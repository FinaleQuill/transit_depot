"""Isolated AP-only early stopping for the authorized 25-epoch A/B amendment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import math

import numpy as np
import torch

import study_student
from audit_trail import capture_source_version, record_checkpoint, training_output_has_artifacts
from cv_splits import read_fold_plan
from dataset import read_manifest
from kd_common import sha256_file, write_json
from model_overnight import build_overnight_model
from study_student import StudyConfig
from tight_crop_ab_data import make_ab_valid_loader
from tight_crop_ab_training import (ABConfig, FOLD_PLAN, INIT_SHA, PLAN, ROOT,
                                    _ab_loaders, build_config)
from train_distill import _configuration, evaluate_student
from train_overnight import calculate_metrics, cpu_tree, save_checkpoint


EARLY_PLAN = ROOT / "docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json"
MONITOR_PROTOCOL = "native_fp32_ema_fp16_autocast_batch4_hflip2"
ORIGINAL_E1 = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/A_tiny/last.pt")
IMMUTABLE_E1 = ROOT / "artifacts/tight_crop_ab_20261102/earlystop25/migration/original_E1/last.pt"
LOCAL_E1_COPY = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/earlystop25/migration/A_tiny_original_E1_last.pt")


def _verify_e1_source(source: Path):
    source = Path(source).resolve(strict=True)
    allowed = {ORIGINAL_E1.resolve(), IMMUTABLE_E1.resolve(), LOCAL_E1_COPY.resolve()}
    if source not in allowed:
        raise ValueError("E1 source must be original last.pt or its registered immutable copy")
    original = ORIGINAL_E1.resolve(strict=True)
    if sha256_file(source) != sha256_file(original):
        raise ValueError("Immutable E1 source differs from the original checkpoint")
    return source


@dataclass(frozen=True)
class AB25Config(ABConfig):
    epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    earlystop_plan_sha256: str = ""

    def validate(self):
        StudyConfig.validate(self)
        if (self.arm not in ("A", "B") or self.member not in ("tiny", "base")
                or self.seed != 48 or self.epochs != 25 or self.patience != 5
                or self.min_delta != 1e-4 or self.pack_images != 8
                or self.warmup_epochs != 1 or self.kd_warmup_epochs != 0
                or self.alpha != 0.2 or self.temperature != 2
                or self.ema_decay != 0.999 or self.valid_batch_size != 4
                or len(self.earlystop_plan_sha256) != 64
                or self.earlystop_plan_sha256 != sha256_file(EARLY_PLAN)):
            raise ValueError("A/B 25-epoch amendment protocol changed")
        backbone = study_student.TINY_BACKBONE if self.member == "tiny" else study_student.BASE_BACKBONE
        lr = (1e-5, 1e-4, 0.1) if self.member == "tiny" else (5e-6, 5e-5, 0.15)
        if (self.backbone_name != backbone
                or (self.backbone_lr, self.head_lr, self.drop_path) != lr
                or self.erase_probability != (0.5 if self.arm == "B" else 0.0)
                or self.ab_plan_sha256 != sha256_file(PLAN)
                or self.ab_preprocess_sha256 != sha256_file(ROOT / "tight_crop_ab_preprocess.py")
                or self.ab_crop_sha256 != sha256_file(ROOT / "tight_crop.py")):
            raise ValueError("A/B member or frozen source differs")
        if self.resume is None and (self.init_student_ema is None
                                    or self.init_student_ema_sha256 != INIT_SHA[self.member]):
            raise ValueError("Fresh A/B run requires its original registered FP32 EMA")


def build_early_config(*, arm, member, output, resume=None, image_root=Path("/home/featurize/data/cxr_first"),
                       num_workers=4):
    if not EARLY_PLAN.is_file():
        raise FileNotFoundError(EARLY_PLAN)
    base = build_config(arm=arm, member=member, output=Path(output), resume=resume,
                        image_root=Path(image_root), num_workers=num_workers,
                        valid_batch_size=4)
    values = asdict(base)
    values.update(epochs=25, patience=5, min_delta=1e-4,
                  earlystop_plan_sha256=sha256_file(EARLY_PLAN))
    config = AB25Config(**values)
    config.validate()
    return config


def ap_monitor_metrics(validation, valid_records):
    """Use the native train-validation Study probabilities; all classes must qualify."""
    keys = validation["study_keys"]
    views = {}
    for row in valid_records:
        views.setdefault(row["study_key"], []).append(str(row["view_position"]).upper())
    if len(keys) != len(views) or set(keys) != set(views):
        raise ValueError("Native validation Study coverage differs from official valid")
    selected = [i for i, key in enumerate(keys) if all(view == "AP" for view in views[key])]
    if not selected:
        raise ValueError("AP-only validation slice is empty")
    ap_metrics = calculate_metrics(validation["targets"][selected],
                                   validation["probabilities"][selected])
    native = validation["metrics"]
    native["all_study_macro_auc"] = native["macro_auc"]
    native["all_study_macro_ap"] = native["macro_ap"]
    native["all_study_per_class_auc"] = native["per_class_auc"]
    native["all_study_per_class_ap"] = native["per_class_ap"]
    native["all_study_loss"] = native["loss"]
    native["all_study_num_studies"] = native["num_studies"]
    native["ap_only_macro_auc"] = ap_metrics["macro_auc"]
    native["ap_only_macro_ap"] = ap_metrics["macro_ap"]
    native["ap_only_per_class_auc"] = ap_metrics["per_class_auc"]
    native["ap_only_per_class_ap"] = ap_metrics["per_class_ap"]
    native["ap_only_loss"] = ap_metrics["loss"]
    native["ap_only_num_studies"] = len(selected)
    native["monitor"] = "ap_only_macro_auc"
    native["macro_auc_scope"] = "AP-only Studies; all_study_macro_auc retains fullvalid"
    native["monitor_protocol"] = MONITOR_PROTOCOL
    native["macro_auc"] = ap_metrics["macro_auc"]
    native["macro_ap"] = ap_metrics["macro_ap"]
    native["per_class_auc"] = ap_metrics["per_class_auc"]
    native["per_class_ap"] = ap_metrics["per_class_ap"]
    native["loss"] = ap_metrics["loss"]
    native["num_studies"] = len(selected)
    return validation


def _preprocessing25(config):
    from tight_crop_ab_training import _preprocessing
    result = _preprocessing(config)
    result["earlystop_plan_sha256"] = config.earlystop_plan_sha256
    result["paired_training"] = {"seed": 48, "max_epochs": 25,
        "patience": 5, "min_delta": 1e-4, "checkpoint": "best.pt AP-only EMA",
        "monitor_protocol": MONITOR_PROTOCOL}
    return result


@contextmanager
def _early_runtime(config, valid_records):
    with _ab_loaders(config.arm):
        previous_eval = study_student.evaluate_student
        previous_pre = study_student.student_preprocessing_config
        previous_source = study_student.capture_source_version
        import train_overnight
        previous_save = train_overnight.save_checkpoint
        def native_evaluate(model, loader, records, device, dtype, used_config):
            result = previous_eval(model, loader, records, device, dtype, used_config)
            return ap_monitor_metrics(result, records)
        def source_version(paths):
            return previous_source([*list(paths), Path(__file__), EARLY_PLAN])
        def save_with_ancestry(path, payload):
            migration = config.output / "migration.json"
            if migration.is_file():
                payload = {**payload, "migration": json.loads(migration.read_text(encoding="utf-8"))}
            return previous_save(path, payload)
        study_student.evaluate_student = native_evaluate
        study_student.student_preprocessing_config = _preprocessing25
        study_student.capture_source_version = source_version
        train_overnight.save_checkpoint = save_with_ancestry
        try:
            yield
        finally:
            study_student.evaluate_student = previous_eval
            study_student.student_preprocessing_config = previous_pre
            study_student.capture_source_version = previous_source
            train_overnight.save_checkpoint = previous_save


def monitor_source(*, arm, member, source, output):
    """Score an old E1 full EMA with the exact native validation arithmetic."""
    source, output = _verify_e1_source(source), Path(output)
    if output.exists():
        raise ValueError("E1 monitor needs original last.pt and a new output directory")
    config = build_early_config(arm=arm, member=member, output=output)
    if (arm, member) != ("A", "tiny"):
        raise ValueError("Only the interrupted original A_tiny E1 can be migrated")
    digest = sha256_file(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if (sha256_file(source) != digest or checkpoint.get("epoch") != 1
            or checkpoint.get("training_config", {}).get("epochs") != 2
            or checkpoint.get("training_config", {}).get("arm") != arm
            or checkpoint.get("training_config", {}).get("member") != member
            or checkpoint.get("training_config", {}).get("ab_plan_sha256") != config.ab_plan_sha256
            or checkpoint.get("preprocessing", {}).get("ab_preprocess_sha256")
                != config.ab_preprocess_sha256):
        raise ValueError("Original E1 source differs from frozen A_tiny run")
    plan = read_fold_plan(FOLD_PLAN)
    records = read_manifest(plan["official_holdout"]["path"], split="valid")
    if not torch.cuda.is_available():
        raise RuntimeError("Native E1 monitor requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True
    model = build_overnight_model(pretrained=False, **config.model_config())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
    loader = make_ab_valid_loader(records, config.image_root, arm=arm, training=False,
        image_size=768, batch_size=4, num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor, seed=(48 + 1009) % 2**32)
    validation = ap_monitor_metrics(evaluate_student(model, loader, records,
        device, torch.float16, config), records)
    receipt = {"arm": arm, "member": member, "epoch": 1,
        "source_sha256": digest, "ap_only_macro_auc": validation["metrics"]["ap_only_macro_auc"],
        "all_study_macro_auc": validation["metrics"]["all_study_macro_auc"],
        "monitor_protocol": MONITOR_PROTOCOL, "ab_plan_sha256": config.ab_plan_sha256,
        "earlystop_plan_sha256": config.earlystop_plan_sha256,
        "ab_preprocess_sha256": config.ab_preprocess_sha256,
        "ab_crop_sha256": config.ab_crop_sha256,
        "validation": validation["metrics"]}
    output.mkdir(parents=True)
    write_json(output / "monitor.json", receipt)
    return receipt


def migrate_e1(*, config: AB25Config, source: Path, monitor_path: Path):
    """Copy original E1 state into a new 25-epoch run without changing source."""
    if (config.arm, config.member) != ("A", "tiny") or config.resume is not None:
        raise ValueError("Only original A_tiny E1 may be migrated")
    source = _verify_e1_source(source)
    monitor_path = Path(monitor_path).resolve(strict=True)
    if training_output_has_artifacts(config.output):
        raise ValueError("Migration requires original last.pt and an empty new run")
    digest = sha256_file(source)
    old = torch.load(source, map_location="cpu", weights_only=True)
    score = json.loads(monitor_path.read_text(encoding="utf-8"))
    if (sha256_file(source) != digest or score.get("source_sha256") != digest
            or score.get("monitor_protocol") != MONITOR_PROTOCOL
            or not isinstance(score.get("ap_only_macro_auc"), (int, float))
            or not math.isfinite(score["ap_only_macro_auc"])
            or score.get("earlystop_plan_sha256") != config.earlystop_plan_sha256
            or old.get("epoch") != 1 or old.get("progress", {}).get("completed_epochs") != 1
            or old.get("training_config", {}).get("epochs") != 2
            or old.get("training_config", {}).get("arm") != "A"
            or old.get("training_config", {}).get("member") != "tiny"
            or old.get("training_config", {}).get("ab_plan_sha256") != config.ab_plan_sha256
            or old.get("preprocessing", {}).get("ab_preprocess_sha256")
                != config.ab_preprocess_sha256):
        raise ValueError("E1 migration source or native AP score differs")
    old_history = ORIGINAL_E1.parent / "history.jsonl"
    rows = [json.loads(line) for line in old_history.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(rows) != 1 or rows[0].get("epoch") != 1
            or rows[0].get("last_checkpoint_sha256") != digest
            or not isinstance(rows[0].get("training"), dict)):
        raise ValueError("Original E1 optimizer work has no matching epoch history")
    from tight_crop_ab_training import _preprocessing
    with _ab_loaders(config.arm):
        preprocessing = _preprocessing25(config)
    migrated = dict(old)
    migrated["training_config"] = _configuration(config)
    migrated["preprocessing"] = preprocessing
    migrated["validation"] = score["validation"]
    migrated["best_auc"] = float(score["ap_only_macro_auc"])
    migrated["best_epoch"] = 1
    migrated["early_stopping"] = {"reference_auc": migrated["best_auc"], "bad_epochs": 0}
    migrated["source_version"] = capture_source_version([ROOT / name for name in
        ("tight_crop_ab_earlystop_training.py", "tight_crop_ab_training.py",
         "tight_crop_ab_data.py", "tight_crop_ab_preprocess.py", "tight_crop.py",
         "study_student.py", "study_packing.py", "train_distill.py")]
        + [PLAN, EARLY_PLAN])
    migrated["migration"] = {"source_path": str(source), "source_sha256": digest,
        "monitor_receipt_path": str(monitor_path), "monitor_receipt_sha256": sha256_file(monitor_path),
        "source_epoch": 1, "source_training_epochs": 2,
        "preserved": ["raw_model_state_dict", "model_state_dict", "optimizer_state_dict",
                      "scaler_state_dict", "rng_state", "progress"],
        "optimizer_rng_reset": False, "policy_change": "warmup1 to cosine25 AP-only patience5"}
    out = config.output.resolve()
    out.mkdir(parents=True)
    write_json(out / "migration.json", migrated["migration"])
    write_json(out / "config.json", {**_configuration(config), "model_config": config.model_config(),
        "preprocessing": preprocessing, "migration": migrated["migration"]})
    write_json(out / "pretrained_provenance.json", old["pretrained_provenance"])
    best_hash = save_checkpoint(out / "best.pt", migrated)
    record_checkpoint(out / "best.pt", best_hash, epoch=1, role="best", source_version=migrated["source_version"])
    last_hash = save_checkpoint(out / "last.pt", migrated)
    audit = record_checkpoint(out / "last.pt", last_hash, epoch=1, role="last",
                              source_version=migrated["source_version"])
    with (out / "history.jsonl").open("x") as stream:
        stream.write(json.dumps({"epoch": 1, "migrated_from_sha256": digest,
            "training": rows[0]["training"],
            "validation": migrated["validation"], "is_best": True,
            "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash,
            "checkpoint_saved_at_utc": audit["saved_at_utc"],
            "checkpoint_audit_record_sha256": audit["record_sha256"]}) + "\n")
    write_json(out / "status.json", {"state": "paused", **old["progress"],
        "best_auc": migrated["best_auc"], "best_epoch": 1,
        "selected_checkpoint": "best.pt", "monitor": "ap_only_macro_auc",
        "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash})
    if sha256_file(source) != digest:
        raise RuntimeError("Original E1 checkpoint changed during migration")
    return out / "last.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--member", choices=("tiny", "base"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--migrate-from", type=Path)
    parser.add_argument("--migrate-ap-score", type=Path)
    parser.add_argument("--monitor-source", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = build_early_config(arm=args.arm, member=args.member, output=args.output,
        resume=args.resume, num_workers=args.num_workers)
    if not args.execute:
        print(json.dumps({"dry_run": True, "config": _configuration(config),
                          "monitor_protocol": MONITOR_PROTOCOL}, indent=2))
        return 0
    if args.monitor_source is not None:
        if args.resume or args.migrate_from or args.migrate_ap_score:
            parser.error("Native monitor cannot combine with training or migration")
        monitor_source(arm=args.arm, member=args.member,
                       source=args.monitor_source, output=args.output)
        return 0
    if bool(args.migrate_from) != bool(args.migrate_ap_score):
        parser.error("Migration requires both --migrate-from and --migrate-ap-score")
    if args.migrate_from:
        if args.resume:
            parser.error("Migration and --resume are separate actions")
        migrate_e1(config=config, source=args.migrate_from, monitor_path=args.migrate_ap_score)
        return 0
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=config.output / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    plan = read_fold_plan(FOLD_PLAN)
    valid_records = read_manifest(plan["official_holdout"]["path"], split="valid")
    with _early_runtime(config, valid_records):
        result = study_student.run_training(config)
    status = json.loads((config.output / "status.json").read_text())
    status.update(selected_checkpoint="best.pt", monitor="ap_only_macro_auc",
                  max_epochs=25, patience=5, min_delta=1e-4)
    write_json(config.output / "status.json", status)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
