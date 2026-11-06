"""Diagnose saved V1 Base teacher EMA checkpoints on official valid only.

Default is a metadata/file preview. --execute performs evaluation, never training,
and refuses to overlap a known active training process. Intermediate checkpoints
are allowed for diagnostics without changing their real completion/OOF flags.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / "data/clean_v1_reviewed/valid_manifest.csv"
DEFAULT_FOLD_PLAN = PROJECT / "data/cv5_overnight/folds.json"
APPROVED_VALID_SHA256 = "c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112"
APPROVED_TRAIN_SHA256 = "f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef"
BASE_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
MODEL_NAME = "convnext_v1_base_gap"
CLASS_IDS = list(range(1, 11))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def official_binding(manifest: str | Path, fold_plan: str | Path) -> dict[str, Any]:
    """Read small manifests only; the CLI cannot substitute train/test/OOF rows."""
    manifest, fold_plan = Path(manifest).resolve(strict=True), Path(fold_plan).resolve(strict=True)
    valid_sha = sha256_file(manifest)
    if valid_sha != APPROVED_VALID_SHA256:
        raise ValueError("Only the approved, untouched official valid manifest is permitted")
    plan = json.loads(fold_plan.read_text(encoding="utf-8"))
    if plan.get("n_splits") != 5 or plan.get("class_ids") != CLASS_IDS:
        raise ValueError("Teacher diagnostic requires the approved five-fold ten-class plan")
    holdout = plan.get("official_holdout", {})
    train = plan.get("source_train_manifest", {})
    if (holdout.get("sha256") != valid_sha or sha256_file(holdout["path"]) != valid_sha
            or train.get("sha256") != APPROVED_TRAIN_SHA256
            or plan.get("source_manifest_sha256") != APPROVED_TRAIN_SHA256
            or sha256_file(train["path"]) != APPROVED_TRAIN_SHA256):
        raise ValueError("Official valid/train binding differs from the registered fold plan")
    return {"manifest": str(manifest), "manifest_sha256": valid_sha,
            "fold_plan": str(fold_plan), "fold_plan_sha256": sha256_file(fold_plan), "plan": plan}


def active_training_processes(proc_root: str | Path = "/proc") -> list[dict[str, Any]]:
    known = {"train_teacher.py", "train_distill.py", "train_overnight.py", "train_tiny.py"}
    active = []
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            command = (folder / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        scripts = [Path(arg.decode(errors="replace")).name for arg in command if arg]
        matches = sorted(set(scripts) & known)
        if matches:
            active.append({"pid": int(folder.name), "scripts": matches})
    return active


def validate_teacher_metadata(checkpoint: Mapping[str, Any], *, plan: Mapping[str, Any],
                              plan_sha256: str, train_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate intermediate/final teacher identity without making it OOF-ready."""
    fold = checkpoint.get("fold_index")
    epoch, horizon = checkpoint.get("epoch"), checkpoint.get("training_config", {}).get("epochs")
    progress = checkpoint.get("progress", {})
    complete = type(epoch) is int and type(horizon) is int and epoch == horizon
    if (checkpoint.get("schema_version") != 6 or checkpoint.get("artifact_type") != "kd_teacher"
            or checkpoint.get("model_name") != MODEL_NAME
            or checkpoint.get("model_config", {}).get("backbone_name") != BASE_BACKBONE
            or checkpoint.get("class_ids") != CLASS_IDS or checkpoint.get("image_size") != 768
            or type(fold) is not int or not 0 <= fold < 5
            or type(epoch) is not int or type(horizon) is not int or not 1 <= epoch <= horizon
            or checkpoint.get("fold_plan_sha256") != plan_sha256
            or checkpoint.get("source_manifest_sha256") != plan["source_manifest_sha256"]
            or checkpoint.get("selection_protocol") != "fixed_epoch_final_ema"
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("training_complete") is not complete
            or checkpoint.get("inference_ready") is not complete
            or checkpoint.get("outer_fold_used_for_selection") is not False
            or checkpoint.get("official_valid_used_for_training") is not False
            or checkpoint.get("test_used_for_training") is not False
            or progress.get("completed_epochs") != epoch
            or progress.get("optimizer_updates", 0) <= 0
            or progress.get("optimizer_updates") != progress.get("ema_updates")):
        raise ValueError("Checkpoint is not a consistent approved V1 Base teacher diagnostic source")
    mapping = plan["study_to_fold"]
    training = [r for r in train_records if mapping[r["study_key"]] != fold]
    held_out = [r for r in train_records if mapping[r["study_key"]] == fold]
    for field, rows, key in (("training_subject_ids", training, "subject_id"),
                             ("oof_subject_ids", held_out, "subject_id"),
                             ("training_image_ids", training, "image_id"),
                             ("oof_image_ids", held_out, "image_id")):
        expected = sorted({r[key] for r in rows}) if key == "subject_id" else sorted(r[key] for r in rows)
        if checkpoint.get(field) != expected:
            raise ValueError(f"Teacher diagnostic source differs from the accepted fold membership: {field}")
    if set(checkpoint["training_subject_ids"]) & set(checkpoint["oof_subject_ids"]):
        raise ValueError("Teacher fold patient leakage")
    preprocessing = checkpoint.get("preprocessing", {})
    expected_preprocessing = {"image_size": 768, "validation_size": 768,
        "resize": "letterboxLANCZOS", "validation_tta": "identity", "color_mode": "RGB",
        "exif_transpose": True, "data_source": "raw_readonly", "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]}
    if any(preprocessing.get(k) != v for k, v in expected_preprocessing.items()):
        raise ValueError("Teacher diagnostic preprocessing must remain identity 768 letterbox")
    from kd_common import validate_pretrained_provenance
    validate_pretrained_provenance(checkpoint.get("pretrained_provenance"), BASE_BACKBONE)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Teacher EMA state is absent")
    return {"fold": fold, "epoch": epoch, "fixed_horizon": horizon,
            "training_complete": checkpoint["training_complete"],
            "inference_ready": checkpoint["inference_ready"], "weights_source": "ema"}


def compare_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports or not all(report.get("complete") is True for report in reports):
        raise ValueError("Comparison requires complete diagnostic reports")
    protocol_keys = ("manifest_sha256", "fold_plan_sha256", "fold", "image_size", "precision",
                     "tta", "aggregation", "num_images", "num_studies")
    reference = reports[0]
    if any(any(report.get(k) != reference.get(k) for k in protocol_keys) for report in reports[1:]):
        raise ValueError("Teacher comparisons must use the same data, fold and evaluation protocol")
    values = [float(report["macro_auc"]) for report in reports]
    return {"complete": True, "diagnostic_only": True, "checkpoint_selection_performed": False,
            "thresholds_fitted": False, "reference_checkpoint": reference["checkpoint"],
            "results": [{"checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"],
                         "epoch": r["epoch"], "macro_auc": float(r["macro_auc"]),
                         "macro_auc_delta_from_reference": value - values[0],
                         "training_complete": r["training_complete"],
                         "inference_ready": r["inference_ready"]} for r, value in zip(reports, values)],
            "note": "Descriptive official-valid comparison only. No teacher completion flag, weight, epoch budget or OOF selection is changed."}


def evaluate_checkpoints(*, checkpoints: Sequence[str | Path], manifest: str | Path,
                         fold_plan: str | Path, image_root: str | Path, output_dir: str | Path,
                         batch_size: int = 4, num_workers: int = 8, amp_dtype: str = "auto") -> dict[str, Any]:
    """Only called with explicit --execute, after the active teacher has exited."""
    if active_training_processes():
        raise RuntimeError("Teacher diagnostic refuses to overlap active training; run it after the current GPU trainer exits")
    binding = official_binding(manifest, fold_plan)
    paths = [Path(path).resolve(strict=True) for path in checkpoints]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Choose one or more distinct, saved checkpoints")
    output, image_root = Path(output_dir).resolve(), Path(image_root).resolve(strict=True)
    if output == image_root or output.is_relative_to(image_root) or image_root.is_relative_to(output):
        raise ValueError("Diagnostic output must not overlap the read-only image root")
    if not 1 <= batch_size or not 0 <= num_workers <= 12 or amp_dtype not in {"auto", "bfloat16", "float16"}:
        raise ValueError("Invalid diagnostic batch/worker/precision setting")
    from audit_trail import training_output_has_artifacts
    if training_output_has_artifacts(output):
        raise FileExistsError("Choose a new diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    from training_evidence_monitor import snapshot_source
    archive = snapshot_source(PROJECT, output / "evidence" / (
        "diagnostic_source_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
    import numpy as np
    import torch
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from dataset_overnight import make_overnight_loader
    from model_overnight import build_overnight_model
    from train_overnight import evaluate
    plan = read_fold_plan(fold_plan)
    train_records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    records = read_manifest(manifest, split="valid")
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher diagnostic requires the explicitly reserved CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if amp_dtype == "bfloat16" and not bf16:
        raise ValueError("The selected GPU does not support BF16")
    precision = "bfloat16" if amp_dtype == "bfloat16" or amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    write_json(output / "diagnostic_config.json", {"started_at": utc_now(),
        "checkpoints": [str(p) for p in paths], "manifest": binding["manifest"],
        "manifest_sha256": binding["manifest_sha256"], "fold_plan_sha256": binding["fold_plan_sha256"],
        "image_root": str(image_root), "image_size": 768, "precision": precision,
        "batch_size": batch_size, "num_workers": num_workers, "seed": 43,
        "diagnostic_only": True, "thresholds_fitted": False, "test_used": False,
        "diagnostic_source_archive": str(archive),
        "checkpoint_selection_performed": False, "pretrained_downloads": False,
        "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__})
    reports = []
    for index, path in enumerate(paths):
        print(f"Teacher diagnostic {index + 1}/{len(paths)}: {path}", flush=True)
        initial_stat = path.stat()
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        source = validate_teacher_metadata(checkpoint, plan=plan,
            plan_sha256=binding["fold_plan_sha256"], train_records=train_records)
        initialization = {key:checkpoint["pretrained_provenance"][key] for key in (
            "backbone_name", "repo_id", "license", "revision", "weights_sha256", "weights_url",
            "pretrain_dataset", "finetune_dataset")}
        model_config = {**checkpoint["model_config"], "gradient_checkpointing": False}
        state = checkpoint["model_state_dict"]
        del checkpoint
        model = build_overnight_model(pretrained=False, **model_config)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
        loader = make_overnight_loader(records, image_root, training=False, image_size=768,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=1, seed=43)
        try:
            values = evaluate(model, loader, records, device, dtype)
        finally:
            del loader, model
            gc.collect()
            torch.cuda.empty_cache()
        after = path.stat()
        if ((initial_stat.st_size, initial_stat.st_mtime_ns, initial_stat.st_ino)
                != (after.st_size, after.st_mtime_ns, after.st_ino)
                or sha256_file(path) != digest):
            raise RuntimeError("Diagnostic checkpoint changed during evaluation")
        if sha256_file(manifest) != binding["manifest_sha256"] or sha256_file(fold_plan) != binding["fold_plan_sha256"]:
            raise RuntimeError("Registered validation manifest/fold plan changed during evaluation")
        stem = f"checkpoint_{index}_epoch_{source['epoch']}"
        with (output / (stem + ".csv")).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability"])
            for subject, study, probability in zip(values["subject_ids"], values["study_ids"], values["probabilities"]):
                for class_id, number in zip(CLASS_IDS, probability):
                    writer.writerow([subject, study, class_id, format(float(number), ".17g")])
        np.savez_compressed(output / (stem + ".npz"),
            study_keys=np.asarray(values["study_keys"], dtype=str),
            targets=values["targets"], probabilities=values["probabilities"])
        report = {**values["metrics"], "complete": True, "diagnostic_only": True, "checkpoint": str(path),
            "checkpoint_sha256": digest, **source, "manifest_sha256": binding["manifest_sha256"],
            "fold_plan_sha256": binding["fold_plan_sha256"], "image_size": 768, "precision": precision,
            "tta": "identity", "aggregation": "study_mean_probability", "num_images": len(records),
            "num_studies": len(values["study_keys"]),
            "initialization": initialization,
            "thresholds_fitted": False, "checkpoint_selection_performed": False,
            "test_used": False, "finished_at": utc_now()}
        write_json(output / (stem + "_report.json"), report)
        reports.append(report)
        print(f"Teacher epoch={source['epoch']} official_valid_MacroAUC={report['macro_auc']:.8f}", flush=True)
    comparison = compare_reports(reports)
    comparison.update(manifest_sha256=binding["manifest_sha256"],
                      fold_plan_sha256=binding["fold_plan_sha256"], reports=reports, finished_at=utc_now())
    write_json(output / "comparison_report.json", comparison)
    return comparison


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", dest="checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fold-plan", type=Path, default=DEFAULT_FOLD_PLAN)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        binding = official_binding(args.manifest, args.fold_plan)
        print(json.dumps({"dry_run": True, "checkpoints_loaded": 0, "gpu_used": False,
            "diagnostic_only": True, "checkpoints": [{"path": str(p.absolute()), "exists": p.is_file()}
            for p in args.checkpoints], "manifest_sha256": binding["manifest_sha256"],
            "fold_plan_sha256": binding["fold_plan_sha256"], "output_dir": str(args.output_dir.absolute()),
            "note": "No image or checkpoint bytes read; --execute is required for evaluation."}, indent=2))
        return 0
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=args.output_dir / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    settings = vars(args)
    settings.pop("execute")
    print(json.dumps(evaluate_checkpoints(**settings), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
