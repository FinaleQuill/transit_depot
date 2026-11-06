"""显式运行的合规 V1 Base 教师训练；外折只用于之后的软标签提取。

严格 OOF 协议不读取外折图片、不计算外折指标，也不以外折标签选择 epoch。
训练轮数在开始前固定，只有完整训练至该轮数的最终 EMA 才写入 best.pt。
默认仅预览；只有 --execute 才训练。导入、--help 和 --dry-run 不构建模型或下载。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import sys
from typing import Any, Sequence

from audit_trail import capture_source_version, record_checkpoint, training_output_has_artifacts
from model_overnight import BASE_BACKBONE_NAME, MODEL_NAME, build_overnight_model
from train_overnight import (
    CLASS_IDS, OvernightConfig, _jsonable, build_optimizer, build_weighted_bce,
    capture_rng, compute_loss_config, cpu_tree, preprocessing_config, restore_rng,
    save_checkpoint, sha256, train_epoch, write_json,
)

SCHEMA_VERSION = 6
SELECTION_PROTOCOL = "fixed_epoch_final_ema"


@dataclass(frozen=True)
class TeacherConfig(OvernightConfig):
    """固定训练终点，显式禁用旧训练器的外折早停设置。"""

    patience: int = 0

    def validate(self) -> None:
        super().validate()
        if self.patience != 0:
            raise ValueError("Strict OOF teachers use a fixed epoch horizon, never outer-fold early stopping")
        if self.model_config()["backbone_name"] != BASE_BACKBONE_NAME:
            raise ValueError("Teacher must be the approved ConvNeXt V1 Base")


def teacher_binding(config: TeacherConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只读训练清单，核对患者/内容隔离；不读取外折图片或拟合其标签。"""
    from cv_splits import load_fold_records, read_fold_plan

    config.validate()
    plan_path = config.fold_plan.resolve(strict=True)
    plan_hash = sha256(plan_path)
    plan = read_fold_plan(plan_path)
    training = load_fold_records(plan_path, config.fold, "train")
    oof = load_fold_records(plan_path, config.fold, "valid")
    for field in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {r.get(field) for r in training} - {None, ""}
        right = {r.get(field) for r in oof} - {None, ""}
        if left & right:
            raise ValueError(f"Teacher outer-fold leakage: overlapping {field}")
    if any(r["split"] != "train" for r in training + oof):
        raise ValueError("Teacher fitting and OOF extraction may use original train only")
    binding = {
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "fold_plan_sha256": plan_hash,
        "fold_index": config.fold,
        "training_subject_ids": sorted({r["subject_id"] for r in training}),
        "oof_subject_ids": sorted({r["subject_id"] for r in oof}),
        "training_image_ids": sorted(r["image_id"] for r in training),
        "oof_image_ids": sorted(r["image_id"] for r in oof),
        "selection_protocol": SELECTION_PROTOCOL,
        "outer_fold_used_for_selection": False,
        "official_valid_used_for_training": False,
        "test_used_for_training": False,
        "loss_config": compute_loss_config(
            training, source_manifest_sha256=plan["source_manifest_sha256"],
            fold=config.fold, fold_plan_sha256=plan_hash),
    }
    return binding, training


def validate_teacher_resume(checkpoint: dict[str, Any], config: TeacherConfig,
                            binding: dict[str, Any], precision: str) -> None:
    """拒绝旧 V2、其他折、旧清单、不同训练目标或不完整的恢复状态。"""
    from kd_common import validate_pretrained_provenance

    expected = {
        "schema_version": SCHEMA_VERSION, "artifact_type": "kd_teacher",
        "model_name": MODEL_NAME, "model_config": config.model_config(),
        "class_ids": CLASS_IDS, "image_size": 768,
        "preprocessing": preprocessing_config(image_size=768),
        "inference_precision": precision, **binding,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"Teacher resume protocol mismatch: {field}")
    validate_pretrained_provenance(checkpoint.get("pretrained_provenance"), BASE_BACKBONE_NAME)
    if checkpoint.get("resume_supported") is not True:
        raise ValueError("Resume requires last.pt with complete optimizer/RNG/EMA state")
    for field in ("model_state_dict", "raw_model_state_dict", "optimizer_state_dict",
                  "scaler_state_dict", "rng_state", "progress", "training_config"):
        if field not in checkpoint:
            raise ValueError(f"Teacher checkpoint lacks complete continuation state: {field}")
    source = checkpoint["training_config"]
    current = _jsonable(asdict(config))
    invariant = ("epochs", "seed", "backbone_lr", "head_lr", "warmup_epochs", "min_lr_ratio",
                 "weight_decay", "dropout", "drop_path", "ema_decay", "grad_clip", "patience")
    if any(source.get(k) != current[k] for k in invariant):
        raise ValueError("Teacher resume cannot change the precommitted epoch horizon or optimization objective")
    if source["batch_size"] * source["accumulation_steps"] != config.batch_size * config.accumulation_steps:
        raise ValueError("Teacher resume must retain effective batch size")
    epoch = checkpoint.get("epoch")
    progress = checkpoint["progress"]
    if (type(epoch) is not int or not 0 < epoch <= config.epochs
            or progress.get("completed_epochs") != epoch
            or progress.get("optimizer_updates", 0) != progress.get("ema_updates", -1)
            or progress.get("optimizer_updates", 0) <= 0
            or checkpoint.get("training_complete") is not (epoch == config.epochs)
            or checkpoint.get("inference_ready") is not (epoch == config.epochs)):
        raise ValueError("Teacher checkpoint has inconsistent epoch/EMA completion state")


def _final_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """部署/OOF只保留一套最终EMA；训练断点留在last.pt。"""
    if payload.get("training_complete") is not True or payload.get("inference_ready") is not True:
        raise ValueError("An unfinished teacher cannot be exported for OOF extraction")
    excluded = {"raw_model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state"}
    result = {k: v for k, v in payload.items() if k not in excluded}
    result["resume_supported"] = False
    return result


def run_teacher(config: TeacherConfig) -> dict[str, Any]:
    """仅显式调用此函数才会加载原图、下载预训练权重并执行训练。"""
    import numpy as np
    import torch
    from dataset_overnight import make_overnight_loader
    from kd_common import prepare_pretrained_provenance, require_training_runtime

    config.validate()
    require_training_runtime()
    source_version = capture_source_version(Path(__file__).with_name(name) for name in (
        "train_teacher.py", "train_overnight.py", "model_overnight.py", "dataset_overnight.py",
        "dataset.py", "cv_splits.py", "kd_common.py", "runtime_compat.py", "audit_trail.py"))
    binding, records = teacher_binding(config)
    image_root = config.image_root.resolve(strict=True)
    output = config.output.resolve()
    if output == image_root or output.is_relative_to(image_root) or image_root.is_relative_to(output):
        raise ValueError("Teacher output must not overlap the read-only image root")
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher training requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if config.amp_dtype == "bfloat16" and not bf16:
        raise ValueError("This GPU does not support BF16")
    precision = "bfloat16" if config.amp_dtype == "bfloat16" or config.amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    checkpoint = None
    if config.resume is not None:
        resume = config.resume.resolve(strict=True)
        if not output.is_dir() or resume.parent != output or resume.name != "last.pt":
            raise ValueError("Teacher continuation must use last.pt in its original output directory")
        fingerprint = sha256(resume)
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if sha256(resume) != fingerprint:
            raise RuntimeError("Teacher checkpoint changed while loading")
        validate_teacher_resume(checkpoint, config, binding, precision)
    elif training_output_has_artifacts(output):
        if not config.restart_empty:
            raise FileExistsError("New teacher requires an empty directory; use --resume for last.pt")
        if (list(output.glob("*.pt")) or list(output.glob("*.pt.tmp"))
                or (output / "history.jsonl").exists() and (output / "history.jsonl").stat().st_size):
            raise ValueError("--restart-empty refuses any saved training progress")
        old_config = output / "config.json"
        if old_config.exists():
            prior = json.loads(old_config.read_text())
            if prior.get("model_config") != config.model_config() or prior.get("binding") != binding:
                raise ValueError("--restart-empty refuses a different model/fold/manifest")
    output.mkdir(parents=True, exist_ok=True)
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    if checkpoint is not None:
        progress.update(checkpoint["progress"])
        provenance = checkpoint["pretrained_provenance"]
        if checkpoint["training_complete"]:
            best_hash = save_checkpoint(output / "best.pt", _final_payload(checkpoint))
            record_checkpoint(output / "best.pt", best_hash, epoch=checkpoint["epoch"],
                              role="best", source_version=source_version)
            result = {"state": "completed", **progress, "fold": config.fold,
                      "training_complete": True, "inference_ready": True,
                      "selection_protocol": SELECTION_PROTOCOL, "best_checkpoint_sha256": best_hash}
            write_json(output / "status.json", result)
            return result
    else:
        provenance = prepare_pretrained_provenance(BASE_BACKBONE_NAME)
    write_json(output / "config.json", {**_jsonable(asdict(config)), "binding": binding,
        "model_config": config.model_config(), "preprocessing": preprocessing_config(image_size=768),
        "pretrained_provenance": provenance, "inference_precision": precision})
    write_json(output / "loss_statistics.json", binding["loss_config"])
    seed = (config.seed + config.fold * 10007) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    hook = None
    old_handlers: dict[Any, Any] = {}
    stop_requested = False
    signal_count = 0

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested, signal_count
        signal_count += 1
        if signal_count > 1:
            raise KeyboardInterrupt("Second signal stops immediately; last complete epoch remains saved")
        stop_requested = True
        print("=> Teacher stop requested: finish current epoch, save last.pt, pause.", flush=True)

    try:
        model = build_overnight_model(pretrained=checkpoint is None,
            pretrained_weights_path=provenance["weights_path"] if checkpoint is None else None,
            **config.model_config()).to(
            device, memory_format=torch.channels_last)
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,
                          warmup_power=0.75, exclude_buffers=True, foreach=True)
        ema.module.requires_grad_(False)
        optimizer = build_optimizer(model, config, fused=dtype == torch.bfloat16)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        criterion = build_weighted_bce(binding["loss_config"], device=device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
            ema.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            restore_rng(checkpoint["rng_state"])
            del checkpoint
            checkpoint = None

        def after_step(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            progress["optimizer_updates"] += 1
            ema.update(model, step=progress["optimizer_updates"])
            progress["ema_updates"] += 1

        hook = optimizer.register_step_post_hook(after_step)
        old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        stop_file = config.stop_file.resolve() if config.stop_file else output / "STOP"
        last_epoch = config.epochs if not config.max_epochs_this_call else min(
            config.epochs, progress["completed_epochs"] + config.max_epochs_this_call)
        result: dict[str, Any] = {"state": "paused", **progress, "fold": config.fold,
                                  "training_complete": False, "inference_ready": False}
        for epoch in range(progress["completed_epochs"] + 1, last_epoch + 1):
            if stop_requested or stop_file.exists():
                break
            loader = make_overnight_loader(records, image_root=image_root, training=True,
                image_size=768, batch_size=config.batch_size, num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor, seed=(seed + epoch * 1009) % 2**32)
            training = train_epoch(model, ema, loader, optimizer, scaler, device, dtype,
                                   epoch, config, progress, 768, criterion)
            del loader
            if training["updates_this_epoch"] < 1 or progress["optimizer_updates"] != progress["ema_updates"]:
                raise RuntimeError("Teacher epoch lacks successful updates or EMA progress diverged")
            if sha256(config.fold_plan) != binding["fold_plan_sha256"]:
                raise RuntimeError("Teacher fold plan changed during training")
            from cv_splits import read_fold_plan
            read_fold_plan(config.fold_plan)  # Recheck immutable CSV bytes without opening images.
            progress["completed_epochs"] = epoch
            complete = epoch == config.epochs
            payload = {"schema_version": SCHEMA_VERSION, "artifact_type": "kd_teacher",
                "model_name": MODEL_NAME, "model_config": config.model_config(),
                "class_ids": CLASS_IDS, "num_classes": 10, "image_size": 768,
                "epoch": epoch, **binding, "preprocessing": preprocessing_config(image_size=768),
                "inference_precision": precision, "weights_source": "ema",
                "aggregation": "study_mean_probability", "training_complete": complete,
                "inference_ready": complete, "pretrained_provenance": provenance,
                "training_config": _jsonable(asdict(config)), "progress": dict(progress),
                "training": training, "model_state_dict": cpu_tree(ema.module.state_dict()),
                "raw_model_state_dict": cpu_tree(model.state_dict()),
                "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                "scaler_state_dict": cpu_tree(scaler.state_dict()), "rng_state": capture_rng(),
                "ema": {"enabled": True, "decay": config.ema_decay, "use_warmup": True,
                        "warmup_power": 0.75, "updates": progress["ema_updates"]},
                "resume_supported": True, "reproducibility": "epoch_boundary_rng_not_bitwise_cuda"}
            best_hash = save_checkpoint(output / "best.pt", _final_payload(payload)) if complete else None
            if best_hash is not None:
                record_checkpoint(output / "best.pt", best_hash, epoch=epoch,
                                  role="best", source_version=source_version)
            last_hash = save_checkpoint(output / "last.pt", payload)
            last_audit = record_checkpoint(output / "last.pt", last_hash, epoch=epoch,
                                           role="last", source_version=source_version)
            del payload
            result = {"state": "completed" if complete else "paused", **progress,
                "fold": config.fold, "training_complete": complete, "inference_ready": complete,
                "selection_protocol": SELECTION_PROTOCOL, "last_checkpoint_sha256": last_hash,
                "best_checkpoint_sha256": best_hash, "updated_utc": datetime.now(timezone.utc).isoformat()}
            write_json(output / "status.json", result)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, "training": training,
                    "selection_protocol": SELECTION_PROTOCOL, "outer_fold_evaluated": False,
                    "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash,
                    "checkpoint_saved_at_utc": last_audit["saved_at_utc"],
                    "checkpoint_audit_record_sha256": last_audit["record_sha256"]}, allow_nan=False) + "\n")
            print(f"=> teacher fold={config.fold} epoch={epoch}/{config.epochs} "
                  f"train_BCE={training['loss']:.6f} final_ready={complete}", flush=True)
            if complete or stop_requested or stop_file.exists():
                break
        write_json(output / "status.json", result)
        return result
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", **progress, "fold": config.fold,
            "training_complete": False, "inference_ready": False, "incomplete_epoch": True,
            "error": f"{type(exc).__name__}: {exc}", "resume_from": str(output / "last.pt")})
        raise
    finally:
        if hook is not None:
            hook.remove()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def parse_args(argv: Sequence[str] | None = None) -> tuple[TeacherConfig, bool]:
    parser = argparse.ArgumentParser(description="Fixed-horizon strict-OOF ConvNeXt V1 Base 768 teacher")
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--restart-empty", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training; otherwise preview metadata only")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV/fold metadata only; no model, images or training")
    for name, default in (("batch-size", 2), ("accumulation-steps", 16), ("num-workers", 8),
                          ("prefetch-factor", 2), ("epochs", 20), ("max-epochs-this-call", 0),
                          ("seed", 43), ("warmup-epochs", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("backbone-lr", 2e-5), ("head-lr", 2e-4), ("min-lr-ratio", 0.05),
                          ("weight-decay", 0.05), ("dropout", 0.2), ("drop-path", 0.15),
                          ("ema-decay", 0.999), ("grad-clip", 1.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    args = vars(parser.parse_args(argv))
    dry_run = args.pop("dry_run")
    execute = args.pop("execute")
    if execute and dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    dry_run = dry_run or not execute
    config = TeacherConfig(**args)
    config.validate()
    return config, dry_run


def main(argv: Sequence[str] | None = None) -> int:
    config, dry_run = parse_args(argv)
    if dry_run:
        binding, records = teacher_binding(config)
        print(json.dumps({"dry_run": True, "training_images": len(records),
            "training_patients": len(binding["training_subject_ids"]),
            "oof_patients": len(binding["oof_subject_ids"]),
            "selection_protocol": SELECTION_PROTOCOL, "epochs": config.epochs,
            "model_config": config.model_config(), "training_started": False}, indent=2))
        return 0
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(
            evidence_dir=config.output / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    run_teacher(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
