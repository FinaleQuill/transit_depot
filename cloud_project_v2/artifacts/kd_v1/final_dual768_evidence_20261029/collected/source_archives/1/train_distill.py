"""单个 V1 Tiny/512 离线蒸馏。仅 main 的 --execute 才会开始训练。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from audit_trail import capture_source_version, record_checkpoint, training_output_has_artifacts

from kd_common import (TINY_BACKBONE, prepare_pretrained_provenance, sha256_file,
                       student_preprocessing, validate_pretrained_provenance, write_json,
                       require_training_runtime)


@dataclass(frozen=True)
class DistillConfig:
    fold_plan: Path
    soft_labels: Path
    output: Path
    image_root: Path
    batch_size: int = 8
    accumulation_steps: int = 4
    valid_batch_size: int = 4
    num_workers: int = 8
    prefetch_factor: int = 2
    epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 43
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    dropout: float = 0.2
    drop_path: float = 0.1
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    amp_dtype: str = "auto"
    alpha: float = 0.2
    temperature: float = 2.0
    kd_warmup_epochs: int = 2
    align_teacher_pos_weight: bool = True
    resume: Path | None = None

    def model_config(self) -> dict[str, Any]:
        from model_overnight import validate_model_config
        return validate_model_config({"backbone_name": TINY_BACKBONE,
                    "drop_rate": self.dropout, "drop_path_rate": self.drop_path,
                    "gradient_checkpointing": True})

    def validate(self) -> None:
        for name in ("batch_size", "accumulation_steps", "valid_batch_size", "epochs", "prefetch_factor"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("patience", "seed", "num_workers", "kd_warmup_epochs", "warmup_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.num_workers > 12 or self.prefetch_factor not in (1, 2) or self.seed >= 2**32:
            raise ValueError("Invalid workers/prefetch/seed")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("LR warmup must be shorter than the training horizon")
        for name in ("backbone_lr", "head_lr", "grad_clip", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.alpha <= 1 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite in [0,1]")
        if not 0 <= self.ema_decay < 1 or not 0 < self.min_lr_ratio <= 1:
            raise ValueError("Invalid EMA or cosine schedule")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if self.amp_dtype not in ("auto", "bfloat16", "float16"):
            raise ValueError("Unsupported training precision")
        if type(self.align_teacher_pos_weight) is not bool:
            raise ValueError("Logit alignment must be boolean")
        self.model_config()


def alpha_at(config: DistillConfig, epoch: int, batch_index: int, batches: int) -> float:
    if not config.kd_warmup_epochs:
        return config.alpha
    return config.alpha * min(1.0, (epoch - 1 + (batch_index + 1) / batches) / config.kd_warmup_epochs)


def distillation_losses(student_logits: torch.Tensor, targets: torch.Tensor,
                        teacher_logits: torch.Tensor, teacher_pos_weights: torch.Tensor,
                        student_pos_weights: torch.Tensor, *, temperature: float = 2.0,
                        align: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """返回两个逐图损失；各自对十类取 mean，外部再施加相同 Study 权重。

    Bernoulli KL 包含阳性和阴性两项，绝不将十类作 softmax。
    加权 BCE logit 对齐为理论先验修正，须与原始 logits 对照验证。
    """
    shape = student_logits.shape
    if (len(shape) != 2 or shape[1] != 10 or targets.shape != shape
            or teacher_logits.shape != shape or teacher_pos_weights.shape != shape
            or student_pos_weights.shape != (10,)):
        raise ValueError("KD expects Nx10 image logits/labels and ten student class weights")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("KD temperature must be positive and finite")
    for value in (student_logits, targets, teacher_logits, teacher_pos_weights, student_pos_weights):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite KD input")
    if bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("Hard labels must be in [0,1]")
    if bool((teacher_pos_weights <= 0).any()) or bool((student_pos_weights <= 0).any()):
        raise ValueError("Class weights must be positive")
    s, t, y = student_logits.float(), teacher_logits.detach().float(), targets.float()
    ws, wt = student_pos_weights.float(), teacher_pos_weights.float()
    if align:
        t = t - wt.log() + ws.log()[None, :]
    t, s = t / temperature, s / temperature
    q = torch.sigmoid(t)
    kl = q * (F.logsigmoid(t) - F.logsigmoid(s))
    kl += (1 - q) * (F.logsigmoid(-t) - F.logsigmoid(-s))
    # 只截断浮点舍入导致的极小负值，不改变正 KL。
    soft = kl.mean(1).clamp_min(0) * temperature**2
    hard = F.binary_cross_entropy_with_logits(student_logits.float(), y,
                                              pos_weight=ws, reduction="none").mean(1)
    return hard, soft


def _configuration(config: DistillConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()}


def validate_student_resume(checkpoint: dict[str, Any], config: DistillConfig,
                            expected: dict[str, Any]) -> None:
    """纯元数据恢复检查；不允许用损坏的计数跳过训练或覆盖更佳结果。"""
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Student resume protocol mismatch: {key}")
    current = {k: v for k, v in _configuration(config).items() if k != "resume"}
    prior = {k: v for k, v in checkpoint.get("training_config", {}).items() if k != "resume"}
    if current != prior:
        raise ValueError("Student resume training configuration changed")
    for field in ("model_state_dict", "raw_model_state_dict", "optimizer_state_dict",
                  "scaler_state_dict", "rng_state", "progress", "early_stopping"):
        if field not in checkpoint:
            raise ValueError(f"Student continuation state is missing: {field}")
    progress = checkpoint["progress"]
    epoch, best_epoch, auc = checkpoint.get("epoch"), checkpoint.get("best_epoch"), checkpoint.get("best_auc")
    if (type(epoch) is not int or not 1 <= epoch <= config.epochs
            or progress.get("completed_epochs") != epoch or progress.get("validated_epochs") != epoch
            or type(best_epoch) is not int or not 1 <= best_epoch <= epoch
            or type(auc) not in (int, float) or not math.isfinite(auc) or not 0 <= auc <= 1
            or progress.get("optimizer_updates", 0) <= 0
            or progress.get("optimizer_updates") != progress.get("ema_updates")
            or checkpoint.get("inference_ready") is not True
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("weights_source") != "ema"):
        raise ValueError("Inconsistent student epoch/validation/EMA/best state")
    early = checkpoint["early_stopping"]
    reference, bad = early.get("reference_auc"), early.get("bad_epochs")
    if (type(reference) not in (int, float) or not math.isfinite(reference)
            or type(bad) is not int or not 0 <= bad <= epoch):
        raise ValueError("Invalid student early-stopping state")
    validate_pretrained_provenance(checkpoint.get("pretrained_provenance"), TINY_BACKBONE)


def reconcile_student_checkpoints(last: dict[str, Any], best: dict[str, Any],
                                  config: DistillConfig, expected: dict[str, Any]) -> dict[str, Any]:
    validate_student_resume(last, config, expected)
    validate_student_resume(best, config, expected)
    if (best["best_epoch"] != best["epoch"]
            or best.get("validation", {}).get("macro_auc") != best["best_auc"]):
        raise ValueError("Student best.pt does not contain the claimed best epoch")
    if best["epoch"] > last["epoch"]:
        if best["best_auc"] < last["best_auc"]:
            raise ValueError("Newer student best has a lower recorded best AUC")
        return best
    if best["epoch"] != last["best_epoch"] or best["best_auc"] != last["best_auc"]:
        raise ValueError("Student best/last checkpoint bindings disagree")
    return last


def _train_epoch(model, ema, loader, optimizer, scaler, device, dtype, epoch,
                 config, progress, cache, student_weights):
    from train_overnight import set_learning_rates
    model.train()
    expected = {row["image_id"] for row in loader.dataset.records}
    seen: set[str] = set()
    total = hard_sum = soft_sum = weight_sum = 0.0
    windows = math.ceil(len(loader) / config.accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    updates_before = progress["optimizer_updates"]
    for batch_index, batch in enumerate(loader):
        ids = batch["image_ids"]
        if len(set(ids)) != len(ids) or seen.intersection(ids) or not set(ids) <= expected:
            raise ValueError("Student training image coverage is invalid")
        seen.update(ids)
        indices = [cache.index[image_id] for image_id in ids]
        teacher = torch.as_tensor(cache.logits[indices], device=device, dtype=torch.float32)
        teacher_weights = torch.as_tensor(cache.teacher_pos_weights[indices], device=device, dtype=torch.float32)
        images = batch["images"].to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = batch["targets"].to(device, non_blocking=True, dtype=torch.float32)
        weights = batch["loss_weights"].to(device, non_blocking=True, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Invalid student Study weights")
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(images)
        hard, soft = distillation_losses(logits, targets, teacher, teacher_weights, student_weights,
                                        temperature=config.temperature, align=config.align_teacher_pos_weight)
        alpha = alpha_at(config, epoch, batch_index, len(loader))
        weighted_sum = ((1 - alpha) * hard + alpha * soft).mul(weights).sum()
        if not bool(torch.isfinite(weighted_sum)):
            raise FloatingPointError("Nonfinite student loss")
        first = batch_index // config.accumulation_steps * config.accumulation_steps
        window_images = min(config.batch_size * config.accumulation_steps,
                            len(loader.dataset) - first * config.batch_size)
        scaler.scale(weighted_sum / window_images).backward()
        progress["backward_batches"] += 1
        total += float(weighted_sum.detach())
        hard_sum += float((hard.detach() * weights).sum())
        soft_sum += float((soft.detach() * weights).sum())
        weight_sum += float(weights.sum())
        if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(loader):
            rates = set_learning_rates(optimizer, config, progress["optimizer_updates"], windows)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                     error_if_nonfinite=not scaler.is_enabled())
            before = progress["optimizer_updates"]
            scaler.step(optimizer)
            scaler.update()
            if progress["optimizer_updates"] == before:
                progress["amp_skipped_steps"] += 1
            optimizer.zero_grad(set_to_none=True)
        if batch_index == 0 or (batch_index + 1) % 100 == 0:
            print(f"student epoch={epoch} batch={batch_index+1}/{len(loader)} "
                  f"loss={total/len(seen):.6f} alpha={alpha:.4f}", flush=True)
    if seen != expected or not math.isclose(weight_sum, len(expected), rel_tol=2e-6, abs_tol=1e-3):
        raise RuntimeError("Student epoch is incomplete or Study normalization changed")
    if progress["optimizer_updates"] <= updates_before or progress["optimizer_updates"] != progress["ema_updates"]:
        raise RuntimeError("Student optimizer/EMA did not advance together")
    return {"loss": total / len(expected), "hard_bce": hard_sum / len(expected),
            "soft_kl_T2": soft_sum / len(expected), "alpha_end": alpha,
            "images": len(expected), "full_manifest_coverage": True,
            "seconds": time.monotonic() - started, "learning_rates": rates,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


def run_training(config: DistillConfig) -> dict[str, Any]:
    config.validate()
    require_training_runtime()
    source_version = capture_source_version(Path(__file__).with_name(name) for name in (
        "train_distill.py", "train_overnight.py", "distill_data.py", "model_overnight.py",
        "dataset_overnight.py", "dataset.py", "cv_splits.py", "kd_common.py",
        "runtime_compat.py", "audit_trail.py"))
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from dataset_overnight import make_overnight_loader
    from distill_data import load_soft_labels
    from model_overnight import TINY_MODEL_NAME, build_overnight_model
    from train_overnight import (build_optimizer, capture_rng, compute_loss_config, cpu_tree,
                                 evaluate, restore_rng, save_checkpoint)
    from timm.utils import ModelEmaV3

    plan = read_fold_plan(config.fold_plan)
    plan_hash = sha256_file(config.fold_plan)
    train_records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    valid_path = Path(plan["official_holdout"]["path"])
    valid_records = read_manifest(valid_path, split="valid")
    # 所有选择只用官方 valid；test 不会被本程序打开。
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        if ({row.get(key) for row in train_records} - {None, ""}) & ({row.get(key) for row in valid_records} - {None, ""}):
            raise ValueError(f"Student train/official-valid leakage: {key}")
    cache = load_soft_labels(config.soft_labels, config.fold_plan, train_records)
    soft_file = config.soft_labels / "soft_labels.npz"
    meta_file = config.soft_labels / "metadata.json"
    loss_config = compute_loss_config(train_records, source_manifest_sha256=plan["source_manifest_sha256"], fold=None)
    protocol = {"alpha": config.alpha, "temperature": config.temperature,
                "warmup_epochs": config.kd_warmup_epochs,
                "loss": "bernoulli_kl_T2_plus_weighted_bce", "class_reduction": "mean",
                "study_equal_weighting": True, "align_teacher_pos_weight": config.align_teacher_pos_weight,
                "soft_labels_sha256": sha256_file(soft_file), "soft_metadata_sha256": sha256_file(meta_file),
                "teacher_tta": "identity", "student_training_size": 512,
                "deployment_tta": "identity", "deployment_ensemble": False}
    if not torch.cuda.is_available():
        raise RuntimeError("Student training requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if config.amp_dtype == "bfloat16" and not bf16:
        raise ValueError("This GPU does not support BF16")
    precision = "bfloat16" if config.amp_dtype == "bfloat16" or config.amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    out = config.output.resolve()
    root = config.image_root.resolve(strict=True)
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Student output must not overlap the read-only image root")
    if config.resume is None:
        if training_output_has_artifacts(out):
            raise FileExistsError("New student training requires an empty output directory")
        out.mkdir(parents=True, exist_ok=True)
    elif config.resume.resolve().parent != out or config.resume.name != "last.pt":
        raise ValueError("Resume must use last.pt from the same student output directory")
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    early = {"reference_auc": None, "bad_epochs": 0}
    best_auc, best_epoch = None, None
    best_hash = None
    checkpoint = None
    serialized = _configuration(config)
    if config.resume:
        # 第一次 best 已原子提交但第一次 last 未提交时，也可恢复这份完整状态。
        resume_source = config.resume if config.resume.is_file() else out / "best.pt"
        resume_hash = sha256_file(resume_source)
        checkpoint = torch.load(resume_source, map_location="cpu", weights_only=True)
        if sha256_file(resume_source) != resume_hash:
            raise RuntimeError("Student last.pt changed while loading")
        expected = {"schema_version": 7, "artifact_type": "kd_student", "model_name": TINY_MODEL_NAME,
                    "fold_plan_sha256": plan_hash, "source_manifest_sha256": plan["source_manifest_sha256"],
                    "official_valid_manifest_sha256": sha256_file(valid_path), "distillation": protocol,
                    "loss_config": loss_config, "model_config": config.model_config(),
                    "preprocessing": student_preprocessing(), "training_precision": precision,
                    "teacher_provenance": cache.metadata["teachers"]}
        validate_student_resume(checkpoint, config, expected)
        # 丢失最佳文件时不能把更差的 last 伪装成最佳。
        if not (out / "best.pt").is_file():
            raise FileNotFoundError("Saved student best.pt is missing; retain both best and last for resume")
        best_hash = sha256_file(out / "best.pt")
        best_record = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        if sha256_file(out / "best.pt") != best_hash:
            raise RuntimeError("Student best.pt changed while loading")
        # best 先提交、last 后提交；中间中断时提升已完成且完整的更新 best。
        checkpoint = reconcile_student_checkpoints(checkpoint, best_record, config, expected)
        progress.update(checkpoint["progress"])
        early = checkpoint["early_stopping"]
        best_auc, best_epoch = checkpoint["best_auc"], checkpoint["best_epoch"]
        provenance = checkpoint["pretrained_provenance"]
        del best_record
    else:
        provenance = prepare_pretrained_provenance(TINY_BACKBONE)
    if progress["completed_epochs"] >= config.epochs or config.patience and early["bad_epochs"] >= config.patience:
        return {"state": "already_terminal", **progress, "best_auc": best_auc, "best_epoch": best_epoch}
    model = build_overnight_model(pretrained=checkpoint is None,
                pretrained_weights_path=provenance["weights_path"] if checkpoint is None else None,
                **config.model_config()).to(device, memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,
                    warmup_power=0.75, exclude_buffers=True, foreach=True)
    ema.module.requires_grad_(False)
    optimizer = build_optimizer(model, config, fused=dtype == torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    if checkpoint:
        model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
        ema.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng(checkpoint["rng_state"])
        del checkpoint
    student_weights = torch.tensor(loss_config["pos_weight"], device=device, dtype=torch.float32)
    write_json(out / "config.json", {**serialized, "model_config": config.model_config(),
                                   "preprocessing": student_preprocessing(), "distillation": protocol})
    write_json(out / "pretrained_provenance.json", provenance)
    stop_requested = False

    def request_stop(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Student stop requested: finishing the current epoch before checkpointing", flush=True)

    def after_step(_optimizer, _args, _kwargs):
        progress["optimizer_updates"] += 1
        ema.update(model, step=progress["optimizer_updates"])
        progress["ema_updates"] += 1

    hook = optimizer.register_step_post_hook(after_step)
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    state = "paused"
    last_hash = None
    last_audit = None
    try:
        for epoch in range(progress["completed_epochs"] + 1, config.epochs + 1):
            if stop_requested or (out / "STOP").exists():
                break
            options = {"image_root": root, "num_workers": config.num_workers,
                       "prefetch_factor": config.prefetch_factor,
                       "seed": (seed + epoch * 1009) % 2**32}
            loader = make_overnight_loader(train_records, training=True, image_size=512,
                                          batch_size=config.batch_size, **options)
            training = _train_epoch(model, ema, loader, optimizer, scaler, device, dtype,
                                     epoch, config, progress, cache, student_weights)
            del loader
            valid_loader = make_overnight_loader(valid_records, training=False, image_size=512,
                                                batch_size=config.valid_batch_size, **options)
            validation = evaluate(ema.module, valid_loader, valid_records, device, dtype)
            validation["metrics"]["tta"] = "identity512"
            del valid_loader
            for path, expected in ((config.fold_plan, plan_hash),
                    (Path(plan["source_train_manifest"]["path"]), plan["source_manifest_sha256"]),
                    (valid_path, plan["official_holdout"]["sha256"]),
                    (soft_file, protocol["soft_labels_sha256"]), (meta_file, protocol["soft_metadata_sha256"])):
                if sha256_file(path) != expected:
                    raise RuntimeError("A bound student input changed during training")
            auc = validation["metrics"]["macro_auc"]
            is_best = best_auc is None or auc > best_auc
            if is_best:
                best_auc, best_epoch = auc, epoch
            if early["reference_auc"] is None or auc > early["reference_auc"] + config.min_delta:
                early = {"reference_auc": auc, "bad_epochs": 0}
            else:
                early["bad_epochs"] += 1
            progress.update(completed_epochs=epoch, validated_epochs=progress["validated_epochs"] + 1)
            payload = {"schema_version": 7, "artifact_type": "kd_student", "model_name": TINY_MODEL_NAME,
                "model_config": config.model_config(), "class_ids": list(range(1, 11)), "num_classes": 10,
                "image_size": 512, "preprocessing": student_preprocessing(), "epoch": epoch,
                "weights_source": "ema", "training_precision": precision, "inference_precision": "float16",
                "inference_ready": True, "validation_complete": True, "validation": validation["metrics"],
                "official_valid_manifest_sha256": sha256_file(valid_path),
                "source_manifest_sha256": plan["source_manifest_sha256"], "fold_plan_sha256": plan_hash,
                "pretrained_provenance": provenance, "teacher_provenance": cache.metadata["teachers"],
                "distillation": protocol, "loss_config": loss_config, "thresholds": [0.5] * 10,
                "aggregation": "study_mean_probability", "deployment_tta": "identity", "deployment_ensemble": False,
                "validation_precision": precision, "deployment_validation_pending": True,
                "model_state_dict": cpu_tree(ema.module.state_dict()), "raw_model_state_dict": cpu_tree(model.state_dict()),
                "optimizer_state_dict": cpu_tree(optimizer.state_dict()), "scaler_state_dict": cpu_tree(scaler.state_dict()),
                "rng_state": capture_rng(), "progress": dict(progress), "early_stopping": dict(early),
                "best_auc": best_auc, "best_epoch": best_epoch, "training_config": serialized}
            if is_best:
                best_hash = save_checkpoint(out / "best.pt", payload)
                record_checkpoint(out / "best.pt", best_hash, epoch=epoch,
                                  role="best", source_version=source_version)
                write_json(out / "best_metrics.json", {"epoch": epoch, **validation["metrics"]})
            last_hash = save_checkpoint(out / "last.pt", payload)
            last_audit = record_checkpoint(out / "last.pt", last_hash, epoch=epoch,
                                           role="last", source_version=source_version)
            del payload
            with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, "training": training,
                                        "validation": validation["metrics"], "is_best": is_best,
                                        "last_checkpoint_sha256": last_hash,
                                        "best_checkpoint_sha256": best_hash,
                                        "checkpoint_saved_at_utc": last_audit["saved_at_utc"],
                                        "checkpoint_audit_record_sha256": last_audit["record_sha256"]}) + "\n")
            state = "completed" if epoch == config.epochs else "paused"
            if config.patience and early["bad_epochs"] >= config.patience:
                state = "early_stopped"
            write_json(out / "status.json", {"state": state, **progress, "best_auc": best_auc,
                "best_epoch": best_epoch, "last_checkpoint_sha256": last_hash,
                "best_checkpoint_sha256": best_hash,
                "updated_utc": datetime.now(timezone.utc).isoformat()})
            print(f"student epoch={epoch} valid_AUC={auc:.8f} best={best_auc:.8f}@{best_epoch}", flush=True)
            if state in ("completed", "early_stopped"):
                break
    except BaseException as exc:
        write_json(out / "status.json", {"state": "failed", **progress,
                    "incomplete_epoch": True, "error": f"{type(exc).__name__}: {exc}",
                    "resume_from": str(out / "last.pt")})
        raise
    finally:
        hook.remove()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    result = {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch}
    if last_hash is not None:
        result.update(last_checkpoint_sha256=last_hash, best_checkpoint_sha256=best_hash,
                      updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "status.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline OOF V1 Tiny512 KD; dry-run unless --execute")
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--soft-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly start training")
    for name in ("batch_size", "accumulation_steps", "valid_batch_size", "num_workers", "prefetch_factor",
                 "epochs", "patience", "seed", "warmup_epochs", "kd_warmup_epochs"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DistillConfig.__dataclass_fields__[name].default)
    for name in ("backbone_lr", "head_lr", "min_lr_ratio", "weight_decay", "dropout", "drop_path", "ema_decay",
                 "grad_clip", "alpha", "temperature", "min_delta"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DistillConfig.__dataclass_fields__[name].default)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--no-align-teacher-pos-weight", dest="align_teacher_pos_weight", action="store_false")
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    config = DistillConfig(**args)
    config.validate()
    if not execute:
        print(json.dumps({"dry_run": True, "training_started": False, "config": _configuration(config),
                          "model_config": config.model_config(), "preprocessing": student_preprocessing()}, indent=2))
        return 0
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(
            evidence_dir=config.output / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    print(json.dumps(run_training(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
