"""Explicit V1 Tiny/Base training with complete-Study probability-mean supervision.

The frozen teachers and original OOF cache are read-only inputs. Importing this
module never initializes a model, opens images, or downloads weights. Only
--execute starts training; all mutable run assets belong in a new local output.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, Dataset

from audit_trail import capture_source_version, record_checkpoint, training_output_has_artifacts
from training_hooks import compliance_run, start_run, epoch_completed
from kd_common import BASE_BACKBONE, TINY_BACKBONE, require_training_runtime, sha256_file, write_json
from train_distill import (DistillConfig, _configuration, alpha_at, distillation_losses,
                           evaluate_student, prepare_student_objective,
                           student_preprocessing_config, validate_output_paths,
                           load_student_ema_initialization, initialize_model_from_student_ema,
                           resume_training_configuration)

SCHEMA_VERSION = 10
MODEL_NAME = "convnext_v1_base_gap"


@dataclass(frozen=True)
class StudyConfig(DistillConfig):
    backbone_name: str = BASE_BACKBONE
    pack_images: int = 0
    batch_size: int = 1
    accumulation_steps: int = 32
    seed: int = 44
    image_size: int = 768
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    drop_path: float = 0.15
    amp_dtype: str = "float16"
    pos_weight_power: float = 0.5
    pretrained_provenance_path: Path | None = None
    pretrained_weights_path: Path | None = None

    def model_config(self) -> dict[str, Any]:
        from model_overnight import validate_model_config
        return validate_model_config({"backbone_name": self.backbone_name,
            "drop_rate": self.dropout, "drop_path_rate": self.drop_path,
            "gradient_checkpointing": True})

    def validate(self) -> None:
        super().validate()
        if type(self.pack_images) is not int or self.pack_images not in (0, 8):
            raise ValueError('Use the original single-Study execution or registered packing cap 8')
        if self.training_mode != "distillation" or self.soft_labels is None:
            raise ValueError("The complete-Study student requires frozen OOF distillation")
        if self.batch_size != 1 or self.accumulation_steps != 32:
            raise ValueError("P3 uses one complete Study per microbatch and 32 Studies per update")
        if self.pos_weight_power != 0.5 or self.image_size != 768 or self.amp_dtype != "float16":
            raise ValueError("P3 retains its precommitted sqrt weights, 768 input and FP16 training")
        if (self.init_student_ema is None and self.resume is None
                and (self.pretrained_provenance_path is None or self.pretrained_weights_path is None)):
            raise ValueError("P3 requires explicit offline original-pretrain materials and weights")

    def model_name(self):
        from model_overnight import TINY_MODEL_NAME
        return TINY_MODEL_NAME if self.backbone_name == TINY_BACKBONE else MODEL_NAME


def complete_study_groups(records: list[dict[str, Any]]) -> list[list[int]]:
    """Retain every image once, preserving first-seen Study and image order."""
    if not records:
        raise ValueError("A complete-Study dataset cannot be empty")
    groups: dict[str, list[int]] = {}
    labels: dict[str, tuple[Any, ...]] = {}
    image_ids: set[str] = set()
    for index, row in enumerate(records):
        key, image_id = row.get("study_key"), row.get("image_id")
        if row.get("split") != "train" or not isinstance(key, str) or not key:
            raise ValueError("Complete-Study training uses original train records only")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError("Complete-Study image IDs must be unique nonempty strings")
        target = row.get("targets")
        if target is None or len(target) != 10 or any(v not in (0, 1) for v in target):
            raise ValueError("Complete-Study training requires ten original binary labels")
        target = tuple(target)
        if key in labels and labels[key] != target:
            raise ValueError("Original labels disagree within a Study")
        labels[key] = target
        groups.setdefault(key, []).append(index)
        image_ids.add(image_id)
    return list(groups.values())


class CompleteStudyDataset(Dataset):
    """The existing immutable image loader, grouped without dropping views."""
    def __init__(self, records, image_root, *, image_size=768, seed=44):
        from dataset_overnight import OvernightImageDataset
        self.records = [dict(row) for row in records]
        self.groups = complete_study_groups(self.records)
        self.image_dataset = OvernightImageDataset(self.records, image_root,
            training=True, image_size=image_size, seed=seed)

    def __len__(self):
        return len(self.groups)

    def reseed(self, seed):
        self.image_dataset.reseed(seed)

    def __getitem__(self, index):
        from dataset_overnight import collate_overnight
        batch = collate_overnight([self.image_dataset[i] for i in self.groups[index]])
        if len(set(batch["study_keys"])) != 1:
            raise RuntimeError("A complete-Study microbatch mixed multiple Studies")
        return batch


def make_study_loader(records, image_root, *, image_size, num_workers, prefetch_factor,
                      seed, pin_memory=True):
    from dataset_overnight import _seed_worker
    if not 0 <= num_workers <= 12 or prefetch_factor not in (1, 2):
        raise ValueError("Invalid worker or prefetch configuration")
    dataset = CompleteStudyDataset(records, image_root, image_size=image_size, seed=seed)
    options: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory,
        "generator": torch.Generator().manual_seed(seed), "worker_init_fn": _seed_worker}
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False,
                       multiprocessing_context="spawn", timeout=180.0)
    # Disable automatic batching: one dataset element already contains all of a
    # Study's image tensors. default_convert preserves that complete dictionary.
    return DataLoader(dataset, batch_size=None, shuffle=True, **options)


def study_probability_bce(logits: torch.Tensor, targets: torch.Tensor,
                          pos_weights: torch.Tensor) -> torch.Tensor:
    """Weighted BCE of mean image probabilities, in stable FP32 log space."""
    if (logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 10
            or targets.shape != logits.shape or pos_weights.shape != (10,)):
        raise ValueError("Study loss expects Vx10 logits/labels and ten class weights")
    if not all(bool(torch.isfinite(v).all()) for v in (logits, targets, pos_weights)):
        raise FloatingPointError("Nonfinite Study loss input")
    if bool((pos_weights <= 0).any()) or bool(((targets != 0) & (targets != 1)).any()):
        raise ValueError("Study loss requires binary labels and positive weights")
    if not bool((targets == targets[0]).all()):
        raise ValueError("Study supervision cannot mix conflicting image labels")
    z, y, weight = logits.float(), targets[0].float(), pos_weights.float()
    log_n = math.log(logits.shape[0])
    log_p = torch.logsumexp(F.logsigmoid(z), dim=0) - log_n
    log_negative = torch.logsumexp(F.logsigmoid(-z), dim=0) - log_n
    return -(weight * y * log_p + (1 - y) * log_negative).mean()


def study_distillation_losses(logits, targets, teacher_logits, teacher_weights,
                              student_weights, *, temperature=2., align=True):
    hard = study_probability_bce(logits, targets, student_weights)
    _, per_image_kl = distillation_losses(logits, targets, teacher_logits,
        teacher_weights, student_weights, temperature=temperature, align=align)
    return hard, per_image_kl.mean()


def accumulation_window_size(index: int, total_studies: int, accumulation: int) -> int:
    if (type(index) is not int or type(total_studies) is not int or type(accumulation) is not int
            or not 0 <= index < total_studies or accumulation < 1):
        raise ValueError("Invalid Study accumulation window")
    return min(accumulation, total_studies - (index // accumulation) * accumulation)


def train_study_epoch(model, ema, loader, optimizer, scaler, device, dtype,
                      epoch, config, progress, cache, student_weights):
    if config.pack_images:
        from study_packing import train_packed_study_epoch
        return train_packed_study_epoch(model, ema, loader, optimizer, scaler, device,
            dtype, epoch, config, progress, cache, student_weights)
    from train_overnight import set_learning_rates
    model.train()
    expected_ids = {r["image_id"] for r in loader.dataset.records}
    expected_studies = {r["study_key"] for r in loader.dataset.records}
    expected_by_study: dict[str, set[str]] = {}
    for row in loader.dataset.records:
        expected_by_study.setdefault(row["study_key"], set()).add(row["image_id"])
    seen_ids, seen_studies = set(), set()
    total = hard_total = kl_total = 0.
    windows = math.ceil(len(loader) / config.accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    updates_before = progress["optimizer_updates"]
    for index, batch in enumerate(loader):
        ids, keys = batch["image_ids"], set(batch["study_keys"])
        if (len(keys) != 1 or len(ids) != len(set(ids)) or seen_ids.intersection(ids)
                or not set(ids) <= expected_ids or seen_studies.intersection(keys)):
            raise ValueError("Complete-Study image/Study epoch coverage is invalid")
        key = next(iter(keys))
        if key not in expected_studies or set(ids) != expected_by_study[key]:
            raise ValueError("A Study microbatch omitted or added an original view")
        seen_ids.update(ids)
        seen_studies.add(key)
        images = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        positions = [cache.index[image_id] for image_id in ids]
        teacher = torch.as_tensor(cache.logits[positions], device=device, dtype=torch.float32)
        teacher_weights = torch.as_tensor(cache.teacher_pos_weights[positions], device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(images)
        hard, soft = study_distillation_losses(logits, targets, teacher, teacher_weights,
            student_weights, temperature=config.temperature, align=config.align_teacher_pos_weight)
        alpha = alpha_at(config, epoch, index, len(loader))
        loss = (1 - alpha) * hard + alpha * soft
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite complete-Study loss")
        count = accumulation_window_size(index, len(loader), config.accumulation_steps)
        scaler.scale(loss / count).backward()
        progress["backward_batches"] += 1
        total += float(loss.detach())
        hard_total += float(hard.detach())
        kl_total += float(soft.detach())
        if (index + 1) % config.accumulation_steps == 0 or index + 1 == len(loader):
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
        if index == 0 or (index + 1) % 100 == 0:
            print(f"study_student epoch={epoch} studies={index+1}/{len(loader)} "
                  f"images={len(seen_ids)} loss={total/(index+1):.6f} alpha={alpha:.4f}", flush=True)
    if seen_ids != expected_ids or seen_studies != expected_studies:
        raise RuntimeError("Complete-Study epoch omitted an original image or Study")
    if progress["optimizer_updates"] <= updates_before or progress["optimizer_updates"] != progress["ema_updates"]:
        raise RuntimeError("Student optimizer and EMA did not advance together")
    return {"loss": total / len(loader), "hard_bce": hard_total / len(loader),
        "soft_kl_T2": kl_total / len(loader), "alpha_end": alpha,
        "images": len(seen_ids), "studies": len(seen_studies), "full_manifest_coverage": True,
        "effective_batch_unit": "Study", "effective_batch_size": 32,
        "seconds": time.monotonic() - started, "learning_rates": rates,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


def validate_study_resume(checkpoint, config, expected):
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Complete-Study resume protocol changed: {name}")
    current, prior = resume_training_configuration(config, checkpoint)
    prior.setdefault('backbone_name', BASE_BACKBONE)
    prior.setdefault('pack_images', 0)
    if current != prior:
        raise ValueError("Complete-Study resume configuration changed")
    for name in ("model_state_dict", "raw_model_state_dict", "optimizer_state_dict",
                 "scaler_state_dict", "rng_state", "progress", "early_stopping", "validation"):
        if name not in checkpoint:
            raise ValueError(f"Incomplete Study checkpoint: {name}")
    epoch, best_epoch = checkpoint.get("epoch"), checkpoint.get("best_epoch")
    progress, early = checkpoint["progress"], checkpoint["early_stopping"]
    for name in ("completed_epochs", "validated_epochs", "backward_batches", "optimizer_updates",
                 "ema_updates", "amp_skipped_steps"):
        if type(progress.get(name)) is not int or progress[name] < 0:
            raise ValueError(f"Invalid complete-Study progress counter: {name}")
    current_auc = checkpoint["validation"].get("macro_auc")
    if (type(epoch) is not int or not 1 <= epoch <= config.epochs
            or progress.get("completed_epochs") != epoch or progress.get("validated_epochs") != epoch
            or type(best_epoch) is not int or not 1 <= best_epoch <= epoch
            or not isinstance(checkpoint.get("best_auc"), (int, float))
            or not math.isfinite(checkpoint["best_auc"]) or not 0 <= checkpoint["best_auc"] <= 1
            or type(current_auc) not in (int, float) or not math.isfinite(current_auc)
            or not 0 <= current_auc <= checkpoint["best_auc"]
            or progress.get("optimizer_updates", 0) <= 0
            or progress.get("optimizer_updates") != progress.get("ema_updates")
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True or checkpoint.get("weights_source") != "ema"
            or type(early.get("bad_epochs")) is not int or not 0 <= early["bad_epochs"] <= epoch
            or not isinstance(early.get("reference_auc"), (int, float))
            or not math.isfinite(early["reference_auc"]) or not 0 <= early["reference_auc"] <= 1):
        raise ValueError("Inconsistent complete-Study continuation state")


@compliance_run
def run_training(config: StudyConfig):
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from dataset_overnight import make_overnight_loader
    from model_overnight import build_overnight_model
    from student_experiments import compute_student_loss_config, load_local_pretrained_provenance
    from train_overnight import build_optimizer, capture_rng, cpu_tree, restore_rng, save_checkpoint
    from timm.utils import ModelEmaV3

    config.validate()
    validate_output_paths(config)
    require_training_runtime()
    source_version = capture_source_version(Path(__file__).with_name(name) for name in (
        "study_student.py", "study_packing.py", "student_experiments.py", "train_distill.py", "train_overnight.py",
        "distill_data.py", "distill_tta_data.py", "model_overnight.py", "dataset_overnight.py",
        "dataset.py", "cv_splits.py", "kd_common.py", "runtime_compat.py", "audit_trail.py",
        "training_hooks.py", "training_compliance.py"))
    plan = read_fold_plan(config.fold_plan)
    train_path = Path(plan["source_train_manifest"]["path"])
    valid_path = Path(plan["official_holdout"]["path"])
    train_records, valid_records = read_manifest(train_path, split="train"), read_manifest(valid_path, split="valid")
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {r.get(key) for r in train_records} - {None, ""}
        right = {r.get(key) for r in valid_records} - {None, ""}
        if left & right:
            raise ValueError(f"Student train/valid leakage: {key}")
    complete_study_groups(train_records)
    objective = prepare_student_objective(config, train_records)
    cache = objective["cache"]
    protocol = {**objective["protocol"], "loss": "study_probability_mean_bce_plus_mean_image_bernoulli_kl_T2",
        "hard_target_unit": "complete_Study", "hard_prediction": "mean_image_sigmoid_probability",
        "soft_reduction": "mean_images_within_Study_then_mean_Studies",
        "effective_batch_unit": "Study", "effective_batch_size": 32, "all_Study_views_retained": True}
    if config.pack_images:
        protocol['forward_packing_preferred_images'] = config.pack_images
        protocol['forward_packing_splits_Studies'] = False
    loss_config = compute_student_loss_config(train_records,
        source_manifest_sha256=plan["source_manifest_sha256"], fold=None,
        pos_weight_power=config.pos_weight_power)
    loss_config = {**loss_config, "name": "StudyMeanProbabilityWeightedBCE",
        "reduction": "mean_classes_then_mean_complete_Studies",
        "study_equal_weighting": "one_loss_per_complete_Study",
        "sampling": "shuffle_Studies_without_replacement"}
    preprocessing = student_preprocessing_config(config)
    initial_state, student_warm_start = None, None
    if config.init_student_ema is not None:
        initial_state, provenance, student_warm_start = load_student_ema_initialization(
            config.init_student_ema, config, expected={
                'source_manifest_sha256': plan['source_manifest_sha256'],
                'fold_plan_sha256': sha256_file(config.fold_plan),
                'official_valid_manifest_sha256': plan['official_holdout']['sha256'],
                'teacher_provenance': objective['teacher_provenance']})
    elif config.resume is None:
        provenance = load_local_pretrained_provenance(config.pretrained_provenance_path,
            config.pretrained_weights_path, config.backbone_name)
    else:
        # Original pretrain/EMA paths are not required to resume a full new-run checkpoint.
        provenance = None
    bindings = [(config.fold_plan, sha256_file(config.fold_plan)),
        (train_path, plan["source_manifest_sha256"]),
        (valid_path, plan["official_holdout"]["sha256"]), *objective["cache_bindings"]]
    expected = {"schema_version": SCHEMA_VERSION, "artifact_type": "kd_student", "model_name": config.model_name(),
        "model_config": config.model_config(), "image_size": config.image_size,
        "source_manifest_sha256": plan["source_manifest_sha256"], "fold_plan_sha256": bindings[0][1],
        "official_valid_manifest_sha256": bindings[2][1], "preprocessing": preprocessing,
        "distillation": protocol, "loss_config": loss_config, "training_precision": "float16",
        "validation_tta": config.validation_tta, "teacher_provenance": objective["teacher_provenance"],
        "pretrained_provenance": provenance}
    if not torch.cuda.is_available():
        raise RuntimeError("Base Study training requires a CUDA GPU")
    device, dtype = torch.device("cuda", 0), torch.float16
    torch.cuda.set_device(device)
    out = config.output.resolve()
    root = config.image_root.resolve(strict=True)
    if config.resume is None:
        if training_output_has_artifacts(out):
            raise FileExistsError("New Base student training requires an empty run directory")
        out.mkdir(parents=True, exist_ok=True)
    elif config.resume.resolve().parent != out or config.resume.name != "last.pt":
        raise ValueError("Resume must name last.pt in the same local run directory")
    start_run(out, trainer="study_student.run_training", configuration={
        **_configuration(config), "binding": expected,
        "source_version": source_version}, source_files=source_version["files"], resume=config.resume)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
        "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    early, best_auc, best_epoch = {"reference_auc": None, "bad_epochs": 0}, None, None
    checkpoint, best_hash, last_hash = None, None, None
    if config.resume is not None:
        source = config.resume if config.resume.is_file() else out / "best.pt"
        digest = sha256_file(source)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        provenance = checkpoint.get('pretrained_provenance')
        from kd_common import validate_pretrained_provenance
        validate_pretrained_provenance(provenance, config.backbone_name)
        expected['pretrained_provenance'] = provenance
        student_warm_start = checkpoint.get('student_warm_start')
        if sha256_file(source) != digest:
            raise RuntimeError("Base resume checkpoint changed while loading")
        validate_study_resume(checkpoint, config, expected)
        if not (out / "best.pt").is_file():
            raise FileNotFoundError("The actual best Base checkpoint is required for continuation")
        best_hash = sha256_file(out / "best.pt")
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        if sha256_file(out / "best.pt") != best_hash:
            raise RuntimeError("Base best checkpoint changed while loading")
        validate_study_resume(best, config, expected)
        if best["epoch"] != best["best_epoch"] or best["validation"]["macro_auc"] != best["best_auc"]:
            raise ValueError("Base best checkpoint does not contain its claimed best epoch")
        if best["epoch"] > checkpoint["epoch"]:
            if best["best_auc"] < checkpoint["best_auc"]:
                raise ValueError("A newer best checkpoint has a lower best AUC")
            checkpoint = best
            last_hash = save_checkpoint(out / "last.pt", checkpoint)
            record_checkpoint(out / "last.pt", last_hash, epoch=checkpoint["epoch"],
                              role="last", source_version=source_version)
        elif best["epoch"] != checkpoint["best_epoch"] or best["best_auc"] != checkpoint["best_auc"]:
            raise ValueError("Base best and last checkpoint bindings disagree")
        progress.update(checkpoint["progress"])
        early = checkpoint["early_stopping"]
        best_auc, best_epoch = checkpoint["best_auc"], checkpoint["best_epoch"]
        del best
        if progress["completed_epochs"] >= config.epochs or config.patience and early["bad_epochs"] >= config.patience:
            state = "completed" if progress["completed_epochs"] >= config.epochs else "early_stopped"
            result = {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch,
                      "best_checkpoint_sha256": best_hash, "last_checkpoint_sha256": sha256_file(out / "last.pt")}
            write_json(out / "status.json", result)
            return result
    use_pretrained = checkpoint is None and initial_state is None
    model = build_overnight_model(pretrained=use_pretrained,
        pretrained_weights_path=provenance["weights_path"] if use_pretrained else None,
        **config.model_config()).to(device, memory_format=torch.channels_last)
    if initial_state is not None:
        initialize_model_from_student_ema(model, initial_state)
        del initial_state
    ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,
                    warmup_power=0.75, exclude_buffers=True, foreach=True)
    ema.module.requires_grad_(False)
    optimizer = build_optimizer(model, config, fused=False)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
        ema.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng(checkpoint["rng_state"])
        del checkpoint
    student_weights = torch.tensor(loss_config["pos_weight"], device=device, dtype=torch.float32)
    write_json(out / "config.json", {**_configuration(config), "model_config": config.model_config(),
        "preprocessing": preprocessing, "distillation": protocol, "loss_config": loss_config,
        **({'student_warm_start': student_warm_start} if student_warm_start else {})})
    write_json(out / "pretrained_provenance.json", provenance)
    stop_requested = False

    def request_stop(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Base stop requested: finish current epoch and save full continuation state", flush=True)

    def after_step(_optimizer, _args, _kwargs):
        progress["optimizer_updates"] += 1
        ema.update(model, step=progress["optimizer_updates"])
        progress["ema_updates"] += 1

    hook = optimizer.register_step_post_hook(after_step)
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    state = "running"
    try:
        write_json(out / "status.json", {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch})
        for epoch in range(progress["completed_epochs"] + 1, config.epochs + 1):
            if stop_requested or (out / "STOP").exists():
                state = "paused"
                break
            options = {"image_root": root, "num_workers": config.num_workers,
                "prefetch_factor": config.prefetch_factor, "seed": (config.seed + epoch * 1009) % 2**32}
            loader = make_study_loader(train_records, image_size=config.image_size, **options)
            training = train_study_epoch(model, ema, loader, optimizer, scaler, device,
                dtype, epoch, config, progress, cache, student_weights)
            del loader
            valid_loader = make_overnight_loader(valid_records, training=False,
                image_size=config.image_size, batch_size=config.valid_batch_size, **options)
            validation = evaluate_student(ema.module, valid_loader, valid_records, device, dtype, config)
            del valid_loader
            for path, digest in bindings:
                if sha256_file(path) != digest:
                    raise RuntimeError("A frozen Base student input changed during training")
            auc = validation["metrics"]["macro_auc"]
            is_best = best_auc is None or auc > best_auc
            if is_best:
                best_auc, best_epoch = auc, epoch
            if early["reference_auc"] is None or auc > early["reference_auc"] + config.min_delta:
                early = {"reference_auc": auc, "bad_epochs": 0}
            else:
                early["bad_epochs"] += 1
            progress.update(completed_epochs=epoch, validated_epochs=progress["validated_epochs"] + 1)
            payload = {**expected, "class_ids": list(range(1, 11)), "num_classes": 10,
                "epoch": epoch, "weights_source": "ema", "inference_precision": "float16",
                "inference_ready": True, "validation_complete": True, "validation": validation["metrics"],
                "thresholds": [0.5] * 10, "aggregation": "study_mean_probability",
                "deployment_tta": config.validation_tta, "deployment_ensemble": False,
                "validation_precision": "float16", "deployment_validation_pending": True,
                "model_state_dict": cpu_tree(ema.module.state_dict()), "raw_model_state_dict": cpu_tree(model.state_dict()),
                "optimizer_state_dict": cpu_tree(optimizer.state_dict()), "scaler_state_dict": cpu_tree(scaler.state_dict()),
                "rng_state": capture_rng(), "progress": dict(progress), "early_stopping": dict(early),
                "best_auc": best_auc, "best_epoch": best_epoch, "training_config": _configuration(config),
                "training_mode": "distillation", "source_version": source_version}
            if student_warm_start is not None:
                payload['student_warm_start'] = student_warm_start
            if is_best:
                best_hash = save_checkpoint(out / "best.pt", payload)
                record_checkpoint(out / "best.pt", best_hash, epoch=epoch, role="best", source_version=source_version)
                write_json(out / "best_metrics.json", {"epoch": epoch, **validation["metrics"]})
            last_hash = save_checkpoint(out / "last.pt", payload)
            audit = record_checkpoint(out / "last.pt", last_hash, epoch=epoch, role="last", source_version=source_version)
            del payload
            with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, "training": training, "validation": validation["metrics"],
                    "is_best": is_best, "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash,
                    "checkpoint_saved_at_utc": audit["saved_at_utc"], "checkpoint_audit_record_sha256": audit["record_sha256"]}) + "\n")
            epoch_completed(epoch, {"training": training, "validation": validation["metrics"],
                "is_best": is_best, "progress": dict(progress),
                "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash})
            state = "completed" if epoch == config.epochs else "running"
            if config.patience and early["bad_epochs"] >= config.patience:
                state = "early_stopped"
            write_json(out / "status.json", {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch,
                "last_checkpoint_sha256": last_hash, "best_checkpoint_sha256": best_hash,
                "updated_utc": datetime.now(timezone.utc).isoformat()})
            print(f"study_student epoch={epoch} valid_AUC={auc:.8f} best={best_auc:.8f}@{best_epoch}", flush=True)
            if state in ("completed", "early_stopped"):
                break
    except BaseException as exc:
        write_json(out / "status.json", {"state": "failed", **progress, "incomplete_epoch": True,
            "error": f"{type(exc).__name__}: {exc}", "resume_from": str(out / "last.pt")})
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fold_plan", "soft_labels", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ('pretrained_provenance_path', 'pretrained_weights_path', 'init_student_ema'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path)
    parser.add_argument('--init-student-ema-sha256')
    parser.add_argument('--backbone-name', choices=(BASE_BACKBONE, TINY_BACKBONE), default=BASE_BACKBONE)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    for name in ("num_workers", "prefetch_factor", "valid_batch_size", "epochs", "patience", "seed",
                 'warmup_epochs', 'kd_warmup_epochs', 'pack_images'):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=StudyConfig.__dataclass_fields__[name].default)
    for name in ('backbone_lr', 'head_lr', 'alpha', 'drop_path', 'min_lr_ratio'):
        parser.add_argument('--' + name.replace('_', '-'), type=float,
                            default=StudyConfig.__dataclass_fields__[name].default)
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    config = StudyConfig(**args)
    config.validate()
    if not execute:
        print(json.dumps({"dry_run": True, "training_started": False, "config": _configuration(config),
            "schema_version": SCHEMA_VERSION, "model_config": config.model_config(),
            "hard_supervision": "BCE(mean_image_sigmoid_probability, original_Study_label)",
            "effective_batch_unit": "Study", "effective_batch_size": 32}, indent=2))
        return 0
    validate_output_paths(config)
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=config.output / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]))
    print(json.dumps(run_training(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
