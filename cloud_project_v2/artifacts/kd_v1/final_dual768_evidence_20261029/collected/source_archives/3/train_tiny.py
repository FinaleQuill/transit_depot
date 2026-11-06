"""ConvNeXt-Tiny/GAP/Linear、完整逐图遍历、Study等权BCE的云端入口。

仅显式执行main才创建模型、读取原图或训练；编辑/语法检查不会执行任务。
原始图像与清洗清单只读；valid完整覆盖才选模、调阈值或推进早停。
历史train_highres及schema 3保持兼容，新权重使用独立schema 4。
"""
from __future__ import annotations

from runtime_compat import prepare_runtime

prepare_runtime()

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import signal
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset import (
    CLASS_IDS, DEFAULT_TRAIN_MANIFEST, DEFAULT_VALID_MANIFEST, IMAGENET_MEAN,
    IMAGENET_STD, NUM_CLASSES, PROJECT_ROOT, StudyPredictionAggregator, read_manifest,
)
from dataset_highres import RAW_ROOT, make_image_loader, resolve_image_root
from dataset_tiny import make_tiny_train_loader, train_study_statistics
from model_tiny import MODEL_CONFIG_KEYS, TINY_MODEL_NAME, build_tiny_model, validate_model_config
from train import (
    EarlyStopping, _aggregate_clean_train, _jsonable, _metrics_imports, _metric_text,
    _now, _safe_eval_batches, _save_checkpoint, _sha256, _unavailable_metrics,
    _write_json, calculate_metrics, search_class_thresholds,
)


@dataclass(frozen=True)
class TinyConfig:
    output: Path
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST
    valid_manifest: Path = DEFAULT_VALID_MANIFEST
    raw_root: Path = RAW_ROOT
    work_root: Path = PROJECT_ROOT
    backbone_name: str = TINY_MODEL_NAME
    image_size: int = 384
    batch_size: int = 32  # 图片数；不再是Study bag数。
    accumulation_steps: int = 1
    valid_batch_size: int = 64
    epochs: int = 25
    warmup_epochs: int = 1
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.01
    dropout: float = 0.1
    drop_path: float = 0.05
    gradient_checkpointing: bool = False
    amp_dtype: str = "auto"
    ema_decay: float = 0.999
    pos_weight_power: float = 0.0
    pos_weight_cap: float = 3.0
    num_workers: int = 12
    prefetch_factor: int = 2
    worker_timeout: float = 120.0
    train_eval_every: int = 0
    grad_clip: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 43
    device: str = "cuda"
    stop_file: Path | None = None
    resume: Path | None = None
    resume_lr_factor: float = 0.1

    def model_config(self) -> dict[str, Any]:
        return validate_model_config({
            "backbone_name": self.backbone_name, "drop_rate": self.dropout,
            "drop_path_rate": self.drop_path,
            "gradient_checkpointing": self.gradient_checkpointing,
        })

    def validate(self) -> None:
        if self.image_size != 384:
            raise ValueError("本轮固定原图384，不能与旧300缓存或512协议混用")
        for name in ("batch_size", "accumulation_steps", "valid_batch_size", "epochs", "prefetch_factor"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}必须为正整数")
        for name in ("num_workers", "train_eval_every", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name}必须为非负整数")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs必须在[0,epochs)内")
        for name in ("backbone_lr", "head_lr", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}必须为有限正数")
        for name in ("weight_decay", "worker_timeout", "min_delta", "pos_weight_power"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}必须为有限非负数")
        if not math.isfinite(self.pos_weight_cap) or self.pos_weight_cap < 1:
            raise ValueError("pos_weight_cap必须是至少1的有限数值")
        if not math.isfinite(self.min_lr_ratio) or not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio必须在(0,1]内")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay必须在[0,1)内")
        if self.amp_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("amp_dtype只支持auto/bfloat16/float16")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed必须为[0,2**32)整数")
        if not math.isfinite(self.resume_lr_factor) or not 0 < self.resume_lr_factor <= 1:
            raise ValueError("resume_lr_factor必须在(0,1]内")
        self.model_config()


def preprocessing_config() -> dict[str, Any]:
    return {"image_size": 384, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
            "cache_mode": "RGB", "data_source": "raw_readonly",
            "eval_transform": "EXIF+RGB+LetterboxLANCZOS+Normalize", "tta": "identity"}


def preprocessing_sha256(preprocessing: dict[str, Any]) -> str:
    payload = json.dumps(preprocessing, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_warm_restart(config: TinyConfig, source_hashes: dict[str, str]
                         ) -> tuple[TinyConfig, dict[str, Any] | None, dict[str, Any] | None]:
    """只在CPU读取/校验检查点；无模型构建、图片读取、优化或输出目录写入。

    schema4旧包未保存AdamW、GradScaler或RNG状态，因此--resume明确表示权重
    热重启：继承raw/EMA与累计进度，重置优化器、局部余弦和早停耐心。
    """
    if config.resume is None:
        return config, None, None
    path = config.resume.resolve(strict=True)
    checkpoint_sha256 = _sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if _sha256(path) != checkpoint_sha256:
        raise RuntimeError("读取期间源检查点发生变化，拒绝使用不一致的续训来源")
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 4
            or checkpoint.get("model_name") != "convnext_tiny_gap_linear"
            or checkpoint.get("num_classes") != NUM_CLASSES
            or checkpoint.get("class_ids") != list(CLASS_IDS)):
        raise ValueError("--resume仅支持类别顺序一致的schema4 ConvNeXt-Tiny检查点")
    saved_model_config = checkpoint.get("model_config")
    if (not isinstance(saved_model_config, dict) or set(saved_model_config) != set(MODEL_CONFIG_KEYS)
            or checkpoint.get("aggregation") != "study_mean_probability"
            or checkpoint.get("preprocessing") != preprocessing_config()
            or validate_model_config(saved_model_config) != config.model_config()):
        raise ValueError("续训模型配置/预处理/Study聚合必须与源检查点一致")
    if (checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True
            or any(checkpoint.get(key) != "ema" for key in
                   ("weights_source", "validation_weights_source", "threshold_weights_source"))):
        raise ValueError("--resume要求完整验证且指标/阈值绑定EMA的检查点")
    for split, digest in source_hashes.items():
        if checkpoint.get(f"{split}_manifest_sha256") != digest:
            raise ValueError(f"续训{split}清单SHA256与源检查点不一致，禁止更换数据划分")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 1 or config.epochs <= epoch:
        raise ValueError("--epochs为全局最后一轮编号，必须大于源检查点epoch")
    progress = checkpoint.get("progress", {})
    counters = ("completed_epochs", "validated_epochs", "backward_batches",
                "optimizer_updates", "ema_updates", "amp_skipped_steps")
    if (any(type(progress.get(key)) is not int or progress[key] < 0 for key in counters)
            or progress["completed_epochs"] != epoch or not 0 < progress["validated_epochs"] <= epoch
            or not progress["optimizer_updates"] == progress["ema_updates"] > 0):
        raise ValueError("源检查点缺少一致的完整轮/优化器/EMA累计更新计数")
    ema = checkpoint.get("ema", {})
    if (ema.get("enabled") is not True or ema.get("use_warmup") is not True
            or ema.get("warmup_power") != 0.75 or ema.get("buffers") != "copy_from_raw"
            or ema.get("updates") != progress["ema_updates"]
            or ema.get("decay") != config.ema_decay):
        raise ValueError("源EMA配置或更新计数不兼容；不能重新预热并覆盖已有EMA")
    raw, averaged = checkpoint.get("raw_model_state_dict"), checkpoint.get("model_state_dict")
    if (not isinstance(raw, dict) or not isinstance(averaged, dict)
            or not raw or raw.keys() != averaged.keys()
            or any(not isinstance(raw[key], torch.Tensor)
                   or not isinstance(averaged[key], torch.Tensor)
                   or raw[key].shape != averaged[key].shape
                   or raw[key].dtype != averaged[key].dtype for key in raw)):
        raise ValueError("源检查点必须同时含结构一致的在线模型与EMA权重")
    validation = checkpoint.get("validation", {})
    if any(type(validation.get(key)) not in (int, float)
           or not math.isfinite(validation[key]) or not 0 <= validation[key] <= 1
           for key in ("macro_auc", "macro_f1")):
        raise ValueError("源检查点缺少有效的验证AUC/F1，不能作为续训best基线")
    thresholds = checkpoint.get("thresholds")
    if (not isinstance(thresholds, list) or len(thresholds) != NUM_CLASSES
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in thresholds)):
        raise ValueError("源检查点缺少有效的同源验证阈值")
    source_config = checkpoint.get("training_config", {})
    source_lrs = [source_config.get(name) for name in ("backbone_lr", "head_lr")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in source_lrs):
        raise ValueError("源检查点缺少有效的基础学习率，不能推算热重启学习率")
    loss = checkpoint.get("loss_config", {})
    if (loss.get("name") != "BCEWithLogitsLoss" or loss.get("pos_weight") is not None
            or source_config.get("pos_weight_power") != 0 or config.pos_weight_power != 0):
        raise ValueError("本热重启策略保留原普通BCE，不同时引入pos_weight或更换损失")
    precision = checkpoint.get("inference_precision")
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("源检查点缺少有效AMP精度")
    config = replace(config, resume=path,
        backbone_lr=source_lrs[0] * config.resume_lr_factor,
        head_lr=source_lrs[1] * config.resume_lr_factor,
        warmup_epochs=0, patience=min(config.patience, 3) if config.patience else 3,
        amp_dtype=precision)
    config.validate()
    origin = {"kind": "weights_warm_restart", "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha256, "epoch": epoch, "start_epoch": epoch + 1,
        "baseline_auc": validation["macro_auc"], "baseline_f1": validation["macro_f1"],
        "raw_weights": "raw_model_state_dict", "ema_weights": "model_state_dict",
        "source_optimizer_updates": progress["optimizer_updates"],
        "lr_factor_of_source_base": config.resume_lr_factor,
        "backbone_lr": config.backbone_lr, "head_lr": config.head_lr,
        "schedule_epochs": config.epochs - epoch, "schedule_updates_reset": True,
        "optimizer_reset": True, "scaler_reset": True, "rng_restored": False,
        "early_stopping_bad_epochs_reset": True, "exact_resume": False,
        "policy": "lower LR; local cosine without warmup; patience<=3; unchanged BCE/augmentation"}
    return config, checkpoint, origin


def restore_warm_restart_weights(model: nn.Module, ema: Any, checkpoint: dict[str, Any]) -> None:
    """严格分别恢复在线模型与EMA；不调用forward或optimizer.step。"""
    model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
    ema.module.load_state_dict(checkpoint["model_state_dict"], strict=True)


def seed_resume_baseline(output: Path, checkpoint: dict[str, Any], origin: dict[str, Any]) -> None:
    """保留源best为新run基线；即使后续全部退步也不丢失可用的最优权重。

    不冒称重新验证过源模型，重写绑定哈希但保留checkpoint原训练配置。
    """
    record = {"epoch": checkpoint["epoch"], "phase": "inherited_resume_baseline",
        "inherited": True, "resume_origin": origin, "val": checkpoint["validation"],
        "validation_complete": True, "weights_source": "ema", "thresholds": checkpoint["thresholds"],
        "is_best": True, "is_best_f1": True, "metrics_unit": "Study"}
    save_bundle(output, checkpoint, record)
    save_bundle(output / "best_f1", checkpoint, record)


def append_run_worklog(work_root: Path, run_id: str, message: str) -> None:
    with (work_root / "WORKLOG.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n### {_now()} · {run_id}\n\n- {message}\n")


def choose_precision(requested: str) -> tuple[torch.dtype, str]:
    """仅在用户真正启动后查询当前CUDA设备能力，不用模型试跑探测。"""
    bf16_supported = torch.cuda.is_bf16_supported()
    if requested == "bfloat16" and not bf16_supported:
        raise RuntimeError("当前设备不支持BF16；可显式选择--amp-dtype float16")
    use_bf16 = requested == "bfloat16" or (requested == "auto" and bf16_supported)
    return (torch.bfloat16, "bfloat16") if use_bf16 else (torch.float16, "float16")


def positive_weights(statistics: dict[str, Any], config: TinyConfig) -> list[float] | None:
    """默认普通BCE；可选权重严格只来自训练清单的Study阳性数。"""
    if config.pos_weight_power == 0:
        return None
    positive = np.asarray(statistics["positive_studies"], dtype=np.float64)
    negative = np.asarray(statistics["negative_studies"], dtype=np.float64)
    if (positive.shape != (NUM_CLASSES,) or negative.shape != (NUM_CLASSES,)
            or (positive <= 0).any() or (negative <= 0).any()):
        raise ValueError("训练清单每类必须有正负Study，不能为缺失类别伪造pos_weight")
    values = np.clip((negative / positive) ** config.pos_weight_power, 1, config.pos_weight_cap)
    return values.tolist()


def build_optimizer(model: nn.Module, config: TinyConfig, *, fused: bool) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    if not head_ids:
        raise ValueError("GAP+Linear分类头参数为空")
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        role = "head" if id(parameter) in head_ids else "backbone"
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        buckets.setdefault((role, decay), []).append(parameter)
    groups = [{"params": parameters, "role": role,
               "name": f"{role}_{'decay' if decay else 'no_decay'}",
               "lr": config.head_lr if role == "head" else config.backbone_lr,
               "weight_decay": config.weight_decay if decay else 0.0}
              for (role, decay), parameters in buckets.items()]
    # FP16 GradScaler必须在发生溢出时完全跳过optimizer.step，EMA的post hook才可信。
    # fused AdamW可能在内部处理found_inf而仍调用hook，因此只在BF16无scaler时融合。
    return torch.optim.AdamW(groups, weight_decay=0.0, fused=fused, foreach=not fused)


def set_learning_rates(optimizer: torch.optim.Optimizer, config: TinyConfig,
                       updates: int, windows_per_epoch: int, *,
                       schedule_epochs: int | None = None) -> dict[str, float]:
    """按本次成功更新数推进；热重启使用剩余轮数，不消耗旧调度进度。"""
    total_updates = (config.epochs if schedule_epochs is None else schedule_epochs) * windows_per_epoch
    warmup_updates = config.warmup_epochs * windows_per_epoch
    if warmup_updates and updates < warmup_updates:
        factor = 0.1 + 0.9 * updates / warmup_updates
    else:
        cosine_updates = max(1, total_updates - warmup_updates - 1)
        fraction = min(1.0, max(0.0, (updates - warmup_updates) / cosine_updates))
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    for group in optimizer.param_groups:
        base = config.head_lr if group["role"] == "head" else config.backbone_lr
        group["lr"] = base * factor
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def train_epoch(model: nn.Module, loader: Any, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, scaler: Any, device: torch.device,
                amp_dtype: torch.dtype, epoch: int, config: TinyConfig,
                progress: dict[str, int], *, schedule_epochs: int | None = None) -> dict[str, Any]:
    """每张训练图恰好一次；不足32的尾批保留，累积按窗口实际图片数归一。"""
    model.train()
    loader.dataset.reseed((config.seed + epoch * 1009) % 2**32)
    expected = {record["image_id"] for record in loader.dataset.records}
    if len(expected) != len(loader.dataset):
        raise ValueError("训练清单含重复image_id，不能声明完整唯一遍历")
    seen: set[str] = set()
    studies: set[str] = set()
    optimizer.zero_grad(set_to_none=True)
    windows = math.ceil(len(loader) / config.accumulation_steps)
    start_updates = progress["optimizer_updates"]
    total_loss = total_plain_bce = total_weights = 0.0
    image_count = 0
    last_rates = set_learning_rates(optimizer, config, progress["schedule_updates"], windows,
                                    schedule_epochs=schedule_epochs)
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        count = len(batch["image_ids"])
        if (batch["split"] != "train" or batch["is_mixed"] or not 0 < count <= config.batch_size
                or batch["targets"].shape != (count, NUM_CLASSES)
                or batch["images"].shape != (count, 3, 384, 384)):
            raise ValueError("训练只允许原清单硬标签逐图批次，尾批可小于batch_size")
        batch_ids = set(batch["image_ids"])
        if len(batch_ids) != count or seen.intersection(batch_ids) or not batch_ids <= expected:
            raise ValueError("训练发现重复/额外图像，shuffle必须每轮无放回完整遍历")
        weights = batch["loss_weights"]
        if weights.shape != (count,) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("每张图必须携带有限正Study等权loss_weight")
        seen.update(batch_ids)
        studies.update(batch["study_keys"])
        images = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        device_weights = weights.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(images)
        if logits.shape != targets.shape or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("训练logits形状异常或出现NaN/Inf")
        # BCE在FP32计算；权重N/(S*n_views)为清单固定量，不能按随机batch再归一。
        per_image = criterion(logits.float(), targets).mean(dim=1)
        weighted_sum = (per_image * device_weights).sum()
        if not bool(torch.isfinite(weighted_sum)):
            raise FloatingPointError("训练BCE出现NaN/Inf")
        window_first_batch = batch_index // config.accumulation_steps * config.accumulation_steps
        window_images = min(config.batch_size * config.accumulation_steps,
                            len(loader.dataset) - window_first_batch * config.batch_size)
        scaler.scale(weighted_sum / window_images).backward()
        progress["backward_batches"] += 1
        plain_bce = nn.functional.binary_cross_entropy_with_logits(logits.detach().float(), targets)
        total_loss += float(weighted_sum.detach())
        total_plain_bce += float(plain_bce.detach()) * count
        total_weights += float(weights.sum())
        image_count += count
        del images, targets, device_weights, logits, per_image, weighted_sum, plain_bce
        if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(loader):
            last_rates = set_learning_rates(optimizer, config, progress["schedule_updates"], windows,
                                            schedule_epochs=schedule_epochs)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                     error_if_nonfinite=not scaler.is_enabled())
            before = progress["optimizer_updates"]
            scaler.step(optimizer)
            scaler.update()
            if progress["optimizer_updates"] == before:
                progress["amp_skipped_steps"] += 1
            optimizer.zero_grad(set_to_none=True)
        if batch_index == 0 or (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            print(f"Epoch {epoch} | images batch {batch_index + 1}/{len(loader)} | "
                  f"Study-weighted BCE={total_loss / image_count:.6f} "
                  f"images={image_count}/{len(loader.dataset)} updates={progress['optimizer_updates']} "
                  f"EMA={progress['ema_updates']} GPU_peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
                  flush=True)
    if seen != expected or image_count != len(loader.dataset):
        raise RuntimeError("训练epoch未覆盖清单全部图片，不能声明完整遍历")
    if not math.isclose(total_weights, len(loader.dataset), rel_tol=2e-6, abs_tol=1e-3):
        raise RuntimeError("全轮Study权重之和异常，不能确认Study等权BCE")
    return {"loss": total_loss / image_count, "bce_unweighted": total_plain_bce / image_count,
            "image_draws": image_count, "unique_images": len(seen), "unique_studies": len(studies),
            "full_manifest_coverage": True, "shuffle": True, "replacement": False, "drop_last": False,
            "loss_weight_sum": total_weights, "updates_this_epoch": progress["optimizer_updates"] - start_updates,
            "scheduled_windows": windows, "learning_rates": last_rates,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


@torch.inference_mode()
def evaluate_tiny(model: nn.Module, loader: Any, device: torch.device,
                  amp_dtype: torch.dtype, *, manifest: Path, split: str,
                  epoch: int, diagnostic_path: Path) -> dict[str, Any]:
    """完整图片概率按原Study求均值，缺任一视角则不给正式指标或阈值。"""
    model.eval()
    aggregator = StudyPredictionAggregator(manifest, split="valid") if split == "valid" else None
    ids: list[str] = []
    probabilities, targets = [], []
    skipped: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    count = 0
    for batch in _safe_eval_batches(loader, skipped, events, diagnostic_path=diagnostic_path,
                                    epoch=epoch, split=split):
        if batch["split"] != split or batch["is_mixed"] or batch["targets"] is None:
            raise ValueError("评估只允许原始硬标签与无随机增强输入")
        images = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(images)
        if logits.shape != (len(batch["image_ids"]), NUM_CLASSES) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("评估logits形状异常或出现NaN/Inf")
        values = torch.sigmoid(logits.float()).cpu()
        if aggregator is not None:
            aggregator.update(batch, values)
        else:
            ids.extend(batch["image_ids"])
            probabilities.append(values.numpy())
            targets.append(batch["targets"].numpy())
        count += len(batch["image_ids"])
    complete = count == len(loader.dataset) and not skipped
    diagnostics = {"complete": complete, "evaluated_images": count,
        "expected_images": len(loader.dataset), "skipped_samples": skipped,
        "fallback_serial": any(event["event"] == "loader_fallback_serial" for event in events),
        "loss": None, "loss_unit": "Study", "loader_events": events}
    if not complete:
        return diagnostics
    result = aggregator.finalize() if aggregator is not None else _aggregate_clean_train(
        loader.dataset.records, ids, np.concatenate(probabilities), np.concatenate(targets))
    p = np.clip(result["probabilities"].astype(np.float64), 1e-7, 1 - 1e-7)
    y = result["targets"]
    diagnostics["loss"] = float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())
    result.update(diagnostics)
    return result


def save_bundle(directory: Path, checkpoint_payload: dict[str, Any], record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    digest = _save_checkpoint(directory / "best.pt", checkpoint_payload)
    binding = {"schema_version": 4, "epoch": checkpoint_payload["epoch"],
        "checkpoint_sha256": digest, "checkpoint_filename": "best.pt",
        "preprocessing_sha256": preprocessing_sha256(checkpoint_payload["preprocessing"]),
        "weights_source": "ema", "inference_precision": checkpoint_payload["inference_precision"]}
    _write_json(directory / "thresholds.json", {**binding, "class_ids": list(CLASS_IDS),
        "thresholds": checkpoint_payload["thresholds"], "aggregation": "study_mean_probability",
        "selection": "complete valid Study metrics; thresholds tuned on same valid"})
    _write_json(directory / "best_metrics.json", {**record, **binding})


def run_training(config: TinyConfig) -> dict[str, Any]:
    config.validate()
    source_hashes = {"train": _sha256(config.train_manifest), "valid": _sha256(config.valid_manifest)}
    config, resume_checkpoint, resume_origin = prepare_warm_restart(config, source_hashes)
    resume_epoch = 0 if resume_checkpoint is None else resume_checkpoint["epoch"]
    schedule_epochs = config.epochs - resume_epoch
    _metrics_imports()
    output, work_root = config.output.resolve(), config.work_root.resolve()
    requested_root = config.raw_root.resolve()
    raw_root = resolve_image_root(requested_root, split="train")
    if work_root.is_relative_to(requested_root) or work_root.is_relative_to(raw_root):
        raise ValueError("work_root不能位于只读原图根内")
    runs_root = (work_root / "runs").resolve()
    if (runs_root != work_root / "runs" or output == runs_root
            or not output.is_relative_to(runs_root) or output.exists()):
        raise ValueError("output必须是work_root/runs内不存在的新目录，禁止覆盖旧实验")
    train_records = read_manifest(config.train_manifest, split="train")
    valid_records = read_manifest(config.valid_manifest, split="valid")
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {row.get(key) for row in train_records} - {None, ""}
        right = {row.get(key) for row in valid_records} - {None, ""}
        if left & right:
            raise ValueError(f"train/valid的{key}发生交叉")
    statistics = train_study_statistics(config.train_manifest)
    selected_pos_weights = positive_weights(statistics, config)
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("云端Tiny入口要求CUDA PyTorch；不自动回退CPU或关闭AMP")
    device = torch.device("cuda", 0) if device.index is None else device
    torch.cuda.set_device(device)
    amp_dtype, precision_name = choose_precision(config.amp_dtype)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config_dict = _jsonable(asdict(config))
    config_dict.update({"resolved_image_root": str(raw_root), "resolved_work_root": str(work_root)})
    preprocessing, model_config = preprocessing_config(), config.model_config()
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    _write_json(output / "config.json", config_dict)
    _write_json(output / "provenance.json", {
        "created": _now(), "config": config_dict, "preprocessing": preprocessing,
        "model_config": model_config, "train_manifest_sha256": source_hashes["train"],
        "valid_manifest_sha256": source_hashes["valid"],
        "code_sha256": {name: _sha256(PROJECT_ROOT / name) for name in
            ("dataset.py", "dataset_highres.py", "dataset_tiny.py", "model_tiny.py", "train_tiny.py",
             "train.py", "inference.py", "runtime_compat.py")},
        "training_statistics": {key: value for key, value in statistics.items() if key != "study_view_counts"},
        "loss": {"name": "BCEWithLogitsLoss", "pos_weight": selected_pos_weights,
            "sample_weight": "N_images / (N_studies * views_in_this_study)"},
        "memory_policy": {"amp": precision_name, "channels_last": True,
            "gradient_checkpointing": config.gradient_checkpointing,
            "batch_unit": "image", "batch_size": config.batch_size,
            "accumulation_steps": config.accumulation_steps, "ema_device": "cuda",
            "adamw_fused": amp_dtype == torch.bfloat16, "image_size": 384},
        "reproducibility": "seeded; CUDA benchmark/TF32 allowed; not bitwise deterministic",
        "resume_supported": True, "resume_mode": "weights_warm_restart_only",
        "exact_resume_supported": False, "resume_origin": resume_origin,
    })
    append_run_worklog(work_root, run_id, f"任务：ConvNeXt-Tiny/GAP/Linear原图384新run；"
        f"逐图shuffle完整遍历，batch={config.batch_size}/accum={config.accumulation_steps}，"
        f"Study等权BCE，AMP={precision_name}/GPU EMA；源检查点完成{resume_epoch}轮，"
        "本次尚未反传。")
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    if resume_checkpoint is not None:
        progress.update({key: resume_checkpoint["progress"][key] for key in progress})
    # 生命周期累计更新驱动EMA；本次局部更新驱动新余弦，两者不能混用。
    progress.update({"schedule_updates": 0, "epochs_this_run": 0})
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta, min_epochs=1)
    stop_requested = False
    interrupt_count = 0
    step_hook = None
    status = "preparing"
    best_auc, best_f1 = (-math.inf, -math.inf), (-math.inf, -math.inf)
    best_epoch = None
    if resume_checkpoint is not None:
        source_metrics = resume_checkpoint["validation"]
        best_auc = (source_metrics["macro_auc"], source_metrics["macro_f1"])
        best_f1 = (source_metrics["macro_f1"], source_metrics["macro_auc"])
        best_epoch = resume_epoch
        early_stop.best_auc, early_stop.best_epoch = best_auc[0], resume_epoch
        _write_json(output / "resume.json", resume_origin)
        print(f"=> Warm restart: {config.resume} | epoch {resume_epoch} -> {resume_epoch + 1} | "
              f"baseline AUC={best_auc[0]:.8f} | LR={config.backbone_lr:g}/{config.head_lr:g} | "
              f"warmup=0 | cosine={schedule_epochs} epochs | patience={config.patience} | "
              "重建AdamW/GradScaler；保留累计EMA步数，不是精确状态续训。", flush=True)
    stop_file = config.stop_file.resolve() if config.stop_file else output / "STOP"

    def request_stop(_signum, _frame):
        nonlocal stop_requested, interrupt_count
        interrupt_count += 1
        if interrupt_count > 1:
            raise KeyboardInterrupt("第二次中断立即退出，未完成轮不视为精确可恢复")
        stop_requested = True
        print("=> 已请求轮末停止；本轮完整验证与保存后退出。", flush=True)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        common = {"num_workers": config.num_workers, "prefetch_factor": config.prefetch_factor,
                  "timeout": config.worker_timeout, "seed": config.seed, "raw_root": raw_root}
        print("=> 构建原图清单白名单loader；训练shuffle，无类别重采样，不丢尾批。", flush=True)
        train_loader = make_tiny_train_loader(config.train_manifest, batch_size=config.batch_size, **common)

        def evaluation_loader(split: str, serial: bool = False):
            options = {**common, "num_workers": 0 if serial else config.num_workers}
            manifest = config.valid_manifest if split == "valid" else config.train_manifest
            return make_image_loader(manifest, split=split, batch_size=config.valid_batch_size,
                                     training=False, image_size=384, **options)

        valid_loader = evaluation_loader("valid")
        train_eval_loader = evaluation_loader("train") if config.train_eval_every else None
        print(f"=> 初始化{config.backbone_name}；" +
              ("从检查点恢复，不下载ImageNet预训练权重。" if resume_checkpoint is not None
               else "首次显式启动可能下载ImageNet预训练权重。"), flush=True)
        model = build_tiny_model(pretrained=resume_checkpoint is None,
                                 **model_config).to(device, memory_format=torch.channels_last)
        model.requires_grad_(True)
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,
                         warmup_power=0.75, exclude_buffers=True, foreach=True)
        ema.module.requires_grad_(False)
        if resume_checkpoint is not None:
            restore_warm_restart_weights(model, ema, resume_checkpoint)
            seed_resume_baseline(output, resume_checkpoint, resume_origin)
            # CPU检查点包含两套权重；加载/保留基线后及时释放，降低主机内存占用。
            del resume_checkpoint
            resume_checkpoint = None
        optimizer = build_optimizer(model, config, fused=amp_dtype == torch.bfloat16)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
        pos_weight = (None if selected_pos_weights is None else
                      torch.tensor(selected_pos_weights, device=device, dtype=torch.float32))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none").to(device)

        def record_optimizer_step(_optimizer, _args, _kwargs):
            progress["optimizer_updates"] += 1
            progress["schedule_updates"] += 1
            ema.update(model, step=progress["optimizer_updates"])
            progress["ema_updates"] += 1

        step_hook = optimizer.register_step_post_hook(record_optimizer_step)
        status = "running"
        for epoch in range(resume_epoch + 1, config.epochs + 1):
            if stop_requested or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
            print(f"=> Epoch {epoch}/{config.epochs} | raw384 | batch={config.batch_size} images "
                  f"× accum{config.accumulation_steps} | AMP={precision_name} | EMA=GPU | "
                  "full fine-tune/shuffle coverage", flush=True)
            training = train_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                   amp_dtype, epoch, config, progress, schedule_epochs=schedule_epochs)
            if progress["ema_updates"] == 0:
                raise RuntimeError("尚无成功优化/EMA更新，不能保存初始模型为best")
            if training["unique_studies"] != statistics["num_studies"]:
                raise RuntimeError("逐图完整遍历未覆盖训练清单所有Study")
            train_metrics, val_metrics, fixed_metrics = (_unavailable_metrics() for _ in range(3))
            train_eval = {"complete": False, "loss": None, "reason": "本轮未安排完整训练诊断"}
            thresholds = None
            validation = evaluate_tiny(ema.module, valid_loader, device, amp_dtype,
                manifest=config.valid_manifest, split="valid", epoch=epoch,
                diagnostic_path=output / "data_loading_events.jsonl")
            should_early_stop = False
            if validation["complete"]:
                thresholds = search_class_thresholds(validation["targets"], validation["probabilities"])
                val_metrics = calculate_metrics(validation["targets"], validation["probabilities"], thresholds)
                fixed_metrics = calculate_metrics(validation["targets"], validation["probabilities"], [0.5] * NUM_CLASSES)
                should_early_stop = early_stop.update(epoch, val_metrics["macro_auc"])
                if train_eval_loader is not None and (epoch == 1 or epoch % config.train_eval_every == 0
                                                       or epoch == config.epochs):
                    train_eval = evaluate_tiny(ema.module, train_eval_loader, device, amp_dtype,
                        manifest=config.train_manifest, split="train", epoch=epoch,
                        diagnostic_path=output / "data_loading_events.jsonl")
                    if train_eval["complete"]:
                        train_metrics = calculate_metrics(train_eval["targets"], train_eval["probabilities"], thresholds)
            if validation["fallback_serial"]:
                valid_loader = evaluation_loader("valid", serial=True)
            if train_eval.get("fallback_serial"):
                train_eval_loader = evaluation_loader("train", serial=True)
            for split, manifest in (("train", config.train_manifest), ("valid", config.valid_manifest)):
                if _sha256(manifest) != source_hashes[split]:
                    raise RuntimeError("训练期间清单发生变化，不能继续或交付权重")
            ema_metadata = {"enabled": True, "decay": config.ema_decay, "use_warmup": True,
                "warmup_power": 0.75, "updates": progress["ema_updates"],
                "buffers": "copy_from_raw", "device": "cuda"}
            score_auc = (val_metrics["macro_auc"], val_metrics["macro_f1"]) if validation["complete"] else None
            score_f1 = (val_metrics["macro_f1"], val_metrics["macro_auc"]) if validation["complete"] else None
            is_best = score_auc is not None and score_auc > best_auc
            is_best_f1 = score_f1 is not None and score_f1 > best_f1
            epoch_progress = {**progress, "completed_epochs": epoch,
                "validated_epochs": progress["validated_epochs"] + int(validation["complete"]),
                "epochs_this_run": progress["epochs_this_run"] + 1}
            early_metadata = asdict(early_stop)
            record = {"epoch": epoch, "phase": "warm_restart" if resume_origin else "full_finetune",
                "resume_origin": resume_origin, "training": training,
                "train_loss": training["loss"], "train": train_metrics, "train_eval_loss": train_eval["loss"],
                "val": val_metrics, "val_loss": validation["loss"], "val_at_05": fixed_metrics,
                "val_macro_f1_at_05": fixed_metrics["macro_f1"], "thresholds": thresholds,
                "validation_complete": validation["complete"], "weights_source": "ema", "ema": ema_metadata,
                "metrics_unit": "Study", "inference_precision": precision_name,
                "f1_threshold_source": "same_epoch_valid" if thresholds is not None else None,
                "is_best": is_best, "is_best_f1": is_best_f1, "early_stopping": early_metadata,
                **epoch_progress, "data_loading": {"valid": {key: validation[key] for key in
                    ("complete", "evaluated_images", "expected_images", "skipped_samples", "fallback_serial")}}}
            payload = {"schema_version": 4, "model_name": "convnext_tiny_gap_linear",
                "model_config": model_config, "num_classes": NUM_CLASSES, "class_ids": list(CLASS_IDS),
                "epoch": epoch, "aggregation": "study_mean_probability", "thresholds": thresholds,
                "weights_source": "ema", "validation_weights_source": "ema", "threshold_weights_source": "ema",
                "ema": ema_metadata, "preprocessing": preprocessing, "inference_precision": precision_name,
                "model_state_dict": {name: value.detach().cpu().clone() for name, value in ema.module.state_dict().items()},
                "raw_model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "train_manifest_sha256": source_hashes["train"], "valid_manifest_sha256": source_hashes["valid"],
                "validation_complete": validation["complete"], "inference_ready": validation["complete"],
                "validation": val_metrics if validation["complete"] else None,
                "loss_config": {"name": "BCEWithLogitsLoss", "pos_weight": selected_pos_weights,
                    "study_equal_weighting": "N_images / (N_studies * views_in_this_study)", "rank_weight": 0.0},
                "training_config": config_dict, "progress": epoch_progress,
                "schedule": {"kind": "successful_optimizer_update_warmup_cosine",
                    "learning_rates": training["learning_rates"], "epochs": schedule_epochs,
                    "start_epoch": resume_epoch + 1, "local_updates": progress["schedule_updates"]},
                "early_stopping": early_metadata, "resume_supported": True,
                "resume_mode": "weights_warm_restart_only", "exact_resume_supported": False,
                "resume_origin": resume_origin}
            _save_checkpoint(output / "last.pt", payload)
            if is_best:
                save_bundle(output, payload, record)
                best_auc, best_epoch = score_auc, epoch
            if is_best_f1:
                save_bundle(output / "best_f1", payload, record)
                best_f1 = score_f1
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress.update(epoch_progress)
            _write_json(output / "status.json", {"state": "epoch_complete", **progress,
                "best_epoch": best_epoch, "validation_complete": validation["complete"],
                "early_stopping": early_metadata, "updated": _now()})
            print(f"epoch={epoch} weighted_train_BCE={training['loss']:.6f} "
                  f"val_AUC={_metric_text(val_metrics['macro_auc'])} "
                  f"val_F1_tuned={_metric_text(val_metrics['macro_f1'])} "
                  f"val_F1_05={_metric_text(fixed_metrics['macro_f1'])} "
                  f"val_AP={_metric_text(val_metrics['macro_ap'])} weights=ema best={is_best} "
                  f"complete={validation['complete']} early_bad={early_stop.bad_epochs}/{config.patience}", flush=True)
            print("per_class_AUC=" + json.dumps(val_metrics["per_class_auc"]) +
                  " per_class_AP=" + json.dumps(val_metrics["per_class_ap"]) +
                  " per_class_F1=" + json.dumps(val_metrics["per_class_f1"]), flush=True)
            append_run_worklog(work_root, run_id, f"结果：epoch={epoch}完整train图数={training['image_draws']}，"
                f"本轮更新={training['updates_this_epoch']}，完整valid={validation['complete']}；"
                f"Study AUC/F1@.5={_metric_text(val_metrics['macro_auc'])}/{_metric_text(fixed_metrics['macro_f1'])}，"
                f"best={is_best}；验证调优指标不作为平台测试成绩。")
            with (work_root / "EXPERIMENTS.md").open("a", encoding="utf-8") as stream:
                stream.write(f"| {run_id}-E{epoch:03d} | Tiny GAP+Linear raw384; Study-weighted BCE; "
                    f"shuffle batch{config.batch_size}/accum{config.accumulation_steps}; EMAcuda | "
                    f"loss={training['loss']:.6f}; images={training['image_draws']}; "
                    f"updates={training['updates_this_epoch']} | AUC={_metric_text(val_metrics['macro_auc'])}; "
                    f"F1@.5={_metric_text(fixed_metrics['macro_f1'])} | 未使用 | "
                    f"完整valid={validation['complete']}；F1调阈值={_metric_text(val_metrics['macro_f1'])} | "
                    f"{'保留best' if is_best else '保留last/日志'} | {output}/history.jsonl |\n")
            del payload
            if stop_requested or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
            if should_early_stop:
                status = "early_stopped"
                break
        else:
            status = "completed"
    except BaseException as exc:
        status = "failed_or_interrupted"
        advice = ""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            advice = ("使用新run目录，改--batch-size 16 --accumulation-steps 2；若发生于验证则改"
                      "--valid-batch-size 16。还可显式启用--gradient-checkpointing；不自动重试。")
            print("=> CUDA OOM：" + advice, flush=True)
        _write_json(output / "status.json", {"state": status, **progress, "best_epoch": best_epoch,
            "error": f"{type(exc).__name__}: {exc}", "oom_advice": advice, "updated": _now()})
        append_run_worklog(work_root, run_id, f"失败/中断：{type(exc).__name__}: {exc}；"
            f"完成{progress['completed_epochs']}轮、反传{progress['backward_batches']}批；"
            "保留此前完整轮权重，未完成轮不冒称已验收或精确可续训。")
        raise
    finally:
        if step_hook is not None:
            step_hook.remove()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    result = {"state": status, **progress, "best_epoch": best_epoch, "output": str(output),
              "early_stopping": asdict(early_stop), "updated": _now()}
    _write_json(output / "status.json", result)
    append_run_worklog(work_root, run_id, f"交接：{status}，完成{progress['completed_epochs']}轮、"
        f"完整验证{progress['validated_epochs']}轮；best epoch={best_epoch}；"
        "训练后推理单独由用户显式执行，未读取test作训练选择。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny/GAP+Linear，原图384、完整shuffle、Study等权BCE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
        help="从schema4检查点分别恢复在线模型/EMA，从epoch+1热重启；重建优化器，不是精确状态续训")
    parser.add_argument("--resume-lr-factor", type=float, default=0.1,
        help="热重启峰值LR=检查点基础LR×此系数；默认0.1，同时取消warmup、早停patience最多3")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--image-root", "--data-root", "--raw-root", dest="raw_root", type=Path, default=RAW_ROOT)
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--backbone", dest="backbone_name", choices=(TINY_MODEL_NAME,), default=TINY_MODEL_NAME)
    parser.add_argument("--image-size", type=int, choices=(384,), default=384)
    parser.add_argument("--batch-size", type=int, default=32, help="图片数；shuffle无放回完整遍历，保留尾批")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--valid-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25, help="全局最后一轮编号；源epoch5加--epochs10即续跑6至10")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="默认关闭；手动节省激活显存")
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--pos-weight-power", type=float, default=0.0, help="默认0为普通BCE；大于0才启用train-only权重")
    parser.add_argument("--pos-weight-cap", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--train-eval-every", type=int, default=0, help="0关闭额外train诊断；valid仍每轮完整")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5,
        help="完整valid连续停滞轮数；新训练0禁用，热重启最多3且0改为3")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="只控制早停，微小AUC新高仍保存best")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    run_training(TinyConfig(**vars(args)))


if __name__ == "__main__":
    main()
