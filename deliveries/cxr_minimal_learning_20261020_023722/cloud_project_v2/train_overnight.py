"""Explicitly invoked fold training with complete epoch-boundary continuation.

The original cleaned train is partitioned by patient. Its held-out fold, and
only that fold, selects best.pt. Official valid/test never enter this module.
No training, image decoding, model construction or download occurs at import.
"""
from __future__ import annotations

from runtime_compat import prepare_runtime

prepare_runtime()

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
# 学习导航：当前主线是 ConvNeXt V2-Base、768 原图输入、Study 等权的正样本加权 BCE。
# 参数更新与 EMA 都按“梯度累积窗口”发生，而不是按每个微批发生。

from model_overnight import MODEL_NAME, build_overnight_model, validate_model_config

CLASS_IDS = list(range(1, 11))
TRAINING_SCALES = (768,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OvernightConfig:
    fold_plan: Path
    fold: int
    output: Path
    image_root: Path
    # 微批 2 表示每次前向/反向处理两张图片，未必属于两个不同 Study。
    # 累积 16 次后更新，完整窗口对应 32 张图；不是 32 个独立 Study。默认值可被运行参数覆盖。
    batch_size: int = 2
    accumulation_steps: int = 16
    valid_batch_size: int = 4
    num_workers: int = 8
    prefetch_factor: int = 2
    epochs: int = 20
    max_epochs_this_call: int = 0
    patience: int = 4
    min_delta: float = 1e-4
    seed: int = 43
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    dropout: float = 0.2
    drop_path: float = 0.15
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    amp_dtype: str = "auto"
    resume: Path | None = None
    restart_empty: bool = False
    stop_file: Path | None = None

    def validate(self) -> None:
        for name in ("batch_size", "accumulation_steps", "valid_batch_size", "epochs", "prefetch_factor"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_workers", "max_epochs_this_call", "patience", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.fold < 5 or self.num_workers > 12 or self.seed >= 2**32:
            raise ValueError("fold must be 0..4, num_workers <= 12 and seed < 2**32")
        if self.prefetch_factor not in (1, 2):
            raise ValueError("prefetch_factor must be 1 or 2 to bound host RAM")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        for name in ("backbone_lr", "head_lr", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_delta", "weight_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.min_lr_ratio <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid minimum LR ratio or EMA decay")
        if self.amp_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("Unsupported AMP dtype")
        if type(self.restart_empty) is not bool or self.resume is not None and self.restart_empty:
            raise ValueError("--restart-empty is only for a fold without a completed checkpoint")
        self.model_config()

    def model_config(self) -> dict[str, Any]:
        # 激活检查点在反向时重算部分中间结果省显存，与保存磁盘 best.pt/last.pt 是两件事。
        return validate_model_config({"drop_rate": self.dropout, "drop_path_rate": self.drop_path,
                                      "gradient_checkpointing": True})


def preprocessing_config(*, image_size: int = 768) -> dict[str, Any]:
    """Describe the fixed 768 original-image protocol used in this learning copy."""
    if type(image_size) is not int or image_size != 768:
        raise ValueError("This learning copy uses 768 inputs only")
    scales = TRAINING_SCALES
    return {"image_size": image_size, "validation_size": image_size,
            "training_scales": list(scales), "cache_mode": "RGB",
            "color_mode": "RGB", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
            "resize": "letterboxLANCZOS", "exif_transpose": True,
            "data_source": "raw_readonly", "eval_transform": "EXIF+RGB+LetterboxLANCZOS+Normalize",
            "validation_tta": "identity"}


def compute_loss_config(records: Sequence[dict[str, Any]], *, source_manifest_sha256: str,
                        fold: int | None, fold_plan_sha256: str | None = None) -> dict[str, Any]:
    """Fit ten class weights on TRAIN Studies only; repeated views count once.

    Fold validation labels must never participate. The existing inverse-view
    image weights equalize Studies; they do not add class/positive reweighting.
    """
    if not records or fold is not None and (type(fold) is not int or not 0 <= fold < 5):
        raise ValueError("Loss weights require nonempty train records and a valid fold")
    studies: dict[str, tuple[int, ...]] = {}
    # 同 Study 的多个视图继承同一份标签；类别频率按 Study 去重，而不是逐图重复计数。
    # 这里只允许当前折训练部分，内部验证折、官方 valid、test 不参加 pos_weight 拟合。
    image_ids: set[str] = set()
    for row in records:
        if (row.get("split") != "train" or str(row.get("training_eligible", "")).lower() != "true"
                or row.get("targets") is None or len(row["targets"]) != len(CLASS_IDS)
                or any(value not in (0, 1) for value in row["targets"])):
            raise ValueError("Positive class weights may use eligible original train labels only")
        key = row.get("study_key")
        if (not row.get("subject_id") or not row.get("study_id")
                or key != f"{row['subject_id']}/{row['study_id']}"):
            raise ValueError("Malformed patient/Study identity in loss statistics")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError("Loss statistics require unique nonempty image IDs")
        image_ids.add(image_id)
        labels = tuple(int(value) for value in row["targets"])
        if key in studies and studies[key] != labels:
            raise ValueError("Labels disagree within a Study in loss statistics")
        studies[key] = labels
    labels = np.asarray(list(studies.values()), dtype=np.int64)
    positives = labels.sum(axis=0)
    # 十列分别统计：q_c = 阴性 Study 数 / 阳性 Study 数；没有正例或负例时拒绝补伪权重。
    negatives = len(studies) - positives
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError("Each weighted-BCE class requires positive and negative training Studies")
    canonical = json.dumps([[key, list(studies[key])] for key in sorted(studies)],
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"name": "BCEWithLogitsLoss", "class_ids": CLASS_IDS,
            "pos_weight": (negatives / positives).tolist(),
            "pos_weight_formula": "negative_studies/positive_studies",
            "pos_weight_counting_unit": "Study", "positive_studies": positives.tolist(),
            "negative_studies": negatives.tolist(), "num_studies": len(studies),
            "num_images": len(records), "counted_split": "train",
            "scope": "fold_training_subset" if fold is not None else "full_clean_train",
            "fold_index": fold, "source_manifest_sha256": source_manifest_sha256,
            "fold_plan_sha256": fold_plan_sha256,
            "study_targets_sha256": hashlib.sha256(canonical).hexdigest(),
            "reduction": "mean_classes_then_study_weighted_image_mean",
            "study_equal_weighting": "N_images/(N_studies*retained_study_views)",
            "sampling": "shuffle_without_replacement", "auxiliary_losses": []}


def build_weighted_bce(loss_config: dict[str, Any], *, device: torch.device | str = "cpu") -> nn.Module:
    """Compute stable FP32 BCE; pos_weight multiplies the positive term only."""
    values = torch.tensor(loss_config["pos_weight"], dtype=torch.float32, device=device)
    if values.shape != (10,) or not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("BCE pos_weight must contain ten positive finite values")
    # 一类的损失为 -q*y*log(sigmoid(z)) -(1-y)*log(1-sigmoid(z))，q 只乘阳性项。
    # reduction=none 留下 [B,10]；训练循环先平均十类，再施加每图的 Study 权重。
    # WithLogits 内含稳定的 sigmoid+BCE 运算，调用者不要提前 sigmoid。
    return nn.BCEWithLogitsLoss(pos_weight=values, reduction="none")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # 先写临时文件，再原子替换目标，避免半个 checkpoint 被当成完整恢复点。
    return sha256(path)


def cpu_tree(value: Any) -> Any:
    """Checkpoint tensors are CPU only, including optimizer state and RNG."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng() -> dict[str, Any]:
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"algorithm": state[0],
            "keys": torch.tensor(state[1].astype(np.int64)), "position": state[2],
            "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    item = state["numpy"]
    np.random.set_state((item["algorithm"], item["keys"].numpy().astype(np.uint32),
                         item["position"], item["has_gauss"], item["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def build_optimizer(model: nn.Module, config: OvernightConfig, *, fused: bool) -> torch.optim.Optimizer:
    # 新分类头默认学习率 2e-4，预训练骨干 2e-5；全模型可训练，没有先冻结阶段。
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    if not head_ids:
        raise ValueError("The standard classification head has no parameters")
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = "head" if id(parameter) in head_ids else "backbone"
        # 矩阵/卷积核使用 weight decay；bias 和一维归一化参数不衰减。
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        buckets.setdefault((role, decay), []).append(parameter)
    groups = [{"params": parameters, "role": role,
               "name": f"{role}_{'decay' if decay else 'no_decay'}",
               "lr": config.head_lr if role == "head" else config.backbone_lr,
               "weight_decay": config.weight_decay if decay else 0.0}
              for (role, decay), parameters in buckets.items()]
    # FP16 overflow must skip the entire step and its EMA post hook.
    return torch.optim.AdamW(groups, weight_decay=0.0, fused=fused, foreach=not fused)


def set_learning_rates(optimizer: torch.optim.Optimizer, config: OvernightConfig,
                       updates: int, windows_per_epoch: int) -> dict[str, float]:
    # 学习率按成功优化器更新次数前进，恢复时不重新从零预热，AMP 跳步也不推进计数。
    # 先从基础学习率 0.1 倍预热，再余弦下降到 min_lr_ratio 倍。
    total = config.epochs * windows_per_epoch
    warmup = config.warmup_epochs * windows_per_epoch
    if warmup and updates < warmup:
        factor = 0.1 + 0.9 * updates / warmup
    else:
        fraction = min(1.0, max(0.0, (updates - warmup) / max(1, total - warmup - 1)))
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    for group in optimizer.param_groups:
        group["lr"] = (config.head_lr if group["role"] == "head" else config.backbone_lr) * factor
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if (targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != 10
            or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Invalid Study predictions for complete-fold evaluation")
    # 此处目标已经聚合为 Study；AUC/AP 使用连续概率，不依赖二值阈值。
    aucs, aps = [], []
    for index in range(10):
        if len(np.unique(targets[:, index])) != 2:
            raise ValueError(f"Fold validation class {index + 1} lacks positive/negative Studies")
        aucs.append(float(roc_auc_score(targets[:, index], probabilities[:, index])))
        aps.append(float(average_precision_score(targets[:, index], probabilities[:, index])))
    # 诊断 loss 是聚合后普通无权重 BCE，不等于在线增强训练中的带权 BCE。
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    return {"macro_auc": float(np.mean(aucs)), "per_class_auc": aucs,
            "macro_ap": float(np.mean(aps)), "per_class_ap": aps,
            "loss": float(-(targets * np.log(p) + (1 - targets) * np.log1p(-p)).mean()),
            "num_studies": len(targets), "unit": "Study", "tta": "identity768"}


def train_epoch(model: nn.Module, ema: Any, loader: Any, optimizer: Any, scaler: Any,
                device: torch.device, dtype: torch.dtype, epoch: int, config: OvernightConfig,
                progress: dict[str, int], image_size: int, criterion: nn.Module) -> dict[str, Any]:
    del ema  # EMA is updated only by successful optimizer-step post hooks.
    # train() 启用 dropout/随机深度等训练行为；验证用 eval() 关闭这些随机行为。
    model.train()
    # 每轮无放回覆盖所有图片；用 expected/seen 检查漏图、重复或混入其他图。
    expected = {row["image_id"] for row in loader.dataset.records}
    if len(expected) != len(loader.dataset):
        raise ValueError("Fold train contains duplicate image IDs")
    seen: set[str] = set()
    total_loss = total_weights = 0.0
    image_count = 0
    windows = math.ceil(len(loader) / config.accumulation_steps)
    initial_updates = progress["optimizer_updates"]
    started = time.monotonic()
    # 在累积窗口起点清梯度；若每个微批都清零，先前微批的梯度就会丢失。
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    rates = set_learning_rates(optimizer, config, progress["optimizer_updates"], windows)
    for batch_index, batch in enumerate(loader):
        count = len(batch["image_ids"])
        batch_ids = set(batch["image_ids"])
        if (len(batch_ids) != count or seen.intersection(batch_ids) or not batch_ids <= expected
                or batch["images"].shape != (count, 3, image_size, image_size)
                or batch["targets"] is None or batch["targets"].shape != (count, 10)):
            raise ValueError("Train batch identity, targets or image shape failed validation")
        weights = batch["loss_weights"]
        if weights.shape != (count,) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Train requires positive finite globally normalized Study weights")
        seen.update(batch_ids)
        # 逻辑形状仍是 [B,3,H,W]；channels_last 只改变有利于 GPU 的内存排布。
        images = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        weights_gpu = weights.to(device, dtype=torch.float32, non_blocking=True)
        # AMP 让适合的骨干运算使用 BF16/FP16；输出 [B,10] 仍是未 sigmoid 的 logit。
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(images)
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise FloatingPointError("Train logits are nonfinite or have incorrect shape")
        # Class imbalance is handled exactly once by BCE's positive term.
        # Keep this FP32 even though the large backbone runs under AMP.
        # 先对每图的十类 BCE 求均值，再乘逆视图 Study 权重。损失明确以 FP32 计算。
        # 训练没有先聚合 Study 概率再算 BCE；Study 概率平均是在验证/推理阶段。
        per_image = criterion(logits.float(), targets).mean(1)
        weighted_sum = (per_image * weights_gpu).sum()
        if not torch.isfinite(weighted_sum):
            raise FloatingPointError("Train loss is NaN/Inf")
        # 完整窗口分母是 2*16=32 张图，尾窗按实际剩余图片数归一化。
        # 不能始终除以 16，也不能用当前微批的权重和重新归一化：Study 权重是全局定义。
        first_batch = batch_index // config.accumulation_steps * config.accumulation_steps
        window_images = min(config.batch_size * config.accumulation_steps,
                            len(loader.dataset) - first_batch * config.batch_size)
        # backward 立即计算该微批梯度，加到已有 .grad；它自身不修改参数。
        scaler.scale(weighted_sum / window_images).backward()
        progress["backward_batches"] += 1
        total_loss += float(weighted_sum.detach())
        total_weights += float(weights.sum())
        image_count += count
        del images, targets, weights_gpu, logits, per_image, weighted_sum
        # 满 16 个微批或到 epoch 尾部才更新；不足整窗的尾部也会被保留。
        if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(loader):
            rates = set_learning_rates(optimizer, config, progress["optimizer_updates"], windows)
            # FP16 时先还原真实梯度尺度，再裁剪范数；BF16 的 scaler 是关闭的同形接口。
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                     error_if_nonfinite=not scaler.is_enabled())
            before = progress["optimizer_updates"]
            # FP16 溢出时可跳过 optimizer.step，因此真实更新的 EMA 后置 hook 也不会触发。
            scaler.step(optimizer)
            scaler.update()
            if progress["optimizer_updates"] == before:
                progress["amp_skipped_steps"] += 1
            optimizer.zero_grad(set_to_none=True)
        if batch_index == 0 or (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            print(f"fold={config.fold} epoch={epoch} size={image_size} batch={batch_index + 1}/{len(loader)} "
                  f"loss={total_loss / image_count:.6f} images={image_count}/{len(loader.dataset)} "
                  f"updates={progress['optimizer_updates']} peak={torch.cuda.max_memory_allocated(device)/2**30:.2f}GiB",
                  flush=True)
    if seen != expected or image_count != len(loader.dataset):
        raise RuntimeError("Incomplete fold train image coverage")
    if not math.isclose(total_weights, image_count, rel_tol=2e-6, abs_tol=1e-3):
        raise ValueError("Study weights are not globally normalized to N_images")
    return {"loss": total_loss / image_count, "images": image_count, "full_manifest_coverage": True,
            "image_size": image_size, "seconds": time.monotonic() - started,
            "updates_this_epoch": progress["optimizer_updates"] - initial_updates,
            "learning_rates": rates, "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Any, records: list[dict[str, Any]],
             device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from dataset_overnight import aggregate_study_predictions

    # inference_mode 不建立反传图；调用方传入的是 EMA.module。
    # 内部折验证固定 768，不做随机增强或水平翻转 TTA。
    model.eval()
    image_ids: list[str] = []
    probabilities = []
    for batch in loader:
        images = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(images)
        if logits.shape != (len(batch["image_ids"]), 10) or not torch.isfinite(logits).all():
            raise FloatingPointError("Validation logits are nonfinite or have incorrect shape")
        image_ids.extend(batch["image_ids"])
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    # 先对每图 sigmoid，再平均同 Study 原始视图的概率；不能换成先平均 logit。
    result = aggregate_study_predictions(records, image_ids, np.concatenate(probabilities))
    result["metrics"] = calculate_metrics(result["targets"], result["probabilities"])
    return result


def save_oof(path: Path, validation: dict[str, Any], *, fold: int, epoch: int,
             checkpoint_sha256: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, study_keys=np.asarray(validation["study_keys"], dtype=str),
                            subject_ids=np.asarray(validation["subject_ids"], dtype=str),
                            study_ids=np.asarray(validation["study_ids"], dtype=str),
                            targets=np.asarray(validation["targets"], dtype=np.float32),
                            probabilities=np.asarray(validation["probabilities"], dtype=np.float32),
                            fold=np.asarray(fold), epoch=np.asarray(epoch),
                            checkpoint_sha256=np.asarray(checkpoint_sha256))
    os.replace(temporary, path)


def _resume_check(checkpoint: dict[str, Any], config: OvernightConfig,
                  plan_hash: str, source_hash: str, precision: str,
                  expected_loss_config: dict[str, Any] | None = None) -> None:
    if (checkpoint.get("schema_version") != 5 or checkpoint.get("model_name") != MODEL_NAME
            or checkpoint.get("fold_index") != config.fold or checkpoint.get("fold_plan_sha256") != plan_hash
            or checkpoint.get("source_manifest_sha256") != source_hash
            or checkpoint.get("class_ids") != CLASS_IDS or checkpoint.get("validation_complete") is not True
            or checkpoint.get("model_config") != config.model_config()
            or checkpoint.get("preprocessing") != preprocessing_config()
            or checkpoint.get("inference_precision") != precision):
        raise ValueError("Resume checkpoint architecture/fold/manifest/preprocessing/precision mismatch")
    objective = checkpoint.get("loss_config")
    if (not isinstance(objective, dict) or objective.get("name") != "BCEWithLogitsLoss"
            or objective.get("pos_weight_formula") != "negative_studies/positive_studies"
            or objective.get("pos_weight_counting_unit") != "Study"
            or objective.get("scope") != "fold_training_subset"
            or objective.get("fold_index") != config.fold
            or objective.get("source_manifest_sha256") != source_hash
            or objective.get("fold_plan_sha256") != plan_hash
            or objective.get("auxiliary_losses") != []
            or expected_loss_config is not None and objective != expected_loss_config):
        raise ValueError("Resume weighted-BCE objective/statistics mismatch; use a new run for a new objective")
    saved = checkpoint["training_config"]
    # Batch may shrink after an OOM, preserving the effective batch and update schedule.
    invariant = ("epochs", "warmup_epochs", "min_lr_ratio", "backbone_lr", "head_lr", "weight_decay",
                 "seed", "ema_decay", "patience", "min_delta", "grad_clip")
    if any(saved.get(key) != getattr(config, key) for key in invariant):
        raise ValueError("Resume cannot change LR horizon, objective, seed or early-stop policy")
    if saved["batch_size"] * saved["accumulation_steps"] != config.batch_size * config.accumulation_steps:
        raise ValueError("Resume must retain the same effective batch size")
    required = ("optimizer_state_dict", "scaler_state_dict", "rng_state", "early_stopping", "progress")
    if any(key not in checkpoint for key in required):
        raise ValueError("Resume requires full optimizer/scaler/RNG/early-stopping state")
    if checkpoint["progress"].get("completed_epochs") != checkpoint.get("epoch"):
        raise ValueError("Resume epoch progress is inconsistent")


def _best_record(checkpoint: dict[str, Any], *, actual_best: bool) -> tuple[float, int]:
    """Validate best metadata independently from the model's tensor contents."""
    epoch, best_epoch = checkpoint.get("epoch"), checkpoint.get("best_epoch")
    auc = checkpoint.get("best_auc")
    validation = checkpoint.get("validation", {})
    current_auc = validation.get("macro_auc") if isinstance(validation, dict) else None
    if (type(epoch) is not int or epoch < 1 or type(best_epoch) is not int
            or not 1 <= best_epoch <= epoch or type(auc) not in (int, float)
            or not math.isfinite(auc) or not 0 <= auc <= 1
            or type(current_auc) not in (int, float) or not math.isfinite(current_auc)
            or not 0 <= current_auc <= auc):
        raise ValueError("Checkpoint contains invalid current/best validation metadata")
    if actual_best and (best_epoch != epoch or auc != current_auc):
        raise ValueError("best.pt must hold the weights and metrics from its own best epoch")
    return float(auc), best_epoch


def _checkpoint_validation(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Recover OOF values from CPU metadata; never constructs or runs a model."""
    prediction = checkpoint.get("validation_predictions")
    if not isinstance(prediction, dict):
        return None
    result = {key: prediction.get(key) for key in ("study_keys", "subject_ids", "study_ids")}
    count = len(result["study_keys"]) if isinstance(result["study_keys"], list) else 0
    if (not count or any(not isinstance(result[key], list) or len(result[key]) != count
                         or any(not isinstance(value, str) or not value for value in result[key])
                         for key in result)
            or len(set(result["study_keys"])) != count
            or any(key != f"{subject}/{study}" for key, subject, study in zip(
                result["study_keys"], result["subject_ids"], result["study_ids"]))):
        raise ValueError("Saved validation prediction identities are invalid")
    for key in ("targets", "probabilities"):
        value = prediction.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError("Saved validation predictions must be CPU tensors")
        values = value.detach().numpy()
        if values.shape != (count, 10) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Saved validation prediction values are invalid")
        if key == "targets" and not np.isin(values, [0, 1]).all():
            raise ValueError("Saved OOF targets must be the original binary labels")
        result[key] = values
    return result


def _repair_best_bindings(output: Path, checkpoint: dict[str, Any], digest: str) -> None:
    """Repair a partially committed best bundle using only its saved CPU values."""
    auc, epoch = _best_record(checkpoint, actual_best=True)
    prediction = _checkpoint_validation(checkpoint)
    oof_path = output / "best_oof.npz"
    oof_complete = False
    if oof_path.exists():
        try:
            with np.load(oof_path, allow_pickle=False) as values:
                oof_complete = (str(values["checkpoint_sha256"].item()) == digest
                                and int(values["epoch"].item()) == epoch
                                and int(values["fold"].item()) == checkpoint["fold_index"])
                if prediction is not None:
                    oof_complete = oof_complete and all(np.array_equal(values[key], prediction[key]) for key in
                        ("study_keys", "subject_ids", "study_ids", "targets", "probabilities"))
        except (ValueError, KeyError, OSError, EOFError):
            oof_complete = False
    if not oof_complete and prediction is not None:
        save_oof(oof_path, prediction, fold=checkpoint["fold_index"], epoch=epoch,
                 checkpoint_sha256=digest)
        oof_complete = True
    elif not oof_complete and oof_path.exists():
        # Legacy schema-5 files may lack saved validation values. Preserve a
        # stale artifact for inspection without presenting it as current OOF.
        stale = output / f"best_oof.stale.{time.time_ns()}.npz"
        os.replace(oof_path, stale)
    metrics_path = output / "best_metrics.json"
    previous = {}
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
    expected = {"epoch": epoch, "fold": checkpoint["fold_index"], "checkpoint_sha256": digest,
                "best_auc": auc, "best_epoch": epoch, "validation": checkpoint["validation"],
                "oof_complete": oof_complete}
    if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in expected.items()):
        # Reconstruct from the checkpoint, rather than retaining unrelated
        # training statistics from an older best_metrics.json.
        write_json(metrics_path, {**expected, "is_best": True, "binding_recovered": True,
                   "progress": checkpoint["progress"], "early_stopping": checkpoint["early_stopping"]})
    write_json(output / "thresholds.json", {"schema_version": 5, "class_ids": CLASS_IDS,
               "thresholds": checkpoint["thresholds"], "checkpoint_sha256": digest,
               "selection": "fixed_0.5; continuous probabilities are submitted"})


def reconcile_resume_best(checkpoint: dict[str, Any], config: OvernightConfig,
                          plan_hash: str, source_hash: str, precision: str,
                          expected_loss_config: dict[str, Any] | None = None,
                          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile independently atomic best/last commits before model creation.

    best.pt contains the full continuation state. If its complete epoch is newer
    than last.pt, continue from best.pt and repair missing OOF/JSON companions.
    An older but better disk best remains protected while last's online model
    and optimizer continue. Missing historic best weights are never fabricated.
    """
    # best/last 各自原子提交；中断可能造成 best 更新而 last 还旧，需要先协调完整状态。
    # 先核对代码约定、数据和损失身份；缺失的历史最佳权重不能凭空伪造。
    _resume_check(checkpoint, config, plan_hash, source_hash, precision, expected_loss_config)
    recorded_auc, recorded_epoch = _best_record(checkpoint, actual_best=False)
    output = config.output.resolve()
    best_path = output / "best.pt"
    best = None
    best_hash = None
    if best_path.exists():
        best_hash = sha256(best_path)
        best = checkpoint if config.resume is not None and config.resume.resolve() == best_path else torch.load(
            best_path, map_location="cpu", weights_only=True)
        if sha256(best_path) != best_hash:
            raise RuntimeError("best.pt changed while reconciling continuation state")
        _resume_check(best, config, plan_hash, source_hash, precision, expected_loss_config)
        disk_auc, disk_epoch = _best_record(best, actual_best=True)
    else:
        disk_auc, disk_epoch = -math.inf, 0
    if disk_auc < recorded_auc:
        if checkpoint["epoch"] != recorded_epoch or checkpoint["validation"]["macro_auc"] != recorded_auc:
            raise RuntimeError("Historic best weights are missing or older than last.pt's recorded best; cannot reconstruct them")
        best = checkpoint
        best_hash = save_checkpoint(best_path, best)
        disk_auc, disk_epoch = recorded_auc, recorded_epoch
    assert best is not None and best_hash is not None
    _repair_best_bindings(output, best, best_hash)
    prior_epoch = checkpoint["epoch"]
    promote = best["epoch"] > checkpoint["epoch"]
    if promote:
        checkpoint = best
    else:
        checkpoint["best_auc"], checkpoint["best_epoch"] = disk_auc, disk_epoch
        if disk_auc > recorded_auc:
            early = dict(checkpoint["early_stopping"])
            reference = early.get("reference_auc")
            early["reference_auc"] = disk_auc if reference is None else max(reference, disk_auc)
            checkpoint["early_stopping"] = early
    actual_path = best_path if promote else config.resume.resolve()
    return checkpoint, {"promoted_newer_best_state": promote, "requested_resume_epoch": prior_epoch,
                        "actual_resume_epoch": checkpoint["epoch"], "best_auc": disk_auc,
                        "best_epoch": disk_epoch, "best_checkpoint_sha256": best_hash,
                        "actual_checkpoint": str(actual_path),
                        "actual_checkpoint_sha256": best_hash if promote else sha256(actual_path)}


def run_training(config: OvernightConfig) -> dict[str, Any]:
    """Run one fold slice; full continuation does not restart warmup/cosine."""
    config.validate()
    from cv_splits import load_fold_records
    from dataset_overnight import make_overnight_loader

    plan_path = config.fold_plan.resolve(strict=True)
    plan_hash = sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_hash = plan["source_manifest_sha256"]
    # 内部 train/valid 都来自原始 train；官方 valid 不参与每折训练或 best 选择。
    train_records = load_fold_records(plan_path, config.fold, "train")
    valid_records = load_fold_records(plan_path, config.fold, "valid")
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {row.get(key) for row in train_records} - {None, ""}
        right = {row.get(key) for row in valid_records} - {None, ""}
        if left & right:
            raise ValueError(f"Fold patient/content leakage: overlapping {key}")
    if any(row["split"] != "train" for row in train_records + valid_records):
        raise ValueError("CV training/selection may use only original train records")
    # 先分折，再只用本折训练部分估计 q_c，不能直接用全部 train 或验证集统计代替。
    loss_config = compute_loss_config(train_records, source_manifest_sha256=source_hash,
                                      fold=config.fold, fold_plan_sha256=plan_hash)
    device = torch.device("cuda", 0)
    if not torch.cuda.is_available():
        raise RuntimeError("Overnight training requires CUDA and AMP")
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if config.amp_dtype == "bfloat16" and not bf16:
        raise ValueError("This GPU does not support bfloat16")
    # auto 优先 BF16（4090 支持），否则 FP16；FP16 配合后面的 GradScaler。
    precision = "bfloat16" if config.amp_dtype == "bfloat16" or config.amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    output = config.output.resolve()
    image_root = config.image_root.resolve(strict=True)
    if output == image_root or output.is_relative_to(image_root):
        raise ValueError("Output must not be inside the read-only raw data root")
    if config.resume is None:
        if output.exists() and any(output.iterdir()):
            if not config.restart_empty:
                raise FileExistsError("A new fold must use an empty output directory; pass --resume for last.pt")
            if (list(output.glob("*.pt")) or list(output.glob("*.pt.tmp"))
                    or (output / "best_oof.npz").exists()
                    or (output / "history.jsonl").exists() and (output / "history.jsonl").stat().st_size):
                raise ValueError("--restart-empty refuses a fold with saved checkpoints/history")
            prior_status = json.loads((output / "status.json").read_text(encoding="utf-8")) if (
                output / "status.json").exists() else {}
            if prior_status.get("completed_epochs", 0) != 0:
                raise ValueError("--restart-empty refuses any previously completed epoch")
            with (output / "restart_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"prior_status": prior_status,
                                          "new_config": _jsonable(asdict(config))}, allow_nan=False) + "\n")
        output.mkdir(parents=True, exist_ok=True)
    elif not output.is_dir() or config.resume.resolve().parent != output:
        raise ValueError("Resume must use a checkpoint in its original output directory")
    seed = (config.seed + config.fold * 10007) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Spawn workers each receive one CPU thread; keep the main process modest.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    early = {"reference_auc": None, "bad_epochs": 0, "patience": config.patience, "min_delta": config.min_delta}
    best_auc, best_epoch = None, None
    checkpoint = None
    resume_origin = None
    if config.resume is not None:
        requested_resume_hash = sha256(config.resume)
        checkpoint = torch.load(config.resume.resolve(strict=True), map_location="cpu", weights_only=True)
        if sha256(config.resume) != requested_resume_hash:
            raise RuntimeError("Requested resume checkpoint changed while loading")
        checkpoint, reconciliation = reconcile_resume_best(checkpoint, config, plan_hash, source_hash,
                                                           precision, expected_loss_config=loss_config)
        progress.update(checkpoint["progress"])
        early = dict(checkpoint["early_stopping"])
        best_auc, best_epoch = checkpoint["best_auc"], checkpoint["best_epoch"]
        resume_origin = {"path": reconciliation["actual_checkpoint"],
                         "checkpoint_sha256": reconciliation["actual_checkpoint_sha256"],
                         "requested_resume_path": str(config.resume.resolve()),
                         "requested_resume_sha256": requested_resume_hash,
                         "epoch": checkpoint["epoch"],
                         "source_training_config": checkpoint["training_config"],
                         "best_reconciliation": reconciliation,
                         "current_batch_size": config.batch_size,
                         "current_accumulation_steps": config.accumulation_steps}
        with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(resume_origin, allow_nan=False) + "\n")
    write_json(output / "config.json", {**_jsonable(asdict(config)), "model_config": config.model_config(),
                                        "preprocessing": preprocessing_config(), "inference_precision": precision,
                                        "loss_config": loss_config})
    write_json(output / "loss_statistics.json", loss_config)
    print(f"=> fold={config.fold} {MODEL_NAME} train_images={len(train_records)} valid_images={len(valid_records)} "
          f"AMP={precision} batch={config.batch_size} accum={config.accumulation_steps} workers={config.num_workers} "
          f"resume_epoch={progress['completed_epochs']} horizon={config.epochs}", flush=True)
    print(f"=> pure weighted BCE; train Studies={loss_config['num_studies']} "
          f"pos={loss_config['positive_studies']} "
          f"pos_weight={[round(value, 6) for value in loss_config['pos_weight']]}", flush=True)
    if progress["completed_epochs"] >= config.epochs or config.patience and early["bad_epochs"] >= config.patience:
        state = "completed" if progress["completed_epochs"] >= config.epochs else "early_stopped"
        result = {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch,
                  "fold": config.fold, "output": str(output)}
        write_json(output / "status.json", result)
        return result
    hook = None
    previous_handlers: dict[Any, Any] = {}
    stop_requested = False
    signal_count = 0
    state = "paused"

    def request_stop(_signal, _frame):
        nonlocal stop_requested, signal_count
        signal_count += 1
        if signal_count > 1:
            raise KeyboardInterrupt("Second signal: stop immediately; last complete epoch is retained")
        stop_requested = True
        print("=> Stop requested: finish this epoch, validate, save and pause.", flush=True)

    try:
        model = build_overnight_model(pretrained=checkpoint is None, **config.model_config()).to(
            device, memory_format=torch.channels_last)
        model.requires_grad_(True)
        from timm.utils import ModelEmaV3

        # EMA 保存第二份平滑参数：ema = d_t*旧ema + (1-d_t)*在线模型。
        # 目标 decay=0.999，但 warmup=True 使早期 d_t 变化，不是从第一步就固定 0.999。
        # 非参数 buffers 按 exclude_buffers=True 直接同步；EMA 参数本身不参加反向传播。
        ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,
                         warmup_power=0.75, exclude_buffers=True, foreach=True)
        ema.module.requires_grad_(False)
        optimizer = build_optimizer(model, config, fused=dtype == torch.bfloat16)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        criterion = build_weighted_bce(loss_config, device=device)
        # 完整接续同时恢复在线模型、EMA、AdamW 历史、scaler 与随机状态，不只是推理权重。
        if checkpoint is not None:
            model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
            ema.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            restore_rng(checkpoint["rng_state"])
            del checkpoint
            checkpoint = None

        # 这个后置 hook 只在真实 optimizer.step 后执行：一次累积窗口对应一次 EMA 更新。
        # AMP 跳过更新时两种计数都不增加，保持优化器进度和 EMA 进度一致。
        def after_step(_optimizer, _args, _kwargs):
            progress["optimizer_updates"] += 1
            ema.update(model, step=progress["optimizer_updates"])
            progress["ema_updates"] += 1

        hook = optimizer.register_step_post_hook(after_step)
        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        stop_file = config.stop_file.resolve() if config.stop_file else output / "STOP"
        start_epoch = progress["completed_epochs"] + 1
        end_epoch = config.epochs if not config.max_epochs_this_call else min(
            config.epochs, progress["completed_epochs"] + config.max_epochs_this_call)
        for epoch in range(start_epoch, end_epoch + 1):
            if stop_requested or stop_file.exists():
                break
            # 当前 TRAINING_SCALES 只有 768，每轮都从原图得到该尺寸。
            # 每轮重建非持久 worker，让当前尺寸与随机种子正确进入 worker 副本。
            image_size = TRAINING_SCALES[(epoch - 1) % len(TRAINING_SCALES)]
            loader_options = {"image_root": image_root, "num_workers": config.num_workers,
                              "prefetch_factor": config.prefetch_factor,
                              "seed": (seed + epoch * 1009) % 2**32}
            train_loader = make_overnight_loader(train_records, training=True, image_size=image_size,
                                                  batch_size=config.batch_size, **loader_options)
            training = train_epoch(model, ema, train_loader, optimizer, scaler, device,
                                   dtype, epoch, config, progress, image_size, criterion)
            del train_loader
            if progress["optimizer_updates"] != progress["ema_updates"] or training["updates_this_epoch"] < 1:
                raise RuntimeError("No successful updates, or EMA/optimizer progress diverged")
            valid_loader = make_overnight_loader(valid_records, training=False, image_size=768,
                                                  batch_size=config.valid_batch_size, **loader_options)
            # 内部折评价、best 与之后推理都用 EMA；在线模型保留给 AdamW 继续更新。
            validation = evaluate(ema.module, valid_loader, valid_records, device, dtype)
            del valid_loader
            if sha256(plan_path) != plan_hash:
                raise RuntimeError("Fold plan changed during training")
            # Recheck bound source bytes, without decoding images or changing labels.
            source_path = Path(plan["source_train_manifest"]["path"])
            if sha256(source_path) != source_hash:
                raise RuntimeError("Original train manifest changed during training")
            auc = validation["metrics"]["macro_auc"]
            # best 只看 Study Macro-AUC 严格提升，相等不覆盖；这里没有搜索 F1 阈值。
            is_best = best_auc is None or auc > best_auc
            if is_best:
                best_auc, best_epoch = auc, epoch
            # 早停另要求超过参考 AUC+min_delta；因此刷新 best 与清零坏轮数并非同一条件。
            if early["reference_auc"] is None or auc > early["reference_auc"] + config.min_delta:
                early["reference_auc"], early["bad_epochs"] = auc, 0
            else:
                early["bad_epochs"] += 1
            progress.update(completed_epochs=epoch, validated_epochs=progress["validated_epochs"] + 1)
            record = {"epoch": epoch, "fold": config.fold, "training": training,
                      "validation": validation["metrics"], "is_best": is_best,
                      "best_auc": best_auc, "best_epoch": best_epoch,
                      "early_stopping": dict(early), "progress": dict(progress)}
            payload = {"schema_version": 5, "model_name": MODEL_NAME,
                       "model_config": config.model_config(), "num_classes": 10,
                       "class_ids": CLASS_IDS, "image_size": 768, "epoch": epoch,
                       "fold_index": config.fold, "fold_plan_sha256": plan_hash,
                       "source_manifest_sha256": source_hash,
                       "train_manifest_sha256": source_hash, "aggregation": "study_mean_probability",
                       "weights_source": "ema", "validation_weights_source": "ema",
                       "validation_complete": True, "inference_ready": True,
                       "validation": validation["metrics"], "preprocessing": preprocessing_config(),
                       "validation_predictions": {
                           "study_keys": list(validation["study_keys"]),
                           "subject_ids": list(validation["subject_ids"]),
                           "study_ids": list(validation["study_ids"]),
                           "targets": torch.tensor(validation["targets"], dtype=torch.float32),
                           "probabilities": torch.tensor(validation["probabilities"], dtype=torch.float32)},
                       # 固定 0.5 只是内部折设置；最终官方 valid 阈值搜索在 inference_overnight.py。
                       # AUC 和十类概率提交不靠阈值筛标签，也不按阈值删除提交行。
                       "inference_precision": precision, "thresholds": [0.5] * 10,
                       "threshold_selection": "fixed_0.5_not_used_for_auc_or_probability_submission",
                       "model_state_dict": cpu_tree(ema.module.state_dict()),
                       "raw_model_state_dict": cpu_tree(model.state_dict()),
                       "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                       "scaler_state_dict": cpu_tree(scaler.state_dict()), "rng_state": capture_rng(),
                       "ema": {"enabled": True, "decay": config.ema_decay, "use_warmup": True,
                               "warmup_power": 0.75, "updates": progress["ema_updates"], "device": "cuda"},
                       "progress": dict(progress), "early_stopping": dict(early),
                       "best_auc": best_auc, "best_epoch": best_epoch,
                       "training_config": _jsonable(asdict(config)),
                       "resume_origin": resume_origin,
                       "schedule": {"kind": "successful_optimizer_update_warmup_cosine",
                                    "horizon_epochs": config.epochs,
                                    "updates": progress["optimizer_updates"]},
                       "loss_config": loss_config,
                       "resume_supported": True, "resume_mode": "complete_epoch_state",
                       "exact_resume_supported": False,
                       "reproducibility": "RNG restored at epoch boundaries; TF32/CUDA kernels not bitwise deterministic"}
            # Commit a new best and its OOF first: a committed last.pt never
            # advertises a best epoch whose best.pt has not yet been written.
            # Each checkpoint is replaced atomically, preserving prior weights.
            # 先写 best/OOF 再写 last，避免 last 引用一份尚未落盘的最佳模型。
            # OOF 的 Study 概率绑定 best 文件哈希，避免把旧预测错配到新权重。
            if is_best:
                best_hash = save_checkpoint(output / "best.pt", payload)
                save_oof(output / "best_oof.npz", validation, fold=config.fold, epoch=epoch,
                         checkpoint_sha256=best_hash)
                write_json(output / "best_metrics.json", {**record, "checkpoint_sha256": best_hash})
                write_json(output / "thresholds.json", {"schema_version": 5, "class_ids": CLASS_IDS,
                           "thresholds": [0.5] * 10, "checkpoint_sha256": best_hash,
                           "selection": "fixed_0.5; continuous probabilities are submitted"})
            last_hash = save_checkpoint(output / "last.pt", payload)
            del payload
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            state = "completed" if epoch >= config.epochs else "paused"
            if config.patience and early["bad_epochs"] >= config.patience:
                state = "early_stopped"
            result = {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch,
                      "fold": config.fold, "output": str(output), "last_checkpoint_sha256": last_hash,
                      "early_stopping": dict(early), "updated_utc": datetime.now(timezone.utc).isoformat()}
            write_json(output / "status.json", result)
            print(f"=> fold={config.fold} epoch={epoch} val_AUC={auc:.8f} "
                  f"val_BCE={validation['metrics']['loss']:.6f} best={best_auc:.8f}@{best_epoch} "
                  f"bad_epochs={early['bad_epochs']}/{config.patience} state={state}", flush=True)
            if state in {"completed", "early_stopped"} or stop_requested or stop_file.exists():
                break
    except BaseException as exc:
        result = {"state": "failed", **progress, "best_auc": best_auc, "best_epoch": best_epoch,
                  "fold": config.fold, "output": str(output), "error": f"{type(exc).__name__}: {exc}",
                  "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                  "incomplete_epoch": True, "resume_from": str(output / "last.pt")}
        write_json(output / "status.json", result)
        raise
    finally:
        if hook is not None:
            hook.remove()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    result = {"state": state, **progress, "best_auc": best_auc, "best_epoch": best_epoch,
              "fold": config.fold, "output": str(output), "early_stopping": dict(early)}
    write_json(output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-isolated ConvNeXt V2-Base fold training")
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--restart-empty", action="store_true",
                        help="Retry initialization only when no checkpoint/completed epoch exists")
    parser.add_argument("--stop-file", type=Path)
    for name, default in (("batch-size", 2), ("accumulation-steps", 16), ("valid-batch-size", 4),
                          ("num-workers", 8), ("prefetch-factor", 2), ("epochs", 20),
                          ("max-epochs-this-call", 0), ("patience", 4), ("seed", 43), ("warmup-epochs", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("backbone-lr", 2e-5), ("head-lr", 2e-4), ("min-lr-ratio", 0.05),
                          ("weight-decay", 0.05), ("dropout", 0.2), ("drop-path", 0.15),
                          ("ema-decay", 0.999), ("grad-clip", 1.0), ("min-delta", 1e-4)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    run_training(OvernightConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
