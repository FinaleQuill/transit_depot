"""384原图、ConvNeXt V2类别注意力、Study监督的独立训练入口。

仅显式执行main才运行；导入、编辑和py_compile不读取图片、不创建模型。
旧train.py/model.py/dataset.py保留原样。固定384、AMP与非重入激活检查点；
Study批大小、累积次数、GPU视角分块和工作/原图根均可独立配置。
云端批量、分块和worker参数由run_cloud.sh显式设置，原图始终只读。
"""
from __future__ import annotations

if __name__ == "__main__":
    print("=> 高分辨率384训练入口已进入，正在初始化依赖...", flush=True)

from runtime_compat import prepare_runtime

prepare_runtime()

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataset import (
    CLASS_IDS, DEFAULT_TRAIN_MANIFEST, DEFAULT_VALID_MANIFEST,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, PROJECT_ROOT,
    StudyPredictionAggregator, read_manifest,
)
from dataset_highres import RAW_ROOT, make_image_loader, make_study_loader, resolve_image_root
from model_highres import (
    HIGHRES_MODEL_NAME, SUPPORTED_BACKBONES, StudyBCERankLoss,
    build_highres_model, validate_model_config,
)
# 复用已经审阅的指标、文件原子保存和完整评估取批保护，不调用旧训练入口。
from train import (
    _aggregate_clean_train, _jsonable, _metrics_imports,
    _metric_text, _now, _safe_eval_batches, _save_checkpoint, _sha256,
    _unavailable_metrics, _write_json, calculate_metrics, search_class_thresholds,
)


@dataclass(frozen=True)
class HighResConfig:
    output: Path
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST
    valid_manifest: Path = DEFAULT_VALID_MANIFEST
    raw_root: Path = RAW_ROOT
    work_root: Path = PROJECT_ROOT
    backbone_name: str = HIGHRES_MODEL_NAME
    image_size: int = 384
    batch_size: int = 2  # Study数，不是图片数；云端run_cloud.sh显式设置。
    accumulation_steps: int = 16
    view_chunk_size: int = 1  # GPU同时处理的图片数；云端显式选择8。
    view_pad_multiple: int = 0  # 可选CPU尾块补齐；0保留原行为，不增加有效视角或标签。
    epochs: int = 30
    freeze_epochs: int = 1
    warmup_epochs: int = 2
    backbone_lr: float = 3e-5
    head_lr: float = 3e-4
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.01
    attention_dim: int = 256
    attention_heads: int = 8
    attention_layers: int = 2
    dropout: float = 0.2
    drop_path: float = 0.1
    rank_weight: float = 0.2
    rank_warmup_epochs: int = 3
    rank_ramp_epochs: int = 3
    balanced_probability: float = 0.5
    ema_decay: float = 0.999
    num_workers: int = 0
    prefetch_factor: int = 1
    worker_timeout: float = 120.0
    train_eval_every: int = 5
    grad_clip: float = 1.0
    seed: int = 43
    device: str = "cuda"
    stop_file: Path | None = None

    def model_config(self) -> dict[str, Any]:
        return validate_model_config({
            "backbone_name": self.backbone_name, "attention_dim": self.attention_dim,
            "attention_heads": self.attention_heads, "attention_layers": self.attention_layers,
            "drop_rate": self.dropout, "drop_path_rate": self.drop_path,
            "gradient_checkpointing": True,
        })

    def validate(self) -> None:
        if self.image_size != 384:
            raise ValueError("本版本固定384，不启用512")
        if type(self.batch_size) is not int or self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("Study batch_size必须为至少2的偶数，以构造正负对")
        if type(self.view_chunk_size) is not int or self.view_chunk_size < 1:
            raise ValueError("view_chunk_size必须为正整数")
        _validate_view_padding(self.view_chunk_size, self.view_pad_multiple)
        if self.epochs < 2 or not 0 <= self.freeze_epochs < self.epochs:
            raise ValueError("epochs至少2，freeze_epochs须留出全网微调阶段")
        if not 0 <= self.warmup_epochs < self.epochs - self.freeze_epochs:
            raise ValueError("warmup_epochs须小于全网微调轮数")
        if self.accumulation_steps < 1 or self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("累积/预取必须为正，worker数不能为负")
        if self.train_eval_every < 0 or self.rank_warmup_epochs < 0 or self.rank_ramp_epochs < 1:
            raise ValueError("训练诊断间隔/排序预热不能为负，排序渐增轮数必须为正")
        for name, value in (("backbone_lr", self.backbone_lr), ("head_lr", self.head_lr),
                            ("grad_clip", self.grad_clip)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}必须是有限正数")
        for name, value in (("weight_decay", self.weight_decay), ("rank_weight", self.rank_weight),
                            ("worker_timeout", self.worker_timeout)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}必须是有限非负数")
        if not math.isfinite(self.min_lr_ratio) or not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio须在(0,1]")
        if not math.isfinite(self.balanced_probability) or not 0 <= self.balanced_probability <= 1:
            raise ValueError("balanced_probability须在[0,1]")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay须在[0,1)")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed须在[0,2**32)")
        self.model_config()


def preprocessing_config() -> dict[str, Any]:
    return {"image_size": 384, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
            "cache_mode": "RGB", "data_source": "raw_readonly",
            "eval_transform": "EXIF+RGB+LetterboxLANCZOS+Normalize", "tta": "identity"}


def preprocessing_sha256(preprocessing: dict[str, Any]) -> str:
    payload = json.dumps(preprocessing, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_run_worklog(work_root: Path, run_id: str, message: str) -> None:
    """日志写入可配置工作盘；源码根与原图目录不作为隐式输出位置。"""
    with (work_root / "WORKLOG.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n### {_now()} · {run_id}\n\n- {message}\n")


def study_logits_from_views(view_logits: torch.Tensor, view_indices: torch.Tensor,
                            view_counts: torch.Tensor) -> torch.Tensor:
    """稳定计算logit(mean(sigmoid(view_logits)))，所有有效视角均参与梯度。

    使用logsumexp而非先sigmoid再clamp；极端logits仍保持有限、可用的梯度。
    index/count在CPU，避免按Study查找时GPU同步；仅索引本身按需迁移。
    """
    if view_logits.ndim != 2 or view_logits.shape[1] != NUM_CLASSES:
        raise ValueError("逐图logits必须为[V,10]")
    if view_indices.device.type != "cpu" or view_counts.device.type != "cpu":
        raise ValueError("Study视角索引及计数须保留CPU")
    if view_indices.shape != (view_logits.shape[0],) or view_counts.ndim != 1:
        raise ValueError("视角索引/计数形状异常")
    if int(view_counts.sum()) != view_logits.shape[0] or bool((view_counts <= 0).any()):
        raise ValueError("每Study必须保留全部有效视角，计数不能为0")
    z = view_logits.float()
    outputs = []
    for index, count in enumerate(view_counts.tolist()):
        selected = torch.nonzero(view_indices == index, as_tuple=False).flatten()
        if selected.numel() != count:
            raise ValueError("Study分组索引与视角计数不一致")
        values = z.index_select(0, selected.to(z.device))
        # log(mean(p))-log(mean(1-p))，两边log(count)相消。
        outputs.append(torch.logsumexp(F.logsigmoid(values), dim=0)
                       - torch.logsumexp(F.logsigmoid(-values), dim=0))
    return torch.stack(outputs)


def _validate_view_padding(chunk_size: int, pad_multiple: int) -> None:
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size必须为正整数")
    if (type(pad_multiple) is not int or pad_multiple < 0
            or (pad_multiple and chunk_size % pad_multiple)):
        raise ValueError("view_pad_multiple必须为0或整除view_chunk_size的正整数")


def checkpointed_view_forward(model: nn.Module, cpu_images: torch.Tensor,
                              device: torch.device, cuda_anchor: torch.Tensor,
                              chunk_size: int = 1, *, pad_multiple: int = 0) -> torch.Tensor:
    """输入保留CPU；整模型非重入检查点不保留各视角的GPU大激活。

    空CUDA anchor显式登记随机数设备，确保Dropout/DropPath在重算时重放。
    按图片数分块，而非按Study数整袋搬上GPU；云端可提高块大小以增加吞吐。
    """
    if cpu_images.device.type != "cpu" or cpu_images.ndim != 4 or cpu_images.shape[0] < 1:
        raise ValueError("视角必须是非空CPU图像张量")
    _validate_view_padding(chunk_size, pad_multiple)

    def forward_one(pixels: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(pixels.to(anchor.device, non_blocking=True))

    pieces = []
    for start in range(0, cpu_images.shape[0], chunk_size):
        pixels = cpu_images[start:start + chunk_size]
        real_count = pixels.shape[0]
        processed_count = (math.ceil(real_count / pad_multiple) * pad_multiple
                           if pad_multiple else real_count)
        if processed_count > real_count:
            # Only CPU working memory is padded. Keep pinned-transfer behavior;
            # source pixels, Study indices/counts and hard labels stay unchanged.
            padded = torch.empty((processed_count, *pixels.shape[1:]), dtype=pixels.dtype,
                                 device="cpu", pin_memory=pixels.is_pinned())
            padded[:real_count].copy_(pixels)
            padded[real_count:].copy_(pixels[-1:])
            pixels = padded
        logits = checkpoint(forward_one, pixels, cuda_anchor,
                            use_reentrant=False, preserve_rng_state=True)
        # The current backbone/head operate independently across samples. These
        # trailing model outputs never enter Study aggregation or the loss.
        pieces.append(logits[:real_count])
    return torch.cat(pieces, dim=0)


def build_optimizer(model: nn.Module, config: HighResConfig) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        role = "head" if id(parameter) in head_ids else "backbone"
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        buckets.setdefault((role, decay), []).append(parameter)
    groups = [{"params": parameters, "role": role, "name": f"{role}_{'decay' if decay else 'no_decay'}",
               "lr": config.head_lr if role == "head" else config.backbone_lr,
               "weight_decay": config.weight_decay if decay else 0.0}
              for (role, decay), parameters in buckets.items()]
    # 限制foreach/fused额外临时显存，保持与已验收压测相同的优化器策略。
    return torch.optim.AdamW(groups, weight_decay=0.0, foreach=False, fused=False)


def configure_phase(model: nn.Module, epoch: int, config: HighResConfig) -> None:
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    frozen = epoch <= config.freeze_epochs
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen or id(parameter) in head_ids)
    model.train()
    if frozen:
        model.backbone.eval()


def set_learning_rates(optimizer: torch.optim.Optimizer, config: HighResConfig,
                       epoch: int, updates_in_epoch: int, windows: int) -> dict[str, float]:
    if epoch <= config.freeze_epochs:
        factor = 1.0
    else:
        position = epoch - config.freeze_epochs - 1 + updates_in_epoch / max(windows, 1)
        if config.warmup_epochs and position < config.warmup_epochs:
            factor = 0.1 + 0.9 * position / config.warmup_epochs
        else:
            duration = config.epochs - config.freeze_epochs - config.warmup_epochs
            fraction = min(1.0, max(0.0, (position - config.warmup_epochs) / duration))
            factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    for group in optimizer.param_groups:
        base = config.head_lr if group["role"] == "head" else config.backbone_lr
        group["lr"] = 0.0 if group["role"] == "backbone" and epoch <= config.freeze_epochs else base * factor
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def rank_weight_at(epoch: int, config: HighResConfig) -> float:
    return config.rank_weight * min(1.0, max(0.0,
        (epoch - config.rank_warmup_epochs) / config.rank_ramp_epochs))


def train_epoch(model: nn.Module, loader: Any, criterion: StudyBCERankLoss,
                optimizer: torch.optim.Optimizer, scaler: Any, device: torch.device,
                epoch: int, config: HighResConfig, progress: dict[str, int]) -> dict[str, Any]:
    configure_phase(model, epoch, config)
    loader.batch_sampler.set_epoch(epoch)
    loader.dataset.reseed((config.seed + epoch * 1009) % 2**32)
    optimizer.zero_grad(set_to_none=True)
    windows = math.ceil(len(loader) / config.accumulation_steps)
    start_updates = progress["optimizer_updates"]
    rank_weight = rank_weight_at(epoch, config)
    cuda_anchor = torch.empty(0, device=device)
    total_loss = total_bce = total_rank = 0.0
    study_draws = image_draws = 0
    last_rates = set_learning_rates(optimizer, config, epoch, 0, windows)
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        if batch["split"] != "train" or batch["is_mixed"] or batch["targets"].shape != (config.batch_size, NUM_CLASSES):
            raise ValueError("高分辨率训练必须满足配置的Study批大小，使用原始硬标签、不做逐图混合")
        count = batch["targets"].shape[0]
        # 尾窗口按真实微批数归一化；云端accumulation_steps=1时每微批即一次更新。
        window_start = batch_index // config.accumulation_steps * config.accumulation_steps
        window_size = min(config.accumulation_steps, len(loader) - window_start)
        view_logits = checkpointed_view_forward(model, batch["images"], device, cuda_anchor,
                                                chunk_size=config.view_chunk_size,
                                                pad_multiple=config.view_pad_multiple)
        study_logits = study_logits_from_views(view_logits, batch["view_study_indices"], batch["view_counts"])
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        loss, bce, rank = criterion(study_logits, targets, rank_weight=rank_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Study训练损失出现NaN/Inf")
        scaler.scale(loss / window_size).backward()
        progress["backward_batches"] += 1
        total_loss += float(loss.detach()) * count
        total_bce += float(bce.detach()) * count
        total_rank += float(rank.detach()) * count
        study_draws += count
        image_draws += batch["images"].shape[0]
        # 释放本微批计算图，若启用累积也只保存参数梯度。
        del loss, bce, rank, study_logits, view_logits, targets
        if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(loader):
            last_rates = set_learning_rates(optimizer, config, epoch,
                progress["optimizer_updates"] - start_updates, windows)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=False)
            before = progress["optimizer_updates"]
            scaler.step(optimizer)
            scaler.update()
            if before == progress["optimizer_updates"]:
                progress["amp_skipped_steps"] += 1
            optimizer.zero_grad(set_to_none=True)
        if batch_index == 0 or (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
            print(f"Epoch {epoch} | Study batch {batch_index + 1}/{len(loader)} | "
                  f"loss={total_loss / study_draws:.5f} BCE={total_bce / study_draws:.5f} "
                  f"rank={total_rank / study_draws:.5f} rank_weight={rank_weight:.3f} "
                  f"updates={progress['optimizer_updates']} EMA={progress['ema_updates']} "
                  f"GPU_peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB", flush=True)
    if study_draws != config.batch_size * len(loader):
        raise RuntimeError("Study采样预算未完成，不能记录完整epoch")
    return {"loss": total_loss / study_draws, "bce": total_bce / study_draws,
            "rank": total_rank / study_draws, "rank_weight": rank_weight,
            "study_draws": study_draws, "image_draws": image_draws,
            "learning_rates": last_rates, "sampling": loader.batch_sampler.report(),
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


@torch.inference_mode()
def evaluate_highres(model: nn.Module, loader: Any, device: torch.device,
                     *, manifest: Path, split: str, epoch: int, diagnostic_path: Path) -> dict[str, Any]:
    """完整逐图概率后按原Study均值评估；诊断BCE也按Study等权。

    缺图时不输出正式指标/阈值，不用截断Study选择best。GPU一次仅一张图。
    """
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
            raise ValueError("评估只允许原始硬标签与确定性单图输入")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch["images"].to(device, non_blocking=True))
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


@contextmanager
def ema_evaluation_weights(model: nn.Module, ema: Any):
    """CPU EMA暂载入唯一GPU模型，结束恢复raw；不同时驻留两份GPU模型。"""
    raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(ema.module.state_dict(), strict=True)
        yield model
    finally:
        model.load_state_dict(raw_state, strict=True)


def save_bundle(directory: Path, checkpoint_payload: dict[str, Any], record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    digest = _save_checkpoint(directory / "best.pt", checkpoint_payload)
    _write_json(directory / "thresholds.json", {
        "schema_version": 3, "epoch": checkpoint_payload["epoch"],
        "checkpoint_sha256": digest, "checkpoint_filename": "best.pt",
        "class_ids": list(CLASS_IDS), "thresholds": checkpoint_payload["thresholds"],
        "aggregation": "study_mean_probability", "weights_source": "ema",
        "preprocessing_sha256": preprocessing_sha256(checkpoint_payload["preprocessing"]),
        "selection": "complete valid Study metrics; thresholds tuned on same valid",
    })
    _write_json(directory / "best_metrics.json", record)


def run_training(config: HighResConfig) -> dict[str, Any]:
    """用户手动运行的新run；没有自动探测训练、试跑、续训或OOM重试。"""
    config.validate()
    _metrics_imports()
    output = config.output.resolve()
    work_root = config.work_root.resolve()
    requested_image_root = config.raw_root.resolve()
    raw_root = resolve_image_root(requested_image_root, split="train")
    if work_root.is_relative_to(requested_image_root) or work_root.is_relative_to(raw_root):
        raise ValueError("work_root不能位于只读原图目录内")
    runs_root = (work_root / "runs").resolve()
    if runs_root != work_root / "runs" or output == runs_root or not output.is_relative_to(runs_root) or output.exists():
        raise ValueError("output必须是runs内不存在的新目录，不能覆盖旧实验/数据")
    train_records = read_manifest(config.train_manifest, split="train")
    valid_records = read_manifest(config.valid_manifest, split="valid")
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {row.get(key) for row in train_records} - {None, ""}
        right = {row.get(key) for row in valid_records} - {None, ""}
        if left & right:
            raise ValueError(f"train/valid的{key}发生交叉")
    # 本入口强制CUDA+AMP，避免误用CPU环境开始一次极慢的训练。
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("高分辨率入口需要CUDA版PyTorch；不回退CPU、不关闭AMP")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # set_device不接受无index的torch.device('cuda')，将默认值明确为可见设备0。
    if device.index is None:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False  # 避免卷积自动搜索增加显存峰值。
    source_hashes = {"train": _sha256(config.train_manifest), "valid": _sha256(config.valid_manifest)}
    config_dict = _jsonable(asdict(config))
    config_dict.update({"resolved_image_root": str(raw_root), "resolved_work_root": str(work_root)})
    preprocessing = preprocessing_config()
    model_config = config.model_config()
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    _write_json(output / "config.json", config_dict)
    _write_json(output / "provenance.json", {
        "created": _now(), "config": config_dict, "preprocessing": preprocessing,
        "model_config": model_config, "train_manifest_sha256": source_hashes["train"],
        "valid_manifest_sha256": source_hashes["valid"],
        "code_sha256": {name: _sha256(PROJECT_ROOT / name) for name in
            ("dataset.py", "dataset_highres.py", "model_highres.py", "train_highres.py",
             "train.py", "inference.py", "runtime_compat.py")},
        "memory_policy": {"amp": True, "gradient_checkpointing": True, "cpu_checkpoint_inputs": True,
            "micro_studies": config.batch_size, "view_chunk_size": config.view_chunk_size,
            "view_pad_multiple": config.view_pad_multiple, "padding_outputs_excluded": True,
            "accumulation_steps": config.accumulation_steps,
            "ema_device": "cpu", "adamw_foreach": False, "image_size": 384},
    })
    append_run_worklog(work_root, run_id, "任务：384原图/ConvNeXt V2类别注意力/Study BCE+Rank新run；"
        f"微批{config.batch_size}、累积{config.accumulation_steps}、AMP/检查点/CPU EMA；初始完成0轮，尚未反传。")
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "ema_updates": 0, "amp_skipped_steps": 0}
    stop_requested = False
    interrupt_count = 0
    step_hook = None
    status = "preparing"
    best_auc = (-math.inf, -math.inf)
    best_f1 = (-math.inf, -math.inf)
    best_epoch = None
    stop_file = config.stop_file.resolve() if config.stop_file else output / "STOP"

    def request_stop(_signum, _frame):
        nonlocal stop_requested, interrupt_count
        interrupt_count += 1
        if interrupt_count > 1:
            raise KeyboardInterrupt("第二次中断立即退出，当前未完成轮不冒称可恢复")
        stop_requested = True
        print("=> 已请求轮末停止；本轮完整验证与保存后退出。", flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        shared = {"image_size": 384, "num_workers": config.num_workers,
            "prefetch_factor": config.prefetch_factor, "timeout": config.worker_timeout,
            "seed": config.seed, "raw_root": raw_root}
        print("=> 构建只读原图Study采样器与完整评估loader...", flush=True)
        train_loader = make_study_loader(config.train_manifest, batch_size=config.batch_size,
            balanced_probability=config.balanced_probability, **shared)

        def evaluation_loader(split: str, serial: bool = False):
            options = {**shared, "num_workers": 0 if serial else config.num_workers}
            manifest = config.valid_manifest if split == "valid" else config.train_manifest
            return make_image_loader(manifest, split=split, batch_size=1, training=False, **options)

        valid_loader = evaluation_loader("valid")
        train_eval_loader = evaluation_loader("train") if config.train_eval_every else None
        print(f"=> 初始化{config.backbone_name}，首次用户运行可能下载预训练权重...", flush=True)
        # 先在CPU构建EMA再移动raw，避免ModelEmaV3先deepcopy一个CUDA模型的瞬时峰值。
        model = build_highres_model(pretrained=True, **model_config)
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=config.ema_decay, device="cpu", use_warmup=True,
                         warmup_power=0.75, exclude_buffers=True, foreach=False)
        ema.module.requires_grad_(False)
        model.to(device)
        optimizer = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        criterion = StudyBCERankLoss(rank_weight=config.rank_weight).to(device)

        def record_optimizer_step(_optimizer, _args, _kwargs):
            progress["optimizer_updates"] += 1
            ema.update(model, step=progress["optimizer_updates"])
            progress["ema_updates"] += 1

        step_hook = optimizer.register_step_post_hook(record_optimizer_step)
        status = "running"
        for epoch in range(1, config.epochs + 1):
            if stop_requested or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
            print(f"=> Epoch {epoch}/{config.epochs}，384，{config.batch_size} Study × {config.accumulation_steps}累积，"
                  f"GPU每块{config.view_chunk_size}视角，尾块补齐倍数={config.view_pad_multiple}，"
                  "AMP/检查点启用，EMA=CPU。", flush=True)
            training = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config, progress)
            if progress["ema_updates"] == 0:
                raise RuntimeError("没有成功优化/EMA更新，不能保存初始模型为best")
            train_metrics = _unavailable_metrics()
            train_eval = {"complete": False, "loss": None, "reason": "本轮未安排完整训练诊断"}
            thresholds = None
            val_metrics = _unavailable_metrics()
            fixed_metrics = _unavailable_metrics()
            # 整个验证/训练诊断阶段不触发optimizer hook，结束后恢复raw继续训练。
            with ema_evaluation_weights(model, ema) as evaluation_model:
                validation = evaluate_highres(evaluation_model, valid_loader, device,
                    manifest=config.valid_manifest, split="valid", epoch=epoch,
                    diagnostic_path=output / "data_loading_events.jsonl")
                if validation["complete"]:
                    thresholds = search_class_thresholds(validation["targets"], validation["probabilities"])
                    val_metrics = calculate_metrics(validation["targets"], validation["probabilities"], thresholds)
                    fixed_metrics = calculate_metrics(validation["targets"], validation["probabilities"], [0.5] * NUM_CLASSES)
                    if train_eval_loader is not None and (epoch == 1 or epoch % config.train_eval_every == 0 or epoch == config.epochs):
                        train_eval = evaluate_highres(evaluation_model, train_eval_loader, device,
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
                    raise RuntimeError("训练期间清单发生变化")
            ema_metadata = {"enabled": True, "decay": config.ema_decay, "use_warmup": True,
                "warmup_power": 0.75, "updates": progress["ema_updates"], "buffers": "copy_from_raw",
                "device": "cpu"}
            score_auc = (val_metrics["macro_auc"], val_metrics["macro_f1"]) if validation["complete"] else None
            score_f1 = (val_metrics["macro_f1"], val_metrics["macro_auc"]) if validation["complete"] else None
            is_best = score_auc is not None and score_auc > best_auc
            is_best_f1 = score_f1 is not None and score_f1 > best_f1
            epoch_progress = {**progress, "completed_epochs": epoch,
                "validated_epochs": progress["validated_epochs"] + int(validation["complete"])}
            record = {"epoch": epoch, "phase": "head_only" if epoch <= config.freeze_epochs else "full_finetune",
                "train_loss": training["loss"], "training": training, "train": train_metrics,
                "train_eval_loss": train_eval["loss"], "val": val_metrics, "val_loss": validation["loss"],
                "val_macro_f1_at_05": fixed_metrics["macro_f1"], "thresholds": thresholds,
                "validation_complete": validation["complete"], "weights_source": "ema", "ema": ema_metadata,
                "metrics_unit": "Study", "f1_threshold_source": "same_epoch_valid" if thresholds is not None else None,
                "is_best": is_best, "is_best_f1": is_best_f1, **epoch_progress,
                "data_loading": {"valid": {key: validation[key] for key in
                    ("complete", "evaluated_images", "expected_images", "skipped_samples", "fallback_serial")}}}
            payload = {"schema_version": 3, "model_name": "convnextv2_study_attention",
                "model_config": model_config, "num_classes": NUM_CLASSES, "class_ids": list(CLASS_IDS),
                "epoch": epoch, "aggregation": "study_mean_probability", "thresholds": thresholds,
                "weights_source": "ema", "validation_weights_source": "ema", "threshold_weights_source": "ema",
                "ema": ema_metadata, "preprocessing": preprocessing,
                "model_state_dict": {name: value.detach().cpu().clone() for name, value in ema.module.state_dict().items()},
                "raw_model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "train_manifest_sha256": source_hashes["train"], "valid_manifest_sha256": source_hashes["valid"],
                "validation_complete": validation["complete"], "inference_ready": validation["complete"],
                "validation": val_metrics if validation["complete"] else None,
                "loss_config": {"name": "StudyBCE+PairwiseLogisticRank", "rank_weight": training["rank_weight"],
                    "aggregation": "logit_mean_sigmoid_via_logsumexp", "pos_weight": None},
                "training_config": config_dict, "progress": epoch_progress,
                "schedule": {"kind": "head_warmup_cosine", "learning_rates": training["learning_rates"]},
                "resume_supported": False}
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
                "best_epoch": best_epoch, "validation_complete": validation["complete"], "updated": _now()})
            print(f"epoch={epoch} train_AUC={_metric_text(train_metrics['macro_auc'])} "
                  f"val_AUC={_metric_text(val_metrics['macro_auc'])} "
                  f"val_F1_tuned={_metric_text(val_metrics['macro_f1'])} "
                  f"val_F1_05={_metric_text(fixed_metrics['macro_f1'])} "
                  f"val_AP={_metric_text(val_metrics['macro_ap'])} weights=ema best={is_best} "
                  f"complete={validation['complete']}", flush=True)
            print("per_class_AP=" + json.dumps(val_metrics["per_class_ap"]) +
                  " per_class_F1=" + json.dumps(val_metrics["per_class_f1"]), flush=True)
            append_run_worklog(work_root, run_id, f"结果：epoch={epoch}，完整验证={validation['complete']}，"
                f"Study AUC/F1={_metric_text(val_metrics['macro_auc'])}/{_metric_text(val_metrics['macro_f1'])}；"
                f"EMA更新={progress['ema_updates']}、AMP跳步={progress['amp_skipped_steps']}，best={is_best}；"
                "事实：有效图像不改写；推测：泛化收益仍需独立评价。")
            with (work_root / "EXPERIMENTS.md").open("a", encoding="utf-8") as stream:
                stream.write(f"| {run_id}-E{epoch:03d} | raw384; {config.backbone_name}; Study BCE+Rank; "
                    f"micro{config.batch_size}/accum{config.accumulation_steps}; EMAcpu; rank={training['rank_weight']} | "
                    f"loss={training['loss']:.6f}; draws={training['study_draws']}Study | "
                    f"AUC={_metric_text(val_metrics['macro_auc'])}; F1={_metric_text(val_metrics['macro_f1'])} | "
                    f"未使用 | valid调优成绩；完整={validation['complete']} | "
                    f"{'保留best' if is_best else '保留last/日志'} | {output}/history.jsonl |\n")
            del payload
            if stop_requested or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
        else:
            status = "completed"
    except BaseException as exc:
        status = "failed_or_interrupted"
        advice = ""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            advice = ("可用新output减小--view-chunk-size至1，或减小偶数--batch-size并相应增加累积；也可改"
                      "--backbone convnextv2_tiny.fcmae_ft_in1k --attention-dim 128 --attention-layers 1。"
                      "减小梯度累积次数不会降低单视角前向峰值；当前不自动重试。")
            print("=> CUDA OOM：" + advice, flush=True)
        _write_json(output / "status.json", {"state": status, **progress, "best_epoch": best_epoch,
            "error": f"{type(exc).__name__}: {exc}", "oom_advice": advice, "updated": _now()})
        append_run_worklog(work_root, run_id, f"失败/中断：{type(exc).__name__}: {exc}；"
            f"完成{progress['completed_epochs']}轮、实际反传{progress['backward_batches']}微批；"
            "保留此前完整轮权重；未完成轮不视作已验收或可精确续训。")
        raise
    finally:
        if step_hook is not None:
            step_hook.remove()
        signal.signal(signal.SIGINT, previous_handler)
    result = {"state": status, **progress, "best_epoch": best_epoch, "output": str(output), "updated": _now()}
    _write_json(output / "status.json", result)
    append_run_worklog(work_root, run_id, f"交接：{status}，完成{progress['completed_epochs']}轮、"
        f"完整验证{progress['validated_epochs']}轮；best epoch={best_epoch}，训练停止后由用户另行授权推理。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="用户手动启动的384原图Study训练；默认AMP/检查点/CPU EMA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--image-root", "--data-root", "--raw-root", dest="raw_root", type=Path, default=RAW_ROOT,
                        help="只读原图根；支持直接train/valid/test目录或外层cxr_first目录")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT,
                        help="工作盘根；run输出须位于其runs子目录，日志也写入此根")
    parser.add_argument("--backbone", dest="backbone_name", choices=SUPPORTED_BACKBONES, default=HIGHRES_MODEL_NAME)
    parser.add_argument("--image-size", type=int, choices=(384,), default=384)
    parser.add_argument("--batch-size", type=int, default=2, help="每微批Study数，至少2的偶数；云端run_cloud.sh显式设置")
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--view-chunk-size", type=int, default=1, help="同时送入GPU的视角数，独立于Study batch")
    parser.add_argument("--view-pad-multiple", type=int, default=0,
                        help="可选CPU尾块补齐倍数；须整除chunk，0关闭，额外输出不参与Study/loss")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--rank-warmup-epochs", type=int, default=3)
    parser.add_argument("--rank-ramp-epochs", type=int, default=3)
    parser.add_argument("--balanced-probability", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--train-eval-every", type=int, default=5, help="完整训练集诊断间隔；0关闭，valid每轮完整")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_training(HighResConfig(**vars(args))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
