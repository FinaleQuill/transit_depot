"""EfficientNet-B3训练/验证定义，默认软标签ASL与EMA；代码修改不执行训练。

未来显式运行时才读取缓存、下载预训练权重及创建GPU任务；import不会训练。
必须在用户后续授权后才运行CLI。原图、images300、清洗CSV均保持只读。
依赖还包括scikit-learn（用于标准AUC、F1与精确PR阈值搜索），本次不安装。

指标口径：优化loss来自在线增强/混合的训练批次；train AUC/F1另做全量、
无增强、无混合的训练集评估。train/valid均先按Study平均概率再评价。
启用EMA时，train/valid评估、F1阈值与best均来自同一EMA模型；推理直接
加载checkpoint的model_state_dict，不需要再次平滑。验证独立搜索十类F1
阈值，以Study Macro-AUC选best，完全相等时才比较F1。
调阈值与选模型复用valid，调优F1是样本内成绩，不是独立测试成绩。

接口依据：
https://docs.pytorch.org/docs/2.10/amp.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
"""
from __future__ import annotations

# 必须早于numpy/torch/dataset导入；否则库导入阻塞时，main里的print永远到不了。
# spawn子进程也会执行兼容处理，但只由主入口打印，避免8个worker重复刷屏。
if __name__ == "__main__":
    print("=> train.py 已进入，正在初始化启动环境...", flush=True)

from runtime_compat import prepare_windows_runtime

_runtime_fallback_enabled = prepare_windows_runtime()


def _startup_message(message: str) -> None:
    if __name__ == "__main__":
        print(message, flush=True)


if _runtime_fallback_enabled:
    _startup_message("=> 已启用Windows系统信息回退，绕过本机WMI查询阻塞。")

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
from typing import Any

_startup_message("=> 正在导入 NumPy...")
import numpy as np
_startup_message("=> NumPy 就绪，正在导入 PyTorch...")
import torch
from torch import nn
from torch.utils.data import SequentialSampler

_startup_message("=> PyTorch 就绪，正在导入 dataset（OpenCV/Pillow）...")
from dataset import (
    CLASS_IDS, DEFAULT_TRAIN_MANIFEST, DEFAULT_VALID_MANIFEST, IMAGENET_MEAN,
    IMAGENET_STD, NUM_CLASSES, PROJECT_ROOT, MixConfig, StudyPredictionAggregator,
    CachedSampleError, build_transforms, make_dataloader, read_manifest,
)
_startup_message("=> dataset 就绪，正在导入模型定义...")
from model import build_criterion, build_model
_startup_message("=> 启动依赖就绪，进入命令行参数解析。")


@dataclass(frozen=True)
class TrainConfig:
    output: Path
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST
    valid_manifest: Path = DEFAULT_VALID_MANIFEST
    epochs: int = 20
    freeze_epochs: int = 2  # 1、2轮只训头，3轮起解冻；设0表示从第一轮全网微调。
    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 1
    worker_timeout: float = 120.0  # 多进程取批超时；串行loader由工厂自动设为0。
    head_lr: float = 5e-4
    finetune_lr: float = 5e-6
    finetune_head_lr: float = 5e-5
    weight_decay: float = 1e-3
    dropout: float = 0.3
    drop_path: float = 0.1
    freeze_bn: bool = True  # 小batch混合图像下固定预训练BN统计；仿射参数仍可微调。
    loss_name: str = "asl"
    pos_weight_power: float = 0.5
    max_pos_weight: float | None = 10.0
    focal_gamma: float = 2.0  # 仅--loss-name focal时使用。
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    ema: bool = True
    ema_decay: float = 0.999
    ema_warmup: bool = True
    mix_probability: float = 0.5
    cutmix_probability: float = 0.2  # 条件概率；全部batch中约10% CutMix、40% Mixup。
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    early_stopping_patience: int = 5  # 0可关闭；只计完整验证的解冻轮次。
    early_stopping_min_delta: float = 0.001
    early_stopping_min_epochs: int = 8
    grad_clip: float = 1.0
    image_size: int = 300
    seed: int = 43
    device: str = "auto"
    amp: bool = True
    deterministic: bool = True
    stop_file: Path | None = None

    def validate(self) -> None:
        if self.epochs < 1 or not 0 <= self.freeze_epochs < self.epochs:
            raise ValueError("epochs必须为正，freeze_epochs应在[0,epochs)且留出解冻阶段")
        if self.batch_size < 1 or self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("batch_size/prefetch_factor必须为正，num_workers不能为负")
        if not math.isfinite(self.worker_timeout) or self.worker_timeout < 0:
            raise ValueError("worker_timeout必须为有限非负秒数；0表示禁用超时")
        rates = (self.head_lr, self.finetune_lr, self.finetune_head_lr, self.grad_clip)
        if any(not math.isfinite(x) or x <= 0 for x in rates):
            raise ValueError("学习率与grad_clip应为正有限数")
        if max(self.finetune_lr, self.finetune_head_lr) >= self.head_lr:
            raise ValueError("解冻阶段的两个学习率必须均小于冻结阶段头部学习率")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay应为非负有限数")
        if any(not math.isfinite(x) or not 0 <= x < 1 for x in (self.dropout, self.drop_path)):
            raise ValueError("dropout/drop_path应在[0,1)")
        if self.loss_name not in {"bce", "focal", "asl"}:
            raise ValueError("loss_name只能是bce、focal或asl")
        if not math.isfinite(self.pos_weight_power) or not 0 <= self.pos_weight_power <= 1:
            raise ValueError("pos_weight_power应在[0,1]；1为原始比率、0为不加权")
        if self.max_pos_weight is not None and (not math.isfinite(self.max_pos_weight) or self.max_pos_weight < 1):
            raise ValueError("max_pos_weight应至少为1或None")
        if not math.isfinite(self.focal_gamma) or self.focal_gamma < 0:
            raise ValueError("focal_gamma应为非负有限数")
        if any(not math.isfinite(value) or value < 0
               for value in (self.asl_gamma_pos, self.asl_gamma_neg)):
            raise ValueError("ASL正/负gamma应为非负有限数")
        if not math.isfinite(self.asl_clip) or not 0 <= self.asl_clip < 1:
            raise ValueError("asl_clip应在[0,1)")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay应在[0,1)")
        if type(self.ema) is not bool or type(self.ema_warmup) is not bool:
            raise ValueError("ema与ema_warmup必须为布尔值")
        MixConfig(mix_probability=self.mix_probability, cutmix_probability=self.cutmix_probability,
                  mixup_alpha=self.mixup_alpha, cutmix_alpha=self.cutmix_alpha)
        if self.early_stopping_patience < 0 or self.early_stopping_min_epochs < 0:
            raise ValueError("早停patience/min_epochs不能为负；patience=0表示关闭")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("早停min_delta应为非负有限数")
        if self.image_size != 300:
            raise ValueError("本次版本固定300输入，改变尺寸需另做有记录的对照")
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("seed应在[0,2**32)")


@dataclass
class EarlyStopping:
    """仅用完整验证Study Macro-AUC；不让调阈值后的F1噪声驱动停止。

    best.pt仍保留任何真实AUC新高；这里单独记录“至少提升min_delta”的
    参考值，避免1e-5级抖动不断重置耐心。冻结期/缺图验证由调用方跳过。
    最早第min_epochs轮才累计停滞轮数；patience=0关闭自动停止。
    """
    patience: int = 5
    min_delta: float = 0.001
    min_epochs: int = 8
    best_auc: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def update(self, epoch: int, auc: float) -> bool:
        if not math.isfinite(auc):
            raise ValueError("早停不能接收非有限AUC")
        if self.best_auc is None or auc > self.best_auc + self.min_delta:
            self.best_auc, self.best_epoch, self.bad_epochs = auc, epoch, 0
        elif epoch >= self.min_epochs and self.patience > 0:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        return self.patience > 0 and epoch >= self.min_epochs and self.bad_epochs >= self.patience


def _now() -> str:
    # 香港当前使用UTC+8；固定偏移避免Windows缺IANA时区数据库时无法写日志。
    return datetime.now(timezone(timedelta(hours=8), name="Asia/Hong_Kong")).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_worklog(run_id: str, message: str) -> None:
    """仅未来实际运行才记关键事件，不在每个batch写流水账。"""
    with (PROJECT_ROOT / "WORKLOG.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n### {_now()} · {run_id}\n\n- {message}\n")


def _metrics_imports():
    # 延迟导入，单纯阅读/import模型代码不要求安装评估依赖。
    try:
        from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
    except ImportError as exc:
        raise ImportError("未来运行需准备scikit-learn；本文件不会自动安装") from exc
    return roc_auc_score, f1_score, precision_recall_curve, average_precision_score


def _check_hard_predictions(targets: np.ndarray, probabilities: np.ndarray) -> None:
    if targets.ndim != 2 or targets.shape != probabilities.shape or targets.shape[1] != NUM_CLASSES:
        raise ValueError("评估必须使用形状一致的[N,10]硬标签和概率")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("AUC/F1不可直接用Mixup/CutMix的软标签计算")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("评估输入必须是有限的[0,1]概率")
    # 不悄悄nanmean：单类没有正例/负例时宏AUC的口径已经发生变化，直接报错。
    if targets.shape[0] == 0 or any(np.unique(targets[:, c]).size != 2 for c in range(NUM_CLASSES)):
        raise ValueError("每类都必须有正负Study，不能伪造AUC或默填阈值0.5")


def search_class_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> list[float]:
    """遍历每类所有唯一概率断点，精确最大化该类验证F1，不用粗网格。

    precision_recall_curve第k点对应proba>=thresholds[k]；最后一点没有阈值。
    排除最后一点后计算F1；若并列，取更高阈值，规则固定、可复现。
    十类独立最大化F1等价于在此数据上最大化十类的Macro-F1。
    """
    _check_hard_predictions(targets, probabilities)
    _, _, precision_recall_curve, _ = _metrics_imports()
    selected: list[float] = []
    for c in range(NUM_CLASSES):
        precision, recall, cutoffs = precision_recall_curve(targets[:, c], probabilities[:, c])
        denominator = precision[:-1] + recall[:-1]
        scores = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                           out=np.zeros_like(denominator), where=denominator > 0)
        best = np.flatnonzero(scores == scores.max())[-1]
        selected.append(float(cutoffs[best]))
    return selected


def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray,
                      thresholds: list[float]) -> dict[str, Any]:
    _check_hard_predictions(targets, probabilities)
    gates = np.asarray(thresholds, dtype=np.float64)
    if gates.shape != (NUM_CLASSES,) or not np.isfinite(gates).all() or (gates < 0).any() or (gates > 1).any():
        raise ValueError("阈值必须为十维有限[0,1]数组")
    roc_auc_score, f1_score, _, average_precision_score = _metrics_imports()
    aucs = [float(roc_auc_score(targets[:, c], probabilities[:, c])) for c in range(NUM_CLASSES)]
    f1s = [float(f1_score(targets[:, c], probabilities[:, c] >= gates[c], zero_division=0)) for c in range(NUM_CLASSES)]
    aps = [float(average_precision_score(targets[:, c], probabilities[:, c])) for c in range(NUM_CLASSES)]
    predicted, positive = probabilities >= gates, targets == 1
    tp = (predicted & positive).sum(axis=0)
    fp = (predicted & ~positive).sum(axis=0)
    fn = (~predicted & positive).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros(NUM_CLASSES), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(NUM_CLASSES), where=(tp + fn) > 0)
    # 下一轮可据此区分稀有类“漏报多”还是“误报多”，不再只根据低F1猜原因。
    return {"macro_auc": float(np.mean(aucs)), "macro_f1": float(np.mean(f1s)),
            "macro_ap": float(np.mean(aps)), "per_class_ap": aps,
            "per_class_auc": aucs, "per_class_f1": f1s,
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "per_class_positive_studies": positive.sum(axis=0).tolist(),
            "per_class_tp": tp.tolist(), "per_class_fp": fp.tolist(), "per_class_fn": fn.tolist()}


def _configure_phase(model: nn.Module, optimizer: torch.optim.Optimizer,
                     config: TrainConfig, epoch: int) -> str:
    """冻结既包括梯度，也包括BN状态和骨干的随机层。"""
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    frozen = epoch <= config.freeze_epochs
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen or id(parameter) in head_ids)
    if frozen:
        # 先全体eval冻结BN/DropPath，再只把分类头设为train，不能随后model.train()覆盖。
        model.eval()
        head.train()
        # timm B3的head Dropout是forward_head内的F.dropout，读取根模块
        # training标志，不是classifier子模块。只切根标志，不递归model.train()，
        # 让分类头Dropout生效，同时骨干BN/DropPath仍保持eval。
        model.training = True
    else:
        model.train()
        if config.freeze_bn:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    module.eval()
    # 骨干/头部各有衰减/不衰减组。冻结期骨干grad=None，不更新；解冻时下调头lr，
    # 保留头的Adam状态，不因重新创建optimizer无记录地重置已有动量。
    # 只在冻结期/首次解冻设置学习率。后续轮次保留调度器算出的值，
    # 否则每轮重置为初始lr会让余弦退火失效。
    if frozen or epoch == config.freeze_epochs + 1:
        for group in optimizer.param_groups:
            group["lr"] = (config.head_lr if frozen else config.finetune_head_lr) if group["role"] == "head" else config.finetune_lr
    return "head_only" if frozen else "full_finetune"


def _build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """骨干/头部分别设置lr；bias和一维BN缩放参数不施加权重衰减。"""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    if not head_ids:
        raise ValueError("模型分类头没有可训练参数")
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        role = "head" if id(parameter) in head_ids else "backbone"
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        buckets.setdefault((role, decay), []).append(parameter)
    groups = [{"params": parameters, "name": f"{role}_{'decay' if decay else 'no_decay'}",
               "role": role, "lr": config.head_lr if role == "head" else config.finetune_lr,
               "weight_decay": config.weight_decay if decay else 0.0}
              for (role, decay), parameters in buckets.items()]
    return torch.optim.AdamW(groups, weight_decay=0.0)


def train_one_epoch(model: nn.Module, loader: Any, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, scaler: Any,
                    device: torch.device, *, epoch: int, amp_enabled: bool,
                    grad_clip: float, progress: dict[str, int]) -> float:
    """标准优化循环；phase的train/eval模式由调用方设置，不在此覆盖冻结BN。"""
    loss_sum, samples = 0.0, 0
    # 从1计数便于和总批数对照，epoch由调用方显式传入，避免日志引用未定义变量。
    for batch_idx, batch in enumerate(loader, start=1):
        if batch["split"] != "train" or batch["targets"] is None:
            raise ValueError("训练循环只允许有标注的train")
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            # 软标签直接输入带logits的ASL/BCE/Focal，不在损失前sigmoid或argmax。
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练loss非有限数，停止并保留此前完整epoch记录")
        scaler.scale(loss).backward()
        progress["backward_batches"] += 1
        scaler.unscale_(optimizer)
        # AMP暂时溢出应由GradScaler跳过更新并降低scale，不能在这里先抛错。
        # 全精度的非有限梯度则直接报错，避免无效更新。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip,
                                 error_if_nonfinite=not amp_enabled)
        updates_before = progress["optimizer_updates"]
        scaler.step(optimizer)
        scaler.update()
        # post-step hook只在optimizer.step真实执行时触发；AMP跳步不会更新EMA。
        # 不用step返回值判断：AdamW成功更新时通常也返回None。
        if progress["optimizer_updates"] == updates_before:
            progress["amp_skipped_steps"] += 1
        count = images.shape[0]
        # 以实际样本数加权，尾批不足batch_size时不能等权平均batch loss。
        loss_sum += float(loss.detach()) * count
        samples += count
        # 首批确认已经完成一次优化；此后每50批及尾批报告进度。
        # flush=True立即提交输出，重定向终端日志时也不等待缓冲区积满。
        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}", flush=True)
    if samples != len(loader.dataset):
        raise ValueError("训练样本覆盖不完整，不能静默丢弃尾批")
    return loss_sum / samples


def _aggregate_clean_train(records: list[dict], image_ids: list[str],
                           probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """现有StudyPredictionAggregator仅允许valid/test；这里补训练诊断汇总。

    只接收eval模式、无增强无混合的完整训练预测；不能使用优化批次的身份元数据。
    每张图恰好一次、与原始硬标签一致，顺序按训练清单Study首出现顺序。
    """
    expected = {r["image_id"]: r for r in records}
    if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(expected):
        raise ValueError("干净训练评估漏图或重复图")
    if probabilities.shape != (len(image_ids), NUM_CLASSES) or targets.shape != probabilities.shape:
        raise ValueError("干净训练评估概率/标签形状错误")
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        groups.setdefault(r["study_key"], {"sum": np.zeros(NUM_CLASSES, dtype=np.float64),
                                          "count": 0, "target": r["targets"]})
    for image_id, probability, target in zip(image_ids, probabilities, targets):
        row = expected[image_id]
        if not np.array_equal(target, np.asarray(row["targets"])):
            raise ValueError("训练诊断标签不再等于原清单硬标签，可能误用了混合批次")
        group = groups[row["study_key"]]
        group["sum"] += probability
        group["count"] += 1
    return {"study_keys": list(groups),
            "probabilities": np.stack([g["sum"] / g["count"] for g in groups.values()]).astype(np.float32),
            "targets": np.asarray([g["target"] for g in groups.values()], dtype=np.float32)}


def _safe_eval_batches(loader: Any, skipped: list[dict[str, Any]],
                       events: list[dict[str, Any]], *, diagnostic_path: Path | None,
                       epoch: int | None, split: str):
    """仅防御取批/读图故障，模型执行不在异常捕获范围内。

    worker退出通常发生在next(iterator)，此时还没有拿到batch，不能打印
    上一个成功批次的ID冒充故障样本。因此评估必须顺序采样，按索引列出
    本来应读取的ID；再串行逐张复核，才知道具体哪张确实不可读。
    多进程迭代器失效后不再复用，剩余样本在主进程读取，不重跑已成功批次。
    """
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("防御性评估要求shuffle=False、drop_last=False，以准确定位失败ID")
    dataset = loader.dataset
    batch_size = loader.batch_size
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("防御性评估要求固定的正整数batch_size")

    def report(event: dict[str, Any]) -> None:
        entry = {"time": _now(), "epoch": epoch, "split": split, **event}
        events.append(entry)
        print("[数据加载] " + json.dumps(entry, ensure_ascii=False), flush=True)
        if diagnostic_path is not None:
            with diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    iterator = None
    serial = False
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        expected_ids = [dataset.records[i]["image_id"] for i in indices]
        if not serial:
            try:
                if iterator is None:
                    iterator = iter(loader)
                batch = next(iterator)
            except (CachedSampleError, RuntimeError, OSError, EOFError, StopIteration) as exc:
                # 这里的ID只是故障发生时待取的批次，不把整批都判为坏图。
                report({"event": "loader_fallback_serial", "expected_image_ids": expected_ids,
                        "error": f"{type(exc).__name__}: {exc}"})
                serial = True
            # 离开except释放异常traceback后再释放iterator，帮助回收非持久worker。
            # 不杀其他进程，也不通过PyTorch私有API管理worker。
            if serial:
                iterator = None
                gc.collect()
            else:
                if list(batch["image_ids"]) != expected_ids:
                    raise ValueError("评估批次与顺序采样ID不一致，不能继续计算指标")
                yield batch
                continue

        samples = []
        healthy_ids = []
        for index in indices:
            image_id = dataset.records[index]["image_id"]
            try:
                sample = dataset[index]
            except CachedSampleError as exc:
                # 只跳过读取/解码/在线变换异常；标签、身份、模型等错误继续报错。
                failure = {"image_id": image_id, "error": str(exc)}
                skipped.append(failure)
                report({"event": "sample_skipped", **failure})
                continue
            samples.append(sample)
            healthy_ids.append(image_id)
        if samples:
            batch = loader.collate_fn(samples)
            if list(batch["image_ids"]) != healthy_ids:
                raise ValueError("串行评估批次身份改变，不能继续计算指标")
            yield batch


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Any, criterion: nn.Module, device: torch.device,
             *, manifest: Path, split: str, amp_enabled: bool,
             diagnostic_path: Path | None = None, epoch: int | None = None) -> dict[str, Any]:
    """无梯度评估；缺图时返回complete=False，禁止用残缺Study挑选best。

    Worker崩溃后串行重读成功，仍是完整验证；若确认某图不可读，则打印ID、
    跳过并保留故障证据。部分验证loss只作诊断，不输出正式AUC/F1/阈值。
    """
    model.eval()
    aggregator = StudyPredictionAggregator(manifest, split="valid") if split == "valid" else None
    all_ids: list[str] = []
    all_probabilities, all_targets = [], []
    total_loss, total_bce, total_count = 0.0, 0.0, 0
    skipped: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for batch in _safe_eval_batches(loader, skipped, events, diagnostic_path=diagnostic_path,
                                    epoch=epoch, split=split):
        if batch["split"] != split or batch["is_mixed"] or batch["targets"] is None:
            raise ValueError("评估必须是同一划分、未混合的硬标签批次")
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("评估loss非有限数")
        # 训练损失的权重/公式可变；另记固定口径的无权重BCE，以免把
        # 新损失数值较小误称为拟合更好。旧run未记录此值，不能事后伪造。
        plain_bce = nn.functional.binary_cross_entropy_with_logits(logits.float(), targets, reduction="mean")
        # 每张图的logits只转一次概率，再平均概率；不对平均概率重复sigmoid。
        probabilities = torch.sigmoid(logits.float()).cpu()
        total_loss += float(loss) * images.shape[0]
        total_bce += float(plain_bce) * images.shape[0]
        total_count += images.shape[0]
        if aggregator is not None:
            aggregator.update(batch, probabilities)
        else:
            all_ids.extend(batch["image_ids"])
            all_probabilities.append(probabilities.numpy())
            all_targets.append(batch["targets"].numpy())
    complete = total_count == len(loader.dataset) and not skipped
    diagnostics = {"complete": complete, "evaluated_images": total_count,
                   "expected_images": len(loader.dataset), "skipped_samples": skipped,
                   "fallback_serial": any(e["event"] == "loader_fallback_serial" for e in events),
                   "loader_events": events, "loss": total_loss / total_count if total_count else None,
                   "bce_unweighted": total_bce / total_count if total_count else None}
    if not complete:
        # 不放宽Study汇总器的全量要求，也不在缺失某些视角后悄悄重算均值。
        return diagnostics
    result = aggregator.finalize() if aggregator is not None else _aggregate_clean_train(
        loader.dataset.records, all_ids, np.concatenate(all_probabilities), np.concatenate(all_targets))
    result.update(diagnostics)
    return result


def _unavailable_metrics() -> dict[str, Any]:
    result = {"macro_auc": None, "macro_f1": None, "macro_ap": None}
    result.update({key: [None] * NUM_CLASSES for key in
                   ("per_class_auc", "per_class_f1", "per_class_ap", "per_class_precision",
                    "per_class_recall", "per_class_positive_studies", "per_class_tp", "per_class_fp", "per_class_fn")})
    return result


def _metric_text(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def _evaluation_summary(result: dict[str, Any]) -> dict[str, Any]:
    """history只存覆盖情况及错误，不重复写大体积逐图概率。"""
    return {key: result.get(key) for key in
            ("complete", "evaluated_images", "expected_images", "skipped_samples",
             "fallback_serial", "loader_events", "reason")}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return _sha256(path)


def run_training(config: TrainConfig) -> dict[str, Any]:
    """仅未来获得授权后调用。新建run，未实现续训，不会覆盖旧run。"""
    # 清单校验、模型初始化可能耗时；先提示已进入准备阶段，再开始这些动作。
    print("=> 启动准备：检查数据、构建加载器与模型...", flush=True)
    config.validate()
    _metrics_imports()  # 依赖缺失先报错，不留下看似启动的空模型实验。
    output = config.output.resolve()
    runs_root = (PROJECT_ROOT / "runs").resolve()
    if runs_root != PROJECT_ROOT.resolve() / "runs":
        raise ValueError("runs不能是指向data/缓存或其他位置的链接")
    if output == runs_root or not output.is_relative_to(runs_root) or output.exists():
        raise ValueError("output必须是项目runs目录内尚不存在的新子目录")
    train_records = read_manifest(config.train_manifest, split="train")
    valid_records = read_manifest(config.valid_manifest, split="valid")
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        left = {r.get(key) for r in train_records} - {None, ""}
        right = {r.get(key) for r in valid_records} - {None, ""}
        if left & right:
            raise ValueError(f"train/valid的{key}发生交叉，不应在有泄漏时训练")
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    source_hashes = {"train": _sha256(config.train_manifest), "valid": _sha256(config.valid_manifest)}
    config_dict = _jsonable(asdict(config))
    _write_json(output / "config.json", config_dict)
    _write_json(output / "provenance.json", {"created": _now(), "train_manifest_sha256": source_hashes["train"],
        "valid_manifest_sha256": source_hashes["valid"], "code_sha256": {name: _sha256(PROJECT_ROOT / name)
        for name in ("dataset.py", "model.py", "train.py", "inference.py", "runtime_compat.py")}, "config": config_dict})
    _append_worklog(run_id, f"任务：新run注册，数据/配置见{output}；目标：冻结{config.freeze_epochs}轮后全网微调；"
                           f"计划{config.epochs}轮，当前完成0轮/反传0批，尚未启动优化。")
    progress = {"completed_epochs": 0, "validated_epochs": 0, "backward_batches": 0,
                "optimizer_updates": 0, "amp_skipped_steps": 0, "ema_updates": 0}
    optimizer_step_hook = None
    requested_stop = False
    interrupt_count = 0

    def request_stop(signum, frame):
        nonlocal requested_stop, interrupt_count
        del signum, frame
        interrupt_count += 1
        if interrupt_count > 1:
            raise KeyboardInterrupt("第二次中断立即退出；本轮可能不完整，不冒称可精确恢复")
        requested_stop = True
        print("已请求停止：完成本epoch训练、验证和保存后退出；STOP文件适合Windows轮末停止。", flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    stop_file = config.stop_file.resolve() if config.stop_file else output / "STOP"
    best_score = (-math.inf, -math.inf)
    best_epoch = None
    status = "running"
    try:
        if config.deterministic:
            # 在任何CUDA上下文创建之前配置cuBLAS，避免严格确定性下矩阵运算报错。
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace is None:
                if torch.cuda.is_initialized():
                    raise RuntimeError("CUDA已初始化但未配置cuBLAS确定性，请用新进程启动")
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            elif workspace not in {":4096:8", ":16:8"}:
                raise ValueError("严格确定性需要CUBLAS_WORKSPACE_CONFIG=:4096:8或:16:8")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device)
        if device.type not in {"cuda", "cpu"} or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("本版仅支持可用的CUDA或CPU")
        torch.use_deterministic_algorithms(config.deterministic)
        torch.backends.cudnn.benchmark = not config.deterministic
        amp_enabled = config.amp and device.type == "cuda"
        shared = {"batch_size": config.batch_size, "num_workers": config.num_workers,
                  "prefetch_factor": config.prefetch_factor, "pin_memory": device.type == "cuda",
                  "image_size": config.image_size, "seed": config.seed,
                  "persistent_workers": False, "timeout": config.worker_timeout}
        print("正在构建 DataLoader...", flush=True)
        train_loader = make_dataloader(config.train_manifest, split="train",
            mix_config=MixConfig(mix_probability=config.mix_probability,
                                 cutmix_probability=config.cutmix_probability,
                                 mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha), **shared)
        print(f"=> train DataLoader 已构建，workers={config.num_workers}；首次取批时才启动worker。", flush=True)

        def make_evaluation_loader(split: str, *, serial: bool = False):
            options = {**shared, "num_workers": 0 if serial else config.num_workers}
            print("正在构建 DataLoader...", flush=True)
            if split == "valid":
                loader = make_dataloader(config.valid_manifest, split="valid", shuffle=False, **options)
            else:
                loader = make_dataloader(config.train_manifest, split="train", shuffle=False,
                                         mix_config=MixConfig(mix_probability=0.0), **options)
                # 训练诊断只在内存中改为无增强；worker第一次启动前完成，不改缓存。
                loader.dataset.transform = build_transforms("valid", image_size=config.image_size, seed=config.seed)
            print(f"=> {split}评估 DataLoader 已构建，workers={options['num_workers']}。", flush=True)
            return loader

        valid_loader = make_evaluation_loader("valid")
        train_eval_loader = make_evaluation_loader("train")
        print("=> 正在初始化EfficientNet-B3及预训练权重...", flush=True)
        model = build_model(pretrained=True, drop_rate=config.dropout, drop_path_rate=config.drop_path).to(device)
        print("=> 模型已就绪，正在配置损失（ASL不叠加类别权重）...", flush=True)
        criterion = build_criterion(config.train_manifest, unit="image", device=device,
            loss_name=config.loss_name, pos_weight_power=config.pos_weight_power,
            max_weight=config.max_pos_weight, focal_gamma=config.focal_gamma,
            asl_gamma_pos=config.asl_gamma_pos, asl_gamma_neg=config.asl_gamma_neg,
            asl_clip=config.asl_clip)
        print(f"=> 损失={config.loss_name}，有效pos_weight={criterion.pos_weight.detach().cpu().tolist()}", flush=True)
        loss_config = {"loss_name": config.loss_name, "unit": "image",
            "class_weighting": "none" if config.loss_name == "asl" else "train_neg_pos_ratio",
            "pos_weight_power": config.pos_weight_power if config.loss_name != "asl" else None,
            "max_pos_weight": config.max_pos_weight if config.loss_name != "asl" else None,
            "focal_gamma": config.focal_gamma if config.loss_name == "focal" else None,
            "asl_gamma_pos": config.asl_gamma_pos if config.loss_name == "asl" else None,
            "asl_gamma_neg": config.asl_gamma_neg if config.loss_name == "asl" else None,
            "asl_clip": config.asl_clip if config.loss_name == "asl" else None,
            "reduction": "mean_over_images_and_classes",
            "soft_targets": "linear_positive_negative_branches",
            "effective_pos_weight": criterion.pos_weight.detach().cpu().tolist(),
            "loss_comparison": "不同损失/权重的loss不直接比较；另记固定无权重BCE"}
        _write_json(output / "loss_config.json", loss_config)
        optimizer = _build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        model_ema = None
        if config.ema:
            # 延迟构建：导入train.py不会构造模型/EMA。AMP仍使用FP32主权重。
            # BN均值/方差及整数计数器直接拷贝，避免再次平滑统计量。
            from timm.utils import ModelEmaV3
            model_ema = ModelEmaV3(model, decay=config.ema_decay,
                use_warmup=config.ema_warmup, warmup_power=0.75, exclude_buffers=True)
            model_ema.module.requires_grad_(False)
        weights_source = "ema" if model_ema is not None else "raw"

        def record_optimizer_step(_optimizer, _args, _kwargs):
            progress["optimizer_updates"] += 1
            if model_ema is not None:
                model_ema.update(model, step=progress["optimizer_updates"])
                progress["ema_updates"] += 1

        optimizer_step_hook = optimizer.register_step_post_hook(record_optimizer_step)
        print(f"=> 评估/阈值/推理权重来源={weights_source}；EMA decay={config.ema_decay}，"
              f"warmup={config.ema_warmup}，BN buffers直接拷贝。", flush=True)
        # 仅解冻时创建一次。两个参数组各以自身的初始lr做同一余弦衰减，
        # eta_min=0让骨干/分类头保持学习率比例，最后一次step到0。
        # 默认20轮、冻结2轮，T_max=18；冻结阶段不step。保留eta_min=0，
        # 优先修正过长的衰减周期，不同时试验另一调度器或学习率下界。
        scheduler = None
        early_stopping = EarlyStopping(patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta, min_epochs=config.early_stopping_min_epochs)
        print("=> 开始训练，数据加载中...", flush=True)
        for epoch in range(1, config.epochs + 1):
            if requested_stop or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
            phase = _configure_phase(model, optimizer, config, epoch)
            if phase == "full_finetune" and scheduler is None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.epochs - config.freeze_epochs, eta_min=0.0)
            epoch_learning_rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
            _append_worklog(run_id, f"执行：启动epoch={epoch}/{config.epochs}，phase={phase}，"
                f"lr={[(g['name'], g['lr']) for g in optimizer.param_groups]}；只读既有缓存，在线增强。")
            print(f"=> Epoch {epoch}/{config.epochs}：等待首批数据...", flush=True)
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                epoch=epoch, amp_enabled=amp_enabled, grad_clip=config.grad_clip, progress=progress)
            if model_ema is not None and progress["ema_updates"] == 0:
                raise RuntimeError("本轮尚无成功优化/EMA更新，不能将初始EMA权重当已训练best")
            evaluation_model = model_ema.module if model_ema is not None else model
            validation = evaluate(evaluation_model, valid_loader, criterion, device, manifest=config.valid_manifest,
                                  split="valid", amp_enabled=amp_enabled, epoch=epoch,
                                  diagnostic_path=output / "data_loading_events.jsonl")
            if validation["fallback_serial"]:
                # 已发生退出/超时的评估loader，后续轮次继续用串行，避免每轮重复崩溃。
                valid_loader = make_evaluation_loader("valid", serial=True)
            thresholds = None
            val_metrics, train_metrics = _unavailable_metrics(), _unavailable_metrics()
            train_evaluation = {"complete": False, "loss": None,
                                "reason": "本轮valid不完整，跳过依赖其阈值的train诊断"}
            if validation["complete"]:
                thresholds = search_class_thresholds(validation["targets"], validation["probabilities"])
                val_metrics = calculate_metrics(validation["targets"], validation["probabilities"], thresholds)
                train_evaluation = evaluate(evaluation_model, train_eval_loader, criterion, device, manifest=config.train_manifest,
                                            split="train", amp_enabled=amp_enabled, epoch=epoch,
                                            diagnostic_path=output / "data_loading_events.jsonl")
                if train_evaluation["fallback_serial"]:
                    train_eval_loader = make_evaluation_loader("train", serial=True)
                if train_evaluation["complete"]:
                    train_metrics = calculate_metrics(train_evaluation["targets"], train_evaluation["probabilities"], thresholds)
            else:
                print(f"epoch={epoch} 验证不完整：{validation['evaluated_images']}/"
                      f"{validation['expected_images']}张；AUC/F1=NA，不搜索阈值、不更新best；"
                      "保存本轮last后继续。具体ID见data_loading_events.jsonl。", flush=True)
            # 训练诊断使用同轮valid阈值，不再用train拟合阈值；不反传任何评估结果。
            for split, manifest in (("train", config.train_manifest), ("valid", config.valid_manifest)):
                if _sha256(manifest) != source_hashes[split]:
                    raise RuntimeError("运行期间清单改变，停止并保留证据")
            score = (val_metrics["macro_auc"], val_metrics["macro_f1"]) if validation["complete"] else None
            is_best = score is not None and score > best_score
            should_early_stop = False
            if phase == "full_finetune" and validation["complete"]:
                should_early_stop = early_stopping.update(epoch, val_metrics["macro_auc"])
            if scheduler is not None:
                # 本轮优化已经完成，无论验证是否缺图，都只step一次；不是每batch step。
                scheduler.step()
            next_learning_rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
            ema_metadata = {"enabled": config.ema, "decay": config.ema_decay,
                "use_warmup": config.ema_warmup, "warmup_power": 0.75,
                "updates": progress["ema_updates"], "buffers": "copy_from_raw"}
            record = {"epoch": epoch, "phase": phase, "train_loss": train_loss,
                      "train_eval_loss": train_evaluation["loss"], "train": train_metrics,
                      "val_loss": validation["loss"], "val": val_metrics, "thresholds": thresholds,
                      "val_bce_unweighted": validation["bce_unweighted"],
                      "train_bce_unweighted": train_evaluation.get("bce_unweighted"),
                      "loss_name": config.loss_name,
                      "loss_config": loss_config, "weights_source": weights_source,
                      "validation_weights_source": weights_source,
                      "threshold_weights_source": weights_source,
                      "ema": ema_metadata, "optimizer_updates": progress["optimizer_updates"],
                      "amp_skipped_steps": progress["amp_skipped_steps"],
                      "early_stopping": {**asdict(early_stopping), "should_stop": should_early_stop,
                                         "monitor": "complete_valid_Study_macro_auc"},
                      "metrics_unit": "Study", "f1_threshold_source": "same_epoch_valid" if thresholds is not None else None,
                      "validation_complete": validation["complete"],
                      "data_loading": {"valid": _evaluation_summary(validation),
                                       "train_eval": _evaluation_summary(train_evaluation)},
                      "is_best": is_best, "learning_rates": epoch_learning_rates,
                      "next_learning_rates": next_learning_rates,
                      "scheduler": scheduler.state_dict() if scheduler is not None else None}
            checkpoint = {"schema_version": 2, "model_name": "efficientnet_b3", "num_classes": NUM_CLASSES,
                "class_ids": list(CLASS_IDS), "epoch": epoch, "aggregation": "study_mean_probability",
                # model_state_dict就是经过本轮验证并用于阈值搜索的权重。
                # raw备份只供排查，不允许推理静默替换为raw或再次执行EMA。
                "thresholds": thresholds,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in evaluation_model.state_dict().items()},
                "weights_source": weights_source, "validation_weights_source": weights_source,
                "threshold_weights_source": weights_source, "ema": ema_metadata,
                "optimizer_updates": progress["optimizer_updates"],
                "amp_skipped_steps": progress["amp_skipped_steps"], "loss_config": loss_config,
                "pos_weight": criterion.pos_weight.detach().cpu().tolist(),
                "preprocessing": {"image_size": config.image_size, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                                  "cache_mode": "RGB", "eval_transform": "Resize+Normalize"},
                "train_manifest_sha256": source_hashes["train"], "valid_manifest_sha256": source_hashes["valid"],
                "validation": val_metrics if validation["complete"] else None,
                "validation_complete": validation["complete"], "inference_ready": validation["complete"],
                "learning_rates": epoch_learning_rates, "next_learning_rates": next_learning_rates,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "early_stopping": {**asdict(early_stopping), "should_stop": should_early_stop},
                "training_config": config_dict}
            if model_ema is not None:
                checkpoint["raw_model_state_dict"] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # last是最近完成优化轮次的模型快照，不含optimizer/RNG，不能精确续训。
            # 缺验证时显式标记未验证且thresholds=None；inference现有阈值校验会拒绝它。
            # 正式提交只用通过全量验证的best；此处绝不删除/覆盖既有best侧文件。
            _save_checkpoint(output / "last.pt", checkpoint)
            if is_best:
                digest = _save_checkpoint(output / "best.pt", checkpoint)
                _write_json(output / "thresholds.json", {"schema_version": 2, "epoch": epoch,
                    "checkpoint_sha256": digest, "checkpoint_filename": "best.pt", "class_ids": list(CLASS_IDS),
                    "weights_source": weights_source,
                    "thresholds": thresholds, "aggregation": "study_mean_probability",
                    "selection": "max Study Macro-AUC; exact tie uses threshold-tuned Macro-F1"})
                _write_json(output / "best_metrics.json", record)
                best_score, best_epoch = score, epoch
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress["completed_epochs"] = epoch
            if validation["complete"]:
                progress["validated_epochs"] += 1
            print(f"epoch={epoch} phase={phase} train_loss={train_loss:.6f} "
                  f"train_AUC={_metric_text(train_metrics['macro_auc'])} train_F1={_metric_text(train_metrics['macro_f1'])} "
                  f"val_loss={_metric_text(validation['loss'])} val_AUC={_metric_text(val_metrics['macro_auc'])} "
                  f"val_F1_tuned={_metric_text(val_metrics['macro_f1'])} best={is_best} "
                  f"val_BCE_unweighted={_metric_text(validation['bce_unweighted'])} "
                  f"weights={weights_source} ema_updates={progress['ema_updates']} "
                  f"amp_skipped={progress['amp_skipped_steps']} "
                  f"early_stop_bad={early_stopping.bad_epochs}/{early_stopping.patience} "
                  f"lr={epoch_learning_rates} next_lr={next_learning_rates}", flush=True)
            _append_worklog(run_id, f"结果：epoch={epoch}完成，train loss/AUC/F1="
                f"{train_loss:.6f}/{_metric_text(train_metrics['macro_auc'])}/{_metric_text(train_metrics['macro_f1'])}，valid="
                f"{_metric_text(validation['loss'])}/{_metric_text(val_metrics['macro_auc'])}/{_metric_text(val_metrics['macro_f1'])}；"
                f"事实：验证完整={validation['complete']}，跳过图数={len(validation['skipped_samples'])}，"
                f"串行恢复={validation['fallback_serial']}，best={is_best}，权重={weights_source}，"
                f"EMA更新={progress['ema_updates']}、AMP跳步={progress['amp_skipped_steps']}；"
                f"推测：泛化收益待独立评价。证据：{output}/history.jsonl及data_loading_events.jsonl。")
            with (PROJECT_ROOT / "EXPERIMENTS.md").open("a", encoding="utf-8") as stream:
                interpretation = "valid调优成绩，非独立测试" if validation["complete"] else "验证缺图，指标/阈值无效"
                stream.write(f"| {run_id}-E{epoch:03d} | {phase}; loss={config.loss_name}; "
                    f"pos_weight_power={loss_config['pos_weight_power']}; cap={loss_config['max_pos_weight']}; "
                    f"asl_gamma_pos={loss_config['asl_gamma_pos']}; asl_gamma_neg={loss_config['asl_gamma_neg']}; "
                    f"asl_clip={loss_config['asl_clip']}; weights={weights_source}; ema_updates={progress['ema_updates']}; "
                    f"wd={config.weight_decay}; drop={config.dropout}; cutmix_cond={config.cutmix_probability}; "
                    f"cosine; 完整验证={validation['complete']} | "
                    f"loss={train_loss:.6f};AUC={_metric_text(train_metrics['macro_auc'])};F1={_metric_text(train_metrics['macro_f1'])} | "
                    f"loss={_metric_text(validation['loss'])};AUC={_metric_text(val_metrics['macro_auc'])};F1_tuned={_metric_text(val_metrics['macro_f1'])} | "
                    f"未使用 | {interpretation} | {'保留best' if is_best else '保留日志/last，不替换best'} | {output}/history.jsonl |\n")
            epoch_state = "epoch_complete" if validation["complete"] else "epoch_complete_validation_incomplete"
            _write_json(output / "status.json", {"state": epoch_state, **progress,
                "validation_complete": validation["complete"], "best_epoch": best_epoch, "updated": _now()})
            if requested_stop or stop_file.exists():
                status = "stopped_at_epoch_boundary"
                break
            if should_early_stop:
                status = "early_stopped"
                print(f"=> 提前停止：连续{early_stopping.bad_epochs}个完整验证轮次AUC未提升超过"
                      f"{early_stopping.min_delta}；最佳权重仍为epoch={best_epoch}，本轮last已保存。", flush=True)
                _append_worklog(run_id, f"早停：epoch={epoch}，monitor=Study Macro-AUC，"
                    f"patience={early_stopping.patience}，min_delta={early_stopping.min_delta}；"
                    f"已保存last，best epoch={best_epoch}；事实：停止继续拟合，不代表独立测试已提高。")
                break
        else:
            status = "completed"
    except BaseException as exc:
        status = "failed_or_interrupted"
        _write_json(output / "status.json", {"state": status, **progress, "best_epoch": best_epoch,
                                              "error": f"{type(exc).__name__}: {exc}", "updated": _now()})
        _append_worklog(run_id, f"失败/中断：{type(exc).__name__}: {exc}；已保存{progress['completed_epochs']}轮，"
                               f"完整验证{progress['validated_epochs']}轮，实际反传批数{progress['backward_batches']}；"
                               "保留此前模型快照，不把未验证/部分轮称为完整验证训练。")
        raise
    finally:
        if optimizer_step_hook is not None:
            optimizer_step_hook.remove()
        signal.signal(signal.SIGINT, previous_handler)
    result = {"state": status, **progress, "best_epoch": best_epoch, "output": str(output), "updated": _now()}
    _write_json(output / "status.json", result)
    _append_worklog(run_id, f"交接：{status}，完成{progress['completed_epochs']}轮、完整验证{progress['validated_epochs']}轮、反传{progress['backward_batches']}批，"
                           f"best epoch={best_epoch}；证据{output}；下一步经授权用inference.py生成测试提交。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="经用户后续授权后，才可显式运行训练")
    parser.add_argument("--output", type=Path, required=True, help="runs下的新目录，不能覆盖旧run")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--worker-timeout", type=float, default=120.0, help="worker取批超时秒数，0禁用；串行自动为0")
    parser.add_argument("--head-lr", type=float, default=TrainConfig.head_lr)
    parser.add_argument("--finetune-lr", type=float, default=TrainConfig.finetune_lr)
    parser.add_argument("--finetune-head-lr", type=float, default=TrainConfig.finetune_head_lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--drop-path", type=float, default=TrainConfig.drop_path)
    parser.add_argument("--train-bn", action="store_true", help="允许解冻阶段更新BN统计；默认固定统计")
    parser.add_argument("--loss-name", choices=("bce", "focal", "asl"), default=TrainConfig.loss_name)
    parser.add_argument("--pos-weight-power", type=float, default=TrainConfig.pos_weight_power,
                        help="仅BCE/Focal使用；ASL固定不加权")
    parser.add_argument("--max-pos-weight", type=float, default=TrainConfig.max_pos_weight)
    parser.add_argument("--no-pos-weight-cap", dest="max_pos_weight", action="store_const",
                        const=None, default=argparse.SUPPRESS, help="关闭上限供原始BCE对照")
    parser.add_argument("--focal-gamma", type=float, default=TrainConfig.focal_gamma)
    parser.add_argument("--asl-gamma-pos", type=float, default=TrainConfig.asl_gamma_pos)
    parser.add_argument("--asl-gamma-neg", type=float, default=TrainConfig.asl_gamma_neg)
    parser.add_argument("--asl-clip", type=float, default=TrainConfig.asl_clip)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=TrainConfig.ema,
                        help="默认启用EMA；--no-ema可作同配置对照")
    parser.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay)
    parser.add_argument("--ema-warmup", action=argparse.BooleanOptionalAction,
                        default=TrainConfig.ema_warmup, help="默认预热EMA衰减系数")
    parser.add_argument("--mix-probability", type=float, default=TrainConfig.mix_probability)
    parser.add_argument("--cutmix-probability", type=float, default=TrainConfig.cutmix_probability,
                        help="已决定混合时选择CutMix的条件概率")
    parser.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    parser.add_argument("--cutmix-alpha", type=float, default=TrainConfig.cutmix_alpha)
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--early-stopping-min-delta", type=float, default=TrainConfig.early_stopping_min_delta)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=TrainConfig.early_stopping_min_epochs)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    options = vars(args)
    options["amp"] = not options.pop("no_amp")
    options["deterministic"] = not options.pop("allow_nondeterministic")
    options["freeze_bn"] = not options.pop("train_bn")
    print(json.dumps(run_training(TrainConfig(**options)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Windows多进程保护；本次没有执行此入口，保存文件本身不会启动训练。
    main()
