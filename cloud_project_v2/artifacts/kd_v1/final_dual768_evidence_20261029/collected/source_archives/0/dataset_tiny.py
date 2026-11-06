"""ConvNeXt-Tiny逐图训练接口：固定清洗清单、每轮完整随机遍历。

原图身份、SHA256、384保比例缩放及只读约束复用dataset_highres，
不修改旧数据接口，也不扫描图片目录补回清洗时已排除的样本。
所有增强只在DataLoader内存中发生；导入本模块不读取清单或图片。
loss_weight=N/(S*Study保留图数)，使整轮按图片求均值的加权BCE
恰好等于先平均每个Study视图BCE、再对所有Study求均值。
"""
from __future__ import annotations

from runtime_compat import prepare_runtime

prepare_runtime()

from collections import Counter
import math
from pathlib import Path
import random
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from dataset import (
    CLASS_IDS, DEFAULT_TRAIN_MANIFEST, IMAGENET_MEAN, IMAGENET_STD,
    NUM_CLASSES, read_manifest,
)
from dataset_highres import (
    HighResImageDataset, IMAGE_SIZE, RAW_ROOT, collate_images,
)


def _train_record_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """按完整train清单统计Study，不能把多视角重复算成阳性Study。"""
    study_labels: dict[str, tuple[float, ...]] = {}
    view_counts: Counter[str] = Counter()
    for record in records:
        if record["split"] != "train" or record["targets"] is None:
            raise ValueError("类别统计和视图权重只能来自已验收train清单")
        key = record["study_key"]
        targets = tuple(record["targets"])
        if (len(targets) != NUM_CLASSES
                or any(value not in (0, 1) for value in targets)):
            raise ValueError("train Study标签必须为十维原始0/1多热")
        old = study_labels.setdefault(key, targets)
        if old != targets:
            raise ValueError("同Study标签冲突，禁止合并或重写")
        view_counts[key] += 1
    if not study_labels:
        raise ValueError("训练清单为空")
    study_count = len(study_labels)
    positives = [sum(int(labels[c]) for labels in study_labels.values())
                 for c in range(NUM_CLASSES)]
    return {
        "source_split": "train",
        "num_images": len(records),
        "num_studies": study_count,
        "class_ids": list(CLASS_IDS),
        "positive_studies": positives,
        "negative_studies": [study_count - value for value in positives],
        "positive_prevalence": [value / study_count for value in positives],
        "study_view_counts": dict(view_counts),
        "study_loss_weight_scale": len(records) / study_count,
        "loss_weight_definition": "num_images / (num_studies * retained_study_views)",
    }


def train_study_statistics(manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST) -> dict[str, Any]:
    """仅从train计数：每Study只计一次标签，供显式可选pos_weight使用。

    本函数不读取原图，不查看valid/test，也不在模块导入时执行。
    """
    return _train_record_statistics(read_manifest(manifest_path, split="train"))


def _build_tiny_train_transforms(seed: int) -> Any:
    """温和在线亮度/对比与旋转，不加噪声、不裁切、不混合标签。"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("Tiny训练增强需要albumentations 2.x；本模块不安装依赖") from exc
    if A.__version__.split(".")[0] != "2":
        raise RuntimeError("Tiny训练增强使用Albumentations 2.x接口")
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ], seed=seed)


class TinyTrainDataset(HighResImageDataset):
    """逐图继承已验收原图读取，并以逆视图数避免多视角Study支配损失。"""
    def __init__(self, manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST, *,
                 seed: int = 43, raw_root: str | Path = RAW_ROOT) -> None:
        # 先复用原接口的全部路径/标签/身份校验；然后只替换内存增强。
        super().__init__(manifest_path, split="train", image_size=IMAGE_SIZE,
                         seed=seed, raw_root=raw_root, training=False)
        self.training = True
        self.transform = _build_tiny_train_transforms(seed)
        self.study_statistics = _train_record_statistics(self.records)
        self.study_view_counts = self.study_statistics["study_view_counts"]
        scale = self.study_statistics["study_loss_weight_scale"]
        # 全局乘N/S后，全清单平均权重恰为1，每个Study的总权重恰为N/S。
        # 训练端用batch均值，不按每批sum(weight)归一，保持完整目标的无偏估计。
        self.loss_weights = [scale / self.study_view_counts[record["study_key"]]
                             for record in self.records]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["loss_weight"] = self.loss_weights[index]
        sample["study_num_views"] = self.study_view_counts[sample["study_key"]]
        return sample


def collate_tiny_train(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """普通逐图批次加全局归一的逆Study视图权重，绝不混合标签。"""
    if not samples or any(sample["split"] != "train" for sample in samples):
        raise ValueError("Tiny训练批次必须是非空train样本")
    if len({sample["image_id"] for sample in samples}) != len(samples):
        raise ValueError("Tiny训练批次不得重复同一图像")
    if any(sample["targets"] is None for sample in samples):
        raise ValueError("Tiny训练批次不得包含未知标签")
    weights = [sample["loss_weight"] for sample in samples]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Study视图损失权重必须为有限正数")
    batch = collate_images(samples)
    batch["loss_weights"] = torch.tensor(weights, dtype=torch.float32)
    batch["study_num_views"] = torch.tensor(
        [sample["study_num_views"] for sample in samples], dtype=torch.long)
    return batch


def _seed_tiny_worker(worker_id: int) -> None:
    """worker仅CPU增强与解码；不查询/初始化CUDA，不创建任何模型。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    info = get_worker_info()
    if info is not None:
        info.dataset.reseed(worker_seed)


def make_tiny_train_loader(manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST, *,
                           batch_size: int = 32, num_workers: int = 12,
                           prefetch_factor: int = 2, timeout: float = 120,
                           seed: int = 43, raw_root: str | Path = RAW_ROOT,
                           pin_memory: bool = True) -> DataLoader:
    """shuffle=True、drop_last=False，每轮完整无放回覆盖清洗后全部图像。

    worker使用spawn与持久进程；num_workers=0时不传prefetch/context，
    允许串行定位加载问题。Dataset不含任何验证/测试统计或采样权重。
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("逐图batch_size必须为正整数")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers必须为非负整数")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor必须为正整数")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout必须为有限非负数")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed必须是[0,2**32)内的整数")
    if type(pin_memory) is not bool:
        raise ValueError("pin_memory必须为bool")
    dataset = TinyTrainDataset(manifest_path, seed=seed, raw_root=raw_root)
    options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "timeout": float(timeout) if num_workers else 0,
        "generator": torch.Generator().manual_seed(seed),
        "worker_init_fn": _seed_tiny_worker,
    }
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=True,
                       multiprocessing_context="spawn")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      collate_fn=collate_tiny_train, **options)
