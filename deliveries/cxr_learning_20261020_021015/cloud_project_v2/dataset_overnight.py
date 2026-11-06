"""Overnight CV data: immutable manifest records, native images, complete Studies.

Importing this module neither reads images nor creates workers.  A loader must be
rebuilt for every epoch: nonpersistent spawn workers inherit that epoch's image
size and seed.  Images are opened read-only and bound to their accepted SHA256.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import random
import re
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from dataset import (
    CachedSampleError, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
    letterbox_rgb, resolve_native_image_root,
)


ALLOWED_IMAGE_SIZES = (384, 448, 512, 768)


def _transforms(training: bool, seed: int) -> Any:
    # 当前夜间训练使用这条温和增强链；尺寸调整已经在__getitem__中完成。
    # 不同于通用dataset.py，本流程没有Mixup、CutMix、噪声或随机遮挡。
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if A.__version__.split(".")[0] != "2":
        raise RuntimeError("Overnight data augmentation requires Albumentations 2.x")
    operations: list[Any] = []
    if training:
        # Labels contain no laterality, so horizontal flipping preserves them.
        # Keep the complete chest: no random crops, erasing, or strong warps.
        operations += [
            # 每次读取独立抽样：50%水平翻转、50%小幅亮度/对比度变化、
            # 30%在[-5°,5°]内旋转。它们不是每张图都固定执行的三次变换。
            # 当前十类标签不含左右侧别，翻转才保持标签含义；换任务要重新判断。
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
        ]
    operations += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD,
                               max_pixel_value=255.0), ToTensorV2()]
    # Normalize逐通道做(x/255-mean)/std；Pad产生的黑色0也参与这个变换，
    # 因此黑边归一化后一般是负数，而不是继续等于0。
    # ToTensorV2把[高,宽,通道]改成[通道,高,宽]，随后由读取函数保证float32。
    # training=False时只保留这两步，不使用上面的随机增强。
    return A.Compose(operations, seed=int(seed))


class OvernightImageDataset(Dataset):
    """Record subsets retain their original split, path, identities, and labels."""

    def __init__(self, records: Sequence[dict[str, Any]], image_root: str | Path, *,
                 training: bool, image_size: int = 768, seed: int = 43) -> None:
        # records是某一折选出的图片行，复制字典以免误改共享清单记录。
        # 同Study多张视角图分别训练，共享同一组十维标签。
        if not records or type(training) is not bool:
            raise ValueError("A nonempty record sequence and boolean training are required")
        self.records = [dict(record) for record in records]
        splits = {record.get("split") for record in self.records}
        if len(splits) != 1 or not splits.issubset({"train", "valid", "test"}):
            raise ValueError("One loader must contain exactly one original manifest split")
        self.split = str(next(iter(splits)))
        # CV验证行来自原始train目录，record['split']仍是train；是否增强由
        # training这个独立布尔值决定，不能只看原始目录名字来开启增强。
        if training and self.split != "train":
            raise ValueError("Only original training records can receive random augmentation")
        self.training = training
        self.set_image_size(image_size)
        self.raw_root = resolve_native_image_root(image_root, split=self.split)
        self.paths: list[Path] = []
        image_ids: set[str] = set()
        self.study_labels: dict[str, tuple[float, ...] | None] = {}
        self.study_view_counts: dict[str, int] = {}
        for record in self.records:
            image_id = record.get("image_id")
            if not isinstance(image_id, str) or not image_id or image_id in image_ids:
                raise ValueError("Record image IDs must be unique nonempty strings")
            image_ids.add(image_id)
            subject, study, dicom = (record.get(key) for key in
                                     ("subject_id", "study_id", "dicom_id"))
            if any(not isinstance(value, str) or not value for value in (subject, study, dicom)):
                raise ValueError("Patient, Study, and image identities must be original strings")
            if record.get("study_key") != f"{subject}/{study}":
                raise ValueError("Study key does not match patient/Study identity")
            relative = Path(record.get("relative_path", ""))
            if (relative.is_absolute() or relative.drive or ".." in relative.parts
                    or len(relative.parts) != 4
                    or relative.parts[:3] != (self.split, subject, study)
                    or relative.stem != dicom or not relative.suffix):
                raise ValueError("Original relative path does not match record identity/split")
            path = (self.raw_root / relative).resolve()
            # 同时核对患者、Study、图像ID及路径边界，防止跨划分或错图配标签。
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError("Image path escapes its original split")
            if re.fullmatch(r"[0-9a-f]{64}", str(record.get("file_sha256", ""))) is None:
                raise ValueError("Every native image requires an accepted file SHA256")
            targets = record.get("targets")
            if self.split == "test":
                if targets is not None:
                    raise ValueError("Test labels must remain unknown")
                labels = None
            else:
                if (targets is None or len(targets) != NUM_CLASSES
                        or any(value not in (0, 1) for value in targets)):
                    raise ValueError("Labeled records require original ten-class binary targets")
                labels = tuple(float(value) for value in targets)
                if self.split == "train" and str(record.get("training_eligible", "")).lower() != "true":
                    raise ValueError("Training records must remain cleaning-eligible")
            study_key = record["study_key"]
            if study_key in self.study_labels and self.study_labels[study_key] != labels:
                raise ValueError("Labels disagree within one Study")
            self.study_labels[study_key] = labels
            self.study_view_counts[study_key] = self.study_view_counts.get(study_key, 0) + 1
            self.paths.append(path)
        scale = len(self.records) / len(self.study_labels)
        # 视角权重解决“多拍几张就重复贡献更多损失”：若N张图、S个Study，
        # 某Study有k张，则每张w=(N/S)/k。该Study所有视角的权重和均为N/S，
        # 全部图片的权重平均为1；这与按类别计算的pos_weight用途不同。
        # 例：6张图来自3个Study，各有1、2、3张，则每图权重依次为2、1、2/3。
        self.loss_weights = [scale / self.study_view_counts[record["study_key"]]
                             for record in self.records]
        self.transform = _transforms(training, seed)

    def set_image_size(self, image_size: int) -> None:
        """Set before building a NEW loader; existing worker copies cannot be changed."""
        # 这里仅设置目标尺寸；真正等比缩放发生在每张原图读取时。
        # worker是独立进程，已运行worker内的Dataset不会自动同步主进程改动。
        if type(image_size) is not int or image_size not in ALLOWED_IMAGE_SIZES:
            raise ValueError(f"image_size must be one of {ALLOWED_IMAGE_SIZES}")
        self.image_size = image_size

    def reseed(self, seed: int) -> None:
        # Albumentations自己管理随机源，所以必须显式把worker种子交给它。
        self.transform.set_random_seed(int(seed))

    def __len__(self) -> int:
        # 长度按图片而非Study计算；完整保留同次检查的所有视角。
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # 单张图的读取链：原文件字节 -> SHA校验 -> EXIF摆正 -> RGB ->
        # 等比缩放+黑边 -> 随机增强(仅训练) -> Normalize -> CPU张量。
        record = self.records[index]
        try:
            path = self.paths[index].resolve()
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError("Image path changed outside its original split")
            source_bytes = path.read_bytes()
            # 直接读原始分辨率，不把300像素缓存放大；哈希只用来核对数据版本。
            if hashlib.sha256(source_bytes).hexdigest() != record["file_sha256"]:
                raise ValueError("Image SHA256 differs from the accepted manifest")
            with Image.open(BytesIO(source_bytes)) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                # RGB符合ImageNet预训练模型的三通道接口；原图若为灰度，
                # convert会复制灰度到三个通道，并没有创造额外颜色信息。
                if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
                    raise ValueError("Non-8-bit source requires an explicit intensity policy")
                canvas = letterbox_rgb(upright.convert("RGB"), self.image_size)
                # 以768为例：1200x900图统一缩成768x576，上下各补96黑边。
                # letterbox保留长宽比和原图边界；旋转增强只在这之后发生。
            pixels = np.array(canvas, dtype=np.uint8, copy=True)
            tensor = self.transform(image=pixels)["image"].to(dtype=torch.float32)
            # 输出[3,768,768]，还不是[B,3,768,768]；B这个批次轴由collate添加。
            if tensor.device.type != "cpu" or tuple(tensor.shape) != (3, self.image_size, self.image_size):
                raise ValueError("Dataset must return a CPU 3xHxW image")
        except Exception as exc:
            raise CachedSampleError(f"image_id={record['image_id']}; {type(exc).__name__}: {exc}") from exc
        targets = record["targets"]
        return {"image": tensor, "targets": None if targets is None else
                torch.tensor(targets, dtype=torch.float32),
                "image_id": record["image_id"], "study_key": record["study_key"],
                "subject_id": record["subject_id"], "study_id": record["study_id"],
                "split": self.split, "loss_weight": self.loss_weights[index]}


def collate_overnight(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # 只堆叠，不混合图片/标签：微批2张形成images[2,3,768,768]与targets[2,10]。
    # loss_weights是逐图/视角权重，ID列表供验证时把概率还原到对应Study。
    if not samples or len({sample["split"] for sample in samples}) != 1:
        raise ValueError("Batch is empty or mixes original splits")
    present = [sample["targets"] is not None for sample in samples]
    if any(present) and not all(present):
        raise ValueError("Batch mixes known and unknown labels")
    return {"images": torch.stack([sample["image"] for sample in samples]),
            "targets": torch.stack([sample["targets"] for sample in samples]) if all(present) else None,
            "image_ids": [sample["image_id"] for sample in samples],
            "study_keys": [sample["study_key"] for sample in samples],
            "subject_ids": [sample["subject_id"] for sample in samples],
            "study_ids": [sample["study_id"] for sample in samples],
            "loss_weights": torch.tensor([sample["loss_weight"] for sample in samples], dtype=torch.float32),
            "split": samples[0]["split"]}


def _seed_worker(worker_id: int) -> None:
    # 每个CPU worker准备独立随机序列；同时限制线程，避免8个worker各自再
    # 开一套多线程池而争抢16个CPU核心。这里只处理CPU，不创建CUDA上下文。
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    info = get_worker_info()
    if info is not None:
        info.dataset.reseed(seed)


def make_overnight_loader(records: Sequence[dict[str, Any]], image_root: str | Path, *,
                          training: bool, image_size: int, batch_size: int,
                          num_workers: int = 8, prefetch_factor: int = 2,
                          seed: int = 43, pin_memory: bool = True) -> DataLoader:
    """Create once per epoch, leaving CPU cores for the main process and GPU work."""
    # worker并行准备下一批，pin_memory帮助CPU到GPU传输；GPU计算仍在主进程。
    # 8个worker、prefetch_factor=2表示可预取约16个批次，并非16次梯度累积。
    # 梯度累积由训练循环控制，与DataLoader的预取队列没有直接关系。
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or not 0 <= num_workers <= 12:
        raise ValueError("This sixteen-CPU server requires 0 <= num_workers <= 12")
    if type(prefetch_factor) is not int or prefetch_factor not in (1, 2):
        raise ValueError("prefetch_factor must be 1 or 2 to bound host RAM")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    dataset = OvernightImageDataset(records, image_root, training=training,
                                    image_size=image_size, seed=seed)
    options: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory,
                               "generator": torch.Generator().manual_seed(seed),
                               "worker_init_fn": _seed_worker}
    if num_workers:
        # spawn启动独立解释器，persistent_workers=False让每轮重新建立worker，
        # 从而使用当轮分辨率和随机种子；本函数定义被导入时不会自行启动它们。
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False,
                       multiprocessing_context="spawn", timeout=180.0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      drop_last=False, collate_fn=collate_overnight, **options)


def aggregate_study_predictions(records: Sequence[dict[str, Any]], image_ids: Sequence[str],
                                probabilities: Any) -> dict[str, Any]:
    """Strict full-coverage image mean, in first-seen manifest Study order."""
    # 调用者先对模型十个logit逐类sigmoid；本函数只接收[N,10]概率。
    # 不是平均特征或logit，也不是先对每张图作0/1判决再投票。
    expected = {record["image_id"]: record for record in records}
    if not expected or len(expected) != len(records):
        raise ValueError("Manifest image IDs are empty or duplicated")
    image_ids = [str(value) for value in image_ids]
    if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(expected):
        # 丢一张视角、重复一张视角都会改变均值，必须完整且恰好覆盖一次。
        raise ValueError("Predictions must cover every manifest image exactly once")
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != (len(image_ids), NUM_CLASSES) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError("Predictions must be finite Nx10 probabilities in [0,1]")
    studies: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["study_key"]
        labels = None if record["targets"] is None else tuple(record["targets"])
        if key not in studies:
            studies[key] = {"subject_id": record["subject_id"], "study_id": record["study_id"],
                            "targets": labels, "sum": np.zeros(NUM_CLASSES), "count": 0}
        elif (studies[key]["targets"] != labels
              or studies[key]["subject_id"] != record["subject_id"]
              or studies[key]["study_id"] != record["study_id"]):
            raise ValueError("Study identities or labels disagree")
    for image_id, value in zip(image_ids, values):
        item = studies[expected[image_id]["study_key"]]
        item["sum"] += value
        item["count"] += 1
    # 某Study两张图的某类概率0.2与0.8 -> Study概率0.5；另一Study只有
    # 一张图也只输出一行，所以后续Study级指标不会让多视角Study占更多行。
    labels_present = [item["targets"] is not None for item in studies.values()]
    if any(labels_present) and not all(labels_present):
        raise ValueError("Manifest mixes known and unknown Study labels")
    return {"study_keys": list(studies),
            "subject_ids": [item["subject_id"] for item in studies.values()],
            "study_ids": [item["study_id"] for item in studies.values()],
            "probabilities": np.stack([item["sum"] / item["count"] for item in studies.values()]),
            "targets": np.asarray([item["targets"] for item in studies.values()], dtype=np.float32)
            if all(labels_present) else None}
