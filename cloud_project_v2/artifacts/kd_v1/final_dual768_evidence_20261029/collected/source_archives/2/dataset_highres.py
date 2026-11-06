"""从已验收清单读取原图，在DataLoader内存中生成384输入与完整Study-bag。

不重写CSV、原图或images300，不遍历原图目录重新纳入排除项。
导入仅定义接口；只有显式创建Dataset才读清单，取样时才读图。
"""
from __future__ import annotations

from runtime_compat import prepare_runtime

prepare_runtime()

import hashlib
from io import BytesIO
import math
from pathlib import Path
import random
import re
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from dataset import (
    CachedSampleError, DEFAULT_TEST_MANIFEST, DEFAULT_TRAIN_MANIFEST,
    DEFAULT_VALID_MANIFEST, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
    PROJECT_ROOT, read_manifest,
)


RAW_ROOT = PROJECT_ROOT / "cxr_first" / "cxr_first"
IMAGE_SIZE = 384


def _checked_split(split: str) -> str:
    split = "valid" if split == "val" else split
    if split not in {"train", "valid", "test"}:
        raise ValueError("split只能为train、valid/val或test")
    return split


def resolve_image_root(image_root: str | Path, *, split: str = "train") -> Path:
    """兼容直接数据根与整层cxr_first上传；只探测两个明确目录，不扫描图片。"""
    split = _checked_split(split)
    root = Path(image_root).resolve()
    # 直接布局优先，避免两种布局同时存在时静默改用内层数据。
    if (root / split).is_dir():
        return root
    nested = root / "cxr_first"
    if (nested / split).is_dir():
        return nested.resolve()
    raise FileNotFoundError(
        f"找不到原图数据根：{root}；应存在{root / split}或{nested / split}。"
        "请将--image-root/--raw-root指向包含划分目录的数据根或其外层cxr_first目录。"
    )


def _build_transforms(*, training: bool, seed: int) -> Any:
    """只对训练启用原有强度的温和增强，不裁掉胸廓、不做批次混合。"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("高分辨率读取需要albumentations 2.x；本模块不安装依赖") from exc
    if A.__version__.split(".")[0] != "2":
        raise RuntimeError("高分辨率增强使用Albumentations 2.x接口")
    transforms: list[Any] = []
    if training:
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            A.GaussNoise(std_range=(0.005, 0.025), mean_range=(0.0, 0.0),
                         per_channel=False, p=0.2),
            A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
        ])
    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return A.Compose(transforms, seed=seed)


class HighResImageDataset(Dataset):
    """原清单逐图接口：只读原文件，逐次核对SHA，所有像素变换只在内存。"""
    def __init__(self, manifest_path: str | Path, *, split: str,
                 image_size: int = IMAGE_SIZE, seed: int = 43,
                 raw_root: str | Path = RAW_ROOT, training: bool = False) -> None:
        self.split = _checked_split(split)
        if type(image_size) is not int or image_size != IMAGE_SIZE:
            raise ValueError("本轮高分辨率版本固定384，其他尺寸须另做获批实验")
        if type(training) is not bool or (training and self.split != "train"):
            raise ValueError("随机增强只允许显式用于train")
        self.image_size = image_size
        self.training = training
        self.manifest_path = Path(manifest_path).resolve()
        # 标签、清洗资格、Study一致性和固定split全部复用原校验，不重解析CSV。
        self.records = read_manifest(self.manifest_path, split=self.split)
        self.raw_root = resolve_image_root(raw_root, split=self.split)
        self.paths: list[Path] = []
        for record in self.records:
            relative_text = record.get("relative_path")
            if not isinstance(relative_text, str) or not relative_text:
                raise ValueError("清单缺少原图relative_path，禁止扫描目录猜测路径")
            relative = Path(relative_text)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("原图relative_path必须为不含上级跳转的相对路径")
            subject, study, dicom = (record.get(name) for name in ("subject_id", "study_id", "dicom_id"))
            if any(not isinstance(value, str) or not value for value in (subject, study, dicom)):
                raise ValueError("清单缺少原始患者/Study/图像字符串身份")
            parts = relative.parts
            if (len(parts) != 4 or parts[:3] != (self.split, subject, study)
                    or relative.stem != dicom or not relative.suffix
                    or record["study_key"] != f"{subject}/{study}"):
                raise ValueError("原图路径与清单split/患者/Study/图像身份不一致")
            path = (self.raw_root / relative).resolve()
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError("原图解析路径越过允许的根目录或split，禁止链接逃逸")
            digest = record.get("file_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("每张原图必须绑定清单中的有效file_sha256")
            self.paths.append(path)
        self.transform = _build_transforms(training=training, seed=seed)

    def reseed(self, seed: int) -> None:
        self.transform.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            # 每次取图再resolve，避免构造Dataset后链接发生变化；SHA绑定实际读取字节。
            path = self.paths[index].resolve()
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError("原图读取路径越过允许的split")
            source_bytes = path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != record["file_sha256"]:
                raise ValueError("原图SHA与已验收清单不一致，禁止退回缓存或重建标签")
            with Image.open(BytesIO(source_bytes)) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
                    raise ValueError("非8-bit原图需要单独强度映射策略，禁止直接RGB截断")
                rgb = upright.convert("RGB")
                scale = self.image_size / max(rgb.size)
                size = (max(1, min(self.image_size, round(rgb.width * scale))),
                        max(1, min(self.image_size, round(rgb.height * scale))))
                resized = rgb.resize(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            canvas.paste(resized, ((self.image_size - size[0]) // 2,
                                  (self.image_size - size[1]) // 2))
            pixels = np.array(canvas, dtype=np.uint8, copy=True)
            tensor = self.transform(image=pixels)["image"].to(dtype=torch.float32)
            if tensor.device.type != "cpu" or tuple(tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError("Dataset必须返回CPU上的[3,384,384]图像")
        except Exception as exc:
            raise CachedSampleError(
                f"image_id={record['image_id']}; raw={self.paths[index].name}; "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        targets = (None if record["targets"] is None
                   else torch.tensor(record["targets"], dtype=torch.float32))
        return {"image": tensor, "targets": targets, "image_id": record["image_id"],
                "study_key": record["study_key"], "split": self.split}


class HighResStudyDataset(Dataset):
    """仅train的完整Study-bag；每次取样保留该Study清单中的全部视角。"""
    def __init__(self, manifest_path: str | Path, *, split: str,
                 image_size: int = IMAGE_SIZE, seed: int = 43,
                 raw_root: str | Path = RAW_ROOT, training: bool = False) -> None:
        if _checked_split(split) != "train":
            raise ValueError("Study-bag训练接口只接收train，valid/test必须逐图完整评估")
        self.image_dataset = HighResImageDataset(
            manifest_path, split="train", image_size=image_size, seed=seed,
            raw_root=raw_root, training=training)
        self.records = self.image_dataset.records
        self.split = "train"
        self.training = training
        self.image_size = image_size
        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            grouped.setdefault(record["study_key"], []).append(index)
        self.study_keys = list(grouped)
        self.groups = list(grouped.values())
        self.study_targets = np.asarray(
            [self.records[indices[0]]["targets"] for indices in self.groups], dtype=np.float32)

    def reseed(self, seed: int) -> None:
        self.image_dataset.reseed(seed)

    def __len__(self) -> int:
        return len(self.study_keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        samples = [self.image_dataset[image_index] for image_index in self.groups[index]]
        return {"images": torch.stack([sample["image"] for sample in samples]),
                "targets": torch.tensor(self.study_targets[index], dtype=torch.float32),
                "image_ids": [sample["image_id"] for sample in samples],
                "study_key": self.study_keys[index], "split": "train"}


def collate_images(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """无混合逐图collate，允许train完整诊断；test未知标签仍保持None。"""
    if not samples or len({sample["split"] for sample in samples}) != 1:
        raise ValueError("逐图批次为空或混入多个split")
    present = [sample["targets"] is not None for sample in samples]
    if any(present) and not all(present):
        raise ValueError("逐图批次不能混合未知与已知标签")
    return {"images": torch.stack([sample["image"] for sample in samples]),
            "targets": torch.stack([sample["targets"] for sample in samples]) if all(present) else None,
            "image_ids": [sample["image_id"] for sample in samples],
            "study_keys": [sample["study_key"] for sample in samples],
            "split": samples[0]["split"], "is_mixed": False,
            "mix_method": "none", "mix_lambda": 1.0,
            "mix_partner_indices": torch.arange(len(samples), dtype=torch.long)}


def collate_studies(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """将完整bag展平并保留视角到Study映射，不padding、不删视角、不混标签。"""
    if not samples or any(sample["split"] != "train" for sample in samples):
        raise ValueError("Study训练批次必须是非空train样本")
    if len({sample["study_key"] for sample in samples}) != len(samples):
        raise ValueError("同一训练批次不得重复同一个Study")
    counts = []
    for sample in samples:
        images, targets = sample["images"], sample["targets"]
        if (not isinstance(images, torch.Tensor) or images.device.type != "cpu"
                or images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE)
                or images.shape[0] < 1 or images.shape[0] != len(sample["image_ids"])):
            raise ValueError("Study-bag必须是含全部视角的CPU[V,3,384,384]张量")
        if (not isinstance(targets, torch.Tensor) or targets.device.type != "cpu"
                or tuple(targets.shape) != (NUM_CLASSES,)
                or not bool(torch.all((targets == 0) | (targets == 1)))):
            raise ValueError("Study标签必须保持原始CPU十维0/1多热")
        counts.append(images.shape[0])
    view_counts = torch.tensor(counts, dtype=torch.long)
    image_ids = [image_id for sample in samples for image_id in sample["image_ids"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Study训练批次包含重复图像身份")
    return {"images": torch.cat([sample["images"] for sample in samples], dim=0),
            "view_study_indices": torch.repeat_interleave(
                torch.arange(len(samples), dtype=torch.long), view_counts),
            "view_counts": view_counts,
            "targets": torch.stack([sample["targets"] for sample in samples]),
            "study_keys": [sample["study_key"] for sample in samples],
            "image_ids": image_ids, "split": "train", "is_mixed": False}


class ClassAwareStudyBatchSampler(Sampler[list[int]]):
    """只依赖train Study标签的可重放采样；每批为偶数个不同Study。

    均衡分支逐对从尚有正负候选的类别中均匀选类，再各取一正一负。
    每对最多检查十类；剩余候选无法组成任何正负对时，显式均匀回退。
    其余批从全部Study均匀无放回抽取。完整标签不变，跨批允许重复。
    """
    def __init__(self, study_targets: np.ndarray | torch.Tensor, *, batch_size: int = 2,
                 seed: int = 43, balanced_probability: float = 0.5,
                 batches_per_epoch: int | None = None) -> None:
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("Study batch_size必须为至少2的偶数")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("sampler seed必须是[0,2**32)内的整数")
        if not math.isfinite(balanced_probability) or not 0 <= balanced_probability <= 1:
            raise ValueError("balanced_probability必须在[0,1]")
        if isinstance(study_targets, torch.Tensor):
            if study_targets.device.type != "cpu":
                raise ValueError("sampler只接受CPU train Study标签")
            study_targets = study_targets.detach().numpy()
        targets = np.asarray(study_targets)
        if (targets.ndim != 2 or targets.shape[1] != NUM_CLASSES or targets.shape[0] < 2
                or not np.isin(targets, (0, 1)).all()):
            raise ValueError("sampler需要至少两个Study的[N,10]原始0/1标签")
        self.study_targets = targets.astype(np.int64, copy=True)
        self.num_studies = targets.shape[0]
        if self.num_studies < batch_size:
            raise ValueError("Study总数必须至少等于batch_size，批内不能重复Study凑数")
        self.batch_size = batch_size
        self.seed = seed
        self.balanced_probability = float(balanced_probability)
        if batches_per_epoch is not None and (type(batches_per_epoch) is not int or batches_per_epoch < 1):
            raise ValueError("batches_per_epoch必须为正整数或None")
        self.batches_per_epoch = (math.ceil(self.num_studies / self.batch_size)
                                  if batches_per_epoch is None else batches_per_epoch)
        self.positives = [np.flatnonzero(self.study_targets[:, c] == 1) for c in range(NUM_CLASSES)]
        self.negatives = [np.flatnonzero(self.study_targets[:, c] == 0) for c in range(NUM_CLASSES)]
        if self.balanced_probability > 0 and any(
            len(positive) == 0 or len(negative) == 0
            for positive, negative in zip(self.positives, self.negatives)
        ):
            raise ValueError("类均衡分支要求train的十类均存在阳性和阴性Study，不静默跳类")
        self.epoch = 0
        self._draw_counts = np.zeros(self.num_studies, dtype=np.int64)
        self._selected_classes = np.zeros(NUM_CLASSES, dtype=np.int64)
        self._balanced_batches = 0
        self._uniform_batches = 0
        self._balanced_pairs = 0
        self._fallback_pairs = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch必须是非负整数")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self._draw_counts.fill(0)
        self._selected_classes.fill(0)
        self._balanced_batches = self._uniform_batches = 0
        self._balanced_pairs = self._fallback_pairs = 0
        for _ in range(len(self)):
            if rng.random() < self.balanced_probability:
                # 随机类别排列的首个可行类，在当下可行类别中仍为均匀选择。
                # 每对最多检查NUM_CLASSES次，无重试死循环；整批共享可用掩码。
                available = np.ones(self.num_studies, dtype=bool)
                selected: list[int] = []
                for _pair in range(self.batch_size // 2):
                    pair = None
                    for class_index_value in rng.permutation(NUM_CLASSES):
                        class_index = int(class_index_value)
                        positives = self.positives[class_index]
                        negatives = self.negatives[class_index]
                        positive_candidates = positives[available[positives]]
                        negative_candidates = negatives[available[negatives]]
                        if not len(positive_candidates) or not len(negative_candidates):
                            continue
                        pair = np.asarray([rng.choice(positive_candidates),
                                           rng.choice(negative_candidates)], dtype=np.int64)
                        self._selected_classes[class_index] += 1
                        self._balanced_pairs += 1
                        break
                    if pair is None:
                        # 此时剩余Study在十个标签上完全一致，任何类别都无法配对。
                        # N>=batch_size保证仍有至少两个未用Study，回退也不重复。
                        pair = rng.choice(np.flatnonzero(available), size=2, replace=False)
                        self._fallback_pairs += 1
                    available[pair] = False
                    selected.extend(int(index) for index in pair)
                indices = np.asarray(selected, dtype=np.int64)
                self._balanced_batches += 1
            else:
                indices = rng.choice(self.num_studies, size=self.batch_size, replace=False)
                self._uniform_batches += 1
            rng.shuffle(indices)
            self._draw_counts[indices] += 1
            yield [int(index) for index in indices]

    def summary(self, *, include_study_counts: bool = False) -> dict[str, Any]:
        """报告已yield的采样曝光；worker预取可能领先优化，不当成成功反传次数。"""
        draws = int(self._draw_counts.sum())
        squared = float(np.square(self._draw_counts.astype(np.float64)).sum())
        positive_exposure = self._draw_counts @ self.study_targets
        result = {"epoch": self.epoch, "seed": self.seed, "num_studies": self.num_studies,
                  "batch_size": self.batch_size,
                  "planned_batches": len(self), "planned_study_draws": self.batch_size * len(self),
                  "batches_yielded": self._balanced_batches + self._uniform_batches,
                  "study_draws_yielded": draws,
                  "balanced_probability": self.balanced_probability,
                  "balanced_batches": self._balanced_batches, "uniform_batches": self._uniform_batches,
                  "positive_negative_pairs": self._balanced_pairs,
                  "uniform_fallback_pairs": self._fallback_pairs,
                  "pairs_per_balanced_batch": self.batch_size // 2,
                  "class_selection_unit": "feasible_positive_negative_pair",
                  "selected_class_counts": self._selected_classes.tolist(),
                  "per_class_positive_exposure": positive_exposure.tolist(),
                  "per_class_negative_exposure": (draws - positive_exposure).tolist(),
                  "unique_studies_sampled": int(np.count_nonzero(self._draw_counts)),
                  "max_study_repeats": int(self._draw_counts.max()),
                  "effective_sample_size": draws * draws / squared if squared else 0.0,
                  "coverage": "sampled_with_replacement_across_batches_not_full_manifest_coverage"}
        if include_study_counts:
            result["per_study_draw_counts"] = self._draw_counts.tolist()
        return result

    def report(self, *, include_study_counts: bool = False) -> dict[str, Any]:
        return self.summary(include_study_counts=include_study_counts)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    if info is not None:
        info.dataset.reseed(seed)


def _loader_options(*, num_workers: int, prefetch_factor: int, pin_memory: bool,
                    timeout: float, seed: int) -> dict[str, Any]:
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers必须为非负整数")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor必须为正整数")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout必须为有限非负数")
    options = {"num_workers": num_workers, "pin_memory": pin_memory,
               "timeout": float(timeout) if num_workers else 0,
               "generator": torch.Generator().manual_seed(seed), "worker_init_fn": _seed_worker}
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False,
                       multiprocessing_context="spawn")
    return options


def make_image_loader(manifest_path: str | Path | None = None, *, split: str,
                      batch_size: int = 1, num_workers: int = 0,
                      prefetch_factor: int = 1, training: bool = False,
                      timeout: float = 120, image_size: int = IMAGE_SIZE,
                      seed: int = 43, raw_root: str | Path = RAW_ROOT,
                      pin_memory: bool = True) -> DataLoader:
    """逐图顺序loader；train诊断同样全覆盖，valid/test不随机增强。"""
    split = _checked_split(split)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("逐图batch_size必须为正整数")
    defaults = {"train": DEFAULT_TRAIN_MANIFEST, "valid": DEFAULT_VALID_MANIFEST,
                "test": DEFAULT_TEST_MANIFEST}
    dataset = HighResImageDataset(manifest_path or defaults[split], split=split,
        image_size=image_size, seed=seed, raw_root=raw_root, training=training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_images, **_loader_options(num_workers=num_workers,
            prefetch_factor=prefetch_factor, pin_memory=pin_memory, timeout=timeout, seed=seed))


def make_study_loader(manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST, *,
                      split: str = "train", batch_size: int = 2, num_workers: int = 0,
                      prefetch_factor: int = 1, balanced_probability: float = 0.5,
                      batches_per_epoch: int | None = None, training: bool = True,
                      timeout: float = 120, image_size: int = IMAGE_SIZE,
                      seed: int = 43, raw_root: str | Path = RAW_ROOT,
                      pin_memory: bool = True) -> DataLoader:
    """仅train的完整Study loader；batch_sampler可set_epoch，批大小须为偶数。"""
    dataset = HighResStudyDataset(manifest_path, split=split, image_size=image_size,
        seed=seed, raw_root=raw_root, training=training)
    sampler = ClassAwareStudyBatchSampler(dataset.study_targets, batch_size=batch_size,
        seed=seed, balanced_probability=balanced_probability, batches_per_epoch=batches_per_epoch)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_studies,
        **_loader_options(num_workers=num_workers, prefetch_factor=prefetch_factor,
                          pin_memory=pin_memory, timeout=timeout, seed=seed))
