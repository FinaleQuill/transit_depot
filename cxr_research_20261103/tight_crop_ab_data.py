"""Isolated image loaders for the registered tight-crop A/B study training."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CachedSampleError, IMAGENET_MEAN, IMAGENET_STD
from dataset_overnight import OvernightImageDataset, _seed_worker, collate_overnight
from study_student import complete_study_groups
from tight_crop_ab_preprocess import load_canvas


def _augmentation(seed: int):
    import albumentations as A
    if A.__version__.split(".")[0] != "2":
        raise RuntimeError("A/B training requires Albumentations 2.x")
    geometry = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
    ], bbox_params=A.BboxParams(format="pascal_voc", clip=True,
                               filter_invalid_bboxes=True), seed=int(seed))
    from albumentations.pytorch import ToTensorV2
    normalization = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return geometry, normalization


def random_erasing_ab_u8(image: np.ndarray, content_box, *, training=False,
                         rng=None, p=0.5, area_range=(0.005, 0.02)):
    """Erase one train-only patch; half of triggered centers favor the content edge."""
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Erasing requires uint8 image pixels")
    if not 0 <= p <= 1 or not 0 < area_range[0] <= area_range[1] <= 0.02:
        raise ValueError("Invalid A/B erasing probability or patch area")
    out = image.copy()
    record = {"applied": False, "probability": p, "training": bool(training),
              "selection": None, "box": None, "edge_band_fraction": 0.2}
    if not training:
        return out, record
    if rng is None:
        raise ValueError("Train erasing requires an independent worker RNG")
    if rng.random() >= p:
        return out, record
    x0, y0, x1, y1 = map(int, content_box)
    height, width = image.shape[:2]
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Content box lies outside the image")
    cw, ch = x1 - x0, y1 - y0
    area = rng.uniform(*area_range) * cw * ch
    aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
    ew = min(cw, max(1, math.floor(math.sqrt(area * aspect))))
    eh = min(ch, max(1, math.floor(math.sqrt(area / aspect))))
    edge = bool(rng.random() < 0.5)
    ranges = [
        ("left", x0, min(x1 - ew, math.floor(x0 + 0.2 * cw - ew / 2)),
         y0, y1 - eh),
        ("right", max(x0, math.ceil(x1 - 0.2 * cw - ew / 2)), x1 - ew,
         y0, y1 - eh),
        ("top", x0, x1 - ew, y0,
         min(y1 - eh, math.floor(y0 + 0.2 * ch - eh / 2))),
        ("bottom", x0, x1 - ew,
         max(y0, math.ceil(y1 - 0.2 * ch - eh / 2)), y1 - eh),
    ]
    eligible = [entry for entry in ranges if entry[1] <= entry[2] and entry[3] <= entry[4]]
    if edge and eligible:
        side, low_x, high_x, low_y, high_y = eligible[int(rng.integers(len(eligible)))]
        ex = int(rng.integers(low_x, high_x + 1))
        ey = int(rng.integers(low_y, high_y + 1))
        selection = "outer_20pct_content_edge_band"
    else:
        ex = int(rng.integers(x0, x1 - ew + 1))
        ey = int(rng.integers(y0, y1 - eh + 1))
        side = None
        selection = "uniform_position_in_content"
    out[ey:ey + eh, ex:ex + ew] = 0
    record.update(applied=True, box=[ex, ey, ex + ew, ey + eh],
                  selection=selection, edge_side=side,
                  area_fraction_of_content=ew * eh / (cw * ch),
                  value_before_normalization=0)
    return out, record


class ABImageDataset(OvernightImageDataset):
    """Original record validation with one shared A/B decoder and train-only erasing."""

    def __init__(self, records: Sequence[dict[str, Any]], image_root: str | Path, *,
                 training: bool, arm: str, image_size: int = 768, seed: int = 48):
        if arm not in ("A", "B") or image_size != 768:
            raise ValueError("A/B dataset requires arm A or B at 768 pixels")
        if not records or records[0].get("split") not in ("train", "valid"):
            raise ValueError("A/B dataset accepts official train or valid only")
        super().__init__(records, image_root, training=training, image_size=image_size,
                         seed=seed)
        if self.split not in ("train", "valid"):
            raise ValueError("A/B dataset must not read test")
        self.arm = arm
        self.geometry, self.normalization = _augmentation(seed)
        self.erase_rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        super().reseed(seed)
        self.geometry.set_random_seed(int(seed))
        self.erase_rng = np.random.default_rng(int(seed))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            path = self.paths[index].resolve(strict=True)
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError("Image path escaped its official split")
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != record["file_sha256"]:
                raise ValueError("Image SHA256 differs from accepted manifest")
            canvas, content_box, _ = load_canvas(source, self.arm)
            if canvas.mode != "RGB" or canvas.size != (768, 768):
                raise ValueError("Shared A/B preprocessing returned an invalid canvas")
            pixels = np.asarray(canvas, dtype=np.uint8).copy()
            if self.training:
                transformed = self.geometry(image=pixels, bboxes=[tuple(content_box)])
                pixels = transformed["image"]
                boxes = transformed["bboxes"]
                if len(boxes) != 1:
                    raise ValueError("Base augmentation removed the image content box")
                x0, y0, x1, y1 = boxes[0][:4]
                box = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
                       min(768, int(np.ceil(x1))), min(768, int(np.ceil(y1))))
                if self.arm == "B":
                    pixels, _ = random_erasing_ab_u8(pixels, box, training=True,
                        rng=self.erase_rng, p=0.5, area_range=(0.005, 0.02))
            tensor = self.normalization(image=pixels)["image"].to(dtype=torch.float32)
            if tensor.device.type != "cpu" or tuple(tensor.shape) != (3, 768, 768):
                raise ValueError("A/B dataset must return a CPU 3x768x768 tensor")
        except Exception as exc:
            raise CachedSampleError(f"image_id={record['image_id']}; {type(exc).__name__}: {exc}") from exc
        return {"image": tensor, "targets": torch.tensor(record["targets"], dtype=torch.float32),
                "image_id": record["image_id"], "study_key": record["study_key"],
                "subject_id": record["subject_id"], "study_id": record["study_id"],
                "split": self.split, "loss_weight": self.loss_weights[index]}


class ABCompleteStudyDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_root, *, image_size=768, seed=48, arm):
        self.records = [dict(row) for row in records]
        self.groups = complete_study_groups(self.records)
        self.image_dataset = ABImageDataset(self.records, image_root, training=True,
                                            arm=arm, image_size=image_size, seed=seed)

    def __len__(self):
        return len(self.groups)

    def reseed(self, seed):
        self.image_dataset.reseed(seed)

    def __getitem__(self, index):
        batch = collate_overnight([self.image_dataset[i] for i in self.groups[index]])
        if len(set(batch["study_keys"])) != 1:
            raise RuntimeError("A/B Study microbatch mixed multiple Studies")
        return batch


def make_ab_study_loader(records, image_root, *, arm, image_size, num_workers,
                         prefetch_factor, seed, pin_memory=True):
    if not 0 <= num_workers <= 12 or prefetch_factor not in (1, 2):
        raise ValueError("Invalid worker or prefetch configuration")
    dataset = ABCompleteStudyDataset(records, image_root, arm=arm,
                                     image_size=image_size, seed=seed)
    options = {"num_workers": num_workers, "pin_memory": pin_memory,
               "generator": torch.Generator().manual_seed(seed),
               "worker_init_fn": _seed_worker}
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False,
                       multiprocessing_context="spawn", timeout=180.0)
    return DataLoader(dataset, batch_size=None, shuffle=True, **options)


def make_ab_valid_loader(records, image_root, *, arm, training, image_size, batch_size,
                         num_workers=0, prefetch_factor=2, seed=48, pin_memory=True):
    if training or batch_size < 1 or not 0 <= num_workers <= 12:
        raise ValueError("A/B validation loader accepts only valid records")
    dataset = ABImageDataset(records, image_root, training=False, arm=arm,
                             image_size=image_size, seed=seed)
    if dataset.split != "valid":
        raise ValueError("A/B validation loader requires valid split")
    options = {"num_workers": num_workers, "pin_memory": pin_memory,
               "generator": torch.Generator().manual_seed(seed),
               "worker_init_fn": _seed_worker}
    if num_workers:
        if prefetch_factor not in (1, 2):
            raise ValueError("Invalid prefetch factor")
        options.update(prefetch_factor=prefetch_factor, persistent_workers=False,
                       multiprocessing_context="spawn", timeout=180.0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, collate_fn=collate_overnight, **options)
