"""Complete-Study input for isolated reconstruction experiments.

Only explicitly supplied train/valid records are read.  Targets remain Study
labels; a local crop is never given a separate hard label.  Full source decoding,
EXIF orientation, aspect-preserving Pillow LANCZOS and ImageNet normalization are
shared by training and evaluation.  Hashes are checked on the first read of each
image in each dataset/worker instance, not claimed to be checked every epoch.
"""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10


def validate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate identity, original binary labels and within-Study agreement."""
    if not records:
        raise ValueError("A Study dataset requires nonempty records")
    result, seen, labels = [], set(), {}
    splits = set()
    for original in records:
        row = dict(original)
        split = row.get("split")
        if split not in {"train", "valid"}:
            raise ValueError("Reconstruction data accepts official train/valid only")
        splits.add(split)
        for name in ("image_id", "subject_id", "study_id", "study_key", "relative_path"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Missing original identity field: {name}")
        if row["image_id"] in seen:
            raise ValueError("Duplicate image identity")
        seen.add(row["image_id"])
        if row["study_key"] != f"{row['subject_id']}/{row['study_id']}":
            raise ValueError("Study identity does not match its patient")
        relative = Path(row["relative_path"])
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4
                or relative.parts[:3] != (split, row["subject_id"], row["study_id"])):
            raise ValueError("Original image path escapes its Study")
        if row.get("dicom_id") and relative.stem != row["dicom_id"]:
            raise ValueError("Image path and dicom identity disagree")
        target = row.get("targets")
        if isinstance(target, str):
            target = json.loads(target)
        if (not isinstance(target, (list, tuple)) or len(target) != NUM_CLASSES
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or v not in (0, 1) for v in target)):
            raise ValueError("Original targets must be ten binary Study labels")
        target = tuple(int(v) for v in target)
        previous = labels.setdefault(row["study_key"], target)
        if previous != target:
            raise ValueError("Conflicting original labels inside one Study")
        for name in ("file_sha256", "pixel_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(name, ""))) is None:
                raise ValueError(f"Missing accepted source {name}")
        if split == "train" and str(row.get("training_eligible", "")).lower() != "true":
            raise ValueError("Only accepted training images are permitted")
        if row.get("status", "ok") != "ok":
            raise ValueError("Unresolved image status")
        row["targets"] = target
        result.append(row)
    if len(splits) != 1:
        raise ValueError("A dataset cannot mix original train and valid splits")
    return result


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicated manifest columns")
        records = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in records):
        raise ValueError("Malformed manifest row")
    return validate_records(records)


def study_groups(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(row["study_key"], []).append(index)
    return list(groups.values())


def study_pos_weights(records: Sequence[dict[str, Any]], power: float = 0.5) -> torch.Tensor:
    """Compute weights from the explicitly supplied training subset, once/Study."""
    checked = validate_records(records)
    if any(r["split"] != "train" for r in checked) or not 0 <= power <= 1:
        raise ValueError("Class weights require training records and power in [0,1]")
    labels = np.asarray([checked[g[0]]["targets"] for g in study_groups(checked)], dtype=np.float64)
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError("Every class requires positive and negative training Studies")
    return torch.tensor((negative / positive) ** power, dtype=torch.float32)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    if type(size) is not int or size < 1:
        raise ValueError("Input size must be a positive integer")
    image = image.convert("RGB")
    scale = size / max(image.size)
    dimensions = tuple(max(1, min(size, round(v * scale))) for v in image.size)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image.resize(dimensions, Image.Resampling.LANCZOS),
                 ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
    return canvas


def tile_boxes(width: int, height: int, local_tiles: bool) -> list[tuple[int, int, int, int]]:
    """Global plus four overlapping 60% windows in original image coordinates."""
    boxes = [(0, 0, width, height)]
    if local_tiles:
        w, h = max(1, round(width * 0.6)), max(1, round(height * 0.6))
        boxes.extend((x, y, x + w, y + h) for y in (0, height - h) for x in (0, width - w))
    return boxes


class StudyDataset(Dataset):
    """One element contains all original views of one Study, with a single label.

    images is Vx3xSxS by default, or Vx5x3xSxS with local_tiles=True.  tile_boxes
    is VxKx4 normalized xyxy in each oriented source image.  Training geometry
    is sampled per image and shared across its global/local inputs.  views is
    diagnostic metadata, never a required model input.
    """
    def __init__(self, records, image_root, *, image_size=768, training=False,
                 seed=43, local_tiles=False, verify_hash=True, augment=True):
        self.records = validate_records(records)
        if training and self.records[0]["split"] != "train":
            raise ValueError("Only official training records can receive augmentation")
        if type(image_size) is not int or image_size < 1:
            raise ValueError("Input size must be a positive integer")
        self.groups = study_groups(self.records)
        self.image_size, self.training, self.seed = image_size, bool(training), int(seed)
        self.local_tiles, self.verify_hash, self.augment = bool(local_tiles), bool(verify_hash), bool(augment)
        self.epoch = 0
        self.root = Path(image_root).resolve(strict=True)
        split = self.records[0]["split"]
        if not (self.root / split).is_dir() and (self.root / "cxr_first" / split).is_dir():
            self.root = self.root / "cxr_first"
        if not (self.root / split).is_dir():
            raise FileNotFoundError(f"Image root does not contain {split}")
        self.paths = []
        for row in self.records:
            path = (self.root / row["relative_path"]).resolve(strict=True)
            if not path.is_relative_to(self.root / split):
                raise ValueError("Source image symlink leaves its original split")
            self.paths.append(path)
        self._verified: set[int] = set()
        self._mean = np.asarray(MEAN, dtype=np.float32)[:, None, None]
        self._std = np.asarray(STD, dtype=np.float32)[:, None, None]

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        # Set before constructing workers; persistent worker replicas need their
        # own epoch update, so the first trainer rebuilds workers each epoch.
        self.epoch = epoch

    def __len__(self):
        return len(self.groups)

    def _image(self, index):
        row, path = self.records[index], self.paths[index]
        path = path.resolve(strict=True)
        if not path.is_relative_to(self.root / row["split"]):
            raise ValueError("Source image moved outside its original split")
        content = path.read_bytes()
        if self.verify_hash and index not in self._verified:
            if hashlib.sha256(content).hexdigest() != row["file_sha256"]:
                raise ValueError(f"Source image SHA256 changed: {row['image_id']}")
            self._verified.add(index)
        with Image.open(BytesIO(content)) as source:
            source.load()
            upright = ImageOps.exif_transpose(source)
            if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
                raise ValueError("Non-8-bit source requires an explicit intensity policy")
            image = upright.convert("RGB")
        boxes = tile_boxes(*image.size, self.local_tiles)
        seed_bytes = f"{self.seed}/{self.epoch}/{row['image_id']}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big"))
        active = self.training and self.augment
        flip = active and rng.random() < 0.5
        brightness, contrast, angle = 0., 1., 0.
        if active and rng.random() < 0.5:
            brightness, contrast = rng.uniform(-0.10, 0.10) * 255., rng.uniform(0.90, 1.10)
        if active and rng.random() < 0.3:
            angle = rng.uniform(-5., 5.)
        tensors = []
        for box in boxes:
            canvas = letterbox(image.crop(box), self.image_size)
            if flip:
                canvas = ImageOps.mirror(canvas)
            if angle:
                canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            pixels = np.asarray(canvas, dtype=np.float32)
            if brightness or contrast != 1.:
                pixels = np.clip(pixels * contrast + brightness, 0., 255.)
            pixels = pixels.transpose(2, 0, 1) / np.float32(255.)
            tensors.append(torch.from_numpy(np.ascontiguousarray((pixels - self._mean) / self._std)))
        coords = torch.tensor([(a/image.width, b/image.height, c/image.width, d/image.height)
                               for a, b, c, d in boxes], dtype=torch.float32)
        return torch.stack(tensors), coords

    def __getitem__(self, index):
        group = self.groups[index]
        rows = [self.records[i] for i in group]
        loaded = [self._image(i) for i in group]
        images = torch.stack([x[0] for x in loaded])
        if not self.local_tiles:
            images = images[:, 0]
        return {"images": images, "targets": torch.tensor(rows[0]["targets"], dtype=torch.float32),
                "study_key": rows[0]["study_key"], "subject_id": rows[0]["subject_id"],
                "study_id": rows[0]["study_id"], "image_ids": [r["image_id"] for r in rows],
                "views": [r.get("view_position", "") for r in rows],
                "tile_boxes": torch.stack([x[1] for x in loaded])}


def collate_studies(samples):
    """Explicit variable-length batching; batch_size=None needs no collator."""
    return list(samples)
