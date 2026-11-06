"""Original-pixel local inputs for the E2 candidate; no training entrypoint.

The global tensor and random augmentation sequence match ``StudyDataset``.
Each local input is cropped from the fully decoded, oriented source, before
resizing. ``tile_boxes`` remains diagnostic SOURCE xyxy. ``region_affines``
maps normalized NETWORK coordinates of each tile to the augmented global
network frame, including actual rounded letterboxing, reflection and rotation.
These affine maps describe the continuous resampling geometry, not a claim
that interpolated pixels or the zero-filled rotation corners are invertible.

One item contains every image in a Study and exactly one ten-class label.
The caller remains responsible for the registered train/dev/H split boundary.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import math
import random

import numpy as np
from PIL import Image, ImageOps
import torch

from reconstruction.data import StudyDataset, letterbox, tile_boxes


def augmentation_parameters(seed: int, epoch: int, image_id: str, active: bool):
    """Use the frozen E0 per-image RNG sequence without changing global pixels."""
    seed_bytes = f"{seed}/{epoch}/{image_id}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big"))
    flip = active and rng.random() < 0.5
    brightness, contrast, angle = 0., 1., 0.
    if active and rng.random() < 0.5:
        brightness, contrast = rng.uniform(-0.10, 0.10) * 255., rng.uniform(0.90, 1.10)
    if active and rng.random() < 0.3:
        angle = rng.uniform(-5., 5.)
    return bool(flip), brightness, contrast, angle


def tile_to_global_affines(width: int, height: int, size: int, boxes, *,
                           flip: bool = False, angle: float = 0.) -> torch.Tensor:
    """Return [R,2,3] maps from tile [0,1] edge coordinates to global [0,1].

    Positions can lie outside [0,1] after rotation; they must not be clamped.
    Region zero must be the whole image and has an exactly identity map.
    """
    if any(type(v) is not int or v < 1 for v in (width, height, size)):
        raise ValueError("Source and network dimensions must be positive integers")
    if not boxes or tuple(boxes[0]) != (0, 0, width, height):
        raise ValueError("Region zero must be the full source image")
    if type(flip) is not bool or not math.isfinite(angle):
        raise ValueError("Invalid geometric augmentation")
    radians = math.radians(angle)
    c, s, center = math.cos(radians), math.sin(radians), size / 2
    # Pillow positive angles rotate counterclockwise; y increases downwards.
    rotate = np.array([[c, s, center * (1 - c - s)],
                       [-s, c, center * (1 + s - c)], [0., 0., 1.]])
    reflect = np.array([[-1., 0., size], [0., 1., 0.], [0., 0., 1.]]) if flip else np.eye(3)
    transforms = []
    for box in boxes:
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("A source tile must be nonempty and inside its image")
        w, h = right - left, bottom - top
        scale = size / max(w, h)
        rw, rh = tuple(max(1, min(size, round(v * scale))) for v in (w, h))
        sx, sy = rw / w, rh / h
        resize = np.array([[sx, 0., (size-rw)//2 - left*sx],
                           [0., sy, (size-rh)//2 - top*sy], [0., 0., 1.]])
        transforms.append(rotate @ reflect @ resize)
    normalized_to_pixels = np.diag([size, size, 1.])
    pixels_to_normalized = np.diag([1/size, 1/size, 1.])
    result = [np.eye(3)[:2]]
    for transform in transforms[1:]:
        mapping = pixels_to_normalized @ transforms[0] @ np.linalg.inv(transform) @ normalized_to_pixels
        result.append(mapping[:2])
    return torch.from_numpy(np.asarray(result, dtype=np.float32))


class LocalStudyDataset(StudyDataset):
    """E2 full image plus four 60% source crops with explicit pixel geometry.

    The default input is [V,5,3,S,S]. Set local_tiles=False only for a matched
    global continuation/control; it returns [V,1,3,S,S] with identity affines.
    No source file is modified and no crop is promoted to a labeled example.
    """
    def __init__(self, records, image_root, *, image_size=768, training=False,
                 seed=43, local_tiles=True, verify_hash=True, augment=True):
        super().__init__(records, image_root, image_size=image_size, training=training,
                         seed=seed, local_tiles=local_tiles, verify_hash=verify_hash, augment=augment)

    def _image(self, index):
        row = self.records[index]
        path = self.paths[index].resolve(strict=True)
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
        flip, brightness, contrast, angle = augmentation_parameters(
            self.seed, self.epoch, row["image_id"], self.training and self.augment)
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
            tensors.append(torch.from_numpy(np.ascontiguousarray((pixels-self._mean)/self._std)))
        coords = torch.tensor([(a/image.width, b/image.height, c/image.width, d/image.height)
                               for a, b, c, d in boxes], dtype=torch.float32)
        affines = tile_to_global_affines(*image.size, self.image_size, boxes, flip=flip, angle=angle)
        return torch.stack(tensors), coords, affines, (flip, brightness, contrast, angle)

    def __getitem__(self, index):
        group = self.groups[index]
        rows = [self.records[i] for i in group]
        loaded = [self._image(i) for i in group]
        return {"images": torch.stack([x[0] for x in loaded]),
                "targets": torch.tensor(rows[0]["targets"], dtype=torch.float32),
                "study_key": rows[0]["study_key"], "subject_id": rows[0]["subject_id"],
                "study_id": rows[0]["study_id"], "image_ids": [r["image_id"] for r in rows],
                "views": [r.get("view_position", "") for r in rows],
                "tile_boxes": torch.stack([x[1] for x in loaded]),
                "region_affines": torch.stack([x[2] for x in loaded]),
                "augmentation": torch.tensor([x[3] for x in loaded], dtype=torch.float32)}
