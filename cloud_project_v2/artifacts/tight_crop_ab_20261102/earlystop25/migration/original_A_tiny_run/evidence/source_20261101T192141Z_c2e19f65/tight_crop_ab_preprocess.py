"""Frozen full-decode A/B canvas shared by training and validation inference.

This experimental preprocessing does not imply deployment timing certification.
It uses the visually approved body-envelope crop without EXACT-BLACK trimming.
Random erasing belongs exclusively to the training dataset, never this module.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from fast_lanczos import resize
from tight_crop import detect_tight_crop

IMAGE_SIZE = 768
DECODER = "full_decode_exact_reference_v1"


def _load(source, arm: str):
    if arm not in ("A", "B"):
        raise ValueError("Expected registered A/B arm")
    if isinstance(source, (str, Path)):
        path = Path(source).resolve(strict=True)
        if "test" in {part.lower() for part in path.parts}:
            raise ValueError("Experimental A/B preprocessing must not read test")
        source = path
    elif isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    else:
        raise TypeError("Expected source bytes or official train/valid image path")
    cv2.setNumThreads(1)
    with Image.open(source) as image:
        image.load()  # Full original decode, explicitly no JPEG draft.
        ImageOps.exif_transpose(image, in_place=True)
        if image.mode in {"I", "F"} or image.mode.startswith("I;"):
            raise ValueError("Non-8-bit source requires an explicit intensity policy")
        original_size = image.size
        pixels = image if image.mode == "L" else image.convert("RGB")
        box = (0, 0, image.width, image.height)
        diagnostics = {"applied": False, "reason": "baseline_full_frame"}
        if arm == "B":
            gray = np.asarray(pixels.convert("L"), dtype=np.uint8)
            result = detect_tight_crop(gray)
            box, diagnostics = result.box, result.diagnostics
            pixels = pixels.crop(box)
        scale = IMAGE_SIZE / max(pixels.size)
        size = tuple(max(1, min(IMAGE_SIZE, round(length * scale)))
                     for length in pixels.size)
        # One thread avoids nested native pools in parallel DataLoader workers.
        # Native resize is pixel-exact with the pinned Pillow LANCZOS baseline.
        resized = resize(pixels, size, threads=1)
        x, y = ((IMAGE_SIZE - value) // 2 for value in size)
        canvas = Image.new(pixels.mode, (IMAGE_SIZE, IMAGE_SIZE), 0)
        canvas.paste(resized, (x, y))
        content_box = (x, y, x + size[0], y + size[1])
        metadata = {"arm": arm, "decoder": DECODER,
                    "original_size": list(original_size), "crop_box": list(box),
                    "content_box": list(content_box), "single_scale": scale,
                    "crop_diagnostics": diagnostics, "random_erasing": False}
        return canvas, content_box, metadata


def load_canvas(source, arm: str):
    """Return RGB PIL canvas, half-open content box, and geometry provenance."""
    canvas, box, metadata = _load(source, arm)
    return canvas.convert("RGB"), box, metadata


def load_pixels(source, arm: str) -> np.ndarray:
    """Return identical uint8 pixels, retaining grayscale for efficient inference."""
    canvas, _, _ = _load(source, arm)
    return np.asarray(canvas, dtype=np.uint8).copy()
