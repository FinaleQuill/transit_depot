"""Replay one approved AP train image through the actual A/B augmentation path."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tight_crop_ab_data import ABImageDataset, random_erasing_ab_u8
from tight_crop_ab_preprocess import load_canvas

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/clean_v1_reviewed/train_clean.csv"
SOURCE_ROOT = Path("/home/featurize/data/cxr_first")
OUTPUT = ROOT / "artifacts/tight_crop_ab_20261102/augmentation_preview"
IMAGE_ID = "train:005642"
PANEL = 440
GAP = 18
TOP = 58
BOTTOM = 37


def file_sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_record():
    from dataset import read_manifest
    rows = [row for row in read_manifest(MANIFEST, split="train") if row["image_id"] == IMAGE_ID]
    if len(rows) != 1 or rows[0]["view_position"].upper() != "AP":
        raise ValueError("Expected exactly one approved AP train record")
    record = rows[0]
    relative = Path(record["relative_path"])
    if (relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "train"):
        raise ValueError("Source leaves official train")
    source = (SOURCE_ROOT / relative).resolve(strict=True)
    if not source.is_relative_to((SOURCE_ROOT / "train").resolve(strict=True)):
        raise ValueError("Source leaves official train")
    if file_sha(source) != record["file_sha256"]:
        raise ValueError("Approved source SHA256 changed")
    return record, source


def geometry_box(box):
    x0, y0, x1, y1 = box[0][:4]
    result = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
              min(768, int(np.ceil(x1))), min(768, int(np.ceil(y1))))
    if not 0 <= result[0] < result[2] <= 768 or not 0 <= result[1] < result[3] <= 768:
        raise ValueError("Transformed content box is invalid")
    return result


def replay(dataset, source_bytes, *, seed):
    canvas, content_box, meta = load_canvas(source_bytes, "B")
    pixels = np.asarray(canvas, dtype=np.uint8).copy()
    dataset.reseed(seed)
    transformed = dataset.geometry(image=pixels, bboxes=[tuple(content_box)])
    if len(transformed["bboxes"]) != 1:
        raise ValueError("Actual training geometry removed the content box")
    before = transformed["image"]
    box = geometry_box(transformed["bboxes"])
    after, erase = random_erasing_ab_u8(before, box, training=True,
        rng=dataset.erase_rng, p=0.5, area_range=(0.005, 0.02))
    return canvas, before, after, content_box, box, meta, erase


def panel_image(image, *, box=None, box_color=(255, 78, 66)):
    image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    scale = min(PANEL / image.width, PANEL / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    tile = Image.new("RGB", (PANEL, PANEL), (0, 0, 0))
    offset = ((PANEL - size[0]) // 2, (PANEL - size[1]) // 2)
    tile.paste(image.resize(size, Image.Resampling.LANCZOS), offset)
    if box is not None:
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(tile)
        draw.rectangle((offset[0] + round(x0 * scale), offset[1] + round(y0 * scale),
                        offset[0] + round((x1 - 1) * scale), offset[1] + round((y1 - 1) * scale)),
                       outline=box_color, width=3)
    return tile


def render(original, canvas, before, after, erase_box, *, seed):
    width = 4 * PANEL + 5 * GAP
    board = Image.new("RGB", (width, TOP + PANEL + BOTTOM), (29, 34, 40))
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = font
    title = ["Original full image", "B crop + 768 canvas",
             "Training geometry, pre-erase", "Same geometry, post-erase"]
    images = [panel_image(original), panel_image(canvas),
              panel_image(before, box=erase_box), panel_image(after, box=erase_box)]
    for index, (tile, label) in enumerate(zip(images, title)):
        x = GAP + index * (PANEL + GAP)
        board.paste(tile, (x, TOP))
        draw.text((x, 16), label, fill=(240, 244, 245), font=font)
    draw.text((GAP, TOP + PANEL + 11),
              f"{IMAGE_ID} | AP | seed {seed} | red: actual erased patch | full decode, no draft",
              fill=(204, 216, 219), font=small)
    return board


def main():
    import torch
    record, source = selected_record()
    source_bytes = source.read_bytes()
    dataset = ABImageDataset([record], SOURCE_ROOT, training=True, arm="B", seed=48)
    selected = None
    for seed in range(48, 4144):
        replayed = replay(dataset, source_bytes, seed=seed)
        if replayed[-1]["applied"]:
            selected = (seed, replayed)
            break
    if selected is None:
        raise RuntimeError("No deterministic erasing trigger found")
    seed, (canvas, before, after, original_box, transformed_box, canvas_meta, erase) = selected
    if erase["box"] is None or np.array_equal(before, after):
        raise AssertionError("Erasing did not alter the training image")
    dataset.reseed(seed)
    actual = dataset[0]["image"]
    expected = dataset.normalization(image=after)["image"].to(dtype=torch.float32)
    if not torch.equal(actual, expected):
        raise AssertionError("Rendered augmentation differs from the actual training tensor")
    with Image.open(source) as image:
        image.load()
        original = ImageOps.exif_transpose(image).convert("RGB")
    board = render(original, canvas, before, after, erase["box"], seed=seed)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    image_path = OUTPUT / "train_005642_real_augmentation.png"
    board.save(image_path)
    metadata = {"image_id": IMAGE_ID, "split": "train", "view_position": "AP",
                "source_path": str(source), "source_sha256": record["file_sha256"],
                "manifest_sha256": file_sha(MANIFEST), "seed": seed,
                "rng_first_uniform": float(np.random.default_rng(seed).random()),
                "geometry": "ABImageDataset.geometry", "erasing": "random_erasing_ab_u8",
                "erase_record": erase, "canvas_metadata": canvas_meta,
                "original_canvas_content_box": list(original_box),
                "transformed_content_box": list(transformed_box),
                "training_tensor_exact_match": True,
                "normalization_applied_after_erasing": True,
                "selection_method": "first triggered seed from 48 upward, independent of pixel intensity",
                "image_sha256": file_sha(image_path),
                "source_files_sha256": {name: file_sha(ROOT / name) for name in
                    ("tight_crop_ab_data.py", "tight_crop_ab_preprocess.py", "tight_crop.py")}}
    with (OUTPUT / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"image": str(image_path), "seed": seed, "erasing": erase,
                      "training_tensor_exact_match": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
