"""Two-image, train-only visual review of the tight crop proposal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tight_crop import detect_tight_crop, letterbox, random_erasing_u8


PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "data/clean_v1_reviewed/train_clean.csv"
SOURCE_ROOT = Path("/home/featurize/data/cxr_first")
OUTPUT = PROJECT / "artifacts/tight_crop_two_image_20261102"
IDS = ("train:006758", "train:005642")
TARGET_SIZE = 768
CYAN = (24, 224, 235)
RED = (255, 72, 65)
GREEN = (73, 230, 124)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows_for_review() -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_id = row["image_id"]
            if image_id not in IDS:
                continue
            if image_id in found:
                raise ValueError(f"Duplicate manifest ID: {image_id}")
            if row["split"] != "train" or not image_id.startswith("train:"):
                raise ValueError(f"Non-train record: {image_id}")
            found[image_id] = row
    if set(found) != set(IDS):
        raise ValueError(f"Missing train records: {set(IDS) - set(found)}")
    return [found[image_id] for image_id in IDS]


def source_for(row: dict[str, str]) -> Path:
    relative = Path(row["relative_path"])
    if relative.is_absolute() or relative.parts[0] != "train" or ".." in relative.parts:
        raise ValueError(f"Invalid train path: {relative}")
    train_root = (SOURCE_ROOT / "train").resolve(strict=True)
    source = (SOURCE_ROOT / relative).resolve(strict=True)
    if not source.is_relative_to(train_root):
        raise ValueError(f"Path escapes train root: {source}")
    return source


def checked_box(box, width: int, height: int, name: str) -> tuple[int, int, int, int]:
    if len(box) != 4 or any(int(v) != v for v in box):
        raise ValueError(f"{name} must be four integer pixel coordinates")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{name} outside original frame: {box}")
    return x0, y0, x1, y1


def draw_box(draw: ImageDraw.ImageDraw, box, scale: float, offset: tuple[int, int], color):
    x0, y0, x1, y1 = box
    ox, oy = offset
    draw.rectangle((ox + round(x0 * scale), oy + round(y0 * scale),
                    ox + round((x1 - 1) * scale), oy + round((y1 - 1) * scale)),
                   outline=color, width=3)


def fit_panel(image: Image.Image, panel: int, resampling=Image.Resampling.LANCZOS):
    scale = min(panel / image.width, panel / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    offset = ((panel - size[0]) // 2, (panel - size[1]) // 2)
    tile = Image.new("RGB", (panel, panel), (0, 0, 0))
    tile.paste(image.resize(size, resampling), offset)
    return tile, scale, offset


def review_panel(image: Image.Image, mask: np.ndarray, crop_box, body_box,
                 letterboxed: Image.Image, image_id: str) -> Image.Image:
    panel = 520
    gap = 18
    top = 92
    bottom = 42
    canvas = Image.new("RGB", (4 * panel + 5 * gap, top + panel + bottom), (29, 34, 39))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    draw.text((gap, 10), f"{image_id} | proposal, not anatomy segmentation", fill="white", font=font)
    draw.text((gap, 43), "Cyan: original frame   Red: proposed crop   Green: body bound", fill="white", font=small)
    original, scale, offset = fit_panel(image.convert("RGB"), panel)
    mask_image, _, _ = fit_panel(Image.fromarray(mask).convert("RGB"), panel, Image.Resampling.NEAREST)
    overlay = original.copy()
    outline = ImageDraw.Draw(overlay)
    draw_box(outline, (0, 0, image.width, image.height), scale, offset, CYAN)
    draw_box(outline, crop_box, scale, offset, RED)
    draw_box(outline, body_box, scale, offset, GREEN)
    result = letterboxed.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
    for index, (tile, label) in enumerate(zip(
        (original, mask_image, overlay, result),
        ("Original full frame", "Analysis mask", "Proposal over original", "768 letterbox, no erasing"),
    )):
        x = gap + index * (panel + gap)
        canvas.paste(tile, (x, top))
        draw.text((x, top + panel + 9), label, fill="white", font=small)
    return canvas


def raw_strips(image: Image.Image, box, out_dir: Path) -> list[dict]:
    w, h = image.size
    x0, y0, x1, y1 = box
    cuts = {"left": x0, "top": y0, "right": w - x1, "bottom": h - y1}
    records = []
    for side, cut in cuts.items():
        if cut == 0:
            continue
        reach = min(160, (w if side in ("left", "right") else h) - cut)
        if side == "left":
            strip_box = (0, 0, x0 + reach, h)
        elif side == "right":
            strip_box = (x1 - reach, 0, w, h)
        elif side == "top":
            strip_box = (0, 0, w, y0 + reach)
        else:
            strip_box = (0, y1 - reach, w, h)
        path = out_dir / f"{side}_raw_1to1.png"
        image.crop(strip_box).save(path)
        records.append({"side": side, "removed_pixels": cut, "source_box": strip_box,
                        "path": str(path.relative_to(OUTPUT))})
    return records


def render_one(row: dict[str, str]) -> dict:
    image_id = row["image_id"]
    source = source_for(row)
    before = sha256(source)
    if before != row["file_sha256"]:
        raise ValueError(f"Manifest/source SHA mismatch: {image_id}")
    with Image.open(source) as opened:
        image = opened.convert("L")
    w, h = image.size
    if (w, h) != (int(row["width"]), int(row["height"])):
        raise ValueError(f"Manifest/source size mismatch: {image_id}")
    gray = np.asarray(image)
    proposal = detect_tight_crop(gray)
    box = checked_box(proposal.box, w, h, "crop box")
    body_box = checked_box(proposal.body_box, w, h, "body box")
    crop = image.crop(box)
    result, letterbox_meta = letterbox(crop, TARGET_SIZE)
    if result.size != (TARGET_SIZE, TARGET_SIZE):
        raise ValueError(f"Unexpected letterbox size: {result.size}")
    old_scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_scale = min(TARGET_SIZE / crop.width, TARGET_SIZE / crop.height)
    out_dir = OUTPUT / image_id.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=False)
    crop.save(out_dir / "crop.png")
    result.save(out_dir / "letterbox768.png")
    mask = np.asarray(proposal.mask)
    seed = np.asarray(proposal.seed_mask)
    if mask.ndim != 2 or seed.shape != mask.shape:
        raise ValueError(f"Invalid analysis masks: {image_id}")
    mask_vis = np.where(mask > 0, 255, 0).astype(np.uint8)
    seed_vis = np.where(seed > 0, 255, 0).astype(np.uint8)
    Image.fromarray(mask_vis).save(out_dir / "mask.png")
    Image.fromarray(seed_vis).save(out_dir / "seed_mask.png")
    review_panel(image, mask_vis, box, body_box, result, image_id).save(out_dir / "review_four_panel.png")
    strips = raw_strips(image, box, out_dir)
    after = sha256(source)
    if after != before:
        raise ValueError(f"Source changed during render: {image_id}")
    cuts = {"left": box[0], "top": box[1], "right": w - box[2], "bottom": h - box[3]}
    fractions = {side: count / (w if side in ("left", "right") else h)
                 for side, count in cuts.items()}
    return {
        "image_id": image_id, "split": "train", "view": row["view_position"],
        "source_path": str(source), "relative_path": row["relative_path"],
        "source_sha256_manifest": row["file_sha256"],
        "source_sha256_before": before, "source_sha256_after": after,
        "original_size": [w, h], "crop_box_half_open": box, "body_box_half_open": body_box,
        "crop_size": list(crop.size), "removed_pixels_by_side": cuts,
        "removed_fraction_by_side": fractions,
        "removed_area_fraction": 1 - (crop.width * crop.height) / (w * h),
        "full_frame_768_scale": old_scale, "crop_768_scale": new_scale,
        "effective_linear_magnification": new_scale / old_scale,
        "letterbox": json_value(letterbox_meta),
        "analysis_size": [int(mask.shape[1]), int(mask.shape[0])],
        "diagnostics": json_value(proposal.diagnostics),
        "files": {name: str((out_dir / filename).relative_to(OUTPUT)) for name, filename in {
            "crop": "crop.png", "letterbox768": "letterbox768.png", "mask": "mask.png",
            "seed_mask": "seed_mask.png", "review_four_panel": "review_four_panel.png",
        }.items()},
        "raw_strips": strips,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Review output already exists: {OUTPUT}")
    rows = rows_for_review()
    provenance = {
        "manifest_path": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "runner_sha256": sha256(Path(__file__)),
        "detector_sha256": sha256(PROJECT / "tight_crop.py"),
        "target_size": TARGET_SIZE,
    }
    OUTPUT.mkdir(parents=True)
    items = [render_one(row) for row in rows]
    report = {"scope": "Two hard-coded train originals only; crop proposal is not anatomy segmentation.",
              "provenance": provenance, "items": items}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    lines = ["# Tight crop two-image preview", "",
             "Proposal only, not anatomy segmentation. Letterbox results have no random erasing.", "",
             f"Manifest SHA256: `{provenance['manifest_sha256']}`  ",
             f"Detector SHA256: `{provenance['detector_sha256']}`  ",
             f"Runner SHA256: `{provenance['runner_sha256']}`", ""]
    for item in items:
        lines += [f"## {item['image_id']}", "",
                  f"Source: `{item['source_path']}`; SHA256 before/after: `{item['source_sha256_before']}` / `{item['source_sha256_after']}`", "",
                  f"Box: `{item['crop_box_half_open']}`; body: `{item['body_box_half_open']}`; size: `{item['crop_size']}`", "",
                  f"Removed px: `{item['removed_pixels_by_side']}`; area: `{item['removed_area_fraction']:.6f}`", "",
                  f"768 scale full/crop: `{item['full_frame_768_scale']:.6f}` / `{item['crop_768_scale']:.6f}`; linear factor: `{item['effective_linear_magnification']:.4f}`", "",
                  f"[Four-panel review]({item['files']['review_four_panel']}) | [Raw crop]({item['files']['crop']}) | [768 letterbox]({item['files']['letterbox768']}) | [Mask]({item['files']['mask']}) | [Seed]({item['files']['seed_mask']})", "",
                  *[f"[Raw {strip['side']} strip, 1:1]({strip['path']})  " for strip in item["raw_strips"]], ""]
    (OUTPUT / "index.md").write_text("\n".join(lines))
    print(OUTPUT / "report.json")


def add_erasing_demo() -> None:
    """Append fixed-seed examples to an already verified two-image preview."""
    report_path = OUTPUT / "report.json"
    report = json.loads(report_path.read_text())
    if [item["image_id"] for item in report["items"]] != list(IDS):
        raise ValueError("Existing report is not the fixed two-image train preview")
    if report["provenance"]["manifest_sha256"] != sha256(MANIFEST):
        raise ValueError("Manifest changed since original preview")
    if report["provenance"]["detector_sha256"] != sha256(PROJECT / "tight_crop.py"):
        raise ValueError("Core changed since original preview")
    seeds = {"train:006758": 0, "train:005642": 2}
    examples = []
    for item in report["items"]:
        image_id = item["image_id"]
        source = Path(item["source_path"])
        if source != source_for(next(r for r in rows_for_review() if r["image_id"] == image_id)):
            raise ValueError(f"Source path mismatch: {image_id}")
        if sha256(source) != item["source_sha256_before"]:
            raise ValueError(f"Source SHA changed: {image_id}")
        main_image = Image.open(OUTPUT / item["files"]["letterbox768"]).convert("RGB")
        seed = seeds[image_id]
        augmented, meta = random_erasing_u8(
            np.asarray(main_image), item["letterbox"]["content_box"],
            training=True, rng=np.random.default_rng(seed), p=.5,
        )
        expected = image_id == "train:005642"
        if bool(meta["applied"]) != expected:
            raise ValueError(f"Fixed seed trigger changed: {image_id}")
        out_dir = OUTPUT / image_id.replace(":", "_")
        example_path = out_dir / "training_erasing_example.png"
        if example_path.exists():
            raise FileExistsError(example_path)
        Image.fromarray(augmented).save(example_path)
        comparison = Image.new("RGB", (2 * TARGET_SIZE + 36, TARGET_SIZE + 94), (29, 34, 39))
        draw = ImageDraw.Draw(comparison)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        title = (f"TRAINING AUGMENTATION EXAMPLE ONLY | {image_id} | seed={seed} | "
                 f"actual trigger: {'YES' if meta['applied'] else 'NO'}")
        draw.text((12, 10), title, font=font, fill="white")
        draw.text((12, 39), "p=0.5, uniform position, 0.5%-2% of content; no brightness targeting", font=font, fill="white")
        comparison.paste(main_image, (12, 78))
        comparison.paste(Image.fromarray(augmented), (TARGET_SIZE + 24, 78))
        comparison_path = out_dir / "training_erasing_comparison.png"
        comparison.save(comparison_path)
        examples.append({"image_id": image_id, "seed": seed,
                         "meaning": "Extra training augmentation example; not the no-erasing main result.",
                         "actual_trigger": bool(meta["applied"]), "metadata": json_value(meta),
                         "example_path": str(example_path.relative_to(OUTPUT)),
                         "comparison_path": str(comparison_path.relative_to(OUTPUT))})
    report["training_augmentation_examples"] = examples
    report["provenance"]["runner_sha256_after_demo_addition"] = sha256(Path(__file__))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    with (OUTPUT / "index.md").open("a") as stream:
        stream.write("\n## Extra training augmentation examples\n\n")
        stream.write("These fixed-seed p=0.5 examples do not replace the no-erasing 768 previews. ")
        stream.write("A non-triggered draw stays unchanged; a triggered draw uses one uniformly located black patch of at most 2% content area.\n\n")
        for example in examples:
            stream.write(f"- {example['image_id']} seed {example['seed']}: actual trigger "
                         f"{'YES' if example['actual_trigger'] else 'NO'}; "
                         f"[comparison]({example['comparison_path']}) / "
                         f"[example]({example['example_path']})\n")
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-erasing-demo", action="store_true")
    args = parser.parse_args()
    if args.add_erasing_demo:
        add_erasing_demo()
    else:
        main()
