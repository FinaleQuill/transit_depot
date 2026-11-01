"""Five fixed, read-only source previews of exact-zero trim after frozen tight crop."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tight_crop import letterbox


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/home/featurize/data/cxr_first")
METRICS = ROOT / "artifacts/tight_crop_gallery_20261102/metrics.json"
INDEX = ROOT / "artifacts/geometry_review_20261102/index.json"
OUTPUT = ROOT / "artifacts/exact_black_trim_5case_20261102"
MANIFESTS = {
    "train": ROOT / "data/clean_v1_reviewed/train_clean.csv",
    "valid": ROOT / "data/clean_v1_reviewed/valid_manifest.csv",
}
CASES = ("train:006758", "train:005642", "valid:001034", "train:010083", "train:009682")
DETECTOR_SHA = "85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d"
METRICS_SHA = "2496467931184dc1faa146e542f13c6f2c465fc3ff2892eca76d642c06b4ab22"
INDEX_SHA = "4dccd0d13893068470ea2a5ef98ab24cfc1ad4880fab1f62b374a3ef8afacbeb"
TARGET = 768
CYAN, GREEN, RED = (24, 224, 235), (73, 230, 124), (255, 72, 65)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError(f"Box must contain four integers: {box}")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Box outside original: {box}")
    return x0, y0, x1, y1


def exact_zero_trim(gray: np.ndarray, first_box: tuple[int, int, int, int]):
    if gray.dtype != np.uint8 or gray.ndim not in (2, 3) or (gray.ndim == 3 and gray.shape[2] != 3):
        raise ValueError("Expected original 8-bit L or RGB pixels")
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = checked_box(first_box, width, height)
    first = gray[y0:y1, x0:x1]
    support = first > 0 if gray.ndim == 2 else np.any(first > 0, axis=2)
    ys, xs = np.nonzero(support)
    if not len(xs):
        return first_box, True
    second = (x0 + int(xs.min()), y0 + int(ys.min()),
              x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
    sx0, sy0, sx1, sy1 = second
    retained = gray[sy0:sy1, sx0:sx1]
    if np.count_nonzero(first) != np.count_nonzero(retained):
        raise AssertionError("Exact trim discarded nonzero pixels")
    return second, False


def load_inputs():
    for path, expected in ((ROOT / "tight_crop.py", DETECTOR_SHA),
                           (METRICS, METRICS_SHA), (INDEX, INDEX_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input SHA changed: {path}")
    report = json.loads(METRICS.read_text())
    index = json.loads(INDEX.read_text())
    if report["count"] != 96 or not report["complete"] or index["count"] != 96 or not index["complete"]:
        raise ValueError("Incomplete frozen 96-image records")
    if (report["provenance"]["index_sha256"] != INDEX_SHA or
            report["provenance"]["detector_sha256"] != DETECTOR_SHA):
        raise ValueError("Metrics provenance mismatch")
    metric_rows = [r for r in report["items"] if r["image_id"] in CASES]
    index_rows = [r for r in index["items"] if r["image_id"] in CASES]
    if len(metric_rows) != len(CASES) or len(index_rows) != len(CASES):
        raise ValueError("Missing or duplicated selected IDs")
    metrics_by_id = {r["image_id"]: r for r in metric_rows}
    index_by_id = {r["image_id"]: r for r in index_rows}
    if len(metrics_by_id) != len(CASES) or len(index_by_id) != len(CASES):
        raise ValueError("Duplicate selected ID")
    manifests = {}
    manifest_shas = {}
    for split, path in MANIFESTS.items():
        manifest_shas[split] = sha256(path)
        if manifest_shas[split] != report["provenance"]["manifest_sha256"][split]:
            raise ValueError(f"Manifest SHA mismatch: {split}")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["image_id"] in CASES:
                    if row["image_id"] in manifests or row["split"] != split:
                        raise ValueError("Duplicate or wrong manifest split")
                    manifests[row["image_id"]] = row
    if set(manifests) != set(CASES):
        raise ValueError("Missing selected manifest identity")
    return metrics_by_id, index_by_id, manifests, manifest_shas, report


def source_for(metric, indexed, row):
    image_id, split = metric["image_id"], metric["split"]
    relative = Path(row["relative_path"])
    if (image_id != indexed["image_id"] or image_id != row["image_id"] or
            split != indexed["split"] or split != row["split"] or
            relative.is_absolute() or ".." in relative.parts or relative.parts[0] != split or
            indexed["relative_path"] != row["relative_path"] or
            metric["view"] != indexed["view"] or
            metric["source_sha256_index_manifest_before_after"] != indexed["source_sha256"] or
            indexed["source_sha256"] != row["file_sha256"]):
        raise ValueError(f"Frozen identity mismatch: {image_id}")
    view = row["view_position"]
    expected_view = "Lateral" if view.upper() in ("LATERAL", "LL") else view or "Missing"
    if metric["view"] != expected_view:
        raise ValueError(f"View mismatch: {image_id}")
    source = (SOURCE_ROOT / relative).resolve(strict=True)
    split_root = (SOURCE_ROOT / split).resolve(strict=True)
    if not source.is_relative_to(split_root) or source != Path(indexed["source_path"]).resolve(strict=True) or source != Path(metric["source_path"]).resolve(strict=True):
        raise ValueError(f"Source path mismatch: {image_id}")
    return source


def fit(image, side=420):
    scale = min(side / image.width, side / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    panel = Image.new("RGB", (side, side), "black")
    offset = ((side - size[0]) // 2, (side - size[1]) // 2)
    panel.paste(image.convert("RGB").resize(size, Image.Resampling.LANCZOS), offset)
    return panel, scale, offset


def comparison(image, first, second, first_768, second_768, image_id, discarded):
    side, gap, header, footer = 420, 16, 84, 34
    canvas = Image.new("RGB", (3 * side + 4 * gap, header + 2 * (side + footer + gap)), (29, 34, 39))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    label = "NO EXTRA TRIM" if first == second else "EXACT ZERO TRIM"
    draw.text((gap, 10), f"{image_id} | {label}", fill="white", font=font)
    draw.text((gap, 44), "Cyan: original  Green: first crop  Red: exact-black trim  |  discarded nonzero: 0", fill="white", font=small)
    annotated, scale, offset = fit(image)
    marks = ImageDraw.Draw(annotated)
    for box, color in (((0, 0, image.width, image.height), CYAN), (first, GREEN), (second, RED)):
        x0, y0, x1, y1 = box
        marks.rectangle((offset[0] + round(x0 * scale), offset[1] + round(y0 * scale),
                         offset[0] + round((x1 - 1) * scale), offset[1] + round((y1 - 1) * scale)),
                        outline=color, width=3)
    first_raw = image.crop(first)
    second_raw = image.crop(second)
    tiles = (annotated, fit(first_raw)[0], fit(second_raw)[0],
             fit(first_768)[0], fit(second_768)[0], fit(discarded)[0])
    labels = ("Original with 3 boxes", "First crop, original pixels", "Second crop, original pixels",
              "First isotropic 768", "Second isotropic 768", "Removed-pixel map (white = zero)")
    for n, (tile, title) in enumerate(zip(tiles, labels)):
        x = gap + (n % 3) * (side + gap)
        y = header + (n // 3) * (side + footer + gap)
        canvas.paste(tile, (x, y))
        draw.text((x, y + side + 6), title, fill="white", font=small)
    return canvas


def render(metric, indexed, row):
    source = source_for(metric, indexed, row)
    before = sha256(source)
    if before != row["file_sha256"]:
        raise ValueError(f"Source SHA mismatch: {metric['image_id']}")
    with Image.open(source) as opened:
        if opened.mode not in ("L", "RGB"):
            raise ValueError(f"Expected 8-bit L or RGB original: {metric['image_id']}")
        source_mode = opened.mode
        image = opened.copy()
    width, height = image.size
    if [width, height] != metric["original_size"] or [width, height] != indexed["original_size"] or (width, height) != (int(row["width"]), int(row["height"])):
        raise ValueError(f"Original size mismatch: {metric['image_id']}")
    first = checked_box(metric["crop_box_half_open"], width, height)
    if list(first) != list(indexed.get("crop_box_half_open", first)) and "crop_box_half_open" in indexed:
        raise ValueError("Frozen first box mismatch")
    second, all_black = exact_zero_trim(np.asarray(image), first)
    first_raw, second_raw = image.crop(first), image.crop(second)
    discarded_count = int(np.count_nonzero(np.asarray(first_raw)) - np.count_nonzero(np.asarray(second_raw)))
    if discarded_count != 0:
        raise AssertionError("Nonzero pixels discarded")
    first_768, first_meta = letterbox(first_raw, TARGET)
    second_768, second_meta = letterbox(second_raw, TARGET)
    fx0, fy0, fx1, fy1 = first
    sx0, sy0, sx1, sy1 = second
    removed = {"left": sx0 - fx0, "top": sy0 - fy0, "right": fx1 - sx1, "bottom": fy1 - sy1}
    removed_map = np.zeros((fy1 - fy0, fx1 - fx0), dtype=np.uint8)
    removed_map[:removed["top"], :] = 255
    if removed["bottom"]:
        removed_map[-removed["bottom"]:, :] = 255
    removed_map[:, :removed["left"]] = 255
    if removed["right"]:
        removed_map[:, -removed["right"]:] = 255
    if np.count_nonzero(np.asarray(first_raw)[removed_map > 0]):
        raise AssertionError("Removed map includes nonzero pixels")
    folder = OUTPUT / metric["image_id"].replace(":", "_")
    folder.mkdir(exist_ok=False)
    first_raw.save(folder / "first_raw.png")
    second_raw.save(folder / "second_raw.png")
    second_768.save(folder / "new768.png")
    comparison(image, first, second, first_768, second_768, metric["image_id"],
               Image.fromarray(removed_map)).save(folder / "comparison.png")
    after = sha256(source)
    if after != before:
        raise ValueError(f"Source changed during render: {metric['image_id']}")
    first_padding = TARGET * TARGET - np.prod(first_meta["resized_size"])
    second_padding = TARGET * TARGET - np.prod(second_meta["resized_size"])
    return {
        "image_id": metric["image_id"], "view": metric["view"], "source_path": str(source),
        "source_sha256_before": before, "source_sha256_after": after,
        "original_size": [width, height], "source_mode": source_mode,
        "first_box_half_open": first, "new_box_half_open": second,
        "all_black_fallback": all_black, "no_extra_trim": first == second,
        "extra_removed_pixels_by_side": removed,
        "extra_removed_fraction_of_first_by_side": {key: value / (first_raw.width if key in ("left", "right") else first_raw.height)
                                                    for key, value in removed.items()},
        "discarded_nonzero_count": discarded_count,
        "first_size": first_raw.size, "new_size": second_raw.size,
        "first_aspect_width_over_height": first_raw.width / first_raw.height,
        "new_aspect_width_over_height": second_raw.width / second_raw.height,
        "first_768": {**first_meta, "padding_area_px": int(first_padding),
                      "padding_size_total_px": [TARGET - first_meta["resized_size"][0], TARGET - first_meta["resized_size"][1]]},
        "new_768": {**second_meta, "padding_area_px": int(second_padding),
                    "padding_size_total_px": [TARGET - second_meta["resized_size"][0], TARGET - second_meta["resized_size"][1]]},
        "linear_magnification_new_over_first": second_meta["scale"] / first_meta["scale"],
        "padding_area_delta_px": int(second_padding - first_padding),
        "first_raw": str((folder / "first_raw.png").relative_to(OUTPUT)),
        "second_raw": str((folder / "second_raw.png").relative_to(OUTPUT)),
        "new768": str((folder / "new768.png").relative_to(OUTPUT)),
        "comparison": str((folder / "comparison.png").relative_to(OUTPUT)),
    }


def create_overview(rows):
    side, gap, label_height = 768, 16, 52
    width = 2 * side + 3 * gap
    height = len(rows) * (side + label_height + gap) + gap
    canvas = Image.new("RGB", (width, height), (29, 34, 39))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for n, row in enumerate(rows):
        y = gap + n * (side + label_height + gap)
        first_path = OUTPUT / row["first_raw"]
        second_path = OUTPUT / row["new768"]
        with Image.open(first_path) as first_raw:
            first_768, _ = letterbox(first_raw, TARGET)
        with Image.open(second_path) as new_768:
            if new_768.size != (TARGET, TARGET):
                raise ValueError(f"Non-768 output: {row['image_id']}")
            canvas.paste(new_768, (2 * gap + side, y + label_height))
        canvas.paste(first_768, (gap, y + label_height))
        cuts = row["extra_removed_pixels_by_side"]
        name = f"{row['image_id']} | +cut L{cuts['left']} T{cuts['top']} R{cuts['right']} B{cuts['bottom']} | {row['linear_magnification_new_over_first']:.4f}x"
        draw.text((gap, y), name, fill="white", font=font)
        draw.text((gap, y + 27), "FIRST 768", fill=GREEN, font=small)
        draw.text((2 * gap + side, y + 27), "EXACT-BLACK 768" if not row["no_extra_trim"] else "NO EXTRA TRIM", fill=RED, font=small)
    canvas.save(OUTPUT / "overview_5cases.png")


def complete_existing_preview():
    """Finish the already rendered five cases without decoding source images again."""
    path = OUTPUT / "summary.json"
    summary = json.loads(path.read_text())
    rows = summary["items"]
    if [row["image_id"] for row in rows] != list(CASES) or any(row["discarded_nonzero_count"] for row in rows):
        raise ValueError("Unexpected five-case output")
    if (OUTPUT / "overview_5cases.png").exists():
        raise FileExistsError("Overview already exists")
    for row in rows:
        row["source_mode"] = "L"  # The initial render rejected other modes before copying pixels.
    create_overview(rows)
    old_renderer = summary["provenance"]["renderer_sha256"]
    new_renderer = sha256(Path(__file__))
    summary["provenance"]["renderer_sha256"] = new_renderer
    summary["overview"] = "overview_5cases.png"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    index_path = OUTPUT / "index.md"
    index_text = index_path.read_text()
    index_path.write_text(index_text.replace("# Exact-black trim: five fixed cases\n", "# Exact-black trim: five fixed cases\n\n[Five-case 768 overview](overview_5cases.png)\n")
                          .replace(f"renderer SHA256: `{old_renderer}`", f"renderer SHA256: `{new_renderer}`"))


def verify_existing_preview():
    """Recompute all five original-pixel outputs with the final source revision."""
    metrics, index, manifests, _, _ = load_inputs()
    path = OUTPUT / "summary.json"
    summary = json.loads(path.read_text())
    rows = summary["items"]
    if [row["image_id"] for row in rows] != list(CASES):
        raise ValueError("Unexpected output identities")
    for row in rows:
        image_id = row["image_id"]
        source = source_for(metrics[image_id], index[image_id], manifests[image_id])
        before = sha256(source)
        if before != row["source_sha256_before"]:
            raise ValueError(f"Source SHA changed: {image_id}")
        with Image.open(source) as opened:
            if opened.mode != row["source_mode"]:
                raise ValueError(f"Source mode changed: {image_id}")
            image = opened.copy()
        first = checked_box(metrics[image_id]["crop_box_half_open"], *image.size)
        second, fallback = exact_zero_trim(np.asarray(image), first)
        if list(first) != list(row["first_box_half_open"]) or list(second) != list(row["new_box_half_open"]) or fallback != row["all_black_fallback"]:
            raise ValueError(f"Box mismatch: {image_id}")
        for key, expected in (("first_raw", image.crop(first)), ("second_raw", image.crop(second)),
                              ("new768", letterbox(image.crop(second), TARGET)[0])):
            with Image.open(OUTPUT / row[key]) as saved:
                if saved.size != expected.size or saved.mode != expected.mode or saved.tobytes() != expected.tobytes():
                    raise ValueError(f"Saved output mismatch: {image_id}/{key}")
        if sha256(source) != before:
            raise ValueError(f"Source changed during verification: {image_id}")
    current_sha = sha256(Path(__file__))
    prior_sha = summary["provenance"]["renderer_sha256"]
    summary["provenance"]["renderer_sha256"] = current_sha
    summary["provenance"]["final_source_recomputed_and_pixel_verified"] = True
    summary["provenance"]["rendering_note"] = "PNG files were first rendered by an earlier local revision; this final source recomputed all five boxes and pixel-compared first raw, second raw and new768 outputs against the saved PNGs."
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    index_path = OUTPUT / "index.md"
    index_path.write_text(index_path.read_text().replace(f"renderer SHA256: `{prior_sha}`", f"renderer SHA256: `{current_sha}`"))


def main():
    metrics, index, manifests, manifest_shas, prior = load_inputs()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for image_id in CASES:
        row = render(metrics[image_id], index[image_id], manifests[image_id])
        rows.append(row)
        print(f"{image_id}: {row['first_box_half_open']} -> {row['new_box_half_open']} | {row['comparison']}", flush=True)
    summary = {
        "scope": "Only five fixed selected train/valid gallery cases; no training, model, test or deployment use",
        "algorithm": "Original 8-bit L pixels > 0, or RGB pixels with any channel > 0; reuse saved first box; smallest half-open support bbox inside first crop; all-black fallback to first box; no threshold, erasure or detector rerun",
        "target": TARGET,
        "provenance": {"metrics_sha256": METRICS_SHA, "index_sha256": INDEX_SHA,
                       "detector_sha256": DETECTOR_SHA, "gallery_renderer_sha256": prior["provenance"]["runner_sha256"],
                       "renderer_sha256": sha256(Path(__file__)), "manifest_sha256": manifest_shas,
                       "first_crop_config": prior["provenance"]["detector_config"],
                       "letterbox_function": "tight_crop.letterbox; RGB LANCZOS, isotropic scale, black padding"},
        "items": rows,
    }
    create_overview(rows)
    summary["overview"] = "overview_5cases.png"
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    lines = ["# Exact-black trim: five fixed cases", "", "[Five-case 768 overview](overview_5cases.png)", "",
             "Experimental visual review only. Trim is computed on original grayscale pixels inside the saved first crop, before 768 letterbox. Any nonzero pixel is retained. No model, training, test, or production preprocessing.", "",
             f"Metrics SHA256: `{METRICS_SHA}`; detector SHA256: `{DETECTOR_SHA}`; renderer SHA256: `{summary['provenance']['renderer_sha256']}`.", ""]
    for row in rows:
        lines.extend((f"## {row['image_id']} ({row['view']})", "",
                      f"{'NO EXTRA TRIM' if row['no_extra_trim'] else 'EXACT ZERO TRIM'} | `{row['first_box_half_open']}` -> `{row['new_box_half_open']}` | extra pixels `{row['extra_removed_pixels_by_side']}` | scale {row['linear_magnification_new_over_first']:.4f}x | discarded nonzero {row['discarded_nonzero_count']}", "",
                      f"[Comparison]({row['comparison']}) | [First original pixels]({row['first_raw']}) | [Second original pixels]({row['second_raw']}) | [New 768]({row['new768']})", ""))
    (OUTPUT / "index.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
