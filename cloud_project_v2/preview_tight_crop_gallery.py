"""Frozen 96-image geometry gallery; visual review only, never model input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tight_crop import detect_tight_crop, letterbox


PROJECT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/home/featurize/data/cxr_first")
INDEX = PROJECT / "artifacts/geometry_review_20261102/index.json"
OUTPUT = PROJECT / "artifacts/tight_crop_gallery_20261102"
MANIFESTS = {split: PROJECT / f"data/clean_v1_reviewed/{name}" for split, name in
             (("train", "train_clean.csv"), ("valid", "valid_manifest.csv"))}
INDEX_SHA = "4dccd0d13893068470ea2a5ef98ab24cfc1ad4880fab1f62b374a3ef8afacbeb"
DETECTOR_SHA = "85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d"
TARGET = 768
CYAN, RED, GREEN = (24, 224, 235), (255, 72, 65), (73, 230, 124)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simple(value):
    if isinstance(value, dict):
        return {str(k): simple(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [simple(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_inputs():
    if sha256(INDEX) != INDEX_SHA:
        raise ValueError("Frozen source index SHA changed")
    if sha256(PROJECT / "tight_crop.py") != DETECTOR_SHA:
        raise ValueError("Frozen detector SHA changed")
    source_index = json.loads(INDEX.read_text())
    items = source_index["items"]
    if not source_index["complete"] or len(items) != 96 or source_index["count"] != 96:
        raise ValueError("Expected the complete frozen 96-image index")
    groups = Counter((a["split"], a["view"]) for a in items)
    expected = {(split, view): 12 for split in MANIFESTS for view in ("AP", "PA", "Lateral", "Missing")}
    if groups != expected or len({a["image_id"] for a in items}) != 96:
        raise ValueError(f"Unexpected index composition: {groups}")
    selected = {a["image_id"] for a in items}
    manifests = {}
    manifest_shas = {}
    for split, path in MANIFESTS.items():
        manifest_shas[split] = sha256(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["image_id"] in selected:
                    if row["image_id"] in manifests:
                        raise ValueError(f"Duplicate manifest identity: {row['image_id']}")
                    if row["split"] != split:
                        raise ValueError(f"Manifest split mismatch: {row['image_id']}")
                    manifests[row["image_id"]] = row
    if set(manifests) != selected:
        raise ValueError(f"Index/manifest IDs differ: {selected ^ set(manifests)}")
    return items, manifests, manifest_shas


def source_for(item, row):
    split = item["split"]
    relative = Path(item["relative_path"])
    if (item["image_id"] != row["image_id"] or not item["image_id"].startswith(split + ":")
            or row["relative_path"] != item["relative_path"] or relative.is_absolute()
            or ".." in relative.parts or relative.parts[0] != split):
        raise ValueError(f"Index/manifest path mismatch: {item['image_id']}")
    root = (SOURCE_ROOT / split).resolve(strict=True)
    source = (SOURCE_ROOT / relative).resolve(strict=True)
    if not source.is_relative_to(root) or source != Path(item["source_path"]).resolve(strict=True):
        raise ValueError(f"Source outside indexed split: {item['image_id']}")
    view = row["view_position"]
    group = "Lateral" if view.upper() in ("LATERAL", "LL") else view or "Missing"
    if item["view"] != group:
        raise ValueError(f"View mismatch: {item['image_id']}")
    if item["source_sha256"] != row["file_sha256"]:
        raise ValueError(f"Index/manifest SHA mismatch: {item['image_id']}")
    return source


def box4(box, w, h):
    if len(box) != 4 or any(int(v) != v for v in box):
        raise ValueError(f"Invalid box: {box}")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"Box outside image: {box}")
    return x0, y0, x1, y1


def fit(image, size=420, resample=Image.Resampling.LANCZOS):
    scale = min(size / image.width, size / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    offset = ((size - width) // 2, (size - height) // 2)
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(image.convert("RGB").resize((width, height), resample), offset)
    return panel, scale, offset


def mark_box(draw, box, scale, offset, color):
    x0, y0, x1, y1 = box
    ox, oy = offset
    draw.rectangle((ox + round(x0 * scale), oy + round(y0 * scale),
                    ox + round((x1 - 1) * scale), oy + round((y1 - 1) * scale)),
                   outline=color, width=3)


def composite(image, proposal, box, body, baseline, candidate, item, status):
    side, gap, top, bottom = 420, 16, 106, 38
    canvas = Image.new("RGB", (3 * side + 4 * gap, 2 * side + top + bottom + gap), (29, 34, 39))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((gap, 8), f"{item['split']} / {item['view']} / {item['image_id']} | {status}", fill="white", font=font)
    draw.text((gap, 39), "Crop proposal, not anatomy segmentation. Black border does not imply expendable tissue.", fill="white", font=small)
    draw.text((gap, 66), "Cyan: original frame  Red: proposed crop  Green: body bound. Baseline uses isotropic letterbox.", fill="white", font=small)
    original, scale, offset = fit(image)
    seed, _, _ = fit(Image.fromarray(np.uint8(proposal.seed_mask > 0) * 255), resample=Image.Resampling.NEAREST)
    mask, _, _ = fit(Image.fromarray(np.uint8(proposal.mask > 0) * 255), resample=Image.Resampling.NEAREST)
    overlay = original.copy()
    outline = ImageDraw.Draw(overlay)
    mark_box(outline, (0, 0, image.width, image.height), scale, offset, CYAN)
    mark_box(outline, box, scale, offset, RED)
    mark_box(outline, body, scale, offset, GREEN)
    baseline_tile, _, _ = fit(baseline)
    candidate_tile, _, _ = fit(candidate)
    tiles = (original, seed, mask, overlay, baseline_tile, candidate_tile)
    names = ("Original full frame", "Bright seed (analysis)", "Support mask (analysis)",
             "Proposal on original", "Full-view 768 baseline", "New crop 768, no erasing")
    for index, (tile, name) in enumerate(zip(tiles, names)):
        col, row = index % 3, index // 3
        x, y = gap + col * (side + gap), top + row * (side + bottom + gap)
        canvas.paste(tile, (x, y))
        draw.text((x, y + side + 6), name, fill="white", font=small)
    return canvas


def strips(image, box, folder):
    w, h = image.size
    x0, y0, x1, y1 = box
    cuts = {"left": x0, "top": y0, "right": w - x1, "bottom": h - y1}
    files = []
    for side, cut in cuts.items():
        if cut <= 0:
            continue
        reach = min(160, (w if side in ("left", "right") else h) - cut)
        coords = {"left": (0, 0, x0 + reach, h),
                  "top": (0, 0, w, y0 + reach),
                  "right": (x1 - reach, 0, w, h),
                  "bottom": (0, y1 - reach, w, h)}[side]
        path = folder / f"{side}_raw_1to1.png"
        image.crop(coords).save(path)
        files.append({"side": side, "cut_pixels": cut, "source_box": coords,
                      "path": str(path.relative_to(OUTPUT))})
    return files


def render(item, row):
    source = source_for(item, row)
    before = sha256(source)
    if before != item["source_sha256"]:
        raise ValueError(f"Source SHA mismatch: {item['image_id']}")
    with Image.open(source) as opened:
        image = opened.convert("L")
    w, h = image.size
    if list(image.size) != item["original_size"] or image.size != (int(row["width"]), int(row["height"])):
        raise ValueError(f"Source size mismatch: {item['image_id']}")
    proposal = detect_tight_crop(np.asarray(image))
    box, body = box4(proposal.box, w, h), box4(proposal.body_box, w, h)
    baseline, base_meta = letterbox(image, TARGET)
    candidate, crop_meta = letterbox(image.crop(box), TARGET)
    is_no_cut = box == (0, 0, w, h)
    if is_no_cut and baseline.tobytes() != candidate.tobytes():
        raise ValueError(f"NO CUT changed pixels: {item['image_id']}")
    status = "NO CUT / full-view fallback" if is_no_cut else "new crop candidate"
    folder = OUTPUT / item["split"] / item["view"] / item["image_id"].replace(":", "_")
    folder.mkdir(parents=True, exist_ok=False)
    composite(image, proposal, box, body, baseline, candidate, item, status).save(
        folder / "comparison.jpg", quality=91, subsampling=0)
    candidate.save(folder / "crop768.png")
    raw = strips(image, box, folder)
    after = sha256(source)
    if after != before:
        raise ValueError(f"Source changed during render: {item['image_id']}")
    cuts = {"left": box[0], "top": box[1], "right": w - box[2], "bottom": h - box[3]}
    return {"image_id": item["image_id"], "split": item["split"], "view": item["view"],
            "selection_reason": item["selection_reason"], "source_path": str(source),
            "source_sha256_index_manifest_before_after": before,
            "original_size": [w, h], "crop_box_half_open": box, "body_box_half_open": body,
            "crop_size": [box[2] - box[0], box[3] - box[1]],
            "removed_pixels_by_side": cuts,
            "removed_fraction_by_side": {side: count / (w if side in ("left", "right") else h)
                                         for side, count in cuts.items()},
            "removed_area_fraction": 1 - (box[2]-box[0]) * (box[3]-box[1]) / (w*h),
            "baseline_768_scale": base_meta["scale"], "candidate_768_scale": crop_meta["scale"],
            "effective_linear_magnification": crop_meta["scale"] / base_meta["scale"],
            "baseline_letterbox": simple(base_meta), "candidate_letterbox": simple(crop_meta),
            "applied": bool(proposal.diagnostics["applied"]),
            "reason": proposal.diagnostics["reason"], "no_cut": is_no_cut,
            "diagnostics": simple(proposal.diagnostics),
            "comparison": str((folder / "comparison.jpg").relative_to(OUTPUT)),
            "crop768": str((folder / "crop768.png").relative_to(OUTPUT)),
            "raw_strips": raw}


def write_reports(records, manifest_shas):
    provenance = {"index_path": str(INDEX), "index_sha256": INDEX_SHA,
                  "manifest_sha256": manifest_shas, "detector_sha256": DETECTOR_SHA,
                  "runner_sha256": sha256(Path(__file__)), "target_size": TARGET,
                  "detector_config": records[0]["diagnostics"]["config"] if records else None}
    report = {"scope": "Frozen, selected 96-image gallery only; not a full-dataset estimate.",
              "count": len(records), "complete": len(records) == 96,
              "provenance": provenance, "items": records}
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    fields = ("image_id", "split", "view", "selection_reason", "source_sha256_index_manifest_before_after",
              "original_size", "crop_box_half_open", "body_box_half_open", "crop_size",
              "removed_pixels_by_side", "removed_fraction_by_side", "removed_area_fraction",
              "baseline_768_scale", "candidate_768_scale", "effective_linear_magnification",
              "applied", "reason", "no_cut", "diagnostics", "comparison", "crop768", "raw_strips")
    with (OUTPUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key], ensure_ascii=True) if isinstance(record[key], (list, dict, tuple))
                             else record[key] for key in fields})
    counts = Counter((r["split"], r["view"], r["applied"]) for r in records)
    lines = ["# Tight crop frozen 96-image gallery", "",
             "Selected review cases only. These counts do not estimate full train or valid behavior.",
             "Crop proposals are not anatomy segmentation; black borders do not imply expendable tissue.",
             "Full-view baseline and candidate use isotropic 768 letterbox. No erasing, training, or model inference.", "",
             f"Index SHA256: `{INDEX_SHA}`; detector SHA256: `{DETECTOR_SHA}`; runner SHA256: `{provenance['runner_sha256']}`.", "",
             f"Progress: {len(records)}/96; complete: {report['complete']}.", "",
             "## Selected-case counts", ""]
    for (split, view, applied), count in sorted(counts.items()):
        lines.append(f"- {split}/{view}, {'candidate' if applied else 'fallback'}: {count}")
    lines.append("")
    for r in records:
        lines += [f"## {r['split']} / {r['view']} / {r['image_id']}", "",
                  f"{r['reason']} | box `{r['crop_box_half_open']}` | removed area {r['removed_area_fraction']:.4%} | linear scale {r['effective_linear_magnification']:.4f}x", "",
                  f"[Six-panel comparison]({r['comparison']}) | [New 768]({r['crop768']})" +
                  "".join(f" | [Raw {s['side']} 1:1]({s['path']})" for s in r["raw_strips"]), ""]
    (OUTPUT / "index.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--first-12", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    items, manifests, manifest_shas = load_inputs()
    if args.first_12:
        if OUTPUT.exists():
            raise FileExistsError(f"Gallery exists: {OUTPUT}")
        OUTPUT.mkdir(parents=True)
        records = []
        pending = items[:12]
    else:
        report = json.loads((OUTPUT / "metrics.json").read_text())
        if (report["provenance"]["index_sha256"] != INDEX_SHA or
                report["provenance"]["detector_sha256"] != DETECTOR_SHA or
                report["provenance"]["manifest_sha256"] != manifest_shas or
                report["provenance"]["runner_sha256"] != sha256(Path(__file__))):
            raise ValueError("Gallery provenance changed during resume")
        records = report["items"]
        if [r["image_id"] for r in records] != [a["image_id"] for a in items[:len(records)]]:
            raise ValueError("Gallery progress is not an index prefix")
        pending = items[len(records):]
    for item in pending:
        records.append(render(item, manifests[item["image_id"]]))
    write_reports(records, manifest_shas)
    print(f"{len(records)}/96: {OUTPUT / 'index.md'}")


if __name__ == "__main__":
    main()
