"""Train-only deterministic scale-normalization atlas; no inference/training.

The selection deliberately overrepresents geometric and detector extremes.  Its
applied fraction is not a dataset prevalence estimate.  Labels are never used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from roi_scale import normalize_roi, VERSION
from tight_crop import letterbox


PROJECT = Path(__file__).resolve().parent
VIEWS = ("AP", "PA", "Lateral", "Missing")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def view_group(value):
    value = value.strip().upper()
    if value in ("LATERAL", "LL"):
        return "Lateral"
    if value in ("AP", "PA"):
        return value
    return "Missing"


def sample_key(row):
    return hashlib.sha256((VERSION+":"+row["image_id"]).encode()).hexdigest()


def source_for(row, source_root):
    relative = Path(row["relative_path"])
    if row["split"] != "train" or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Atlas must only use safe train paths")
    if len(relative.parts) != 4 or relative.parts[0] != "train":
        raise ValueError("Invalid train path layout")
    source = (source_root/relative).resolve(strict=True)
    if not source.is_relative_to((source_root/"train").resolve(strict=True)):
        raise ValueError("Source escaped the train root")
    return source


def decode(row, source_root):
    source = source_for(row, source_root)
    before = sha256(source)
    if before != row["file_sha256"]:
        raise ValueError(f"Source SHA mismatch: {row['image_id']}")
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if sha256(source) != before:
        raise ValueError("Read-only source changed")
    return image, source, before


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def geometric_pool(rows, count):
    unique = {}
    orders = [sorted(rows, key=lambda r: int(r["width"])/int(r["height"])),
              sorted(rows, key=lambda r: int(r["width"])*int(r["height"])),
              sorted(rows, key=lambda r: int(r["width"])),
              sorted(rows, key=lambda r: int(r["height"]))]
    for order in orders:
        for row in order[:3]+order[-3:]:
            unique[row["image_id"]] = row
    for row in sorted(rows, key=sample_key)[:count]:
        unique[row["image_id"]] = row
    return sorted(unique.values(), key=lambda r: r["image_id"])


def select_cases(candidates, count):
    chosen = {}

    def add_order(reason, order):
        for value in order:
            image_id = value["row"]["image_id"]
            if image_id not in chosen:
                chosen[image_id] = {**value, "selection_reason": reason}
                return

    reliable = [c for c in candidates if c["diagnostics"]["applied"]]
    fallback = [c for c in candidates if not c["diagnostics"]["applied"]]
    for reason, order in (
        ("lowest_effective_magnification", sorted(reliable, key=lambda c: c["diagnostics"]["effective_magnification"])),
        ("highest_effective_magnification", sorted(reliable, key=lambda c: -c["diagnostics"]["effective_magnification"])),
        ("widest_frame", sorted(candidates, key=lambda c: -int(c["row"]["width"])/int(c["row"]["height"]))),
        ("tallest_frame", sorted(candidates, key=lambda c: int(c["row"]["width"])/int(c["row"]["height"]))),
        ("smallest_native_image", sorted(candidates, key=lambda c: int(c["row"]["width"])*int(c["row"]["height"]))),
        ("largest_native_image", sorted(candidates, key=lambda c: -int(c["row"]["width"])*int(c["row"]["height"]))),
        ("largest_removed_source_area", sorted(reliable, key=lambda c: -c["diagnostics"]["source_removed_area_fraction"])),
        ("source_edge_contact_veto", sorted([c for c in fallback if c["diagnostics"].get("source_edge_contact")], key=lambda c: sample_key(c["row"]))),
        ("no_cut_fallback", sorted(fallback, key=lambda c: sample_key(c["row"]))),
    ):
        add_order(reason, order)
    for c in sorted(candidates, key=lambda c: sample_key(c["row"])):
        if len(chosen) >= count:
            break
        image_id = c["row"]["image_id"]
        if image_id not in chosen:
            chosen[image_id] = {**c, "selection_reason": "deterministic_hash_sample"}
    return list(chosen.values())[:count]


def comparison(source, baseline, candidate, metadata, label, side=512):
    gap, head, foot = 14, 84, 40
    canvas = Image.new("RGB", (4*side+5*gap, side+head+foot), (28, 32, 37))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), label, fill="white", font=font(18))
    draw.text((gap, 34), f"{metadata['reason']} | effective magnification {metadata['effective_magnification']:.3f}x", fill="white", font=font(16))
    original, original_meta = letterbox(source, side)
    overlay = original.copy()
    marks = ImageDraw.Draw(overlay)
    ox, oy = original_meta["content_box"][:2]
    scale = original_meta["scale"]
    for key, color in (("crop_box", (255, 80, 70)), ("proposed_crop_box", (255, 80, 70)),
                       ("body_box", (60, 230, 130))):
        if key not in metadata:
            continue
        x0, y0, x1, y1 = metadata[key]
        marks.rectangle((ox+x0*scale, oy+y0*scale, ox+(x1-1)*scale, oy+(y1-1)*scale), outline=color, width=3)
    tiles = (original, overlay, baseline.resize((side, side), Image.Resampling.LANCZOS), candidate.resize((side, side), Image.Resampling.LANCZOS))
    titles = ("Full original", "Green body / red accepted or proposed crop", "Full-frame letterbox", "Magnify-only crop / exact NO CUT fallback")
    for i, (tile, title) in enumerate(zip(tiles, titles)):
        x = gap+i*(side+gap)
        canvas.paste(tile, (x, head))
        draw.text((x, head+side+9), title, fill="white", font=font(13))
    return canvas


def contact_sheet(items, output, view):
    side, gap, header, label = 256, 12, 50, 52
    pair_width = side*2+gap
    canvas = Image.new("RGB", (3*pair_width+4*gap, 4*(side+label)+header+gap), (28, 32, 37))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"TRAIN {view}: original letterbox | normalized (12 deliberately varied cases)", fill="white", font=font(20))
    for i, item in enumerate(items):
        x, y = gap+(i%3)*(pair_width+gap), header+(i//3)*(side+label)
        with Image.open(output/item["baseline_png"]) as opened:
            baseline = opened.resize((side, side), Image.Resampling.LANCZOS)
        with Image.open(output/item["candidate_png"]) as opened:
            candidate = opened.resize((side, side), Image.Resampling.LANCZOS)
        canvas.paste(baseline, (x, y))
        canvas.paste(candidate, (x+side+gap, y))
        meta = item["diagnostics"]
        draw.text((x, y+side+4), f"{i+1:02d} {item['image_id']} {meta['effective_magnification']:.3f}x {meta['reason']}", fill="white", font=font(14))
        draw.text((x, y+side+25), item["selection_reason"], fill="white", font=font(13))
    destination = output/f"train_{view}_contact.jpg"
    canvas.save(destination, quality=95, subsampling=0)
    return destination.name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT/"data/clean_v1_reviewed/train_clean.csv")
    parser.add_argument("--source-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output", type=Path, default=PROJECT/"artifacts/roi_magnify_explore_20261103/gallery")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--pool-per-view", type=int, default=64)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing atlas")
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if any(r["split"] != "train" or r["training_eligible"].lower() != "true" for r in rows):
        raise ValueError("Atlas requires only eligible official train images")
    manifest_sha = sha256(args.manifest)
    args.output.mkdir(parents=True)
    records, pool_counts, atlases = [], {}, {}
    for view in VIEWS:
        group = [r for r in rows if view_group(r["view_position"]) == view]
        if not group:
            continue
        candidates = []
        for row in geometric_pool(group, args.pool_per_view):
            image, _, _ = decode(row, args.source_root)
            _, diagnostics = normalize_roi(image, args.size)
            candidates.append({"row": row, "diagnostics": diagnostics})
        pool_counts[view] = {"available_train_rows": len(group), "examined_pool": len(candidates),
                             "applied_in_selected_pool": sum(c["diagnostics"]["applied"] for c in candidates)}
        cases = select_cases(candidates, min(12, len(candidates)))
        view_records = []
        for index, case in enumerate(cases, 1):
            row = case["row"]
            source, path, source_sha = decode(row, args.source_root)
            candidate, metadata = normalize_roi(source, args.size)
            if metadata != case["diagnostics"]:
                raise ValueError("Geometry replay mismatch")
            baseline, _ = letterbox(source, args.size)
            folder = args.output/view/f"{index:02d}_{row['image_id'].replace(':', '_')}"
            folder.mkdir(parents=True)
            for name, image in (("baseline.png", baseline), ("normalized.png", candidate)):
                image.save(folder/name)
            comparison(source, baseline, candidate, metadata,
                       f"{view} {index:02d} {row['image_id']} | {case['selection_reason']}").save(folder/"comparison.jpg", quality=95, subsampling=0)
            record = {"image_id": row["image_id"], "split": "train", "view": view,
                      "original_view_position": row["view_position"],
                      "selection_reason": case["selection_reason"], "source_path": str(path),
                      "source_sha256_manifest_before_after": source_sha, "diagnostics": metadata,
                      "baseline_png": str((folder/"baseline.png").relative_to(args.output)),
                      "candidate_png": str((folder/"normalized.png").relative_to(args.output)),
                      "comparison": str((folder/"comparison.jpg").relative_to(args.output))}
            (folder/"metadata.json").write_text(json.dumps(record, indent=2)+"\n")
            records.append(record)
            view_records.append(record)
        atlases[view] = contact_sheet(view_records, args.output, view)
        print(f"{view}: {len(candidates)} train candidates inspected; {len(cases)} atlas cases rendered", flush=True)
    if sha256(args.manifest) != manifest_sha:
        raise ValueError("Train manifest changed")
    report = {"method_version": VERSION, "split": "train", "test_accessed": False,
              "valid_accessed": False, "training_or_model_inference": False,
              "safety_policy": {"minimum_effective_magnification": 1.01,
                  "maximum_effective_magnification": 1.25,
                  "crop_short_long_ratio_must_not_worsen_vs_source": True,
                  "minimum_crop_short_long_ratio": .55,
                  "source_edge_clipped_body_veto": True,
                  "detector_existing_seed_support_and_removed_area_gates": True},
              "selection": "deterministic metadata + detector extremes and hashed image IDs; labels not used",
              "count": len(records), "counts_by_view": dict(Counter(r["view"] for r in records)),
              "pool_counts": pool_counts, "prevalence_estimate": False,
              "manifest_path": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
              "code_sha256": {name: sha256(PROJECT/name) for name in ("roi_scale.py", "tight_crop.py", Path(__file__).name)},
              "contact_sheets": atlases, "items": records}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Train-only ROI magnify atlas", "", "12 deliberately varied images per available view; no labels selected, no valid/test read.",
             "Classical body-envelope heuristic, not a medical segmentation. Source-edge clipping is rejected, not repaired.",
             "Crop body envelope with about 2% margin per side; long side fills 768 and short side is padded isotropically.",
             "Only proposals with effective magnification 1.01-1.25 that do not worsen square padding are applied. Other cases exactly fall back to full-frame letterbox.",
             "Selected-case counts do not estimate train prevalence; no erasing, training, or model inference.", ""]
    for view, atlas in atlases.items():
        lines.extend((f"## {view}", "", f"![{view}]({atlas})", ""))
        for item in [r for r in records if r["view"] == view]:
            lines.append(f"- [{item['image_id']}: {item['selection_reason']}]({item['comparison']}) — {item['diagnostics']['reason']}")
        lines.append("")
    (args.output/"index.md").write_text("\n".join(lines)+"\n")
    print(str(args.output/"metrics.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
