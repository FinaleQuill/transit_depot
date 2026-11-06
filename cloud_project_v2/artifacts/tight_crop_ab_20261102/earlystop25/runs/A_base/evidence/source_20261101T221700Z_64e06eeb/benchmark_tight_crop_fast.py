"""Fixed-train comparison of draft-pixel crop against frozen full-pixel crop."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from tight_crop import TightCropConfig, detect_tight_crop
from tight_crop_fast import analysis_from_draft, detect_prepared, draft_crop_box


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "artifacts/geometry_review_20261102/index.json"
OUTPUT = ROOT / "artifacts/tight_crop_ab_20261102/latency_preflight"
INDEX_SHA = "4dccd0d13893068470ea2a5ef98ab24cfc1ad4880fab1f62b374a3ef8afacbeb"
DETECTOR_SHA = "85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(fn):
    start = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - start) / 1e6


def read_draft(path):
    with Image.open(path) as image:
        original_size = image.size
        if image.format == "JPEG" and image.mode in ("L", "RGB"):
            image.draft(image.mode, (768, 768))
        image.load()
        ImageOps.exif_transpose(image, in_place=True)
        if image.mode in ("I", "F") or image.mode.startswith("I;"):
            raise ValueError("Unsupported non-8-bit image")
        return image, original_size


def full_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean_ms": float(a.mean()),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max())}


def extra_removed_stats(gray, reference_box, candidate_box):
    height, width = gray.shape
    rx0, ry0, rx1, ry1 = reference_box
    cx0, cy0, cx1, cy1 = candidate_box
    inside_reference = np.zeros((height, width), dtype=bool)
    inside_reference[ry0:ry1, rx0:rx1] = True
    inside_candidate = np.zeros((height, width), dtype=bool)
    inside_candidate[cy0:cy1, cx0:cx1] = True
    lost = gray[inside_reference & ~inside_candidate]
    return {"pixels": int(lost.size), "nonzero_pixels": int(np.count_nonzero(lost)),
            "ge_16_pixels": int(np.count_nonzero(lost >= 16)),
            "ge_128_pixels": int(np.count_nonzero(lost >= 128)),
            "max_intensity": int(lost.max()) if lost.size else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cv2-threads", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()
    if args.repeats < 2 or sha256(INDEX) != INDEX_SHA or sha256(ROOT / "tight_crop.py") != DETECTOR_SHA:
        raise ValueError("Invalid repeats or frozen source SHA")
    cv2.setNumThreads(args.cv2_threads)
    index = json.loads(INDEX.read_text())
    selected = []
    for view in ("AP", "PA", "Lateral", "Missing"):
        group = [x for x in index["items"] if x["split"] == "train" and x["view"] == view]
        if len(group) != 12:
            raise ValueError("Unexpected group")
        selected.extend(group[:4])
    rows = []
    for item in selected:
        path = Path(item["source_path"])
        if sha256(path) != item["source_sha256"]:
            raise ValueError("Source SHA before timing differs")
        original = full_gray(path)
        reference = detect_tight_crop(original)
        full_analysis = cv2.resize(original, reference.analysis_gray.shape[::-1],
                                   interpolation=cv2.INTER_AREA)
        replay = detect_prepared(full_analysis, original.shape[::-1],
                                 original_range=int(original.max()) - int(original.min()))
        if (replay.box != reference.box or replay.body_box != reference.body_box
                or not np.array_equal(replay.mask, reference.mask)):
            raise ValueError(f"Prepared full replay differs: {item['image_id']}")
        draft, original_size = read_draft(path)
        if original_size != tuple(item["original_size"]):
            raise ValueError("Original size mismatch")
        small = analysis_from_draft(draft)
        candidate = detect_prepared(small, original_size)
        draft_box = draft_crop_box(candidate.box, original_size, draft.size)
        if draft_box[0] >= draft_box[2] or draft_box[1] >= draft_box[3]:
            raise ValueError("Empty draft box")
        draft.crop(draft_box)
        times = defaultdict(list)
        for _ in range(args.repeats):
            (_, _), ms = measure(lambda: read_draft(path))
            times["draft_decode"].append(ms)
            start = time.perf_counter_ns()
            small, ms = measure(lambda: analysis_from_draft(draft))
            times["draft_to_analysis"].append(ms)
            detected, ms = measure(lambda: detect_prepared(small, original_size))
            times["geometry"].append(ms)
            box = draft_crop_box(detected.box, original_size, draft.size)
            _, ms = measure(lambda: draft.crop(box))
            times["crop_draft_pixels"].append(ms)
            if detected.box != candidate.box:
                raise ValueError("Unstable candidate box")
            increment = (time.perf_counter_ns() - start) / 1e6
            times["added_cpu"].append(increment)
            times["draft_decode_plus_added_cpu"].append(times["draft_decode"][-1] + increment)
        rows.append({"image_id": item["image_id"], "view": item["view"],
                     "source_sha256": item["source_sha256"],
                     "original_size": original_size, "draft_size": draft.size,
                     "reference_analysis_size": reference.analysis_gray.shape[::-1],
                     "draft_analysis_size": small.shape[::-1],
                     "reference_box": reference.box, "candidate_box": candidate.box,
                     "draft_crop_box": draft_box, "reference_applied": reference.diagnostics["applied"],
                     "candidate_applied": candidate.diagnostics["applied"],
                     "extra_removed_vs_reference": extra_removed_stats(original, reference.box, candidate.box),
                     "timings_ms": dict(times)})
        print(f"{item['image_id']} added={statistics.mean(times['added_cpu']):.2f}ms "
              f"box_equal={reference.box == candidate.box}", flush=True)
    for item in selected:
        if sha256(Path(item["source_path"])) != item["source_sha256"]:
            raise ValueError("Source SHA after timing differs")
    keys = tuple(rows[0]["timings_ms"])
    aggregate = {key: stats([v for row in rows for v in row["timings_ms"][key]]) for key in keys}
    report = {"scope": "16 fixed train images; no valid/test; no model or GPU",
              "provenance": {"index_sha256": INDEX_SHA, "reference_sha256": DETECTOR_SHA,
                             "fast_module_sha256": sha256(ROOT / "tight_crop_fast.py"),
                             "runner_sha256": sha256(Path(__file__)),
                             "cv2_threads": cv2.getNumThreads(), "repeats": args.repeats},
              "aggregate": aggregate, "items": rows,
              "caveats": ["draft analysis pixels differ from full original pixels",
                          "source-pixel removal counts are mechanical, not medical safety claims",
                          "crop operation is PIL crop only, not downstream LANCZOS letterbox",
                          "historical 84.82ms end-to-end GPU baseline is from a separate run"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"fast_report_threads{args.cv2_threads}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target}")
    for key, value in aggregate.items():
        print(f"{key}: mean={value['mean_ms']:.3f} p95={value['p95_ms']:.3f} ms")
    print("box_equal", sum(row["reference_box"] == row["candidate_box"] for row in rows), "/", len(rows))
    print("extra_removed_nonzero", sum(row["extra_removed_vs_reference"]["nonzero_pixels"] for row in rows))


if __name__ == "__main__":
    main()
