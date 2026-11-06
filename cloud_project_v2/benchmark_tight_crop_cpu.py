"""Read-only CPU latency preflight for the frozen body-envelope detector."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tight_crop import detect_tight_crop


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "artifacts/geometry_review_20261102/index.json"
OUTPUT = ROOT / "artifacts/tight_crop_ab_20261102/latency_preflight"
INDEX_SHA = "4dccd0d13893068470ea2a5ef98ab24cfc1ad4880fab1f62b374a3ef8afacbeb"
DETECTOR_SHA = "85ec065e2b4a9bb23a0defcafc3681cd474a579563affd3501ddcb4d2426cd8d"
STAGES = {
    "resize": "resize512",
    "GaussianBlur": "blur_otsu",
    "threshold": "blur_otsu",
    "morphologyEx": "morphology",
    "connectedComponentsWithStats": "connected_components",
    "findContours": "contour_hull_fill",
    "drawContours": "contour_hull_fill",
    "convexHull": "contour_hull_fill",
    "boundingRect": "contour_hull_fill",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(fn):
    start = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - start) / 1e6


def full_gray(path):
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"))


def draft_decode(path):
    with Image.open(path) as opened:
        if opened.format == "JPEG" and opened.mode in ("L", "RGB"):
            opened.draft(opened.mode, (768, 768))
        opened.load()
        return opened.size


def profile_detector(gray):
    times = defaultdict(float)
    counts = defaultdict(int)
    patchers = []
    for name, stage in STAGES.items():
        original = getattr(cv2, name)

        def wrapped(*args, _original=original, _stage=stage, **kwargs):
            result, elapsed = measure(lambda: _original(*args, **kwargs))
            times[_stage] += elapsed
            counts[_stage] += 1
            return result

        patchers.append(patch.object(cv2, name, wrapped))
    try:
        for patcher in patchers:
            patcher.start()
        result, elapsed = measure(lambda: detect_tight_crop(gray))
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
    return result, elapsed, dict(times), dict(counts)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cv2-threads", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("Need at least two repeats")
    if sha256(INDEX) != INDEX_SHA or sha256(ROOT / "tight_crop.py") != DETECTOR_SHA:
        raise ValueError("Frozen index or detector SHA changed")
    cv2.setNumThreads(args.cv2_threads)
    index = json.loads(INDEX.read_text())
    if not index["complete"] or index["count"] != 96:
        raise ValueError("Incomplete frozen index")
    selected = []
    for view in ("AP", "PA", "Lateral", "Missing"):
        group = [item for item in index["items"] if item["split"] == "train" and item["view"] == view]
        if len(group) != 12:
            raise ValueError(f"Unexpected train group: {view}")
        selected.extend(group[:4])
    if len({item["image_id"] for item in selected}) != 16:
        raise ValueError("Duplicate train image")
    for item in selected:
        path = Path(item["source_path"])
        if sha256(path) != item["source_sha256"]:
            raise ValueError(f"Original SHA mismatch before timing: {item['image_id']}")

    rows = []
    for item in selected:
        path = Path(item["source_path"])
        gray = full_gray(path)
        if list(gray.shape[::-1]) != item["original_size"]:
            raise ValueError(f"Decoded shape mismatch: {item['image_id']}")
        reference = detect_tight_crop(gray)
        # Warm each path once before timed repetitions.
        draft_decode(path)
        times = defaultdict(list)
        for _ in range(args.repeats):
            size, ms = measure(lambda: draft_decode(path))
            times["draft768_decode"].append(ms)
            decoded, ms = measure(lambda: full_gray(path))
            times["full_gray_decode"].append(ms)
            if not np.array_equal(decoded, gray):
                raise ValueError(f"Unstable full decode: {item['image_id']}")
            result, ms = measure(lambda: detect_tight_crop(gray))
            times["decoded_detector"].append(ms)
            if result.box != reference.box or not np.array_equal(result.mask, reference.mask):
                raise ValueError(f"Detector changed output: {item['image_id']}")
            times["full_decode_plus_detector"].append(
                times["full_gray_decode"][-1] + times["decoded_detector"][-1])
            times["delta_reuse_full_instead_of_draft"].append(
                times["full_decode_plus_detector"][-1] - times["draft768_decode"][-1])
        profiled, profile_ms, stage_ms, counts = profile_detector(gray)
        if profiled.box != reference.box or not np.array_equal(profiled.mask, reference.mask):
            raise ValueError(f"Instrumented output differs: {item['image_id']}")
        row = {"image_id": item["image_id"], "view": item["view"],
               "source_sha256": item["source_sha256"], "original_size": item["original_size"],
               "draft_decoded_size": size, "box": reference.box,
               "applied": reference.diagnostics["applied"],
               "timings_ms": dict(times), "profile_ms": profile_ms,
               "profile_stage_ms": stage_ms, "profile_call_counts": counts,
               "profile_python_and_other_ms": profile_ms - sum(stage_ms.values())}
        rows.append(row)
        print(f"{item['image_id']} {item['view']} detector={statistics.mean(times['decoded_detector']):.2f}ms "
              f"full+detector={statistics.mean(times['full_decode_plus_detector']):.2f}ms", flush=True)

    for item in selected:
        if sha256(Path(item["source_path"])) != item["source_sha256"]:
            raise ValueError(f"Original SHA mismatch after timing: {item['image_id']}")
    keys = tuple(rows[0]["timings_ms"])
    aggregate = {key: stats([sample for row in rows for sample in row["timings_ms"][key]])
                 for key in keys}
    per_image_means = {key: stats([statistics.mean(row["timings_ms"][key]) for row in rows])
                       for key in keys}
    profile_keys = sorted(set().union(*(row["profile_stage_ms"] for row in rows)))
    stages = {key: stats([row["profile_stage_ms"].get(key, 0) for row in rows])
              for key in profile_keys}
    stages["python_and_other"] = stats([row["profile_python_and_other_ms"] for row in rows])
    report = {
        "scope": "16 fixed train images, four per view; descriptive latency only, no valid/test",
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "opencv": cv2.__version__, "pillow": Image.__version__,
                    "cv2_threads": cv2.getNumThreads()},
        "provenance": {"index_sha256": INDEX_SHA, "detector_sha256": DETECTOR_SHA,
                       "runner_sha256": sha256(Path(__file__)), "repeats": args.repeats,
                       "selection": "first four train records in each AP/PA/Lateral/Missing index group"},
        "aggregate_repetition": aggregate,
        "aggregate_per_image_mean": per_image_means,
        "profile_stages_single_instrumented_pass": stages,
        "items": rows,
        "interpretation": {
            "decoded_detector": "extra work only if identical full grayscale pixels are already available",
            "full_decode_plus_detector": "additional work if existing draft768 decoder still runs separately",
            "delta_reuse_full_instead_of_draft": "decode substitution plus detection; changes model input pixels unless separately proved equivalent",
            "profile_stages": "one instrumented detector pass per image; wrapper overhead included in residual; not additive to uninstrumented detector timing",
            "target_mean_ms": 5.18,
            "historical_2080ti_baseline_ms": 84.820141257,
            "historical_baseline_decoder": "Pillow JPEG draft768; historical GPU/model path differs from detector-only CPU process",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"report_threads{args.cv2_threads}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target}")
    for key in keys:
        print(f"{key}: mean={aggregate[key]['mean_ms']:.3f} p95={aggregate[key]['p95_ms']:.3f} ms")
    for key in profile_keys:
        print(f"stage {key}: mean={stages[key]['mean_ms']:.3f} ms")


if __name__ == "__main__":
    main()
