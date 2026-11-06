"""Summarize completed 043/045 previews and optional 044; stdlib, read-only.

Usage: python summarize_study_batch_previews.py --preview-045 /tmp/.../preview.json
       --preview-043 /tmp/.../preview.json [--preview-044 /tmp/.../preview.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED = {
    "045_X1_B_full_decode": "09f2c96debb6afbb83e5a99c4c1e80b5fcbb6a2a87c343eb4c70c32659ec563d",
    "044_A_B_draft768": "f722b9c3e0e62ac0a675c49163dfac58bc1b52481ded8f2e9e05c1602c2fe92d",
    "043_base_B": "ec8d155fdf1b4a8ca5de7c914ab098efa5d75c449f3a986e950a811d0c632011",
}
EXPECTED_DECODER = {
    "045_X1_B_full_decode": "full_decode_exact_reference_v1",
    "044_A_B_draft768": "JPEG_draft_target768",
    "043_base_B": "full_decode_exact_reference_v1",
}
ROUTES = ("serial_image_reference", "whole_study_batch")
COMPARISON_FIELDS = (
    "binary_mismatched_studies", "sample_class_order_mismatched_studies",
    "cross_study_class_order_mismatched_studies", "probability_mismatched_studies",
    "missing_result_studies", "unexpected_result_studies",
    "missing_reference_studies", "invalid_probability_studies",
)


def finite_positive(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be finite and positive")
    return value


def load_preview(path, expected_model):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("model") != expected_model
            or data.get("config_sha256") != EXPECTED[expected_model]
            or data.get("decoder") != EXPECTED_DECODER[expected_model]
            or data.get("official_cost_certification") is not False):
        raise ValueError(f"{path}: wrong frozen model or preview scope")
    routes = data.get("routes", {})
    comparisons = data.get("comparisons", [])
    if (data.get("rounds") != 3 or any(len(routes.get(name, [])) != 3 for name in ROUTES)
            or len(comparisons) != 3):
        raise ValueError(f"{path}: require three complete serial/batch rounds")
    if not isinstance(data.get("environment", {}).get("model_inventory_bytes"), int):
        raise ValueError(f"{path}: missing measured model inventory size")
    return path, data


def round_pairs(serial, batch, expected_studies, expected_images):
    serial_rows, batch_rows = serial["per_study"], batch["per_study"]
    if len(serial_rows) != expected_studies or len(batch_rows) != expected_studies:
        raise ValueError("Incomplete per-Study timing rows")
    if sum(row["images"] for row in serial_rows) != expected_images:
        raise ValueError("Per-Study image total differs from input summary")
    if [(r["study"], r["images"]) for r in serial_rows] != [
            (r["study"], r["images"]) for r in batch_rows]:
        raise ValueError("Serial and batch timings cover different Studies or view counts")
    if len({r["study"] for r in serial_rows}) != expected_studies:
        raise ValueError("Duplicate Study timing")
    return [(r["images"], finite_positive(r["milliseconds"], "serial Study ms"),
             finite_positive(b["milliseconds"], "batch Study ms"))
            for r, b in zip(serial_rows, batch_rows)]


def comparison_summary(comparison):
    if not isinstance(comparison, dict) or comparison.get("identity_coverage_match") is not True:
        raise ValueError("Preview comparison has incomplete identity coverage")
    value = comparison.get("max_absolute_error")
    if value is not None and (not isinstance(value, (int, float)) or value < 0
                              or not math.isfinite(float(value))):
        raise ValueError("Invalid maximum absolute error")
    result = {"max_absolute_error": value,
              "semantic_match": comparison.get("semantic_match"),
              "tolerance_match": comparison.get("match")}
    for field in COMPARISON_FIELDS:
        names = comparison.get(field)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"Missing comparison field: {field}")
        result[field.removesuffix("_studies") + "_count"] = len(names)
    order = (set(comparison["sample_class_order_mismatched_studies"])
             | set(comparison["cross_study_class_order_mismatched_studies"]))
    result["any_order_changed_count"] = len(order)
    return result


def summarize(path, data):
    observed = []
    rounds = []
    for index in range(3):
        serial, batch = (data["routes"][name][index] for name in ROUTES)
        pairs = round_pairs(serial, batch, data["measured_studies"], data["measured_images"])
        observed.extend(pairs)
        serial_sum = sum(s for _, s, _ in pairs)
        batch_sum = sum(b for _, _, b in pairs)
        comparisons = data["comparisons"][index]
        rounds.append({
            "round": index + 1,
            "serial_mean_ms_per_study": serial_sum / len(pairs),
            "batch_mean_ms_per_study": batch_sum / len(pairs),
            "speedup": serial_sum / batch_sum,
            "saved_ms_all_studies": serial_sum - batch_sum,
            "image_comparison": comparison_summary(comparisons["image"]),
            "study_comparison": comparison_summary(comparisons["study"]),
        })
    serial_total = sum(s for _, s, _ in observed)
    batch_total = sum(b for _, _, b in observed)
    by_size = {}
    for count in sorted({n for n, _, _ in observed}):
        rows = [(s, b) for n, s, b in observed if n == count]
        serial = sum(s for s, _ in rows)
        batch = sum(b for _, b in rows)
        by_size[str(count)] = {
            "study_round_observations": len(rows),
            "serial_mean_ms_per_study": serial / len(rows),
            "batch_mean_ms_per_study": batch / len(rows),
            "speedup": serial / batch,
            "saved_ms_across_three_rounds": serial - batch,
        }
    max_errors = {}
    for level in ("image", "study"):
        values = [round_[f"{level}_comparison"]["max_absolute_error"] for round_ in rounds]
        max_errors[level] = None if any(value is None for value in values) else max(values)
    return {
        "preview_json": str(path), "model": data["model"],
        "config_sha256": data["config_sha256"], "decoder": data["decoder"],
        "environment": data["environment"],
        "measured_studies": data["measured_studies"],
        "measured_images": data["measured_images"],
        "sample_study_signature": [(row["study"], row["images"]) for row in
                                   data["routes"]["serial_image_reference"][0]["per_study"]],
        "rounds": rounds,
        "three_rounds": {
            "study_round_observations": len(observed),
            "serial_mean_ms_per_study": serial_total / len(observed),
            "batch_mean_ms_per_study": batch_total / len(observed),
            "speedup": serial_total / batch_total,
            "saved_ms_across_three_rounds": serial_total - batch_total,
            "mean_saved_ms_per_round": (serial_total - batch_total) / 3,
            "max_absolute_error": max_errors,
            "by_images_per_study": by_size,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-045", type=Path, required=True)
    parser.add_argument("--preview-043", type=Path, required=True)
    parser.add_argument("--preview-044", type=Path)
    args = parser.parse_args()
    results = {}
    inputs = [("045_X1_B_full_decode", args.preview_045),
              ("043_base_B", args.preview_043)]
    if args.preview_044 is not None:
        inputs.append(("044_A_B_draft768", args.preview_044))
    for model, path in inputs:
        resolved, data = load_preview(path, model)
        results[model] = summarize(resolved, data)
    if len({(result["measured_studies"], result["measured_images"],
             tuple(tuple(row) for row in result["sample_study_signature"]))
            for result in results.values()}) != 1:
        raise ValueError("The previews used different sample Study identities or counts")
    print(json.dumps({
        "scope": "Local Study-batch previews; not official cost certification",
        "model_difference": {
            "045": "Fixed X1 Tiny + B Base; class 1 B only, other classes equal probability blend",
            "044": "Fixed A Tiny + B Base; class 1 B only, other classes equal blend; JPEG draft target 768 decoder",
            "043": "Original B Base alone; all classes use its probability",
            "shared": "768 input, FP16 and fixed hflip2; 043/045 full decode, 044 JPEG draft decoder",
        },
        "models": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
