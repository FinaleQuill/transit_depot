"""Render final A/B review tables from saved evidence; no model or GPU imports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
NAMES = ("A_tiny", "A_base", "B_tiny", "B_base", "A_fused", "B_fused")
PAIRS = (("B_tiny", "A_tiny"), ("B_base", "A_base"), ("B_fused", "A_fused"))
SLICES = ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image")
HEADERS = {
    "all": "整体",
    "ap_only": "AP-only",
    "ap_only_single_image": "AP单图",
    "ap_only_multiple_image": "AP多图",
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def number(value, *, signed=False):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Expected a finite metric, got {value!r}")
    return f"{value:+.9f}" if signed else f"{value:.9f}"


def interval(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Missing 95% interval: {value!r}")
    return f"[{number(value[0], signed=True)}, {number(value[1], signed=True)}]"


def table(header, rows):
    return "\n".join(("| " + " | ".join(header) + " |",
                      "| " + " | ".join("---" for _ in header) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)))


def score(scores, name, subset):
    return scores[name][subset]["macro_auc"]


def comparison(review, candidate, reference):
    key = f"{candidate}_minus_{reference}"
    row = review["paired_tests"][key]
    if row["candidate"] != candidate or row["reference"] != reference:
        raise ValueError(f"Comparison identity differs: {key}")
    return row


def delta_interval(row, subset):
    bootstrap = row["patient_bootstrap"]
    if bootstrap["sampling_unit"] != "patient_with_all_Studies" or bootstrap["paired"] is not True:
        raise ValueError("Expected paired patient resampling")
    entry = bootstrap["slices"][subset]["intervals"]["B_minus_A"]
    valid = entry["valid_resamples"]
    if type(valid) is not int or not 0 < valid <= bootstrap["samples"]:
        raise ValueError("Invalid paired resample count")
    return interval(entry["ci_95"]), valid


def class_table(scores, subset):
    reference = scores[NAMES[0]][subset]
    counts = reference["class_counts"]
    if len(counts) != 10:
        raise ValueError("Expected ten original classes")
    for name in NAMES:
        current = scores[name][subset]
        if current["class_counts"] != counts or len(current["per_class_auc"]) != 10:
            raise ValueError(f"Class support differs: {name}/{subset}")
    rows = []
    for index, count in enumerate(counts):
        if not count["eligible"] or count["positive"] <= 0 or count["negative"] <= 0:
            raise ValueError(f"Ineligible original class {index + 1}/{subset}")
        values = [scores[name][subset]["per_class_auc"][index] for name in NAMES]
        rows.append([index + 1, f"{count['positive']}/{count['negative']}",
                     *(number(value) for value in values),
                     number(values[5] - values[4], signed=True)])
    return table(["类", "阳/阴 Study", *NAMES, "B融合-A融合"], rows)


def checked_asset_bytes(timing):
    files = timing["source_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Timing source_files are missing")
    paths = [entry["path"] for entry in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate inference asset in source_files")
    sizes = [entry["bytes"] for entry in files]
    if any(type(value) is not int or value <= 0 for value in sizes):
        raise ValueError("Invalid inference asset size")
    if not any(path.endswith("state_fp16.pt") for path in paths):
        raise ValueError("No FP16 export weight in timing source_files")
    if not timing["exact_full_valid_probability_match"]:
        raise ValueError("Timing probability output differs from FP16 full valid")
    return sum(sizes)


def validate(review, evidence):
    if review.get("test_used") is not False or evidence.get("test_used") is not False:
        raise ValueError("These tables require test-free evidence")
    if review.get("selection_bias_corrected") is not False or review.get("T4_certified") is not False:
        raise ValueError("Unexpected selection or T4 claim")
    if (review.get("bootstrap_samples") != 1000 or review.get("num_studies") != 859
            or review.get("num_images") != 1422 or review.get("num_patients") != 228):
        raise ValueError("Final review has unexpected valid or bootstrap coverage")
    if set(review["candidates"]) != set(NAMES) or set(review["timing"]) != set(NAMES):
        raise ValueError("Final review lacks six scored and timed candidates")
    if set(review["selected_epochs"]) != set(NAMES[:4]):
        raise ValueError("Final review lacks four selected epochs")
    if evidence.get("historical_latency_is_proxy_only") is not True:
        raise ValueError("Historical image latency is not marked as proxy")
    for name in NAMES:
        for subset in SLICES:
            value = score(review["candidates"], name, subset)
            number(value)
        timing = review["timing"][name]
        number(timing["mean_ms"])
        checked_asset_bytes(timing)
    for candidate, reference in PAIRS:
        row = comparison(review, candidate, reference)
        for subset in SLICES:
            number(row["delta_macro_auc"][subset])
            delta_interval(row, subset)


def build_tables(review, evidence):
    validate(review, evidence)
    scores = review["candidates"]
    main_rows = []
    for name in NAMES:
        epoch = f"E{review['selected_epochs'][name]}" if name in review["selected_epochs"] else "固定融合"
        main_rows.append([name, epoch, *(number(score(scores, name, subset)) for subset in SLICES)])
    main = table(["候选", "最佳EMA", *(HEADERS[s] for s in SLICES)], main_rows)

    interval_rows = []
    slice_rows = []
    for candidate, reference in PAIRS:
        row = comparison(review, candidate, reference)
        label = f"{candidate} - {reference}"
        interval_rows.append([label, number(row["delta_macro_auc"]["all"], signed=True),
                              *delta_interval(row, "all"),
                              number(row["delta_macro_auc"]["ap_only"], signed=True),
                              *delta_interval(row, "ap_only")])
        for subset in ("ap_only_single_image", "ap_only_multiple_image"):
            slice_rows.append([label, HEADERS[subset],
                               scores[candidate][subset]["num_studies"],
                               number(row["delta_macro_auc"][subset], signed=True),
                               *delta_interval(row, subset)])
    intervals = table(["配对", "整体差", "整体95% CI", "有效抽样", "AP差", "AP95% CI", "有效抽样"],
                      interval_rows)
    slices = table(["配对", "切片", "Study", "AUC差", "患者配对95% CI", "有效抽样"], slice_rows)

    matrix_rows = []
    for name in NAMES:
        timing = review["timing"][name]
        asset_bytes = checked_asset_bytes(timing)
        gpu = timing["environment"]["gpu_name"]
        matrix_rows.append([name, number(score(scores, name, "all")),
                            number(score(scores, name, "ap_only")),
                            f"{asset_bytes:,}", f"{asset_bytes / 1_000_000:.3f}",
                            f"{timing['mean_ms']:.3f}", gpu, "本轮旧单图代理", "待决策"])
    for name in ("043", "044", "045"):
        candidate = evidence["historical_candidates"][name]
        asset_bytes = candidate["asset_bytes"]
        if asset_bytes != (candidate["weight_bytes"] + candidate["config_bytes"]
                           + candidate.get("make_report_bytes_in_inventory", 0)):
            raise ValueError(f"Historical asset inventory differs: {name}")
        matrix_rows.append([name, number(candidate["valid_fp16_auc"]),
                            number(candidate["valid_ap_only_auc"]),
                            f"{asset_bytes:,}", f"{asset_bytes / 1_000_000:.3f}",
                            f"{candidate['historical_ms_per_image']:.3f}",
                            candidate["hardware"], f"历史旧单图代理；{candidate['decoder']}", "待决策"])
    matrix = table(["候选", "整体valid AUC", "AP-only AUC", "登记字节", "MB", "旧ms/图",
                    "本地GPU", "计时口径", "部署排序"], matrix_rows)
    return {
        "AB_MAIN_TABLE": main,
        "AB_INTERVAL_TABLE": intervals,
        "AP_CLASS_TABLE": class_table(scores, "ap_only"),
        "ALL_CLASS_TABLE": class_table(scores, "all"),
        "AP_SLICE_INTERVAL_TABLE": slices,
        "DEPLOYMENT_MATRIX": matrix,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=HERE / "ab_review/results.json",
                        help="Final results.json or directory containing it")
    parser.add_argument("--evidence", type=Path, default=HERE / "AB_EVIDENCE.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    source = args.review / "results.json" if args.review.is_dir() else args.review
    tables = build_tables(load_json(source), load_json(args.evidence))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "tables.md").write_text(
        "\n\n".join(f"## {key}\n\n{value}" for key, value in tables.items()) + "\n",
        encoding="utf-8")
    print(f"Rendered {len(tables)} tables from {source} into {args.output_dir}")


if __name__ == "__main__":
    main()
