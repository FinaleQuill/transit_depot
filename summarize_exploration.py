"""Write a no-CSV morning report from completed exploration evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DISEASES = {
    1: "Enlarged Cardiomediastinum", 2: "Pneumothorax", 3: "Consolidation",
    4: "Pneumonia", 5: "Edema", 6: "Cardiomegaly", 7: "Atelectasis",
    8: "Lung Opacity", 9: "Pleural Effusion", 10: "No Finding",
}
REVIEW_SINGLES = ("champion041", "res640", "res768", "res896", "roi", "mil")
REVIEW_PAIRS = (("res768", "res896"), ("res768", "roi"))
LOCAL_GAIN_MARGIN = .004
AP_DECLINE_MARGIN = -.003
THRESHOLD_TOLERANCE = 1e-12


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value):
    return "not complete" if value is None else f"{value:.6f}"


def fmt_delta(delta, interval):
    if delta is None or not interval:
        return "not complete"
    return f"{delta:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]"


def candidate_timing(candidate_key, candidate, reports):
    timing = candidate.get("timing") if isinstance(candidate, dict) else None
    if timing:
        return timing
    for report in reports:
        if not report:
            continue
        fallback = (report.get("candidates", {}).get(candidate_key) or {}).get("timing")
        if fallback:
            return fallback
    return None


def selection_is_best(candidate):
    value = candidate.get("selection_kind")
    kinds = value if isinstance(value, list) else [value]
    return bool(kinds) and all(kind in {"valid_selected_best", "best", "best_checkpoint"} for kind in kinds)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def review_singles(comparison):
    """Use the five registered best arms and the original initialization reference."""
    candidates = (comparison or {}).get("candidates", {})
    singles = {}
    for name in REVIEW_SINGLES:
        candidate = candidates.get(name + "_hflip2", {})
        selection_ok = (candidate.get("selection_kind") == "initialization_reference"
                        if name == "champion041" else selection_is_best(candidate))
        if (selection_ok and candidate.get("members") == [name]
                and all(finite_number(candidate.get("scores", {}).get(view, {}).get("macro_auc"))
                        for view in ("all", "ap_only"))):
            singles[name + "_hflip2"] = candidate
    return singles


def paired_interval(comparison, candidate_key, reference_key, view):
    comparisons = comparison.get("comparisons", {})
    forward = comparisons.get(f"{candidate_key}_minus_{reference_key}")
    reverse = comparisons.get(f"{reference_key}_minus_{candidate_key}")
    contrast = forward if forward is not None else reverse
    interval = ((contrast or {}).get("slices", {}).get(view, {}).get("delta_macro", {}).get("ci95"))
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(finite_number(value) for value in interval) or interval[0] > interval[1]):
        return None
    return list(interval) if forward is not None else [-interval[1], -interval[0]]


def assess_local_candidate(comparison, candidate_key, reference_key, complete_singles):
    candidates = comparison.get("candidates", {})
    candidate, reference = candidates[candidate_key], candidates[reference_key]
    auc = candidate.get("scores", {}).get("all", {}).get("macro_auc")
    ap_auc = candidate.get("scores", {}).get("ap_only", {}).get("macro_auc")
    reference_auc = reference["scores"]["all"]["macro_auc"]
    reference_ap = reference["scores"]["ap_only"]["macro_auc"]
    if not finite_number(auc):
        return None
    gain = auc - reference_auc
    ap_delta = ap_auc - reference_ap if finite_number(ap_auc) else None
    interval = paired_interval(comparison, candidate_key, reference_key, "all")
    ap_interval = paired_interval(comparison, candidate_key, reference_key, "ap_only")
    point_gates = (gain >= LOCAL_GAIN_MARGIN - THRESHOLD_TOLERANCE
                   and ap_delta is not None and ap_delta >= AP_DECLINE_MARGIN - THRESHOLD_TOLERANCE)
    reliable_gain = interval is not None and interval[0] > 0
    reasons = []
    if not complete_singles:
        reasons.append("the original reference or one of the five registered best single candidates is incomplete")
    if gain < LOCAL_GAIN_MARGIN - THRESHOLD_TOLERANCE:
        reasons.append("gain is below +0.004")
    if ap_delta is None:
        reasons.append("AP-only scores are unavailable")
    elif ap_delta < AP_DECLINE_MARGIN - THRESHOLD_TOLERANCE:
        reasons.append("AP-only point decline exceeds 0.003")
    if interval is None:
        reasons.append("paired valid CI is unavailable")
    elif not reliable_gain:
        reasons.append("paired valid CI does not support a positive gain")
    if ap_interval is None:
        reasons.append("paired AP-only CI is unavailable")
    return {"candidate": candidate_key, "candidate_auc": auc, "dual_auc": auc,
            "best_single_name": reference_key.removesuffix("_hflip2"), "best_single_key": reference_key,
            "best_single_auc": reference_auc, "gain": gain, "ap_only_delta": ap_delta,
            "paired_ci95": interval, "ap_only_paired_ci95": ap_interval,
            "passes_registered_local_gates": point_gates, "paired_gain_supported": reliable_gain,
            "ap_only_ci_allows_decline_over_margin": ap_interval is not None and ap_interval[0] < AP_DECLINE_MARGIN,
            "all_registered_singles_available": complete_singles,
            "recommend_for_human_review": complete_singles and point_gates and reliable_gain and ap_interval is not None,
            "reasons": reasons}


def best_dual_tiny_recommendation(comparison):
    """Assess both fixed pairs against the best of all registered singles, including 041."""
    if not comparison:
        return None
    candidates = comparison.get("candidates", {})
    singles = review_singles(comparison)
    if not singles:
        return None
    best_single_key = max(singles, key=lambda name: singles[name]["scores"]["all"]["macro_auc"])
    complete_singles = len(singles) == len(REVIEW_SINGLES)
    recommendations = []
    for members in REVIEW_PAIRS:
        dual_name = "_".join(members) + "_equal_hflip2"
        dual = candidates.get(dual_name, {})
        if (dual.get("members") != list(members) or dual.get("probability_weights") != [.5, .5]
                or not selection_is_best(dual)
                or any(member + "_hflip2" not in singles for member in members)):
            continue
        result = assess_local_candidate(comparison, dual_name, best_single_key, complete_singles)
        if result:
            recommendations.append(result)
    if not recommendations:
        return None
    best = max(recommendations, key=lambda item: (item["recommend_for_human_review"], item["gain"]))
    return {**best, "pair_assessments": recommendations}


def final_local_recommendation(comparison, dual_recommendation):
    """Name one review candidate only when it improves on the measured original reference."""
    singles = review_singles(comparison)
    original = "champion041_hflip2"
    baseline_result = {"candidate": original, "retain_original_baseline": True, "assessment": None}
    if original not in singles:
        return baseline_result
    complete_singles = len(singles) == len(REVIEW_SINGLES)
    eligible = []
    for name in singles:
        if name == original:
            continue
        result = assess_local_candidate(comparison, name, original, complete_singles)
        if result and result["recommend_for_human_review"]:
            eligible.append(result)
    for result in (dual_recommendation or {}).get("pair_assessments", []):
        if result["recommend_for_human_review"]:
            eligible.append(result)
    if not eligible:
        return baseline_result
    best = max(eligible, key=lambda result: result["candidate_auc"])
    return {"candidate": best["candidate"], "retain_original_baseline": False, "assessment": best}


def resolution_direction(resolution):
    rows = (resolution or {}).get("resolution_sensitivity", [])
    class_ids = [row.get("class_id") for row in rows]
    if len(rows) != 10 or set(class_ids) != set(range(1, 11)):
        return None, "Sensitivity map is incomplete; no training-resolution direction is supported yet."
    supported = [row for row in rows if row.get("train896_benefit_at_infer768") is True]
    count = len(supported)
    if count == 0:
        conclusion = "No class is supported for higher training resolution at common inference768; keep inference768 and do not branch by disease yet."
    elif count <= 4:
        conclusion = f"Only {count}/10 classes support higher training resolution at common inference768; continue only those classes as a targeted training-resolution follow-up."
    elif count == 5:
        conclusion = "Half (5/10) of classes support higher training resolution at common inference768; continue targeted class-level follow-up without claiming a majority benefit."
    else:
        conclusion = f"A majority ({count}/10) classes support higher training resolution at common inference768; continue the higher-resolution training direction while retaining inference768."
    return count, conclusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    status = load(plan["status_file"]) or {"state": "STATUS MISSING"}
    lines = ["# Morning exploration report", "",
             f"Queue state: **{status.get('state')}**; current arm: **{status.get('current_job')} / {status.get('phase')}**.",
             f"Estimated finish UTC: {status.get('estimated_finish_utc') or 'not available'}.",
             f"Failure: {status.get('failure_reason') or 'none recorded'}.",
             "Test access: no. Platform submission: no. Declaration changes: no.", "",
             "Class names follow the official task label sheet (SHA-256 034a098696d60c6a3d6ee041d75b6d22ec85c5c1dc272fc0ad44846c12364cb5).", "",
             "Metrics below are emitted only when their evidence files exist; missing results are marked incomplete.", ""]
    lines += ["## Job evidence", ""]
    receipts = status.get("compliance_records", {})
    if receipts:
        lines += ["| Job | Receipt | Preregistration SHA-256 | Run ledger SHA-256 | Receipt SHA-256 |",
                  "| --- | --- | --- | --- | --- |"]
        for name, record in receipts.items():
            lines.append(f"| {name} | {record.get('receipt_path', 'missing')} | "
                         f"{record.get('preregistration_sha256', 'missing')} | "
                         f"{record.get('run_ledger_sha256') or 'not applicable'} | "
                         f"{record.get('receipt_sha256', 'missing')} |")
    else:
        lines.append("- No completed job receipts recorded.")
    lines += ["## ROI atlas", ""]
    gallery = Path(plan["roi_gallery"])
    if gallery.is_dir():
        for view in ("AP", "PA", "Lateral", "Missing"):
            path = gallery / f"train_{view}_contact.jpg"
            lines.append(f"- {view}: [{path.name}]({path})" if path.is_file() else f"- {view}: not complete")
        index = gallery / "index.md"
        if index.is_file():
            lines += ["", f"Gallery review: {index}"]
    else:
        lines.append("- not complete")
    lines += ["", "## Candidate metrics", "",
              "| Candidate | Selection | Macro-AUC | AP-only Macro-AUC | Registered sampled-train minus full-valid gap | Three-image P50/P95 ms |",
              "| --- | --- | ---: | ---: | --- | ---: |"]
    evaluations = plan.get("evaluation_reports", {})
    ordered_evaluations = sorted(evaluations.items(), key=lambda item:
        (load(Path(item[1]).parent / "metadata.json") or {}).get("selection_kind") != "fixed_final_E3")
    evaluation_details = []
    for name, value in ordered_evaluations:
        report = load(value)
        if not report or not report.get("complete"):
            lines.append(f"| {name} | not complete | not complete | not complete | not complete | not measured |")
            continue
        scores = report["scores"]["hflip2"]
        gap = report.get("sampled_train_gap") or {}
        timing = ((report.get("timing") or {}).get("modes") or {}).get("hflip2") or {}
        p50, p95 = timing.get("p50_ms"), timing.get("p95_ms")
        cost = f"{p50:.2f}/{p95:.2f}" if p50 is not None and p95 is not None else "not measured"
        gap_value = gap.get("sampled_train_minus_full_valid_macro_auc")
        gap_text = (f"{gap.get('num_studies')} train Studies: {fmt(gap_value)}"
                    if gap_value is not None else "not measured")
        metadata = load(Path(value).parent / "metadata.json") or {}
        lines.append(f"| {name} | {metadata.get('selection_kind', 'unavailable')} | "
                     f"{fmt(scores['all']['macro_auc'])} | {fmt(scores['ap_only']['macro_auc'])} | "
                     f"{gap_text} | {cost} |")
        evaluation_details.append("")
        evaluation_details.append(f"Per-class valid AUC ({name}): " + ", ".join(
            f"{DISEASES[i+1]}={fmt(v)}" for i, v in enumerate(scores['all']['per_class_auc'])))
        evaluation_details.append(f"Per-class AP-only AUC ({name}): " + ", ".join(
            f"{DISEASES[i+1]}={fmt(v)}" for i, v in enumerate(scores['ap_only']['per_class_auc'])))
        compliance = metadata.get("training_compliance", {})
        evaluation_details.append(f"Compliance: checkpoint SHA {metadata.get('source_checkpoint_sha256', 'unavailable')}; "
                     f"lifecycle ledger SHA {compliance.get('ledger_sha256', 'unavailable')}.")
    lines += evaluation_details
    lines += ["", "## Resolution sensitivity", ""]
    comparison_files = plan.get("comparison_reports", {})
    resolution = load(comparison_files.get("resolution", "")) if comparison_files.get("resolution") else None
    supported_count = None
    direction = None
    noninferior_count = None
    if not resolution:
        lines.append("- not complete")
    else:
        lines += ["| Class | 896 vs 768 valid delta (95% patient CI) | 896 vs 768 AP-only delta (95% patient CI) | "
                  "640 vs 768 valid delta (95% patient CI) | 640 vs 768 AP-only delta (95% patient CI) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        contrasts = resolution.get("comparisons", {})
        high = contrasts.get("res896_hflip2_minus_res768_hflip2", {}).get("slices", {})
        low = contrasts.get("res640_hflip2_minus_res768_hflip2", {}).get("slices", {})
        high_all = high.get("all", {}).get("per_class", [])
        high_ap = high.get("ap_only", {}).get("per_class", [])
        low_all = low.get("all", {}).get("per_class", [])
        low_ap = low.get("ap_only", {}).get("per_class", [])
        for index in range(10):
            values = [rows[index] if len(rows) > index else {} for rows in
                      (high_all, high_ap, low_all, low_ap)]
            lines.append(f"| {DISEASES[index+1]} | " + " | ".join(
                fmt_delta(value.get("delta"), value.get("ci95")) for value in values) + " |")
        lines.append("")
        for row in resolution.get("resolution_sensitivity", []):
            class_index = row['class_id'] - 1
            high_stat = high_all[class_index] if len(high_all) > class_index else {}
            low_stat = low_all[class_index] if len(low_all) > class_index else {}
            lines.append(f"- {DISEASES.get(row['class_id'], '')}: {row['status']}; "
                         f"train896 benefit={row['train896_benefit_at_infer768']} "
                         f"({fmt_delta(high_stat.get('delta'), high_stat.get('ci95'))}); "
                         f"train640 noninferior={row['train640_noninferior_at_infer768_margin_0_01']} "
                         f"({fmt_delta(low_stat.get('delta'), low_stat.get('ci95'))}).")
        lines += ["", "Paired contrasts:"]
        for name, result in resolution.get("comparisons", {}).items():
            if "res896_hflip2_minus_res768_hflip2" in name or "res640_hflip2_minus_res768_hflip2" in name:
                ci = result["slices"]["all"]["delta_macro"]["ci95"]
                delta = result["slices"]["all"]["macro_auc_delta"]
                lines.append(f"- {name}: delta={fmt(delta)}, patient 95% CI={ci}.")
        supported_count, direction = resolution_direction(resolution)
        noninferior_count = sum(row.get("train640_noninferior_at_infer768_margin_0_01") is True
                                for row in resolution.get("resolution_sensitivity", []))
        support_text = f"{supported_count}/10" if supported_count is not None else "incomplete"
        noninferior_text = f"{noninferior_count}/10" if supported_count is not None else "incomplete"
        lines += ["", "Next direction (automatic, valid-only evidence):", "",
                  f"- {direction}",
                  f"- Higher-training-resolution support: {support_text} classes; 640 noninferior within 0.01: {noninferior_text}.",
                  "- This map compares training size with inference fixed at 768. It cannot establish disease-specific inference sizes; no per-disease inference-size experiment was run."]
    lines += ["", "## Fusion and spatial pooling", ""]
    final_path = comparison_files.get("final")
    final = load(final_path) if final_path else None
    roi_report = load(comparison_files.get("roi", "")) if comparison_files.get("roi") else None
    resolution_best = load(comparison_files.get("resolution_best", "")) if comparison_files.get("resolution_best") else None
    resolution_for_timing = resolution
    timing_reports = [final, resolution_for_timing, roi_report]
    if not final:
        lines.append("- not complete")
    else:
        lines += ["| Fixed member set | Macro-AUC | AP-only Macro-AUC | P50/P95 ms |",
                  "| --- | ---: | ---: | ---: |"]
        preferred = ("champion041_hflip2", "res640_hflip2", "res768_hflip2", "res896_hflip2",
                     "res768_res896_equal_hflip2", "res768_roi_equal_hflip2", "roi_hflip2", "mil_hflip2",
                     "tiny_a_hflip2", "tiny_b_hflip2", "tiny_a_tiny_b_equal_hflip2")
        final_candidates = final.get("candidates", {})
        for candidate_key in preferred:
            candidate = final_candidates.get(candidate_key)
            if not candidate:
                continue
            timing = candidate_timing(candidate_key, candidate, timing_reports)
            cost = (f"{timing['p50_ms']:.2f}/{timing['p95_ms']:.2f}"
                    if timing and timing.get('p50_ms') is not None and timing.get('p95_ms') is not None
                    else "not measured")
            name = candidate_key.removesuffix("_hflip2")
            cost_text = cost + (" ms" if cost != "not measured" else "")
            lines.append(f"| {name} | {fmt(candidate['scores']['all']['macro_auc'])} | "
                         f"{fmt(candidate['scores']['ap_only']['macro_auc'])} | {cost_text} |")
        if "res768_hflip2" in final_candidates:
            lines += ["", "Paired candidate contrasts vs res768 (patient-bootstrap 95% CI):"]
            for candidate_key in preferred:
                if candidate_key == "res768_hflip2" or candidate_key not in final_candidates:
                    continue
                candidate = final_candidates[candidate_key]
                label = candidate_key.removesuffix("_hflip2")
                contrast_key = f"{candidate_key}_minus_res768_hflip2"
                contrast = final.get("comparisons", {}).get(contrast_key)
                if not contrast:
                    continue
                all_slice = contrast["slices"]["all"]
                ap_slice = contrast["slices"]["ap_only"]
                lines.append(f"- {label}: valid {fmt_delta(all_slice.get('macro_auc_delta'), all_slice.get('delta_macro', {}).get('ci95'))}; "
                             f"AP-only {fmt_delta(ap_slice.get('macro_auc_delta'), ap_slice.get('delta_macro', {}).get('ci95'))}.")
        for name, candidate in final.get("candidates", {}).items():
            if len(candidate.get("members", [])) > 1:
                timing = candidate_timing(name, candidate, timing_reports)
                timing_text = (f"{timing['p50_ms']:.2f}/{timing['p95_ms']:.2f}"
                               if timing and timing.get('p50_ms') is not None and timing.get('p95_ms') is not None
                               else "not measured")
                lines.append(f"- {name}: members={candidate.get('members')}; weights={candidate.get('probability_weights')}; "
                             f"Macro-AUC={fmt(candidate['scores']['all']['macro_auc'])}; "
                             f"AP-only={fmt(candidate['scores']['ap_only']['macro_auc'])}; P50/P95={timing_text} ms.")
        if roi_report:
            roi_contrast = roi_report.get("comparisons", {}).get("roi_hflip2_minus_res768_hflip2")
            if roi_contrast:
                all_slice = roi_contrast["slices"]["all"]
                ap_slice = roi_contrast["slices"]["ap_only"]
                lines += ["", f"ROI-only vs matched full-frame: Macro-AUC delta={fmt(all_slice['macro_auc_delta'])}, "
                          f"patient 95% CI={all_slice['delta_macro']['ci95']}; AP-only delta="
                          f"{fmt(ap_slice['macro_auc_delta'])}, CI={ap_slice['delta_macro']['ci95']}."]
                focus = {2, 3, 7, 8, 9}
                rows = all_slice["per_class"]
                lines.append("ROI local-finding focus (Pneumothorax, Consolidation, Atelectasis, Lung Opacity, Pleural Effusion): " +
                    "; ".join(f"{DISEASES[row['class_id']]} delta={fmt(row['delta'])}, CI={row['ci95']}"
                              for row in rows if row['class_id'] in focus))
        mil_contrast = final.get("comparisons", {}).get("mil_hflip2_minus_res768_hflip2")
        if mil_contrast:
            lines.append(f"MIL vs GAP 768: Macro-AUC delta={fmt(mil_contrast['slices']['all']['macro_auc_delta'])}, "
                         f"patient 95% CI={mil_contrast['slices']['all']['delta_macro']['ci95']}.")
    recommendation = best_dual_tiny_recommendation(resolution_best)
    lines += ["", "## Best-checkpoint comparison (supplementary)", ""]
    if resolution_best:
        best_candidates = resolution_best.get("candidates", {})
        best_keys = tuple(name + "_hflip2" for name in REVIEW_SINGLES) + (
            "res768_res896_equal_hflip2", "res768_roi_equal_hflip2")
        lines += ["Valid-selected best checkpoints are secondary to fixed-final E3 results.", "",
                  "The champion041 initialization_reference is the original 041/E9 snapshot, evaluated with the same protocol; it is not a newly trained run.", "",
                  "| Candidate | Selection | Macro-AUC | AP-only Macro-AUC | P50/P95 ms |",
                  "| --- | --- | ---: | ---: | ---: |"]
        best_details = []
        for key in best_keys:
            candidate = best_candidates.get(key)
            if not candidate:
                continue
            timing = candidate_timing(key, candidate, [resolution_best] + timing_reports)
            cost = (f"{timing['p50_ms']:.2f}/{timing['p95_ms']:.2f}"
                    if timing and timing.get('p50_ms') is not None and timing.get('p95_ms') is not None
                    else "not measured")
            lines.append(f"| {key.removesuffix('_hflip2')} | {candidate.get('selection_kind', 'unavailable')} | "
                         f"{fmt(candidate['scores']['all']['macro_auc'])} | "
                         f"{fmt(candidate['scores']['ap_only']['macro_auc'])} | {cost} |")
            best_details.append(f"Per-class valid AUC ({key.removesuffix('_hflip2')} best/reference): " + ", ".join(
                f"{DISEASES[i+1]}={fmt(value)}" for i, value in enumerate(candidate['scores']['all'].get('per_class_auc', []))))
        lines += [""] + best_details
    else:
        lines.append("- Not complete.")
    lines += ["", "## Best-checkpoint review suggestion", ""]
    if recommendation is None:
        lines.append("- Neither registered 50:50 pair (res768+res896, res768+roi) has sufficient best-checkpoint evidence for a fusion assessment.")
    else:
        for assessment in recommendation["pair_assessments"]:
            gate_text = ("passes" if assessment["passes_registered_local_gates"] else "fails")
            lines.append(f"- {assessment['candidate']}: dual Macro-AUC={fmt(assessment['dual_auc'])}; "
                         f"best single across the original reference and five registered arms "
                         f"{assessment['best_single_name']}={fmt(assessment['best_single_auc'])}; "
                         f"gain={assessment['gain']:+.6f}; patient-paired 95% CI="
                         f"{assessment['paired_ci95'] or 'not available'}; AP-only delta="
                         f"{fmt(assessment['ap_only_delta'])}, patient-paired 95% CI="
                         f"{assessment['ap_only_paired_ci95'] or 'not available'}. "
                         f"This {gate_text} the registered local point gates: gain >=0.004 and AP-only delta >=-0.003.")
            if assessment["reasons"]:
                lines.append("  Review limitation: " + "; ".join(assessment["reasons"]) + ".")
            if assessment["ap_only_ci_allows_decline_over_margin"]:
                lines.append("  AP-only risk: the paired interval still includes a decline greater than 0.003; passing the point gate does not establish AP-only noninferiority.")
    final_suggestion = final_local_recommendation(resolution_best, recommendation)
    if final_suggestion["retain_original_baseline"]:
        lines.append("- Suggested decision: retain the original champion041 baseline. No new candidate has established the required local gain with paired support and the AP-only point safeguard; do not claim a stronger model from this round.")
    else:
        assessment = final_suggestion["assessment"]
        lines.append(f"- One candidate for owner choice: **{final_suggestion['candidate']}**. "
                     f"Versus {assessment['best_single_name']}: valid delta="
                     f"{fmt_delta(assessment['gain'], assessment['paired_ci95'])}; AP-only delta="
                     f"{fmt_delta(assessment['ap_only_delta'], assessment['ap_only_paired_ci95'])}. "
                     "This is a local comparison suggestion, subject to the measured latency and primary fixed-final E3 evidence.")
        if assessment["ap_only_ci_allows_decline_over_margin"]:
            lines.append("- The suggested candidate passes the AP-only point safeguard, but its interval still permits a decline greater than 0.003.")
    lines.append("- The >=0.004 and AP-only >=-0.003 gates use local point estimates. A positive paired valid CI is separately required for this report to suggest a new candidate; a CI crossing zero does not establish reliable improvement.")
    lines.append("- This is local valid screening only; valid-selected results are supplementary and selection-conditional. Online performance is unknown, final members require human choice, and this report does not freeze members.")
    lines += ["", "## Human review", "",
              "- Anatomy: confirm no lung tissue was cut in the ROI atlas.",
              "- Final members: owner review remains required; this report does not freeze members.",
              "- Rules: only unresolved interpretation boundaries need review.",
              "- Platform: no submission interface was used; platform outcome remains unavailable.", "",
              "The resolution and fusion sensitivity estimates are conditional on this fixed valid set and checkpoint selection; they are not online-score claims.", ""]
    output = Path(plan["report_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
