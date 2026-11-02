"""Write a no-CSV morning report from completed exploration evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DISEASES = {
    1: "Enlarged Cardiomediastinum", 2: "Pneumothorax", 3: "Consolidation",
    4: "Pneumonia", 5: "Edema", 6: "Cardiomegaly", 7: "Atelectasis",
    8: "Lung Opacity", 9: "Pleural Effusion", 10: "No Finding",
}


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


def best_dual_tiny_recommendation(comparison):
    """Apply the preregistered fixed-dual point threshold for human review only."""
    if not comparison:
        return None
    candidates = comparison.get("candidates", {})
    dual_items = [(name, candidates[name]) for name in ("tiny_a_tiny_b_equal_hflip2",)
                  if name in candidates and candidates[name].get("members") == ["tiny_a", "tiny_b"]
                  and candidates[name].get("probability_weights") == [.5, .5]]
    recommendations = []
    for dual_name, dual in dual_items:
        members = dual.get("members", [])
        singles = [candidates.get(f"{member}_hflip2") for member in members]
        if (not selection_is_best(dual) or any(item is None or not selection_is_best(item)
                                               for item in singles)):
            continue
        single_auc = [item.get("scores", {}).get("all", {}).get("macro_auc") for item in singles]
        dual_auc = dual.get("scores", {}).get("all", {}).get("macro_auc")
        if dual_auc is None or any(value is None for value in single_auc):
            continue
        best_single = max(single_auc)
        best_index = single_auc.index(best_single)
        best_single_name = members[best_index]
        contrast_key = f"{dual_name}_minus_{best_single_name}_hflip2"
        contrast = comparison.get("comparisons", {}).get(contrast_key, {})
        interval = (contrast.get("slices", {}).get("all", {}).get("delta_macro", {}).get("ci95"))
        gain = round(dual_auc - best_single, 6)
        recommendations.append({"candidate": dual_name, "gain": gain,
                                "best_single_name": best_single_name, "best_single_auc": best_single,
                                "dual_auc": dual_auc, "paired_ci95": interval,
                                "recommend_for_human_review": gain >= .004})
    if not recommendations:
        return None
    return max(recommendations, key=lambda item: item["gain"])


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
        lines.append("")
        lines.append(f"Per-class valid AUC ({name}): " + ", ".join(
            f"{DISEASES[i+1]}={fmt(v)}" for i, v in enumerate(scores['all']['per_class_auc'])))
        lines.append(f"Per-class AP-only AUC ({name}): " + ", ".join(
            f"{DISEASES[i+1]}={fmt(v)}" for i, v in enumerate(scores['ap_only']['per_class_auc'])))
        compliance = metadata.get("training_compliance", {})
        lines.append(f"Compliance: checkpoint SHA {metadata.get('source_checkpoint_sha256', 'unavailable')}; "
                     f"lifecycle ledger SHA {compliance.get('ledger_sha256', 'unavailable')}.")
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
        preferred = ("res640_hflip2", "res768_hflip2", "res896_hflip2",
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
        best_keys = ("res640_hflip2", "res768_hflip2", "res896_hflip2", "roi_hflip2", "mil_hflip2")
        lines += ["Valid-selected best checkpoints are secondary to fixed-final E3 results.", "",
                  "| Candidate | Selection | Macro-AUC | AP-only Macro-AUC |",
                  "| --- | --- | ---: | ---: |"]
        for key in best_keys:
            candidate = best_candidates.get(key)
            if not candidate:
                continue
            lines.append(f"| {key.removesuffix('_hflip2')} | {candidate.get('selection_kind', 'unavailable')} | "
                         f"{fmt(candidate['scores']['all']['macro_auc'])} | "
                         f"{fmt(candidate['scores']['ap_only']['macro_auc'])} |")
            lines.append(f"Per-class valid AUC ({key.removesuffix('_hflip2')} best): " + ", ".join(
                f"{DISEASES[i+1]}={fmt(value)}" for i, value in enumerate(candidate['scores']['all']['per_class_auc'])))
    else:
        lines.append("- Not complete.")
    lines += ["", "## Best-checkpoint review suggestion", ""]
    if recommendation is None:
        lines.append("- Best-checkpoint comparison is absent or lacks fixed equal-weight tiny_a/tiny_b and both matched single candidates; no suggestion emitted.")
    else:
        status_text = ("meets" if recommendation["recommend_for_human_review"] else "does not meet")
        lines.append(f"- {recommendation['candidate']}: dual Macro-AUC={fmt(recommendation['dual_auc'])}; "
                     f"best matched single {recommendation['best_single_name']}={fmt(recommendation['best_single_auc'])}; "
                     f"gain={recommendation['gain']:+.6f}; patient-paired 95% CI="
                     f"{recommendation['paired_ci95'] or 'not available'}. This {status_text} the fixed >=0.004 local threshold for human final-member review.")
        lines.append("- This is local valid evidence only. It is not an online-score claim and does not freeze final members.")
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
