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
    lines += ["", "## Fusion and spatial pooling", ""]
    final_path = comparison_files.get("final")
    final = load(final_path) if final_path else None
    resolution_for_timing = load(comparison_files.get("resolution", "")) if comparison_files.get("resolution") else None
    if not final:
        lines.append("- not complete")
    else:
        lines += ["| Tiny member set | TTA | Macro-AUC | AP-only Macro-AUC | P50/P95 ms |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for name in ("tiny_a", "tiny_b", "tiny_a_tiny_b_equal"):
            for mode in ("identity", "hflip2"):
                candidate_key = f"{name}_{mode}"
                candidate = final.get("candidates", {}).get(candidate_key)
                if not candidate:
                    lines.append(f"| {name} | {mode} | not complete | not complete | not measured |")
                    continue
                timing = candidate.get("timing")
                if timing is None and resolution_for_timing:
                    timing = resolution_for_timing.get("candidates", {}).get(candidate_key, {}).get("timing")
                cost = (f"{timing['p50_ms']:.2f}/{timing['p95_ms']:.2f}" if timing else "not measured")
                lines.append(f"| {name} | {mode} | {fmt(candidate['scores']['all']['macro_auc'])} | "
                             f"{fmt(candidate['scores']['ap_only']['macro_auc'])} | {cost} |")
        for name, candidate in final.get("candidates", {}).items():
            if len(candidate.get("members", [])) > 1:
                lines.append(f"- {name}: Macro-AUC={fmt(candidate['scores']['all']['macro_auc'])}; "
                             f"AP-only={fmt(candidate['scores']['ap_only']['macro_auc'])}; "
                             f"P50/P95={candidate.get('timing', {}).get('p50_ms', 'not measured')}/"
                             f"{candidate.get('timing', {}).get('p95_ms', 'not measured')} ms.")
        for name, result in final.get("comparisons", {}).items():
            if "tiny_a_tiny_b_equal_hflip2" in name or "res768_roi_equal_hflip2" in name or "mil_hflip2" in name:
                lines.append(f"- {name}: patient-paired delta={fmt(result['slices']['all']['macro_auc_delta'])}; "
                             f"CI={result['slices']['all']['delta_macro']['ci95']}.")
        roi_path = comparison_files.get("roi")
        roi_report = load(roi_path) if roi_path else None
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
    for label, key in (("Dual Tiny timing", "resolution"), ("ROI-pair timing", "roi")):
        path = comparison_files.get(key)
        report = load(path) if path else None
        if report:
            for name, candidate in report.get("candidates", {}).items():
                if name in {"tiny_a_tiny_b_equal_hflip2", "res768_roi_equal_hflip2"}:
                    timing = candidate.get("timing") or {}
                    lines.append(f"- {label} ({name}): P50/P95="
                                 f"{timing.get('p50_ms', 'not measured')}/{timing.get('p95_ms', 'not measured')} ms.")
    lines += ["", "## Human review", "",
              "- Anatomy: confirm no lung tissue was cut in the ROI atlas.",
              "- Final members: pending owner choice after the candidate comparison.",
              "- Rules: only unresolved interpretation boundaries need review.",
              "- Platform: no submission interface was used; platform outcome remains unavailable.", "",
              "The resolution and fusion sensitivity estimates are conditional on this fixed valid set and checkpoint selection; they are not online-score claims.", ""]
    output = Path(plan["report_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
