"""CPU-only, bound inner-development diagnostics; never reads images or weights.

Patient leave-one-out influence is the exact change in the original Study AUC
after removing ALL Studies of that patient. It is sensitivity evidence, not a
confidence interval, a reason to remove patients, or proof of incorrect labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from reconstruction.evaluate import _RankedClass, _validated, json_native

RUN_ROOT = Path(__file__).resolve().parents[2] / "reconstruction_runs_20261103"
REGISTERED_SPLIT = RUN_ROOT / "splits/split_plan.json"
REGISTERED_SPLIT_SHA = "b7bf913c38756b9b4db42a87847c7d4879f477b8644cc66a62172c50fa3f4827"
REGISTERED_DEV_SHA = "fd889cd16f0c2f3156aa79ec08521741238ea0e362ebd86169c13ad3593e96af"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON: {value}")))


def _macro(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def _difference(a, b):
    return a - b if a is not None and b is not None else None


def _metadata_from_manifest(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Malformed registered dev header")
        rows = list(reader)
    studies, image_ids = {}, set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed registered dev row")
        if row["split"] != "train" or row["training_eligible"].lower() != "true":
            raise ValueError("Internal dev must originate only from eligible official train")
        key, patient = row["study_key"], row["subject_id"]
        if key != f"{patient}/{row['study_id']}" or not patient or row["image_id"] in image_ids:
            raise ValueError("Invalid registered Study or duplicate image identity")
        image_ids.add(row["image_id"])
        labels = json.loads(row["targets"])
        if (not isinstance(labels, list) or len(labels) != 10
                or any(type(v) not in (int, float) or v not in (0, 1) for v in labels)):
            raise ValueError("Registered original labels must be ten binary values")
        record = studies.setdefault(key, {"patient": patient, "targets": labels, "views": []})
        if record["patient"] != patient or record["targets"] != labels:
            raise ValueError("Conflicting registered Study metadata")
        record["views"].append(row["view_position"].strip().upper() or "MISSING")
    if (len(studies) != expected["studies"] or len(rows) != expected["images"]
            or len({v["patient"] for v in studies.values()}) != expected["patients"]):
        raise ValueError("Registered dev counts differ from the fixed plan")
    return studies


def load_bound_predictions(prediction_dir, *, split_plan=REGISTERED_SPLIT,
                           _expected_split_sha=REGISTERED_SPLIT_SHA,
                           _expected_dev_sha=REGISTERED_DEV_SHA, _expected_studies=3703):
    """Verify exact registered dev bytes, original identities/labels and report.

    Underscored overrides exist only for temporary synthetic CPU fixtures; the
    CLI exposes none. This function does not open the H manifest or source images.
    """
    split_plan = Path(split_plan).resolve(strict=True)
    if sha(split_plan) != _expected_split_sha:
        raise ValueError("Split plan differs from the registered immutable plan")
    plan = _json(split_plan)
    if (plan.get("schema") != "reconstruction_patient_content_holdout_v1"
            or plan.get("holdout_fold") != 4 or plan.get("dev_fold") != 0
            or plan.get("train_folds") != [1, 2, 3]
            or any(value for pair in plan.get("isolation", {}).values() for value in pair.values())):
        raise ValueError("Invalid registered isolation roles")
    expected = plan["roles"]["dev"]
    dev = Path(expected["path"]).resolve(strict=True)
    if (dev != split_plan.parent / "dev.csv" or expected["studies"] != _expected_studies
            or expected["sha256"] != _expected_dev_sha or sha(dev) != _expected_dev_sha):
        raise ValueError("Dev manifest path/hash/count differs from registered dev")
    registered = _metadata_from_manifest(dev, expected)
    directory = Path(prediction_dir).resolve(strict=True)
    report_path, prediction_path = directory / "report.json", directory / "predictions.npz"
    report_hash = sha(report_path)
    report = _json(report_path)
    if (report.get("scope") != "inner_validation" or report.get("num_studies") != _expected_studies
            or any(report.get(k) is not False for k in
                   ["holdout_H_accessed", "official_validation_accessed", "test_accessed"])):
        raise ValueError("Report is not an isolated complete internal-development evaluation")
    prediction_hash = sha(prediction_path)
    if prediction_hash != report.get("predictions_sha256"):
        raise ValueError("Predictions SHA differs from the bound report")
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {"probabilities", "targets", "study_keys", "patient_ids", "views_json", "image_counts"}
        if not required <= set(stored.files):
            raise ValueError("Predictions lack complete original metadata")
        values = {k: stored[k].copy() for k in required}
    views = [json.loads(v) for v in values["views_json"]]
    p, y, keys, patients, normalized = _validated(values["probabilities"], values["targets"],
        values["study_keys"], values["patient_ids"], views)
    if set(keys.tolist()) != set(registered):
        raise ValueError("Predictions do not cover exactly the registered dev Studies")
    for i, key in enumerate(keys):
        original = registered[key]
        if (patients[i] != original["patient"] or y[i].tolist() != original["targets"]
                or normalized[i] != original["views"] or values["image_counts"][i] != len(original["views"])):
            raise ValueError(f"Predictions disagree with original Study metadata: {key}")
    actual = [_RankedClass(y[:, c], p[:, c]).score(np.ones(len(p)))[0] for c in range(10)]
    if (report.get("per_class_auc") != actual or report.get("macro_auc") != _macro(actual)
            or report.get("num_patients") != expected["patients"]
            or report.get("num_images") != expected["images"]):
        raise ValueError("Report metrics/counts disagree with bound predictions")
    # Fail if files changed while this CPU reader validated them.
    if (sha(prediction_path) != prediction_hash or sha(dev) != _expected_dev_sha
            or sha(report_path) != report_hash or sha(split_plan) != _expected_split_sha):
        raise ValueError("Bound inputs changed during diagnostics")
    return {"probabilities": p, "targets": y, "study_keys": keys, "patient_ids": patients,
            "views": normalized, "image_counts": values["image_counts"], "report": report,
            "binding": {"prediction_path": str(prediction_path), "predictions_sha256": prediction_hash,
                        "report_path": str(report_path), "report_sha256": report_hash,
                        "split_plan_path": str(split_plan), "split_plan_sha256": _expected_split_sha,
                        "dev_manifest_path": str(dev), "dev_manifest_sha256": _expected_dev_sha}}


def align_reference(candidate, reference):
    keys = reference["study_keys"].tolist()
    if len(set(keys)) != len(keys) or set(keys) != set(candidate["study_keys"].tolist()):
        raise ValueError("Paired reference does not have identical unique Study coverage")
    positions = {key: i for i, key in enumerate(keys)}
    order = np.asarray([positions[k] for k in candidate["study_keys"]])
    for name in ("targets", "patient_ids", "image_counts"):
        if not np.array_equal(candidate[name], reference[name][order]):
            raise ValueError(f"Paired original metadata differs: {name}")
    if candidate["views"] != [reference["views"][i] for i in order]:
        raise ValueError("Paired original metadata differs: views")
    return reference["probabilities"][order]


def exact_patient_loo(probabilities, targets, patient_ids):
    """Exact AUC after deleting each patient, including ties and mixed labels.

    For each class remove winning positive/negative pairs touching the patient,
    adding internal pairs back once because they were subtracted twice. Work is
    O(C N log N + C sum_g n_g log n_g), without N_patient complete re-sorts.
    """
    p, y = np.asarray(probabilities), np.asarray(targets)
    patients = np.asarray(patient_ids)
    if p.ndim != 2 or p.shape[1] != 10 or y.shape != p.shape or len(patients) != len(p):
        raise ValueError("Expected complete Nx10 Study arrays and matching patients")
    if (not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid Study probability/target arrays")
    unique, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])
    matrix = np.full((len(unique), 10), np.nan)
    full = []
    for c in range(10):
        scores, labels = p[:, c], y[:, c].astype(np.int64)
        _, bins = np.unique(scores, return_inverse=True)
        pos = np.bincount(bins, weights=labels)
        neg = np.bincount(bins, weights=1-labels)
        neg_before = np.cumsum(neg) - neg
        pos_after = pos.sum() - np.cumsum(pos)
        positive_contribution = neg_before[bins] + .5 * neg[bins]
        negative_contribution = pos_after[bins] + .5 * pos[bins]
        numerator = float(np.dot(labels, positive_contribution))
        npos, nneg = int(labels.sum()), len(labels) - int(labels.sum())
        full.append(numerator / (npos * nneg) if npos and nneg else None)
        for g, indices in enumerate(groups):
            local_y, local_p = labels[indices], scores[indices]
            gp, gn = int(local_y.sum()), len(indices) - int(local_y.sum())
            remaining_p, remaining_n = npos - gp, nneg - gn
            if not remaining_p or not remaining_n:
                continue
            internal_auc = _RankedClass(local_y, local_p).score(np.ones(len(indices)))[0]
            internal = internal_auc * gp * gn if internal_auc is not None else 0.
            removed_positive = float(np.dot(local_y, positive_contribution[indices]))
            removed_negative = float(np.dot(1-local_y, negative_contribution[indices]))
            matrix[g, c] = (numerator - removed_positive - removed_negative + internal) / (remaining_p * remaining_n)
    return {"patient_ids": unique, "study_counts": np.asarray([len(g) for g in groups]),
            "positive_studies": np.asarray([y[g].sum(0) for g in groups]),
            "full_per_class_auc": full, "full_macro_auc": _macro(full), "loo_per_class_auc": matrix}


def _slice_stats(p, y, patients):
    weights = np.ones(len(p))
    unique, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    equal = 1. / counts[inverse] if len(p) else weights
    ranked = [_RankedClass(y[:, c], p[:, c]) for c in range(10)]
    aucs, aps, equal_aucs = [], [], []
    for rank in ranked:
        a, ap = rank.score(weights)
        aucs.append(a); aps.append(ap); equal_aucs.append(rank.score(equal)[0])
    return {"num_studies": len(p), "num_patients": len(unique),
            "positive_studies": y.sum(0).astype(int).tolist(),
            "negative_studies": (len(y)-y.sum(0)).astype(int).tolist(),
            "positive_patients": [len(np.unique(patients[y[:, c] == 1])) for c in range(10)],
            "negative_patients": [len(np.unique(patients[y[:, c] == 0])) for c in range(10)],
            "per_class_auc": aucs, "macro_auc": _macro(aucs),
            "per_class_average_precision": aps, "macro_average_precision": _macro(aps),
            "patient_equal_weight": {"definition": "Each patient's Studies within this slice have weights 1/n_studies_in_slice; original Study labels retained",
                                     "per_class_auc": equal_aucs, "macro_auc": _macro(equal_aucs)}}


def _verify_selected_completion(item):
    """Bind stable terminal/history evidence; does not audit upstream weights."""
    bound_report = item.get("binding", {}).get("report_path")
    if not bound_report:
        raise ValueError("Completed-run diagnostics require bound disk evidence")
    run_dir = Path(bound_report).parent.parent
    paths = {"status": run_dir / "status.json", "history": run_dir / "history.jsonl",
             "compliance_status": run_dir / "compliance/status.json"}
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    status = json.loads(snapshots["status"])
    compliance = json.loads(snapshots["compliance_status"])
    history_bytes = snapshots["history"]
    if not history_bytes or not history_bytes.endswith(b"\n"):
        raise ValueError("Selected completed-run history lacks a complete final record")
    history = [json.loads(line) for line in history_bytes.splitlines() if line]
    epoch = item["report"].get("metadata", {}).get("epoch")
    aucs = [row.get("macro_auc") for row in history]
    valid_epoch = type(epoch) is int and 1 <= epoch <= len(history)
    if (status.get("state") not in {"completed", "early_stopped"}
            or compliance.get("state") != status.get("state")
            or compliance.get("last_event") != "training_ended"
            or not valid_epoch or type(status.get("best_epoch")) is not int
            or status["best_epoch"] != epoch
            or not history or any(type(row.get("epoch")) is not int for row in history)
            or [row.get("epoch") for row in history] != list(range(1, len(history)+1))
            or any(type(auc) not in (int, float) or not np.isfinite(auc) or not 0 <= auc <= 1 for auc in aucs)):
        raise ValueError("Selected completed-run stage lacks matching terminal/history evidence")
    best_auc = status.get("best_macro_auc")
    if (type(best_auc) not in (int, float) or best_auc != item["report"].get("macro_auc")
            or aucs[epoch-1] != best_auc or max(aucs) != best_auc):
        raise ValueError("Selected best epoch does not match its history row and maximum AUC")
    binding = {name: {"path": str(paths[name]), "sha256": hashlib.sha256(data).hexdigest()}
               for name, data in snapshots.items()}
    _check_completion_stable(binding)
    return {"state": status["state"], "best_epoch": epoch, "completed_epochs": len(history),
            "best_macro_auc": best_auc, "files": binding}


def _check_completion_stable(binding):
    for record in binding.values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Completion evidence changed during diagnostics")


def diagnose(candidate, *, reference=None, top_k=20, model_stage="incomplete_training_epoch"):
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be in 1..100")
    if model_stage not in {"incomplete_training_epoch", "selected_completed_run"}:
        raise ValueError("Unknown diagnostic model stage")
    completion = {}
    if model_stage == "selected_completed_run":
        completion["candidate"] = _verify_selected_completion(candidate)
        if reference is not None:
            completion["reference"] = _verify_selected_completion(reference)
    p, y, keys, patients, views = _validated(candidate["probabilities"], candidate["targets"],
        candidate["study_keys"], candidate["patient_ids"], candidate["views"])
    reference_p = align_reference(candidate, reference) if reference is not None else None
    loo = exact_patient_loo(p, y, patients)
    ref_loo = exact_patient_loo(reference_p, y, patients) if reference_p is not None else None
    influence = []
    full = loo["full_per_class_auc"]
    for i, patient in enumerate(loo["patient_ids"]):
        vals = [float(v) if np.isfinite(v) else None for v in loo["loo_per_class_auc"][i]]
        row = {"patient_id": str(patient), "num_studies": int(loo["study_counts"][i]),
               "positive_studies": loo["positive_studies"][i].astype(int).tolist(),
               "per_class_auc_without_patient": vals, "macro_auc_without_patient": _macro(vals),
               "per_class_auc_removal_delta": [_difference(a, b) for a, b in zip(vals, full)],
               "macro_auc_removal_delta": _difference(_macro(vals), loo["full_macro_auc"])}
        if ref_loo is not None:
            ref_values = [float(v) if np.isfinite(v) else None for v in ref_loo["loo_per_class_auc"][i]]
            deltas = [_difference(a, b) for a, b in zip(vals, ref_values)]
            original_deltas = [_difference(a, b) for a, b in zip(full, ref_loo["full_per_class_auc"])]
            row["paired"] = {"candidate_minus_reference_per_class_auc_without_patient": deltas,
                "candidate_minus_reference_macro_auc_without_patient": _difference(_macro(vals), _macro(ref_values)),
                "per_class_delta_change_on_removal": [_difference(a, b) for a, b in zip(deltas, original_deltas)],
                "macro_delta_change_on_removal": _difference(_difference(_macro(vals), _macro(ref_values)),
                    _difference(loo["full_macro_auc"], ref_loo["full_macro_auc"]))}
        influence.append(row)
    def top(field, c=None, paired=False):
        def value(row):
            source = row["paired"] if paired else row
            return source[field] if c is None else source[field][c]
        usable = [row for row in influence if value(row) is not None]
        return [{"patient_id": r["patient_id"], "num_studies": r["num_studies"],
                 "removal_delta": value(r)} for r in sorted(usable, key=lambda r: (-abs(value(r)), r["patient_id"]))[:top_k]]
    classes = []
    for c in range(10):
        def candidates(label):
            eligible = np.flatnonzero(y[:, c] == label)
            ordered = sorted(eligible, key=lambda i: ((p[i, c] if label else -p[i, c]), str(keys[i])))[:top_k]
            result = [{"study_key": str(keys[i]), "patient_id": str(patients[i]),
                       "target": label, "probability": float(p[i, c]), "views": views[i]} for i in ordered]
            if reference_p is not None:
                for record, i in zip(result, ordered):
                    record["reference_probability"] = float(reference_p[i, c])
            return result
        row = {"class_id": c+1, "auc": full[c], "positive_studies": int(y[:, c].sum()),
               "negative_studies": int(len(y)-y[:, c].sum()),
               "top_patient_influence": top("per_class_auc_removal_delta", c),
               "high_probability_negative_candidates": candidates(0),
               "low_probability_positive_candidates": candidates(1)}
        if ref_loo is not None:
            row["paired_top_patient_influence"] = top("per_class_delta_change_on_removal", c, paired=True)
        classes.append(row)
    masks = {"all": np.ones(len(p), dtype=bool),
        "ap_projection_only": np.asarray([set(v) == {"AP"} for v in views]),
        "pa_plus_lateral_strict": np.asarray([set(v) == {"PA", "LATERAL"} for v in views]),
        "pa_plus_lateral_including_LL": np.asarray(["PA" in v and any(x in {"LATERAL", "LL"} for x in v)
                                                   and set(v) <= {"PA", "LATERAL", "LL"} for v in views]),
        "single_image_study": np.asarray([len(v) == 1 for v in views]),
        "multiple_image_study": np.asarray([len(v) > 1 for v in views])}
    slices = {}
    for name, mask in masks.items():
        stats = _slice_stats(p[mask], y[mask], patients[mask])
        if reference_p is not None:
            ref = _slice_stats(reference_p[mask], y[mask], patients[mask])
            stats.update(reference=ref, paired_macro_auc_delta=_difference(stats["macro_auc"], ref["macro_auc"]),
                         paired_per_class_auc_delta=[_difference(a,b) for a,b in zip(stats["per_class_auc"], ref["per_class_auc"])])
        slices[name] = stats
    result = {"schema": "bound_inner_dev_patient_diagnostics_v1", "scope": "inner_validation",
        "model_stage": model_stage, "final_model_judgment": False,
        "caveats": ["Exploratory internal-development diagnostics after epoch selection; no independent confirmation",
                    "Removal influence is sensitivity, not a confidence interval or permission to exclude patients",
                    "Error rankings are candidates for review, not diagnoses or proof of wrong labels",
                    "Patient-equal metrics are diagnostics and do not replace official Study-weighted AUC",
                    "AP projection denotes an acquisition view; average_precision is a separate metric"],
        "candidate_binding": candidate.get("binding", {}), "candidate_epoch": candidate.get("report", {}).get("metadata", {}).get("epoch"),
        "completed_run_evidence": completion,
        "diagnostic_source_sha256": {"diagnostics.py": sha(__file__),
                                      "evaluate.py": sha(Path(__file__).with_name("evaluate.py"))},
        "num_studies": len(p), "num_patients": len(loo["patient_ids"]),
        "macro_auc": loo["full_macro_auc"], "per_class_auc": full,
        "classes_lowest_auc_first": sorted(classes, key=lambda row: (row["auc"] is None, row["auc"] or 0, row["class_id"])),
        "patient_influence_definition": "AUC_without_patient minus AUC_all; remove every Study of one patient, preserve all original labels",
        "top_macro_patient_influence": top("macro_auc_removal_delta"),
        "all_patient_influence": influence, "slices": slices,
        "slice_definitions": {"pa_plus_lateral_strict": "At least one PA and LATERAL, no other projection",
            "pa_plus_lateral_including_LL": "At least one PA and LATERAL or LL, no other projection; LL explicitly treated as left lateral",
            "ap_projection_only": "All retained image projections are AP; excludes missing and mixed views"},
        "holdout_H_accessed": False, "official_validation_accessed": False, "test_accessed": False,
        "images_read": False, "weights_read": False, "thresholds_fitted": False}
    if ref_loo is not None:
        result.update(reference_binding=reference.get("binding", {}),
            paired_macro_auc_delta=_difference(loo["full_macro_auc"], ref_loo["full_macro_auc"]),
            paired_top_macro_patient_influence=top("macro_delta_change_on_removal", paired=True))
    for evidence in completion.values():
        _check_completion_stable(evidence["files"])
    return json_native(result)


def write_diagnostics(output, report):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = ["# 内部验证患者敏感性与弱类诊断", "", f"模型阶段：{report['model_stage']}；epoch={report['candidate_epoch']}。",
             "本报告是CPU机制与探索性诊断，不是完成模型的终局判断；不据此删除患者或改写标签。", "",
             f"{report['num_studies']} Study / {report['num_patients']}患者；Macro-AUC={report['macro_auc']:.8f}。", "",
             "| 类别 | AUC | 阳性Study | 阴性Study |", "| --- | ---: | ---: | ---: |"]
    for row in report["classes_lowest_auc_first"]:
        lines.append(f"| {row['class_id']} | {row['auc']} | {row['positive_studies']} | {row['negative_studies']} |")
    lines += ["", "| 切片 | Study | 患者 | Study Macro-AUC | 患者等权诊断AUC |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in report["slices"].items():
        lines.append(f"| {name} | {row['num_studies']} | {row['num_patients']} | {row['macro_auc']} | {row['patient_equal_weight']['macro_auc']} |")
    lines += ["", "完整逐患者、逐类删除影响及高分负例/低分正例候选见 report.json。正的removal_delta表示删除该患者后AUC升高，并非排除该患者的理由。",
              "患者等权保留每次Study的原始标签，仅将切片内同患者每Study权重设为1/Study数，不做跨Study概率或标签平均。",
              "PA＋侧位分别提供严格LATERAL与包含明确LL别名的结果；未知、混合AP及PA LLD不纳入这些切片。",
              "AP投影与average precision是不同概念。未读取H、官方valid、test、图片或权重。", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Explicit completed epoch prediction/report directory")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-stage", choices=["incomplete_training_epoch", "selected_completed_run"], default="incomplete_training_epoch")
    args = parser.parse_args()
    output = args.output.resolve()
    if RUN_ROOT.resolve() not in output.parents:
        parser.error("Diagnostics output must stay in isolated reconstruction runs")
    candidate = load_bound_predictions(args.candidate)
    reference = load_bound_predictions(args.reference) if args.reference else None
    report = diagnose(candidate, reference=reference, top_k=args.top_k, model_stage=args.model_stage)
    write_diagnostics(output, report)
    print(json.dumps({"output": str(output), "macro_auc": report["macro_auc"], "model_stage": report["model_stage"]}))


if __name__ == "__main__":
    main()
