"""CPU-only, read-only audit of saved fold metrics/OOF/checkpoint bindings.

Creates reports only in this delivery directory. Does not load torch/models or
touch training process/state. F1 is derived at a fixed 0.5 threshold, never tuned.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(output_name):
    root = Path("/home/featurize/work/runs/020_convnextv2_base768_posbce_cv5_4090")
    plan_path = Path("/home/featurize/work/cloud_project_v2/data/cv5_overnight/folds.json")
    plan = load_json(plan_path)
    report = {
        "created_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "run_root": str(root),
        "fold_plan": str(plan_path),
        "mode": "CPU-only saved artifact audit; no model load or inference",
        "metric_sources": {
            "val_macro_auc": "fold_N/best_metrics.json:validation.macro_auc, independently recomputed from best_oof.npz",
            "val_macro_f1_at_0_5": "derived from same best_oof.npz targets/probabilities using sklearn f1_score(average=macro, zero_division=0), probabilities>=0.5",
            "json_macro_f1": "not present; do not imply this field was read from best_metrics.json",
            "validation_tta": "identity768 (saved training validation); not the requested test identity+horizontal-flip TTA",
            "auc_rounding": "dataset_overnight.py:221,231,246 aggregates Study probabilities in float64; train_overnight.py:393 saves OOF probabilities in float32. New ties may change AUC slightly. Both raw deltas and a 1e-5 tolerance are reported; checkpoint/fold/epoch/Study bindings must still match exactly.",
        },
        "folds": [],
    }
    all_keys, all_subjects, all_targets, all_probabilities = [], [], [], []
    for index in range(5):
        fold = root / f"fold_{index}"
        tracked = [fold / x for x in ("best_metrics.json", "status.json", "history.jsonl", "best_oof.npz", "best.pt")]
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in tracked}
        metrics, status, config = [load_json(fold / x) for x in ("best_metrics.json", "status.json", "config.json")]
        history = [json.loads(line) for line in (fold / "history.jsonl").read_text().splitlines() if line.strip()]
        with np.load(fold / "best_oof.npz", allow_pickle=False) as saved:
            targets = saved["targets"].copy()
            probabilities = saved["probabilities"].copy()
            keys = saved["study_keys"].astype(str).tolist()
            subjects = saved["subject_ids"].astype(str).tolist()
            binding = {k: saved[k].item() for k in ("fold", "epoch", "checkpoint_sha256")}
        checkpoint_hash = sha256(fold / "best.pt")
        auc_per_class = roc_auc_score(targets, probabilities, average=None)
        auc, f1 = float(np.mean(auc_per_class)), float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0))
        matching_history = [entry for entry in history if entry["epoch"] == metrics["epoch"]]
        history_matches = len(matching_history) == 1 and matching_history[0]["validation"]["macro_auc"] == metrics["validation"]["macro_auc"]
        expected = {key for key, fold_number in plan["study_to_fold"].items() if fold_number == index}
        checks = {
            "oof_fold_matches": binding["fold"] == index == metrics["fold"],
            "oof_epoch_matches": binding["epoch"] == metrics["epoch"] == metrics["best_epoch"],
            "sha256_binds_checkpoint_json_oof": checkpoint_hash == metrics["checkpoint_sha256"] == binding["checkpoint_sha256"],
            "recomputed_auc_consistent_with_float32_oof": abs(auc - metrics["validation"]["macro_auc"]) <= 1e-5,
            "recomputed_per_class_auc_consistent_with_float32_oof": bool(np.allclose(auc_per_class, metrics["validation"]["per_class_auc"], rtol=0, atol=1e-5)),
            "best_epoch_matches_history": history_matches,
            "all_history_epochs_complete_and_validated": sorted(entry["epoch"] for entry in history) == list(range(1, status["completed_epochs"] + 1)) and status["completed_epochs"] == status["validated_epochs"],
            "oof_not_explicitly_incomplete": metrics.get("oof_complete") is not False,
            "oof_targets_binary": bool(np.all((targets == 0) | (targets == 1))),
            "oof_probabilities_finite_and_in_range": bool(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()),
            "oof_shape": targets.shape == probabilities.shape == (len(keys), 10),
            "study_keys_unique": len(keys) == len(set(keys)),
            "study_keys_match_fold_plan_exactly": set(keys) == expected,
            "study_to_patient_matches_plan": all(plan["study_to_patient"][key] == subject for key, subject in zip(keys, subjects)),
        }
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in tracked}
        checks["artifacts_unchanged_during_read"] = before == after
        reached_end = status["completed_epochs"] >= config["epochs"] or status["state"] in {"early_stopped", "completed", "finished"}
        entry = {
            "fold": index,
            "state": status["state"],
            "completed_epochs": status["completed_epochs"],
            "validated_epochs": status["validated_epochs"],
            "target_epochs": config["epochs"],
            "reached_planned_end_or_early_stopping": reached_end,
            "best_epoch": metrics["epoch"],
            "val_macro_auc": metrics["validation"]["macro_auc"],
            "recomputed_macro_auc": auc,
            "recomputed_minus_json_macro_auc": auc - metrics["validation"]["macro_auc"],
            "max_absolute_per_class_auc_difference": float(np.max(np.abs(auc_per_class - np.asarray(metrics["validation"]["per_class_auc"])))),
            "optional_oof_complete_flag": metrics.get("oof_complete"),
            "val_macro_f1_at_0_5": f1,
            "num_studies": len(keys),
            "tta": metrics["validation"].get("tta"),
            "checkpoint": str(fold / "best.pt"),
            "checkpoint_sha256": checkpoint_hash,
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "checks": checks,
            "eligible_for_inference": all(checks.values()),
        }
        report["folds"].append(entry)
        all_keys.extend(keys)
        all_subjects.append(set(subjects))
        all_targets.append(targets)
        all_probabilities.append(probabilities)
    report["cross_fold_checks"] = {
        "study_disjoint": len(all_keys) == len(set(all_keys)),
        "covers_all_train_studies_in_plan": set(all_keys) == set(plan["study_to_fold"]),
        "patients_disjoint": all(not all_subjects[i].intersection(all_subjects[j]) for i in range(5) for j in range(i)),
    }
    y, p = np.concatenate(all_targets), np.concatenate(all_probabilities)
    report["pooled_oof"] = {
        "num_studies": len(y),
        "macro_auc": float(roc_auc_score(y, p, average="macro")),
        "macro_f1_at_0_5": float(f1_score(y, p >= 0.5, average="macro", zero_division=0)),
        "meaning": "Each Study predicted only by its held-out fold; this is pooled OOF, not an ensemble validation score.",
    }
    report["eligible_folds"] = [entry["fold"] for entry in report["folds"] if entry["eligible_for_inference"]]
    report["folds_reached_planned_end_or_early_stopping"] = [entry["fold"] for entry in report["folds"] if entry["reached_planned_end_or_early_stopping"]]
    output = Path(__file__).parent / output_name
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "folds": [{k: e[k] for k in ("fold", "completed_epochs", "best_epoch", "val_macro_auc", "val_macro_f1_at_0_5", "eligible_for_inference")} for e in report["folds"]], "cross_fold_checks": report["cross_fold_checks"], "pooled_oof": report["pooled_oof"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="interim_summary.json")
    audit(parser.parse_args().output)
