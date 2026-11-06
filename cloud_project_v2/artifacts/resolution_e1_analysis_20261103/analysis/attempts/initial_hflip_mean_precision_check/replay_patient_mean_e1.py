#!/usr/bin/env python3
"""CPU-only, preregistered E1 validation diagnostic with fixed input paths."""
from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "artifacts/resolution_explore_20261103/evaluations/res768_best"
OUT_DIR = Path(__file__).resolve().parent
PREREG = OUT_DIR.parent / "prereg/patient_mean_e1.txt"
INPUT_FILES = ("predictions.npz", "metadata.json", "report.json")
N_BOOTSTRAP = 2000
SEED = 20261103


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def json_number(value):
    return float(value) if np.isfinite(value) else None


def interval(values):
    valid = values[np.isfinite(values)]
    return {
        "percentile_95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
        "valid_replicates": int(len(valid)),
        "invalid_replicates": int(len(values) - len(valid)),
    }


def weighted_auc_many(target, prediction, weights):
    """AUC for many weighted samples; exact equal-score ties receive 1/2 credit."""
    order = np.argsort(prediction, kind="stable")
    scores = prediction[order]
    labels = target[order]
    group_starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    sorted_weights = weights[:, order]
    positive = np.add.reduceat(sorted_weights * labels[None, :], group_starts, axis=1)
    negative = np.add.reduceat(sorted_weights * (1 - labels[None, :]), group_starts, axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(axis=1)
    result = np.full(len(weights), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def auc_many(targets, probabilities, weights):
    return np.stack([
        weighted_auc_many(targets[:, c], probabilities[:, c], weights)
        for c in range(targets.shape[1])
    ], axis=1)


def main():
    started_utc, started_ns, tick = utc_now(), time.time_ns(), time.perf_counter()
    assert PREREG.exists(), "Preregistration is required before execution"
    prereg_lines = PREREG.read_text().splitlines()
    assert len(prereg_lines) == 4 and all(prereg_lines), "Require exactly four preregistration lines"
    assert PREREG.stat().st_mtime_ns < started_ns
    source_paths = {name: INPUT_DIR / name for name in INPUT_FILES}
    for path in source_paths.values():
        assert path.resolve().parent == INPUT_DIR.resolve(), "Input escaped fixed valid directory"
    input_hashes = {name: sha256(path) for name, path in source_paths.items()}
    write_json(OUT_DIR / "patient_mean_e1_receipt.json", {
        "status": "running", "started_utc": started_utc,
        "preregistration_mtime_ns": PREREG.stat().st_mtime_ns,
        "preregistration_sha256": sha256(PREREG),
        "input_sha256": input_hashes,
        "script_sha256": sha256(Path(__file__).resolve()),
        "compute_device": "CPU", "training_started": False,
        "test_or_submission_files_read_by_this_script": False,
    })
    metadata = json.loads(source_paths["metadata.json"].read_text())
    previous_report = json.loads(source_paths["report.json"].read_text())
    assert metadata["source_epoch"] == 1
    assert metadata["class_ids"] == list(range(1, 11))
    assert metadata["test_accessed"] is False and previous_report["test_accessed"] is False
    assert previous_report["complete"] is True
    assert input_hashes["predictions.npz"] == metadata["prediction_files_sha256"]["predictions.npz"]
    assert input_hashes["metadata.json"] == previous_report["metadata_sha256"]
    with np.load(source_paths["predictions.npz"], allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}

    keys = data["study_keys"]
    assert len(np.unique(keys)) == len(keys)
    assert len(keys) == previous_report["num_studies"] == 859
    targets = data["targets"].astype(np.float64)
    assert targets.shape == (len(keys), 10) and np.isin(targets, [0, 1]).all()
    key_to_index = {key: index for index, key in enumerate(keys)}
    image_indices = np.array([key_to_index[key] for key in data["image_study_keys"]])
    counts = np.bincount(image_indices, minlength=len(keys))
    assert np.array_equal(counts, data["image_counts"]) and np.all(counts > 0)
    views = data["per_view_probabilities"]
    assert views.shape == (1422, 2, 10)
    assert np.isfinite(views).all() and ((views >= 0) & (views <= 1)).all()
    rebuilt = {}
    verification = {}
    for mode, image_probabilities in (("identity", views[:, 0, :]), ("hflip2", views.mean(axis=1))):
        study_probabilities = np.zeros((len(keys), 10), dtype=np.float64)
        np.add.at(study_probabilities, image_indices, image_probabilities)
        study_probabilities /= counts[:, None]
        difference = float(np.max(np.abs(study_probabilities - data[mode])))
        assert difference < 1e-12, f"Cannot reproduce saved {mode} Study probabilities: {difference}"
        rebuilt[mode] = study_probabilities
        verification[mode + "_max_probability_difference"] = difference

    patients, patient_index = np.unique(data["subject_ids"], return_inverse=True)
    num_patients = len(patients)
    assert num_patients == previous_report["num_patients"] == 228
    patient_counts = np.bincount(patient_index)
    ap_only = data["ap_only"].astype(bool)
    assert int(ap_only.sum()) == previous_report["scores"]["hflip2"]["ap_only"]["num_studies"] == 442
    rng = np.random.default_rng(SEED)
    sampled_patients = rng.integers(0, num_patients, size=(N_BOOTSTRAP, num_patients))
    bootstrap_patient_counts = np.zeros((N_BOOTSTRAP, num_patients), dtype=np.int64)
    for row, sample in enumerate(sampled_patients):
        bootstrap_patient_counts[row] = np.bincount(sample, minlength=num_patients)
    bootstrap_study_weights = bootstrap_patient_counts[:, patient_index].astype(np.float64)
    results = {}
    for mode, baseline in rebuilt.items():
        patient_means = np.zeros((num_patients, 10), dtype=np.float64)
        np.add.at(patient_means, patient_index, baseline)
        patient_means /= patient_counts[:, None]
        smoothed = patient_means[patient_index]
        results[mode] = {}
        for cohort, mask in (("all", np.ones(len(keys), dtype=bool)), ("ap_only", ap_only)):
            y, p, q = targets[mask], baseline[mask], smoothed[mask]
            original = auc_many(y, p, np.ones((1, len(y))))[0]
            patient_mean = auc_many(y, q, np.ones((1, len(y))))[0]
            expected = previous_report["scores"][mode][cohort]
            assert np.allclose(original, expected["per_class_auc"], rtol=0, atol=1e-12)
            assert abs(original.mean() - expected["macro_auc"]) < 1e-12
            verification[mode + "_" + cohort + "_max_auc_difference"] = float(np.max(np.abs(original - expected["per_class_auc"])))
            weights = bootstrap_study_weights[:, mask]
            bootstrap_original = auc_many(y, p, weights)
            bootstrap_mean = auc_many(y, q, weights)
            bootstrap_delta = bootstrap_mean - bootstrap_original
            # Complete-class macro preserves the same ten-class estimand in every retained draw.
            bootstrap_macro = bootstrap_delta.mean(axis=1)
            class_results = []
            for c in range(10):
                class_results.append({
                    "class_id": c + 1,
                    "positive_studies": int(y[:, c].sum()),
                    "negative_studies": int((1 - y[:, c]).sum()),
                    "baseline_auc": json_number(original[c]),
                    "patient_mean_auc": json_number(patient_mean[c]),
                    "delta": json_number(patient_mean[c] - original[c]),
                    "paired_patient_bootstrap_delta": interval(bootstrap_delta[:, c]),
                })
            results[mode][cohort] = {
                "num_studies": int(mask.sum()),
                "num_patients_represented": int(len(np.unique(patient_index[mask]))),
                "macro_auc": {
                    "baseline": float(original.mean()),
                    "patient_mean": float(patient_mean.mean()),
                    "delta": float((patient_mean - original).mean()),
                    "paired_patient_bootstrap_delta": interval(bootstrap_macro),
                },
                "classes": class_results,
                "class_1": class_results[0],
                "class_4": class_results[3],
            }
        print(mode, "all delta", results[mode]["all"]["macro_auc"]["delta"], "AP-only delta", results[mode]["ap_only"]["macro_auc"]["delta"], flush=True)

    output = {
        "complete": True,
        "created_utc": utc_now(),
        "diagnostic": "res768 E1 fixed patient mean versus Study mean",
        "status": "non-independent exploratory recheck after observing unregistered E3 negative result",
        "source_epoch": metadata["source_epoch"],
        "selection_kind": metadata["selection_kind"],
        "checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "primary_mode": "hflip2",
        "input_paths": {name: str(path.relative_to(ROOT)) for name, path in source_paths.items()},
        "input_sha256": input_hashes,
        "preregistration_path": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG),
        "replay_script_sha256": sha256(Path(__file__).resolve()),
        "protocol": {
            "study_aggregation": "equal mean of image probabilities; hflip2 first averages identity and flipped probabilities per image",
            "patient_aggregation": "equal mean of all official-valid Study probabilities for each official patient, copied to each Study",
            "ap_only": "stored AP-only mask; patient means formed from all valid Studies before AP-only filtering",
            "bootstrap": "paired patients with replacement from all 228 patients; retain every Study of each sampled patient; same draws for all modes/cohorts",
            "bootstrap_replicates": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
            "ci": "unadjusted percentile 95%; class AUC undefined if a draw has only one label; macro requires all ten classes defined; undefined replicates counted",
            "confidence_limitations": "conditional on validation-selected E1; no correction for repeated validation use, prior E3 inspection, or multiple class/cohort comparisons",
            "fitted_parameters": 0,
            "coefficient_search": False,
            "device": "CPU / numpy",
        },
        "num_images": int(len(image_indices)),
        "num_studies": int(len(keys)),
        "num_patients": num_patients,
        "verification": verification,
        "results": results,
        "access_scope": {
            "this_script_data_inputs": "only the three allowlisted official-valid files; no raw images, manifests, other predictions or weights read",
            "this_script_test_or_submission_reads": False,
            "earlier_incident": "An earlier overbroad rule-review search saw submission-derived lines; see ACCESS_INCIDENT.md. This scope statement does not deny that incident.",
        },
    }
    write_json(OUT_DIR / "patient_mean_e1_inputs_sha256.json", {
        "created_utc": started_utc,
        "input_paths": output["input_paths"],
        "input_sha256": input_hashes,
        "preregistration_sha256": output["preregistration_sha256"],
        "replay_script_sha256": output["replay_script_sha256"],
    })
    write_json(OUT_DIR / "patient_mean_e1.json", output)
    receipt = {
        "status": "completed",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "seconds": time.perf_counter() - tick,
        "run_dir": str(OUT_DIR),
        "argv": "python artifacts/resolution_e1_analysis_20261103/analysis/replay_patient_mean_e1.py",
        "preregistration_preceded_run": True,
        "preregistration_mtime_ns": PREREG.stat().st_mtime_ns,
        "preregistration_sha256": output["preregistration_sha256"],
        "input_sha256": input_hashes,
        "script_sha256": output["replay_script_sha256"],
        "output_sha256": {name: sha256(OUT_DIR / name) for name in ("patient_mean_e1.json", "patient_mean_e1_inputs_sha256.json")},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compute_device": "CPU",
        "training_started": False,
        "test_or_submission_files_read_by_this_script": False,
        "prior_access_incident_record": "ACCESS_INCIDENT.md",
    }
    write_json(OUT_DIR / "patient_mean_e1_receipt.json", receipt)
    print("Completed", OUT_DIR / "patient_mean_e1.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        receipt_path = OUT_DIR / "patient_mean_e1_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt.update({"status": "error", "finished_utc": utc_now(),
                        "error": repr(error), "traceback": traceback.format_exc()})
        write_json(receipt_path, receipt)
        raise
