"""Read-only CPU audit of registered final EMAs and original-valid predictions.

Uses only the standard library. The independent AUC implementation groups tied
scores and counts positive/negative pairs; it does not import the training or
evaluation implementation. Patient bootstrap uses Python's random.Random(46),
independent of the production comparison's NumPy generator. Only this script's
JSON review is written. No model is loaded, GPU called, test file read, or
training asset changed.
"""
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
SAMPLES = 1000
SEED = 46


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


def main():
    report_path = EVIDENCE / "comparison_after_X2/comparison_report.json"
    report = read_json(report_path)
    manifest_path = ROOT / "data/clean_v1_reviewed/valid_manifest.csv"
    manifest_sha = sha(manifest_path)
    require(manifest_sha == report["manifest_sha256"], "Official-valid manifest identity differs")
    with manifest_path.open(encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    labels, image_counts = {}, Counter()
    for row in manifest:
        require(row["split"] == "valid", "Only original valid may be reviewed")
        key = (row["subject_id"], row["study_id"])
        target = tuple(json.loads(row["targets"]))
        require(len(target) == 10 and set(target) <= {0, 1}, "Invalid original ten-class labels")
        require(key not in labels or labels[key] == target, "Conflicting Study labels")
        labels[key] = target
        image_counts[key] += 1
    subjects = sorted({key[0] for key in labels})
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    require((len(manifest), len(labels), len(subjects)) == (1422, 859, 228), "Incomplete valid coverage")
    inputs_sha256 = {str(report_path): sha(report_path), str(manifest_path): manifest_sha}
    member_images = {}
    source_checks = {}
    for name, source in report["source_validations"].items():
        for path, expected_digest in source["files_sha256"].items():
            actual_digest = sha(path)
            require(actual_digest == expected_digest, "Cached-valid source changed: " + path)
            inputs_sha256[path] = actual_digest
        directory = Path(source["validation_dir"])
        validation = read_json(directory / "validation_report.json")
        require(validation["complete"] is True and validation["manifest_sha256"] == manifest_sha,
                "Member validation is incomplete or bound to a different manifest")
        image_values = {}
        with (directory / "valid.image_predictions.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = (row["Image_id"], int(row["Predict_class"]) - 1)
                require(key not in image_values, "Duplicate image-class probability")
                probability = float(row["Probability"])
                require(0 <= probability <= 1, "Invalid probability")
                image_values[key] = probability
        require(len(image_values) == 14220, "Incomplete image-class coverage")
        member_images[name] = image_values
        grouped = aggregate_images(manifest, image_counts, image_values)
        saved = {}
        with (directory / "valid_predictions.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = (row["Subject_id"], row["Study_id"])
                position = int(row["Predict_class"]) - 1
                saved.setdefault(key, {})[position] = float(row["Probability"])
        require(set(saved) == set(labels), "Study coverage differs")
        require(all(set(values) == set(range(10)) for values in saved.values()), "Missing Study class")
        max_error = max(abs(values[c] - saved[key][c])
                        for key, values in grouped.items() for c in range(10))
        require(max_error == 0, "Original-manifest-order Study arithmetic differs")
        source_checks[name] = {"images": 1422, "studies": 859,
                              "source_sha256_match": True,
                              "image_to_study_max_abs_error": max_error}

    candidates = {}
    for name, tiny in [("old_AB", "A"), ("new_X1_B", "X1")]:
        images = {}
        for row in manifest:
            for c in range(10):
                key = (row["image_id"], c)
                b, a = member_images["B"][key], member_images[tiny][key]
                images[key] = b if c == 0 else .5 * a + .5 * b
        candidates[name] = aggregate_images(manifest, image_counts, images)

    def score_groups(predictions, column, predicate=None):
        grouped = []
        for key in sorted(labels, key=lambda value: predictions[value][column]):
            if predicate is not None and not predicate(key):
                continue
            value = predictions[key][column]
            if not grouped or grouped[-1][0] != value:
                grouped.append((value, []))
            grouped[-1][1].append((subject_index[key[0]], labels[key][column]))
        return [entries for _, entries in grouped]

    def auc(groups, patient_weights):
        positive = negative = credit = 0.0
        for entries in groups:
            p = sum(patient_weights[i] for i, y in entries if y)
            n = sum(patient_weights[i] for i, y in entries if not y)
            credit += p * (negative + .5 * n)
            positive += p
            negative += n
        return credit / (positive * negative) if positive and negative else None

    unit_weights = [1] * len(subjects)
    independent_scores = {}
    groups_by_candidate = {}
    for name, predictions in candidates.items():
        groups = [score_groups(predictions, c) for c in range(10)]
        groups_by_candidate[name] = groups
        per_class = [auc(group, unit_weights) for group in groups]
        slices = {}
        for label, predicate in [("single_image_study", lambda key: image_counts[key] == 1),
                                 ("multiple_image_study", lambda key: image_counts[key] > 1)]:
            values = [auc(score_groups(predictions, c, predicate), unit_weights) for c in range(10)]
            slices[label] = {"num_studies": sum(predicate(key) for key in labels),
                             "macro_auc": sum(values) / 10, "per_class_auc": values}
        original_name = "A_B_class1_B_baseline" if name == "old_AB" else "X1_B_class1_B"
        original = report["candidates"][original_name]
        error = max(abs(a - b) for a, b in zip(per_class, original["per_class_auc"]))
        require(error < 1e-15, "Independent pair-count AUC disagrees with comparison")
        independent_scores[name] = {"macro_auc": sum(per_class) / 10, "per_class_auc": per_class,
                                   "class_1_4_8_mean_auc": sum(per_class[c - 1] for c in [1, 4, 8]) / 3,
                                   "slices": slices, "max_abs_per_class_error_from_source": error}
    delta_per_class = [new - old for old, new in zip(independent_scores["old_AB"]["per_class_auc"],
                                                    independent_scores["new_X1_B"]["per_class_auc"])]
    rng = random.Random(SEED)
    deltas = []
    for _ in range(SAMPLES):
        weights = [0] * len(subjects)
        for j in range(len(subjects)):
            weights[rng.randrange(len(subjects))] += 1
        old = [auc(group, weights) for group in groups_by_candidate["old_AB"]]
        new = [auc(group, weights) for group in groups_by_candidate["new_X1_B"]]
        if None not in old and None not in new:
            deltas.append(sum(b - a for a, b in zip(old, new)) / 10)
    independent_bootstrap = {"samples_requested": SAMPLES, "valid_resamples": len(deltas), "seed": SEED,
        "rng": "Python standard-library random.Random (independent of production NumPy RNG)",
        "sampling_unit": "patient_with_all_Studies", "paired": True,
        "method": "Draw 228 patient indices with replacement; use multiplicities in exact weighted tie-aware AUC pair counts.",
        "quantiles": "Linear interpolation at (n-1)*q, q in [0.025,0.975]",
        "macro_auc_delta_ci95": [quantile(deltas, .025), quantile(deltas, .975)],
        "positive_delta_resample_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
        "selection_bias_corrected": False}

    identity_reviews, resources = {}, {}
    for name in ["X1", "X2"]:
        receipt_path = EVIDENCE / f"{name}.final_ema.json"
        receipt = read_json(receipt_path)
        config_path = EVIDENCE / "models" / name / "config.json"
        config = read_json(config_path)
        weights_path = config_path.parent / config["weights_file"]
        execution_path = EVIDENCE / f"{name}_packed.execution.json"
        execution = read_json(execution_path)
        status_path = EVIDENCE / "training" / f"{name}_packed" / "status.json"
        status = read_json(status_path)
        history_path = status_path.parent / "history.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        binding_path = EVIDENCE / "evaluations" / name / "final_export_binding.json"
        binding = read_json(binding_path)
        for path in [receipt_path, config_path, weights_path, execution_path, status_path, history_path, binding_path]:
            inputs_sha256[str(path)] = sha(path)
        require(execution["exit_code"] == 0 and execution["state"] == "training_complete", "Training did not finish")
        require(status["state"] == "completed" and status["completed_epochs"] == 4, "Final epoch is incomplete")
        require(len(history) == 4 and [entry["epoch"] for entry in history] == [1, 2, 3, 4], "Wrong training horizon")
        require(config["source_epoch"] == receipt["source_epoch"] == 4, "Export is not final E4")
        require(config["source_checkpoint_sha256"] == receipt["source_checkpoint_sha256"] == status["last_checkpoint_sha256"],
                "Final EMA/checkpoint identity differs")
        require(config["student_warm_start"] == receipt["student_warm_start"], "Export lost initialization ancestry")
        require(inputs_sha256[str(weights_path)] == config["weights_sha256"] == receipt["weights_sha256"] == binding["weights_sha256"],
                "Actual FP16 weight identity differs")
        require(inputs_sha256[str(config_path)] == receipt["export_config_sha256"] == binding["deployment_config_sha256"],
                "Actual deployment configuration identity differs")
        require(inputs_sha256[str(receipt_path)] == binding["final_ema_receipt_sha256"], "Evaluation receipt binding differs")
        require(all(entry["training"]["full_manifest_coverage"] and entry["training"]["images"] == 30328
                    and entry["training"]["studies"] == 18513 for entry in history), "Training omitted original data")
        identity_reviews[name] = {"source_epoch": 4, "checkpoint_selection": receipt["checkpoint_selection"],
            "full_checkpoint_sha256_as_bound_by_receipt_status_and_config": receipt["source_checkpoint_sha256"],
            "full_checkpoint_blob_independently_rehashed": False,
            "fp16_weights_sha256": config["weights_sha256"], "config_sha256": inputs_sha256[str(config_path)],
            "actual_weights_bytes": weights_path.stat().st_size, "actual_config_bytes": config_path.stat().st_size,
            "final_epoch_not_intermediate_best": status["best_epoch"] != 4,
            "parent_epoch": config["student_warm_start"]["source_epoch"],
            "parent_snapshot_sha256": config["student_warm_start"]["source_snapshot_sha256"],
            "ancestry_matches_receipt": True, "identity_pass": True}
        resources[name] = {"completed_epochs": 4,
            "training_seconds_excluding_epoch_validation": sum(entry["training"]["seconds"] for entry in history),
            "wall_seconds_execution_receipt": (datetime.fromisoformat(execution["finished_hong_kong"])
                - datetime.fromisoformat(execution["started_hong_kong"])).total_seconds(),
            "peak_gpu_allocated_gib": max(entry["training"]["gpu_peak_allocated_gib"] for entry in history),
            "optimizer_updates": status["optimizer_updates"], "ema_updates": status["ema_updates"],
            "amp_skipped_steps": status["amp_skipped_steps"], "full_manifest_coverage_all_epochs": True}

    source_bootstrap = report["paired_patient_bootstrap"]["A_B_class1_B_baseline"]["results"]["X1_B_class1_B"]
    plan_path = ROOT / "docs/BOUNDED_STUDY_FINETUNE_20261101.json"
    packing_path = ROOT / "docs/BOUNDED_STUDY_PACKING_20261101.json"
    inputs_sha256[str(plan_path)], inputs_sha256[str(packing_path)] = sha(plan_path), sha(packing_path)
    review = {"schema_version": 1, "reviewed_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "reviewer": "independent audit_newer subagent; standard-library CPU arithmetic",
        "script_sha256": sha(__file__), "input_files_sha256": inputs_sha256,
        "complete_valid": {"images": 1422, "studies": 859, "patients": 228, "original_labels_unchanged": True},
        "model_identity_reviews": identity_reviews, "source_validation_checks": source_checks,
        "independent_scores": independent_scores,
        "macro_auc_delta_new_minus_old": independent_scores["new_X1_B"]["macro_auc"] - independent_scores["old_AB"]["macro_auc"],
        "per_class_auc_delta_new_minus_old": delta_per_class,
        "source_bootstrap_reference": "A_B_class1_B_baseline",
        "source_bootstrap_new_minus_old": source_bootstrap,
        "independent_patient_bootstrap": independent_bootstrap, "training_resources": resources,
        "training_attempt_accounting": {"completed_registered_runs": 2, "completed_epochs_total": 8,
            "additional_unpacked_X1_interruption": "0 complete epochs, 0 valid; preserved in X1_unpacked_interruption.json"},
        "audit_actions": {"GPU_calls": 0, "model_forwards": 0, "test_reads": 0,
                          "models_loaded": 0, "training": 0, "source_files_modified": 0},
        "conclusions": {"target_0_82_reached": False,
            "new_single_model_improvement": False,
            "stable_generalization_gain_supported": False,
            "release_recommendation": "Keep X1+B as an experimental candidate. Complete actual combined full-valid/latency validation, but do not automatically replace delivered versions or claim platform improvement from this small point difference.",
            "additional_training_or_weight_search_recommended_this_round": False},
        "limitations": ["Historical checkpoints and class-1 fusion rule were selected on reused official valid; neither bootstrap corrects selection bias.",
            "Independent RNG results differ slightly from production by design; a positive-bootstrap fraction is descriptive, not a posterior probability or calibrated p-value.",
            "Only class 9 has an individually positive source CI against old AB; no multiplicity correction was applied across classes and fixed candidates.",
            "Combined live-model validation, end-to-end latency and real T4 certification are separate pending checks; cached fusion is insufficient for deployment.",
            "Full local checkpoint blobs were not rehashed or loaded in this audit. Actual persistent FP16 weights and configuration files were independently rehashed.",
            "This review is written before the root agent's subsequent actual combined-predictor validation and latency measurements."]}
    output = EVIDENCE / "independent_training_review.json"
    output.write_text(json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"review_path": str(output), "macro_auc_delta": review["macro_auc_delta_new_minus_old"],
        "independent_ci95": independent_bootstrap["macro_auc_delta_ci95"], "identity_pass": True,
        "GPU_calls": 0, "test_reads": 0}, indent=2))


def aggregate_images(manifest, image_counts, image_values):
    sums = defaultdict(lambda: [0.0] * 10)
    for row in manifest:
        key = (row["subject_id"], row["study_id"])
        for c in range(10):
            sums[key][c] += image_values[row["image_id"], c]
    return {key: [value / image_counts[key] for value in values] for key, values in sums.items()}


if __name__ == "__main__":
    main()
