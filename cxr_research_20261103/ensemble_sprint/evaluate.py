"""Exactly one fixed ensemble prediction set and one final valid comparison.

Member predictions are diagnostic records only: no alternative combination,
checkpoint, threshold or test-driven selection is evaluated here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, StudyAccumulator,
                       prediction_records, read_submission, sha256_file,
                       write_image_prediction, write_prediction)
from student_experiment_evaluation import score_probabilities, validation_labels
from sprint.evaluate import _paired_bootstrap, _write_json
from ensemble_sprint.predictor import EnsemblePredictor, MEMBER_IDS, validate_recipe

EXPECTED_VALID_SHA = "c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112"
EXPECTED_REFERENCE_SHA = "f6b47f09a80682578fe1ec11773870cb3db8ae81e2024e3026a40c4e03e7aae7"
EXPECTED_REFERENCE_MACRO = 0.8179628779375786


def selection_checks(candidate, reference):
    return {"macro_strictly_better": candidate["macro_auc"] > reference["macro_auc"],
            "class8_not_worse": candidate["per_class_auc"][7] >= reference["per_class_auc"][7]}


def _reference(config, labels, counts):
    path = Path(config["reference_predictions"]).resolve(strict=True)
    report_path = Path(config["reference_report"]).resolve(strict=True)
    identity = sha256_file(path)
    report = json.loads(report_path.read_text())
    if (identity != EXPECTED_REFERENCE_SHA or report.get("complete") is not True
            or report.get("candidate_predictions_sha256") != identity
            or report.get("manifest_sha256") != EXPECTED_VALID_SHA
            or report.get("candidate_weights_sha256") != config["base_weights_sha256"]):
        raise ValueError("Reference is not the frozen completed Sprint candidate")
    predictions = read_submission(path)
    scores = score_probabilities(labels, predictions, counts)
    if (abs(scores["macro_auc"] - EXPECTED_REFERENCE_MACRO) > 1e-12
            or not np.allclose(scores["per_class_auc"], report["candidate"]["per_class_auc"], rtol=0, atol=1e-12)):
        raise ValueError("Reference predictions do not reproduce the recorded Sprint scores")
    return predictions, scores, identity


def evaluate(config):
    config = dict(config)
    validate_recipe(config)
    manifest = Path(config["valid_manifest"]).resolve(strict=True)
    if sha256_file(manifest) != EXPECTED_VALID_SHA:
        raise ValueError("Require the unchanged complete official validation manifest")
    for field in ("reference_predictions", "reference_report", "online_best_predictions", "online_best_report"):
        if field + "_sha256" in config and sha256_file(config[field]) != config[field + "_sha256"]:
            raise ValueError("Frozen evaluation input changed: " + field)
    output = Path(config["output"]).resolve() / "final_comparison"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final comparison already exists; do not rerun or overwrite it")
    records = prediction_records(config["image_root"], manifest, split="valid")
    labels, counts = validation_labels(records)
    reference, reference_scores, reference_sha = _reference(config, labels, counts)
    online_predictions, online_scores = None, None
    if config.get("online_best_predictions"):
        from sprint.evaluate import (_reference as historical_reference,
            EXPECTED_ONLINE_BEST_SHA256, EXPECTED_ONLINE_BEST_MACRO)
        online_predictions, online_scores, _ = historical_reference(
            config["online_best_predictions"], config["online_best_report"], labels, counts,
            expected_sha=EXPECTED_ONLINE_BEST_SHA256, expected_macro=EXPECTED_ONLINE_BEST_MACRO,
            manifest_sha=EXPECTED_VALID_SHA)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "evaluation_protocol.json", {
        "configuration": config, "validation_runs": 1, "candidate_count": 1,
        "reference": "Sprint_20261104 fixed fusion", "reference_predictions_sha256": reference_sha,
        "manifest_sha256": EXPECTED_VALID_SHA, "tta": "hflip2", "all_original_study_views": True,
        "fusion_weights": config["fusion_weights"], "member_order": list(MEMBER_IDS),
        "member_predictions_diagnostic_only": True, "alternative_fusions_evaluated": 0,
        "selection_rule": "Macro-AUC strictly above Sprint and class8 AUC not below Sprint",
        "test_images_used": False, "thresholds_fitted": False, "patient_smoothing": False})
    shutil.copyfile(config["reference_predictions"], output / "baseline_predictions.csv")
    started = time.monotonic()
    try:
        predictor = EnsemblePredictor(config)
        predictions = {}
        accumulators = [StudyAccumulator(records) for _ in range(4)]
        member_predictions = [{} for _ in MEMBER_IDS]
        with ExitStack() as stack:
            paths = [output / "candidate_predictions.csv", output / "candidate.image_predictions.csv"]
            for identity in MEMBER_IDS:
                paths.extend([output / f"{identity}.predictions.csv", output / f"{identity}.image_predictions.csv"])
            streams = [stack.enter_context(path.open("x", newline="", encoding="utf-8")) for path in paths]
            writers = [csv.writer(stream) for stream in streams]
            for index, writer in enumerate(writers):
                writer.writerow(CSV_HEADER if index % 2 == 0 else IMAGE_CSV_HEADER)
            for index, row in enumerate(records, 1):
                fused, values = predictor.predict_image_components(row["path"])
                for member_index, value in enumerate([fused, *values]):
                    write_image_prediction(writers[member_index * 2 + 1], row, value)
                    result = accumulators[member_index].add(row, value)
                    if result is not None:
                        subject, study, probability = result
                        write_prediction(writers[member_index * 2], subject, study, probability)
                        target = predictions if member_index == 0 else member_predictions[member_index - 1]
                        target[(subject, study)] = probability.copy()
                if index % 100 == 0 or index == len(records):
                    print(f"[ENSEMBLE FINAL VALID] {index}/{len(records)} images", flush=True)
            for stream in streams:
                stream.flush()
                os.fsync(stream.fileno())
        if not all(accumulator.complete() for accumulator in accumulators):
            raise RuntimeError("Incomplete candidate or member Study coverage")
        predictor.synchronize()
        inference_seconds = time.monotonic() - started
        inventory = predictor.inventory
        torch = predictor.torch
        del predictor
        torch.cuda.empty_cache()
        scores = score_probabilities(labels, predictions, counts)
        deltas = [float(a-b) for a, b in zip(scores["per_class_auc"], reference_scores["per_class_auc"])]
        # A zero Tiny class1 fusion weight protects the unchanged Base path.
        class1_max_difference = max(abs(float(predictions[k][0] - reference[k][0])) for k in labels)
        if class1_max_difference > 1e-7 or abs(deltas[0]) > 1e-12:
            raise ValueError("Frozen Base class1 predictions changed relative to Sprint")
        checks = selection_checks(scores, reference_scores)
        won = all(checks.values())
        bootstrap = _paired_bootstrap(labels, predictions, reference,
            seed=int(config.get("seed", 20261105)), samples=200)
        report = {
            "complete": True, "candidate_name": "Base768_context512_opacity512",
            "baseline_name": "Sprint_20261104", "candidate": scores, "baseline": reference_scores,
            "macro_auc_delta": float(scores["macro_auc"] - reference_scores["macro_auc"]),
            "per_class_auc_delta": deltas, "class1_auc_delta": deltas[0], "class8_auc_delta": deltas[7],
            "class1_max_abs_probability_difference": class1_max_difference,
            "class1_frozen_path_preserved": True, "bootstrap": bootstrap,
            "candidate_won": won, "selection_checks": checks,
            "fusion_weights": config["fusion_weights"], "fixed_classwise_probability_weights": config["fusion_weights"],
            "member_order": list(MEMBER_IDS), "member_scores_diagnostic_only": {
                name: score_probabilities(labels, probability, counts)
                for name, probability in zip(MEMBER_IDS, member_predictions)},
            "alternative_fusions_evaluated": 0, "validation_images": len(records),
            "manifest_sha256": EXPECTED_VALID_SHA, "baseline_predictions_sha256": reference_sha,
            "candidate_predictions_sha256": sha256_file(output / "candidate_predictions.csv"),
            "candidate_image_predictions_sha256": sha256_file(output / "candidate.image_predictions.csv"),
            "member_prediction_sha256": {name: sha256_file(output / f"{name}.predictions.csv") for name in MEMBER_IDS},
            "candidate_inference_assets": inventory["files"], "candidate_inference_bytes": inventory["total_bytes"],
            "asset_inventory": inventory, "size_pass": True,
            "validation_elapsed_seconds": inference_seconds, "elapsed_is_official_latency_benchmark": False,
            "test_images_used": False, "thresholds_fitted": False, "tta": "hflip2",
            "formal_T4_certification": False, "deployment_accepted": False,
            "selection_note": "Local validation improvement does not establish online improvement."}
        if online_scores is not None:
            report["submission_best_041"] = online_scores
            report["macro_auc_delta_vs_041"] = scores["macro_auc"] - online_scores["macro_auc"]
        _write_json(output / "report.json", report)
        selection = {"selected": "ensemble_sprint" if won else "sprint_candidate",
            "selected_current": "ensemble_sprint" if won else "sprint_candidate", "candidate_won": won,
            "selection_checks": checks, "candidate_macro_auc": scores["macro_auc"],
            "baseline_macro_auc": reference_scores["macro_auc"], "class8_auc_delta": deltas[7],
            "report": str(output / "report.json"), "fusion_weights": config["fusion_weights"],
            "candidate_inference_assets": inventory["files"], "tta": "hflip2",
            "research_selection_only": True, "deployment_accepted": False}
        for name in ("selection.json", "selected_current.json", "current_model.json"):
            _write_json(output / name, selection)
        print(f"[ENSEMBLE RESULT] old={reference_scores['macro_auc']:.9f} new={scores['macro_auc']:.9f} "
              f"class8_delta={deltas[7]:+.9f} selected={selection['selected']}", flush=True)
        return report
    except BaseException as error:
        _write_json(output / "failure.json", {"complete": False, "error_type": type(error).__name__,
            "message": str(error), "selection_changed": False, "test_images_used": False,
            "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
