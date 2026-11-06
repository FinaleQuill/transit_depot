"""One final, fixed validation comparison; no training or test-set access.

Only the final Sprint EMA plus the frozen Tiny is run, once, with fixed hflip2.
That one prediction set is compared with two authenticated, existing validation
CSVs: the user's actual submission_best (041/A) and the historical local best.
The test submission is hashed only to establish identity, never parsed or scored.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from deploy_kd import (
    CLASS_IDS, CSV_HEADER, IMAGE_CSV_HEADER, MAX_MODEL_BYTES, StudyAccumulator,
    _letterbox_image, aggregate_member_probabilities, prediction_records,
    read_submission, sha256_file, write_image_prediction, write_prediction,
)
from student_experiment_evaluation import score_probabilities, validation_labels

EXPECTED_MANIFEST_SHA256 = "c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112"
EXPECTED_BASELINE_SHA256 = "050d3ff3dba70db0377253abdb6bb9227693b4f9d5a79d4e1bf77a1f4e29ae7c"
EXPECTED_BASELINE_MACRO = 0.817614546496497
EXPECTED_ONLINE_BEST_SHA256 = "672cd43cb85c2bc7544fcc0dac0550c3f6b8f878fabc4b80cf20684710bee900"
EXPECTED_ONLINE_BEST_MACRO = 0.8091159408519895
EXPECTED_SUBMISSION_BEST_SHA256 = "44dde95be64f42c151af7b23ee4a81ffb2158822eabb9f6249a59a73f795f66e"
WORK_ROOT = Path(__file__).resolve().parents[2]
ONLINE_BEST_DIRECTORY = WORK_ROOT / "cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/reports/A/valid_hflip2"
SELECTION_RULE = ("candidate Macro-AUC strictly exceeds both submission_best_041 and historical local best; "
                  "class 1 AUC is at least historical local best; otherwise retain submission_best_041")
MODEL_WEIGHTS = [[0.0] + [0.5] * 9, [1.0] + [0.5] * 9]


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _reference(csv_path, report_path, labels, counts, *, expected_sha, expected_macro, manifest_sha):
    """Authenticate and rescore an existing official-valid prediction artifact."""
    identity = sha256_file(csv_path)
    report = json.loads(Path(report_path).read_text())
    if (identity != expected_sha or report.get("predictions_sha256") != identity
            or report.get("manifest_sha256") != manifest_sha or report.get("complete") is not True):
        raise ValueError("Historical validation provenance does not match: " + str(csv_path))
    predictions = read_submission(csv_path)
    scores = score_probabilities(labels, predictions, counts)
    if (abs(scores["macro_auc"] - expected_macro) > 1e-12
            or not np.allclose(scores["per_class_auc"], report["per_class_auc"], atol=1e-12, rtol=0)):
        raise ValueError("Historical validation CSV does not reproduce its frozen scores")
    return predictions, scores, identity


def selection_checks(candidate, online_best, local_best):
    """Fixed decision only; no weights, thresholds or epochs are searched."""
    return {
        "beats_submission_best_macro": candidate["macro_auc"] > online_best["macro_auc"],
        "beats_historical_local_best_macro": candidate["macro_auc"] > local_best["macro_auc"],
        "class1_not_below_historical_local_best": candidate["per_class_auc"][0] >= local_best["per_class_auc"][0],
    }


def _paired_bootstrap(labels, candidate, reference, *, seed, samples=200):
    """Patient-cluster uncertainty from the sole saved prediction pair."""
    from sklearn.metrics import roc_auc_score

    keys = sorted(labels)
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    new = np.asarray([candidate[key] for key in keys], dtype=np.float64)
    old = np.asarray([reference[key] for key in keys], dtype=np.float64)
    patients = sorted({key[0] for key in keys})
    groups = [np.asarray([i for i, key in enumerate(keys) if key[0] == patient])
              for patient in patients]
    rng = np.random.default_rng(seed)
    deltas, attempts = [], 0
    while len(deltas) < samples and attempts < samples * 5:
        attempts += 1
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        y = targets[indices]
        if any(np.unique(y[:, c]).size != 2 for c in range(10)):
            continue
        delta = [float(roc_auc_score(y[:, c], new[indices, c])
                       - roc_auc_score(y[:, c], old[indices, c])) for c in range(10)]
        deltas.append([float(np.mean(delta)), *delta])
    values = np.asarray(deltas, dtype=np.float64)
    if not len(values):
        return {"samples": 0, "attempts": attempts, "seed": seed, "interval": None}
    limits = np.quantile(values, [.025, .975], axis=0)
    return {"samples": len(deltas), "attempts": attempts, "seed": seed,
            "macro_auc_delta_95_percentile_interval": limits[:, 0].tolist(),
            "per_class_auc_delta_95_percentile_intervals": limits[:, 1:].T.tolist(),
            "interpretation": "Exploratory paired interval; historical validation selection is not corrected."}


def _candidate(config, torch):
    from sprint.model import SprintStudyModel

    weights = Path(config["candidate_weights"]).resolve(strict=True)
    expected = config.get("candidate_weights_sha256")
    identity = sha256_file(weights)
    if expected is not None and identity != expected:
        raise ValueError("Candidate weights differ from the fixed configuration")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"state_dict", "configuration"} <= payload.keys():
        raise ValueError("Require a Sprint final EMA export with state_dict/configuration")
    configuration = payload["configuration"]
    if (not isinstance(configuration, dict) or configuration.get("image_size", 768) != 768
            or configuration.get("aggregation", "mean") != "mean"):
        raise ValueError("Sprint final evaluation requires the frozen 768 mean architecture")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Sprint export has no model tensors")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.is_complex()
                or tensor.is_floating_point() and (
                    tensor.dtype != torch.float16 or not bool(torch.isfinite(tensor).all()))):
            raise ValueError("Require finite FP16 export tensors: " + str(name))
    model = SprintStudyModel(
        backbone_name=configuration.get("backbone_name", "convnext_base.fb_in22k_ft_in1k_384"),
        pretrained_path=None, spatial_dim=configuration.get("spatial_dim", 256),
        aggregation="mean", gradient_checkpointing=False, view_chunk_size=1,
        channels_last=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to("cuda", dtype=torch.float16,
                                        memory_format=torch.channels_last)
    del payload, state
    return model, configuration, identity


def _tiny(config):
    from frozen_head_predictor import FrozenHeadPredictor

    entry = config.get("tiny_patch_config", config.get("tiny_patch"))
    if not entry:
        raise ValueError("Declare the frozen A_F2 patch in tiny_patch_config or tiny_patch")
    path = Path(entry).resolve(strict=True)
    root = path if path.is_dir() else path.parent
    tiny = FrozenHeadPredictor(root, device="cuda")
    if (tiny.config["backbone_name"] != "convnext_tiny.fb_in22k_ft_in1k_384"
            or tiny.config["official_valid_manifest_sha256"] != EXPECTED_MANIFEST_SHA256
            or tiny.config["patch_sha256"] !=
            "fd92ffc02908f22bd68e4ddfb2b191c8899230fe058cf9b576bcf61b23cf1fa3"):
        raise ValueError("The frozen candidate Tiny must be the registered A_F2 model")
    return tiny


def evaluate(config) -> dict:
    """Evaluate exactly the registered final fusion and select on Macro-AUC.

    ``output`` must not already contain results. No epoch selection, alternative
    blend, threshold fitting, test prediction or deployment acceptance occurs.
    Selection changes only this run's current_model.json research pointer.
    """
    config = dict(config)
    baseline = Path(config["baseline_directory"]).resolve(strict=True)
    manifest = Path(config["valid_manifest"]).resolve(strict=True)
    output = Path(config["output"]).resolve()
    manifest_sha = sha256_file(manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Require the immutable original official validation manifest")
    reference_csv = Path(config.get("baseline_predictions", baseline / "valid_predictions.csv")).resolve(strict=True)
    reference_report = Path(config.get("baseline_report", baseline / "validation_report.json")).resolve(strict=True)
    online_directory = Path(config.get("online_best_directory", ONLINE_BEST_DIRECTORY)).resolve(strict=True)
    online_csv = Path(config.get("online_best_predictions", online_directory / "valid_predictions.csv")).resolve(strict=True)
    online_report = Path(config.get("online_best_report", online_directory / "validation_report.json")).resolve(strict=True)
    submission_best = Path(config.get("submission_best", WORK_ROOT / "submission_best.csv")).resolve(strict=True)
    # Identity only: test probabilities are neither parsed nor used for selection.
    if sha256_file(submission_best) != EXPECTED_SUBMISSION_BEST_SHA256:
        raise ValueError("submission_best.csv is no longer the registered 041/A submission")
    records = prediction_records(config["image_root"], manifest, split="valid")
    labels, counts = validation_labels(records)
    reference, baseline_scores, reference_sha = _reference(reference_csv, reference_report, labels, counts,
        expected_sha=EXPECTED_BASELINE_SHA256, expected_macro=EXPECTED_BASELINE_MACRO, manifest_sha=manifest_sha)
    online_reference, online_scores, online_sha = _reference(online_csv, online_report, labels, counts,
        expected_sha=EXPECTED_ONLINE_BEST_SHA256, expected_macro=EXPECTED_ONLINE_BEST_MACRO, manifest_sha=manifest_sha)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final comparison output is already occupied; do not rerun or overwrite it")
    if any(output.is_relative_to(path) for path in
           (Path(config["image_root"]).resolve(), baseline, online_directory, Path(config["candidate_weights"]).resolve())):
        raise ValueError("Evaluation output must not overwrite inputs")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "evaluation_protocol.json", {
        "configuration": config, "baseline": "submission_best_041_A", "candidate": "A_F2+Sprint_final_EMA",
        "historical_local_best": "A_F2+B_F2",
        "manifest_sha256": manifest_sha, "baseline_predictions_sha256": online_sha,
        "historical_local_best_predictions_sha256": reference_sha,
        "submission_best_sha256": EXPECTED_SUBMISSION_BEST_SHA256,
        "fixed_classwise_probability_weights": MODEL_WEIGHTS, "tta": "hflip2",
        "image_size": 768, "random_augmentation": False, "patient_smoothing": False,
        "selection_rule": SELECTION_RULE,
        "validation_runs": 1, "epoch_search": False, "test_images_used": False,
        "test_submission_identity_hash_only": True, "test_predictions_parsed": False,
    })
    shutil.copyfile(online_csv, output / "baseline_predictions.csv")
    shutil.copyfile(reference_csv, output / "historical_local_best_predictions.csv")
    started = time.monotonic()
    try:
        from inference_runtime import activate_dependencies
        activate_dependencies()
        import torch
        from inference_inputs import load_image_pixels, member_results_to_cpu

        if not torch.cuda.is_available():
            raise RuntimeError("Final FP16 validation requires CUDA")
        seed = int(config.get("seed", 57))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tiny = _tiny(config)
        model, model_config, candidate_sha = _candidate(config, torch)
        weights = Path(config["candidate_weights"]).resolve(strict=True)
        asset_paths = {weights, *(Path(item["path"]).resolve() for item in tiny.inventory["files"])}
        assets = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in sorted(asset_paths)]
        model_bytes = sum(item["bytes"] for item in assets)
        if model_bytes > MAX_MODEL_BYTES:
            raise ValueError("Actual candidate weights and required Tiny files exceed 500MB")
        predictions, accumulator = {}, StudyAccumulator(records)
        with (output / "candidate_predictions.csv").open("x", newline="", encoding="utf-8") as stream, \
             (output / "candidate.image_predictions.csv").open("x", newline="", encoding="utf-8") as image_stream:
            writer, image_writer = csv.writer(stream), csv.writer(image_stream)
            writer.writerow(CSV_HEADER)
            image_writer.writerow(IMAGE_CSV_HEADER)
            for index, row in enumerate(records, 1):
                pixels = load_image_pixels(row["path"], [768], _letterbox_image)[768]
                with torch.inference_mode():
                    image = tiny.input_lookup.to_tensor(pixels)
                    views = [image, torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)]
                    means, finite = [], []
                    for member in (tiny.model, model):
                        probabilities = []
                        for view in views:
                            logits = (member(view) if member is tiny.model else
                                      member(view, return_components=True)["image_logits"])
                            if tuple(logits.shape) != (1, 10):
                                raise ValueError("Candidate member must produce ten single-image logits")
                            finite.append(torch.isfinite(logits).all())
                            probabilities.append(torch.sigmoid(logits.float())[0])
                        means.append(torch.stack(probabilities).mean(0))
                    values = member_results_to_cpu(torch, means, finite)
                probability = aggregate_member_probabilities(values, MODEL_WEIGHTS)
                write_image_prediction(image_writer, row, probability)
                result = accumulator.add(row, probability)
                if result is not None:
                    subject, study, value = result
                    predictions[(subject, study)] = value.copy()
                    write_prediction(writer, subject, study, value)
                if index % 100 == 0 or index == len(records):
                    print(f"[SPRINT FINAL VALID] {index}/{len(records)} images", flush=True)
            stream.flush()
            image_stream.flush()
            os.fsync(stream.fileno())
            os.fsync(image_stream.fileno())
        if not accumulator.complete():
            raise RuntimeError("Incomplete official validation coverage")
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - started
        del model, tiny
        torch.cuda.empty_cache()
        candidate_scores = score_probabilities(labels, predictions, counts)
        delta = float(candidate_scores["macro_auc"] - online_scores["macro_auc"])
        per_class = [float(a - b) for a, b in zip(candidate_scores["per_class_auc"], online_scores["per_class_auc"])]
        local_delta = float(candidate_scores["macro_auc"] - baseline_scores["macro_auc"])
        local_per_class = [float(a - b) for a, b in zip(candidate_scores["per_class_auc"], baseline_scores["per_class_auc"])]
        uncertainty = _paired_bootstrap(labels, predictions, online_reference, seed=seed, samples=200)
        local_uncertainty = _paired_bootstrap(labels, predictions, reference, seed=seed, samples=200)
        checks = selection_checks(candidate_scores, online_scores, baseline_scores)
        won = all(checks.values())
        report = {
            "complete": True, "baseline_name": "submission_best_041_A", "candidate_name": "A_F2+Sprint_final_EMA",
            "baseline": online_scores, "candidate": candidate_scores,
            "macro_auc_delta": delta, "per_class_auc_delta": per_class,
            "class1_auc_delta": per_class[0], "candidate_won": won,
            "selection_checks": checks, "selection_rule": SELECTION_RULE,
            "historical_local_best_name": "A_F2+B_F2", "historical_local_best": baseline_scores,
            "macro_auc_delta_vs_historical_local_best": local_delta,
            "per_class_auc_delta_vs_historical_local_best": local_per_class,
            "class1_auc_delta_vs_historical_local_best": local_per_class[0],
            "bootstrap_vs_historical_local_best": local_uncertainty,
            "bootstrap": uncertainty, "validation_images": len(records),
            "manifest_sha256": manifest_sha, "baseline_predictions_sha256": online_sha,
            "historical_local_best_predictions_sha256": reference_sha,
            "submission_best_sha256": EXPECTED_SUBMISSION_BEST_SHA256,
            "candidate_predictions_sha256": sha256_file(output / "candidate_predictions.csv"),
            "candidate_image_predictions_sha256": sha256_file(output / "candidate.image_predictions.csv"),
            "candidate_weights_sha256": candidate_sha, "model_configuration": model_config,
            "candidate_inference_assets": assets, "candidate_inference_bytes": model_bytes,
            "size_pass": True, "formal_T4_certification": False, "deployment_accepted": False,
            "validation_elapsed_seconds": inference_seconds,
            "elapsed_is_official_latency_benchmark": False,
            "test_images_used": False, "thresholds_fitted": False,
            "test_submission_identity_hash_only": True, "test_predictions_parsed": False,
            "fixed_classwise_probability_weights": MODEL_WEIGHTS,
            "tta": "hflip2", "patient_smoothing": False,
            "selection_note": "A local validation win is not proof of an online improvement.",
        }
        _write_json(output / "report.json", report)
        selection = {
            "selected": "sprint_candidate" if won else "submission_best_041_A",
            "selected_current": "sprint_candidate" if won else "submission_best_041_A",
            "deployment_preferred": "sprint_candidate_pending_acceptance" if won else "submission_best_041_A",
            "candidate_won": won, "baseline_macro_auc": online_scores["macro_auc"],
            "candidate_macro_auc": candidate_scores["macro_auc"], "macro_auc_delta": delta,
            "historical_local_best_macro_auc": baseline_scores["macro_auc"],
            "macro_auc_delta_vs_historical_local_best": local_delta,
            "selection_checks": checks, "rule": SELECTION_RULE,
            "candidate_weights": str(weights), "candidate_weights_sha256": candidate_sha,
            "baseline_directory": str(online_directory), "historical_local_best_directory": str(baseline),
            "baseline_submission": str(submission_best), "baseline_submission_sha256": EXPECTED_SUBMISSION_BEST_SHA256,
            "tiny_patch_config": str(Path(config.get("tiny_patch_config", config.get("tiny_patch"))).resolve()),
            "fixed_classwise_probability_weights": MODEL_WEIGHTS, "tta": "hflip2",
            "report": str(output / "report.json"), "research_selection_only": True,
            "deployment_accepted": False,
            "deployment_requirements": ["official mean <=100ms per Study", "actual inference assets <=500MB",
                                        "predict/cost/submission consistency and complete compliance materials"],
        }
        _write_json(output / "selection.json", selection)
        _write_json(output / "current_model.json", selection)
        _write_json(output / "selected_current.json", selection)
        print(f"[SPRINT RESULT] submission_best={online_scores['macro_auc']:.9f} "
              f"historical_local_best={baseline_scores['macro_auc']:.9f} "
              f"new={candidate_scores['macro_auc']:.9f} delta_vs_local_best={local_delta:+.9f}; "
              f"selected={selection['selected']}", flush=True)
        return report
    except BaseException as error:
        _write_json(output / "failure.json", {"complete": False, "error_type": type(error).__name__,
                    "message": str(error), "elapsed_seconds": time.monotonic() - started,
                    "selection_changed": False, "test_images_used": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
