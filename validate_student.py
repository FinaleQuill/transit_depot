"""Evaluate the exported FP16 deployment on the untouched official validation set.

Uses exactly Predictor.predict_image, matching predict.py/cost.py. Never fits
thresholds, changes the deployment, reads test labels, or selects a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deploy_kd import (CSV_HEADER, Predictor, StudyAccumulator, prediction_records,
                       sha256_file, write_json, write_prediction, deployment_members)


def validate_student(*, model_dir: str | Path, image_root: str | Path,
                     manifest: str | Path, output_dir: str | Path) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new validation output directory")
    if destination.is_relative_to(Path(image_root).resolve()):
        raise ValueError("Validation output cannot be inside the read-only image root")
    predictor = Predictor(model_dir)
    manifest_hash = sha256_file(manifest)
    if manifest_hash != predictor.config.get("official_valid_manifest_sha256"):
        raise ValueError("Validation manifest differs from the student's declared official holdout")
    records = prediction_records(image_root, manifest, split="valid")
    labels = {}
    for row in records:
        target = json.loads(row["targets"])
        if len(target) != 10 or any(value not in (0, 1) for value in target):
            raise ValueError("Official valid must retain all ten original binary labels")
        key = row["study_key"]
        if key in labels and labels[key] != target:
            raise ValueError("Inconsistent labels within a validation Study")
        labels[key] = target
    accumulator = StudyAccumulator(records)
    destination.mkdir(parents=True, exist_ok=False)
    targets, probabilities = [], []
    with (destination / "valid_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        for row in records:
            result = accumulator.add(row, predictor.predict_image(row["path"]))
            if result is not None:
                subject, study, values = result
                write_prediction(writer, subject, study, values)
                targets.append(labels[f"{subject}/{study}"])
                probabilities.append(values)
    if not accumulator.complete():
        raise RuntimeError("Validation image coverage is incomplete")
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if any(len(np.unique(targets[:, c])) != 2 for c in range(10)):
        raise ValueError("Every validation class requires positive and negative Studies")
    aucs = [float(roc_auc_score(targets[:, c], probabilities[:, c])) for c in range(10)]
    macro_auc = float(np.mean(aucs))
    source = predictor.config.get("source_validation", {})
    source_auc = source.get("macro_auc")
    members = deployment_members(predictor.config)
    report = {"complete": True, "precision": "float16", "image_size": predictor.config["image_size"],
              "image_sizes": sorted({m["image_size"] for m in members}),
              "tta": predictor.config["tta"], "ensemble": predictor.config["ensemble"], "batch_size": 1,
              "member_count": len(members),
              "members": [{k: m.get(k) for k in ("source_checkpoint_sha256", "weights_sha256", "image_size", "tta", "seed")} for m in members],
              "deployment_config_sha256": sha256_file(Path(model_dir) / "config.json"),
              "predictions_sha256": sha256_file(destination / "valid_predictions.csv"),
              "num_images": len(records), "num_studies": len(labels),
              "macro_auc": macro_auc, "per_class_auc": aucs,
              "macro_f1_fixed_0_5": float(f1_score(targets, probabilities >= 0.5,
                                                    average="macro", zero_division=0)),
              "thresholds_fitted": False, "checkpoint_selection_performed": False,
              "manifest_sha256": manifest_hash,
              "deployment_inventory": predictor.inventory,
              "source_training_validation": source,
              "auc_difference_from_source": None if source_auc is None else macro_auc - float(source_auc),
              "note": "FP16 deployment validation is distinct from training-precision validation; no test data is used."}
    write_json(destination / "validation_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent /
                        "data/clean_v1_reviewed/valid_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(validate_student(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
