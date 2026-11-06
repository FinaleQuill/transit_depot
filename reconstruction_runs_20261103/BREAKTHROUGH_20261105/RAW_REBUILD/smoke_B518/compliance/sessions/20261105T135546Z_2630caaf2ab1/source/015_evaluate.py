"""Study-level metrics for the reconstruction's train-only development splits.

No image loading, model execution, threshold fitting, or external split access occurs
here. The caller must supply one row per complete Study and enforce its split ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

NUM_CLASSES = 10
ALLOWED_SCOPES = frozenset({"inner_validation", "train_diagnostic", "synthetic_test"})


def json_native(value: Any) -> Any:
    """Return JSON-native data, representing undefined floating values as null."""
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _identities(values: Sequence[str], count: int, name: str) -> np.ndarray:
    items = list(values)
    if len(items) != count or any(not isinstance(x, str) or not x.strip() for x in items):
        raise ValueError(f"{name} must contain {count} nonempty strings")
    return np.asarray(items, dtype=np.str_)


def _views(values: Sequence[Sequence[str] | str], count: int) -> list[list[str]]:
    if len(values) != count:
        raise ValueError("views must contain one projection list per Study")
    normalized = []
    for raw in values:
        if isinstance(raw, str):
            raw = json.loads(raw) if raw.strip().startswith("[") else [raw]
        if not isinstance(raw, (list, tuple, np.ndarray)) or not len(raw):
            raise ValueError("Every Study requires a nonempty list of image projections")
        if any(not isinstance(v, str) for v in raw):
            raise ValueError("Projection names must be strings; use Missing for unknown views")
        normalized.append([v.strip().upper() or "MISSING" for v in raw])
    return normalized


def _validated(probabilities, targets, study_keys, patient_ids, views):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    raw_targets = np.asarray(targets)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError("probabilities must have shape [N, 10]")
    if len(probabilities) == 0:
        raise ValueError("Evaluation requires at least one complete Study")
    if raw_targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have identical [N, 10] shapes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.isin(raw_targets, [0, 1]).all():
        raise ValueError("Targets must contain original binary labels only")
    keys = _identities(study_keys, len(probabilities), "study_keys")
    patients = _identities(patient_ids, len(probabilities), "patient_ids")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("study_keys must be unique: aggregate images before evaluation")
    return probabilities, raw_targets.astype(np.uint8), keys, patients, _views(views, len(keys))


class _RankedClass:
    """Fixed probability ordering supporting fast patient-weighted AUC and AP."""

    def __init__(self, labels: np.ndarray, probabilities: np.ndarray):
        self.order = np.argsort(probabilities, kind="stable")
        self.labels = labels[self.order].astype(np.float64)
        sorted_scores = probabilities[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1] if len(labels) else np.array([], dtype=int)

    def score(self, weights: np.ndarray) -> tuple[float | None, float | None]:
        if not len(self.starts):
            return None, None
        ordered_weights = weights[self.order]
        positives = np.add.reduceat(ordered_weights * self.labels, self.starts)
        negatives = np.add.reduceat(ordered_weights * (1.0 - self.labels), self.starts)
        p, n = float(positives.sum()), float(negatives.sum())
        auc = None
        if p > 0 and n > 0:
            negatives_before = np.cumsum(negatives) - negatives
            auc = float(np.dot(positives, negatives_before + negatives * 0.5) / (p * n))
        ap = None
        if p > 0:
            # Equal-score rows enter the precision/recall curve together.
            descending_positive = positives[::-1]
            denominator = np.cumsum((positives + negatives)[::-1])
            precision = np.divide(np.cumsum(descending_positive), denominator,
                                  out=np.zeros_like(denominator), where=denominator > 0)
            ap = float(np.dot(descending_positive, precision) / p)
        return auc, ap


def _macro(values: Sequence[float | None]) -> float | None:
    return float(np.mean(values)) if all(v is not None for v in values) else None


def _available_macro(values: Sequence[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    return float(np.mean(usable)) if usable else None


def _scores(ranks, weights):
    scores = [r.score(weights) for r in ranks]
    aucs, aps = [v[0] for v in scores], [v[1] for v in scores]
    return {"macro_auc": _macro(aucs), "per_class_auc": aucs,
            "macro_average_precision": _macro(aps), "per_class_average_precision": aps,
            "available_class_macro_auc": _available_macro(aucs),
            "available_class_macro_average_precision": _available_macro(aps),
            "num_auc_evaluable_classes": sum(v is not None for v in aucs)}


def _interval(values):
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return {"ci95": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
            "valid_replicates": int(len(valid)), "attempted_replicates": len(values)}


def evaluate_studies(probabilities, targets, study_keys, patient_ids, views, *,
                     bootstrap_repeats: int = 500, seed: int = 20261103,
                     reference_probabilities=None, scope: str = "inner_validation") -> dict:
    """Evaluate Nx10 probabilities; AP-only refers to the AP projection, not AP metric.

    ``views[i]`` lists the projection of *every retained image* of Study i. Unknown
    projections must remain Missing. A Study enters ap_projection_only only when all
    its image projections are AP. Bootstrap samples patients with replacement and
    keeps all of their Studies, including each sampled patient's multiplicity.

    Missing positive/negative support yields null AUC. The selection metric macro_auc
    is null unless all ten classes are evaluable; available_class_macro_auc is only a
    diagnostic and must not be substituted for the ten-class selection metric.
    """
    if scope not in ALLOWED_SCOPES:
        raise ValueError("Only registered internal train development splits are enabled")
    if type(bootstrap_repeats) is not int or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    probabilities, targets, keys, patients, projections = _validated(
        probabilities, targets, study_keys, patient_ids, views)
    if reference_probabilities is not None:
        reference_probabilities = _validated(reference_probabilities, targets, keys, patients, projections)[0]
    count = len(keys)
    image_counts = np.asarray([len(v) for v in projections])
    masks = {"all": np.ones(count, dtype=bool),
             "ap_projection_only": np.asarray([all(v == "AP" for v in row) for row in projections]),
             "single_image_study": image_counts == 1,
             "multiple_image_study": image_counts > 1}
    unique_patients, patient_indices = np.unique(patients, return_inverse=True)
    slice_reports, ranked, reference_ranked = {}, {}, {}
    for name, mask in masks.items():
        local_targets, local_probabilities = targets[mask], probabilities[mask]
        ranked[name] = [_RankedClass(local_targets[:, c], local_probabilities[:, c]) for c in range(NUM_CLASSES)]
        stats = _scores(ranked[name], np.ones(int(mask.sum())))
        stats.update(num_studies=int(mask.sum()), num_patients=int(len(np.unique(patients[mask]))),
                     positive_studies=local_targets.sum(axis=0).astype(int).tolist(),
                     positive_patients=[int(len(np.unique(patients[mask][local_targets[:, c] == 1])))
                                        for c in range(NUM_CLASSES)])
        if reference_probabilities is not None:
            reference_ranked[name] = [_RankedClass(local_targets[:, c], reference_probabilities[mask, c])
                                      for c in range(NUM_CLASSES)]
            reference = _scores(reference_ranked[name], np.ones(int(mask.sum())))
            stats["reference"] = reference
            stats["macro_auc_delta"] = (stats["macro_auc"] - reference["macro_auc"]
                                         if stats["macro_auc"] is not None and reference["macro_auc"] is not None else None)
        slice_reports[name] = stats
    bootstrap_values = {name: {"macro_auc": [], "per_class_auc": [[] for _ in range(NUM_CLASSES)],
                               "delta_macro_auc": [], "delta_per_class_auc": [[] for _ in range(NUM_CLASSES)]}
                        for name in masks}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_repeats):
        patient_counts = np.bincount(rng.integers(0, len(unique_patients), size=len(unique_patients)),
                                     minlength=len(unique_patients))
        study_weights = patient_counts[patient_indices]
        for name, mask in masks.items():
            aucs = [rank.score(study_weights[mask])[0] for rank in ranked[name]]
            macro = _macro(aucs)
            record = bootstrap_values[name]
            record["macro_auc"].append(macro)
            for c, value in enumerate(aucs):
                record["per_class_auc"][c].append(value)
            if reference_probabilities is not None:
                references = [rank.score(study_weights[mask])[0] for rank in reference_ranked[name]]
                ref_macro = _macro(references)
                record["delta_macro_auc"].append(macro - ref_macro if macro is not None and ref_macro is not None else None)
                for c, (value, reference) in enumerate(zip(aucs, references)):
                    record["delta_per_class_auc"][c].append(value - reference if value is not None and reference is not None else None)
    if bootstrap_repeats:
        for name, record in bootstrap_values.items():
            result = {"macro_auc": _interval(record["macro_auc"]),
                      "per_class_auc": [_interval(v) for v in record["per_class_auc"]]}
            if reference_probabilities is not None:
                result["paired_delta_macro_auc"] = _interval(record["delta_macro_auc"])
                result["paired_delta_per_class_auc"] = [_interval(v) for v in record["delta_per_class_auc"]]
            slice_reports[name]["patient_bootstrap"] = result
    return json_native({"schema_version": 1, "scope": scope, "unit": "Study",
                        "class_ids": list(range(1, 11)), "num_studies": count,
                        "num_patients": len(unique_patients), "num_images": int(image_counts.sum()),
                        **{k: v for k, v in slice_reports["all"].items()
                           if k in ["macro_auc", "per_class_auc", "macro_average_precision", "per_class_average_precision"]},
                        "overall": slice_reports["all"], "slices": slice_reports,
                        "bootstrap": {"sampling_unit": "patient_with_all_studies_and_multiplicity",
                                      "repeats": bootstrap_repeats, "seed": seed, "paired": reference_probabilities is not None,
                                      "interval": "percentile_95", "selection_bias_corrected": False},
                        "ap_projection_only_definition": "All retained image projections of the Study equal AP; unknown views excluded",
                        "macro_auc_definition": "Unweighted mean of all 10 class AUCs; null if any class lacks positive or negative support",
                        "test_accessed": False, "official_validation_accessed": False,
                        "holdout_H_accessed": False, "thresholds_fitted": False})


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_evaluation(output_dir, probabilities, targets, study_keys, patient_ids, views, *,
                    bootstrap_repeats: int = 500, seed: int = 20261103,
                    reference_probabilities=None, scope: str = "inner_validation", metadata=None) -> dict:
    """Atomically write portable, pickle-free predictions.npz and report.json."""
    report = evaluate_studies(probabilities, targets, study_keys, patient_ids, views,
                              bootstrap_repeats=bootstrap_repeats, seed=seed,
                              reference_probabilities=reference_probabilities, scope=scope)
    probabilities, targets, keys, patients, projections = _validated(
        probabilities, targets, study_keys, patient_ids, views)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"probabilities": probabilities, "targets": targets, "study_keys": keys,
               "patient_ids": patients, "views_json": np.asarray([json.dumps(v) for v in projections]),
               "image_counts": np.asarray([len(v) for v in projections], dtype=np.int64)}
    if reference_probabilities is not None:
        payload["reference_probabilities"] = np.asarray(reference_probabilities, dtype=np.float64)
    with tempfile.NamedTemporaryFile(dir=destination, prefix=".predictions.", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination / "predictions.npz")
    report["predictions_sha256"] = _file_hash(destination / "predictions.npz")
    report["metadata"] = json_native(metadata or {})
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination, prefix=".report.", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination / "report.json")
    return report
