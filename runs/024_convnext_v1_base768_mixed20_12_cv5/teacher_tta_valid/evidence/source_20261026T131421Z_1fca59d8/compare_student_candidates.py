"""Compare saved official-valid probabilities, including fixed soft ensembles.

No model is loaded. No weights, thresholds, or test statistics are fitted.
Patient-cluster paired bootstrap is optional and uses the same resample for all
candidates. An AUC comparison does not certify T4 latency or choose deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deploy_kd import read_json, read_submission, sha256_file, write_json


def class_aucs(targets, probabilities):
    from sklearn.metrics import roc_auc_score
    return [float(roc_auc_score(targets[:, c], probabilities[:, c]))
            if len(np.unique(targets[:, c])) == 2 else None for c in range(10)]


def normalized_weights(values, count):
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape == (count,):
        weights = np.repeat(weights[:, None], 10, axis=1)
    if weights.shape != (count, 10) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Fixed ensemble weights must be nonnegative M or Mx10 values")
    sums = weights.sum(0)
    if (sums <= 0).any():
        raise ValueError("Every class requires a positive total weight")
    return weights / sums


def paired_intervals(targets, candidates, subjects, *, samples, seed):
    if type(samples) is not int or not 0 <= samples <= 10000:
        raise ValueError("Bootstrap samples must be in 0..10000")
    if not samples:
        return {}
    groups = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    rng = np.random.default_rng(seed)
    names = list(candidates)
    deltas = {name: [] for name in names[1:]}
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        reference = class_aucs(targets[indices], candidates[names[0]][indices])
        for name in names[1:]:
            auc = class_aucs(targets[indices], candidates[name][indices])
            row = [None if a is None or b is None else a - b for a, b in zip(auc, reference)]
            macro = None if any(v is None for v in row) else float(np.mean(row))
            deltas[name].append(row + [macro])
    result = {}
    for name, rows in deltas.items():
        intervals = []
        for column in range(11):
            valid = [r[column] for r in rows if r[column] is not None]
            intervals.append({"valid_resamples": len(valid), "ci_95":
                              None if not valid else np.quantile(valid, [.025, .975]).tolist()})
        result[name] = {"per_class_delta": intervals[:10], "macro_auc_delta": intervals[10]}
    return {"reference": names[0], "samples": samples, "seed": seed,
            "sampling_unit": "patient_with_all_Studies", "paired": True, "results": result}


def compare(*, validation_dirs, manifest, output, names=None, combinations_json=None, bootstrap=0, seed=43):
    from dataset import read_manifest
    if not validation_dirs or len(validation_dirs) > 16:
        raise ValueError("Provide 1..16 saved validation runs")
    roots = [Path(p).resolve(strict=True) for p in validation_dirs]
    names = list(names) if names else [p.name for p in roots]
    if len(names) != len(roots) or len(set(names)) != len(names) or any(not x for x in names):
        raise ValueError("Candidate names must be unique and match the validation runs")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new comparison output")
    if any(output == p or output.is_relative_to(p) for p in roots):
        raise ValueError("Comparison outputs cannot be inside saved validation inputs")
    records = read_manifest(manifest, split="valid")
    labels, views = {}, {}
    for row in records:
        key = (row["subject_id"], row["study_id"])
        target = tuple(row["targets"])
        if key in labels and labels[key] != target:
            raise ValueError("Inconsistent valid labels within a Study")
        labels[key] = target
        views[key] = views.get(key, 0) + 1
    keys = sorted(labels)
    targets = np.asarray([labels[k] for k in keys], dtype=np.int8)
    subjects = np.asarray([k[0] for k in keys])
    manifest_hash = sha256_file(manifest)
    candidates, provenance = {}, []
    for name, root in zip(names, roots):
        report = read_json(root / "validation_report.json")
        if report.get("complete") is not True or report.get("manifest_sha256") != manifest_hash:
            raise ValueError("All reports must cover this complete official-valid manifest")
        prediction_hash = sha256_file(root / "valid_predictions.csv")
        if report.get("predictions_sha256") is not None and report["predictions_sha256"] != prediction_hash:
            raise ValueError("Saved probabilities differ from the validation report's SHA256 binding")
        submitted = read_submission(root / "valid_predictions.csv")
        if set(submitted) != set(keys):
            raise ValueError("All predictions must cover exactly the same original valid Studies")
        values = np.asarray([submitted[k] for k in keys], dtype=np.float64)
        aucs = class_aucs(targets, values)
        if any(v is None for v in aucs) or not np.isclose(np.mean(aucs), report["macro_auc"], atol=1e-9, rtol=0):
            raise ValueError("Saved probabilities differ from their validation report")
        candidates[name] = values
        provenance.append({"name": name, "validation_dir": str(root), "tta": report.get("tta"),
            "image_size": report.get("image_size"), "image_sizes": report.get("image_sizes"),
            "member_count": report.get("member_count", 1), "report_sha256": sha256_file(root / "validation_report.json"),
            "predictions_sha256": prediction_hash,
            "probability_report_binding": "sha256" if report.get("predictions_sha256") else "legacy_macro_auc_only"})
    combinations = {} if combinations_json is None else read_json(combinations_json)
    if not isinstance(combinations, dict):
        raise ValueError("Combination JSON maps a name to fixed M or Mx10 weights")
    if len(roots) > 1:
        combinations.setdefault("equal_mean", [1.] * len(roots))
    fixed = {}
    stack = np.stack(list(candidates.values()))
    for name, raw in combinations.items():
        if name in candidates:
            raise ValueError("Combination names must not overwrite an individual candidate")
        weights = normalized_weights(raw, len(roots))
        candidates[name] = (stack * weights[:, None, :]).sum(0)
        fixed[name] = weights.tolist()
    reference = class_aucs(targets, candidates[names[0]])
    scores = {}
    for name, values in candidates.items():
        aucs = class_aucs(targets, values)
        slices = {}
        for label, predicate in [("single_image_study", lambda n: n == 1), ("multiple_image_study", lambda n: n > 1)]:
            indices = np.asarray([i for i, k in enumerate(keys) if predicate(views[k])], dtype=int)
            sliced = class_aucs(targets[indices], values[indices]) if len(indices) else [None] * 10
            slices[label] = {"num_studies": len(indices), "per_class_auc": sliced,
                             "macro_auc": None if any(v is None for v in sliced) else float(np.mean(sliced))}
        scores[name] = {"macro_auc": float(np.mean(aucs)), "per_class_auc": aucs,
                        "per_class_delta_from_reference": [a - b for a, b in zip(aucs, reference)],
                        "macro_auc_delta_from_reference": float(np.mean(aucs) - np.mean(reference)), "slices": slices}
    result = {"manifest_sha256": manifest_hash, "num_studies": len(keys), "num_patients": len(np.unique(subjects)),
        "reference": names[0], "candidates": scores, "source_validations": provenance,
        "fixed_probability_weights": fixed, "weights_fitted": False, "thresholds_fitted": False,
        "deployment_selected": False, "T4_latency_certified": False,
        "bootstrap": paired_intervals(targets, candidates, subjects, samples=bootstrap, seed=seed),
        "note": "Validation comparisons are exploratory; no early-disease or lesion-size annotation is inferred."}
    output.mkdir(parents=True)
    write_json(output / "comparison_report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "data/clean_v1_reviewed/valid_manifest.csv")
    parser.add_argument("--combinations-json", type=Path)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    result = compare(**vars(parser.parse_args(argv)))
    print(json.dumps({name: score["macro_auc"] for name, score in result["candidates"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
