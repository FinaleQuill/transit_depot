"""CPU-only evaluation of registered final-EMA fine-tuning candidates.

Reads the unchanged official valid manifest and SHA-bound saved predictions.
No model, image decoder, teacher, or test split is loaded. Fixed image-level
probability rules are replayed before the original-order FP64 Study mean.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from audit_student_submission import IMAGE_HEADER, SUBMISSION_HEADER, read_probabilities
from compare_student_candidates import normalized_weights, paired_intervals
from deploy_kd import sha256_file
from student_experiment_evaluation import score_probabilities, validation_labels, write_new_json


OFFICIAL_VALID_SHA256 = "c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112"
BASELINE_NAME = "A_B_class1_B_baseline"
PAIR_WEIGHTS = [[0.0] + [0.5] * 9, [1.0] + [0.5] * 9]
CATALOG = {
    "X1_single": (["X1"], [[1.0] * 10]),
    "X2_single": (["X2"], [[1.0] * 10]),
    "X1_B_class1_B": (["X1", "B"], PAIR_WEIGHTS),
    "A_X2_class1_X2": (["A", "X2"], PAIR_WEIGHTS),
    "X1_X2_class1_X2": (["X1", "X2"], PAIR_WEIGHTS),
    "B_X2_equal": (["B", "X2"], [[0.5] * 10, [0.5] * 10]),
}


def load_registered_plan(path, expected_sha256, manifest_sha256):
    path = Path(path).resolve(strict=True)
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError("Pre-registered plan SHA256 differs from the supplied frozen identity")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("test_used") is not False
            or plan.get("registered_before_new_model_evaluation") is not True
            or plan.get("manifest_sha256") != manifest_sha256):
        raise ValueError("Plan must bind original valid and declare pre-registration without test use")
    # Do not require/claim that the old class-1 rule was chosen without valid:
    # it was selected in an earlier official-valid experiment.
    rules = plan.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Plan must contain its finite registered rules")
    seen = set()
    for rule in rules:
        name = rule.get("name") if isinstance(rule, dict) else None
        if name not in CATALOG or name in seen:
            raise ValueError("Unregistered or duplicate fine-tuning comparison rule")
        seen.add(name)
        members, weights = CATALOG[name]
        if rule.get("members") != members:
            raise ValueError("Rule member order differs from the fixed catalog: " + name)
        actual = normalized_weights(rule.get("weights"), len(members))
        expected = normalized_weights(weights, len(members))
        if not np.array_equal(actual, expected):
            raise ValueError("Rule weights differ from the fixed catalog: " + name)
    return plan, digest


def aggregate_images(records, image_probabilities):
    """Independent replay in original manifest order; never sort image IDs."""
    sums, counts = {}, {}
    for row in records:
        key = (row["subject_id"], row["study_id"])
        sums.setdefault(key, np.zeros(10, dtype=np.float64))
        counts[key] = counts.get(key, 0) + 1
        sums[key] += image_probabilities[row["image_id"]]
    return {key: value / counts[key] for key, value in sums.items()}


def load_validation(directory, records, manifest_sha256, labels, views):
    root = Path(directory).resolve(strict=True)
    files = [root / name for name in (
        "validation_report.json", "valid_predictions.csv", "valid.image_predictions.csv")]
    frozen = {str(path): sha256_file(path) for path in files}
    report = json.loads(files[0].read_text(encoding="utf-8"))
    if (report.get("complete") is not True
            or report.get("manifest_sha256") != manifest_sha256
            or report.get("predictions_sha256") != frozen[str(files[1])]
            or report.get("image_predictions_sha256") != frozen[str(files[2])]
            or report.get("tta") != "hflip2" or report.get("image_size") != 768
            or report.get("precision") != "float16" or report.get("member_count") != 1
            or report.get("test_images_used") is not False):
        raise ValueError("Require SHA-bound complete single-student FP16/768/hflip2 valid predictions")
    identities = {r["image_id"]: (r["subject_id"], r["study_id"]) for r in records}
    if len(identities) != len(records):
        raise ValueError("Manifest repeats an image identity")
    images, image_rows = read_probabilities(files[2], IMAGE_HEADER, identities, image_level=True)
    submitted, study_rows = read_probabilities(files[1], SUBMISSION_HEADER, set(labels), image_level=False)
    replay = aggregate_images(records, images)
    if any(replay[key].tobytes() != submitted[key].tobytes() for key in labels):
        raise ValueError("Saved Study predictions differ from original-order FP64 image means")
    scores = score_probabilities(labels, replay, views)
    if (report.get("num_images") != len(records) or report.get("num_studies") != len(labels)
            or image_rows != len(records) * 10 or study_rows != len(labels) * 10
            or not np.isclose(report.get("macro_auc"), scores["macro_auc"], atol=1e-12, rtol=0)
            or not np.allclose(report.get("per_class_auc"), scores["per_class_auc"], atol=1e-12, rtol=0)):
        raise ValueError("Saved valid counts/AUC differ from independent replay")
    if any(sha256_file(path) != digest for path, digest in frozen.items()):
        raise RuntimeError("Saved validation inputs changed during verification")
    return images, replay, scores, {"validation_dir": str(root), "files_sha256": frozen,
        "deployment_config_sha256": report.get("deployment_config_sha256"),
        "images": len(records), "studies": len(labels), "image_to_study_bitwise_equal": True}


def fixed_image_rule(records, images, names, weights):
    weights = normalized_weights(weights, len(names))
    result = {}
    for row in records:
        image_id = row["image_id"]
        result[image_id] = sum((weights[i] * images[name][image_id]
                               for i, name in enumerate(names)), np.zeros(10, dtype=np.float64))
    return aggregate_images(records, result)


def evaluate_finetune_candidates(*, a, b, x1, x2, manifest, plan, plan_sha256, out,
                                 bootstrap=1000, seed=20261101):
    from dataset import read_manifest
    if x1 is None and x2 is None:
        raise ValueError("Provide at least one new final-EMA validation directory")
    manifest = Path(manifest).resolve(strict=True)
    manifest_sha = sha256_file(manifest)
    if manifest_sha != OFFICIAL_VALID_SHA256:
        raise ValueError("Only the unchanged complete official valid manifest is accepted")
    registered, digest = load_registered_plan(plan, plan_sha256, manifest_sha)
    records = read_manifest(manifest, split="valid")
    labels, views = validation_labels(records)
    if len(records) != 1422 or len(labels) != 859 or len({k[0] for k in labels}) != 228:
        raise ValueError("Require all 1422 images, 859 Studies and 228 original valid patients")
    roots = {"A": Path(a), "B": Path(b)}
    roots.update({name: Path(path) for name, path in (("X1", x1), ("X2", x2)) if path is not None})
    output = Path(out).resolve()
    if output.exists():
        raise FileExistsError("Choose a new fine-tuning comparison output directory")
    if any(output == root.resolve() or output.is_relative_to(root.resolve()) for root in roots.values()):
        raise ValueError("Comparison output cannot be inside saved validation inputs")
    images, predictions, scores, provenance = {}, {}, {}, {}
    for name, root in roots.items():
        im, study, score, identity = load_validation(root, records, manifest_sha, labels, views)
        images[name], predictions[name], scores[name], provenance[name] = im, study, score, identity
    baseline = fixed_image_rule(records, images, ["A", "B"], PAIR_WEIGHTS)
    predictions[BASELINE_NAME] = baseline
    scores[BASELINE_NAME] = score_probabilities(labels, baseline, views)
    executed, pending = [], []
    for rule in registered["rules"]:
        name, names = rule["name"], rule["members"]
        if not set(names) <= set(images):
            pending.append(name)
            continue
        predictions[name] = fixed_image_rule(records, images, names, rule["weights"])
        scores[name] = score_probabilities(labels, predictions[name], views)
        executed.append(name)
    if not executed:
        raise ValueError("No registered rule has all required saved members")
    keys = sorted(labels)
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    subjects = np.asarray([key[0] for key in keys])
    arrays = {name: np.asarray([values[key] for key in keys], dtype=np.float64)
              for name, values in predictions.items()}
    intervals = {}
    for reference in ("B", BASELINE_NAME):
        reference_scores = scores[reference]
        for name in scores:
            scores[name].setdefault("delta_from", {})[reference] = {
                "macro_auc": scores[name]["macro_auc"] - reference_scores["macro_auc"],
                "per_class_auc": [a-b for a, b in zip(scores[name]["per_class_auc"], reference_scores["per_class_auc"])]}
        intervals[reference] = paired_intervals(targets, arrays, subjects, samples=bootstrap,
                                                seed=seed, reference=reference)
    if sha256_file(manifest) != manifest_sha or sha256_file(plan) != digest:
        raise RuntimeError("Manifest or registered plan changed during comparison")
    for identity in provenance.values():
        if any(sha256_file(path) != value for path, value in identity["files_sha256"].items()):
            raise RuntimeError("A validation input changed during comparison")
    report = {"complete": True, "time_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "manifest_sha256": manifest_sha, "registered_plan_sha256": digest,
        "registration_claim_source": "Supplied frozen plan; this CPU tool does not independently prove registration chronology.",
        "num_images": len(records), "num_studies": len(labels), "num_patients": len(set(subjects)),
        "candidates": scores, "registered_rules_evaluated": executed,
        "registered_rules_pending_missing_member": pending, "source_validations": provenance,
        "paired_patient_bootstrap": intervals, "bootstrap_samples": bootstrap, "bootstrap_seed": seed,
        "fixed_rules": [rule for rule in registered["rules"] if rule["name"] in executed],
        "aggregation": "fixed image-probability fusion, then FP64 original-manifest-order Study arithmetic mean",
        "weights_fitted": False, "thresholds_fitted": False, "test_used": False,
        "models_loaded": False, "deployment_selected": False, "T4_latency_certified": False,
        "evaluation_source_sha256": sha256_file(__file__),
        "limitations": ["Official valid was reused in earlier checkpoint and class-1 rule selection; intervals do not correct selection bias.",
            "The caller must separately bind X1/X2 exports to the registered final-EMA checkpoints; this comparison audits saved probability inputs, not training completion.",
            "Cache fusion requires actual combined predictor reproduction and latency measurement before deployment.",
            "These are local valid AUCs, not platform test scores; no independent generalization claim is made."]}
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "registered_plan.json", registered)
    write_new_json(output / "comparison_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a", "b", "manifest", "plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("x1", "x2"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261101)
    report = evaluate_finetune_candidates(**vars(parser.parse_args(argv)))
    print(json.dumps({name: score["macro_auc"] for name, score in report["candidates"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
