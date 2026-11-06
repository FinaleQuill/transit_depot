"""Study-matched CPU diagnostics against a frozen, patient-excluded CNN.

The historical CNN reference used the old image manifest. Restoring original
images may change a Study's image count or view list; it must never change its
identity or labels. Slice membership uses the current original-data metadata,
while the report preserves the historical metadata difference. This is not an
identical-image or single-factor architecture comparison.

No images, model weights, test predictions, thresholds or fitted blend weights
are used here. The caller must verify that both inputs are registered internal
development outputs and that CNN training patients exclude the current dev.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from breakthrough.generalization_monitor import compare, read_prediction, write
from reconstruction.evaluate import _validated

DIFFICULT_SLICES = (
    "ap_projection_only", "single_image_study", "at_least_5_studies_patient",
)
VARIANTS = ("single_candidate", "fixed_50_50_blend")
MINIMUM_VALID_REPLICATES = 100
MINIMUM_VALID_FRACTION = .90


def _checked(value):
    required = {"probabilities", "targets", "study_keys", "patient_ids", "views_json"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("Missing prediction identity fields")
    p, y, keys, patients, views = _validated(
        value["probabilities"], value["targets"], value["study_keys"],
        value["patient_ids"], value["views_json"])
    if any(not key.startswith(patient + "/") for key, patient in zip(keys, patients)):
        raise ValueError("Study key does not belong to the supplied patient")
    order = np.argsort(keys)
    return dict(probabilities=p[order], targets=y[order], study_keys=keys[order],
                patient_ids=patients[order],
                views_json=np.asarray([json.dumps(views[i]) for i in order]))


def align_predictions(candidate, reference):
    """Return checked Study alignment and an explicit historical-input audit.

    Inputs are not modified. The returned reference retains its original view
    metadata; only ``compare_predictions`` builds a separate projection of that
    metadata for assigning both predictions to the same current-data slices.
    """
    candidate, reference = _checked(candidate), _checked(reference)
    for name in ("study_keys", "patient_ids", "targets"):
        if not np.array_equal(candidate[name], reference[name]):
            raise ValueError("Reference and candidate differ in " + name)
    changes, current_images, reference_images = [], 0, 0
    for key, patient, new, old in zip(candidate["study_keys"], candidate["patient_ids"],
                                    candidate["views_json"], reference["views_json"]):
        new_views, old_views = json.loads(new), json.loads(old)
        current_images += len(new_views)
        reference_images += len(old_views)
        if new_views != old_views:
            changes.append(dict(study_key=str(key), patient_id=str(patient),
                candidate_views=new_views, reference_views=old_views,
                candidate_image_count=len(new_views), reference_image_count=len(old_views),
                view_multiset_changed=Counter(new_views) != Counter(old_views)))
    coverage = dict(studies=len(candidate["study_keys"]),
        patients=len(set(candidate["patient_ids"])),
        candidate_images=current_images, reference_images=reference_images,
        image_count_difference=current_images-reference_images,
        studies_with_view_metadata_changes=len(changes), changed_studies=changes,
        identical_study_patient_label_coverage=True,
        identical_image_coverage_claimed=False,
        slice_membership_source="candidate original uncleaned development metadata",
        reference_predictions_recomputed=False,
        reference_input_policy="historical CNN predictions used the old cleaned image manifest",
        limitation="Same Studies and labels; historical and current image coverage can differ. "
                   "View lists alone cannot certify identical image identities.")
    return candidate, reference, coverage


def compare_predictions(candidate, reference, *, repeats=300):
    """Evaluate one candidate and one fixed all-class 50:50 probability blend."""
    if type(repeats) is not int or repeats < 0:
        raise ValueError("repeats must be a nonnegative integer")
    candidate, reference, coverage = align_predictions(candidate, reference)
    # This copy changes only diagnostic slice membership, never CNN outputs.
    sliced_reference = dict(reference, views_json=candidate["views_json"].copy())
    blend = dict(candidate, probabilities=(candidate["probabilities"] +
                                          reference["probabilities"]) * .5)
    reports = {}
    for name, prediction in ((VARIANTS[0], candidate), (VARIANTS[1], blend)):
        report = compare(prediction, sliced_reference, repeats=repeats)
        report["comparison_variant"] = name
        report["blend_weights"] = {"candidate": 1., "cnn": 0.} if name == VARIANTS[0] else {
            "candidate": .5, "cnn": .5}
        report["blend_weights_fitted"] = False
        report["reference_slice_metadata_replaced_for_diagnostics_only"] = True
        report["identical_image_coverage_claimed"] = False
        report["limitations"].append(coverage["limitation"])
        reports[name] = report
    return dict(schema_version=1, scope="registered_internal_development",
        coverage_comparison=coverage, **reports,
        selection_metric="Study macro AUC; these diagnostics do not select checkpoints",
        patient_training_isolation_verified_by="caller must bind frozen CNN training provenance",
        test_accessed=False, official_validation_accessed=False,
        model_weights_loaded=False, automatic_submission=False,
        automatic_training_stop=False, fitted_parameters=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_files(candidate_path, reference_path, *, repeats=300,
                  expected_reference_sha256=None, output_path=None):
    """Read frozen NPZs, preserve their identities and optionally save JSON."""
    candidate_path, reference_path = Path(candidate_path), Path(reference_path)
    initial = (_sha(candidate_path), _sha(reference_path))
    if expected_reference_sha256 is not None and initial[1] != expected_reference_sha256:
        raise ValueError("Frozen CNN reference checksum differs")
    report = compare_predictions(read_prediction(candidate_path),
                                 read_prediction(reference_path), repeats=repeats)
    if (_sha(candidate_path), _sha(reference_path)) != initial:
        raise ValueError("Prediction inputs changed during comparison")
    report["inputs"] = dict(candidate_path=str(candidate_path.resolve()),
        candidate_sha256=initial[0], reference_path=str(reference_path.resolve()),
        reference_sha256=initial[1], reference_sha256_checked=expected_reference_sha256 is not None)
    if output_path is not None:
        write(output_path, report)
    return report


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_score(score):
    if not isinstance(score, dict) or not _finite_number(score.get("macro_auc")):
        return False
    per_class = score.get("per_class_auc")
    reference = score.get("reference", {})
    reference_classes = reference.get("per_class_auc")
    return (isinstance(per_class, list) and len(per_class) == 10
        and all(_finite_number(v) and 0 <= v <= 1 for v in per_class)
        and isinstance(reference_classes, list) and len(reference_classes) == 10
        and all(_finite_number(v) and 0 <= v <= 1 for v in reference_classes)
        and _finite_number(reference.get("macro_auc"))
        and _finite_number(score.get("macro_auc_delta")))


def _paired_interval(score):
    if not isinstance(score, dict):
        return None
    bootstrap = score.get("patient_bootstrap", {}).get("paired_delta_macro_auc")
    if bootstrap is None:
        bootstrap = score.get("paired_patient_bootstrap")
    if not isinstance(bootstrap, dict):
        return None
    interval = bootstrap.get("ci95")
    valid, attempted = bootstrap.get("valid_replicates"), bootstrap.get("attempted_replicates")
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(_finite_number(v) for v in interval)
            or not -1 <= interval[0] <= interval[1] <= 1
            or type(valid) is not int or type(attempted) is not int
            or valid < MINIMUM_VALID_REPLICATES or attempted < valid
            or valid / attempted < MINIMUM_VALID_FRACTION):
        return None
    return interval


def stage_decision(comparison, *, training_complete=False):
    """Conservative eligibility for another fold, never permission to deploy.

    Call only on the checkpoint selected by the registered Macro-AUC protocol;
    scanning all epochs for a passing gate would introduce extra selection.
    Confidence intervals remain exploratory after repeated dev selection.
    """
    if type(training_complete) is not bool:
        raise TypeError("training_complete must be boolean")
    eligible, details = [], {}
    for name in VARIANTS:
        report = comparison.get(name, {})
        overall = report.get("overall", {})
        reasons = []
        if not training_complete:
            reasons.append("training has not completed; no stage advancement")
        interval = _paired_interval(overall)
        if not _valid_score(overall) or interval is None:
            reasons.append("overall ten-class result or paired patient CI is missing/insufficient")
        elif interval[0] <= 0:
            reasons.append("overall paired patient CI does not establish a positive difference")
        for key in DIFFICULT_SLICES:
            score = report.get("slices", {}).get(key, {})
            difficult_ci = _paired_interval(score)
            if not _valid_score(score) or difficult_ci is None:
                reasons.append(key + ": ten-class result or paired patient CI is missing/insufficient")
            elif difficult_ci[1] < 0:
                reasons.append(key + ": clear negative difference in paired patient CI")
        passed = not reasons
        details[name] = dict(eligible=passed, reasons=reasons)
        if passed:
            eligible.append(name)
    return dict(state="eligible_for_second_fold" if eligible else "needs_review",
        eligible_variants=eligible, variants=details, training_complete=training_complete,
        minimum_valid_bootstrap_replicates=MINIMUM_VALID_REPLICATES,
        minimum_valid_bootstrap_fraction=MINIMUM_VALID_FRACTION,
        difficult_slices=list(DIFFICULT_SLICES), automatic_submission=False,
        automatic_training_stop=False, hidden_test_improvement_established=False,
        meaning="Eligibility for a separately held-out patient-fold confirmation only; "
                "absence of significant slice harm is not proof of non-inferiority.")
