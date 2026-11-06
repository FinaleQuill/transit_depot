"""Integrity and conservative advancement checks using synthetic Studies."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from raw_rebuild.comparison import (align_predictions, compare_files,
    compare_predictions, DIFFICULT_SLICES, stage_decision, VARIANTS)


def fixture():
    rng = np.random.default_rng(732)
    patients = np.asarray([f"p{i // 6}" for i in range(72)])
    keys = np.asarray([f"{patient}/s{i}" for i, patient in enumerate(patients)])
    y = np.asarray([[(i + c) % 2 for c in range(10)] for i in range(72)])
    views = np.asarray([json.dumps(["AP"] if i % 3 == 0 else
                                   ["PA"] if i % 3 == 1 else ["PA", "LATERAL"])
                        for i in range(72)])
    candidate = dict(probabilities=.2 + .6*y, targets=y, study_keys=keys,
                     patient_ids=patients, views_json=views)
    reference = dict(candidate, probabilities=.35*y + .65*rng.random(y.shape))
    return candidate, reference


def passing_report():
    score = dict(macro_auc=.81, per_class_auc=[.81]*10, macro_auc_delta=.01,
        reference=dict(macro_auc=.80, per_class_auc=[.80]*10),
        patient_bootstrap=dict(paired_delta_macro_auc=dict(
            ci95=[.002, .018], valid_replicates=300, attempted_replicates=300)))
    report = dict(overall=deepcopy(score), slices={key:deepcopy(score) for key in DIFFICULT_SLICES})
    return {name:deepcopy(report) for name in VARIANTS}


class ComparisonTests(unittest.TestCase):
    def test_permutation_aligns_but_study_patient_and_label_changes_fail(self):
        candidate, reference = fixture()
        permuted = {key:values[::-1].copy() for key, values in reference.items()}
        left, right, coverage = align_predictions(candidate, permuted)
        np.testing.assert_array_equal(left["study_keys"], right["study_keys"])
        self.assertEqual(coverage["studies"], 72)
        for fault in ("study", "patient", "label", "duplicate"):
            damaged = deepcopy(reference)
            if fault == "study":
                damaged["study_keys"][0] = "p0/sx"
            elif fault == "patient":
                damaged["patient_ids"][0] = "p9"
            elif fault == "label":
                damaged["targets"][0,0] = 1-damaged["targets"][0,0]
            else:
                damaged["study_keys"][1] = damaged["study_keys"][0]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                align_predictions(candidate, damaged)

    def test_restored_views_are_audited_and_only_current_metadata_defines_slices(self):
        candidate, reference = fixture()
        original_reference_views = reference["views_json"].copy()
        # New array avoids NumPy fixed-width string truncation in this fixture.
        views = candidate["views_json"].tolist()
        views[0] = json.dumps(["AP", "AP"])
        candidate["views_json"] = np.asarray(views)
        result = compare_predictions(candidate, reference, repeats=0)
        coverage = result["coverage_comparison"]
        self.assertEqual(coverage["image_count_difference"], 1)
        self.assertEqual(coverage["studies_with_view_metadata_changes"], 1)
        self.assertFalse(coverage["identical_image_coverage_claimed"])
        self.assertEqual(coverage["changed_studies"][0]["reference_views"], ["AP"])
        self.assertEqual(result["single_candidate"]["slices"]["single_image_study"]["num_studies"], 47)
        self.assertEqual(result["fixed_50_50_blend"]["blend_weights"], {"candidate":.5,"cnn":.5})
        np.testing.assert_array_equal(reference["views_json"], original_reference_views)
        self.assertEqual(stage_decision(result, training_complete=True)["state"], "needs_review")

    def test_identical_reference_has_exact_zero_pair_delta_even_after_reordering(self):
        candidate, _ = fixture()
        result = compare_predictions(candidate, {k:v[::-1] for k,v in candidate.items()}, repeats=5)
        for name in VARIANTS:
            self.assertEqual(result[name]["overall"]["macro_auc_delta"], 0.)
            self.assertEqual(result[name]["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"], [0.,0.])

    def test_saved_files_bind_frozen_checksum_and_reject_changed_identity(self):
        candidate, reference = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/"candidate.npz", **candidate)
            np.savez(root/"reference.npz", **reference)
            with self.assertRaisesRegex(ValueError, "checksum"):
                compare_files(root/"candidate.npz", root/"reference.npz", repeats=0,
                              expected_reference_sha256="0"*64)
            result = compare_files(root/"candidate.npz", root/"reference.npz", repeats=0,
                                   output_path=root/"report.json")
            self.assertEqual(json.loads((root/"report.json").read_text())["inputs"], result["inputs"])

    def test_incomplete_training_cannot_advance_even_with_positive_intervals(self):
        report = passing_report()
        self.assertEqual(stage_decision(report)["state"], "needs_review")
        decision = stage_decision(report, training_complete=True)
        self.assertEqual(decision["state"], "eligible_for_second_fold")
        self.assertEqual(decision["eligible_variants"], list(VARIANTS))
        self.assertFalse(decision["automatic_submission"])

    def test_missing_low_support_nonfinite_or_negative_ci_does_not_advance(self):
        for fault in ("missing_ci", "few_replicates", "low_fraction", "nan", "none_class", "harm", "not_positive"):
            report = passing_report()
            for name in VARIANTS:
                score = report[name]["slices"][DIFFICULT_SLICES[0]]
                interval = score["patient_bootstrap"]["paired_delta_macro_auc"]
                if fault == "missing_ci":
                    del score["patient_bootstrap"]
                elif fault == "few_replicates":
                    interval["valid_replicates"] = 1
                elif fault == "low_fraction":
                    interval["valid_replicates"] = 200
                elif fault == "nan":
                    interval["ci95"][0] = float("nan")
                elif fault == "none_class":
                    score["per_class_auc"][0] = None
                elif fault == "harm":
                    interval["ci95"] = [-.02, -.001]
                else:
                    report[name]["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"] = [0.,.02]
            with self.subTest(fault=fault):
                self.assertEqual(stage_decision(report, training_complete=True)["state"], "needs_review")

    def test_blend_can_qualify_without_single_model_and_frequency_ci_is_supported(self):
        report = passing_report()
        report[VARIANTS[0]]["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"] = [-.03,.01]
        for name in VARIANTS:
            frequency = report[name]["slices"]["at_least_5_studies_patient"]
            frequency["paired_patient_bootstrap"] = frequency.pop("patient_bootstrap")["paired_delta_macro_auc"]
        decision = stage_decision(report, training_complete=True)
        self.assertEqual(decision["eligible_variants"], ["fixed_50_50_blend"])


if __name__ == "__main__":
    unittest.main()
