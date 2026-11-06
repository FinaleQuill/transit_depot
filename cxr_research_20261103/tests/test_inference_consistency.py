"""CPU-only semantic and entrypoint checks; no model, CUDA work or downloads."""
from __future__ import annotations

from pathlib import Path
import csv
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cost
import predict
from inference_reproducibility import INFERENCE_SEED, configure_inference


class InferenceConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.key = ("subject_1", "study_1")
        self.other = ("subject_2", "study_2")
        self.values = np.asarray([.1, .2, .3, .35, .4, .45, .6, .7, .8, .9])

    def compare(self, actual, reference, **kwargs):
        return cost.compare_predictions({self.key: actual}, {self.key: reference}, **kwargs)

    def test_equal_outputs_and_ties_pass_and_report_zero_error(self):
        values = np.full(10, .2)
        report = self.compare(values, values)
        self.assertTrue(report["match"])
        self.assertEqual(report["mismatched_study_count"], 0)
        self.assertEqual(report["max_absolute_error"], 0)

    def test_small_semantically_identical_change_keeps_strict_numeric_check(self):
        actual = self.values + 5e-8
        self.assertTrue(self.compare(actual, self.values)["match"])
        report = self.compare(self.values + 2e-7, self.values)
        self.assertTrue(report["semantic_match"])
        self.assertFalse(report["match"])
        self.assertEqual(report["absolute_tolerance"], 1e-7)

    def test_within_sample_order_flip_under_allclose_tolerance_fails(self):
        reference, actual = self.values.copy(), self.values.copy()
        reference[:2] = [.2, .20000004]
        actual[:2] = [.20000006, .2]
        self.assertTrue(np.allclose(actual, reference, atol=1e-7, rtol=0))
        report = self.compare(actual, reference)
        self.assertFalse(report["match"])
        self.assertEqual(report["sample_class_order_mismatched_studies"], ["subject_1/study_1"])

    def test_breaking_within_sample_tie_under_tolerance_fails(self):
        reference, actual = self.values.copy(), self.values.copy()
        reference[:2] = [.2, .2]
        actual[:2] = [.20000004, .2]
        self.assertFalse(self.compare(actual, reference)["semantic_match"])

    def test_fixed_half_threshold_flip_under_tolerance_fails(self):
        reference, actual = self.values.copy(), self.values.copy()
        reference[4], actual[4] = .49999998, .50000002
        report = self.compare(actual, reference)
        self.assertFalse(report["match"])
        self.assertEqual(report["binary_mismatched_studies"], ["subject_1/study_1"])

    def test_config_gate_flip_is_checked_in_addition_to_default_half(self):
        reference, actual = self.values.copy(), self.values.copy()
        reference[7], actual[7] = .74999998, .75000002
        report = self.compare(actual, reference, gates=[.75] * 10)
        self.assertFalse(report["match"])
        self.assertEqual(report["binary_mismatched_studies"], [])
        self.assertEqual(report["gate_mismatched_studies"], ["subject_1/study_1"])

    def test_cross_study_order_flip_under_tolerance_fails(self):
        first, second = self.values.copy(), self.values.copy()
        first[2], second[2] = .3, .30000004
        actual_first, actual_second = first.copy(), second.copy()
        actual_first[2], actual_second[2] = .30000006, .3
        report = cost.compare_predictions(
            {self.key: actual_first, self.other: actual_second},
            {self.key: first, self.other: second})
        self.assertFalse(report["match"])
        self.assertEqual(report["sample_class_order_mismatched_studies"], [])
        self.assertEqual(report["cross_study_class_order_mismatched_studies"],
                         ["subject_1/study_1", "subject_2/study_2"])

    def test_cross_study_tie_creation_under_tolerance_fails(self):
        first, second = self.values.copy(), self.values.copy()
        first[2], second[2] = .3, .30000004
        actual_second = second.copy()
        actual_second[2] = .3
        report = cost.compare_predictions(
            {self.key: first, self.other: actual_second}, {self.key: first, self.other: second})
        self.assertFalse(report["match"])
        self.assertTrue(report["cross_study_class_order_mismatched_studies"])

    def test_missing_and_unexpected_identities_fail(self):
        reference = {self.key: self.values, self.other: self.values}
        report = cost.compare_predictions({self.key: self.values}, reference)
        self.assertFalse(report["identity_coverage_match"])
        self.assertEqual(report["missing_result_studies"], ["subject_2/study_2"])
        report = cost.compare_predictions({self.other: self.values}, {self.key: self.values})
        self.assertFalse(report["match"])
        self.assertEqual(report["missing_reference_studies"], ["subject_2/study_2"])

    def test_timing_subset_needs_explicit_identities_and_reference_permission(self):
        reference = {self.key: self.values, self.other: self.values}
        with self.assertRaisesRegex(ValueError, "explicit expected"):
            cost.compare_predictions({self.key: self.values}, reference, allow_reference_superset=True)
        report = cost.compare_predictions({self.key: self.values}, reference,
                                          expected_keys={self.key}, allow_reference_superset=True)
        self.assertTrue(report["match"])
        self.assertEqual(report["uncompared_reference_studies"], 1)
        report = cost.compare_predictions({self.key: self.values}, reference,
                                          expected_keys={self.key, self.other}, allow_reference_superset=True)
        self.assertFalse(report["match"])
        report = cost.compare_predictions(reference, reference, expected_keys={self.key},
                                          allow_reference_superset=True)
        self.assertFalse(report["match"])

    def test_legacy_subjectless_csv_matches_once_but_never_two_subjects(self):
        reference = {(None, "study_1"): self.values}
        self.assertTrue(cost.compare_predictions({self.key: self.values}, reference)["match"])
        report = cost.compare_predictions(
            {self.key: self.values, ("subject_2", "study_1"): self.values}, reference)
        self.assertFalse(report["match"])
        self.assertEqual(len(report["reused_reference_studies"]), 1)

    def test_invalid_probabilities_and_empty_results_never_pass(self):
        for value in (np.nan, np.inf, -1., 1.1):
            report = self.compare(np.full(10, value), self.values)
            self.assertFalse(report["match"])
            self.assertIsNone(report["max_absolute_error"])
        self.assertFalse(self.compare(np.ones(9), self.values)["match"])
        self.assertFalse(cost.compare_predictions({}, {})["match"])
        self.assertFalse(cost.compare_predictions({}, {self.key: self.values})["match"])

    def test_invalid_tolerance_or_gate_is_rejected(self):
        for tolerance in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.compare(self.values, self.values, atol=tolerance)
        for gates in ([.5] * 9, [.5] * 9 + [np.nan], [.5] * 9 + [1.1]):
            with self.assertRaises(ValueError):
                self.compare(self.values, self.values, gates=gates)

    def test_shared_seed_replays_python_numpy_without_model_or_cuda(self):
        calls = []
        fake = SimpleNamespace(
            manual_seed=lambda seed: calls.append(("torch", seed)),
            cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda_seed_only", seed))),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
                                     cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        report = configure_inference(torch_module=fake)
        first = random.random(), np.random.random()
        configure_inference(torch_module=fake)
        self.assertEqual(first, (random.random(), np.random.random()))
        self.assertEqual(report["seed"], INFERENCE_SEED)
        self.assertEqual(calls, [("torch", 43), ("cuda_seed_only", 43)] * 2)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)

    def test_image_sidecar_maps_manifest_id_and_checks_each_image(self):
        with tempfile.TemporaryDirectory(prefix="image-consistency-test-") as tmp:
            root = Path(tmp)
            path = root / "round.image_predictions.csv"
            record = {"subject_id": "subject_1", "study_id": "study_1",
                      "image_id": "subject_1/study_1/a.jpg", "path": root / "a.jpg"}
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Image_id", "Subject_id", "Study_id", "Predict_class", "Probability"])
                for c, p in enumerate(self.values, 1):
                    writer.writerow([record["image_id"], "subject_1", "study_1", c, p])
            reference = {("subject_1", "study_1", "test:00001"): self.values}
            ids = {("subject_1", "study_1", "a.jpg"): "test:00001"}
            report = cost.compare_image_output([record], path, reference, ids)
            self.assertTrue(report["match"])
            self.assertEqual(report["compared_images"], 1)
            self.assertEqual(report["mismatched_image_count"], 0)
            self.assertFalse(cost.compare_image_output([record], path, {}, ids)["match"])
            with path.open("a", newline="") as stream:
                writer = csv.writer(stream)
                for c, p in enumerate(self.values, 1):
                    writer.writerow(["unexpected", "subject_1", "study_1", c, p])
            self.assertFalse(cost.compare_image_output([record], path, reference, ids)["match"])

    def test_image_sidecar_rejects_missing_class_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix="image-sidecar-test-") as tmp:
            path = Path(tmp) / "image.csv"
            header = "Image_id,Subject_id,Study_id,Predict_class,Probability\n"
            rows = [f"a,subject_1,study_1,{c},0.2\n" for c in range(1, 11)]
            for body in (rows[:-1], rows + rows[:1], rows[:-1] + ["a,subject_1,study_1,10,nan\n"]):
                path.write_text(header + "".join(body))
                with self.assertRaises(ValueError):
                    cost.read_image_probabilities(path)

    def test_predict_and_cost_apply_shared_seed_before_model_loading(self):
        events = []
        config = {"schema_version": 1, "image_size": 512, "tta": "identity", "ensemble": False}

        class FakePredictor:
            inventory = {"size_pass": True}

            def __init__(self, model_dir):
                self.config = config
                events.append("load")

            def predict_image(self, path):
                return np.full(10, .2)

            def synchronize(self):
                pass

        def configure():
            events.append("seed")
            return {"seed": 43}

        with tempfile.TemporaryDirectory(prefix="inference-entry-test-") as tmp:
            root = Path(tmp)
            model, images = root / "model", root / "images"
            model.mkdir()
            images.mkdir()
            (model / "config.json").write_text("{}")
            record = {"image_id": "a", "study_key": "subject_1/study_1", "subject_id": "subject_1",
                      "study_id": "study_1", "path": images / "a.jpg"}
            with patch.object(predict, "prediction_records", return_value=[record]), \
                    patch.object(predict, "configure_inference", side_effect=configure), \
                    patch.object(predict, "Predictor", FakePredictor):
                report = predict.predict_submission(model_dir=model, image_root=images,
                                                    output=root / "predict/submission.csv")
            self.assertEqual(events, ["seed", "load"])
            self.assertEqual(report["reproducibility"]["seed"], 43)
            result = {self.key: np.full(10, .2)}
            for has_image_reference in (True, False):
                events.clear()
                if not has_image_reference:
                    (root / "predict/submission.image_predictions.csv").unlink()
                with patch.object(cost, "discover_images", side_effect=[[record] * 100, [record] * 5]), \
                        patch.object(cost, "match_timing_manifest", return_value={"complete_studies_proven": True}), \
                        patch.object(cost, "read_submission", return_value=result), \
                        patch.object(cost, "configure_inference", side_effect=configure), \
                        patch.object(cost, "Predictor", FakePredictor), \
                        patch.object(cost, "timing_environment", return_value={"official_environment_compatible": True, "gpu_name": "T4"}), \
                        patch.object(cost, "time_round", return_value=([1.] * 100, result)), \
                        patch.object(cost, "compare_image_output", return_value={"match": True}), \
                        patch.object(cost, "sha256_file", return_value="a" * 64), \
                        patch.object(cost, "print", create=True):
                    report = cost.run_cost(submission=root / "predict/submission.csv", pictures_100=images,
                                           pictures_5=images, model_dir=model,
                                           output_dir=root / f"cost_{has_image_reference}")
                self.assertEqual(events, ["seed", "load"])
                self.assertEqual(report["official_compliance_pass"], has_image_reference)
                self.assertEqual(report["reproducibility"]["seed"], 43)
                if not has_image_reference:
                    self.assertTrue(all(c["status"] == "pending_missing_final_image_sidecar"
                                        for c in report["comparison"]["image_rounds"]))


if __name__ == "__main__":
    unittest.main()
