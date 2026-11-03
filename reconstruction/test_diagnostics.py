"""Independent semantic checks for bound dev and exact patient deletion AUC."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reconstruction.diagnostics import (align_reference, diagnose, exact_patient_loo,
    load_bound_predictions, sha, _verify_selected_completion)
from reconstruction.evaluate import save_evaluation


def pairwise_auc(labels, scores, weights=None):
    labels, scores = np.asarray(labels), np.asarray(scores)
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights)
    positive, negative = labels == 1, labels == 0
    denominator = weights[positive].sum() * weights[negative].sum()
    if denominator == 0:
        return None
    differences = scores[positive, None] - scores[negative][None, :]
    pairs = (differences > 0) + 0.5 * (differences == 0)
    pair_weights = weights[positive, None] * weights[negative][None, :]
    return float((pairs * pair_weights).sum() / denominator)


def fixture():
    y = np.tile(np.array([0, 1, 0, 1, 0, 1, 1])[:, None], (1, 10))
    p = np.tile(np.array([.1, .4, .4, .9, .8, .5, .4])[:, None], (1, 10))
    return {"probabilities": p, "targets": y,
            "study_keys": np.asarray(["p1/s1", "p1/s2", "p2/s3", "p2/s4", "p3/s5", "p3/s6", "p4/s7"]),
            "patient_ids": np.asarray(["p1", "p1", "p2", "p2", "p3", "p3", "p4"]),
            "views": [["AP"], ["AP", "AP"], ["PA", "LATERAL"], ["PA", "LATERAL"],
                      ["PA", "LL"], ["PA", "LL"], ["PA", "LATERAL", "MISSING"]],
            "image_counts": np.array([1, 2, 2, 2, 2, 2, 3])}


class ExactDeletionTests(unittest.TestCase):
    def test_ties_mixed_patient_labels_and_all_deletions_match_independent_pairs(self):
        rng = np.random.default_rng(851)
        p = rng.integers(0, 6, size=(67, 10)) / 5
        y = rng.integers(0, 2, size=(67, 10))
        patients = np.array([f"p{i//4}" for i in range(67)])
        order = rng.permutation(len(patients))
        p, y, patients = p[order], y[order], patients[order]
        result = exact_patient_loo(p, y, patients)
        for c in range(10):
            self.assertAlmostEqual(result["full_per_class_auc"][c], pairwise_auc(y[:, c], p[:, c]), places=14)
        for i, patient in enumerate(result["patient_ids"]):
            keep = patients != patient
            for c in range(10):
                self.assertAlmostEqual(result["loo_per_class_auc"][i, c],
                    pairwise_auc(y[keep, c], p[keep, c]), places=14)

    def test_deleting_only_positive_patient_produces_undefined_auc(self):
        p = np.tile(np.array([.2, .8, .3])[:, None], (1, 10))
        y = np.tile(np.array([0, 1, 0])[:, None], (1, 10))
        result = exact_patient_loo(p, y, ["negative", "only_positive", "negative"])
        self.assertTrue(np.isnan(result["loo_per_class_auc"]).all())

    def test_paired_alignment_and_loo_delta_change(self):
        candidate = fixture()
        reference = deepcopy(candidate)
        reference["probabilities"] = np.flip(candidate["probabilities"], axis=0).copy()
        aligned = reference["probabilities"].copy()
        order = np.array([4, 2, 6, 0, 1, 3, 5])
        for field in ["probabilities", "targets", "study_keys", "patient_ids", "image_counts"]:
            reference[field] = reference[field][order]
        reference["views"] = [reference["views"][i] for i in order]
        np.testing.assert_array_equal(align_reference(candidate, reference), aligned)
        report = diagnose(candidate, reference=reference)
        full = pairwise_auc(candidate["targets"][:, 0], candidate["probabilities"][:, 0]) - pairwise_auc(candidate["targets"][:, 0], aligned[:, 0])
        self.assertAlmostEqual(report["paired_macro_auc_delta"], full)
        for row in report["all_patient_influence"]:
            keep = candidate["patient_ids"] != row["patient_id"]
            delta = pairwise_auc(candidate["targets"][keep, 0], candidate["probabilities"][keep, 0]) - pairwise_auc(candidate["targets"][keep, 0], aligned[keep, 0])
            self.assertAlmostEqual(row["paired"]["macro_delta_change_on_removal"], delta-full)
        reference["targets"][0, 0] = 1-reference["targets"][0, 0]
        with self.assertRaisesRegex(ValueError, "targets"):
            align_reference(candidate, reference)

    def test_lateral_alias_support_patient_equal_and_candidate_rankings(self):
        values = fixture()
        report = diagnose(values, top_k=2)
        slices = report["slices"]
        self.assertEqual(slices["pa_plus_lateral_strict"]["num_studies"], 2)
        self.assertEqual(slices["pa_plus_lateral_including_LL"]["num_studies"], 4)
        self.assertEqual(slices["ap_projection_only"]["num_studies"], 2)
        weights = np.array([.5, .5, .5, .5, .5, .5, 1.])
        self.assertAlmostEqual(slices["all"]["patient_equal_weight"]["macro_auc"],
            pairwise_auc(values["targets"][:, 0], values["probabilities"][:, 0], weights))
        first_class = report["classes_lowest_auc_first"][0]
        self.assertEqual([r["probability"] for r in first_class["high_probability_negative_candidates"]], [.8, .4])
        self.assertEqual([r["probability"] for r in first_class["low_probability_positive_candidates"]], [.4, .4])
        json.dumps(report, allow_nan=False)


class BindingTests(unittest.TestCase):
    def setup_fixture(self, folder):
        values = fixture()
        root = Path(folder)
        dev = root / "dev.csv"
        rows = []
        for i, (key, patient, labels, views) in enumerate(zip(values["study_keys"], values["patient_ids"], values["targets"], values["views"])):
            for v, view in enumerate(views):
                rows.append({"split": "train", "training_eligible": "true", "subject_id": patient,
                    "study_id": key.split("/")[1], "study_key": key, "image_id": f"image{i}_{v}",
                    "targets": json.dumps(labels.tolist()), "view_position": view})
        with dev.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        plan = {"schema": "reconstruction_patient_content_holdout_v1", "holdout_fold": 4,
            "dev_fold": 0, "train_folds": [1,2,3], "isolation": {"train__dev": {"patients": 0}},
            "roles": {"dev": {"path": str(dev), "sha256": sha(dev), "studies": 7, "patients": 4, "images": len(rows)}}}
        plan_path = root / "split_plan.json"; plan_path.write_text(json.dumps(plan))
        prediction = root / "epoch_01"
        save_evaluation(prediction, values["probabilities"], values["targets"], values["study_keys"],
            values["patient_ids"], values["views"], bootstrap_repeats=0, metadata={"epoch": 1})
        options = {"split_plan": plan_path, "_expected_split_sha": sha(plan_path),
                   "_expected_dev_sha": sha(dev), "_expected_studies": 7}
        return values, prediction, options

    def test_bound_complete_dev_is_accepted_without_image_or_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            values, prediction, options = self.setup_fixture(folder)
            loaded = load_bound_predictions(prediction, **options)
            np.testing.assert_array_equal(loaded["targets"], values["targets"])
            report = diagnose(loaded)
            self.assertEqual(report["candidate_epoch"], 1)
            self.assertFalse(report["holdout_H_accessed"])

    def test_prediction_tampering_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            _, prediction, options = self.setup_fixture(folder)
            with (prediction/"predictions.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "Predictions SHA"):
                load_bound_predictions(prediction, **options)

    def test_altered_labels_rejected_even_if_evaluation_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            values, prediction, options = self.setup_fixture(folder)
            values["targets"][0, 0] = 1
            save_evaluation(prediction, values["probabilities"], values["targets"], values["study_keys"],
                values["patient_ids"], values["views"], bootstrap_repeats=0)
            with self.assertRaisesRegex(ValueError, "original Study metadata"):
                load_bound_predictions(prediction, **options)

    def test_changed_registered_manifest_and_forged_plan_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, prediction, options = self.setup_fixture(folder)
            path = Path(folder)/"dev.csv"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "Dev manifest path/hash/count"):
                load_bound_predictions(prediction, **options)
        with tempfile.TemporaryDirectory() as folder:
            _, prediction, options = self.setup_fixture(folder)
            options["split_plan"].write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable plan"):
                load_bound_predictions(prediction, **options)

    def test_incomplete_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            values, prediction, options = self.setup_fixture(folder)
            save_evaluation(prediction, values["probabilities"][:-1], values["targets"][:-1], values["study_keys"][:-1],
                values["patient_ids"][:-1], values["views"][:-1], bootstrap_repeats=0)
            with self.assertRaisesRegex(ValueError, "complete internal-development"):
                load_bound_predictions(prediction, **options)

    def test_completed_stage_requires_terminal_and_matching_best_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            _, prediction, options = self.setup_fixture(folder)
            loaded = load_bound_predictions(prediction, **options)
            auc = loaded["report"]["macro_auc"]
            run_dir = prediction.parent
            (run_dir/"compliance").mkdir()
            (run_dir/"compliance/status.json").write_text(json.dumps({"state": "early_stopped", "last_event": "training_ended"}))
            (run_dir/"status.json").write_text(json.dumps({"state": "running", "best_epoch": 1, "best_macro_auc": auc}))
            (run_dir/"history.jsonl").write_text(json.dumps({"epoch": 1, "macro_auc": auc})+"\n")
            with self.assertRaisesRegex(ValueError, "terminal/history"):
                diagnose(loaded, model_stage="selected_completed_run")
            (run_dir/"status.json").write_text(json.dumps({"state": "early_stopped", "best_epoch": 1, "best_macro_auc": auc}))
            report = diagnose(loaded, model_stage="selected_completed_run")
            self.assertEqual(report["model_stage"], "selected_completed_run")
            self.assertFalse(report["final_model_judgment"])
            self.assertEqual(report["completed_run_evidence"]["candidate"]["files"]["history"]["sha256"], sha(run_dir/"history.jsonl"))

    def terminal_fixture(self, folder):
        _, prediction, options = self.setup_fixture(folder)
        loaded = load_bound_predictions(prediction, **options)
        auc = loaded["report"]["macro_auc"]
        root = prediction.parent
        (root/"compliance").mkdir()
        (root/"status.json").write_text(json.dumps({"state": "early_stopped", "best_epoch": 1, "best_macro_auc": auc}))
        (root/"compliance/status.json").write_text(json.dumps({"state": "early_stopped", "last_event": "training_ended"}))
        (root/"history.jsonl").write_text(json.dumps({"epoch": 1, "macro_auc": auc})+"\n")
        return loaded, root, auc

    def test_best_epoch_out_of_range_and_wrong_matching_row_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            loaded, root, auc = self.terminal_fixture(folder)
            loaded["report"]["metadata"]["epoch"] = 7
            (root/"status.json").write_text(json.dumps({"state": "early_stopped", "best_epoch": 7, "best_macro_auc": auc}))
            with self.assertRaisesRegex(ValueError, "terminal/history"):
                _verify_selected_completion(loaded)
            loaded["report"]["metadata"]["epoch"] = 1
            (root/"status.json").write_text(json.dumps({"state": "early_stopped", "best_epoch": 1, "best_macro_auc": auc}))
            (root/"history.jsonl").write_text(json.dumps({"epoch": 1, "macro_auc": auc-.1})+"\n"+json.dumps({"epoch": 2, "macro_auc": auc})+"\n")
            with self.assertRaisesRegex(ValueError, "history row"):
                _verify_selected_completion(loaded)

    def test_reference_terminal_status_must_also_match(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            candidate, _, _ = self.terminal_fixture(first)
            reference, root, _ = self.terminal_fixture(second)
            (root/"compliance/status.json").write_text(json.dumps({"state": "running", "last_event": "epoch_completed"}))
            with self.assertRaisesRegex(ValueError, "terminal/history"):
                diagnose(candidate, reference=reference, model_stage="selected_completed_run")

    def test_concurrent_history_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            loaded, root, _ = self.terminal_fixture(folder)
            original_read = Path.read_bytes
            target = root/"history.jsonl"
            calls = 0
            def changing_read(path):
                nonlocal calls
                contents = original_read(path)
                if path == target:
                    calls += 1
                    if calls > 1:
                        return contents+b"\n"
                return contents
            with mock.patch.object(Path, "read_bytes", changing_read):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    _verify_selected_completion(loaded)


if __name__ == "__main__":
    unittest.main()
