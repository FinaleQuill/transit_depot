"""CPU tests for actual evaluation semantics and original-weight identity gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reconstruction.evaluate import evaluate_studies, json_native, save_evaluation
from reconstruction import provenance


def fixture():
    y = np.tile(np.array([0, 1, 0, 1])[:, None], (1, 10))
    p = np.tile(np.array([0.1, 0.4, 0.4, 0.9])[:, None], (1, 10))
    return p, y, ["s1", "s2", "s3", "s4"], ["p1", "p1", "p2", "p2"], [["AP"], ["AP", "AP"], ["PA", "LATERAL"], ["AP"]]


class EvaluationTests(unittest.TestCase):
    def test_exact_ties_and_projection_slices(self):
        result = evaluate_studies(*fixture(), bootstrap_repeats=0, scope="synthetic_test")
        self.assertEqual(result["macro_auc"], 0.875)
        self.assertAlmostEqual(result["macro_average_precision"], 5 / 6)
        self.assertEqual(result["slices"]["ap_projection_only"]["num_studies"], 3)
        self.assertEqual(result["slices"]["single_image_study"]["num_studies"], 2)
        self.assertEqual(result["slices"]["multiple_image_study"]["num_studies"], 2)

    def test_bootstrap_preserves_patient_clusters_and_is_paired(self):
        p, y, keys, patients, views = fixture()
        result = evaluate_studies(p, y, keys, patients, views, bootstrap_repeats=50, seed=4,
                                  reference_probabilities=p, scope="synthetic_test")
        repeat = evaluate_studies(p, y, keys, patients, views, bootstrap_repeats=50, seed=4,
                                  reference_probabilities=p, scope="synthetic_test")
        self.assertEqual(result, repeat)
        actual = result["slices"]["all"]["patient_bootstrap"]
        self.assertEqual(actual["paired_delta_macro_auc"]["ci95"], [0.0, 0.0])
        # Every patient contains one positive and one negative: all draws valid.
        self.assertEqual(actual["macro_auc"]["valid_replicates"], 50)
        self.assertEqual(actual["macro_auc"]["ci95"], [0.875, 1.0])

    def test_rank_metrics_match_pairwise_auc_and_sklearn_when_available(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=(83, 10))
        p = rng.integers(0, 8, size=(83, 10)) / 7
        result = evaluate_studies(p, y, [f"s{i}" for i in range(83)],
                                  [f"p{i // 3}" for i in range(83)], [["AP"]] * 83,
                                  bootstrap_repeats=0, scope="synthetic_test")
        for c in range(10):
            pairwise = p[y[:, c] == 1, c][:, None] - p[y[:, c] == 0, c][None, :]
            expected = np.mean((pairwise > 0) + 0.5 * (pairwise == 0))
            self.assertAlmostEqual(result["per_class_auc"][c], expected)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            return
        np.testing.assert_allclose(result["per_class_auc"], roc_auc_score(y, p, average=None))
        np.testing.assert_allclose(result["per_class_average_precision"], average_precision_score(y, p, average=None))

    def test_missing_class_is_null_and_json_contains_no_nan(self):
        p, y, keys, patients, views = fixture()
        y[:, 0] = 0
        result = evaluate_studies(p, y, keys, patients, views, bootstrap_repeats=8, scope="synthetic_test")
        self.assertIsNone(result["macro_auc"])
        self.assertIsNone(result["per_class_auc"][0])
        self.assertEqual(result["slices"]["all"]["num_auc_evaluable_classes"], 9)
        self.assertIsNone(result["slices"]["all"]["patient_bootstrap"]["macro_auc"]["ci95"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(json_native({"x": np.float64(np.nan)}), {"x": None})

    def test_empty_ap_slice_and_unknown_projection_are_not_silently_ap(self):
        p, y, keys, patients, _ = fixture()
        result = evaluate_studies(p, y, keys, patients, [["AP", "Missing"], ["PA"], [""], ["LATERAL"]],
                                  bootstrap_repeats=2, scope="synthetic_test")
        self.assertEqual(result["slices"]["ap_projection_only"]["num_studies"], 0)
        self.assertIsNone(result["slices"]["ap_projection_only"]["macro_auc"])

    def test_invalid_inputs_and_external_scope_rejected(self):
        p, y, keys, patients, views = fixture()
        for scope in ["H", "official_valid", "test"]:
            with self.assertRaises(ValueError):
                evaluate_studies(p, y, keys, patients, views, scope=scope)
        with self.assertRaises(ValueError):
            evaluate_studies(p, y, ["duplicate"] * 4, patients, views)
        for bad in [np.nan, -0.1, 1.1]:
            altered = p.copy()
            altered[0, 0] = bad
            with self.assertRaises(ValueError):
                evaluate_studies(altered, y, keys, patients, views)
        with self.assertRaises(ValueError):
            evaluate_studies(p, np.full_like(y, 2), keys, patients, views)

    def test_npz_roundtrip_without_pickle_and_report_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            report = save_evaluation(folder, *fixture(), bootstrap_repeats=5, scope="synthetic_test",
                                      metadata={"epoch": np.int64(3)})
            with np.load(Path(folder) / "predictions.npz", allow_pickle=False) as stored:
                self.assertEqual(stored["probabilities"].shape, (4, 10))
                views = [json.loads(v) for v in stored["views_json"]]
                restored = evaluate_studies(stored["probabilities"], stored["targets"], stored["study_keys"],
                                            stored["patient_ids"], views, bootstrap_repeats=5, scope="synthetic_test")
                self.assertEqual(restored["slices"], report["slices"])
            self.assertEqual(report["predictions_sha256"], hashlib.sha256((Path(folder) / "predictions.npz").read_bytes()).hexdigest())
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text()), report)


class ProvenanceTests(unittest.TestCase):
    def _source(self, weight_hash):
        materials = {k: {"text": "fixture source material", "url": "https://example.invalid/source",
                         "sha256": hashlib.sha256(b"fixture source material").hexdigest()}
                     for k in ["model_card", "apache_license", "upstream_license"]}
        return {"repo_id": provenance.EXPECTED_REPOSITORY, "backbone_name": provenance.EXPECTED_BACKBONE,
                "revision": provenance.EXPECTED_REVISION, "weights_sha256": weight_hash,
                "weights_url": provenance.EXPECTED_WEIGHTS_URL, "license": "apache-2.0",
                "pretrain_dataset": "ImageNet-22k", "finetune_dataset": "ImageNet-1k", "materials": materials}

    def test_exact_original_copy_accepted_but_modified_task_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            weights, source = Path(folder) / "original.bin", Path(folder) / "source.json"
            weights.write_bytes(b"synthetic original public weights")
            weight_hash = provenance.file_sha256(weights)
            source.write_text(json.dumps(self._source(weight_hash)))
            with mock.patch.object(provenance, "EXPECTED_WEIGHTS_SHA256", weight_hash), \
                 mock.patch.object(provenance, "EXPECTED_PROVENANCE_SHA256", provenance.file_sha256(source)):
                verified = provenance.verify_public_initialization(weights, source)
                self.assertTrue(verified["verified"])
                self.assertFalse(verified["legacy_task_checkpoint_used"])
                weights.write_bytes(b"fine tuned task checkpoint")
                with self.assertRaisesRegex(ValueError, "original public ImageNet checkpoint"):
                    provenance.verify_public_initialization(weights, source)

    def test_forged_source_record_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            weights, source = Path(folder) / "trained.pt", Path(folder) / "forged.json"
            weights.write_bytes(b"trained")
            source.write_text(json.dumps(self._source(provenance.EXPECTED_WEIGHTS_SHA256)))
            with self.assertRaisesRegex(ValueError, "trusted original source record"):
                provenance.verify_public_initialization(weights, source)


if __name__ == "__main__":
    unittest.main()
