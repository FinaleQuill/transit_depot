import hashlib
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import exploration_evaluate as evaluate
from training_compliance import ComplianceRecorder


def records_for_studies(count=24, *, images_per_study=2, ap_only=False):
    rows = []
    for study_index in range(count):
        subject = f"P{study_index // 2:03d}"
        study = f"S{study_index:03d}"
        targets = [(study_index + class_index) % 2 for class_index in range(10)]
        for image_index in range(images_per_study):
            rows.append({
                "split": "valid", "subject_id": subject, "study_id": study,
                "study_key": f"{subject}/{study}", "image_id": f"I{study_index:03d}_{image_index}",
                "targets": targets, "view_position": "AP" if ap_only or image_index == 0 else "PA",
            })
    return rows


def predictions_for(records):
    values = np.empty((len(records), 2, 10), dtype=np.float64)
    for index, row in enumerate(records):
        target = np.asarray(row["targets"], dtype=np.float64)
        values[index, 0] = .2 + .6 * target
        values[index, 1] = .3 + .4 * target
    return values


class ExplorationEvaluationTests(unittest.TestCase):
    def test_aggregate_uses_image_tta_then_study_means_and_keeps_patient_identity(self):
        records = records_for_studies(2, images_per_study=2, ap_only=True)
        views = predictions_for(records)
        bundle = evaluate.aggregate_predictions(records, views)

        self.assertEqual(bundle["study_keys"].tolist(), ["P000/S000", "P000/S001"])
        self.assertEqual(bundle["subject_ids"].tolist(), ["P000", "P000"])
        self.assertEqual(bundle["image_counts"].tolist(), [2, 2])
        self.assertTrue(bundle["ap_only"].all())
        np.testing.assert_allclose(bundle["identity"][0], views[:2, 0].mean(axis=0))
        np.testing.assert_allclose(bundle["hflip2"][0], views[:2].astype(np.float32).mean(axis=1).mean(axis=0))

    def test_aggregate_rejects_test_records_duplicate_images_and_mixed_study_labels(self):
        records = records_for_studies(1)
        views = predictions_for(records)
        bad = [dict(records[0], split="test"), *records[1:]]
        with self.assertRaisesRegex(ValueError, "train/valid"):
            evaluate.aggregate_predictions(bad, views)
        bad = [records[0], dict(records[1], image_id=records[0]["image_id"])]
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            evaluate.aggregate_predictions(bad, views)
        bad = [records[0], dict(records[1], targets=[1 - records[1]["targets"][0], *records[1]["targets"][1:]])]
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            evaluate.aggregate_predictions(bad, views)
        with self.assertRaisesRegex(ValueError, "binary targets"):
            evaluate.aggregate_predictions([dict(records[0], targets=[.5] * 10), records[1]], views)

    def test_tie_aware_auc_gives_half_credit(self):
        labels = np.tile(np.asarray([0, 0, 1, 1])[:, None], (1, 10))
        probabilities = np.tile(np.asarray([.1, .5, .5, .9])[:, None], (1, 10))
        self.assertEqual(evaluate.class_aucs(labels, probabilities), [.875] * 10)

    def test_study_scoring_reports_ap_only_as_a_separate_slice(self):
        records = records_for_studies(24, ap_only=False)
        bundle = evaluate.aggregate_predictions(records, predictions_for(records))
        result = evaluate.score_predictions(bundle)
        self.assertEqual(result["all"]["num_studies"], 24)
        self.assertEqual(result["ap_only"]["num_studies"], 0)
        self.assertIsNone(result["ap_only"]["macro_auc"])
        self.assertAlmostEqual(result["all"]["macro_auc"], 1.0)

    def test_patient_bootstrap_is_paired_reproducible_and_classwise(self):
        records = records_for_studies(24, images_per_study=1, ap_only=True)
        bundle = evaluate.aggregate_predictions(records, predictions_for(records))
        reference = bundle["hflip2"]
        candidate = np.where(bundle["targets"] == 1, .95, .05)
        first = evaluate.paired_bootstrap(bundle, reference, candidate, samples=80, seed=41)
        second = evaluate.paired_bootstrap(bundle, reference, candidate, samples=80, seed=41)

        self.assertEqual(first, second)
        self.assertEqual(first["sampling_unit"], "patient_with_all_Studies")
        self.assertTrue(first["paired"])
        self.assertEqual(first["slices"]["all"]["eligible_classes"], list(range(1, 11)))
        self.assertAlmostEqual(first["slices"]["all"]["macro_auc_delta"], 0.0)
        self.assertEqual(len(first["slices"]["all"]["per_class"]), 10)

    def test_patient_bootstrap_matches_explicit_cluster_replication(self):
        records = records_for_studies(8, images_per_study=1, ap_only=True)
        bundle = evaluate.aggregate_predictions(records, predictions_for(records))
        a = np.repeat(np.asarray([.1, .3, .5, .2, .9, .6, .7, .4])[:, None], 10, axis=1)
        b = np.repeat(np.asarray([.2, .5, .3, .9, .7, .4, .1, .6])[:, None], 10, axis=1)
        result = evaluate.paired_bootstrap(bundle, a, b, samples=60, seed=15)
        rng = np.random.default_rng(15)
        patients = np.unique(bundle["subject_ids"])
        class_deltas = []
        for _ in range(60):
            drawn = rng.integers(len(patients), size=len(patients))
            indices = np.concatenate([np.flatnonzero(bundle["subject_ids"] == patients[i]) for i in drawn])
            # Direct pairwise concordance provides independent tie handling.
            def direct_auc(predictions):
                y, p = bundle["targets"][indices, 0], predictions[indices, 0]
                comparisons = p[y == 1, None] - p[y == 0][None, :]
                return np.mean((comparisons > 0) + .5 * (comparisons == 0))
            class_deltas.append(direct_auc(b) - direct_auc(a))
        np.testing.assert_allclose(result["slices"]["all"]["per_class"][0]["ci95"],
                                   np.quantile(class_deltas, [.025, .975]))

    def test_sensitivity_threshold_is_stricter_for_rare_positive_classes(self):
        high = {"slices": {"all": {"per_class": [
            {"positive_studies": 12, "delta": .02, "ci95": [.005, .04]},
            {"positive_studies": 40, "delta": .02, "ci95": [.005, .04]},
        ] + [{"positive_studies": 40, "delta": None, "ci95": None} for _ in range(8)]}}}
        low = {"slices": {"all": {"per_class": [
            {"positive_studies": 12, "delta": -.001, "ci95": [-.005, .01]},
        ] + [{"positive_studies": 40, "delta": None, "ci95": None} for _ in range(9)]}}}
        result = evaluate.sensitivity_map(high, low)

        self.assertEqual(result[0]["high_resolution_required_delta"], .03)
        self.assertFalse(result[0]["train896_benefit_at_infer768"])
        self.assertEqual(result[1]["high_resolution_required_delta"], .01)
        self.assertTrue(result[1]["train896_benefit_at_infer768"])
        self.assertTrue(result[0]["train640_noninferior_at_infer768_margin_0_01"])

    def test_checkpoint_metadata_requires_official_valid_and_declared_roi_mil(self):
        checkpoint = {
            "artifact_type": "kd_student", "weights_source": "ema", "validation_complete": True,
            "inference_ready": True, "class_ids": list(range(1, 11)),
            "official_valid_manifest_sha256": "a" * 64, "image_size": 768, "epoch": 3,
            "experiment": {"roi": True, "mil": False, "roi_version": "synthetic_roi_version",
                           "inference_size": 768},
            "model_config": {"backbone_name": "convnext_tiny.fb_in22k_ft_in1k_384"},
            "model_state_dict": {"weight": object()},
        }
        with patch.dict("sys.modules", {"roi_scale": SimpleNamespace(VERSION="synthetic_roi_version")}):
            protocol = evaluate.validate_checkpoint_metadata(checkpoint, valid_sha="a" * 64, roi=True, mil=False)
        self.assertTrue(protocol["roi"])
        self.assertEqual(protocol["training_size"], 768)
        with self.assertRaisesRegex(ValueError, "CLI ROI/MIL"):
            evaluate.validate_checkpoint_metadata(checkpoint, valid_sha="a" * 64, roi=False, mil=False)
        with self.assertRaisesRegex(ValueError, "official-valid"):
            evaluate.validate_checkpoint_metadata(checkpoint, valid_sha="b" * 64)

    def test_checkpoint_registration_binds_explicit_sha_and_latest_compliance_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"synthetic checkpoint fixture")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            recorder = ComplianceRecorder(root, trainer="synthetic evaluator test",
                                          configuration={}, demonstration=True)
            recorder.checkpoint_saved(checkpoint, digest, epoch=3, role="last")
            registration = evaluate.validate_checkpoint_registration(checkpoint, digest)
            self.assertTrue(registration["chain_verified"])
            self.assertEqual(registration["checkpoint_epoch"], 3)
            self.assertEqual(registration["checkpoint_role"], "last")
            recorder.checkpoint_saved(checkpoint, "0" * 64, epoch=4, role="last")
            recorder.finish("completed")
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                evaluate.validate_checkpoint_registration(checkpoint, "not-a-sha")
            with self.assertRaisesRegex(ValueError, "latest compliance save"):
                evaluate.validate_checkpoint_registration(checkpoint, digest)

    def test_unmanaged_checkpoint_still_requires_sha_and_records_missing_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "snapshot.pt"
            checkpoint.write_bytes(b"synthetic historical checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result = evaluate.validate_checkpoint_registration(checkpoint, digest)
            self.assertFalse(result["chain_verified"])
            self.assertIn("no lifecycle ledger", result["status"])

    def test_saved_outputs_are_round_trip_loadable_and_hash_checked(self):
        records = records_for_studies(24, images_per_study=1, ap_only=True)
        views = predictions_for(records)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle = evaluate.save_predictions(output, records, views)
            metadata = {"valid_manifest_sha256": "a" * 64,
                        "prediction_files_sha256": {
                            name: evaluate.sha256(output / name)
                            for name in ("predictions.npz",)}}
            evaluate.write_json(output / "metadata.json", metadata)
            evaluate.write_json(output / "report.json", {
                "complete": True, "metadata_sha256": evaluate.sha256(output / "metadata.json")})
            loaded, _, _ = evaluate.load_evaluation(output)
            np.testing.assert_array_equal(loaded["study_keys"], bundle["study_keys"])
            np.testing.assert_allclose(loaded["hflip2"], bundle["hflip2"])
            self.assertEqual(list(output.glob("*.csv")), [])
            (output / "predictions.npz").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "predictions changed"):
                evaluate.load_evaluation(output)

    def test_timing_selection_uses_disjoint_three_image_valid_studies(self):
        records = records_for_studies(35, images_per_study=3, ap_only=True)
        measured, warmups = evaluate.timing_selection(records, count=30, seed=17)
        measured_keys = {group[0]["study_key"] for group in measured}
        warmup_keys = {group[0]["study_key"] for group in warmups}

        self.assertEqual(len(measured), 30)
        self.assertEqual(len(warmups), 5)
        self.assertFalse(measured_keys & warmup_keys)
        self.assertTrue(all(len(group) == 3 for group in [*measured, *warmups]))
        with self.assertRaisesRegex(ValueError, "official valid only"):
            evaluate.timing_selection([dict(records[0], split="train"), *records[1:]], count=30, seed=17)

    def test_official_records_has_no_test_route(self):
        with self.assertRaisesRegex(ValueError, "Only official train/valid"):
            evaluate.official_records(Path("unused"), split="test")

    def test_official_records_decodes_original_json_targets_at_manifest_boundary(self):
        row = {"targets": "[0,1,0,1,0,1,0,1,0,1]", "study_key": "P/S",
               "subject_id": "P", "study_id": "S"}
        with patch("deploy_kd.prediction_records", return_value=[row]) as read_records:
            with patch.object(evaluate, "TRAIN", Path("official-train-manifest.csv")):
                records = evaluate.official_records(Path("official-images"), split="train")
        read_records.assert_called_once_with(Path("official-images"), Path("official-train-manifest.csv"), split="train")
        self.assertEqual(records[0]["targets"], [0,1,0,1,0,1,0,1,0,1])
        for invalid in ("[0,1]", "not json", "[0,1,0,1,0,1,0,1,0,0.5]"):
            with patch("deploy_kd.prediction_records", return_value=[{**row, "targets": invalid}]):
                with patch.object(evaluate, "TRAIN", Path("official-train-manifest.csv")):
                    with self.assertRaises(ValueError):
                        evaluate.official_records(Path("official-images"), split="train")

    def test_csv_manifest_target_text_reaches_aggregation_as_binary_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'train/subject_00000/study_00000/pic_000000.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture path only; no image decoding')
            manifest=root/'train_manifest.csv'
            fields=['image_id','split','subject_id','study_id','study_key','relative_path','targets']
            with manifest.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                writer.writerow({'image_id':'train:000000','split':'train','subject_id':'subject_00000',
                    'study_id':'study_00000','study_key':'subject_00000/study_00000',
                    'relative_path':'train/subject_00000/study_00000/pic_000000.jpg',
                    'targets':'[0,1,0,1,0,1,0,1,0,1]'})
            with patch.object(evaluate,'TRAIN',manifest):
                records=evaluate.official_records(root,split='train')
            predictions=np.full((1,2,10),.5,dtype=np.float64)
            bundle=evaluate.aggregate_predictions(records,predictions)
            np.testing.assert_array_equal(bundle['targets'],[[0,1,0,1,0,1,0,1,0,1]])

    def test_train_gap_sample_is_study_complete_stable_and_train_only(self):
        records = [dict(row, split="train") for row in records_for_studies(20, images_per_study=3)]
        sampled = evaluate.train_gap_sample(records, count=7)
        keys = {row["study_key"] for row in sampled}
        self.assertEqual(len(keys), 7)
        self.assertEqual(len(sampled), 21)
        reordered = evaluate.train_gap_sample(list(reversed(records)), count=7)
        self.assertEqual(keys, {row["study_key"] for row in reordered})
        with self.assertRaisesRegex(ValueError, "official train only"):
            evaluate.train_gap_sample(records_for_studies(20), count=7)

    def test_compare_creates_only_fixed_equal_fusions_and_keeps_different_patients(self):
        records = records_for_studies(24, images_per_study=1, ap_only=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "valid_fixture.csv"
            manifest.write_text("synthetic manifest, no real images\n")
            inputs = []
            expected = []
            for name, offset in (("tiny_a", 0.), ("tiny_b", .02), ("mil", .01), ("champion041", .015)):
                output = root / name
                output.mkdir()
                bundle = evaluate.save_predictions(output, records, predictions_for(records) + offset)
                expected.append(bundle["hflip2"])
                metadata = {"valid_manifest_sha256": evaluate.sha256(manifest),
                    "source_checkpoint_sha256": "a" * 64, "selection_kind": "fixed_final_E3",
                    "prediction_files_sha256": {filename: evaluate.sha256(output / filename) for filename in
                        ("predictions.npz",)}}
                evaluate.write_json(output / "metadata.json", metadata)
                evaluate.write_json(output / "report.json", {"complete": True,
                    "metadata_sha256": evaluate.sha256(output / "metadata.json")})
                inputs.append(f"{name}={output}")
            args = SimpleNamespace(inputs=inputs, output=root / "comparison", bootstrap=5,
                                   seed=13, timing_fusions=False, image_root="unused")
            with patch.object(evaluate, "VALID", manifest):
                report = evaluate.compare(args)
            fusion = report["candidates"]["tiny_a_tiny_b_equal_hflip2"]
            self.assertEqual(fusion["probability_weights"], [.5, .5])
            self.assertTrue(all(abs(row["correlation"] - 1.) < 1e-12
                                for row in fusion["member_residual_correlations"]["per_class"]))
            self.assertEqual(report["num_patients"], 12)
            self.assertIn("tiny_a_tiny_b_equal_hflip2_minus_tiny_b_hflip2", report["comparisons"])
            self.assertIn("tiny_a_tiny_b_equal_hflip2_minus_mil_hflip2", report["comparisons"])
            self.assertIn("tiny_a_tiny_b_equal_hflip2_minus_champion041_hflip2", report["comparisons"])
            self.assertIn("mil_hflip2_minus_champion041_hflip2", report["comparisons"])
            with np.load(args.output / "candidate_study_probabilities.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["tiny_a_tiny_b_equal_hflip2"], .5 * (expected[0] + expected[1]))


if __name__ == "__main__":
    unittest.main()
