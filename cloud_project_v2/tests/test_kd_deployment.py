"""CPU protocol tests only: no actual model, model forward, or weight download."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost import compare_predictions, summarize_timings, time_round
from deploy_kd import (CLASS_IDS, CSV_HEADER, MEAN, STD, StudyAccumulator,
                       checked_probabilities, deployment_inventory, letterbox_rgb,
                       match_timing_manifest, read_submission, validate_preprocessing,
                       validate_student_checkpoint, validate_provenance, write_prediction,
                       TINY_BACKBONE, BASE_BACKBONE)
from export_student import estimate_size


class DeploymentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kd-deploy-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [
            {"image_id": "a", "study_key": "subject_1/study_1", "subject_id": "subject_1",
             "study_id": "study_1", "path": self.root / "a.jpg"},
            {"image_id": "b", "study_key": "subject_1/study_1", "subject_id": "subject_1",
             "study_id": "study_1", "path": self.root / "b.jpg"},
        ]

    def test_fp16_size_estimate_does_not_need_weights(self):
        estimate = estimate_size()
        self.assertEqual(estimate["fp16_parameter_bytes"], 55_655_636)
        self.assertEqual(estimate["fp32_parameter_bytes"], 111_311_272)
        self.assertEqual(estimate["latency_status"], "pending_T4_measurement")

    def test_legacy_v2_checkpoint_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            validate_student_checkpoint({"schema_version": 5, "model_name": "convnextv2_base_gap"})

    def test_actual_common_provenance_validator_rejects_nc_and_missing_teachers(self):
        # Synthetic unit-test materials, never represented as downloaded licenses.
        from kd_common import APPROVED_PRETRAINS, text_sha256
        def fixture(backbone):
            repo = APPROVED_PRETRAINS[backbone]["repo_id"]
            revision = "a" * 40
            materials = {"model_card": "license: apache-2.0\nSynthetic test fixture",
                         "upstream_license": "MIT License\nSynthetic test fixture",
                         "apache_license": "Apache License\nSynthetic test fixture"}
            return {**APPROVED_PRETRAINS[backbone], "backbone_name": backbone,
                    "revision": revision, "weights_sha256": "b" * 64,
                    "weights_url": f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors",
                    "materials": {k: {"text": v, "sha256": text_sha256(v)} for k, v in materials.items()}}
        student = fixture(TINY_BACKBONE)
        teachers = [{"fold_index": fold, "checkpoint_sha256": "c" * 64,
                     "selection_protocol": "fixed_epoch_final_ema", "training_complete": True,
                     "schema_version": 6, "artifact_type": "kd_teacher", "weights_source": "ema",
                     "model_name": "convnext_v1_base_gap", "outer_fold_used_for_selection": False,
                     "official_valid_used_for_training": False, "test_used_for_training": False,
                     "epoch": 20, "training_config": {"epochs": 20},
                     "training_subject_ids": ["train_patient"], "oof_subject_ids": ["heldout_patient"],
                     "pretrained_provenance": fixture(BASE_BACKBONE)} for fold in range(5)]
        validate_provenance(student, teachers)
        with self.assertRaises(ValueError):
            validate_provenance({**student, "license": "cc-by-nc-4.0"}, teachers)
        with self.assertRaises(ValueError):
            validate_provenance(student, teachers[:-1])
        teachers[0]["training_complete"] = False
        with self.assertRaises(ValueError):
            validate_provenance(student, teachers)

    def test_preprocessing_rejects_extra_scale_and_tta(self):
        valid = {"image_size": 512, "resize": "letterboxLANCZOS", "color_mode": "RGB",
                 "exif_transpose": True, "mean": MEAN, "std": STD,
                 "training_scales": [512], "validation_tta": "identity"}
        validate_preprocessing(valid)
        for update in ({"training_scales": [512, 384]}, {"validation_tta": "hflip"},
                       {"image_size": 768}, {"exif_transpose": False}):
            with self.assertRaises(ValueError):
                validate_preprocessing({**valid, **update})

    def test_letterbox_keeps_aspect_ratio_and_black_padding(self):
        result = np.asarray(letterbox_rgb(Image.new("RGB", (4, 2), (255, 255, 255)), 8))
        self.assertEqual(result.shape, (8, 8, 3))
        self.assertTrue((result[:2] == 0).all())
        self.assertTrue((result[2:6] == 255).all())
        self.assertTrue((result[6:] == 0).all())

    def test_study_aggregation_uses_all_views_once(self):
        acc = StudyAccumulator(self.records)
        self.assertIsNone(acc.add(self.records[0], np.full(10, 0.2)))
        subject, study, values = acc.add(self.records[1], np.full(10, 0.8))
        np.testing.assert_allclose(values, 0.5)
        self.assertEqual((subject, study), ("subject_1", "study_1"))
        self.assertTrue(acc.complete())
        with self.assertRaises(ValueError):
            acc.add(self.records[0], np.zeros(10))

    def test_invalid_probabilities_rejected(self):
        for value in (np.nan, np.inf, -0.01, 1.01):
            with self.assertRaises(ValueError):
                checked_probabilities(np.full(10, value))

    def test_timing_calls_same_per_image_predict_and_includes_writer(self):
        events = []
        class FakePredictor:
            config = {}
            def synchronize(self):
                events.append("sync")
            def predict_image(self, path):
                events.append(path.name)
                return np.full(10, 0.2 if path.name == "a.jpg" else 0.8)
        clock_values = iter((1.0, 1.01, 2.0, 2.02))
        durations, results = time_round(FakePredictor(), self.records, self.root / "round.csv",
                                         clock=lambda: next(clock_values))
        np.testing.assert_allclose(durations, [10, 20])
        self.assertEqual(events, ["sync", "a.jpg", "sync", "sync", "b.jpg", "sync"])
        self.assertTrue(compare_predictions(results, read_submission(self.root / "round.csv"))["match"])
        self.assertEqual(len((self.root / "round.csv").read_text().splitlines()), 11)
        with (self.root / "round.image_predictions.csv").open() as image_stream:
            image_rows = list(csv.DictReader(image_stream))
        self.assertEqual(len(image_rows), 20)
        self.assertEqual({r["Image_id"] for r in image_rows}, {"a", "b"})
        self.assertEqual([float(r["Probability"]) for r in image_rows[:10]], [0.2] * 10)

    def test_official_averaging_and_complete_rounds(self):
        rounds = [[float(i + r) for i in range(100)] for r in (0, 10, 20)]
        _, per_image, mean = summarize_timings(rounds)
        np.testing.assert_allclose(per_image, np.arange(100) + 10)
        self.assertEqual(mean, 59.5)
        with self.assertRaises(ValueError):
            summarize_timings([[1.0] * 100] * 2)

    def test_incomplete_study_timing_never_certifies_csv_agreement(self):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "split", "subject_id", "study_id", "study_key", "relative_path", "targets"])
            writer.writeheader()
            for record in self.records:
                writer.writerow({**{k: record[k] for k in ("image_id", "subject_id", "study_id", "study_key")},
                                 "split": "test", "relative_path": f"test/subject_1/study_1/{record['path'].name}", "targets": ""})
        self.assertFalse(match_timing_manifest(self.records[:1], manifest)["complete_studies_proven"])
        self.assertTrue(match_timing_manifest(self.records, manifest)["complete_studies_proven"])
        self.assertFalse(match_timing_manifest(self.records, None)["complete_studies_proven"])

    def test_submission_keeps_all_ten_probabilities(self):
        path = self.root / "submission.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_HEADER)
            write_prediction(writer, "subject_1", "study_1", np.zeros(10))
        self.assertEqual(len(read_submission(path)), 1)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]))
        with self.assertRaises(ValueError):
            read_submission(path)

    def test_inventory_counts_actual_weights_and_config(self):
        (self.root / "student_fp16.pt").write_bytes(b"x" * 123)
        (self.root / "config.json").write_text("{}")
        report = deployment_inventory(self.root, {"inference_files": ["student_fp16.pt", "config.json"]})
        self.assertEqual(report["total_bytes"], 125)
        self.assertTrue(report["size_pass"])
        with self.assertRaises(ValueError):
            deployment_inventory(self.root, {"inference_files": ["student_fp16.pt", "../config.json"]})


if __name__ == "__main__":
    unittest.main()
