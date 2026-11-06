# 【学习导航】检查Study概率聚合、集成协议和提交格式，帮助理解为什么不能丢类别。
# 本次只打包与静态解析，没有执行此测试文件。
"""CPU-only probability/identity/provenance tests: no models, images or inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_overnight import (
    SAMPLE_HEADER, _checkpoint, load_ensemble_plan, normalized_weights, read_submission_template,
    sha256, write_matched_submission,
)
from dataset_overnight import aggregate_study_predictions


class EnsembleProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="overnight-protocol-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [
            {"image_id": "image-b2", "subject_id": "subject-b", "study_id": "study-b", "study_key": "subject-b/study-b", "targets": None},
            {"image_id": "image-a", "subject_id": "subject-a", "study_id": "study-a", "study_key": "subject-a/study-a", "targets": None},
            {"image_id": "image-b1", "subject_id": "subject-b", "study_id": "study-b", "study_key": "subject-b/study-b", "targets": None},
        ]
        self.sample = self.root / "sample.csv"
        # Deliberately interleave studies and reverse class order: no sorting.
        self.rows = [[f"subject-{subject}", f"study-{subject}", str(class_id), "nan"]
                     for class_id in range(10, 0, -1) for subject in ("a", "b")]
        self.write_sample(self.rows)

    def write_sample(self, rows: list[list[str]]) -> None:
        with self.sample.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(SAMPLE_HEADER)
            writer.writerows(rows)

    def result(self) -> dict:
        return aggregate_study_predictions(self.records, ["image-a", "image-b1", "image-b2"],
            np.stack([np.linspace(0, 1, 10), np.full(10, 0.2), np.full(10, 0.8)]))

    def test_nonuniform_view_counts_do_not_change_study_identity_or_mean(self) -> None:
        result = self.result()
        self.assertEqual(result["study_ids"], ["study-b", "study-a"])
        np.testing.assert_allclose(result["probabilities"][0], 0.5)
        self.assertIsNone(result["targets"])

    def test_exact_sample_identity_order_and_all_ten_probabilities(self) -> None:
        rows = read_submission_template(self.sample, self.records)
        destination = self.root / "submission_final.csv"
        report = write_matched_submission(destination, rows, self.result())
        self.assertEqual(report["rows"], 20)
        self.assertTrue(report["template_identity_order_equal"])
        with destination.open(newline="") as stream:
            reader = csv.reader(stream)
            self.assertEqual(next(reader), SAMPLE_HEADER)
            output = list(reader)
        self.assertEqual([r[:3] for r in output], [r[:3] for r in self.rows])
        self.assertEqual(output[0][3], "1.0")
        self.assertEqual(output[-2][3], "0.0")
        self.assertTrue(all(float(row[3]) == 0.5 for row in output[1::2]))

    def test_missing_or_duplicate_sample_class_is_rejected(self) -> None:
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            self.write_sample(rows)
            with self.assertRaises(ValueError):
                read_submission_template(self.sample, self.records)

    def test_wrong_subject_is_rejected(self) -> None:
        rows = [list(row) for row in self.rows]
        rows[0][0] = "other-subject"
        self.write_sample(rows)
        with self.assertRaises(ValueError):
            read_submission_template(self.sample, self.records)

    def test_nonfinite_and_out_of_range_predictions_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            result = self.result()
            result["probabilities"][0, 0] = value
            with self.assertRaises(ValueError):
                write_matched_submission(self.root / "bad.csv", self.rows, result)

    def test_weights_are_convex_and_equal_by_default(self) -> None:
        np.testing.assert_allclose(normalized_weights([{"fold": 0}, {"fold": 3}]), [0.5, 0.5])
        np.testing.assert_allclose(normalized_weights([{"fold": 0, "weight": 1}, {"fold": 4, "weight": 3}]), [0.25, 0.75])
        np.testing.assert_allclose(normalized_weights([{"fold": 0, "weight": 1e308}, {"fold": 4, "weight": 1e308}]), [0.5, 0.5])

    def test_invalid_weights_and_repeated_folds_rejected(self) -> None:
        for entries in ([], [{"fold": 0}, {"fold": 0}], [{"fold": 5}], [{"fold": False}],
                        [{"fold": 0, "weight": -1}], [{"fold": 0, "weight": 0}],
                        [{"fold": 0, "weight": float("nan")}], [{"fold": 0, "weight": True}]):
            with self.assertRaises(ValueError):
                normalized_weights(entries)

    def test_missing_and_duplicate_view_predictions_rejected(self) -> None:
        for ids in (["image-a", "image-b1"], ["image-a", "image-b1", "image-b1"]):
            with self.assertRaises(ValueError):
                aggregate_study_predictions(self.records, ids, np.full((len(ids), 10), 0.5))

    def make_plan(self) -> Path:
        fold_plan, checkpoint = self.root / "folds.json", self.root / "best.pt"
        fold_plan.write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(b"synthetic-file-no-tensors")
        plan = {"schema_version": 1, "fold_plan": fold_plan.name, "fold_plan_sha256": sha256(fold_plan),
            "checkpoints": [{"fold": 2, "path": checkpoint.name, "sha256": sha256(checkpoint), "weight": 1}],
            "tta": {"scales": [384, 512], "hflip": True}}
        plan_path = self.root / "ensemble_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan_path

    def test_partial_ensemble_plan_has_verified_hash_and_relative_paths(self) -> None:
        plan = load_ensemble_plan(self.make_plan())
        self.assertEqual(len(plan["checkpoints"]), 1)
        self.assertEqual(plan["checkpoints"][0]["normalized_weight"], 1.0)
        self.assertEqual(plan["checkpoints"][0]["path"], str(self.root / "best.pt"))

    def test_modified_checkpoint_rejected_before_model_load(self) -> None:
        path = self.make_plan()
        (self.root / "best.pt").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load_ensemble_plan(path)

    def test_duplicate_checkpoint_hash_rejected(self) -> None:
        path = self.make_plan()
        plan = json.loads(path.read_text())
        plan["checkpoints"].append({**plan["checkpoints"][0], "fold": 4})
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            load_ensemble_plan(path)

    def test_schema_five_metadata_accepts_training_contract_and_uses_ema(self) -> None:
        # These are two scalar CPU tensors, not a model or model forward.
        from model_overnight import MODEL_NAME, validate_model_config
        from train_overnight import preprocessing_config
        fold_path = self.root / "folds.json"
        fold_path.write_text(json.dumps({"source_manifest_sha256": "a" * 64}))
        plan = {"fold_plan": str(fold_path), "fold_plan_sha256": sha256(fold_path)}
        payload = {"schema_version": 5, "model_name": MODEL_NAME, "num_classes": 10,
            "class_ids": list(range(1, 11)), "image_size": 512, "epoch": 2,
            "weights_source": "ema", "validation_complete": True, "inference_ready": True,
            "aggregation": "study_mean_probability", "fold_index": 1,
            "fold_plan_sha256": plan["fold_plan_sha256"], "source_manifest_sha256": "a" * 64,
            "model_config": validate_model_config({}), "preprocessing": preprocessing_config(image_size=512),
            "inference_precision": "bfloat16", "validation": {"macro_auc": 0.8},
            "model_state_dict": {"weight": torch.tensor([7.0])},
            "raw_model_state_dict": {"weight": torch.tensor([3.0])}}
        path = self.root / "metadata_only.pt"
        torch.save(payload, path)
        entry = {"fold": 1, "path": str(path), "sha256": sha256(path)}
        loaded = _checkpoint(entry, plan)
        self.assertEqual(float(loaded["model_state_dict"]["weight"][0]), 7.0)
        self.assertNotIn("raw_model_state_dict", loaded)
        with self.assertRaises(ValueError):
            _checkpoint({**entry, "fold": 4}, plan)
        payload["image_size"] = 768
        payload["preprocessing"] = preprocessing_config()
        torch.save(payload, path)
        entry["sha256"] = sha256(path)
        self.assertEqual(_checkpoint(entry, plan)["image_size"], 768)
        payload["preprocessing"] = preprocessing_config(image_size=512)
        torch.save(payload, path)
        entry["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            _checkpoint(entry, plan)


if __name__ == "__main__":
    unittest.main()
