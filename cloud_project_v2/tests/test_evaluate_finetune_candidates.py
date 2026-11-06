"""CPU arithmetic and tamper guards for the finite fine-tuning comparison."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_finetune_candidates as evaluation


class FineTuneComparisonTests(unittest.TestCase):
    def fixture(self, root):
        records = []
        images = {}
        for index in range(4):
            for view in range(1 if index < 2 else 2):
                image_id = f"{index}_{view}"
                row = {"image_id": image_id, "split": "valid", "subject_id": f"p{index//2}",
                       "study_id": f"s{index}", "targets": [index % 2] * 10}
                records.append(row)
                images[image_id] = np.full(10, 0.2 + 0.5 * (index % 2) + 0.1 * view)
        labels, views = evaluation.validation_labels(records)
        predictions = evaluation.aggregate_images(records, images)
        scores = evaluation.score_probabilities(labels, predictions, views)
        image_path = root / "valid.image_predictions.csv"
        with image_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(evaluation.IMAGE_HEADER)
            for row in records:
                for c, value in enumerate(images[row["image_id"]], 1):
                    writer.writerow([row["image_id"], row["subject_id"], row["study_id"], c, repr(float(value))])
        study_path = root / "valid_predictions.csv"
        with study_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(evaluation.SUBMISSION_HEADER)
            for (subject, study), values in predictions.items():
                for c, value in enumerate(values, 1):
                    writer.writerow([subject, study, c, repr(float(value))])
        report = {**scores, "num_images": len(records), "complete": True,
            "manifest_sha256": "f" * 64, "tta": "hflip2", "image_size": 768, "precision": "float16",
            "member_count": 1, "test_images_used": False,
            "predictions_sha256": evaluation.sha256_file(study_path),
            "image_predictions_sha256": evaluation.sha256_file(image_path)}
        (root / "validation_report.json").write_text(json.dumps(report))
        return records, labels, views, images, report

    def test_plan_rejects_unregistered_weight_search_and_wrong_sha(self):
        plan = {"manifest_sha256": "f" * 64, "test_used": False,
            "registered_before_new_model_evaluation": True,
            "rules": [{"name": "X1_B_class1_B", "members": ["X1", "B"],
                       "weights": copy.deepcopy(evaluation.PAIR_WEIGHTS)}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            digest = evaluation.sha256_file(path)
            evaluation.load_registered_plan(path, digest, "f" * 64)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluation.load_registered_plan(path, "0" * 64, "f" * 64)
            plan["rules"][0]["weights"][0][1] = 0.9
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "weights differ"):
                evaluation.load_registered_plan(path, evaluation.sha256_file(path), "f" * 64)

    def test_full_image_and_study_audit_then_detects_bound_probability_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, labels, views, _, _ = self.fixture(root)
            _, _, scores, identity = evaluation.load_validation(root, records, "f" * 64, labels, views)
            self.assertEqual(scores["macro_auc"], 1.0)
            self.assertTrue(identity["image_to_study_bitwise_equal"])
            with (root / "valid.image_predictions.csv").open("a") as stream:
                stream.write("bad\n")
            with self.assertRaisesRegex(ValueError, "SHA-bound"):
                evaluation.load_validation(root, records, "f" * 64, labels, views)

    def test_hash_updated_wrong_study_mean_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, labels, views, _, report = self.fixture(root)
            path = root / "valid_predictions.csv"
            path.write_text(path.read_text().replace("0.2", "0.3", 1))
            report["predictions_sha256"] = evaluation.sha256_file(path)
            (root / "validation_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Study predictions differ"):
                evaluation.load_validation(root, records, "f" * 64, labels, views)

    def test_class1_uses_only_base_and_other_classes_are_fixed_equal_means(self):
        records = [{"image_id": "a", "subject_id": "p", "study_id": "s"},
                   {"image_id": "b", "subject_id": "p", "study_id": "s"}]
        images = {"X1": {"a": np.full(10, .2), "b": np.full(10, .4)},
                  "B": {"a": np.full(10, .8), "b": np.full(10, .6)}}
        result = evaluation.fixed_image_rule(records, images, ["X1", "B"], evaluation.PAIR_WEIGHTS)
        np.testing.assert_allclose(result[("p", "s")], [.7] + [.5] * 9, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
