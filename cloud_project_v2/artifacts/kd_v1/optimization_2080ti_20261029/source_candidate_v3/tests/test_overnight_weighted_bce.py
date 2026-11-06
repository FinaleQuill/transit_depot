"""CPU loss/statistics checks only: no model, image decoding or CUDA access."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

import torch

from train_overnight import compute_loss_config, build_weighted_bce, preprocessing_config


def record(image: str, study: int, targets: list[int]) -> dict:
    return {"image_id": image, "split": "train", "training_eligible": True,
            "subject_id": f"p{study}", "study_id": f"s{study}",
            "study_key": f"p{study}/s{study}", "targets": targets}


class WeightedBceTests(unittest.TestCase):
    def setUp(self) -> None:
        # Three views from the positive Study must still count as one positive.
        self.records = [record(f"a{view}", 0, [1] * 10) for view in range(3)]
        self.records += [record("b", 1, [0] * 10), record("c", 2, [0] * 10)]

    def statistics(self, records: list[dict] | None = None) -> dict:
        return compute_loss_config(records if records is not None else self.records,
                                   source_manifest_sha256="a" * 64,
                                   fold=0, fold_plan_sha256="b" * 64)

    def test_study_not_image_positive_counts_and_order_independence(self) -> None:
        result = self.statistics()
        self.assertEqual(result["num_images"], 5)
        self.assertEqual(result["num_studies"], 3)
        self.assertEqual(result["positive_studies"], [1] * 10)
        self.assertEqual(result["negative_studies"], [2] * 10)
        self.assertEqual(result["pos_weight"], [2.] * 10)
        self.assertEqual(result, self.statistics(list(reversed(self.records))))

    def test_bce_scales_only_positive_term(self) -> None:
        criterion = build_weighted_bce(self.statistics())
        targets = torch.tensor([[1.] * 10, [0.] * 10])
        values = criterion(torch.zeros_like(targets), targets)
        torch.testing.assert_close(values[0], torch.full((10,), 2 * math.log(2)))
        torch.testing.assert_close(values[1], torch.full((10,), math.log(2)))

    def test_study_weights_and_pos_weight_have_distinct_roles(self) -> None:
        criterion = build_weighted_bce(self.statistics())
        targets = torch.tensor([row["targets"] for row in self.records], dtype=torch.float32)
        per_image = criterion(torch.zeros_like(targets), targets).mean(1)
        # N/(S*n_views) preserves equal Study weighting despite different views.
        weights = torch.tensor([5 / (3 * 3)] * 3 + [5 / 3] * 2)
        observed = (per_image * weights).sum() / 5
        expected = (2 * math.log(2) + 2 * math.log(2)) / 3
        self.assertAlmostEqual(observed.item(), expected, places=6)

    def test_extreme_logits_remain_finite(self) -> None:
        criterion = build_weighted_bce(self.statistics())
        targets = torch.tensor([[1.] * 10, [0.] * 10])
        values = criterion(torch.tensor([[-10000.] * 10, [10000.] * 10]), targets)
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(values[0, 0].item(), 20000.)
        self.assertEqual(values[1, 0].item(), 10000.)

    def test_holdout_or_ineligible_labels_cannot_fit_weights(self) -> None:
        for field, invalid in (("split", "valid"), ("training_eligible", False)):
            records = deepcopy(self.records)
            records[0][field] = invalid
            with self.assertRaisesRegex(ValueError, "original train labels only"):
                self.statistics(records)

    def test_inconsistent_study_or_duplicate_image_is_rejected(self) -> None:
        records = deepcopy(self.records)
        records[0]["targets"][0] = 0
        with self.assertRaisesRegex(ValueError, "Labels disagree"):
            self.statistics(records)
        records = deepcopy(self.records)
        records[1]["image_id"] = records[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "unique nonempty image IDs"):
            self.statistics(records)

    def test_absent_class_fails_instead_of_silent_infinite_weight(self) -> None:
        records = deepcopy(self.records)
        for row in records:
            row["targets"][0] = 0
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            self.statistics(records)

    def test_fixed_768_and_historical_preprocessing_are_explicit(self) -> None:
        current = preprocessing_config()
        self.assertEqual(current["image_size"], 768)
        self.assertEqual(current["validation_size"], 768)
        self.assertEqual(current["training_scales"], [768])
        legacy = preprocessing_config(image_size=512)
        self.assertEqual(legacy["training_scales"], [512, 448, 512, 384, 512])
        self.assertEqual(legacy["validation_size"], 512)


if __name__ == "__main__":
    unittest.main()
