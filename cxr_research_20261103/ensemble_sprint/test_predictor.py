"""CPU mechanics tests: fixed routing, averaging, coverage and final selection."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ensemble_sprint.predictor import (EnsemblePredictor, FUSION_WEIGHTS,
    checked_member_payload, validate_recipe)
from ensemble_sprint.evaluate import selection_checks
from inference_inputs import ExactU8Input


def recipe():
    return {"fusion_weights": deepcopy(FUSION_WEIGHTS), "members": [
        {"id": "context512", "branch": "global", "image_size": 512, "epochs": 4},
        {"id": "opacity512", "branch": "opacity", "image_size": 512, "epochs": 6}]}


class DummyImageModel:
    def __init__(self, offset):
        self.offset = offset
        self.calls = 0

    def __call__(self, image, return_components=False):
        self.calls += 1
        value = image[0, 0, 0, 0].float() + self.offset
        return {"image_logits": value.expand(1, 10)}


class PredictorTests(unittest.TestCase):
    def test_frozen_class_routing_and_tta_mean_of_probabilities(self):
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.config = recipe()
        predictor.torch = torch
        predictor.image_sizes = [768, 512, 512]
        predictor.input_lookup = ExactU8Input(torch, "cpu", (0., 0., 0.), (1., 1., 1.), pin_memory=False)
        predictor.models = [DummyImageModel(v) for v in (-2., 0., 2.)]
        pixels = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        with patch("inference_inputs.load_image_pixels", return_value={512: pixels, 768: pixels}) as decode:
            fused, members = predictor.predict_image_components("synthetic")
        decode.assert_called_once()
        expected = np.array([(torch.sigmoid(torch.tensor(float(v))) +
            torch.sigmoid(torch.tensor(float(v+1)))) / 2 for v in (-2., 0., 2.)])
        np.testing.assert_allclose(members[:, 0], expected, rtol=0, atol=1e-7)
        self.assertEqual(float(fused[0]), float(members[0, 0]))
        self.assertAlmostEqual(float(fused[1]), float(members[:, 1] @ np.array([.5, .25, .25])))
        self.assertAlmostEqual(float(fused[7]), float(members[:, 7] @ np.array([.4, .2, .4])))
        self.assertEqual([model.calls for model in predictor.models], [2, 2, 2])

    def test_study_preserves_every_view_and_rejects_duplicate_or_mixed_studies(self):
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.predict_image = lambda path: np.full(10, float(path))
        rows = [{"study_key": "p/s", "subject_id": "p", "study_id": "s",
                 "image_id": str(i), "path": value} for i, value in enumerate((.1, .3, .8))]
        result, images = predictor.predict_study(rows)
        self.assertEqual(result[:2], ("p", "s"))
        self.assertEqual(len(images), 3)
        np.testing.assert_allclose(result[2], .4)
        with self.assertRaises(ValueError):
            predictor.predict_study(rows + [rows[0]])
        with self.assertRaises(ValueError):
            predictor.predict_study(rows + [dict(rows[0], study_key="p/other")])

    def test_recipe_cannot_silently_change_class8_or_member_identity(self):
        config = recipe()
        validate_recipe(config)
        config["fusion_weights"][0][7] = .5
        with self.assertRaises(ValueError):
            validate_recipe(config)
        config = recipe()
        config["members"].reverse()
        with self.assertRaises(ValueError):
            validate_recipe(config)

    def test_only_final_finite_fp16_member_export_is_accepted(self):
        member = recipe()["members"][0]
        payload = {"configuration": member, "epoch": 4, "state_dict": {"weight": torch.ones(2).half()}}
        checked_member_payload(payload, member, torch)
        with self.assertRaises(ValueError):
            checked_member_payload(dict(payload, epoch=3), member, torch)
        with self.assertRaises(ValueError):
            checked_member_payload(dict(payload, state_dict={"weight": torch.ones(2)}), member, torch)
        with self.assertRaises(ValueError):
            checked_member_payload(dict(payload, state_dict={"weight": torch.tensor([float("nan")]).half()}), member, torch)

    def test_class8_regression_blocks_macro_win_and_macro_tie_is_not_win(self):
        old = {"macro_auc": .818, "per_class_auc": [.8] * 10}
        new = {"macro_auc": .819, "per_class_auc": [.8] * 10}
        self.assertTrue(all(selection_checks(new, old).values()))
        new["per_class_auc"][7] = .799
        self.assertFalse(all(selection_checks(new, old).values()))
        new["per_class_auc"][7] = .801
        new["macro_auc"] = .818
        self.assertFalse(all(selection_checks(new, old).values()))


if __name__ == "__main__":
    unittest.main()
