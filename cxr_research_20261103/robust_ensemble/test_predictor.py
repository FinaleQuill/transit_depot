"""CPU checks for retained prediction math, fixed routing and input coverage."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from deploy_kd import sha256_file
from inference_inputs import ExactU8Input
from robust_ensemble.predictor import (FUSION_WEIGHTS, RobustPredictor,
                                       _checked_path, inference_inventory, validate_recipe)


def recipe():
    return {"fusion_weights": deepcopy(FUSION_WEIGHTS),
        "sprint_config": {"candidate_weights": "/unopened/base.pt",
            "candidate_weights_sha256": "a" * 64, "tiny_patch_config": "/unopened/tiny/config.json"},
        "opacity_member": {"id": "opacity512", "branch": "opacity", "image_size": 512,
            "epochs": 6, "weights": "/unopened/opacity.pt", "weights_sha256": "b" * 64}}


class DummyImageModel:
    def __init__(self, offset):
        self.offset, self.calls = offset, 0

    def __call__(self, image, return_components=False):
        self.calls += 1
        value = image[0, 0, 0, 0].float() + self.offset
        logits = value.expand(1, 10)
        return {"image_logits": logits} if return_components else logits


class RobustPredictorTests(unittest.TestCase):
    def predictor(self):
        predictor = RobustPredictor.__new__(RobustPredictor)
        predictor.config = recipe()
        predictor.torch = torch
        predictor.image_sizes = (768, 512)
        predictor.input_lookup = ExactU8Input(torch, "cpu", (0., 0., 0.), (1., 1., 1.), pin_memory=False)
        predictor.tiny = SimpleNamespace(model=DummyImageModel(-2.))
        predictor.base = DummyImageModel(0.)
        predictor.opacity = DummyImageModel(2.)
        return predictor

    def test_retained_sprint_math_tta_and_fixed_class1_are_exact(self):
        predictor = self.predictor()
        pixels = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        with patch("inference_inputs.load_image_pixels", return_value={512: pixels, 768: pixels}) as decode:
            fused, components = predictor.predict_image_components("synthetic")
        decode.assert_called_once()
        # This reproduces the original Sprint FP32 TTA then FP64 fusion order.
        means = np.array([torch.stack((torch.sigmoid(torch.tensor(float(v))),
                          torch.sigmoid(torch.tensor(float(v + 1))))).mean().item()
                          for v in (-2., 0., 2.)], dtype=np.float32)
        expected_sprint = np.full(10, .5 * float(means[0]) + .5 * float(means[1]), dtype=np.float64)
        expected_sprint[0] = float(means[1])
        np.testing.assert_array_equal(components[0], expected_sprint)
        np.testing.assert_array_equal(components[1], np.full(10, means[2], dtype=np.float64))
        self.assertEqual(fused[0], components[0, 0])
        np.testing.assert_allclose(fused[1:], .8 * components[0, 1:] + .2 * components[1, 1:], rtol=0, atol=1e-15)
        self.assertEqual([predictor.tiny.model.calls, predictor.base.calls, predictor.opacity.calls], [2, 2, 2])

    def test_finite_logits_guard_rejects_infinity_hidden_by_sigmoid(self):
        predictor = self.predictor()
        predictor.opacity = DummyImageModel(float("inf"))
        pixels = np.zeros((2, 2), dtype=np.uint8)
        with patch("inference_inputs.load_image_pixels", return_value={512: pixels, 768: pixels}):
            with self.assertRaises(FloatingPointError):
                predictor.predict_image_components("synthetic")

    def test_study_mean_keeps_all_views_and_rejects_invalid_coverage(self):
        predictor = RobustPredictor.__new__(RobustPredictor)
        predictor.predict_image = lambda path: np.full(10, float(path))
        rows = [{"study_key": "p/s", "subject_id": "p", "study_id": "s",
                 "image_id": str(i), "path": value} for i, value in enumerate((.1, .3, .8))]
        result, images = predictor.predict_study(rows)
        self.assertEqual(result[:2], ("p", "s"))
        self.assertEqual(len(images), 3)
        np.testing.assert_allclose(result[2], .4)
        for malformed in ([], rows + [rows[0]], rows + [dict(rows[0], study_key="p/other")]):
            with self.assertRaises(ValueError):
                predictor.predict_study(malformed)

    def test_recipe_rejects_drift_without_opening_any_weights(self):
        validate_recipe(recipe())
        for change in (
            lambda c: c["fusion_weights"][1].__setitem__(7, .25),
            lambda c: c["fusion_weights"][1].__setitem__(0, .2),
            lambda c: c["opacity_member"].__setitem__("epochs", 5),
            lambda c: c["sprint_config"].__setitem__("image_size", 512),
            lambda c: c.__setitem__("patient_smoothing", True),
            lambda c: c["opacity_member"].pop("weights_sha256")):
            config = recipe()
            change(config)
            with self.assertRaises(ValueError):
                validate_recipe(config)

    def test_asset_inventory_counts_configs_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("tiny.pt", "tiny_patch.pt", "config.json", "base.pt", "opacity.pt")]
            for path in paths:
                path.write_bytes(path.name.encode())
            tiny = {"files": [{"path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)} for path in paths[:3]]}
            inventory = inference_inventory(recipe(), tiny, *paths[-2:])
            self.assertEqual(len(inventory["files"]), 5)
            self.assertEqual(inventory["total_bytes"], sum(path.stat().st_size for path in paths)
                             + inventory["configuration_bytes"])
            identity = sha256_file(paths[-1])
            _checked_path(paths[-1], identity, "opacity")
            paths[-1].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                _checked_path(paths[-1], identity, "opacity")
            with patch("robust_ensemble.predictor.MAX_MODEL_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "500MB"):
                    inference_inventory(recipe(), tiny, *paths[-2:])


if __name__ == "__main__":
    unittest.main()
