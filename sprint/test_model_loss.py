"""Small CPU checks for the new architecture and training objective."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from reconstruction.model import StudyModel
from reconstruction.test_model import DummyBackbone
from sprint.model import SprintStudyModel
from sprint.loss import SprintLoss


class SprintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def test_zero_residual_preserves_legacy_predictions_and_single_encoder(self):
        backbone = DummyBackbone()
        base = StudyModel.from_backbone(copy.deepcopy(backbone)).eval()
        sprint = SprintStudyModel.from_backbone(copy.deepcopy(backbone), spatial_dim=12).eval()
        images = torch.randn(3, 3, 8, 8)
        with patch.object(sprint.backbone, "forward_features", wraps=sprint.backbone.forward_features) as encoder:
            actual = sprint(images)
            self.assertEqual(encoder.call_count, 1)
        torch.testing.assert_close(actual, base(images), rtol=0, atol=0)
        actual.sum().backward()
        self.assertGreater(float(sprint.spatial_head[-1].weight.grad.abs().sum()), 0)
        head_ids = {id(p) for _, p in sprint.named_parameter_groups()["head"]}
        self.assertTrue(all(id(p) in head_ids for p in sprint.spatial_head.parameters()))

    def test_packing_does_not_mix_studies_and_legacy_restore_is_strict(self):
        model = SprintStudyModel.from_backbone(DummyBackbone(), spatial_dim=12).eval()
        with torch.no_grad():
            model.spatial_head[-1].weight.normal_(std=0.1)
        studies = [torch.randn(n, 3, 8, 8) for n in (2, 1, 3)]
        packed = model.forward_studies(studies)
        for x, result in zip(studies, packed):
            torch.testing.assert_close(model(x), result)
        changed = [x.clone() for x in studies]
        changed[1] += 7
        again = model.forward_studies(changed)
        torch.testing.assert_close(packed[0], again[0], atol=0, rtol=0)
        torch.testing.assert_close(packed[2], again[2], atol=0, rtol=0)
        state = {"backbone." + key: value for key, value in model.backbone.state_dict().items()}
        model.load_legacy_backbone(state, prefix="backbone.")
        with self.assertRaises(ValueError):
            model.load_legacy_backbone({**state, "other": torch.zeros(1)}, prefix="backbone.")
        with self.assertRaises(RuntimeError):
            model.load_legacy_backbone({"fc.bias": torch.zeros(10)})

    def test_pairwise_ranking_pushes_positives_up_and_negatives_down(self):
        loss = SprintLoss(torch.ones(10))
        loss.enqueue(torch.zeros(2, 10), torch.stack([torch.zeros(10), torch.ones(10)]))
        logits = torch.zeros(2, 10, requires_grad=True)
        targets = torch.stack([torch.ones(10), torch.zeros(10)])
        parts = loss(logits, targets, return_components=True)
        parts["ranking"].backward()
        self.assertTrue(bool((logits.grad[0] < 0).all()))
        self.assertTrue(bool((logits.grad[1] > 0).all()))
        self.assertAlmostEqual(float(logits.grad[0, 0] / logits.grad[0, 1]), 2.0)
        self.assertEqual(int(loss.queue_size), 2)  # forward never enqueues

    def test_single_class_fallback_and_queue_is_detached_bounded_and_restorable(self):
        loss = SprintLoss(torch.ones(10), queue_capacity=3)
        logits = torch.zeros(2, 10, requires_grad=True)
        labels = torch.zeros(2, 10)
        self.assertEqual(float(loss(logits, labels, return_components=True)["ranking"]), 0.0)
        for _ in range(4):
            loss.enqueue(logits, labels)
        self.assertEqual(int(loss.queue_size), 3)
        self.assertFalse(loss.queue_logits.requires_grad)
        self.assertIsNone(loss.queue_logits.grad_fn)
        parts = loss(logits, labels, return_components=True)
        self.assertEqual(float(parts["ranking"]), 0.0)
        self.assertTrue(bool(torch.isfinite(parts["loss"])))
        parts["loss"].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        restored = SprintLoss(torch.ones(10), queue_capacity=3)
        restored.load_state_dict(loss.state_dict(), strict=True)
        torch.testing.assert_close(restored.queue_logits, loss.queue_logits)
        torch.testing.assert_close(restored.queue_cursor, loss.queue_cursor)
        with self.assertRaises(RuntimeError):
            loss.eval()(logits, labels)
        with self.assertRaises(RuntimeError):
            loss.enqueue(logits, labels)

    def test_same_patient_contradictory_studies_are_excluded_from_ranking(self):
        loss = SprintLoss(torch.ones(10))
        loss.enqueue(torch.zeros(1, 10), torch.zeros(1, 10), patient_ids=["patient-a"])
        logits = torch.zeros(1, 10, requires_grad=True)
        labels = torch.ones(1, 10)
        same = loss(logits, labels, patient_ids=["patient-a"], return_components=True)
        other = loss(logits, labels, patient_ids=["patient-b"], return_components=True)
        self.assertEqual(float(same["ranking"]), 0.0)
        self.assertGreater(float(other["ranking"]), 0.0)
        restored = SprintLoss(torch.ones(10))
        restored.load_state_dict(loss.state_dict(), strict=True)
        torch.testing.assert_close(restored.queue_patients, loss.queue_patients)


if __name__ == "__main__":
    unittest.main()
