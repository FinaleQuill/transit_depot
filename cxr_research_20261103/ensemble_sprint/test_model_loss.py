"""CPU mechanics: warm starts, attention gradients, Study isolation and ranking."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from ensemble_sprint.loss import EnsembleLoss
from ensemble_sprint.model import EnsembleStudyModel
from reconstruction.model import StudyModel
from reconstruction.test_model import DummyBackbone


class EnsembleModelLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261105)
        torch.set_num_threads(1)

    def test_both_branches_preserve_parent_and_opacity_gate_can_start_learning(self):
        backbone = DummyBackbone()
        reference = StudyModel.from_backbone(copy.deepcopy(backbone)).eval()
        pixels = torch.randn(3, 3, 7, 7)
        for branch in ("global", "opacity"):
            model = EnsembleStudyModel.from_backbone(copy.deepcopy(backbone), branch=branch).eval()
            with patch.object(model.backbone, "forward_features", wraps=model.backbone.forward_features) as encoder:
                output = model(pixels, return_components=True)
            self.assertEqual(encoder.call_count, 1)
            torch.testing.assert_close(output["study_logits"], reference(pixels), rtol=0, atol=0)
            self.assertEqual(output["image_logits"].shape, (3, 10))
            if branch == "opacity":
                output["study_logits"].square().sum().backward()
                self.assertGreater(float(model.spatial_gate.grad.abs().sum()), 0.)
                self.assertEqual(float(model.spatial_attention.weight.grad.abs().sum()), 0.)
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.spatial_gate.fill_(.1)
                model(pixels).square().sum().backward()
                self.assertGreater(float(model.spatial_attention.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.backbone.conv.weight.grad.abs().sum()), 0.)

    def test_packed_studies_do_not_mix_and_all_views_receive_gradients(self):
        model = EnsembleStudyModel.from_backbone(DummyBackbone(), branch="opacity", view_chunk_size=2,
            gradient_checkpointing=True, checkpoint_mode="chunks").eval()
        with torch.no_grad():
            model.spatial_gate.fill_(.2)
        studies = [torch.randn(n, 3, 6, 6, requires_grad=True) for n in (1, 3, 2)]
        packed = model.forward_studies(studies)
        for pixels, output in zip(studies, packed):
            torch.testing.assert_close(output, model(pixels), atol=2e-6, rtol=2e-6)
        changed = [pixels.detach().clone() for pixels in studies]
        changed[1] += 5
        again = model.forward_studies(changed)
        torch.testing.assert_close(packed[0], again[0], atol=0, rtol=0)
        torch.testing.assert_close(packed[2], again[2], atol=0, rtol=0)
        model.train()
        torch.stack(model.forward_studies(studies)).square().mean().backward()
        for pixels in studies:
            self.assertTrue(bool((pixels.grad.abs().sum((1, 2, 3)) > 0).all()))

    def test_restore_is_strict_and_new_parameters_belong_to_head(self):
        model = EnsembleStudyModel.from_backbone(DummyBackbone(), branch="opacity")
        parent = {"backbone." + key: value for key, value in model.backbone.state_dict().items()}
        model.load_legacy_backbone(parent, prefix="backbone.")
        with self.assertRaises(ValueError):
            model.load_legacy_backbone({**parent, "foreign": torch.zeros(1)}, prefix="backbone.")
        with self.assertRaises(RuntimeError):
            model.load_legacy_backbone({"fc.bias": torch.zeros(10)})
        heads = {id(parameter) for _, parameter in model.named_parameter_groups()["head"]}
        self.assertIn(id(model.spatial_gate), heads)
        self.assertIn(id(model.spatial_attention.weight), heads)
        with self.assertRaises(ValueError):
            EnsembleStudyModel.from_backbone(DummyBackbone(), branch="opacity", use_local=True)

    def test_class_weighted_ranking_and_same_patient_exclusion(self):
        criterion = EnsembleLoss(torch.ones(10), queue_capacity=4)
        historical = torch.zeros(1, 10)
        criterion.enqueue(historical, torch.zeros_like(historical), patient_ids=["a"])
        logits = torch.zeros(1, 10, requires_grad=True)
        positive = torch.ones_like(logits)
        excluded = criterion(logits, positive, patient_ids=["a"], return_components=True)
        self.assertEqual(float(excluded["ranking"]), 0.)
        loss = criterion(logits, positive, patient_ids=["b"], return_components=True)
        loss["ranking"].backward()
        self.assertTrue(bool((logits.grad < 0).all()))
        self.assertAlmostEqual(float(logits.grad[0, 7] / logits.grad[0, 1]), 2.)
        self.assertAlmostEqual(float(logits.grad[0, 0] / logits.grad[0, 1]), 1.5)
        self.assertAlmostEqual(float(logits.grad[0, 3] / logits.grad[0, 1]), 1.25)
        self.assertFalse(criterion.queue_logits.requires_grad)
        restored = EnsembleLoss(torch.ones(10), queue_capacity=4)
        restored.load_state_dict(criterion.state_dict(), strict=True)
        torch.testing.assert_close(restored.class_weight, criterion.class_weight)
        torch.testing.assert_close(restored.queue_patients, criterion.queue_patients)
        with self.assertRaises(RuntimeError):
            criterion.eval()(logits, positive, patient_ids=["b"])
        for values in ([1.] * 9, [0.] * 10, [float("nan")] * 10):
            with self.assertRaises(ValueError):
                EnsembleLoss(torch.ones(10), class_weights10=values)


if __name__ == "__main__":
    unittest.main()
