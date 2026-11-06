"""CPU semantic checks for Study labels, all-view pooling and local hierarchy."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from reconstruction.model import StudyModel, mean_probability_logits, study_bce_loss


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.norm = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 10)

    def forward_features(self, images):
        return self.conv(images).tanh()

    def forward_head(self, features, pre_logits=False):
        values = self.norm(features.mean((-2, -1)))
        return values if pre_logits else self.fc(values)

    def get_classifier(self):
        return self.fc

    def forward(self, images):
        return self.forward_head(self.forward_features(images))


class StudyModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)
        self.backbone = DummyBackbone()
        self.images = torch.randn(7, 3, 6, 6)
        self.targets = torch.tensor([1., 0., 1., 0., 0., 1., 0., 0., 1., 0.])

    def make(self, **kwargs):
        return StudyModel.from_backbone(copy.deepcopy(self.backbone), attention_dim=12, **kwargs)

    def test_probability_mean_not_logit_mean_and_extreme_gradient(self):
        z = torch.tensor([[-5., 1000.], [1., -1000.], [2., 1000.]], requires_grad=True)
        pooled = mean_probability_logits(z)
        torch.testing.assert_close(pooled.sigmoid(), z.sigmoid().mean(0))
        self.assertGreater(float((pooled - z.mean(0)).abs().max().detach()), 100.)
        extreme = torch.full((3, 10), -1000., requires_grad=True)
        loss = study_bce_loss(mean_probability_logits(extreme), torch.ones(10))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(extreme.grad).all()))
        self.assertTrue(bool((extreme.grad < 0).all()))

    def test_e0_matches_original_classifier_and_preserves_every_view(self):
        model = self.make(view_chunk_size=2).eval()
        output = model(self.images, return_components=True)
        logits = model.backbone(self.images)
        torch.testing.assert_close(output["study_logits"].sigmoid(), logits.sigmoid().mean(0))
        self.assertEqual(output["region_logits"].shape, (7, 1, 10))
        torch.testing.assert_close(output["region_logits"][:, 0], logits)
        single = model(self.images[:1])
        torch.testing.assert_close(single, logits[0])
        changed = self.images.clone()
        changed[-1] += 3.
        self.assertFalse(torch.allclose(output["study_logits"], model(changed)))

    def test_attention_initialization_exactly_preserves_mean_baseline(self):
        baseline = self.make().eval()
        attention = self.make(aggregation="attention").eval()
        torch.testing.assert_close(attention(self.images), baseline(self.images), rtol=0, atol=0)
        components = attention(self.images, return_components=True)
        torch.testing.assert_close(components["view_attention"].sum(0), torch.ones(10))
        study_bce_loss(attention(self.images), self.targets).backward()
        self.assertGreater(float(attention.view_residual_weight.grad.abs().sum()), 0.)
        self.assertGreater(float(attention.backbone.conv.weight.grad.abs().sum()), 0.)

    def test_nonzero_attention_is_permutation_invariant_and_class_specific(self):
        model = self.make(aggregation="attention").eval()
        with torch.no_grad():
            model.view_query.normal_()
            model.view_residual_weight.normal_(std=.1)
        result = model(self.images, return_components=True)
        order = torch.tensor([5, 2, 6, 0, 3, 1, 4])
        shuffled = model(self.images[order], return_components=True)
        torch.testing.assert_close(result["study_logits"], shuffled["study_logits"], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(result["view_attention"][order], shuffled["view_attention"])
        self.assertFalse(torch.allclose(result["view_attention"][:, 0], result["view_attention"][:, 1]))

    def test_local_starts_global_only_then_receives_local_gradients(self):
        model = self.make(use_local=True, aggregation="attention").eval()
        pixels = torch.randn(3, 5, 3, 6, 6, requires_grad=True)
        result = model(pixels, return_components=True)
        global_reference = self.make().eval()(pixels[:, 0])
        torch.testing.assert_close(result["study_logits"], global_reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result["region_attention"].sum(1), torch.ones(3, 10))
        torch.testing.assert_close(result["view_attention"].sum(0), torch.ones(10))
        # After the zero residual head starts learning, every local input must
        # have a gradient path from the ONE Study label.
        with torch.no_grad():
            model.region_residual_weight.normal_(std=.1)
            model.view_residual_weight.normal_(std=.1)
        result = model(pixels, return_components=True)
        loss = study_bce_loss(result["study_logits"], self.targets,
                              global_study_logits=result["global_study_logits"], global_aux_weight=.2)
        loss.backward()
        self.assertTrue(bool((pixels.grad.abs().sum((2, 3, 4)) > 0).all()))

    def test_local_and_view_hierarchies_permute_with_coordinates(self):
        model = self.make(use_local=True, aggregation="attention").eval()
        pixels = torch.randn(3, 5, 3, 6, 6)
        coords = torch.tensor([[0, 0, 1, 1], [0, 0, .6, .6], [.4, 0, 1, .6],
                               [0, .4, .6, 1], [.4, .4, 1, 1.]]).expand(3, -1, -1)
        with torch.no_grad():
            model.region_query.normal_()
            model.region_residual_weight.normal_(std=.1)
            model.view_query.normal_()
            model.view_residual_weight.normal_(std=.1)
        expected = model(pixels, region_coordinates=coords)
        vorder, rorder = torch.tensor([2, 0, 1]), torch.tensor([0, 4, 2, 1, 3])
        actual = model(pixels[vorder][:, rorder], region_coordinates=coords[vorder][:, rorder])
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_checkpoint_chunks_retain_all_gradients_without_input_requires_grad(self):
        plain = self.make(view_chunk_size=3).train()
        checkpointed = self.make(view_chunk_size=2, gradient_checkpointing=True).train()
        study_bce_loss(plain(self.images), self.targets).backward()
        study_bce_loss(checkpointed(self.images), self.targets).backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            self.assertIsNotNone(b.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_parameter_groups_put_original_classifier_at_head_lr(self):
        model = self.make(aggregation="attention", use_local=True)
        groups = model.named_parameter_groups()
        head_ids = {id(p) for _, p in groups["head"]}
        backbone_ids = {id(p) for _, p in groups["backbone"]}
        self.assertEqual(head_ids & backbone_ids, set())
        self.assertEqual(head_ids | backbone_ids, {id(p) for p in model.parameters()})
        self.assertIn(id(model.backbone.fc.weight), head_ids)
        self.assertIn(id(model.backbone.norm.weight), backbone_ids)
        self.assertIn(id(model.region_residual_weight), head_ids)

    def test_packed_studies_share_encoder_and_match_independent_predictions(self):
        for local in (False, True):
            with self.subTest(local=local):
                model = self.make(aggregation="attention", use_local=local, view_chunk_size=3).eval()
                with torch.no_grad():
                    model.view_residual_weight.normal_(std=.1)
                    model.view_query.normal_()
                    if local:
                        model.region_residual_weight.normal_(std=.1)
                        model.region_query.normal_()
                studies = [torch.randn(v, 5, 3, 6, 6) if local else torch.randn(v, 3, 6, 6)
                           for v in (1, 3, 2)]
                expected = [model(x, return_components=True) for x in studies]
                with patch.object(model, "_encode", wraps=model._encode) as encoder:
                    actual = model.forward_studies(studies, return_components=True)
                self.assertEqual(encoder.call_count, 1)
                self.assertEqual(encoder.call_args.args[0].shape[0], 6 * (5 if local else 1))
                for one, packed in zip(expected, actual):
                    for key in one:
                        torch.testing.assert_close(one[key], packed[key], atol=2e-6, rtol=2e-6)

    def test_packed_and_independent_equal_study_gradients_match(self):
        independent = self.make(aggregation="attention", view_chunk_size=2).train()
        with torch.no_grad():
            independent.view_residual_weight.normal_(std=.1)
            independent.view_query.normal_()
        packed = copy.deepcopy(independent)
        packed.gradient_checkpointing = True
        packed.checkpoint_mode = "chunks"
        packed.channels_last = True
        packed.backbone.to(memory_format=torch.channels_last)
        studies = [torch.randn(v, 3, 6, 6) for v in (1, 4, 2)]
        targets = [self.targets, 1 - self.targets, self.targets]
        separate_loss = torch.stack([study_bce_loss(independent(x), y) for x, y in zip(studies, targets)]).mean()
        packed_loss = torch.stack([study_bce_loss(z, y) for z, y in zip(packed.forward_studies(studies), targets)]).mean()
        separate_loss.backward()
        packed_loss.backward()
        torch.testing.assert_close(separate_loss, packed_loss)
        for (name, a), (_, b) in zip(independent.named_parameters(), packed.named_parameters()):
            self.assertIsNotNone(b.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)

    def test_loss_matches_probability_bce_and_rejects_patch_targets(self):
        logits = self.make().eval()(self.images)
        weights = torch.linspace(1, 3, 10)
        p = logits.sigmoid()
        manual = -(weights * self.targets * p.log() + (1-self.targets) * torch.log1p(-p)).mean()
        actual = study_bce_loss(logits, self.targets, weights)
        torch.testing.assert_close(actual, manual)
        self.assertEqual(actual.dtype, torch.float32)
        with self.assertRaises(ValueError):
            study_bce_loss(logits, self.targets.expand(7, -1))
        with self.assertRaises(ValueError):
            self.make()(torch.randn(2, 5, 3, 6, 6))
        with self.assertRaises(ValueError):
            study_bce_loss(logits, self.targets, global_aux_weight=.2)
        with self.assertRaises(ValueError):
            self.make()(torch.empty(0, 3, 6, 6))


if __name__ == "__main__":
    unittest.main()
