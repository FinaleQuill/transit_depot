"""Meaningful CPU checks for token selection, all-view labels and recomputation."""
from __future__ import annotations

from copy import deepcopy
import unittest

import torch
from torch import nn

from breakthrough.model import DinoStudyModel


class TokenEncoder(nn.Module):
    num_features = 8
    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.reg_token = nn.Parameter(torch.randn(1, 4, 8))

    def forward_features(self, pixels):
        patches = self.projection(pixels).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(len(pixels), -1, -1) + patches.mean(1, keepdim=True)
        return torch.cat((cls, self.reg_token.expand(len(pixels), -1, -1), patches), dim=1)


class DinoStudyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return DinoStudyModel.from_backbone(TokenEncoder(), drop_rate=0., **kwargs)

    def test_registers_are_excluded_and_every_patch_is_used(self):
        model = self.model().eval()
        images = torch.randn(2, 3, 5, 5)
        tokens = model.backbone.forward_features(images)
        expected = model.head(torch.cat((tokens[:, 0], tokens[:, 5:].mean(1)), dim=-1))
        torch.testing.assert_close(model.image_logits(images), expected)
        with torch.no_grad():
            model.backbone.reg_token.fill_(1e5)
        torch.testing.assert_close(model.image_logits(images), expected, rtol=0, atol=0)
        images.requires_grad_(True)
        model(images).sum().backward()
        self.assertTrue(bool((images.grad.abs().sum(1) > 0).all()))

    def test_complete_studies_match_probability_means_without_mixing(self):
        model = self.model(view_chunk_size=2).eval()
        studies = [torch.randn(v, 3, 5, 5) for v in (1, 5, 2)]
        outputs = model.forward_studies(studies, return_components=True)
        for study, output in zip(studies, outputs):
            expected = model.image_logits(study).sigmoid().mean(0)
            torch.testing.assert_close(output["study_logits"].sigmoid(), expected)
            torch.testing.assert_close(output["study_logits"], model(study))
            torch.testing.assert_close(model(study.flip(0)), model(study))
        changed = [studies[0], studies[1] + 5, studies[2]]
        again = model.forward_studies(changed)
        torch.testing.assert_close(again[0], outputs[0]["study_logits"], rtol=0, atol=0)
        torch.testing.assert_close(again[2], outputs[2]["study_logits"], rtol=0, atol=0)

    def test_checkpointed_packed_backward_matches_uncheckpointed(self):
        plain = self.model(view_chunk_size=2).train()
        recomputed = deepcopy(plain)
        recomputed.gradient_checkpointing = True
        studies = [torch.randn(v, 3, 5, 5) for v in (1, 4, 2)]
        labels = torch.randint(0, 2, (3, 10)).float()
        for model in (plain, recomputed):
            logits = torch.stack(model.forward_studies(studies))
            nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), recomputed.named_parameters()):
            # The synthetic encoder deliberately keeps its registers uninvolved.
            if name == "backbone.reg_token":
                self.assertTrue(a.grad is None or bool((a.grad == 0).all()))
                self.assertTrue(b.grad is None or bool((b.grad == 0).all()))
            else:
                self.assertIsNotNone(a.grad, name)
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_checkpoint_preserves_dropout_rng(self):
        plain = DinoStudyModel.from_backbone(TokenEncoder(), drop_rate=.3, view_chunk_size=2).train()
        recomputed = deepcopy(plain)
        recomputed.gradient_checkpointing = True
        studies = [torch.randn(v, 3, 5, 5) for v in (3, 2)]
        observed = []
        for model in (plain, recomputed):
            torch.manual_seed(818)
            value = torch.stack(model.forward_studies(studies)).square().mean()
            value.backward()
            observed.append(value.detach())
        torch.testing.assert_close(*observed, rtol=0, atol=0)
        for (name, a), (_, b) in zip(plain.named_parameters(), recomputed.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_public_restore_is_strict_and_only_discards_imagenet_head(self):
        model = self.model()
        source = deepcopy(model.backbone.state_dict())
        model.load_public_state(source)
        with_head = dict(source, **{"head.weight": torch.ones(1000, 8), "head.bias": torch.zeros(1000)})
        model.load_public_state(with_head)
        with self.assertRaises(RuntimeError):
            model.load_public_state({"projection.weight": source["projection.weight"]})
        with self.assertRaises(ValueError):
            model.load_public_state(dict(source, **{"head.weight": torch.ones(10, 8), "head.bias": torch.zeros(10)}))

    def test_optimizer_roles_cover_each_parameter_once(self):
        model = self.model()
        groups = model.named_parameter_groups()
        base = {id(p) for _, p in groups["backbone"]}
        head = {id(p) for _, p in groups["head"]}
        self.assertFalse(base.intersection(head))
        self.assertEqual(base | head, {id(p) for p in model.parameters()})
        self.assertEqual(head, {id(p) for p in model.head.parameters()})


if __name__ == "__main__":
    unittest.main()
