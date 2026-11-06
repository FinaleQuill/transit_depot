"""CPU-only MIL checks with random timm weights; no images or downloads."""
import copy
import unittest

import torch
import timm

from spatial_mil import (SpatialMILHead, benchmark_spatial_mil,
                         forward_attention, install_spatial_mil)


class SpatialMILTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(4403)
        cls.model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k_384",
                                     pretrained=False, num_classes=10,
                                     global_pool="avg", drop_rate=.2).eval()
        cls.gap_head = copy.deepcopy(cls.model.head).eval()
        cls.original_keys = set(cls.model.state_dict())
        cls.original_norm = cls.model.head.norm
        cls.original_fc = cls.model.head.fc
        install_spatial_mil(cls.model)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def make_head(self):
        return SpatialMILHead(copy.deepcopy(self.gap_head)).eval()

    def test_uniform_attention_matches_existing_post_pool_norm_head(self):
        for shape in ((2, 768, 3, 5), (1, 768, 1, 1), (2, 768, 4, 4)):
            features = torch.randn(shape)
            head = self.make_head()
            with torch.no_grad():
                expected = self.gap_head(features)
                actual, maps = head.forward_with_attention(features)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(maps.flatten(2).sum(-1), torch.ones(shape[0], 10))
            torch.testing.assert_close(maps, torch.full_like(maps, 1 / (shape[2] * shape[3])))
            self.assertEqual(head(features, pre_logits=True).shape, (shape[0], 10, 768))

    def test_parameter_names_and_objects_preserve_strict_warm_start(self):
        model = self.model
        self.assertIs(model.head.norm, self.original_norm)
        self.assertIs(model.head.fc, self.original_fc)
        self.assertEqual(set(model.state_dict()) - self.original_keys, {"head.attention.weight"})
        self.assertEqual(self.original_keys - set(model.state_dict()), set())
        self.assertEqual(model.head.attention.weight.numel(), 7680)
        self.assertEqual(torch.count_nonzero(model.head.attention.weight).item(), 0)
        state = copy.deepcopy(model.head.state_dict())
        restored = self.make_head()
        restored.load_state_dict(state, strict=True)
        self.assertEqual(set(restored.state_dict()), set(state))
        same_head = model.head
        self.assertIs(install_spatial_mil(model), model)
        self.assertIs(model.head, same_head)

    def test_uniform_fp16_logits_preserve_gemm_accumulation(self):
        gap = copy.deepcopy(self.gap_head).half().eval()
        head = SpatialMILHead(gap).eval()
        features = torch.randn(3, 768, 24, 24).half()
        with torch.no_grad():
            expected = gap(features)
            actual = head(features)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_class_attention_changes_only_its_own_pooled_logit(self):
        features = torch.randn(2, 768, 3, 4)
        head = self.make_head()
        with torch.no_grad():
            baseline = head(features)
            head.attention.weight[0, 0, 0, 0] = 5
            changed, maps = head.forward_with_attention(features)
        torch.testing.assert_close(changed[:, 1:], baseline[:, 1:], rtol=0, atol=0)
        self.assertGreater((changed[:, 0] - baseline[:, 0]).abs().max().item(), 1e-4)
        self.assertGreater(maps[:, 0].std().item(), 1e-3)
        torch.testing.assert_close(maps[:, 1:], torch.full_like(maps[:, 1:], 1 / 12))

    def test_zero_initial_attention_receives_gradient_and_trunk_remains_differentiable(self):
        head = self.make_head()
        features = torch.randn(2, 768, 3, 4, requires_grad=True)
        head(features).square().mean().backward()
        for gradient in (head.attention.weight.grad, head.norm.weight.grad,
                         head.fc.weight.grad, features.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_real_timm_model_runs_backbone_once_for_logits_and_heatmaps(self):
        calls = []
        handle = self.model.stem.register_forward_hook(lambda *_: calls.append(1))
        images = torch.randn(1, 3, 64, 64)
        try:
            with torch.no_grad():
                logits, maps = forward_attention(self.model, images)
            self.assertEqual(len(calls), 1)
            self.assertEqual(logits.shape, (1, 10))
            self.assertEqual(maps.shape, (1, 10, 2, 2))
            calls.clear()
            with torch.no_grad():
                ordinary = self.model(images)
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(logits, ordinary, rtol=0, atol=0)
        finally:
            handle.remove()

    def test_benchmark_restores_training_modes_and_reports_only_head_scope(self):
        head = self.make_head().train()
        gap = copy.deepcopy(self.gap_head).eval()
        head.norm.eval()
        states = {id(module): module.training for item in (gap, head) for module in item.modules()}
        result = benchmark_spatial_mil(gap, head, torch.randn(1, 768, 2, 2), warmup=1, repeats=3)
        self.assertIn("head_only", result["scope"])
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["additional_parameters"], 7680)
        self.assertGreater(result["mil_ms"]["p50"], 0)
        self.assertEqual(states, {id(module): module.training for item in (gap, head)
                                  for module in item.modules()})

    def test_rejects_incompatible_heads_and_feature_shapes(self):
        gap = copy.deepcopy(self.gap_head)
        gap.norm = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "post-pooling"):
            SpatialMILHead(gap)
        with self.assertRaisesRegex(ValueError, "NCHW"):
            self.make_head()(torch.randn(2, 10, 3, 3))
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_spatial_mil(self.gap_head, self.make_head(), torch.randn(1, 768, 2, 2), repeats=0)


if __name__ == "__main__":
    unittest.main()
