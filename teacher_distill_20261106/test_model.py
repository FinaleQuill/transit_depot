"""CPU checks for complete views, gradients, public loading and phase changes."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from teacher_distill_20261106.model import TeacherStudyModel, study_bce_loss


class TokenEncoder(nn.Module):
    num_features = 8
    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 16, stride=16)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.reg_token = nn.Parameter(torch.randn(1, 4, 8))
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
        self.norm = nn.LayerNorm(8)
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable):
        self.grad_checkpointing = enable

    def forward_features(self, pixels):
        patches = self.patch_embed(pixels).flatten(2).transpose(1, 2)
        for block in self.blocks:
            if self.grad_checkpointing and torch.is_grad_enabled():
                patches = torch.utils.checkpoint.checkpoint(block, patches, use_reentrant=False)
            else:
                patches = block(patches)
        cls = self.cls_token.expand(len(pixels), -1, -1) + patches.mean(1, keepdim=True)
        return self.norm(torch.cat((cls,
            self.reg_token.expand(len(pixels), -1, -1), patches), dim=1))


class ConvStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(8, 8, 1) for _ in range(2)])
        self.downsample = nn.Conv2d(8, 8, 1)

    def forward(self, pixels):
        pixels = self.downsample(pixels)
        for block in self.blocks:
            pixels = torch.nn.functional.gelu(block(pixels))
        return pixels


class ConvEncoder(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=4, padding=1)
        self.stages = nn.Sequential(ConvStage(), ConvStage())
        self.head = nn.Module()
        self.head.norm = nn.LayerNorm(8)
        self.head.fc = nn.Identity()
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable):
        self.grad_checkpointing = enable

    def forward_features(self, pixels):
        return self.stages(self.stem(pixels))

    def forward_head(self, spatial, pre_logits=False):
        pooled = self.head.norm(spatial.mean((-2, -1)))
        return pooled if pre_logits else self.head.fc(pooled)


class TeacherStudyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261106)
        torch.set_num_threads(1)

    def model(self, family="dinov3_l", **kwargs):
        encoder = TokenEncoder() if family == "dinov3_l" else ConvEncoder()
        return TeacherStudyModel.from_backbone(encoder, family=family, **kwargs)

    def studies(self):
        return [torch.randn(v, 3, 32, 32) for v in (1, 5, 2)]

    def test_public_construction_is_offline_and_uses_registered_topologies(self):
        for family, encoder in (("dinov3_l", TokenEncoder()),
                                ("convnext_l", ConvEncoder())):
            calls = []

            def create_model(name, **options):
                calls.append((name, options))
                return encoder

            with patch.dict(sys.modules, timm=SimpleNamespace(create_model=create_model)):
                model = TeacherStudyModel(family)
            name, options = calls[0]
            self.assertIs(options["pretrained"], False)
            self.assertEqual(options["num_classes"], 0)
            self.assertTrue(model.backbone.grad_checkpointing)
            if family == "dinov3_l":
                self.assertEqual(name, "vit_large_patch16_dinov3.lvd1689m")
                self.assertTrue(options["dynamic_img_size"])
            else:
                self.assertEqual(name, "convnext_large.fb_in22k_ft_in1k_384")

    def test_all_views_probability_mean_and_no_cross_study_mixing(self):
        for family in ("dinov3_l", "convnext_l"):
            with self.subTest(family=family):
                model = self.model(family, view_chunk_size=2).eval()
                studies = self.studies()
                outputs = model.forward_studies(studies, return_components=True)
                for images, output in zip(studies, outputs):
                    torch.testing.assert_close(output["study_logits"].sigmoid(),
                        model.image_logits(images).sigmoid().mean(0))
                    torch.testing.assert_close(output["study_logits"], model(images.flip(0)))
                    self.assertEqual(output["image_logits"].shape, (len(images), 10))
                changed = model.forward_studies([studies[0], studies[1] + 9, studies[2]])
                torch.testing.assert_close(changed[0], outputs[0]["study_logits"], rtol=0, atol=0)
                torch.testing.assert_close(changed[2], outputs[2]["study_logits"], rtol=0, atol=0)

    def test_dino_excludes_registers_and_every_view_gets_gradient(self):
        model = self.model(view_chunk_size=2).eval()
        images = torch.randn(7, 3, 32, 32, requires_grad=True)
        before = model.image_logits(images)
        with torch.no_grad():
            model.backbone.reg_token.fill_(10000.)
        torch.testing.assert_close(before, model.image_logits(images), rtol=0, atol=0)
        study_bce_loss(model(images), torch.randint(0, 2, (10,)).float()).backward()
        self.assertTrue(bool((images.grad.abs().flatten(1).sum(1) > 0).all()))

    def test_nested_checkpoint_gradients_match_and_training_toggle_is_real(self):
        for family in ("dinov3_l", "convnext_l"):
            with self.subTest(family=family):
                plain = self.model(family, view_chunk_size=2, drop_rate=.2).train()
                recomputed = deepcopy(plain)
                recomputed.gradient_checkpointing = True
                recomputed.block_checkpointing = True
                recomputed.train()
                self.assertTrue(recomputed.backbone.grad_checkpointing)
                studies = self.studies()
                labels = torch.randint(0, 2, (3, 10)).float()
                for model in (plain, recomputed):
                    torch.manual_seed(818)
                    logits = torch.stack(model.forward_studies(studies))
                    nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
                for (name, a), (_, b) in zip(plain.named_parameters(), recomputed.named_parameters()):
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
                recomputed.eval()
                self.assertFalse(recomputed.backbone.grad_checkpointing)
                recomputed.train()
                self.assertTrue(recomputed.backbone.grad_checkpointing)

    def test_public_state_is_strict_and_only_original_cnn_head_is_discarded(self):
        for family in ("dinov3_l", "convnext_l"):
            model = self.model(family)
            state = deepcopy(model.backbone.state_dict())
            model.load_public_state(state)
            missing = dict(state)
            missing.pop(next(iter(missing)))
            with self.assertRaises(RuntimeError):
                model.load_public_state(missing)
            with self.assertRaises(RuntimeError):
                model.load_public_state(dict(state, unwanted=torch.ones(1)))
            with self.assertRaises(TypeError):
                model.load_public_state({"model": state})
        cnn = self.model("convnext_l")
        source = deepcopy(cnn.backbone.state_dict())
        classifier = {"head.fc.weight": torch.randn(1000, 8), "head.fc.bias": torch.randn(1000)}
        cnn.load_public_state(dict(source, **classifier))
        for head in ({"head.fc.weight": classifier["head.fc.weight"]},
                     {"head.fc.weight": torch.randn(10, 8), "head.fc.bias": torch.zeros(10)}):
            with self.assertRaises(ValueError):
                cnn.load_public_state(dict(source, **head))
        dino = self.model()
        with self.assertRaises(RuntimeError):
            dino.load_public_state(dict(dino.backbone.state_dict(), **{
                "head.weight": torch.zeros(1000, 8), "head.bias": torch.zeros(1000)}))

    def test_local_initialization_rejects_non_safetensors_without_import_or_download(self):
        model = self.model()
        with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
            with self.assertRaises(ValueError):
                model.load_public_pretrained(stream.name)
        with self.assertRaises(FileNotFoundError):
            model.load_public_pretrained("/nonexistent/public.safetensors")

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors not installed")
    def test_actual_local_safetensors_load_preserves_all_public_parameters(self):
        from safetensors.torch import save_file

        for family in ("dinov3_l", "convnext_l"):
            source = self.model(family)
            state = source.backbone.state_dict()
            if family == "convnext_l":
                state.update({"head.fc.weight": torch.randn(1000, 8),
                              "head.fc.bias": torch.randn(1000)})
            with tempfile.NamedTemporaryFile(suffix=".safetensors") as stream:
                save_file(state, stream.name)
                target = self.model(family)
                target.load_public_pretrained(stream.name)
            for name, expected in source.backbone.state_dict().items():
                torch.testing.assert_close(target.backbone.state_dict()[name], expected, rtol=0, atol=0)

    def test_layer_lr_roles_cover_all_parameters_once_and_keep_public_norm(self):
        for family in ("dinov3_l", "convnext_l"):
            model = self.model(family)
            groups = model.optimizer_parameter_groups(backbone_lr=1e-5,
                head_lr=2e-4, layer_decay=.8, weight_decay=.05)
            params = [p for group in groups for p in group["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
            named = {name: group for group in groups for name in group["param_names"]}
            earliest = "backbone.patch_embed.weight" if family == "dinov3_l" else "backbone.stem.weight"
            last = "backbone.norm.weight" if family == "dinov3_l" else "backbone.head.norm.weight"
            self.assertLess(named[earliest]["lr"], named[last]["lr"])
            self.assertEqual(named[last]["role"], "backbone")
            self.assertEqual(named[last]["weight_decay"], 0.)
            self.assertEqual(named["head." + str(len(model.head) - 1) + ".weight"]["lr"], 2e-4)
            if family == "dinov3_l":
                self.assertEqual(named["backbone.cls_token"]["weight_decay"], 0.)

    def test_phase_changes_preserve_parameters_and_optimizer_references(self):
        model = self.model(image_size=512)
        identities = {name: id(p) for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(backbone_lr=1e-5, head_lr=2e-4))
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        model.set_image_size(768)
        self.assertEqual(identities, {name: id(p) for name, p in model.named_parameters()})
        self.assertEqual(parameter_ids, {id(p) for p in model.parameters()})
        model.set_image_size(32)
        self.assertEqual(model(torch.randn(1, 3, 32, 32)).shape, (10,))
        model.set_image_size(48)
        self.assertEqual(model(torch.randn(1, 3, 48, 48)).shape, (10,))
        with self.assertRaises(ValueError):
            model.set_image_size(30)

    def test_real_fp16_evaluation_and_full_state_round_trip(self):
        # torch 2.2 CPU lacks the Half LayerNorm kernel.  Use fp32 arithmetic
        # solely for CPU normalization in this check (as GPU kernels also
        # accumulate normalization in fp32); all weights, inputs and linear/
        # convolution operations remain fp16.  Real CUDA is the runner smoke.
        original_norm = nn.functional.layer_norm

        def cpu_half_norm(value, normalized_shape, weight=None, bias=None, eps=1e-5):
            if value.device.type == "cpu" and value.dtype == torch.float16:
                return original_norm(value.float(), normalized_shape,
                    weight.float() if weight is not None else None,
                    bias.float() if bias is not None else None, eps).to(value.dtype)
            return original_norm(value, normalized_shape, weight, bias, eps)

        for family in ("dinov3_l", "convnext_l"):
            with self.subTest(family=family), patch("torch.nn.functional.layer_norm", cpu_half_norm):
                model = self.model(family, view_chunk_size=2).half().eval()
                images = torch.randn(3, 3, 32, 32).half()
                with torch.inference_mode():
                    expected = model(images)
                self.assertEqual(expected.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(expected).all()))
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = self.model(family, view_chunk_size=2).half().eval()
                restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
                with torch.inference_mode():
                    torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_invalid_views_and_parameter_options_fail_explicitly(self):
        model = self.model()
        for studies in ([], [torch.empty(0, 3, 32, 32)],
                        [torch.randn(1, 3, 32, 32), torch.randn(1, 3, 48, 48)]):
            with self.assertRaises(ValueError):
                model.forward_studies(studies)
        with self.assertRaises(ValueError):
            model.image_logits(torch.randn(1, 3, 31, 32))
        for options in ({"layer_decay": 0}, {"head_lr": 0}, {"weight_decay": -1}):
            values = dict(backbone_lr=1e-5, head_lr=2e-4)
            values.update(options)
            with self.assertRaises(ValueError):
                model.optimizer_parameter_groups(**values)


if __name__ == "__main__":
    unittest.main()
