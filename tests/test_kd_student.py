"""只验证张量公式与元数据，不构建网络、调用 backward 或更新参数。"""
import math
import copy
from dataclasses import asdict
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_distill import (DistillConfig, alpha_at, distillation_losses,
                           validate_student_resume, reconcile_student_checkpoints)
from kd_common import (BASE_BACKBONE, TINY_BACKBONE, student_preprocessing,
                       text_sha256, validate_pretrained_provenance)


class StudentMathTests(unittest.TestCase):
    def test_independent_bernoulli_kl_matches_scalar_reference(self):
        s = torch.linspace(-4, 4, 20).reshape(2, 10)
        t = torch.linspace(3, -2, 20).reshape(2, 10)
        y = (torch.arange(20).reshape(2, 10) % 2).float()
        hard, soft = distillation_losses(s, y, t, torch.ones(2, 10), torch.ones(10))
        p, q = torch.sigmoid(s / 2).double().numpy(), torch.sigmoid(t / 2).double().numpy()
        expected = (q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))).mean(1) * 4
        np.testing.assert_allclose(soft.numpy(), expected, atol=2e-7)
        torch.testing.assert_close(hard, F.binary_cross_entropy_with_logits(s, y, reduction="none").mean(1))

    def test_identical_scores_zero_kl_even_with_many_simultaneous_positives(self):
        logits = torch.full((2, 10), 4.)
        _, soft = distillation_losses(logits, torch.ones(2, 10), logits, torch.ones(2, 10), torch.ones(10))
        torch.testing.assert_close(soft, torch.zeros(2))
        self.assertGreater(float(torch.sigmoid(logits).sum(1)[0]), 9)

    def test_weighted_logit_alignment_uses_both_teacher_and_student_prior(self):
        w_t, w_s = torch.full((2, 10), 4.), torch.full((10,), 2.)
        s = torch.full((2, 10), math.log(2.))
        t = torch.full((2, 10), math.log(4.))
        _, aligned = distillation_losses(s, torch.zeros(2, 10), t, w_t, w_s)
        _, unaligned = distillation_losses(s, torch.zeros(2, 10), t, w_t, w_s, align=False)
        torch.testing.assert_close(aligned, torch.zeros(2), atol=1e-7, rtol=0)
        self.assertTrue(bool((unaligned > 0).all()))

    def test_extreme_logits_remain_finite(self):
        logits = torch.full((2, 10), 1000.)
        hard, soft = distillation_losses(logits, torch.zeros_like(logits), -logits,
                                         torch.ones_like(logits), torch.ones(10))
        self.assertTrue(bool(torch.isfinite(hard).all() & torch.isfinite(soft).all()))
        self.assertTrue(bool((soft > 0).all()))

    def test_bad_labels_weights_temperature_and_shape_rejected(self):
        z = torch.zeros(1, 10)
        with self.assertRaises(ValueError):
            distillation_losses(z, z, z, torch.zeros_like(z), torch.ones(10))
        with self.assertRaises(ValueError):
            distillation_losses(z, z, z, torch.ones_like(z), torch.ones(10), temperature=0)
        with self.assertRaises(ValueError):
            distillation_losses(z, z + 2, z, torch.ones_like(z), torch.ones(10))
        with self.assertRaises(FloatingPointError):
            distillation_losses(z * float("nan"), z, z, torch.ones_like(z), torch.ones(10))
        with self.assertRaises(ValueError):
            distillation_losses(z[:, :9], z, z, torch.ones_like(z), torch.ones(10))

    def test_alpha_warmup_and_no_kd_baseline(self):
        c = DistillConfig(Path("fold.json"), Path("soft"), Path("out"), Path("images"))
        self.assertAlmostEqual(alpha_at(c, 1, 0, 10), .01)
        self.assertAlmostEqual(alpha_at(c, 1, 9, 10), .1)
        self.assertAlmostEqual(alpha_at(c, 2, 9, 10), .2)
        self.assertAlmostEqual(alpha_at(c, 3, 0, 10), .2)
        zero = DistillConfig(Path("f"), Path("s"), Path("o"), Path("i"), alpha=0)
        self.assertEqual(alpha_at(zero, 10, 0, 10), 0)

    def test_student_preprocessing_is_fixed_512(self):
        p = student_preprocessing()
        self.assertEqual(p["training_scales"], [512])
        self.assertEqual(p["validation_tta"], "identity")


class SourceTests(unittest.TestCase):
    def provenance(self):
        revision = "a" * 40
        materials = {
            "model_card": "---\nlicense: apache-2.0\n---\nImageNet-22k and ImageNet-1k",
            "apache_license": "Apache License\nVersion 2.0",
            "upstream_license": "MIT License\nPermission is hereby granted",
        }
        return {"backbone_name": TINY_BACKBONE, "repo_id": f"timm/{TINY_BACKBONE}",
                "license": "apache-2.0", "pretrain_dataset": "ImageNet-22k", "finetune_dataset": "ImageNet-1k",
                "weights_sha256": "b" * 64, "revision": revision,
                "weights_url": f"https://huggingface.co/timm/{TINY_BACKBONE}/resolve/{revision}/model.safetensors",
                "materials": {k: {"text": v, "sha256": text_sha256(v)} for k, v in materials.items()}}

    def test_valid_provenance_and_tampering(self):
        p = self.provenance()
        validate_pretrained_provenance(p, TINY_BACKBONE)
        with self.assertRaises(ValueError):
            validate_pretrained_provenance(p, BASE_BACKBONE)
        p["materials"]["model_card"]["text"] += " changed"
        with self.assertRaises(ValueError):
            validate_pretrained_provenance(p, TINY_BACKBONE)

    def test_nc_and_unpinned_sources_rejected(self):
        p = self.provenance()
        p["license"] = "cc-by-nc-4.0"
        with self.assertRaises(ValueError):
            validate_pretrained_provenance(p, TINY_BACKBONE)
        p = self.provenance()
        p["revision"] = "main"
        with self.assertRaises(ValueError):
            validate_pretrained_provenance(p, TINY_BACKBONE)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = DistillConfig(Path("f"), Path("s"), Path("o"), Path("i"), validation_tta="identity")
        serialized = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.config).items()}
        self.expected = {"schema_version": 7, "artifact_type": "kd_student"}
        self.last = {**self.expected, "training_config": serialized, "epoch": 2, "best_epoch": 1,
            "best_auc": .8, "model_state_dict": {}, "raw_model_state_dict": {},
            "optimizer_state_dict": {}, "scaler_state_dict": {}, "rng_state": {},
            "progress": {"completed_epochs": 2, "validated_epochs": 2, "optimizer_updates": 20, "ema_updates": 20},
            "early_stopping": {"reference_auc": .8, "bad_epochs": 1},
            "inference_ready": True, "validation_complete": True, "weights_source": "ema",
            "pretrained_provenance": SourceTests().provenance()}
        self.best = copy.deepcopy(self.last)
        self.best.update(epoch=1, best_epoch=1, validation={"macro_auc": .8})
        self.best["progress"].update(completed_epochs=1, validated_epochs=1, optimizer_updates=10, ema_updates=10)
        self.best["early_stopping"]["bad_epochs"] = 0

    def test_normal_resume_retains_last(self):
        self.assertIs(reconcile_student_checkpoints(self.last, self.best, self.config, self.expected), self.last)

    def test_new_best_commit_is_promoted_after_interruption(self):
        newer = copy.deepcopy(self.last)
        newer.update(epoch=3, best_epoch=3, best_auc=.81, validation={"macro_auc": .81})
        newer["progress"].update(completed_epochs=3, validated_epochs=3, optimizer_updates=30, ema_updates=30)
        newer["early_stopping"] = {"reference_auc": .81, "bad_epochs": 0}
        self.assertIs(reconcile_student_checkpoints(self.last, newer, self.config, self.expected), newer)

    def test_first_best_is_a_complete_resume_source(self):
        self.assertIs(reconcile_student_checkpoints(self.best, self.best, self.config, self.expected), self.best)

    def test_progress_and_missing_best_bindings_are_rejected(self):
        bad = copy.deepcopy(self.last)
        bad["progress"]["ema_updates"] = 19
        with self.assertRaises(ValueError):
            validate_student_resume(bad, self.config, self.expected)
        bad = copy.deepcopy(self.best)
        bad["best_auc"] = .7
        bad["validation"]["macro_auc"] = .7
        with self.assertRaises(ValueError):
            reconcile_student_checkpoints(self.last, bad, self.config, self.expected)


if __name__ == "__main__":
    unittest.main()
