"""Independent CPU checks of complete-Study FC objectives and train-only fitting."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import frozen_head_fit as fit


class FrozenHeadObjectiveTests(unittest.TestCase):
    def test_complete_study_probability_bce_matches_float64_formula_and_gradient(self):
        logits = torch.tensor([[[-2., 1.] * 5, [1., -1.] * 5]], requires_grad=True)
        mask = torch.ones((1, 2), dtype=torch.bool)
        targets, weights = torch.tensor([[1., 0.] * 5]), torch.arange(1., 11.)
        config = replace(fit.FitRecipe(), anchor_weight=0)
        total, hard, _, _, _ = fit.objective(logits, logits.detach(), mask, targets, weights, config)
        p = logits.detach().double().sigmoid()
        mean = p.mean(1)
        expected = -(weights.double() * targets.double() * mean.log()
                     + (1 - targets.double()) * (1 - mean).log()).mean()
        self.assertAlmostEqual(float(hard), float(expected), places=6)
        total.backward()
        analytic = (((1 - targets.double()) / (1 - mean)
                     - weights.double() * targets.double() / mean)[:, None] * p * (1 - p) / 2 / 10)
        torch.testing.assert_close(logits.grad.double(), analytic, atol=1e-7, rtol=2e-6)
        image_bce = F.binary_cross_entropy_with_logits(logits.detach(), targets[:, None].expand_as(logits),
                                                      pos_weight=weights)
        self.assertGreater(abs(float(hard) - float(image_bce)), .1)

    def test_padding_is_excluded_and_has_zero_gradient(self):
        logits = torch.tensor([[[.3] * 10, [700.] * 10]], requires_grad=True)
        original = torch.zeros_like(logits)
        mask, targets = torch.tensor([[True, False]]), torch.ones((1, 10))
        total, hard, _, anchor, _ = fit.objective(logits, original, mask, targets, torch.ones(10), fit.FitRecipe())
        self.assertAlmostEqual(float(hard), float(F.softplus(torch.tensor(-.3))), places=6)
        self.assertAlmostEqual(float(anchor), .09, places=6)
        total.backward()
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])))

    def test_rank_rewards_positive_negative_order_and_is_class_balanced(self):
        labels = torch.tensor([[1.] * 10, [0.] * 10])
        good = torch.tensor([[[2.] * 10], [[-2.] * 10]], requires_grad=True)
        mask = torch.ones((2, 1), dtype=torch.bool)
        config = replace(fit.FitRecipe(), bce_weight=.5, rank_weight=.5, anchor_weight=0)
        _, _, correct, _, count = fit.objective(good, good.detach(), mask, labels, torch.ones(10), config)
        _, _, reversed_rank, _, _ = fit.objective(-good, -good.detach(), mask, labels, torch.ones(10), config)
        self.assertEqual(count, 10)
        self.assertAlmostEqual(float(correct), float(F.softplus(torch.tensor(-4.))), places=6)
        self.assertLess(float(correct), float(reversed_rank))
        correct.backward()
        self.assertTrue(bool((good.grad[0] < 0).all()))
        self.assertTrue(bool((good.grad[1] > 0).all()))

    def test_study_anchor_gives_equal_weight_to_unequal_view_counts(self):
        logits = torch.tensor([[[1.] * 10, [100.] * 10], [[2.] * 10, [2.] * 10]])
        mask = torch.tensor([[True, False], [True, True]])
        _, _, _, anchor, _ = fit.objective(logits, torch.zeros_like(logits), mask,
            torch.ones((2, 10)), torch.ones(10), fit.FitRecipe())
        self.assertAlmostEqual(float(anchor), 2.5, places=6)

    def test_lr_has_registered_warmup_and_exact_final_minimum(self):
        config = fit.FitRecipe()
        self.assertAlmostEqual(fit.learning_rate_at(config, 0, 10), 2e-5)
        self.assertAlmostEqual(fit.learning_rate_at(config, 9, 10), 2e-4)
        self.assertAlmostEqual(fit.learning_rate_at(config, 119, 10), 1e-5)


class TrainOnlyCacheTests(unittest.TestCase):
    def fixture(self, root):
        cache = root / "cache"
        (cache / "train").mkdir(parents=True)
        features = np.ones((6, 2, 3), dtype=np.float16)
        features[2:4] = -1
        features[5] = -1
        np.save(cache / "train/A_features.npy", features)
        groups = ["s0", "s0", "s1", "s1", "s2", "s3"]
        labels = {"s0": [1] * 10, "s1": [0] * 10, "s2": [1] * 10, "s3": [0] * 10}
        with (cache / "train/records.jsonl").open("w") as stream:
            for i, group in enumerate(groups):
                stream.write(json.dumps({"image_id": "i" + str(i), "study_key": group,
                    "subject_id": "patient" + group, "targets": labels[group]}) + "\n")
        torch.save({"head.fc.weight": torch.zeros((10, 3)), "head.fc.bias": torch.zeros(10)}, cache / "A_head_fp32.pt")
        encoder = {"feature_dim": 3, "base_model_dir": "/immutable/A",
            "base_config_sha256": "a" * 64, "base_weights_sha256": "b" * 64,
            "head_file": "A_head_fp32.pt", "head_sha256": fit.sha256_file(cache / "A_head_fp32.pt")}
        meta = {"schema": "frozen_fc_cache_v1", "encoders": {"A": encoder}, "splits": {
            "train": {"records_file": "train/records.jsonl", "records_sha256": fit.sha256_file(cache / "train/records.jsonl"),
                "manifest_sha256": "c" * 64,
                "features": {"A": {"file": "train/A_features.npy", "sha256": fit.sha256_file(cache / "train/A_features.npy"),
                                     "shape": list(features.shape)}}},
            "valid": {"features": {"A": {"file": "valid/THIS_MUST_NEVER_BE_OPENED.npy"}}}}}
        (cache / "meta.json").write_text(json.dumps(meta))
        plan = {"plan_id": "synthetic_train_only_fixture", "test_used_for_design_or_fitting": False,
            "registered_before_new_head_validation": True, "encoders": {"A": encoder},
            "cache": {"train_manifest_sha256": "c" * 64},
            "common": {"epochs": 2, "warmup_epochs": 1, "batch_studies": 4,
                       "learning_rate": .01, "min_learning_rate": .001, "num_threads": 1},
            "recipes": {"F2": {"seed": 49, "bce_weight": .5, "rank_weight": .5, "anchor_weight": .1}}}
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return cache, plan_path

    def test_small_fit_reads_only_train_and_emits_exact_two_tensor_cpu_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, plan = self.fixture(root)
            with redirect_stdout(io.StringIO()):
                result = fit.fit(cache=cache, plan=plan, plan_sha256=fit.sha256_file(plan),
                    encoder="A", recipe="F2", output=root / "head")
            self.assertEqual(result["state"], "completed")
            self.assertFalse(result["validation_arrays_opened"])
            self.assertFalse(result["GPU_used"])
            self.assertEqual(result["num_studies"], 4)
            self.assertEqual(result["completed_epochs"], 2)
            state = torch.load(root / "head/patch_fp16.pt", weights_only=True)
            self.assertEqual(set(state), set(fit.HEAD_KEYS))
            self.assertTrue(all(value.device.type == "cpu" and value.dtype == torch.float16 for value in state.values()))
            self.assertGreater(float(state["head.fc.weight"].mean()), 0)
            history = [json.loads(line) for line in (root / "head/history.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 2)
            self.assertTrue(all(row["valid_evaluated"] is False for row in history))
            with self.assertRaisesRegex(ValueError, "fresh"):
                fit.fit(cache=cache, plan=plan, plan_sha256=fit.sha256_file(plan),
                        encoder="A", recipe="F2", output=root / "head")

    def test_conflicting_labels_tampered_features_and_wrong_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, plan = self.fixture(root)
            original = _meta = json.loads((cache / "meta.json").read_text())
            original["splits"]["train"]["features"]["A"]["sha256"] = "f" * 64
            (cache / "meta.json").write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "cache file changed"):
                fit.load_train_cache(cache, "A")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, plan = self.fixture(root)
            document = json.loads(plan.read_text())
            document["encoders"]["A"]["base_weights_sha256"] = "e" * 64
            plan.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "registered source"):
                fit.fit(cache=cache, plan=plan, plan_sha256=fit.sha256_file(plan),
                        encoder="A", recipe="F2", output=root / "head")

    def test_preview_does_not_open_feature_cache_or_train(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plan = self.fixture(root)
            with patch.object(fit, "load_train_cache", side_effect=AssertionError("Must not open cache")), redirect_stdout(io.StringIO()):
                self.assertEqual(fit.main(["--cache", "/not-present", "--plan", str(plan),
                    "--plan-sha256", fit.sha256_file(plan), "--encoder", "A", "--recipe", "F2",
                    "--output", str(root / "never-created")]), 0)
            self.assertFalse((root / "never-created").exists())

    def test_incomplete_cache_and_explicit_nontrain_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, _ = self.fixture(root)
            metadata = json.loads((cache / "meta.json").read_text())
            metadata["complete"] = False
            (cache / "meta.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "not finalized"):
                fit.load_train_cache(cache, "A")
            metadata["complete"] = True
            rows_path = cache / "train/records.jsonl"
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            rows[0]["split"] = "valid"
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata["splits"]["train"]["records_sha256"] = fit.sha256_file(rows_path)
            (cache / "meta.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Only original train"):
                fit.load_train_cache(cache, "A")


if __name__ == "__main__":
    unittest.main()
