# 【学习导航】历史Tiny热重启的协议检查，与当前768主线分开阅读。
# 本次只打包与静态解析，没有执行此测试文件。
"""CPU-only checks for checkpoint warm restart; no training or model inference.

Run: python -m unittest discover -s tests -p 'test_tiny_warm_restart.py' -v
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from inference import load_checkpoint_bundle
from train import EarlyStopping
from train_tiny import (
    TinyConfig, prepare_warm_restart, preprocessing_config,
    restore_warm_restart_weights, seed_resume_baseline, set_learning_rates,
)


class WarmRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="tiny-warm-restart-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.pt"
        self.hashes = {"train": "a" * 64, "valid": "b" * 64}
        self.config = TinyConfig(
            output=self.root / "new_run", resume=self.source,
            epochs=10, num_workers=4, prefetch_factor=1,
        )
        self.checkpoint = {
            "schema_version": 4, "model_name": "convnext_tiny_gap_linear",
            "num_classes": 10, "class_ids": list(range(1, 11)), "epoch": 5,
            "aggregation": "study_mean_probability",
            "preprocessing": preprocessing_config(),
            "model_config": self.config.model_config(),
            "validation_complete": True, "inference_ready": True,
            "weights_source": "ema", "validation_weights_source": "ema",
            "threshold_weights_source": "ema", "inference_precision": "bfloat16",
            "train_manifest_sha256": self.hashes["train"],
            "valid_manifest_sha256": self.hashes["valid"],
            "progress": {
                "completed_epochs": 5, "validated_epochs": 5,
                "backward_batches": 9480, "optimizer_updates": 4740,
                "ema_updates": 4740, "amp_skipped_steps": 0,
            },
            "ema": {
                "enabled": True, "decay": 0.999, "use_warmup": True,
                "warmup_power": 0.75, "buffers": "copy_from_raw", "updates": 4740,
            },
            # Different sentinel values detect accidental EMA/raw weight swaps.
            "raw_model_state_dict": {
                "weight": torch.full((2, 3), 2.0), "bias": torch.full((2,), 3.0),
            },
            "model_state_dict": {
                "weight": torch.full((2, 3), 7.0), "bias": torch.full((2,), 11.0),
            },
            "validation": {"macro_auc": 0.7999705530021188, "macro_f1": 0.4694609371},
            "thresholds": [0.05 + index * 0.04 for index in range(10)],
            "training_config": {
                "backbone_lr": 5e-5, "head_lr": 5e-4, "pos_weight_power": 0.0,
            },
            "loss_config": {"name": "BCEWithLogitsLoss", "pos_weight": None},
        }
        self.save_source()
        # Fail immediately if a future refactor accidentally runs the model,
        # enters training, or invokes autograd during these metadata tests.
        for target in (
            "torch.nn.Module._call_impl", "torch.Tensor.backward", "torch.autograd.backward",
            "train_tiny.run_training", "train_tiny.train_epoch", "train_tiny.evaluate_tiny",
            "train_tiny.build_tiny_model",
        ):
            guard = patch(target, side_effect=AssertionError(f"Forbidden test operation: {target}"))
            guard.start()
            self.addCleanup(guard.stop)

    def save_source(self, checkpoint: dict | None = None) -> None:
        torch.save(self.checkpoint if checkpoint is None else checkpoint, self.source)

    def prepare(self, config: TinyConfig | None = None):
        return prepare_warm_restart(self.config if config is None else config, self.hashes)

    def test_warm_restart_uses_epoch_six_and_reduces_source_learning_rates(self) -> None:
        # CLI LR overrides must not accidentally restore the original high rates.
        supplied = replace(self.config, backbone_lr=1e-2, head_lr=1e-1)
        prepared, checkpoint, origin = self.prepare(supplied)
        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(origin["start_epoch"], 6)
        self.assertEqual(origin["schedule_epochs"], 5)
        self.assertAlmostEqual(prepared.backbone_lr, 5e-6)
        self.assertAlmostEqual(prepared.head_lr, 5e-5)
        self.assertEqual(prepared.warmup_epochs, 0)
        self.assertEqual(prepared.patience, 3)
        self.assertEqual(prepared.amp_dtype, "bfloat16")
        self.assertEqual(checkpoint["progress"]["ema_updates"], 4740)
        self.assertTrue(origin["optimizer_reset"])
        self.assertTrue(origin["scaler_reset"])
        self.assertFalse(origin["rng_restored"])
        self.assertFalse(origin["exact_resume"])
        # Metadata preparation must not create training outputs.
        self.assertFalse(self.config.output.exists())

    def test_patience_is_enabled_and_bounded_for_warm_restart(self) -> None:
        for supplied, expected in ((0, 3), (1, 1), (2, 2), (5, 3)):
            with self.subTest(supplied=supplied):
                prepared, _, _ = self.prepare(replace(self.config, patience=supplied))
                self.assertEqual(prepared.patience, expected)

    def test_missing_resume_keeps_fresh_training_configuration(self) -> None:
        fresh = replace(self.config, resume=None)
        self.assertEqual(prepare_warm_restart(fresh, self.hashes), (fresh, None, None))

    def test_rejects_changed_manifest_or_invalid_epoch(self) -> None:
        for split in ("train", "valid"):
            with self.subTest(split=split):
                changed = {**self.hashes, split: "c" * 64}
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    prepare_warm_restart(self.config, changed)
        for last_epoch in (4, 5):
            with self.subTest(last_epoch=last_epoch):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    self.prepare(replace(self.config, epochs=last_epoch))

    def test_rejects_missing_raw_weights_and_inconsistent_ema_counters(self) -> None:
        broken = copy.deepcopy(self.checkpoint)
        del broken["raw_model_state_dict"]
        self.save_source(broken)
        with self.assertRaisesRegex(ValueError, "在线模型与EMA"):
            self.prepare()
        for section, key in (("progress", "ema_updates"), ("ema", "updates")):
            with self.subTest(section=section, key=key):
                broken = copy.deepcopy(self.checkpoint)
                broken[section][key] = 4739
                self.save_source(broken)
                with self.assertRaisesRegex(ValueError, "计数"):
                    self.prepare()

    def test_rejects_incomplete_model_metadata(self) -> None:
        broken = copy.deepcopy(self.checkpoint)
        del broken["model_config"]["drop_rate"]
        self.save_source(broken)
        with self.assertRaisesRegex(ValueError, "模型配置"):
            self.prepare()

    def test_rejects_checkpoint_replaced_during_loading(self) -> None:
        with patch("train_tiny._sha256", side_effect=["d" * 64, "e" * 64]):
            with self.assertRaisesRegex(RuntimeError, "发生变化"):
                self.prepare()

    def test_restores_raw_and_ema_separately_with_strict_state_loading(self) -> None:
        online = nn.Linear(3, 2, device="cpu")
        ema = SimpleNamespace(module=nn.Linear(3, 2, device="cpu"))
        restore_warm_restart_weights(online, ema, self.checkpoint)
        for key, value in online.state_dict().items():
            self.assertTrue(torch.equal(value, self.checkpoint["raw_model_state_dict"][key]))
            self.assertTrue(torch.equal(ema.module.state_dict()[key], self.checkpoint["model_state_dict"][key]))
            self.assertEqual(value.device.type, "cpu")
        # Missing keys and incompatible architectures must not be silently ignored.
        broken = copy.deepcopy(self.checkpoint)
        del broken["raw_model_state_dict"]["bias"]
        with self.assertRaises(RuntimeError):
            restore_warm_restart_weights(online, ema, broken)
        broken = copy.deepcopy(self.checkpoint)
        broken["model_state_dict"]["weight"] = torch.zeros(2, 4)
        with self.assertRaises(RuntimeError):
            restore_warm_restart_weights(online, ema, broken)

    def test_local_cosine_starts_at_peak_and_reaches_floor_without_old_updates(self) -> None:
        prepared, checkpoint, origin = self.prepare()
        # A minimal parameter-group container avoids creating an optimizer.
        optimizer = SimpleNamespace(param_groups=[
            {"name": "backbone_decay", "role": "backbone", "lr": 0.0},
            {"name": "head_decay", "role": "head", "lr": 0.0},
        ])
        windows = 948
        cumulative = checkpoint["progress"]["optimizer_updates"]
        local_updates = cumulative - origin["source_optimizer_updates"]
        self.assertEqual(local_updates, 0)
        initial = set_learning_rates(optimizer, prepared, local_updates, windows,
                                     schedule_epochs=origin["schedule_epochs"])
        self.assertAlmostEqual(initial["backbone_decay"], prepared.backbone_lr)
        self.assertAlmostEqual(initial["head_decay"], prepared.head_lr)
        final_update = origin["schedule_epochs"] * windows - 1
        final = set_learning_rates(optimizer, prepared, final_update, windows,
                                   schedule_epochs=origin["schedule_epochs"])
        self.assertAlmostEqual(final["backbone_decay"], prepared.backbone_lr * prepared.min_lr_ratio)
        self.assertAlmostEqual(final["head_decay"], prepared.head_lr * prepared.min_lr_ratio)
        middle = set_learning_rates(optimizer, prepared, final_update // 2, windows,
                                    schedule_epochs=origin["schedule_epochs"])
        self.assertLess(final["head_decay"], middle["head_decay"])
        self.assertLess(middle["head_decay"], initial["head_decay"])
        self.assertEqual(checkpoint["progress"]["optimizer_updates"], 4740)

    def test_baseline_bundle_retains_valid_threshold_and_hash_binding(self) -> None:
        _, checkpoint, origin = self.prepare()
        output = self.root / "baseline"
        seed_resume_baseline(output, checkpoint, origin)
        for directory in (output, output / "best_f1"):
            loaded, thresholds, digest = load_checkpoint_bundle(directory / "best.pt")
            self.assertEqual(loaded["epoch"], 5)
            self.assertEqual(thresholds.tolist(), checkpoint["thresholds"])
            self.assertEqual(digest, hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest())
            metrics = json.loads((directory / "best_metrics.json").read_text())
            self.assertTrue(metrics["inherited"])
            self.assertEqual(metrics["phase"], "inherited_resume_baseline")
            self.assertEqual(metrics["checkpoint_sha256"], digest)
        # A stale threshold sidecar must be rejected before any inference occurs.
        sidecar = output / "thresholds.json"
        metadata = json.loads(sidecar.read_text())
        metadata["checkpoint_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_checkpoint_bundle(output / "best.pt")

    def test_baseline_auc_survives_regression_and_three_bad_epochs_stop(self) -> None:
        prepared, checkpoint, origin = self.prepare()
        baseline = (origin["baseline_auc"], origin["baseline_f1"])
        early = EarlyStopping(
            patience=prepared.patience, min_delta=prepared.min_delta, min_epochs=1,
            best_auc=baseline[0], best_epoch=checkpoint["epoch"], bad_epochs=0,
        )
        stops = []
        for epoch, auc in enumerate((0.7973, 0.7980, 0.7990), start=origin["start_epoch"]):
            self.assertLess((auc, 0.99), baseline)
            stops.append(early.update(epoch, auc))
        self.assertEqual(stops, [False, False, True])
        self.assertEqual(early.best_epoch, 5)
        self.assertEqual(early.best_auc, baseline[0])
        self.assertEqual(early.bad_epochs, 3)


if __name__ == "__main__":
    unittest.main()
