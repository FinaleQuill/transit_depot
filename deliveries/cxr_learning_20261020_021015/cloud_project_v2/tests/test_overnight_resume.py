# 【学习导航】用CPU小张量元数据检查断点保存和best绑定；这里不是训练循环示例。
# 本次只打包与静态解析，没有执行此测试文件。
"""CPU checkpoint-metadata tests: no model, optimizer, CUDA, or image access."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model_overnight import MODEL_NAME
from train_overnight import (
    OvernightConfig, _jsonable, compute_loss_config, preprocessing_config, reconcile_resume_best,
    save_checkpoint, save_oof, sha256,
)


PLAN_HASH = "a" * 64
SOURCE_HASH = "b" * 64


class ResumeCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cxr_resume_metadata_")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = OvernightConfig(fold_plan=self.output / "folds.json", fold=0,
            output=self.output, image_root=self.output / "unused_readonly_images",
            resume=self.output / "last.pt")

    def checkpoint(self, epoch: int, auc: float, *, best_auc: float | None = None,
                   best_epoch: int | None = None) -> dict:
        """Tiny scalar tensors stand for state; never instantiate a model."""
        return {"schema_version": 5, "model_name": MODEL_NAME,
                "fold_index": 0, "fold_plan_sha256": PLAN_HASH,
                "source_manifest_sha256": SOURCE_HASH, "class_ids": list(range(1, 11)),
                "validation_complete": True, "model_config": self.config.model_config(),
                "preprocessing": preprocessing_config(), "inference_precision": "bfloat16",
                "training_config": _jsonable(asdict(self.config)),
                "loss_config": compute_loss_config([
                    {"image_id": f"image{index}", "split": "train", "training_eligible": True,
                     "subject_id": f"p{index}", "study_id": f"s{index}",
                     "study_key": f"p{index}/s{index}", "targets": [index] * 10}
                    for index in (0, 1)], source_manifest_sha256=SOURCE_HASH,
                    fold=0, fold_plan_sha256=PLAN_HASH),
                "epoch": epoch, "best_auc": auc if best_auc is None else best_auc,
                "best_epoch": epoch if best_epoch is None else best_epoch,
                "validation": {"macro_auc": auc}, "thresholds": [0.5] * 10,
                "model_state_dict": {"metadata_test": torch.tensor([float(epoch)])},
                "raw_model_state_dict": {"metadata_test": torch.tensor([float(epoch)])},
                "optimizer_state_dict": {"completed_epoch": epoch},
                "scaler_state_dict": {}, "rng_state": {"metadata_only": True},
                "early_stopping": {"reference_auc": auc if best_auc is None else best_auc,
                                   "bad_epochs": 0, "patience": 4, "min_delta": 1e-4},
                "progress": {"completed_epochs": epoch, "validated_epochs": epoch,
                             "optimizer_updates": epoch * 100, "ema_updates": epoch * 100},
                "validation_predictions": {
                    "study_keys": ["p1/s1", "p2/s2"], "subject_ids": ["p1", "p2"],
                    "study_ids": ["s1", "s2"],
                    "targets": torch.tensor([[0.] * 10, [1.] * 10]),
                    "probabilities": torch.tensor([[0.2] * 10, [0.8] * 10])}}

    def reconcile(self, checkpoint: dict, config: OvernightConfig | None = None):
        return reconcile_resume_best(checkpoint, config or self.config,
                                     PLAN_HASH, SOURCE_HASH, "bfloat16")

    def save_last(self, checkpoint: dict) -> None:
        save_checkpoint(self.output / "last.pt", checkpoint)

    def test_newer_committed_best_promotes_full_state_and_repairs_companions(self) -> None:
        last, best = self.checkpoint(2, 0.75), self.checkpoint(3, 0.81)
        self.save_last(last)
        digest = save_checkpoint(self.output / "best.pt", best)
        recovered, event = self.reconcile(last)
        self.assertTrue(event["promoted_newer_best_state"])
        self.assertEqual(recovered["progress"]["completed_epochs"], 3)
        self.assertEqual(recovered["optimizer_state_dict"]["completed_epoch"], 3)
        self.assertEqual(event["actual_checkpoint"], str(self.output / "best.pt"))
        self.assertEqual(event["actual_checkpoint_sha256"], digest)
        self.assertEqual(recovered["raw_model_state_dict"]["metadata_test"].item(), 3.)
        metrics = json.loads((self.output / "best_metrics.json").read_text())
        self.assertEqual(metrics["checkpoint_sha256"], digest)
        self.assertEqual(metrics["epoch"], 3)
        self.assertTrue(metrics["oof_complete"])
        with np.load(self.output / "best_oof.npz", allow_pickle=False) as oof:
            self.assertEqual(oof["checkpoint_sha256"].item(), digest)
            self.assertEqual(oof["epoch"].item(), 3)
        thresholds = json.loads((self.output / "thresholds.json").read_text())
        self.assertEqual(thresholds["checkpoint_sha256"], digest)

    def test_older_better_best_is_protected_without_rolling_back_optimizer(self) -> None:
        last = self.checkpoint(5, 0.72, best_auc=0.75, best_epoch=2)
        self.save_last(last)
        best_hash = save_checkpoint(self.output / "best.pt", self.checkpoint(3, 0.81))
        recovered, event = self.reconcile(last)
        self.assertFalse(event["promoted_newer_best_state"])
        self.assertEqual(recovered["epoch"], 5)
        self.assertEqual(recovered["optimizer_state_dict"]["completed_epoch"], 5)
        self.assertEqual(recovered["best_auc"], 0.81)
        self.assertEqual(recovered["best_epoch"], 3)
        self.assertEqual(recovered["early_stopping"]["reference_auc"], 0.81)
        self.assertEqual(sha256(self.output / "best.pt"), best_hash)

    def test_normal_resume_repairs_stale_metric_and_oof_bindings(self) -> None:
        last = self.checkpoint(5, 0.78, best_auc=0.82, best_epoch=4)
        self.save_last(last)
        best = self.checkpoint(4, 0.82)
        digest = save_checkpoint(self.output / "best.pt", best)
        (self.output / "best_metrics.json").write_text(json.dumps(
            {"epoch": 1, "checkpoint_sha256": "c" * 64, "training": {"stale": True}}))
        prediction = best["validation_predictions"]
        save_oof(self.output / "best_oof.npz", prediction, fold=0, epoch=1, checkpoint_sha256="c" * 64)
        recovered, event = self.reconcile(last)
        self.assertEqual(recovered["epoch"], 5)
        self.assertFalse(event["promoted_newer_best_state"])
        metrics = json.loads((self.output / "best_metrics.json").read_text())
        self.assertEqual(metrics["checkpoint_sha256"], digest)
        self.assertNotIn("training", metrics)
        with np.load(self.output / "best_oof.npz", allow_pickle=False) as oof:
            self.assertEqual(oof["epoch"].item(), 4)
            self.assertEqual(oof["checkpoint_sha256"].item(), digest)

    def test_last_that_is_itself_best_reconstructs_missing_best(self) -> None:
        last = self.checkpoint(1, 0.7)
        self.save_last(last)
        recovered, event = self.reconcile(last)
        self.assertEqual(recovered["epoch"], 1)
        self.assertTrue((self.output / "best.pt").is_file())
        self.assertTrue((self.output / "best_oof.npz").is_file())
        self.assertEqual(event["best_auc"], 0.7)

    def test_missing_historic_best_weights_are_not_fabricated(self) -> None:
        last = self.checkpoint(5, 0.7, best_auc=0.8, best_epoch=3)
        self.save_last(last)
        with self.assertRaisesRegex(RuntimeError, "Historic best weights"):
            self.reconcile(last)
        self.assertFalse((self.output / "best.pt").exists())

    def test_disk_best_from_another_fold_is_rejected(self) -> None:
        last = self.checkpoint(2, 0.7)
        self.save_last(last)
        best = self.checkpoint(3, 0.8)
        best["fold_index"] = 1
        save_checkpoint(self.output / "best.pt", best)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.reconcile(last)

    def test_first_complete_best_without_last_is_a_valid_resume_source(self) -> None:
        best = self.checkpoint(1, 0.7)
        save_checkpoint(self.output / "best.pt", best)
        config = replace(self.config, resume=self.output / "best.pt")
        # Exercise the same weights-only load the real CLI performs.
        best = torch.load(config.resume, map_location="cpu", weights_only=True)
        recovered, event = self.reconcile(best, config)
        self.assertFalse((self.output / "last.pt").exists())
        self.assertEqual(recovered["epoch"], 1)
        self.assertEqual(event["actual_checkpoint"], str(config.resume))
        self.assertTrue((self.output / "best_oof.npz").exists())

    def test_oom_microbatch_change_retains_effective_batch(self) -> None:
        last = self.checkpoint(1, 0.7)
        self.save_last(last)
        save_checkpoint(self.output / "best.pt", last)
        config = replace(self.config, batch_size=1, accumulation_steps=32)
        recovered, _ = self.reconcile(last, config)
        self.assertEqual(recovered["epoch"], 1)
        with self.assertRaisesRegex(ValueError, "effective batch"):
            self.reconcile(last, replace(self.config, batch_size=1, accumulation_steps=16))

    def test_legacy_unweighted_objective_cannot_resume_new_weighted_bce(self) -> None:
        checkpoint = self.checkpoint(1, 0.7)
        checkpoint["loss_config"] = {"name": "BCEWithLogitsLoss", "pos_weight": None}
        with self.assertRaisesRegex(ValueError, "objective/statistics mismatch"):
            self.reconcile(checkpoint)
        self.assertFalse((self.output / "best.pt").exists())

    def test_changed_class_counts_are_rejected_before_repairing_files(self) -> None:
        checkpoint = self.checkpoint(1, 0.7)
        expected = dict(checkpoint["loss_config"])
        expected["pos_weight"] = [2.] * 10
        with self.assertRaisesRegex(ValueError, "objective/statistics mismatch"):
            reconcile_resume_best(checkpoint, self.config, PLAN_HASH, SOURCE_HASH,
                                  "bfloat16", expected_loss_config=expected)
        self.assertFalse((self.output / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
