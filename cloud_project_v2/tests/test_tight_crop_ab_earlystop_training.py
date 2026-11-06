"""CPU checks for the 25-epoch AP monitor and immutable E1 migration."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import study_student
from kd_common import sha256_file
import tight_crop_ab_earlystop_training as early
from tight_crop_ab_training import build_config
from train_distill import _configuration
from train_overnight import calculate_metrics, set_learning_rates


class EarlyStopTrainingTests(unittest.TestCase):
    def test_e1_warmup_lr_is_identical_before_and_after_amendment(self):
        old = SimpleNamespace(epochs=2, warmup_epochs=1, min_lr_ratio=0.05,
                              backbone_lr=1e-5, head_lr=1e-4)
        new = SimpleNamespace(epochs=25, warmup_epochs=1, min_lr_ratio=0.05,
                              backbone_lr=1e-5, head_lr=1e-4)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([{"params": [parameter], "role": "backbone",
                                        "name": "backbone_no_decay", "lr": 1e-5}])
        for step in (0, 1, 100, 300, 578):
            self.assertEqual(set_learning_rates(optimizer, old, step, 579),
                             set_learning_rates(optimizer, new, step, 579))

    def test_ap_monitor_scopes_all_metrics_and_keeps_fullvalid(self):
        records = []
        keys = []
        targets = []
        probabilities = []
        for i in range(24):
            key = f"subject_{i}/study_{i}"
            keys.append(key)
            label = [i % 2] * 10
            targets.append(label)
            probabilities.append([0.2 + 0.6 * (i % 2)] * 10)
            records.append({"study_key": key,
                            "view_position": "AP" if i < 20 else "PA"})
        y, p = np.asarray(targets), np.asarray(probabilities)
        validation = {"study_keys": keys, "targets": y, "probabilities": p,
                      "metrics": calculate_metrics(y, p)}
        result = early.ap_monitor_metrics(validation, records)["metrics"]
        self.assertEqual(result["num_studies"], 20)
        self.assertEqual(result["all_study_num_studies"], 24)
        self.assertEqual(result["macro_auc"], result["ap_only_macro_auc"])
        self.assertEqual(result["per_class_auc"], result["ap_only_per_class_auc"])
        self.assertEqual(result["all_study_macro_auc"], 1.0)
        self.assertEqual(result["monitor_protocol"], early.MONITOR_PROTOCOL)

    def test_migration_preserves_state_and_passes_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_dir, copy_dir, new_dir = root / "old", root / "immutable", root / "new"
            old_dir.mkdir()
            copy_dir.mkdir()
            config = early.build_early_config(arm="A", member="tiny", output=new_dir)
            old_config = build_config(arm="A", member="tiny", output=old_dir)
            raw = {"head.weight": torch.tensor([1.25])}
            ema = {"head.weight": torch.tensor([1.5])}
            optimizer = {"state": {0: {"exp_avg": torch.tensor([2.5])}}, "param_groups": []}
            scaler = {"scale": torch.tensor([1024.])}
            rng = {"python": (1, 2), "numpy": torch.tensor([3]),
                   "torch": torch.tensor([4], dtype=torch.uint8)}
            progress = {"completed_epochs": 1, "validated_epochs": 1,
                        "backward_batches": 18513, "optimizer_updates": 579,
                        "ema_updates": 579, "amp_skipped_steps": 0}
            lineage = {"kind": "student_ema_warm_start", "exact_resume": False,
                       "source_path": str(old_config.init_student_ema.resolve()),
                       "source_snapshot_sha256": old_config.init_student_ema_sha256,
                       "source_epoch": 9}
            old = {"epoch": 1, "artifact_type": "kd_student", "weights_source": "ema",
                   "validation_complete": True, "inference_ready": True,
                   "training_config": _configuration(old_config),
                   "preprocessing": {"ab_preprocess_sha256": config.ab_preprocess_sha256},
                   "model_state_dict": ema, "raw_model_state_dict": raw,
                   "optimizer_state_dict": optimizer, "scaler_state_dict": scaler,
                   "rng_state": rng, "progress": progress,
                   "student_warm_start": lineage,
                   "pretrained_provenance": {"source": "synthetic"},
                   "validation": {"macro_auc": 0.75}, "best_auc": 0.75,
                   "best_epoch": 1, "early_stopping": {"reference_auc": 0.75,
                                                       "bad_epochs": 0}}
            original = old_dir / "last.pt"
            torch.save(old, original)
            digest = sha256_file(original)
            copy = copy_dir / "last.pt"
            copy.write_bytes(original.read_bytes())
            (old_dir / "history.jsonl").write_text(json.dumps({"epoch": 1,
                "last_checkpoint_sha256": digest, "training": {"loss": 0.31}}) + "\n")
            score_dir = root / "monitor"
            score_dir.mkdir()
            monitor = score_dir / "monitor.json"
            metrics = {"macro_auc": 0.81, "ap_only_macro_auc": 0.81,
                       "all_study_macro_auc": 0.79, "monitor": "ap_only_macro_auc",
                       "monitor_protocol": early.MONITOR_PROTOCOL}
            monitor.write_text(json.dumps({"source_sha256": digest,
                "monitor_protocol": early.MONITOR_PROTOCOL,
                "ap_only_macro_auc": 0.81,
                "earlystop_plan_sha256": config.earlystop_plan_sha256,
                "validation": metrics}))
            with patch.object(early, "ORIGINAL_E1", original), \
                 patch.object(early, "IMMUTABLE_E1", copy):
                result = early.migrate_e1(config=config, source=copy, monitor_path=monitor)
            self.assertEqual(result, new_dir / "last.pt")
            self.assertEqual(sha256_file(original), digest)
            self.assertEqual(sha256_file(copy), digest)
            migrated = torch.load(result, map_location="cpu", weights_only=True)
            for field in ("model_state_dict", "raw_model_state_dict", "optimizer_state_dict",
                          "scaler_state_dict", "rng_state", "progress"):
                self.assertEqual(repr(migrated[field]), repr(old[field]))
            self.assertEqual(migrated["migration"]["source_sha256"], digest)
            self.assertEqual(migrated["validation"]["all_study_macro_auc"], 0.79)
            self.assertEqual(migrated["best_auc"], 0.81)
            self.assertEqual(migrated["early_stopping"], {"reference_auc": 0.81,
                                                         "bad_epochs": 0})
            self.assertEqual(json.loads((new_dir / "history.jsonl").read_text())["training"]["loss"], 0.31)
            resume = early.build_early_config(arm="A", member="tiny", output=new_dir,
                                              resume=result)
            expected = {key: value for key, value in migrated.items() if key in
                        ("artifact_type", "preprocessing", "pretrained_provenance")}
            study_student.validate_study_resume(migrated, resume, expected)


if __name__ == "__main__":
    unittest.main()
