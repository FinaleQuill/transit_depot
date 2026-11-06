"""CPU-only checks for AP selection and final best-checkpoint identity."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tight_crop_ab_earlystop_evaluate as evaluation
from train_overnight import calculate_metrics


class APEarlyStopTests(unittest.TestCase):
    def test_native_monitor_keeps_fullvalid_and_scopes_macro_auc_to_ap(self):
        from tight_crop_ab_earlystop_training import ap_monitor_metrics

        keys = [f"subject_{i:04d}/study_0001" for i in range(20)]
        targets = np.asarray([[i % 2] * 10 for i in range(20)], dtype=np.float32)
        values = np.asarray([[0.2 + 0.6 * (i % 2)] * 10 for i in range(20)], dtype=np.float64)
        values[10:] = 1.0 - values[10:]
        records = [{"study_key": key, "view_position": "AP" if i < 10 else "PA"}
                   for i, key in enumerate(keys)]
        full = calculate_metrics(targets, values)
        result = {"study_keys": keys, "targets": targets,
                  "probabilities": values, "metrics": dict(full)}
        monitored = ap_monitor_metrics(result, records)["metrics"]
        self.assertEqual(monitored["monitor"], "ap_only_macro_auc")
        self.assertEqual(monitored["ap_only_num_studies"], 10)
        self.assertAlmostEqual(monitored["ap_only_macro_auc"], 1.0)
        self.assertAlmostEqual(monitored["all_study_macro_auc"], full["macro_auc"])
        self.assertEqual(monitored["macro_auc"], monitored["ap_only_macro_auc"])

    def test_selected_source_requires_best_epoch_and_ap_scope(self):
        digest = "a" * 64
        identities = {"plan_sha256": digest, "earlystop_plan_sha256": digest,
                      "preprocess_sha256": digest, "crop_sha256": digest,
                      "valid_sha256": digest}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "earlystop25"
            source = root / "A_tiny" / "best.pt"
            source.parent.mkdir(parents=True)
            source.touch()
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"orchestration": {"run_root": str(root)}}))
            checkpoint = {"artifact_type": "kd_student", "epoch": 3,
                "weights_source": "ema", "validation_complete": True,
                "inference_ready": True, "official_valid_manifest_sha256": digest,
                "progress": {"completed_epochs": 3, "validated_epochs": 3},
                "training_config": {"arm": "A", "member": "tiny", "epochs": 25,
                    "patience": 5, "min_delta": 1e-4, "seed": 48,
                    "erase_probability": 0.0, "image_size": 768,
                    "validation_tta": "hflip2", "ab_plan_sha256": digest,
                    "ab_preprocess_sha256": digest, "ab_crop_sha256": digest,
                    "earlystop_plan_sha256": digest},
                "preprocessing": {"ab_arm": "A", "ab_member": "tiny",
                    "decoder_candidate": "full_decode_exact_reference_v1",
                    "training_random_erasing": {"p": 0.0},
                    "earlystop_plan_sha256": digest},
                "student_warm_start": {"source_snapshot_sha256": evaluation.INIT_SHA["tiny"]},
                "validation": {"ap_only_macro_auc": 0.7, "all_study_macro_auc": 0.6,
                    "macro_auc": 0.7, "monitor": "ap_only_macro_auc",
                    "macro_auc_scope": "AP-only Studies; all_study_macro_auc retains fullvalid"},
                "best_epoch": 3, "best_auc": 0.7,
                "image_size": 768, "validation_tta": "hflip2",
                "class_ids": list(range(1, 11)), "num_classes": 10,
                "model_config": {"backbone_name": "convnext_tiny.fb_in22k_ft_in1k_384"},
                "model_state_dict": {"head.fc.weight": object()}}
            with patch.object(evaluation, "EARLY_PLAN", plan):
                state, config = evaluation.verify_epoch_source(source, arm="A", member="tiny",
                    epoch=3, identities=identities, checkpoint=checkpoint)
                self.assertIn("head.fc.weight", state)
                self.assertFalse(config["gradient_checkpointing"])
                for change in ({"best_epoch": 2},
                               {"validation": {**checkpoint["validation"], "monitor": "fullvalid"}},
                               {"training_config": {**checkpoint["training_config"], "epochs": 2}},
                               {"training_config": {**checkpoint["training_config"], "patience": 0}},
                               {"training_config": {**checkpoint["training_config"], "min_delta": 0}},
                               {"training_config": {**checkpoint["training_config"], "erase_probability": 0.5}},
                               {"preprocessing": {**checkpoint["preprocessing"],
                                                  "training_random_erasing": {"p": 0.5}}},
                               {"validation_tta": "identity"},
                               {"image_size": 512},
                               {"class_ids": list(range(10))}):
                    altered = {**checkpoint, **change}
                    with self.assertRaises(ValueError):
                        evaluation.verify_epoch_source(source, arm="A", member="tiny",
                            epoch=3, identities=identities, checkpoint=altered)
                with self.assertRaises(ValueError):
                    evaluation.verify_epoch_source(source.with_name("last.pt"), arm="A", member="tiny",
                        epoch=3, identities=identities, checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
