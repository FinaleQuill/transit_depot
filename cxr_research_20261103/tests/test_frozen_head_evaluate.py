"""CPU proxy arithmetic, finite candidate selection and final-head lineage guards."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import frozen_head_evaluate as evaluation
from frozen_head_fit import recipe_from_plan


PLAN_PATH = Path(__file__).resolve().parents[1] / "docs/FROZEN_HEAD_PLAN_20261101.json"


class FrozenHeadEvaluationTests(unittest.TestCase):
    def test_cpu_fp32_linear_two_view_probability_mean_then_study_mean(self):
        features = np.asarray([[[1., 0.], [-1., 0.]], [[2., 0.], [0., 0.]]], dtype=np.float16)
        state = {"head.fc.weight": torch.tensor([[1., 0.]] * 10, dtype=torch.float16),
                 "head.fc.bias": torch.zeros(10, dtype=torch.float16)}
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query forbidden")):
            values = evaluation.image_probabilities(features, state)
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(values[0], [.5] * 10, rtol=0, atol=1e-7)
        np.testing.assert_allclose(values[1], [(1 / (1 + np.exp(-2)) + .5) / 2] * 10, atol=1e-7)
        rows = [{"subject_id": "p", "study_id": "s"}] * 2
        actual = evaluation.study_probabilities(rows, values)[("p", "s")]
        np.testing.assert_array_equal(actual, values.astype(np.float64).mean(axis=0))
        # Averaging logits first would produce a different second image value.
        self.assertGreater(float(abs(values[1, 0] - 1 / (1 + np.exp(-1)))), .03)

    def test_record_order_labels_and_test_split_are_rejected(self):
        original = [{"image_id": "valid:0", "study_key": "p/s", "subject_id": "p", "targets": (0,) * 10},
                    {"image_id": "valid:1", "study_key": "p/t", "subject_id": "p", "targets": (1,) * 10}]
        cached = [{**row, "targets": list(row["targets"]), "split": "valid"} for row in original]
        evaluation.validate_record_order(cached, original)
        for altered in (list(reversed(cached)), [{**cached[0], "targets": [1] * 10}, cached[1]],
                        [{**cached[0], "split": "test"}, cached[1]]):
            with self.assertRaises(ValueError):
                evaluation.validate_record_order(altered, original)

    def test_plan_refuses_new_weight_search_and_changed_horizon(self):
        plan = evaluation.read_json(PLAN_PATH)
        evaluation.validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed["pair_candidates"]["A_F1_B"]["weights"][0][2] = .75
        with self.assertRaisesRegex(ValueError, "Unregistered pair/class weights"):
            evaluation.validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["common"]["epochs"] = 11
        with self.assertRaisesRegex(ValueError, "fixed-final E12"):
            evaluation.validate_plan(changed)

    def test_shortlist_uses_only_registered_pairs_even_if_single_or_extra_scores_higher(self):
        scores = {name: {"macro_auc": .8 + index / 1000} for index, name in enumerate(evaluation.PAIR_MEMBERS)}
        scores["A_F1"] = {"macro_auc": .999}
        scores["unregistered_new_fusion"] = {"macro_auc": 1.}
        selected = evaluation.select_actual_pairs(scores, list(evaluation.PAIR_MEMBERS))
        self.assertEqual(selected, ["A_F2_B_F2", "A_F1_B_F1"])
        with self.assertRaisesRegex(ValueError, "registered pair set"):
            evaluation.select_actual_pairs(scores, [*evaluation.PAIR_MEMBERS, "unregistered_new_fusion"])

    def fit_fixture(self, root):
        plan = evaluation.read_json(PLAN_PATH)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        plan_sha = evaluation.sha256_file(plan_path)
        cache_root = root / "cache"
        cache_root.mkdir()
        (cache_root / "train").mkdir()
        original_head = cache_root / "A_head_fp32.pt"
        original_head.write_bytes(b"Original-head SHA-only fixture; not deserialized in receipt unit test")
        records = cache_root / "train/records.jsonl"
        features = cache_root / "train/A_features.npy"
        records.write_text("immutable train fixture")
        features.write_bytes(b"immutable feature fixture")
        info = {**plan["encoders"]["A"], "head_file": original_head.name,
                "head_sha256": evaluation.sha256_file(original_head)}
        train = {"records_file": "train/records.jsonl", "records_sha256": evaluation.sha256_file(records),
                 "features": {"A": {"file": "train/A_features.npy", "sha256": evaluation.sha256_file(features)}}}
        meta = {"encoders": {"A": info}, "splits": {"train": train}}
        meta_path = cache_root / "meta.json"
        meta_path.write_text(json.dumps(meta))
        cache = {"root": cache_root, "meta": meta, "meta_sha256": evaluation.sha256_file(meta_path)}
        fit = root / "A_F1"
        fit.mkdir()
        config = recipe_from_plan(plan, "F1")
        count = math.ceil(18513 / config.batch_studies)
        history = [{"epoch": epoch, "studies": 18513, "all_original_images_retained": True,
                    "valid_evaluated": False, "optimizer_updates": epoch * count} for epoch in range(1, 13)]
        history_path = fit / "history.jsonl"
        history_path.write_text("".join(json.dumps(row) + "\n" for row in history))
        dimension = info["feature_dim"]
        patch_path = fit / "patch_fp16.pt"
        torch.save({"head.fc.weight": torch.zeros((10, dimension), dtype=torch.float16),
                    "head.fc.bias": torch.zeros(10, dtype=torch.float16)}, patch_path)
        receipt = {"artifact_type": "frozen_fc_training_receipt", "state": "completed", "encoder": "A", "recipe": "F1",
            "training_plan_sha256": plan_sha, "feature_cache_metadata_sha256": cache["meta_sha256"],
            "final_epoch": 12, "completed_epochs": 12, "class_ids": list(range(1, 11)),
            "training_config": asdict(config), "frozen_backbone": True, "head_path": "head.fc",
            "weights_source": "fixed_final_epoch_head_parameters", "checkpoint_selection": "fixed_final_epoch",
            "early_stopping": False, "device": "cpu", "GPU_used": False, "test_used": False,
            "valid_used_for_training": False, "valid_used_for_checkpoint_selection": False,
            "validation_arrays_opened": False, "source_inputs_unchanged": True,
            "num_images": 30328, "num_studies": 18513, "train_manifest_sha256": evaluation.TRAIN_SHA256,
            "optimizer_updates": count * 12, "head_patch_fp16_sha256": evaluation.sha256_file(patch_path),
            "history_sha256": evaluation.sha256_file(history_path),
            "source_files_sha256": {str(path): evaluation.sha256_file(path) for path in
                (meta_path, original_head, records, features, plan_path)}}
        receipt.update({field: info[field] for field in
            ("base_model_dir", "base_config_sha256", "base_weights_sha256", "feature_dim")})
        receipt_path = fit / "training_receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        return fit, cache, plan, plan_sha, receipt_path, receipt

    def test_final_E12_patch_receipt_and_train_cache_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            fit, cache, plan, sha, receipt_path, receipt = self.fit_fixture(Path(directory))
            state, identity = evaluation.load_fitted_head(fit, name="A_F1", cache=cache, plan=plan,
                plan_sha256=sha, bindings={})
            self.assertEqual(identity["final_epoch"], 12)
            self.assertEqual(state["head.fc.weight"].dtype, torch.float16)
            receipt["final_epoch"] = 11
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "final_epoch"):
                evaluation.load_fitted_head(fit, name="A_F1", cache=cache, plan=plan, plan_sha256=sha, bindings={})

    def test_train_receipt_cannot_include_valid_feature_source(self):
        with tempfile.TemporaryDirectory() as directory:
            fit, cache, plan, sha, receipt_path, receipt = self.fit_fixture(Path(directory))
            valid = cache["root"] / "valid"
            valid.mkdir()
            feature = valid / "not_allowed.npy"
            feature.write_bytes(b"forbidden valid fitting source fixture")
            receipt["source_files_sha256"][str(feature)] = evaluation.sha256_file(feature)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "opened valid"):
                evaluation.load_fitted_head(fit, name="A_F1", cache=cache, plan=plan, plan_sha256=sha, bindings={})


if __name__ == "__main__":
    unittest.main()
