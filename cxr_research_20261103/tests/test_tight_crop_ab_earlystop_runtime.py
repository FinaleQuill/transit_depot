"""CPU simulation of the native Study trainer's AP early-stop decisions."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import study_student
from kd_common import sha256_file


class FakeModel:
    def to(self, *args, **kwargs):
        return self

    def requires_grad_(self, value):
        return self

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}


class FakeOptimizer:
    def register_step_post_hook(self, hook):
        return SimpleNamespace(remove=lambda: None)

    def state_dict(self):
        return {"state": {}, "param_groups": []}


class APEarlyStopRuntimeTests(unittest.TestCase):
    def test_ap_patience_and_actual_best_checkpoint(self):
        original_threads = torch.get_num_threads()
        original_benchmark = torch.backends.cudnn.benchmark
        self.addCleanup(torch.set_num_threads, original_threads)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", original_benchmark)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("fold.json", "train.csv", "valid.csv"):
                (root / name).write_text("fixture", encoding="utf-8")
            config = study_student.StudyConfig(
                fold_plan=root / "fold.json", soft_labels=root / "soft",
                output=root / "run", image_root=image_root,
                epochs=25, patience=5, min_delta=1e-4)
            train = [{"subject_id": "train", "study_key": "train/a", "image_id": "a",
                      "split": "train", "targets": [0] * 10}]
            valid = [{"subject_id": "valid", "study_key": "valid/b", "image_id": "b"}]
            plan = {"source_train_manifest": {"path": str(root / "train.csv")},
                    "source_manifest_sha256": sha256_file(root / "train.csv"),
                    "official_holdout": {"path": str(root / "valid.csv"),
                                         "sha256": sha256_file(root / "valid.csv")}}
            # E2 is a real best gain below min_delta; E3 clears min_delta and resets
            # patience. E4 becomes best without resetting patience, then E8 stops.
            ap_scores = [0.70000, 0.70005, 0.70011, 0.70012,
                         0.70010, 0.70009, 0.70008, 0.70007]
            seen_epochs = []

            def train_epoch(model, ema, loader, optimizer, scaler, device, dtype,
                            epoch, supplied, progress, cache, weights):
                seen_epochs.append(epoch)
                progress["backward_batches"] += 1
                progress["optimizer_updates"] += 1
                progress["ema_updates"] += 1
                return {"loss": 0.5, "images": 1, "studies": 1}

            def evaluate(*args):
                score = ap_scores[len(seen_epochs) - 1]
                return {"metrics": {"macro_auc": score, "ap_only_macro_auc": score,
                                    "all_study_macro_auc": 0.95,
                                    "monitor": "ap_only_macro_auc"}}

            objective = {"cache": object(), "protocol": {}, "teacher_provenance": [],
                         "cache_bindings": []}
            scaler = SimpleNamespace(state_dict=lambda: {})
            handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
            original_handlers = dict(handlers)

            def register(sig, handler):
                previous = handlers[sig]
                handlers[sig] = handler
                return previous

            tensor = torch.tensor
            replacements = (
                ("study_student.StudyConfig.validate", lambda self: None),
                ("study_student.StudyConfig.model_config", lambda self: {"backbone_name": "fixture"}),
                ("study_student.validate_output_paths", lambda *args: None),
                ("study_student.require_training_runtime", lambda: None),
                ("study_student.capture_source_version", lambda *args: {}),
                ("cv_splits.read_fold_plan", lambda *args: plan),
                ("dataset.read_manifest", lambda path, **kwargs:
                    train if kwargs["split"] == "train" else valid),
                ("study_student.prepare_student_objective", lambda *args: objective),
                ("student_experiments.compute_student_loss_config", lambda *args, **kwargs:
                    {"pos_weight": [1.0] * 10}),
                ("student_experiments.load_local_pretrained_provenance", lambda *args:
                    {"weights_path": "unused"}),
                ("study_student.student_preprocessing_config", lambda *args: {}),
                ("model_overnight.build_overnight_model", lambda **kwargs: FakeModel()),
                ("timm.utils.ModelEmaV3", lambda *args, **kwargs:
                    SimpleNamespace(module=FakeModel())),
                ("train_overnight.build_optimizer", lambda *args, **kwargs: FakeOptimizer()),
                ("train_overnight.capture_rng", lambda: {}),
                ("study_student.make_study_loader", lambda *args, **kwargs: object()),
                ("dataset_overnight.make_overnight_loader", lambda *args, **kwargs: object()),
                ("study_student.train_study_epoch", train_epoch),
                ("study_student.evaluate_student", evaluate),
                ("study_student.record_checkpoint", lambda *args, **kwargs:
                    {"saved_at_utc": "fixture", "record_sha256": "a" * 64}),
                ("torch.cuda.is_available", lambda: True),
                ("torch.cuda.set_device", lambda *args: None),
                ("torch.cuda.manual_seed_all", lambda *args: None),
                ("torch.amp.GradScaler", lambda *args, **kwargs: scaler),
                ("torch.tensor", lambda *args, **kwargs: tensor(*args, **{
                    key: value for key, value in kwargs.items() if key != "device"})),
                ("signal.signal", register),
            )
            with ExitStack() as stack:
                for name, attributes in {
                    "cv_splits": {"read_fold_plan": lambda *args: plan},
                    "dataset": {"read_manifest": lambda path, **kwargs:
                        train if kwargs["split"] == "train" else valid},
                    "dataset_overnight": {"make_overnight_loader": lambda *args, **kwargs: object()},
                    "student_experiments": {
                        "compute_student_loss_config": lambda *args, **kwargs: {"pos_weight": [1.0] * 10},
                        "load_local_pretrained_provenance": lambda *args: {"weights_path": "unused"}},
                    "timm": {"__path__": []},
                    "timm.utils": {"ModelEmaV3": lambda *args, **kwargs:
                        SimpleNamespace(module=FakeModel())},
                }.items():
                    module = ModuleType(name)
                    module.__dict__.update(attributes)
                    stack.enter_context(patch.dict(sys.modules, {name: module}))
                for name, replacement in replacements:
                    stack.enter_context(patch(name, replacement, create=True))
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = study_student.run_training(config)

            self.assertEqual(seen_epochs, list(range(1, 9)))
            self.assertEqual(result["state"], "early_stopped")
            self.assertEqual(result["completed_epochs"], 8)
            self.assertEqual(result["best_epoch"], 4)
            self.assertAlmostEqual(result["best_auc"], 0.70012)
            best = torch.load(config.output / "best.pt", weights_only=True)
            last = torch.load(config.output / "last.pt", weights_only=True)
            self.assertEqual((best["epoch"], last["epoch"]), (4, 8))
            self.assertEqual(best["validation"]["monitor"], "ap_only_macro_auc")
            self.assertEqual(last["early_stopping"],
                             {"reference_auc": 0.70011, "bad_epochs": 5})
            history = [json.loads(line) for line in
                       (config.output / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["is_best"] for row in history],
                             [True, True, True, True, False, False, False, False])
            self.assertEqual(handlers, original_handlers)


if __name__ == "__main__":
    unittest.main()
