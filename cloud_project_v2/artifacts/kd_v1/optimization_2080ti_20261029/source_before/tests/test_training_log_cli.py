"""CLI recording dispatch only; every training and process-spawn call is mocked."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import train_distill
import train_teacher


class TrainingLogDispatchTests(unittest.TestCase):
    def cases(self):
        common = ["--execute", "--fold-plan", "fold plan.json", "--output", "run output",
                  "--seed", "43", "--batch-size", "2", "--temperature=2"]
        yield (train_teacher, "run_teacher", [x for x in common if x != "--temperature=2"]
               + ["--fold", "0", "--epochs", "20"])
        yield (train_distill, "run_training", common
               + ["--soft-labels", "soft labels", "--no-align-teacher-pos-weight", "--accumulation-steps", "16"])

    def test_explicit_execute_preserves_every_argument_and_returns_wrapper_exit(self):
        for module, training_name, argv in self.cases():
            with self.subTest(module=module.__name__):
                logged = Mock(return_value=7)
                fake = types.SimpleNamespace(run_logged=logged)
                before = list(argv)
                with patch.dict(os.environ, {"KD_TRAINING_LOG_ACTIVE": "0"}), \
                     patch.dict(sys.modules, {"training_evidence_monitor": fake}), \
                     patch.object(module, training_name, side_effect=AssertionError("Training forbidden")):
                    self.assertEqual(module.main(argv), 7)
                options = logged.call_args.args[0]
                self.assertEqual(options.command,
                    [sys.executable, "-u", str(Path(module.__file__).resolve()), *before])
                self.assertEqual(options.evidence_dir, Path("run output/evidence"))
                self.assertEqual(argv, before)

    def test_default_sys_argv_is_preserved(self):
        for module, training_name, argv in self.cases():
            with self.subTest(module=module.__name__):
                logged = Mock(return_value=0)
                with patch.dict(os.environ, {"KD_TRAINING_LOG_ACTIVE": "0"}), \
                     patch.dict(sys.modules, {"training_evidence_monitor": types.SimpleNamespace(run_logged=logged)}), \
                     patch.object(sys, "argv", [module.__file__, *argv]), \
                     patch.object(module, training_name, side_effect=AssertionError("Training forbidden")):
                    self.assertEqual(module.main(), 0)
                self.assertEqual(logged.call_args.args[0].command[3:], argv)

    def test_child_marker_bypasses_wrapper_and_prevents_recursion(self):
        for module, training_name, argv in self.cases():
            with self.subTest(module=module.__name__):
                logged = Mock(side_effect=AssertionError("Recursive logging forbidden"))
                trained = Mock(return_value={"state": "mocked_without_training"})
                with patch.dict(os.environ, {"KD_TRAINING_LOG_ACTIVE": "1"}), \
                     patch.dict(sys.modules, {"training_evidence_monitor": types.SimpleNamespace(run_logged=logged)}), \
                     patch.object(module, training_name, trained), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(argv), 0)
                trained.assert_called_once()
                logged.assert_not_called()

    def test_dry_run_never_starts_recorder_or_training(self):
        for module, training_name, argv in self.cases():
            with self.subTest(module=module.__name__):
                argv = [x for x in argv if x != "--execute"]
                fake = types.SimpleNamespace(run_logged=Mock(side_effect=AssertionError("Recorder forbidden")))
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, {"KD_TRAINING_LOG_ACTIVE": "0"}))
                    stack.enter_context(patch.dict(sys.modules, {"training_evidence_monitor": fake}))
                    stack.enter_context(patch.object(module, training_name,
                                       side_effect=AssertionError("Training forbidden")))
                    if module is train_teacher:
                        stack.enter_context(patch.object(module, "teacher_binding", return_value=(
                            {"training_subject_ids": [], "oof_subject_ids": []}, [])))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    self.assertEqual(module.main(argv), 0)
                fake.run_logged.assert_not_called()


if __name__ == "__main__":
    unittest.main()
