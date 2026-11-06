# 【学习导航】用mock子进程检查五折轮转和OOM降档，不表示制作学习包时实际运行了训练。
# 本次只打包与静态解析，没有执行此测试文件。
"""Runner scheduling tests use fake child processes; no model or training runs."""
from __future__ import annotations

from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_all_folds as runner


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="overnight-runner-mocked-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan_path = self.root / "folds.json"
        self.plan_path.write_text("{}", encoding="utf-8")
        self.args = Namespace(run_dir=self.root / "run", delivery_dir=self.root / "delivery",
            fold_plan=self.plan_path, python=Path(sys.executable), image_root=self.root / "raw",
            valid_manifest=self.root / "valid.csv", test_manifest=self.root / "test.csv",
            sample_submission=self.root / "sample.csv", effective_batch_size=32,
            num_workers=4, epochs=2, patience=4, seed=43, hours=0, skip_inference=True)
        self.config = {"fold_plan_sha256": runner.sha256(self.plan_path),
            "tta": {"scales": [384, 512], "hflip": True},
            "official_valid_manifest_sha256": "a" * 64, "test_manifest_sha256": "b" * 64,
            "sample_submission_sha256": "c" * 64}
        self.visits: list[tuple[int, int, int, list[str]]] = []
        self.counts = {i: 0 for i in range(5)}

    def train_success(self, command: list[str], _log: Path) -> tuple[int, str]:
        self.assertIn("train_overnight.py", command[2])
        fold = int(command[command.index("--fold")+1])
        batch = int(command[command.index("--batch-size")+1])
        accum = int(command[command.index("--accumulation-steps")+1])
        self.assertEqual(batch * accum, self.args.effective_batch_size)
        self.visits.append((fold, batch, accum, command))
        self.counts[fold] += 1
        epoch = self.counts[fold]
        directory = self.args.run_dir / f"fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "last.pt").write_bytes(f"fake-last-{fold}-{epoch}".encode())
        (directory / "best.pt").write_bytes(f"fake-best-{fold}-{epoch}".encode())
        runner.write_json(directory / "status.json", {"state": "completed" if epoch >= self.args.epochs else "paused",
            "completed_epochs": epoch, "fold": fold})
        return 0, "mock only"

    def run_mock(self, callback=None) -> int:
        with patch.object(runner, "run_child", side_effect=callback or self.train_success), \
                patch.object(runner.shutil, "disk_usage", return_value=Namespace(free=100 * 2**30)), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.orchestrate(self.args, self.config, {"name": "fakeGPU", "memory_mib": 16000}, 2, 4)

    def test_round_robin_runs_all_folds_before_any_second_epoch(self) -> None:
        self.assertEqual(self.run_mock(), 0)
        self.assertEqual([v[0] for v in self.visits], list(range(5))*2)
        for _, _, _, command in self.visits[:5]:
            self.assertNotIn("--resume", command)
        for _, _, _, command in self.visits[5:]:
            self.assertIn("--resume", command)
            self.assertTrue(command[command.index("--resume")+1].endswith("last.pt"))
        plan = runner.read_json(self.args.run_dir / "ensemble_plan.json")
        self.assertEqual(len(plan["checkpoints"]), 5)
        self.assertFalse(plan["partial_cv_ensemble"])

    def test_first_epoch_oom_reduces_microbatch_preserving_effective_batch(self) -> None:
        attempts = 0
        def callback(command, log):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                directory = self.args.run_dir / "fold_0"
                directory.mkdir()
                runner.write_json(directory / "status.json", {"state": "failed", "completed_epochs": 0,
                    "oom": True, "error": "CUDA out of memory"})
                return 1, "CUDA out of memory"
            return self.train_success(command, log)
        self.assertEqual(self.run_mock(callback), 0)
        self.assertEqual(self.visits[0][1:3], (1, 32))
        self.assertIn("--restart-empty", self.visits[0][3])
        status = runner.read_json(self.args.run_dir / "runner_status.json")
        self.assertEqual(status["microbatches"]["0"], 1)
        self.assertEqual(status["failures"], {})

    def test_oom_reduction_retains_non_power_of_two_effective_batch(self) -> None:
        self.assertEqual(runner.smaller_batch(18, 54), 9)
        self.assertEqual(runner.smaller_batch(9, 54), 3)
        self.assertEqual(runner.smaller_batch(3, 54), 1)

    def test_non_oom_failure_preserves_best_and_finishes_other_folds(self) -> None:
        def callback(command, log):
            fold = int(command[command.index("--fold")+1])
            if fold == 0 and self.counts[0] == 1:
                runner.write_json(self.args.run_dir / "fold_0" / "status.json",
                    {"state": "failed", "completed_epochs": 1, "error": "synthetic decode failure"})
                return 1, "synthetic decode failure"
            return self.train_success(command, log)
        self.assertEqual(self.run_mock(callback), 0)
        self.assertEqual(self.counts, {0: 1, 1: 2, 2: 2, 3: 2, 4: 2})
        status = runner.read_json(self.args.run_dir / "runner_status.json")
        self.assertIn("0", status["failures"])
        self.assertEqual(len(runner.read_json(self.args.run_dir / "ensemble_plan.json")["checkpoints"]), 5)

    def test_epoch_boundary_budget_leaves_partial_ensemble(self) -> None:
        now = 0.0
        self.args.hours = 30 / 3600
        def callback(command, log):
            nonlocal now
            result = self.train_success(command, log)
            now += 60.0
            return result
        with patch.object(runner.time, "monotonic", side_effect=lambda: now):
            self.assertEqual(self.run_mock(callback), 0)
        self.assertEqual(len(self.visits), 1)
        plan = runner.read_json(self.args.run_dir / "ensemble_plan.json")
        self.assertEqual(len(plan["checkpoints"]), 1)
        self.assertTrue(plan["partial_cv_ensemble"])
        status = runner.read_json(self.args.run_dir / "runner_status.json")
        self.assertEqual(status["stop_reason"], "time_budget_at_epoch_boundary")

    def test_config_mismatch_does_not_call_child_or_overwrite(self) -> None:
        self.args.run_dir.mkdir()
        original = {"different": "settings"}
        runner.write_json(self.args.run_dir / "run_config.json", original)
        with patch.object(runner, "run_child") as child:
            with self.assertRaises(ValueError):
                runner.orchestrate(self.args, self.config, {}, 2, 4)
            child.assert_not_called()
        self.assertEqual(runner.read_json(self.args.run_dir / "run_config.json"), original)

    def test_best_checkpoint_used_when_atomic_last_commit_was_interrupted(self) -> None:
        directory = self.args.run_dir / "fold_0"
        directory.mkdir(parents=True)
        (directory / "best.pt").write_bytes(b"fake-validated-best")
        command = runner.training_command(self.args, 0, 2, 4)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--resume")+1], str(directory / "best.pt"))
        self.assertNotIn("--restart-empty", command)

    def test_inference_oom_retries_fresh_directory_and_smaller_batch(self) -> None:
        self.args.skip_inference = False
        inference_commands = []
        def callback(command, log):
            if "train_overnight.py" in command[2]:
                return self.train_success(command, log)
            inference_commands.append(command)
            destination = Path(command[command.index("--output")+1])
            destination.mkdir()
            if len(inference_commands) == 1:
                runner.write_json(destination / "status.json", {"status": "failed"})
                return 1, "CUDA out of memory"
            (destination / "submission_final.csv").write_text("synthetic-no-model-output\n")
            runner.write_json(destination / "status.json", {"status": "complete"})
            return 0, "mock only"
        self.assertEqual(self.run_mock(callback), 0)
        self.assertEqual([c[c.index("--batch-size")+1] for c in inference_commands], ["4", "2"])
        self.assertNotEqual(inference_commands[0][inference_commands[0].index("--output")+1],
            inference_commands[1][inference_commands[1].index("--output")+1])
        latest = runner.read_json(self.args.run_dir / "latest_submission.json")
        self.assertEqual(latest["sha256"], runner.sha256(Path(latest["path"])))

    def test_initial_stop_file_does_not_launch_any_child(self) -> None:
        self.args.run_dir.mkdir()
        (self.args.run_dir / "STOP").write_text("")
        with patch.object(runner, "run_child") as child, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.orchestrate(self.args, self.config, {}, 2, 4), 1)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
