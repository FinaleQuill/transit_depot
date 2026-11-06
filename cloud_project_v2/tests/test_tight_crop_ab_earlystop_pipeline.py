"""CPU-only control-flow tests for the detached AP early-stop controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_tight_crop_ab_earlystop_pipeline as pipeline


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EarlyStopPipelineTests(unittest.TestCase):
    def test_early_stopped_epoch_six_best_two_and_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "A_base"
            folder.mkdir()
            last, best = b"epoch six", b"epoch two"
            (folder / "last.pt").write_bytes(last)
            (folder / "best.pt").write_bytes(best)
            (folder / "config.json").write_text(json.dumps({
                "arm": "A", "member": "base", "epochs": 25,
                "patience": 5, "min_delta": 0.0001}))
            rows = [{"epoch": epoch, "last_checkpoint_sha256": digest(last),
                     "best_checkpoint_sha256": digest(best)} for epoch in range(1, 7)]
            (folder / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (folder / "status.json").write_text(json.dumps({
                "state": "early_stopped", "completed_epochs": 6, "validated_epochs": 6,
                "best_epoch": 2, "best_auc": 0.82, "selected_checkpoint": "best.pt",
                "last_checkpoint_sha256": digest(last),
                "best_checkpoint_sha256": digest(best)}))
            with patch.object(pipeline, "run_dir", return_value=folder):
                result = pipeline.verify_native_run("A", "base", terminal=True)
                self.assertEqual((result["completed_epochs"], result["best_epoch"]), (6, 2))
                (folder / "best.pt").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checkpoint/hash mismatch"):
                    pipeline.verify_native_run("A", "base", terminal=True)

    def test_migration_then_resume_are_distinct_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "A_tiny"
            state = {"jobs": {}, "stage": None}
            calls = []
            finished = {"value": False}

            def fake_command(command, *args, **kwargs):
                calls.append(command)
                if "--migrate-from" in command:
                    folder.mkdir()
                    (folder / "last.pt").write_bytes(b"migrated E1")
                    (folder / "migration.json").write_text("{}")
                else:
                    finished["value"] = True
                    (folder / "best.pt").write_bytes(b"best")
                return 0

            def fake_completed(*_args):
                return ({"completed_epochs": 6, "best_sha256": "best",
                         "last_sha256": "last", "best_epoch": 2, "best_auc": 0.82}
                        if finished["value"] else None)

            with (patch.object(pipeline, "run_dir", return_value=folder),
                  patch.object(pipeline, "persistent_run", return_value=root / "persistent"),
                  patch.object(pipeline, "run_command", side_effect=fake_command),
                  patch.object(pipeline, "completed_job", side_effect=fake_completed),
                  patch.object(pipeline, "verify_native_run", return_value={"completed_epochs": 1}),
                  patch.object(pipeline.old, "mirror_dir"),
                  patch.object(pipeline.old, "sha256", side_effect=lambda path: "best" if Path(path).name == "best.pt" else "migration"),
                  patch.object(pipeline, "write_status")):
                pipeline.run_job("A", "tiny", state, {}, {},
                                 migrate=root / "original_E1/last.pt",
                                 ap_score=root / "monitor.json")
            self.assertEqual(len(calls), 2)
            self.assertIn("--migrate-from", calls[0])
            self.assertNotIn("--resume", calls[0])
            self.assertIn("--resume", calls[1])
            self.assertNotIn("--migrate-from", calls[1])
            self.assertEqual(state["jobs"]["A_tiny"]["state"], "completed")

    def test_pipeline_runs_b_only_after_both_a_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            work = artifacts / "earlystop25"
            old_root = root / "old_runs"
            old_root.mkdir()
            events = []

            def record_job(arm, member, *_args, **_kwargs):
                events.append(f"{arm}_{member}")

            with (patch.object(pipeline, "ARTIFACTS", artifacts),
                  patch.object(pipeline, "WORK", work),
                  patch.object(pipeline, "RUN_ROOT", root / "new_runs"),
                  patch.object(pipeline.old, "ARTIFACTS", artifacts),
                  patch.object(pipeline, "freeze_sources", return_value=({}, {}, {})),
                  patch.object(pipeline, "old_controller_finished", return_value=True),
                  patch.object(pipeline, "adopt_original_e1", return_value=root / "last.pt"),
                  patch.object(pipeline, "score_original_e1", return_value=root / "monitor.json"),
                  patch.object(pipeline, "run_job", side_effect=record_job),
                  patch.object(pipeline, "score_best"),
                  patch.object(pipeline, "final_evaluation", side_effect=lambda *_: events.append("final"))):
                pipeline.pipeline(False)
            self.assertEqual(events, ["A_tiny", "A_base", "B_tiny", "B_base", "final"])

    def test_resume_rejects_live_child_before_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            work = artifacts / "earlystop25"
            work.mkdir()
            (work / "status.json").write_text(json.dumps({
                "child_pid": 321, "child_start_ticks": 456,
                "jobs": {}, "started_utc": "2026-11-01T00:00:00+00:00"}))
            with (patch.object(pipeline, "ARTIFACTS", artifacts),
                  patch.object(pipeline, "WORK", work),
                  patch.object(pipeline, "RUN_ROOT", root / "new_runs"),
                  patch.object(pipeline, "freeze_sources", return_value=({}, {}, {})),
                  patch.object(pipeline, "process_identity", return_value={
                      "state": "S", "start_ticks": 456,
                      "command": f"docker run --output {root / 'new_runs'}/A_tiny"}),
                  patch.object(pipeline, "adopt_original_e1") as adopt):
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    pipeline.pipeline(True)
                adopt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
