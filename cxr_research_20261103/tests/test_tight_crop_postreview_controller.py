"""CPU-only control-flow checks for the detached postreview controller."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tight_crop_postreview as controller


NAMES = {"A_tiny", "A_base", "B_tiny", "B_base", "A_fused", "B_fused"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PostreviewControllerTests(unittest.TestCase):
    def test_strategy_receipt_detects_partial_and_changed_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, source = root / "STRATEGY_REVIEW.md", root / "results.json"
            self.assertFalse(controller.strategy_complete(report))
            report.write_text("review", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both exist"):
                controller.strategy_complete(report)
            source.write_text("{}", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(report.with_suffix(".sources.json"), {
                "report_sha256": digest(report), "sha256": {str(source): digest(source)}})
            self.assertTrue(controller.strategy_complete(report))
            source.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                controller.strategy_complete(report)

    def test_review_complete_requires_all_final_files_and_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            write_json(output / "review.json", {"candidate_scores": dict.fromkeys(NAMES, {})})
            write_json(output / "timing_summary.json", {
                name: {"mean_ms": 1.0} for name in NAMES})
            (output / "REPORT.md").write_text("review", encoding="utf-8")
            for name in NAMES:
                write_json(output / "timing" / name / "latency_report.json", {"mean_ms": 1.0})
            self.assertFalse(controller.review_complete(output))
            write_json(output / "results.json", {
                "candidates": dict.fromkeys(NAMES, {}),
                "timing": dict.fromkeys(NAMES, {"mean_ms": 1.0})})
            self.assertTrue(controller.review_complete(output))

    def test_training_failure_blocks_without_gpu_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status = root / "training.json"
            write_json(status, {"state": "failed", "error": "trainer failed"})
            with patch.object(controller, "CONTROL", root / "controller"), \
                 patch.object(controller, "TRAINING_STATUS", status), \
                 patch.object(controller, "freeze_sources", return_value={}), \
                 patch.object(controller, "check_sources"), \
                 patch.object(controller, "active_gpu_pids") as gpu:
                controller.pipeline(False)
            result = json.loads((root / "controller/status.json").read_text())
            self.assertEqual((result["state"], result["stage"]),
                             ("blocked", "training_failed"))
            gpu.assert_not_called()

    def test_completed_training_waits_for_gpu_idle_before_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = root / "controller"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            strategy = artifacts / "STRATEGY_REVIEW.md"
            strategy.write_text("review", encoding="utf-8")
            status = root / "training.json"
            write_json(status, {"state": "completed", "child_pid": None})
            gpu_pids = iter(([42], []))
            events = []

            def gpu_query():
                pids = next(gpu_pids)
                events.append(("gpu", pids))
                return pids

            def review_done(_):
                events.append(("review", None))
                return True

            with patch.object(controller, "CONTROL", control), \
                 patch.object(controller, "ARTIFACTS", artifacts), \
                 patch.object(controller, "TRAINING_STATUS", status), \
                 patch.object(controller, "ORIGINAL_LOCK", root / "training.lock"), \
                 patch.object(controller, "freeze_sources", return_value={}), \
                 patch.object(controller, "wait_for_training", return_value=True), \
                 patch.object(controller, "active_gpu_pids", side_effect=gpu_query), \
                 patch.object(controller, "review_complete", side_effect=review_done), \
                 patch.object(controller, "strategy_complete", return_value=True), \
                 patch.object(controller, "mirror_directory"), \
                 patch.object(controller.time, "sleep"):
                controller.pipeline(False)
            self.assertEqual(events, [("gpu", [42]), ("gpu", []),
                                      ("review", None), ("review", None)])
            self.assertEqual(json.loads((control / "status.json").read_text())["state"],
                             "completed")

    def test_resume_rejects_live_renderer_child(self):
        with tempfile.TemporaryDirectory() as folder:
            control = Path(folder) / "controller"
            write_json(control / "status.json", {"child_pid": 1234,
                                                  "child_start_ticks": 5678})
            with patch.object(controller, "CONTROL", control), \
                 patch.object(controller, "freeze_sources", return_value={}), \
                 patch.object(controller, "process_identity", return_value={
                     "state": "S", "start_ticks": 5678,
                     "command": "python render_tight_crop_strategy_review.py"}):
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    controller.pipeline(True)


if __name__ == "__main__":
    unittest.main()
