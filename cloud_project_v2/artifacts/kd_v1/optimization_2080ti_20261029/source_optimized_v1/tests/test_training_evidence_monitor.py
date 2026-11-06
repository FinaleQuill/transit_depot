"""Evidence tests use tiny files and harmless subprocesses, never a model/GPU."""
from argparse import Namespace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_trail import read_checkpoint_audit
import training_evidence_monitor as monitor
import run_teachers


class EvidenceMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kd-evidence-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, committed_offset=5):
        folder = self.root / "fold_0"
        folder.mkdir()
        checkpoint = folder / "last.pt"
        checkpoint.write_bytes(b"tiny checkpoint fixture")
        status = {"completed_epochs": 1, "last_checkpoint_sha256": hashlib.sha256(
                  b"tiny checkpoint fixture").hexdigest(), "training_complete": False,
                  "updated_utc": (datetime.now(timezone.utc) + timedelta(seconds=committed_offset)).isoformat()}
        (folder / "status.json").write_text(json.dumps(status))
        return folder, checkpoint, status

    def test_observer_reuses_digest_without_opening_checkpoint(self):
        folder, checkpoint, status = self.fixture()
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path == checkpoint:
                raise AssertionError("Observer must never open checkpoint contents")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            rows = monitor.observe_teacher_checkpoints(self.root)
            monitor.observe_teacher_checkpoints(self.root)
        ledger = read_checkpoint_audit(folder / "checkpoint_audit.jsonl")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(rows[0]["sha256"], status["last_checkpoint_sha256"])
        self.assertEqual(ledger[0]["checkpoint_metadata"]["size_bytes"], checkpoint.stat().st_size)

    def test_replaced_checkpoint_cannot_be_bound_to_old_status(self):
        folder, _, _ = self.fixture(committed_offset=-20)
        self.assertEqual(monitor.observe_teacher_checkpoints(self.root), [])
        self.assertFalse((folder / "checkpoint_audit.jsonl").exists())

    def test_coverage_reports_missing_historical_epoch(self):
        folder, _, _ = self.fixture()
        (folder / "history.jsonl").write_text('{"epoch":1}\n{"epoch":2}\n')
        monitor.observe_teacher_checkpoints(self.root)
        self.assertEqual(monitor.checkpoint_coverage(self.root)["0"]["missing_epochs"], [2])

    def test_record_command_keeps_stdout_stderr_exit_status_and_guard(self):
        project = self.root / "project"
        project.mkdir()
        (project / "train_teacher.py").write_text("# source fixture\n")
        (project / "requirements_kd.txt").write_text("fixture==1\n")
        args = Namespace(evidence_dir=self.root / "evidence", command=[
            sys.executable, "-u", "-c", "import os,sys; print('OUT'); print('ERR',file=sys.stderr); "
            "print(os.environ['KD_TRAINING_LOG_ACTIVE']); sys.exit(7)"])
        with patch.object(monitor, "PROJECT", project):
            code = monitor.run_logged(args)
        self.assertEqual(code, 7)
        receipt = json.loads(next(args.evidence_dir.glob("command_*.json")).read_text())
        text = Path(receipt["console_log"]).read_text()
        self.assertIn("OUT", text)
        self.assertIn("ERR", text)
        self.assertIn("1\n", text)
        self.assertEqual(receipt["exit_code"], 7)
        self.assertEqual((Path(receipt["source_archive"]) / "train_teacher.py").read_text(), "# source fixture\n")

    def test_manifest_refresh_uses_early_stopped_student_actual_epochs(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        student = self.root / "student"
        student.mkdir()
        (student / "status.json").write_text(json.dumps({"state":"early_stopped", "completed_epochs":7}))
        manifest = {"runs":[{"id":"student", "role":"student", "config":str(student / "config.json"),
                             "expected_epochs":25, "complete":False}], "source_archives":[], "console_logs":[]}
        path = evidence / "evidence_manifest.json"
        path.write_text(json.dumps(manifest))
        monitor.refresh_manifest(self.root)
        result = json.loads(path.read_text())["runs"][0]
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected_epochs"], 7)

    def test_resume_receipt_does_not_erase_old_partial_console_coverage(self):
        root_evidence = self.root / "evidence"
        root_evidence.mkdir()
        folder = self.root / "fold_0"
        own = folder / "evidence"
        own.mkdir(parents=True)
        receipt = {"started_at":"2026-10-25T08:00:00Z", "source_archive":str(own / "source_resume"),
                   "console_log":str(own / "resume.log"), "cwd":str(self.root),
                   "preexisting_run_artifacts":["last.pt", "config.json"]}
        (own / "command_resume.json").write_text(json.dumps(receipt))
        manifest = {"runs":[{"id":"teacher_fold_0", "role":"teacher", "config":str(folder / "config.json")}],
                    "console_coverage":[{"run_id":"teacher_fold_0", "coverage_from_start":"partial", "log_ids":[]}],
                    "console_logs":[], "source_archives":[]}
        path = root_evidence / "evidence_manifest.json"
        path.write_text(json.dumps(manifest))
        monitor.refresh_manifest(self.root)
        result = json.loads(path.read_text())
        self.assertEqual(result["console_coverage"][0]["coverage_from_start"], "partial")
        self.assertEqual(result["source_version_timeline"][0]["run_id"], "teacher_fold_0")
        # A second refresh must be stable; no duplicated explanatory text.
        before = path.read_bytes()
        monitor.refresh_manifest(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_parent_scheduler_receipt_applies_to_each_fold(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        receipt = {"started_at":"2026-10-25T08:00:00Z", "source_archive":str(evidence / "source_parent"),
                   "console_log":str(evidence / "parent.log"), "cwd":str(self.root),
                   "preexisting_run_artifacts":[], "command":[sys.executable,"run_teachers.py","--execute"]}
        (evidence / "command_parent.json").write_text(json.dumps(receipt))
        manifest = {"runs":[{"id":f"teacher_fold_{i}", "role":"teacher",
                              "config":str(self.root / f"fold_{i}/config.json")} for i in range(2)],
                    "console_logs":[], "source_archives":[]}
        path = evidence / "evidence_manifest.json"
        path.write_text(json.dumps(manifest))
        monitor.refresh_manifest(self.root)
        result = json.loads(path.read_text())
        self.assertEqual(len(result["source_version_timeline"]), 2)
        self.assertEqual(len(result["console_logs"][0]["run_ids"]), 2)
        self.assertTrue(all(c["coverage_from_start"] == "complete" for c in result["console_coverage"]))

    def test_new_scheduler_accepts_evidence_but_rejects_other_existing_files(self):
        plan = self.root / "folds.json"
        plan.write_text("{}")
        run = self.root / "run"
        (run / "evidence").mkdir(parents=True)
        images = self.root / "images"
        images.mkdir()
        args = run_teachers.parse_args(["--execute","--fold-plan",str(plan),"--run-dir",str(run),
                                       "--image-root",str(images),"--folds","0"])
        fake = Namespace(returncode=0)
        with patch.object(run_teachers.subprocess, "run", return_value=fake):
            self.assertEqual(run_teachers.run(args), 0)
        other = self.root / "other"
        other.mkdir()
        (other / "last.pt").write_bytes(b"fixture")
        args.run_dir = other
        with self.assertRaises(FileExistsError):
            run_teachers.run(args)


if __name__ == "__main__":
    unittest.main()
