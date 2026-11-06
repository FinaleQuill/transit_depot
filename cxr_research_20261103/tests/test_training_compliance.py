"""CPU metadata/byte fixtures only; no model, image, training or inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_compliance import ComplianceRecorder, read_events


class TrainingComplianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "fixture_source.py"
        self.source.write_text("# metadata test only\n")
        self.run = self.root / "run"
        self.packages = patch("training_compliance.metadata.distributions", return_value=[])
        self.packages.start()
        self.addCleanup(self.packages.stop)

    def recorder(self, **kwargs):
        recorder = ComplianceRecorder(self.run, trainer="fixture_only",
            configuration={"seed": 43, "output": self.run},
            source_files=[self.source], demonstration=True, **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def rows(self):
        return read_events(self.run / "compliance/events.jsonl")

    def test_four_points_and_source_bindings(self):
        recorder = self.recorder()
        self.assertEqual(self.rows()[0]["event_type"], "training_started")
        checkpoint = self.run / "last.pt"
        checkpoint.write_bytes(b"NOT A MODEL; demonstration bytes")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        recorder.checkpoint_saved(checkpoint, digest, epoch=1, role="last")
        recorder.epoch_completed(1, {"loss": 0.5})
        recorder.finish("completed", {"fixture_only": True})
        rows = self.rows()
        self.assertEqual([r["event_type"] for r in rows],
            ["training_started", "checkpoint_saved", "epoch_completed", "training_ended"])
        self.assertEqual(rows[2]["checkpoint_events"][0]["record_sha256"], rows[1]["record_sha256"])
        self.assertTrue(all(r["demonstration_only"] for r in rows))
        self.assertTrue(rows[0]["recorded_at_hong_kong"].endswith("+08:00"))
        for filename, key in [("configuration.json", "configuration_sha256"),
                ("source_manifest.json", "source_manifest_sha256"),
                ("environment.json", "environment_sha256"), ("git.json", "git_sha256")]:
            self.assertEqual(hashlib.sha256((recorder.session_dir / filename).read_bytes()).hexdigest(), rows[0][key])
        for source in json.loads((recorder.session_dir / "source_manifest.json").read_text()):
            self.assertEqual(hashlib.sha256((recorder.directory / source["archived_path"]).read_bytes()).hexdigest(), source["sha256"])
        status = json.loads((recorder.directory / "status.json").read_text())
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["completed_epochs_this_session"], [1])
        self.assertFalse(status["final_members_frozen"])

    def test_checkpoint_stat_only_and_rolling_versions_retained(self):
        recorder = self.recorder()
        checkpoint = self.run / "last.pt"
        checkpoint.write_bytes(b"first")
        original_open = Path.open
        def refuse_weights(path, *args, **kwargs):
            if path == checkpoint:
                raise AssertionError("Recorder must not reread weights")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", refuse_weights):
            recorder.checkpoint_saved(checkpoint, hashlib.sha256(b"first").hexdigest(), epoch=1, role="last")
        checkpoint.write_bytes(b"second version")
        recorder.checkpoint_saved(checkpoint, hashlib.sha256(b"second version").hexdigest(), epoch=2, role="last")
        recorder.finish("paused")
        rows = [r for r in self.rows() if r["event_type"] == "checkpoint_saved"]
        self.assertEqual([r["checkpoint_metadata"]["size_bytes"] for r in rows], [5, 14])
        self.assertNotEqual(rows[0]["checkpoint_sha256"], rows[1]["checkpoint_sha256"])

    def test_resume_has_new_session_and_preserves_previous_bytes(self):
        first = self.recorder()
        first.epoch_completed(1, {})
        first.finish("paused")
        before = first.ledger.read_bytes()
        second = self.recorder(resume=self.run / "last.pt")
        second.epoch_completed(2, {})
        second.finish("completed")
        self.assertTrue(first.ledger.read_bytes().startswith(before))
        self.assertNotEqual(first.session_id, second.session_id)
        starts = [r for r in self.rows() if r["event_type"] == "training_started"]
        self.assertEqual(starts[-1]["previous_unclosed_sessions"], [])
        self.assertEqual(starts[-1]["resume_from"], str(self.run / "last.pt"))

    def test_hard_kill_gap_is_never_reported_complete(self):
        first = self.recorder()
        first.close()  # emulate lost invocation, with no invented terminal event
        second = self.recorder(resume=self.run / "last.pt")
        self.assertEqual(self.rows()[-1]["previous_unclosed_sessions"], [first.session_id])
        second.finish("failed", {"error": "fixture"})
        self.assertFalse(any(r["event_type"] == "training_ended" and r["session_id"] == first.session_id for r in self.rows()))

    def test_concurrent_writer_rejected(self):
        first = self.recorder()
        with self.assertRaisesRegex(RuntimeError, "Another compliance writer"):
            self.recorder()
        first.finish("completed")

    def test_complete_tampering_fails_without_rewrite(self):
        recorder = self.recorder()
        row = self.rows()[0]
        row["trainer"] = "tampered"
        corrupt = (json.dumps(row) + "\n").encode()
        recorder.ledger.write_bytes(corrupt)
        with self.assertRaisesRegex(ValueError, "hash chain"):
            recorder.epoch_completed(1, {})
        self.assertEqual(recorder.ledger.read_bytes(), corrupt)
        with self.assertRaisesRegex(ValueError, "hash chain"):
            read_events(recorder.ledger)

    def test_interrupted_tail_preserved_and_recovered(self):
        recorder = self.recorder()
        fragment = b'{"interrupted":'
        with recorder.ledger.open("ab") as stream:
            stream.write(fragment)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.rows()
        recorder.epoch_completed(1, {})
        recovery = self.rows()[1]
        self.assertEqual(recovery["event_type"], "truncated_tail_recovery")
        self.assertEqual(Path(recovery["preserved_fragment"]).read_bytes(), fragment)
        recorder.finish("interrupted")

    def test_missing_checkpoint_and_invalid_digest_never_record_save(self):
        recorder = self.recorder()
        with self.assertRaises(FileNotFoundError):
            recorder.checkpoint_saved(self.run / "missing.pt", "a" * 64, epoch=1, role="last")
        with self.assertRaises(ValueError):
            recorder.checkpoint_saved(self.run / "missing.pt", "bad", epoch=1, role="last")
        self.assertEqual(len(self.rows()), 1)
        recorder.finish("failed")

    def test_nonfinite_metric_explicit_and_finish_idempotent(self):
        recorder = self.recorder()
        recorder.epoch_completed(1, {"auc": float("nan")})
        self.assertEqual(self.rows()[-1]["metrics"]["auc"], {"nonfinite_float": "nan"})
        recorder.finish("early_stopped")
        before = recorder.ledger.read_bytes()
        recorder.finish("completed")
        self.assertEqual(recorder.ledger.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            recorder.epoch_completed(2, {})


if __name__ == "__main__":
    unittest.main()
