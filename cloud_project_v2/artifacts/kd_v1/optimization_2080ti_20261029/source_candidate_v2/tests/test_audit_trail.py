"""Small files only: checkpoint evidence is retained without reading model bytes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_trail import (capture_source_version, read_checkpoint_audit, record_checkpoint,
                         training_output_has_artifacts)


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="checkpoint-evidence-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "last.pt"
        self.checkpoint.write_bytes(b"tiny fixture, not a model")
        self.ledger = self.root / "checkpoint_audit.jsonl"

    def record(self, *, epoch=1, digest="a" * 64, **kwargs):
        return record_checkpoint(self.checkpoint, digest, epoch=epoch, role="last", **kwargs)

    def test_stat_only_record_never_opens_checkpoint_and_keeps_source(self):
        original_open = Path.open

        def reject_checkpoint_read(path, *args, **kwargs):
            if path == self.checkpoint:
                raise AssertionError("Audit must reuse the completed save digest")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", reject_checkpoint_read):
            event = self.record(source_version={"snapshot": "source-v1"})
        self.assertEqual(event["checkpoint_sha256"], "a" * 64)
        self.assertEqual(event["checkpoint_metadata"]["size_bytes"], self.checkpoint.stat().st_size)
        self.assertEqual(event["source_version"], {"snapshot": "source-v1"})
        self.assertIsNotNone(event["saved_at_utc"])

    def test_rolling_file_replacement_preserves_all_epoch_metadata(self):
        first = self.record()
        self.checkpoint.write_bytes(b"different completed checkpoint bytes")
        second = self.record(epoch=2, digest="b" * 64)
        rows = read_checkpoint_audit(self.ledger)
        self.assertEqual([r["epoch"] for r in rows], [1, 2])
        self.assertEqual(rows[0], first)
        self.assertNotEqual(first["checkpoint_metadata"]["size_bytes"],
                            second["checkpoint_metadata"]["size_bytes"])
        self.assertEqual(second["previous_record_sha256"], first["record_sha256"])

    def test_restart_and_observer_duplicate_does_not_append(self):
        first = self.record()
        before = self.ledger.read_bytes()
        observed = self.record(evidence_source="external_monitor")
        self.assertEqual(observed, first)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_repeated_epoch_with_different_actual_digest_is_retained(self):
        self.record()
        self.record(digest="c" * 64)
        self.assertEqual(len(read_checkpoint_audit(self.ledger)), 2)

    def test_historical_recovery_never_borrows_new_file_stat_or_timestamp(self):
        event = self.record(historical=True, observed_status_at="2026-10-25T07:45:32+00:00",
                            evidence_source="original_status_receipt")
        self.assertEqual(event["checkpoint_metadata"], {})
        self.assertIsNone(event["saved_at_utc"])
        self.assertEqual(event["observed_status_at"], "2026-10-25T07:45:32+00:00")
        self.assertTrue(event["historical_recovery"])

    def test_truncated_tail_is_preserved_before_repair(self):
        first = self.record()
        fragment = b'{"event_type":"checkpoint_saved","epoch":'
        with self.ledger.open("ab") as stream:
            stream.write(fragment)
        self.record(epoch=2, digest="b" * 64)
        rows = read_checkpoint_audit(self.ledger)
        self.assertEqual(rows[0], first)
        self.assertEqual(rows[1]["event_type"], "truncated_tail_recovery")
        preserved = Path(rows[1]["preserved_fragment"])
        self.assertEqual(preserved.read_bytes(), fragment)
        self.assertEqual(rows[1]["fragment_sha256"], hashlib.sha256(fragment).hexdigest())
        self.assertEqual(rows[2]["epoch"], 2)

    def test_complete_json_missing_only_newline_keeps_original_record(self):
        first = self.record()
        self.ledger.write_bytes(self.ledger.read_bytes().rstrip(b"\n"))
        self.record(epoch=2, digest="b" * 64)
        self.assertEqual(read_checkpoint_audit(self.ledger)[0], first)

    def test_complete_record_tampering_is_rejected_without_rewrite(self):
        self.record()
        row = json.loads(self.ledger.read_text())
        row["epoch"] = 7
        tampered = (json.dumps(row) + "\n").encode()
        self.ledger.write_bytes(tampered)
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.record(epoch=2, digest="b" * 64)
        self.assertEqual(self.ledger.read_bytes(), tampered)

    def test_corrupt_complete_line_is_rejected_without_dropping_evidence(self):
        self.record()
        with self.ledger.open("ab") as stream:
            stream.write(b"bad complete line\n")
        before = self.ledger.read_bytes()
        with self.assertRaisesRegex(ValueError, "corrupt complete"):
            self.record(epoch=2, digest="b" * 64)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_concurrent_trainer_and_observer_events_remain_valid_and_deduplicated(self):
        def append(epoch):
            return self.record(epoch=epoch)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(append, [1, 2, 3, 4, 1, 2, 3, 4]))
        rows = read_checkpoint_audit(self.ledger)
        self.assertEqual(sorted(r["epoch"] for r in rows), [1, 2, 3, 4])

    def test_source_version_binds_exact_small_files(self):
        source = self.root / "entry.py"
        source.write_text("print('fixture')\n")
        version = capture_source_version([source])
        self.assertEqual(version["files"][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())
        source.write_text("print('changed fixture')\n")
        self.assertNotEqual(capture_source_version([source])["manifest_sha256"], version["manifest_sha256"])

    def test_missing_or_malformed_digest_is_rejected_before_ledger_write(self):
        for digest in (None, "", "a" * 63, "Z" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                self.record(digest=digest)
        self.assertFalse(self.ledger.exists())

    def test_logging_before_first_training_does_not_make_a_new_run_nonempty(self):
        output = self.root / "new-run"
        self.assertFalse(training_output_has_artifacts(output))
        (output / "evidence").mkdir(parents=True)
        (output / "evidence" / "console.log").write_text("Beginning explicit command\n")
        self.assertFalse(training_output_has_artifacts(output))
        for name in ("last.pt", "config.json", "history.jsonl", "checkpoint_audit.jsonl", "unrelated.txt"):
            artifact = output / name
            artifact.write_text("Existing output must remain protected")
            self.assertTrue(training_output_has_artifacts(output), name)
            artifact.unlink()

    def test_a_regular_file_named_evidence_is_not_a_logger_directory(self):
        output = self.root / "new-run"
        output.mkdir()
        (output / "evidence").write_text("unexpected existing user file")
        self.assertTrue(training_output_has_artifacts(output))


if __name__ == "__main__":
    unittest.main()
