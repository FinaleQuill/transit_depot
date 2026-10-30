import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import student_checkpoint_backup as backup


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run, self.local, self.persistent = root / "run", root / "local", root / "work" / "run"
        self.run.mkdir()
        self.publish(1)

    def publish(self, epoch):
        hashes = {}
        for role in ("best", "last"):
            payload = {"epoch": epoch, "model_config": {"backbone_name": "synthetic"},
                       "inference_ready": True, "validation_complete": True,
                       "class_ids": [1], "preprocessing": {"image_size": 2},
                       "model_state_dict": {"weight": torch.tensor([1.125, 2.25]),
                                            "count": torch.tensor([3], dtype=torch.int64)},
                       "raw_model_state_dict": {"secret": torch.ones(2)},
                       "optimizer_state_dict": {"moment": torch.ones(2)},
                       "scaler_state_dict": {}, "rng_state": {}, "training_config": {"seed": 44}}
            path = self.run / (role + ".pt")
            torch.save(payload, path)
            hashes[role + "_checkpoint_sha256"] = backup._sha(path)
        (self.run / "status.json").write_text(json.dumps(hashes))
        (self.run / "config.json").write_text('{"seed":44}')

    def call(self):
        return backup.backup_run(self.run, self.local, self.persistent)

    def test_atomic_full_backup_and_minimal_snapshots(self):
        before = {p.name: backup._sha(p) for p in self.run.iterdir()}
        result = self.call()
        self.assertEqual(result["state"], "complete")
        for role, dtype in (("best", torch.float32), ("last", torch.float16)):
            record = result["checkpoints"][role]
            self.assertEqual(backup._sha(self.local / (role + ".pt")), before[role + ".pt"])
            snap = torch.load(record["snapshot_path"], weights_only=True)
            self.assertEqual(snap["model_state_dict"]["weight"].dtype, dtype)
            self.assertEqual(snap["model_state_dict"]["count"].dtype, torch.int64)
            self.assertEqual(snap["artifact_type"], "student_ema_snapshot")
            self.assertFalse(snap["resume_supported"])
            self.assertFalse(snap["full_training_state_included"])
            self.assertTrue(snap["inference_ready"])
            self.assertTrue(snap["validation_complete"])
            self.assertEqual(snap["original_full_checkpoint_sha256"], before[role + ".pt"])
            for key in ("raw_model_state_dict", "optimizer_state_dict", "rng_state", "scaler_state_dict"):
                self.assertNotIn(key, snap)
        self.assertEqual(before, {p.name: backup._sha(p) for p in self.run.iterdir()})
        self.assertFalse(list(self.persistent.rglob("*.tmp")))
        self.assertTrue((self.persistent / "metadata" / "config.json").exists())

    def test_hash_mismatch_preserves_previous_backups(self):
        self.call()
        old = backup._sha(self.persistent / "best_ema_fp32.pt")
        old_full = backup._sha(self.local / "best.pt")
        (self.run / "best.pt").write_bytes(b"changed before status commit")
        self.assertEqual(self.call()["state"], "retry")
        self.assertEqual(old, backup._sha(self.persistent / "best_ema_fp32.pt"))
        self.assertEqual(old_full, backup._sha(self.local / "best.pt"))

    def test_disk_refusal_preserves_old_snapshot(self):
        self.call()
        old = backup._sha(self.persistent / "best_ema_fp32.pt")
        self.publish(2)
        actual = backup.shutil.disk_usage
        def disk(path):
            if Path(path).resolve().is_relative_to(self.persistent):
                return type(actual(path))(100, 99, 1)
            return actual(path)
        with patch.object(backup.shutil, "disk_usage", side_effect=disk):
            result = self.call()
        self.assertEqual(result["state"], "retry")
        self.assertEqual(old, backup._sha(self.persistent / "best_ema_fp32.pt"))
        self.assertFalse(list(self.persistent.glob("*.tmp")))

    def test_repeat_is_no_op(self):
        self.call()
        files = list(self.persistent.glob("*.pt")) + [self.persistent / "backup_receipt.json"]
        mtimes = {str(p): p.stat().st_mtime_ns for p in files}
        self.assertTrue(self.call()["no_op"])
        self.assertEqual(mtimes, {str(p): p.stat().st_mtime_ns for p in files})

    def test_copy_failure_does_not_replace_old_target(self):
        source, target = self.run / "best.pt", self.local / "best.pt"
        self.local.mkdir()
        target.write_bytes(b"previous")
        with self.assertRaises(ValueError):
            backup._atomic_copy(source, target, "0" * 64)
        self.assertEqual(target.read_bytes(), b"previous")
        self.assertFalse(list(self.local.glob("*.tmp")))

    def test_source_change_during_copy_preserves_target(self):
        source, target = self.run / "best.pt", self.local / "best.pt"
        self.local.mkdir()
        target.write_bytes(b"previous")
        expected = backup._sha(source)
        real_sha = backup._sha
        def changing_sha(path):
            return "0" * 64 if Path(path) == source else real_sha(path)
        with patch.object(backup, "_sha", side_effect=changing_sha):
            with self.assertRaises(ValueError):
                backup._atomic_copy(source, target, expected)
        self.assertEqual(target.read_bytes(), b"previous")
        self.assertFalse(list(self.local.glob("*.tmp")))

    def test_metadata_budget_is_shared_across_runs(self):
        sibling = self.persistent.parent / "other_run" / "metadata"
        sibling.mkdir(parents=True)
        (sibling / "old.log").write_bytes(b"x" * 100)
        with patch.object(backup, "SMALL_FILES_LIMIT", 100):
            result = self.call()
        self.assertEqual(result["small_files"]["state"], "skipped")
        self.assertEqual(result["state"], "retry")
        self.assertEqual((sibling / "old.log").stat().st_size, 100)

    def test_final_metadata_refresh_never_reads_weights(self):
        evidence = self.run / "evidence"
        evidence.mkdir()
        command = evidence / "command.json"
        console = evidence / "console.log"
        command.write_text('{"exit_code":null}')
        console.write_text("training running\n")
        initial = self.call()
        command.write_text('{"exit_code":0}')
        console.write_text("training finished\n")
        real_sha, real_copy = backup._sha, backup._atomic_copy
        def small_sha(path):
            self.assertNotEqual(Path(path).suffix, ".pt")
            return real_sha(path)
        def small_copy(source, target, expected, **kwargs):
            self.assertNotEqual(Path(source).suffix, ".pt")
            self.assertNotEqual(Path(target).suffix, ".pt")
            return real_copy(source, target, expected, **kwargs)
        with patch.object(backup, "_sha", side_effect=small_sha), patch.object(
                backup, "_atomic_copy", side_effect=small_copy):
            result = backup.refresh_run_metadata(self.run, self.persistent)
        self.assertEqual(result["state"], "complete")
        self.assertTrue(result["metadata_refreshed"])
        self.assertEqual(result["checkpoints"], initial["checkpoints"])
        self.assertEqual(result["local_full_backups"], initial["local_full_backups"])
        copied = self.persistent / "metadata" / "evidence"
        self.assertEqual(json.loads((copied / "command.json").read_text())["exit_code"], 0)
        self.assertEqual((copied / "console.log").read_text(), "training finished\n")
        self.assertTrue(json.loads((self.local / "backup_receipt.json").read_text())["metadata_refreshed"])

    def test_refresh_without_receipt_retries(self):
        self.assertEqual(backup.refresh_run_metadata(self.run, self.persistent)["state"], "retry")


if __name__ == "__main__":
    unittest.main()
