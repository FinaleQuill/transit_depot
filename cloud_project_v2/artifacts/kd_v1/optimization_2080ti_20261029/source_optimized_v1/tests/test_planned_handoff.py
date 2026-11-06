"""Small handoff fixtures and mocked dispatch only; no models or training."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planned_handoff import HANDOFF_EXIT_CODE, reject_retired_fold


class PlannedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kd-handoff-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "old_run"
        self.evidence = self.root / "evidence"
        self.evidence.mkdir(parents=True)
        self.marker = self.evidence / "planned_handoff.json"
        self.value = {"artifact_type":"kd_planned_handoff", "original_run":str(self.root),
                      "original_epochs":20, "blocked_folds":[1,2,3,4],
                      "plan_path":str(self.base / "not_to_be_opened.json"),
                      "collection_path":str(self.base / "not_to_be_executed.sh")}

    def write_marker(self):
        self.marker.write_text(json.dumps(self.value))

    def test_exact_retired_launch_has_distinct_receipt_without_creating_fold(self):
        self.write_marker()
        original_marker = self.marker.read_bytes()
        output = self.root / "fold_1"
        result = reject_retired_fold(output, 1, 20)
        persisted = json.loads((self.evidence / "planned_handoff_receipt.json").read_text())
        self.assertEqual(result, persisted)
        self.assertEqual(result["expected_exit_code"], HANDOFF_EXIT_CODE)
        self.assertEqual(result["state"], "planned_handoff")
        self.assertFalse(result["training_started"])
        self.assertEqual(result["blocked_fold"], 1)
        self.assertEqual(result["marker_sha256"], hashlib.sha256(original_marker).hexdigest())
        self.assertEqual(self.marker.read_bytes(), original_marker)
        self.assertFalse(output.exists())
        self.assertFalse(Path(self.value["plan_path"]).exists())

    def test_fold_zero_and_other_schedule_do_not_even_read_marker(self):
        self.marker.write_text("invalid json must not affect fold zero")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Marker read forbidden")):
            self.assertIsNone(reject_retired_fold(self.root / "fold_0", 0, 20))
            self.assertIsNone(reject_retired_fold(self.root / "fold_1", 1, 12))
        self.assertFalse((self.evidence / "planned_handoff_receipt.json").exists())

    def test_absent_marker_is_inert(self):
        self.assertIsNone(reject_retired_fold(self.root / "fold_1", 1, 20))
        self.assertFalse((self.evidence / "planned_handoff_receipt.json").exists())

    def test_marker_must_name_this_absolute_root(self):
        for declared in ("old_run", str(self.base / "different_run")):
            with self.subTest(declared=declared):
                self.value["original_run"] = declared
                self.write_marker()
                self.assertIsNone(reject_retired_fold(self.root / "fold_2", 2, 20))

    def test_new_run_is_not_affected_by_an_old_marker(self):
        new = self.base / "new_run"
        (new / "evidence").mkdir(parents=True)
        (new / "evidence/planned_handoff.json").write_text(json.dumps(self.value))
        self.assertIsNone(reject_retired_fold(new / "fold_1", 1, 20))
        self.assertIsNone(reject_retired_fold(new / "fold_1", 1, 12))

    def test_matching_malformed_control_fields_fail_visibly(self):
        for blocked in ([1,2,3], [True,2,3,4], [1.0,2,3,4], None):
            with self.subTest(blocked=blocked):
                self.value["blocked_folds"] = blocked
                self.write_marker()
                with self.assertRaisesRegex(ValueError, "invalid retired schedule"):
                    reject_retired_fold(self.root / "fold_1", 1, 20)
        self.assertFalse((self.evidence / "planned_handoff_receipt.json").exists())

    def test_foreign_artifact_type_is_inert(self):
        self.value["artifact_type"] = "some_other_record"
        self.write_marker()
        self.assertIsNone(reject_retired_fold(self.root / "fold_4", 4, 20))

    def test_main_exits_before_logger_or_training_with_both_environment_states(self):
        import train_teacher
        self.write_marker()
        argv = ["--execute","--fold-plan","unused_plan.json", "--image-root","unused_images",
                "--output",str(self.root / "fold_1"),"--fold","1","--epochs","20"]
        for active in ("0","1"):
            with self.subTest(logging_active=active), \
                 patch.dict(os.environ,{"KD_TRAINING_LOG_ACTIVE":active}), \
                 patch.object(train_teacher,"run_teacher",side_effect=AssertionError("Training forbidden")), \
                 patch("training_evidence_monitor.run_logged",side_effect=AssertionError("Recorder forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(train_teacher.main(argv), 75)
                self.assertEqual(json.loads(output.getvalue())["state"], "planned_handoff")
        self.assertFalse((self.root / "fold_1").exists())

    def test_main_preview_does_not_run_handoff(self):
        import train_teacher
        argv = ["--fold-plan","unused_plan.json","--image-root","unused_images",
                "--output",str(self.root / "fold_1"),"--fold","1","--epochs","20"]
        with patch.object(train_teacher,"reject_retired_fold",side_effect=AssertionError("Handoff forbidden")), \
             patch.object(train_teacher,"teacher_binding",return_value=(
                 {"training_subject_ids":[],"oof_subject_ids":[]},[])), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train_teacher.main(argv), 0)
        self.assertFalse((self.evidence / "planned_handoff_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
