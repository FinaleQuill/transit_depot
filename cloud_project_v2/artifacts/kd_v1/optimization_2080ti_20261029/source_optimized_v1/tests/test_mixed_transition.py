"""Harmless process mocks test the mixed-budget handoff; no training/GPU."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import tempfile
import unittest
import fcntl
import subprocess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mixed_teachers as mixed


class MixedTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mixed-plan-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old"
        self.new = self.root / "new"
        self.collection = self.root / "collection"
        for folder in (self.old / "fold_0", self.old / "evidence", self.new, self.collection / "evidence"):
            folder.mkdir(parents=True, exist_ok=True)
        self.plan = {
            "original_run":str(self.old), "remaining_run":str(self.new),
            "collection_root":str(self.collection), "teacher_collection":str(self.collection / "teachers.json"),
            "original_scheduler":{"pid":111}, "original_trainer":{"pid":112},
            "handoff_marker_sha256":"a"*64, "retained_epoch13_sha256":"1"*64,
            "retained_epoch13_checkpoint":str(self.old / "retained13.pt"),
            "valid_manifest_sha256":"b"*64, "fold_plan_sha256":"c"*64,
            "validation_output":str(self.collection / "validation"),
            "validation_command":[sys.executable,"validate_teacher.py","--output-dir",str(self.collection / "validation")],
            "remaining_command":[sys.executable,"run_teachers.py","--epochs","12"]}
        (self.old / "fold_0/status.json").write_text(json.dumps({
            "completed_epochs":20,"training_complete":True,"best_checkpoint_sha256":"2"*64}))
        (self.old / "evidence/planned_handoff_receipt.json").write_text(json.dumps({
            "expected_exit_code":75,"blocked_fold":1,"marker_sha256":"a"*64}))

    def report(self):
        return {"complete":True,"diagnostic_only":True,"checkpoint_selection_performed":False,
                "thresholds_fitted":False,"manifest_sha256":"b"*64,"fold_plan_sha256":"c"*64,
                "results":[{"epoch":13,"checkpoint":self.plan["retained_epoch13_checkpoint"],
                            "checkpoint_sha256":"1"*64,"macro_auc":0.8},
                           {"epoch":20,"checkpoint":str(self.old / "fold_0/best.pt"),
                            "checkpoint_sha256":"2"*64,"macro_auc":0.81}],
                "reports":[{"complete":True,"fold":0,"num_images":1422,"num_studies":859,"image_size":768} for _ in range(2)]}

    def test_comparison_accepts_only_bound_complete_results(self):
        self.assertTrue(mixed.validate_comparison(self.report(), self.plan)["complete"])
        for key,value in [("manifest_sha256","0"*64),("complete",False),("thresholds_fitted",True)]:
            report = self.report();report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mixed.validate_comparison(report,self.plan)
        report=self.report();report["results"][0]["checkpoint_sha256"]="f"*64
        with self.assertRaises(ValueError):mixed.validate_comparison(report,self.plan)

    def test_pid_reuse_is_not_the_original_process(self):
        reference={"pid":123,"start_ticks":12,"cmdline_sha256":"one"}
        with patch.object(mixed,"identity",return_value={**reference,"state":"R","start_ticks":13}):
            self.assertFalse(mixed.same_process(reference))

    def run_fixture(self, bad_validation=False):
        calls=[]
        def fake_popen(command,cwd,pass_fds):
            self.assertTrue(pass_fds)
            calls.append(list(command))
            if "validate_teacher.py" in command:
                output=Path(command[command.index("--output-dir")+1]);output.mkdir(parents=True)
                report=self.report()
                if bad_validation:report["manifest_sha256"]="f"*64
                (output/"comparison_report.json").write_text(json.dumps(report))
            else:
                (self.new/"runner_status.json").write_text(json.dumps({"all_requested_complete":True}))
            return SimpleNamespace(pid=999,wait=lambda:0)
        with patch.object(mixed,"register_evidence"),patch.object(mixed,"refresh_manifest"), \
             patch.object(mixed,"same_process",return_value=False), \
             patch.object(mixed,"identity",return_value={"pid":999,"start_ticks":1,"cmdline_sha256":"fake"}), \
             patch.object(mixed.subprocess,"Popen",side_effect=fake_popen), \
             patch.object(mixed,"read_teacher_collection",return_value={}):
            if bad_validation:
                with self.assertRaises(ValueError):mixed.execute(self.plan)
            else:self.assertEqual(mixed.execute(self.plan),0)
        return calls

    def test_natural_handoff_runs_validation_then_twelve_epoch_training(self):
        calls=self.run_fixture()
        self.assertEqual(len(calls),2)
        self.assertIn("validate_teacher.py",calls[0]);self.assertIn("run_teachers.py",calls[1])
        state=json.loads((self.collection/"transition_status.json").read_text())
        self.assertEqual(state["phase"],"completed")

    def test_status_reports_dead_controller_without_rewriting_history(self):
        path = self.collection / "transition_status.json"
        original = json.dumps({"phase":"waiting_for_fold0_twenty_epochs", "controller_pid":123})
        path.write_text(original)
        with patch.object(mixed,"identity",return_value=None):
            state = mixed.status_snapshot(self.plan)
        self.assertFalse(state["controller_alive"])
        self.assertEqual(state["phase"],"controller_not_running")
        self.assertEqual(state["last_recorded_phase"],"waiting_for_fold0_twenty_epochs")
        self.assertEqual(path.read_text(),original)

    def test_legacy_status_rejects_pid_reused_by_another_program(self):
        with patch.object(mixed,"identity",return_value={"state":"R"}), \
             patch.object(Path,"read_bytes",return_value=b"python\0unrelated.py\0--execute\0"):
            self.assertFalse(mixed.controller_is_alive({"controller_pid":123}))
        with patch.object(mixed,"identity",return_value={"state":"R"}), \
             patch.object(Path,"read_bytes",return_value=b"python\0/run/run_mixed_teachers.py\0--execute\0"):
            self.assertTrue(mixed.controller_is_alive({"controller_pid":123}))

    def test_terminal_status_keeps_recorded_result_when_controller_exits(self):
        for phase in ("completed", "failed"):
            (self.collection / "transition_status.json").write_text(json.dumps({"phase":phase,"controller_pid":123}))
            with self.subTest(phase=phase), patch.object(mixed,"identity",return_value=None):
                self.assertEqual(mixed.status_snapshot(self.plan)["phase"],phase)

    def test_invalid_diagnostic_blocks_later_training(self):
        calls=self.run_fixture(bad_validation=True)
        self.assertEqual(len(calls),1)
        self.assertEqual(json.loads((self.collection/"transition_status.json").read_text())["phase"],"failed")

    def test_old_teacher_incomplete_blocks_all_gpu_commands(self):
        (self.old/"fold_0/status.json").write_text(json.dumps({"completed_epochs":13,"training_complete":False}))
        with patch.object(mixed,"register_evidence"),patch.object(mixed,"same_process",return_value=False), \
             patch.object(mixed.subprocess,"Popen") as launch:
            with self.assertRaises(RuntimeError):mixed.execute(self.plan)
        launch.assert_not_called()

    def test_preexisting_partial_validation_uses_a_new_attempt_directory(self):
        Path(self.plan["validation_output"]).mkdir()
        calls=self.run_fixture()
        selected=calls[0][calls[0].index("--output-dir")+1]
        self.assertIn("_retry_",selected)

    def test_stage_inherited_lock_survives_controller_fd_closure(self):
        path=self.root/"lock"
        owner=path.open("a+")
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(0.4)"],
                               pass_fds=(owner.fileno(),))
        self.addCleanup(lambda:child.poll() is None and child.kill())
        owner.close()
        with path.open("a+") as restart:
            with self.assertRaises(BlockingIOError):fcntl.flock(restart,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child.wait()
            fcntl.flock(restart,fcntl.LOCK_EX|fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
