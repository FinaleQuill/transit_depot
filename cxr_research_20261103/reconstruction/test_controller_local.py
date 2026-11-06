"""CPU checks of real lifecycle binding, baseline selection, and queue isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reconstruction import controller_local as queue
from reconstruction.evaluate import save_evaluation
from reconstruction.local_compliance import LocalComplianceRecorder as ComplianceRecorder


def fixture(root, *, epochs=6, state="completed", budget_offset=0):
    output = root / "arm"
    output.mkdir()
    split = root / "split.json"
    queue.base.write_json(split, {"roles": {"train": {"studies": 4, "images": 5}, "dev": {"studies": 4}}})
    config = {"epochs": 6, "effective_studies": 32, "split_plan": str(split)}
    evidence = {**config, "source_sha256": {}}
    queue.base.write_json(output / "configuration.json", evidence)
    recorder = ComplianceRecorder(output, trainer="reconstruction.train_local", configuration=evidence,
                                  source_files=[queue.ROOT / "reconstruction/controller_local.py"])
    labels = np.tile([0, 1, 0, 1], (10, 1)).T
    for epoch in range(epochs + 1):
        scores = [.1, .4, .4, .9] if epoch == 0 else [.1, .3, .4, .5]
        report = save_evaluation(output / f"epoch_{epoch:02d}", np.tile(scores, (10, 1)).T, labels,
                                 ["s1", "s2", "s3", "s4"], ["p1", "p1", "p2", "p2"],
                                 [["AP"], ["AP"], ["PA", "LL"], ["AP"]],
                                 bootstrap_repeats=0, scope="inner_validation", metadata={"epoch": epoch})
        if epoch == 0:
            (output / "best.pt").write_bytes(b"synthetic untrained parent EMA")
            recorder.checkpoint_saved(output / "best.pt", queue.base.sha(output / "best.pt"),
                                      epoch=0, role="best_inner_ema")
        (output / "last.pt").write_bytes(f"synthetic complete epoch {epoch}".encode())
        recorder.checkpoint_saved(output / "last.pt", queue.base.sha(output / "last.pt"),
                                  epoch=epoch, role="last_full_state")
        if epoch:
            row = {"epoch": epoch, "macro_auc": report["macro_auc"], "studies": 4, "images": 5,
                   "attempted_updates": epoch + budget_offset, "best_epoch": 0, "best_macro_auc": .875}
            with (output / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            recorder.epoch_completed(epoch, row)
    status = {"state": state, "best_epoch": 0, "best_macro_auc": .875,
              "confirmation_evaluated": False, "official_valid_evaluated": False}
    queue.base.write_json(output / "status.json", status)
    recorder.finish(state, status)
    return {"name": "global_control", "output": str(output), "config": config}


class LocalQueueTests(unittest.TestCase):
    def test_true_lifecycle_can_select_untrained_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = fixture(Path(tmp))
            result = queue.verify_run(job)
            self.assertEqual(result["best_epoch"], 0)
            self.assertEqual(result["epochs"], 6)
            self.assertEqual(result["best_macro_auc"], .875)

    def test_checkpoint_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = fixture(Path(tmp))
            (Path(job["output"]) / "best.pt").write_bytes(b"changed parent")
            with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
                queue.verify_run(job)

    def test_fewer_epochs_cannot_claim_fixed_budget_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Fixed epoch budget"):
                queue.verify_run(fixture(Path(tmp), epochs=5))

    def test_self_consistent_but_wrong_update_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "attempted-update budget"):
                queue.verify_run(fixture(Path(tmp), budget_offset=1))

    def test_interrupt_cannot_be_silently_declared_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = fixture(Path(tmp), epochs=3, state="interrupted")
            with self.assertRaisesRegex(ValueError, "Unacceptable trainer state"):
                queue.verify_run(job)
            self.assertEqual(queue.verify_run(job, require_complete=False)["epochs"], 3)

    def test_stop_before_start_never_launches_a_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "STOP").touch()
            plan = {"jobs": [{"name": "global_control", "output": str(d / "unused"), "config_path": "x"}]}
            controller = queue.LocalController(d, plan)
            with mock.patch.object(queue.base, "verify_plan"), mock.patch.object(controller, "launch") as launch:
                self.assertEqual(controller.run()["state"], "paused")
                launch.assert_not_called()

    def test_trainer_failure_never_advances_or_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            plan = {"jobs": [{"name": arm, "output": str(d / arm), "config_path": str(d / (arm + ".json"))}
                             for arm in ["global_control", "global_local"]]}
            controller = queue.LocalController(d, plan)
            with mock.patch.object(queue.base, "verify_plan"), mock.patch.object(controller, "launch", return_value=9) as launch:
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    controller.run()
                self.assertEqual(launch.call_count, 1)
                self.assertIn(str(d / "global_control.json"), launch.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
