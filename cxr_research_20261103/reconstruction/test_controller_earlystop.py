"""CPU evidence and queue checks for the user-authorized E2 stopping amendment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reconstruction import controller_earlystop as queue
from reconstruction.early_stop_policy import POLICY, load_policy, decision
from reconstruction.evaluate import save_evaluation
from reconstruction.local_compliance import LocalComplianceRecorder


def fixture(root, *, epochs=3, state="paused", improving=False, new_session=False,
            owned=True, name="global_control"):
    """Real hash-chained lifecycle and bound predictions, without model execution."""
    root.mkdir(parents=True, exist_ok=True)
    output = root / "arm"
    output.mkdir()
    policy_path = root / "policy.json"
    queue.base.write_json(policy_path, deepcopy(POLICY))
    binding = load_policy(policy_path)
    split = root / "split.json"
    queue.base.write_json(split, {"roles": {"train": {"studies": 20, "images": 20},
                                             "dev": {"studies": 20}}})
    config = {"epochs": 6, "effective_studies": 32, "split_plan": str(split),
              "epoch_zero_max_abs_tolerance": 0.000002, "seed": 20261104}
    evidence = {**config, "source_sha256": {}}
    queue.base.write_json(output / "configuration.json", evidence)
    source_sha = {str(p): queue.base.sha(p) for p in queue.sources()}
    adapter = str(Path(queue.__file__).with_name("train_local_earlystop.py").resolve())
    source_paths = [Path(p) for p in source_sha if Path(p).name != "controller_earlystop.py"]
    recorder = LocalComplianceRecorder(output, trainer="reconstruction.train_local",
        configuration=evidence, source_files=[*source_paths, policy_path])
    if new_session:
        recorder._record("stopping_policy_activated", {"policy_binding": binding,
            "adapter_source": adapter, "adapter_source_sha256": source_sha[adapter]})
    targets = np.tile([0] * 10 + [1] * 10, (10, 1)).T
    history, best, best_epoch, baseline = [], -1., 0, None
    for epoch in range(epochs + 1):
        if improving:
            scores = list(np.arange(.05, 1, .1)) + [.1 + epoch * .1] * 10
        else:
            scores = list(np.arange(.05, 1, .1)) + ([.9] * 10 if epoch == 0 else [.7] * 10)
        report = save_evaluation(output / f"epoch_{epoch:02d}", np.tile(scores, (10, 1)).T,
            targets, [f"s{i}" for i in range(20)], [f"p{i // 2}" for i in range(20)],
            [["AP"] for _ in range(20)], bootstrap_repeats=0, scope="inner_validation",
            metadata={"epoch": epoch})
        score = report["macro_auc"]
        if epoch == 0:
            baseline = score
            queue.base.write_json(output / "baseline.json", {"epoch": 0, "macro_auc": baseline,
                "predictions_sha256": report["predictions_sha256"]})
        if score > best:
            best, best_epoch = score, epoch
            (output / "best.pt").write_bytes(f"synthetic selected EMA {epoch}".encode())
            recorder.checkpoint_saved(output / "best.pt", queue.base.sha(output / "best.pt"),
                                      epoch=epoch, role="best_inner_ema")
        (output / "last.pt").write_bytes(f"synthetic full checkpoint {epoch}".encode())
        recorder.checkpoint_saved(output / "last.pt", queue.base.sha(output / "last.pt"),
                                  epoch=epoch, role="last_full_state")
        if epoch:
            row = {"epoch": epoch, "macro_auc": score, "studies": 20, "images": 20,
                   "attempted_updates": epoch, "best_epoch": best_epoch, "best_macro_auc": best}
            history.append(row)
            with (output / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            recorder.epoch_completed(epoch, row)
            if new_session:
                assessment = decision(baseline, history, POLICY)
                record = {"policy_sha256": binding["sha256"], "decision": assessment,
                          "stop_created": False, "stop_sha256": None,
                          "history_sha256": queue.base.sha(output / "history.jsonl"),
                          "baseline_sha256": queue.base.sha(output / "baseline.json")}
                if assessment["stop"] and epoch < 6:
                    marker = {"schema": "cxr_e2_early_stopping_marker_v1",
                              "stopping_policy_sha256": binding["sha256"], "epoch": epoch,
                              "reason": assessment["stop_reason"],
                              "history_sha256": record["history_sha256"]}
                    queue.base.write_json(output / "STOP", marker if owned else {"external": True})
                    record.update(stop_created=owned, stop_sha256=queue.base.sha(output / "STOP"))
                recorder._record("early_stopping_check", record)
                queue.base.write_json(output / "early_stopping.json", record)
    status = {"state": state, "best_epoch": best_epoch, "best_macro_auc": best,
              "completed_epochs": epochs, "confirmation_evaluated": False,
              "official_valid_evaluated": False, "test_accessed": False}
    queue.base.write_json(output / "status.json", status)
    recorder.finish(state, status)
    events = queue.local.closed_sessions(output)
    transition = {"existing": True, "session_ids": [] if new_session else [events[0]["session_id"]],
                  "prefix_files": {}}
    if not new_session:
        (output / "STOP").write_text("authorized migration boundary\n")
        transition.update(adopt_completed_boundary=owned, epochs=epochs,
            ledger_sha256=queue.base.sha(output / "compliance/events.jsonl"),
            stop_sha256=queue.base.sha(output / "STOP"))
    amendment = {"policy_binding": binding, "source_sha256": source_sha,
                 "transition_arms": {name: transition}}
    job = {"name": name, "output": str(output), "config": config,
           "config_path": str(root / "original_config.json")}
    return job, amendment


class StoppedRunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.performance = mock.patch.object(queue.perf, "verify_execution_sessions", return_value={})
        self.performance.start()
        self.addCleanup(self.performance.stop)

    def test_authorized_closed_boundary_stays_paused_but_policy_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            output = Path(job["output"])
            ledger = (output / "compliance/events.jsonl").read_bytes()
            result = queue.verify_stopped_run(job, amendment, {})
            self.assertTrue(result["policy_complete"])
            self.assertEqual(result["policy_state"], "early_stopped")
            self.assertEqual(result["trainer_state"], "paused")
            self.assertEqual(result["epochs"], 3)
            self.assertFalse(result["equal_update_budget"])
            self.assertEqual((output / "compliance/events.jsonl").read_bytes(), ledger)

    def test_two_bad_epochs_need_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), epochs=2)
            result = queue.verify_stopped_run(job, amendment, {}, require_complete=False)
            self.assertFalse(result["policy_complete"])
            with self.assertRaisesRegex(ValueError, "explicit continuation"):
                queue.verify_stopped_run(job, amendment, {})

    def test_failed_and_interrupted_are_never_normal_early_stops(self):
        for state in ["failed", "interrupted"]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                job, amendment = fixture(Path(tmp), state=state)
                with self.assertRaisesRegex(ValueError, "not a normal early stop"):
                    queue.verify_stopped_run(job, amendment, {})

    def test_external_stop_is_not_policy_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), owned=False)
            with self.assertRaisesRegex(ValueError, "External or unbound STOP"):
                queue.verify_stopped_run(job, amendment, {})

    def test_transition_stop_bytes_must_match_authorized_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            (Path(job["output"]) / "STOP").write_text("new user stop\n")
            with self.assertRaisesRegex(ValueError, "External or unbound STOP"):
                queue.verify_stopped_run(job, amendment, {})

    def test_status_count_and_selected_best_cannot_disagree_with_history(self):
        for key, value in [("completed_epochs", 2), ("best_epoch", 1), ("best_macro_auc", .1),
                           ("test_accessed", True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                job, amendment = fixture(Path(tmp))
                path = Path(job["output"]) / "status.json"
                status = queue.base._read_json(path)
                status[key] = value
                queue.base.write_json(path, status)
                with self.assertRaisesRegex(ValueError, "Final status"):
                    queue.verify_stopped_run(job, amendment, {})

    def test_incomplete_history_tail_is_not_hidden_by_paused_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            with (Path(job["output"]) / "history.jsonl").open("ab") as f:
                f.write(b'{"epoch":4')
            with self.assertRaisesRegex(ValueError, "Incomplete history tail"):
                queue.verify_stopped_run(job, amendment, {})

    def test_unchanged_checkpoint_identity_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            (Path(job["output"]) / "last.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
                queue.verify_stopped_run(job, amendment, {})

    def test_six_improving_epochs_finish_at_max_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), epochs=6, state="completed", improving=True)
            result = queue.verify_stopped_run(job, amendment, {})
            self.assertTrue(result["policy_complete"])
            self.assertEqual(result["stopping_decision"]["stop_reason"], "max_epochs_reached")
            self.assertEqual(result["best_epoch"], 6)

    def test_true_new_policy_session_binds_source_and_replays_all_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), new_session=True)
            result = queue.verify_stopped_run(job, amendment, {})
            self.assertTrue(result["policy_complete"])
            self.assertEqual(result["policy_state"], "early_stopped")

    def test_new_policy_session_cannot_claim_preexisting_user_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), new_session=True, owned=False)
            with self.assertRaisesRegex(ValueError, "External or unbound STOP"):
                queue.verify_stopped_run(job, amendment, {})

    def test_sidecar_cannot_relabel_an_external_stop_as_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), new_session=True, owned=False)
            path = Path(job["output"]) / "early_stopping.json"
            side = queue.base._read_json(path)
            side["stop_created"] = True
            queue.base.write_json(path, side)
            with self.assertRaisesRegex(ValueError, "External or unbound STOP"):
                queue.verify_stopped_run(job, amendment, {})

    def test_stopping_baseline_cannot_drift_from_bound_epoch_zero_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            path = Path(job["output"]) / "baseline.json"
            baseline = queue.base._read_json(path)
            baseline["macro_auc"] -= .01
            queue.base.write_json(path, baseline)
            with self.assertRaisesRegex(ValueError, "baseline|Baseline"):
                queue.verify_stopped_run(job, amendment, {})

    def test_missing_policy_activation_cannot_relabel_a_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp))
            amendment["transition_arms"][job["name"]]["session_ids"] = []
            with self.assertRaisesRegex(ValueError, "lacks the bound stopping policy"):
                queue.verify_stopped_run(job, amendment, {})

    def test_new_policy_session_rejects_changed_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, amendment = fixture(Path(tmp), new_session=True)
            amendment["policy_binding"]["sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "bound stopping policy"):
                queue.verify_stopped_run(job, amendment, {})


def controller_fixture(directory, jobs):
    queue.base.write_json(directory / "performance_plan.json", {})
    queue.base.write_json(directory / "early_stopping_plan.json", {})
    return queue.EarlyStopController(directory, {"jobs": jobs}, {"execution_profile": {}},
                                    {"policy_binding": {}})


class QueueIsolationTests(unittest.TestCase):
    def test_user_controller_stop_never_launches_or_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "STOP").touch()
            job = {"name": "global_control", "output": str(root / "unused"), "config_path": "x"}
            ctl = controller_fixture(root, [job])
            with mock.patch.object(ctl, "verify_frozen"), mock.patch.object(ctl, "launch") as launch:
                self.assertEqual(ctl.run()["state"], "paused")
                launch.assert_not_called()

    def test_nonzero_training_exit_is_not_retried_or_advanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [{"name": n, "output": str(root / n), "config_path": str(root / f"{n}.json")}
                    for n in ["global_control", "global_local"]]
            ctl = controller_fixture(root, jobs)
            with mock.patch.object(ctl, "verify_frozen"), mock.patch.object(ctl, "launch", return_value=9) as launch:
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    ctl.run()
                self.assertEqual(launch.call_count, 1)

    def test_policy_completed_global_advances_to_local_without_rerunning_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [{"name": n, "output": str(root / n), "config_path": str(root / f"{n}.json")}
                    for n in ["global_control", "global_local"]]
            Path(jobs[0]["output"]).mkdir()
            (Path(jobs[0]["output"]) / "status.json").write_text("{}")
            ctl = controller_fixture(root, jobs)
            certificate = {"policy_complete": True, "name": "global_control"}
            with mock.patch.object(ctl, "verify_frozen"), \
                 mock.patch.object(queue, "verify_stopped_run", return_value=certificate), \
                 mock.patch.object(ctl, "launch", return_value=9) as launch:
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    ctl.run()
                self.assertEqual(launch.call_count, 1)
                self.assertIn(jobs[1]["config_path"], launch.call_args.args[0])


class SummaryTests(unittest.TestCase):
    def test_summary_accepts_early_stops_and_explains_changed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, amendment = fixture(root / "global", name="global_control")
            second, other = fixture(root / "local", name="global_local")
            amendment["transition_arms"].update(other["transition_arms"])
            directory = root / "controller"
            directory.mkdir()
            queue.base.write_json(directory / "frozen_plan.json", {"jobs": [first, second]})
            queue.base.write_json(directory / "performance_plan.json", {})
            queue.base.write_json(directory / "early_stopping_plan.json", amendment)
            def fast_evaluation(*args, **kwargs):
                kwargs["bootstrap_repeats"] = 2
                return save_evaluation(*args, **kwargs)
            def fixture_predictions(folder):
                # Production diagnostics hard-bind the 3,703-study manifest;
                # this test has twenty synthetic studies and no image access.
                folder = Path(folder)
                with np.load(folder / "predictions.npz", allow_pickle=False) as stored:
                    values = {key: stored[key].copy() for key in stored.files}
                values["views"] = [json.loads(v) for v in values.pop("views_json")]
                values["binding"] = {"prediction_path": str(folder / "predictions.npz"),
                                     "predictions_sha256": queue.base.sha(folder / "predictions.npz")}
                return values
            with mock.patch.object(queue, "verify_early_plan"), \
                 mock.patch.object(queue.perf, "verify_execution_sessions", return_value={}), \
                 mock.patch("reconstruction.diagnostics.load_bound_predictions", side_effect=fixture_predictions), \
                 mock.patch("reconstruction.evaluate.save_evaluation", side_effect=fast_evaluation):
                result = queue.summarize(directory)
            text = (directory / "comparison/REPORT.md").read_text()
            self.assertTrue(result["complete"])
            self.assertFalse(result["equal_update_budget"])
            self.assertEqual([c["epochs"] for c in result["certificates"]], [3, 3])
            for phrase in ["patience=3", "at most 6 epochs", "budgets can differ", "selection bias"]:
                self.assertIn(phrase, text)
            self.assertIn("global epoch 3", result["selection_caveat"])


if __name__ == "__main__":
    unittest.main()
