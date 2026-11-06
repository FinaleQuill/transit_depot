"""CPU-only contract checks for the audited loader-transition controller."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reconstruction import controller_performance as queue
from reconstruction.local_compliance import LocalComplianceRecorder


def profile():
    return {"schema": queue.PROFILE_SCHEMA, "profile_id": "e2_workers6_prefetch2_20261104",
        "scope": "loader_execution_only", "applies_to_arms": ["global_control", "global_local"],
        "loader": {"num_workers": 6, "prefetch_factor": 2, "multiprocessing_context": "spawn",
                   "in_order": True, "pin_memory": True, "persistent_workers": False, "timeout": 180}}


def session_fixture(root, *, loader_workers=6, omit_loader=False, change_seed=False,
                    omit_archive=False, scientific_change=False, old=False):
    output = root / "arm"
    output.mkdir()
    p = root / "profile.json"
    queue.base.write_json(p, profile())
    split = root / "split.json"
    queue.base.write_json(split, {"roles": {"train": {"studies": 4}, "dev": {"studies": 3}}})
    config = {"workers": 2, "seed": 20261104, "epochs": 6, "augmentation_epoch_offset": 8,
              "split_plan": str(split)}
    evidence = {**config, "source_sha256": {}}
    queue.base.write_json(output / "configuration.json", evidence)
    adapter = Path(queue.__file__).with_name("train_local_performance.py").resolve()
    binding = queue.profile_binding(p, queue.base.sha(p))
    perf = {"execution_profile": binding, "source_sha256": {str(adapter): queue.base.sha(adapter)},
            "transition_arms": {"global_control": {"session_ids": []}}}
    actual_configuration = deepcopy(evidence)
    actual_configuration["execution_profile"] = {**binding, "adapter_source": str(adapter),
        "adapter_source_sha256": queue.base.sha(adapter), "registered_workers": 2,
        "registered_prefetch_factor": 1, "overridden_fields": ["loader.num_workers", "loader.prefetch_factor"],
        "in_order_argument_supported": True}
    if scientific_change:
        actual_configuration["seed"] += 1
    if old:
        actual_configuration = evidence
    sources = [adapter] if omit_archive else [adapter, p]
    recorder = LocalComplianceRecorder(output, trainer="reconstruction.train_local",
        configuration=actual_configuration, source_files=sources)
    if not omit_loader and not old:
        for training, epoch, count, seed in [(True, 9, 4, config["seed"] + 1), (False, 0, 3, config["seed"])]:
            loader = {**profile()["loader"], "num_workers": loader_workers, "batch_size": None,
                "shuffle": training, "worker_init_fn": "reconstruction.train.seed_worker",
                "sampler": "RandomSampler" if training else "SequentialSampler"}
            recorder._record("data_loader_created", {"execution_profile_sha256": binding["sha256"],
                "execution_profile_id": profile()["profile_id"], "registered_workers": 2,
                "training": training, "dataset_epoch": epoch, "study_count": count,
                "seed": seed + int(change_seed), "loader": loader})
        recorder.epoch_completed(1, {"epoch": 1})
    if old:
        (output / "last.pt").write_bytes(b"synthetic epoch zero checkpoint")
        recorder.checkpoint_saved(output / "last.pt", queue.base.sha(output / "last.pt"), epoch=0, role="last_full_state")
    recorder.finish("paused", {"state": "paused"})
    events = queue.local.closed_sessions(output)
    if old:
        perf["transition_arms"]["global_control"]["session_ids"] = [events[0]["session_id"]]
    job = {"name": "global_control", "config": config, "output": str(output)}
    return job, perf


class ExecutionEvidenceTests(unittest.TestCase):
    def test_resume_with_stop_before_loader_is_explicitly_an_empty_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), old=True)
            output = Path(job["output"])
            evidence = queue.base._read_json(output / "configuration.json")
            binding = perf["execution_profile"]
            adapter = Path(queue.__file__).with_name("train_local_performance.py").resolve()
            execution = {**binding, "adapter_source": str(adapter), "adapter_source_sha256": queue.base.sha(adapter),
                "registered_workers": 2, "registered_prefetch_factor": 1,
                "overridden_fields": ["loader.num_workers", "loader.prefetch_factor"], "in_order_argument_supported": True}
            recorder = LocalComplianceRecorder(output, trainer="reconstruction.train_local",
                configuration={**evidence, "execution_profile": execution}, source_files=[adapter, binding["path"]],
                resume=output / "last.pt")
            recorder.finish("paused", {"state": "paused"})
            result = queue.verify_execution_sessions(job, perf)
            self.assertTrue(result["performance_sessions"][0]["no_training_executed"])
            self.assertEqual(result["performance_sessions"][0]["loader_events"], 0)

    def test_new_session_binds_real_profile_source_and_loader_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp))
            result = queue.verify_execution_sessions(job, perf)
            self.assertEqual(result["performance_sessions"][0]["completed_epochs"], [1])
            self.assertEqual(result["performance_sessions"][0]["loader_events"], 2)

    def test_old_session_keeps_its_own_settings_without_relabelling(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), old=True)
            result = queue.verify_execution_sessions(job, perf)
            self.assertEqual(result["performance_sessions"], [])
            self.assertEqual(len(result["historical_sessions"]), 1)

    def test_valid_hash_chain_cannot_hide_wrong_actual_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), loader_workers=2)
            with self.assertRaisesRegex(ValueError, "Actual DataLoader"):
                queue.verify_execution_sessions(job, perf)

    def test_valid_hash_chain_cannot_hide_changed_sampler_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), change_seed=True)
            with self.assertRaisesRegex(ValueError, "sampler seed"):
                queue.verify_execution_sessions(job, perf)

    def test_new_session_requires_loader_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), omit_loader=True)
            with self.assertRaisesRegex(ValueError, "no actual DataLoader"):
                queue.verify_execution_sessions(job, perf)

    def test_new_session_requires_archived_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), omit_archive=True)
            with self.assertRaisesRegex(ValueError, "archived bound source"):
                queue.verify_execution_sessions(job, perf)

    def test_execution_override_cannot_change_scientific_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, perf = session_fixture(Path(tmp), scientific_change=True)
            with self.assertRaisesRegex(ValueError, "scientific configuration"):
                queue.verify_execution_sessions(job, perf)


class QueueIsolationTests(unittest.TestCase):
    def test_transition_archives_old_evidence_and_rejects_later_prefix_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, initial = session_fixture(root, old=True)
            output = Path(job["output"])
            queue.base.write_json(output / "status.json", {"state": "paused"})
            history = output / "history.jsonl"
            history.write_bytes(b'{"epoch":1}\n')
            directory = root / "controller"
            directory.mkdir()
            queue.base.write_json(directory / "STATUS.json", {"state": "paused", "active_container": None})
            (directory / "PROGRESS.md").write_text("Original paused state\n")
            plan = {"jobs": [job]}
            queue.base.write_json(directory / "frozen_plan.json", plan)
            before = (directory / "STATUS.json").read_bytes()
            with mock.patch.object(queue.local, "verify_run", return_value={"epochs": 1, "checkpoint_sha256": {"last.pt": "recorded"}}):
                perf = queue.prepare_performance_plan(directory, plan, initial["execution_profile"])
            self.assertEqual((directory / "STATUS.json").read_bytes(), before)
            self.assertEqual(perf["transition_arms"]["global_control"]["session_ids"],
                             initial["transition_arms"]["global_control"]["session_ids"])
            self.assertTrue((directory / "performance_transition/performance_controller_performance.py").exists())
            with mock.patch.object(queue.base, "verify_plan"):
                queue.verify_performance_plan(plan, perf)
                with history.open("ab") as stream:
                    stream.write(b'{"epoch":2}\n')
                queue.verify_performance_plan(plan, perf)
                history.write_bytes(b'{"epoch":9}\n')
                with self.assertRaisesRegex(ValueError, "Historical training evidence"):
                    queue.verify_performance_plan(plan, perf)

    def test_frozen_performance_file_is_checked_before_polling(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            queue.base.write_json(d / "performance_plan.json", {"original": True})
            ctl = queue.PerformanceController(d, {"jobs": []}, {"execution_profile": {}})
            queue.base.write_json(d / "performance_plan.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "performance-plan file changed"):
                ctl.verify_frozen()

    def test_profile_rejects_out_of_order_persistent_and_extra_options(self):
        for key, value in [("in_order", False), ("persistent_workers", True), ("pin_memory", False), ("drop_last", True)]:
            candidate = profile()
            candidate["loader"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.validate_profile(candidate)

    def test_original_ledger_prefix_can_append_but_never_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            original = b'{"original":true}\n'
            path.write_bytes(original + b'{"new":true}\n')
            expected = hashlib.sha256(original).hexdigest()
            self.assertEqual(queue._prefix(path, len(original)), expected)
            path.write_bytes(b'{"original":null}\n')
            self.assertNotEqual(queue._prefix(path, len(original)), expected)

    def test_launch_redirects_only_gpu_training_and_preserves_cpu_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            queue.base.write_json(d / "performance_plan.json", {})
            binding = {"path": str(d / "profile.json"), "sha256": "a" * 64}
            perf = {"execution_profile": binding}
            ctl = queue.PerformanceController(d, {"jobs": []}, perf)
            with mock.patch.object(ctl, "verify_frozen"), mock.patch.object(queue.local.LocalController, "launch", return_value=0) as launch:
                ctl.launch(["-m", "reconstruction.train_local", "--config", "original.json", "--execute", "--resume"],
                           gpu=True, log_name="fake", container_name="fake")
                args = launch.call_args.args[0]
                self.assertEqual(args[1], "reconstruction.train_local_performance")
                self.assertIn("--resume", args)
                self.assertEqual(args[-4:], ["--execution-profile", binding["path"], "--execution-profile-sha256", binding["sha256"]])
                cpu = ["-m", "reconstruction.controller_local", "--summarize"]
                ctl.launch(cpu, gpu=False, log_name="fake2", container_name="fake2")
                self.assertEqual(launch.call_args.args[0], cpu)

    def test_stop_never_starts_or_expands_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            queue.base.write_json(d / "performance_plan.json", {})
            (d / "STOP").touch()
            plan = {"jobs": [{"name": "global_control", "output": str(d / "unused"), "config_path": "x"}]}
            ctl = queue.PerformanceController(d, plan, {"execution_profile": {}})
            with mock.patch.object(ctl, "verify_frozen"), mock.patch.object(ctl, "launch") as launch:
                self.assertEqual(ctl.run()["state"], "paused")
                launch.assert_not_called()

    def test_training_failure_is_not_retried_and_does_not_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            queue.base.write_json(d / "performance_plan.json", {})
            jobs = [{"name": arm, "output": str(d / arm), "config_path": str(d / (arm + ".json"))}
                    for arm in ["global_control", "global_local"]]
            ctl = queue.PerformanceController(d, {"jobs": jobs}, {"execution_profile": {}})
            with mock.patch.object(ctl, "verify_frozen"), mock.patch.object(ctl, "launch", return_value=9) as launch:
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    ctl.run()
                self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
