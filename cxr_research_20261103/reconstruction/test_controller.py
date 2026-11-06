"""CPU-only lifecycle/queue tests; never launch Docker, a GPU, or a trainer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from reconstruction import controller
from reconstruction.evaluate import save_evaluation
from training_compliance import ComplianceRecorder


def config_pair(root):
    root = Path(root)
    split = root / "splits"
    split.mkdir()
    for name in ("train", "dev"):
        (split / f"{name}.csv").write_text(f"synthetic_{name}_manifest\n")
    plan = {"roles": {name: {"path": str(split / f"{name}.csv"),
                            "sha256": controller.sha(split / f"{name}.csv"), "studies": 4}
                      for name in ("train", "dev")}}
    controller.write_json(split / "split_plan.json", plan)
    configs = []
    for name, aggregation in [("E0", "mean"), ("E1", "attention")]:
        config = {"aggregation": aggregation, "use_local": False, "smoke_studies": 0,
                  "train_manifest": str(split / "train.csv"), "dev_manifest": str(split / "dev.csv"),
                  "split_plan": str(split / "split_plan.json"), "output": str(root / name),
                  "seed": 4, "image_size": 768, "epochs": 2, "effective_studies": 32,
                  "backbone_name": "convnext_base.fb_in22k_ft_in1k_384", "pretrained_path": "synthetic_pretrained",
                  "provenance_path": "synthetic_source", "backbone_lr": 5e-5, "head_lr": 5e-4}
        path = root / f"{name}.json"
        controller.write_json(path, config)
        configs.append(path)
    return configs


def complete_run(job, *, state="completed", second_auc=.875, close=True):
    """Write real hash-chained lifecycle events around synthetic model bytes."""
    output = Path(job["output"])
    output.mkdir()
    config = job["config"]
    evidence = {**config, "source_sha256": {"reconstruction/controller.py": controller.sha(controller.ROOT / "reconstruction/controller.py")}}
    controller.write_json(output / "configuration.json", evidence)
    recorder = ComplianceRecorder(output, trainer="reconstruction.train", configuration=evidence,
                                  source_files=[controller.ROOT / "reconstruction/controller.py"])
    y = np.tile([0, 1, 0, 1], (10, 1)).T
    p = np.tile([.1, .4, .4, .9], (10, 1)).T
    best_epoch = 1
    history = []
    for epoch in (1, 2):
        report = save_evaluation(output / f"epoch_{epoch:02d}", p, y, ["s1", "s2", "s3", "s4"],
            ["p1", "p1", "p2", "p2"], [["AP"], ["AP"], ["PA", "LATERAL"], ["AP"]],
            bootstrap_repeats=0, scope="inner_validation")
        auc = report["macro_auc"] if epoch == 1 else second_auc
        if epoch == 1:
            (output / "best.pt").write_bytes(b"synthetic_best_checkpoint")
            recorder.checkpoint_saved(output / "best.pt", controller.sha(output / "best.pt"), epoch=epoch, role="best_inner_ema")
        (output / "last.pt").write_bytes(f"synthetic_last_epoch{epoch}".encode())
        recorder.checkpoint_saved(output / "last.pt", controller.sha(output / "last.pt"), epoch=epoch, role="last_full_state")
        row = {"epoch": epoch, "macro_auc": auc, "best_macro_auc": .875, "best_epoch": best_epoch}
        history.append(row)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        recorder.epoch_completed(epoch, row)
    result = {"state": state, "best_epoch": best_epoch, "best_macro_auc": .875,
              "confirmation_evaluated": False, "official_valid_evaluated": False}
    controller.write_json(output / "status.json", result)
    controller.write_json(output / "progress.json", result)
    if close:
        recorder.finish(state, result)
    else:
        recorder.close()
    return result


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output_patch = mock.patch.object(controller, "RUN_ROOT", self.root)
        self.output_patch.start()
        fixture_source = self.root / "frozen_source.py"
        fixture_source.write_text("# Immutable CPU queue fixture\n")
        self.source_patch = mock.patch.object(controller, "source_files", return_value=[fixture_source])
        self.source_patch.start()
        self.configs = config_pair(self.root)
        self.queue = self.root / "queue"
        self.plan = controller.build_plan(self.configs, self.queue)

    def tearDown(self):
        self.source_patch.stop()
        self.output_patch.stop()
        self.temp.cleanup()

    def test_plan_matches_roles_and_rejects_wrong_order_or_external_output(self):
        self.assertEqual([j["name"] for j in self.plan["jobs"]], ["E0", "E1"])
        with self.assertRaisesRegex(ValueError, "global E0"):
            controller.build_plan(self.configs[::-1], self.queue)
        config = controller._read_json(self.configs[0])
        config["output"] = str(self.root.parent / "old_experiment")
        controller.write_json(self.configs[0], config)
        with self.assertRaisesRegex(ValueError, "isolated research"):
            controller.build_plan(self.configs, self.queue)

    def test_frozen_source_or_input_change_rejected(self):
        controller.verify_plan(self.plan)
        (self.root / "splits/dev.csv").write_text("changed labels\n")
        with self.assertRaisesRegex(ValueError, "Registered train/dev input changed"):
            controller.verify_plan(self.plan)

    def test_genuine_completion_verified_and_checkpoint_mutation_rejected(self):
        job = self.plan["jobs"][0]
        complete_run(job)
        result = controller.verify_success(job)
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["best_macro_auc"], .875)
        (Path(job["output"]) / "best.pt").write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint hash differs"):
            controller.verify_success(job)

    def test_unclosed_lifecycle_never_certifies_success(self):
        job = self.plan["jobs"][0]
        complete_run(job, close=False)
        with self.assertRaisesRegex(ValueError, "Unclosed"):
            controller.verify_success(job)

    def test_nonfinite_history_and_partial_tail_rejected(self):
        job = self.plan["jobs"][0]
        complete_run(job, second_auc=None)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            controller.verify_success(job)
        with (Path(job["output"]) / "history.jsonl").open("ab") as stream:
            stream.write(b'{"epoch":3')
        with self.assertRaisesRegex(ValueError, "Incomplete history"):
            controller.verify_success(job)

    def test_explicit_resume_requires_closed_session_and_matching_configuration(self):
        job = self.plan["jobs"][0]
        complete_run(job, state="paused")
        controller.verify_resume(job)
        different = {**job, "config": {**job["config"], "seed": 999}}
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            controller.verify_resume(different)

    def test_stop_before_first_job_never_launches_trainer(self):
        self.queue.mkdir()
        (self.queue / "STOP").write_text("explicit request\n")
        instance = controller.Controller(self.queue, self.plan)
        with mock.patch.object(instance, "launch") as launch:
            state = instance.run()
        self.assertEqual(state["state"], "paused")
        launch.assert_not_called()

    def test_e1_waits_for_verified_e0_and_failure_does_not_advance(self):
        self.queue.mkdir()
        instance = controller.Controller(self.queue, self.plan)
        calls = []
        def failed_launch(arguments, **kwargs):
            calls.append(arguments)
            return 9
        with mock.patch.object(instance, "launch", side_effect=failed_launch):
            with self.assertRaisesRegex(RuntimeError, "E0 trainer exited 9"):
                instance.run()
        self.assertEqual(len(calls), 1)
        self.assertIn(str(self.configs[0]), calls[0])

    def test_source_change_after_launch_requests_stop_and_waits_for_child(self):
        self.queue.mkdir()
        instance = controller.Controller(self.queue, self.plan, poll_seconds=.01)
        instance.active_job = self.plan["jobs"][0]
        class Child:
            pid = 12345
            returncode = None
            waited = 0
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.waited += 1
                self.returncode = 0
                return 0
        child = Child()
        with mock.patch.object(controller.subprocess, "Popen", return_value=child), \
             mock.patch.object(controller, "verify_plan", side_effect=ValueError("Frozen source changed")), \
             mock.patch.object(controller, "_running_container", return_value=False):
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                instance.launch(["-m", "synthetic_no_gpu_process"], log_name="synthetic.log", gpu=False,
                                container_name="synthetic_never_launched")
        self.assertEqual(child.waited, 1)
        self.assertTrue((Path(instance.active_job["output"]) / "STOP").is_file())

    def test_successful_serial_queue_has_actual_paired_cpu_comparison(self):
        self.queue.mkdir()
        controller.write_json(self.queue / "frozen_plan.json", self.plan)
        instance = controller.Controller(self.queue, self.plan)
        jobs_started = []
        def synthetic_launch(arguments, **kwargs):
            if "reconstruction.train" in arguments:
                job = instance.active_job
                if job["name"] == "E1":
                    self.assertTrue((self.queue / "E0_verified.json").exists())
                complete_run(job)
                jobs_started.append(job["name"])
            else:
                self.assertFalse(kwargs["gpu"])
                controller.summarize(self.queue)
            return 0
        with mock.patch.object(instance, "launch", side_effect=synthetic_launch):
            state = instance.run()
        self.assertEqual(jobs_started, ["E0", "E1"])
        self.assertEqual(state["state"], "completed")
        report = controller._read_json(self.queue / "comparison/report.json")
        self.assertEqual(report["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"], [0., 0.])
        self.assertFalse(report["confirmation_authorized"])

    def test_controller_lock_rejects_duplicate_owner(self):
        import fcntl
        self.queue.mkdir()
        with (self.queue / ".controller.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                controller.run_controller(self.queue, self.configs)


if __name__ == "__main__":
    unittest.main()
