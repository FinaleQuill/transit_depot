"""Explicit E2 loader-only continuation with separately frozen execution evidence.

The original scientific plan, trainer and checkpoint evidence are unchanged.
Historical sessions retain their real loader settings; only subsequent sessions
must bind the new execution profile and record their actual DataLoader options.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from reconstruction import controller as base, controller_local as local
from training_compliance import read_events

SCHEMA = "cxr_e2_loader_performance_plan_v1"
PROFILE_SCHEMA = "cxr_e2_loader_execution_profile_v1"


def validate_profile(profile):
    required = {"schema", "profile_id", "loader", "applies_to_arms", "scope"}
    if set(profile) != required or profile["schema"] != PROFILE_SCHEMA:
        raise ValueError("Unexpected execution-profile schema")
    if (not isinstance(profile["profile_id"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", profile["profile_id"]) is None
            or profile["scope"] != "loader_execution_only"
            or profile["applies_to_arms"] != ["global_control", "global_local"]):
        raise ValueError("Execution profile changes the experiment scope")
    fixed = {"multiprocessing_context": "spawn", "in_order": True, "pin_memory": True,
             "persistent_workers": False, "timeout": 180}
    loader = profile["loader"]
    if not isinstance(loader, dict) or set(loader) != set(fixed) | {"num_workers", "prefetch_factor"}:
        raise ValueError("Unexpected loader options")
    if any(type(loader[k]) is not type(v) or loader[k] != v for k, v in fixed.items()):
        raise ValueError("Only worker count and prefetch may change")
    if (type(loader["num_workers"]) is not int or not 1 <= loader["num_workers"] <= 8
            or type(loader["prefetch_factor"]) is not int or not 1 <= loader["prefetch_factor"] <= 4):
        raise ValueError("Loader concurrency is outside the bounded profile")
    return profile


def profile_binding(path, expected_sha):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Execution profile must be a regular immutable file")
    path = path.resolve(strict=True)
    actual = base.sha(path)
    if actual != expected_sha:
        raise ValueError("Execution profile SHA differs from the explicit request")
    return {"path": str(path), "sha256": actual, "profile": validate_profile(base._read_json(path))}


def performance_sources():
    return [Path(__file__).resolve(), Path(__file__).with_name("train_local_performance.py").resolve()]


def _prefix(path, length):
    with Path(path).open("rb") as stream:
        data = stream.read(length)
    if len(data) != length:
        raise ValueError(f"Historical evidence was truncated: {path}")
    return hashlib.sha256(data).hexdigest()


def prepare_performance_plan(directory, plan, binding):
    """Called only under the original queue lock, after the original job stops."""
    directory = Path(directory)
    destination = directory / "performance_transition"
    if destination.exists():
        raise FileExistsError("Preserve the prior transition; do not overwrite it")
    status = base._read_json(directory / "STATUS.json")
    if status.get("state") != "paused" or status.get("active_container") is not None:
        raise ValueError("Performance transition requires a closed, paused original queue")
    source_hashes = {str(p): base.sha(p) for p in performance_sources()}
    snapshots, arms = {}, {}
    for source in performance_sources():
        snapshots[f"performance_{source.name}"] = source.read_bytes()
    snapshots["execution_profile.json"] = Path(binding["path"]).read_bytes()
    for path in [directory / "STATUS.json", directory / "PROGRESS.md"]:
        snapshots[f"controller_{path.name}"] = path.read_bytes()
    for job in plan["jobs"]:
        output = Path(job["output"])
        if not output.exists() or not any(output.iterdir()):
            arms[job["name"]] = {"existing": False, "session_ids": [], "prefix_files": {}}
            continue
        state = base._read_json(output / "status.json")["state"]
        if state not in {"paused", "completed"}:
            raise ValueError("Transition requires a complete epoch boundary, not a failed/interrupted arm")
        verified = local.verify_run(job, require_complete=state == "completed")
        events = local.closed_sessions(output)
        prefixes = {}
        for name in ["history.jsonl", "compliance/events.jsonl"]:
            path = output / name
            data = path.read_bytes() if path.exists() else b""
            snapshot = f"{job['name']}_{name.replace('/', '_')}"
            snapshots[snapshot] = data
            prefixes[str(path)] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                    "snapshot": str(destination / snapshot)}
        for name in ["status.json", "configuration.json", "compliance/status.json"]:
            snapshots[f"{job['name']}_{name.replace('/', '_')}"] = (output / name).read_bytes()
        arms[job["name"]] = {"existing": True, "state": state, "verified": verified,
            "session_ids": sorted({e["session_id"] for e in events}), "prefix_files": prefixes}
    result = {"schema": SCHEMA, "created_at": base.now(), "original_plan_path": str(directory / "frozen_plan.json"),
        "original_plan_sha256": base.sha(directory / "frozen_plan.json"), "execution_profile": binding,
        "source_sha256": source_hashes, "transition_arms": arms,
        "scientific_configuration_changed": False, "historical_sessions_relabelled": False,
        "original_controller_state": status, "snapshot_sha256": {
            str(destination / name): hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}}
    destination.mkdir()
    for name, data in snapshots.items():
        with (destination / name).open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    base.write_json(directory / "performance_plan.json", result)
    return result


def verify_performance_plan(plan, performance):
    if performance.get("schema") != SCHEMA:
        raise ValueError("Unknown performance-plan schema")
    if base.sha(performance["original_plan_path"]) != performance["original_plan_sha256"]:
        raise ValueError("Original scientific plan changed")
    base.verify_plan(plan)
    binding = performance["execution_profile"]
    if profile_binding(binding["path"], binding["sha256"]) != binding:
        raise ValueError("Execution profile content changed")
    if set(performance["source_sha256"]) != {str(p) for p in performance_sources()}:
        raise ValueError("Performance source inventory changed")
    for path, expected in performance["source_sha256"].items():
        if base.sha(path) != expected:
            raise ValueError(f"Frozen performance source changed: {path}")
    for path, expected in performance["snapshot_sha256"].items():
        if base.sha(path) != expected:
            raise ValueError("Archived transition evidence changed")
    for arm in performance["transition_arms"].values():
        for path, record in arm["prefix_files"].items():
            if record["bytes"] and _prefix(path, record["bytes"]) != record["sha256"]:
                raise ValueError("Historical training evidence changed after loader transition")


def verify_execution_sessions(job, performance):
    """Validate real closed sessions, their profile archive and actual loader use."""
    output = Path(job["output"])
    events = local.closed_sessions(output)
    prior = set(performance["transition_arms"][job["name"]]["session_ids"])
    binding = performance["execution_profile"]
    adapter = Path(__file__).with_name("train_local_performance.py").resolve()
    expected_adapter = performance["source_sha256"][str(adapter)]
    original_evidence = base._read_json(output / "configuration.json")
    sessions = {}
    for event in events:
        sessions.setdefault(event["session_id"], []).append(event)
    if not prior <= set(sessions):
        raise ValueError("Historical training sessions disappeared")
    verified = []
    for sid, rows in sessions.items():
        start = rows[0]
        folder = output / "compliance" / start["session_evidence"]
        configuration = base._read_json(folder / "configuration.json")
        recorded = configuration.get("execution_profile")
        if sid in prior:
            if recorded is not None:
                raise ValueError("Historical session was retroactively assigned a performance profile")
            continue
        if not isinstance(recorded, dict):
            raise ValueError("New training session lacks an execution-profile binding")
        for key in ["path", "sha256", "profile"]:
            if recorded.get(key) != binding[key]:
                raise ValueError("Session execution profile differs from frozen plan")
        if (recorded.get("adapter_source_sha256") != expected_adapter
                or recorded.get("adapter_source") != str(adapter)
                or recorded.get("registered_workers") != job["config"]["workers"]
                or recorded.get("registered_prefetch_factor") != 1
                or recorded.get("overridden_fields") != ["loader.num_workers", "loader.prefetch_factor"]
                or recorded.get("in_order_argument_supported") is not True):
            raise ValueError("Session execution adapter or registered loader differs")
        if {k: v for k, v in configuration.items() if k != "execution_profile"} != original_evidence:
            raise ValueError("Performance session changed scientific configuration evidence")
        sources = base._read_json(folder / "source_manifest.json")
        for path, sha in [(binding["path"], binding["sha256"]), (str(adapter), expected_adapter)]:
            matches = [s for s in sources if s["original_path"] == path and s["sha256"] == sha]
            if len(matches) != 1:
                raise ValueError("Execution adapter/profile lacks an archived bound source")
        loaders = [e for e in rows if e["event_type"] == "data_loader_created"]
        if not loaders:
            # The frozen trainer can resume with a new STOP already present,
            # or close epoch 6 after a previous epoch-boundary STOP. Neither
            # path builds a loader or performs any optimizer update.
            preceding = events[:events.index(start)]
            saved = [e for e in preceding if e["event_type"] == "checkpoint_saved"
                     and e.get("role") == "last_full_state"]
            resume = start.get("resume_from")
            state = rows[-1].get("state")
            if (len(rows) != 2 or not resume or Path(resume).resolve() != output / "last.pt"
                    or not saved or state not in {"paused", "completed"}
                    or (state == "completed" and saved[-1]["epoch"] != job["config"]["epochs"])):
                raise ValueError("New training session has no actual DataLoader evidence")
            verified.append({"session_id": sid, "completed_epochs": [], "loader_events": 0,
                             "profile_sha256": binding["sha256"], "no_training_executed": True,
                             "resumed_checkpoint_epoch": saved[-1]["epoch"], "state": state})
            continue
        split = base._read_json(job["config"]["split_plan"])
        for event in loaders:
            training = event.get("training")
            epoch = event.get("dataset_epoch")
            if type(training) is not bool or type(epoch) is not int:
                raise ValueError("Loader event lacks training/epoch identity")
            expected = {**binding["profile"]["loader"], "batch_size": None, "shuffle": training,
                "worker_init_fn": "reconstruction.train.seed_worker",
                "sampler": "RandomSampler" if training else "SequentialSampler"}
            if (event.get("loader") != expected or event.get("execution_profile_sha256") != binding["sha256"]
                    or event.get("execution_profile_id") != binding["profile"]["profile_id"]
                    or event.get("registered_workers") != job["config"]["workers"]
                    or event.get("study_count") != split["roles"]["train" if training else "dev"]["studies"]):
                raise ValueError("Actual DataLoader settings differ from frozen profile")
            seed = job["config"]["seed"]
            if training:
                actual_epoch = epoch - job["config"]["augmentation_epoch_offset"]
                if not 1 <= actual_epoch <= job["config"]["epochs"]:
                    raise ValueError("Training loader used the wrong augmentation epoch")
                seed += actual_epoch
            elif epoch != 0:
                raise ValueError("Evaluation dataset must keep its original epoch zero")
            if event.get("seed") != seed:
                raise ValueError("Loader sampler seed changed")
        completed = [e["epoch"] for e in rows if e["event_type"] == "epoch_completed"]
        training_epochs = [e["dataset_epoch"] - job["config"]["augmentation_epoch_offset"]
                           for e in loaders if e["training"]]
        evaluation_count = sum(not e["training"] for e in loaders)
        if any(epoch not in training_epochs for epoch in completed) or evaluation_count < len(completed):
            raise ValueError("Completed epochs lack actual train/evaluation loader records")
        verified.append({"session_id": sid, "completed_epochs": completed, "loader_events": len(loaders),
                         "profile_sha256": binding["sha256"]})
    return {"historical_sessions": sorted(prior), "performance_sessions": verified}


class PerformanceController(local.LocalController):
    def __init__(self, directory, plan, performance, **kwargs):
        super().__init__(directory, plan, **kwargs)
        self.performance = performance
        self.performance_plan_sha256 = base.sha(self.directory / "performance_plan.json")
        self.state["execution_profile"] = performance["execution_profile"]
        self.state["performance_plan"] = str(self.directory / "performance_plan.json")
        self.state["performance_plan_sha256"] = self.performance_plan_sha256

    def verify_frozen(self):
        if base.sha(self.directory / "performance_plan.json") != self.performance_plan_sha256:
            raise ValueError("Frozen performance-plan file changed")
        verify_performance_plan(self.plan, self.performance)

    def wait_child(self):
        while self.child.poll() is None:
            reason = self.stop_requested()
            if reason:
                if self.active_job:
                    self.request_current_stop(reason)
                else:
                    self.child.terminate()
            else:
                self.verify_frozen()
                self.publish("training" if self.active_job else "summarizing")
                if self.state.get("monitor_error"):
                    raise ValueError(self.state["monitor_error"])
            try:
                return self.child.wait(timeout=self.poll_seconds)
            except subprocess.TimeoutExpired:
                pass
        return self.child.wait()

    def launch(self, arguments, **kwargs):
        arguments = list(arguments)
        if kwargs["gpu"]:
            if arguments[:2] != ["-m", "reconstruction.train_local"]:
                raise ValueError("Only the registered E2 trainer may use the execution adapter")
            binding = self.performance["execution_profile"]
            arguments[1] = "reconstruction.train_local_performance"
            arguments += ["--execution-profile", binding["path"], "--execution-profile-sha256", binding["sha256"]]
        self.verify_frozen()
        return super().launch(arguments, **kwargs)

    def verify_job(self, job, *, require_complete=True):
        self.verify_frozen()
        result = local.verify_run(job, require_complete=require_complete)
        result["execution_sessions"] = verify_execution_sessions(job, self.performance)
        return result

    def run(self):
        verified = []
        for job in self.plan["jobs"]:
            self.verify_frozen()
            if self.stop_requested():
                self.publish("paused", stop_reason=self.stop_requested()); return self.state
            self.active_job = job
            self.publish("checking", current_job=job["name"], trainer_progress={}, actual_epoch_auc=[])
            output = Path(job["output"])
            existing = output.exists() and any(output.iterdir())
            if existing and (output / "status.json").exists() and base._read_json(output / "status.json")["state"] == "completed":
                result = self.verify_job(job)
            else:
                resume = job["config_path"] in self.resume_runs
                if existing and not resume:
                    raise ValueError("Existing unfinished output needs an explicit --resume-run")
                if resume:
                    self.verify_job(job, require_complete=False)
                    if (output / "STOP").exists():
                        archive = self.directory / "resume_stop_markers"
                        archive.mkdir(exist_ok=True)
                        os.replace(output / "STOP", archive / f"{job['name']}_{time.time_ns()}.STOP")
                stamp = f"{int(time.time())}_{os.getpid()}"
                args = ["-m", "reconstruction.train_local", "--config", job["config_path"], "--execute"]
                if resume:
                    args.append("--resume")
                code = self.launch(args, log_name=f"{job['name']}_performance_{stamp}.log", gpu=True,
                    container_name=f"cxr-e2-{job['name'].replace('_','-')}-perf-{stamp}")
                if self.stop_requested():
                    self.publish("paused", stop_reason=self.stop_requested(), trainer_exit_code=code); return self.state
                if code:
                    raise RuntimeError(f"Trainer exited {code}; failure preserved; no automatic retry")
                result = self.verify_job(job)
            verified.append(result)
            base.write_json(self.directory / f"{job['name']}_verified.json", result)
            self.publish("verified", completed_jobs=list(verified))
            self.active_job = None
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested()); return self.state
        self.publish("summarizing", current_job="paired_comparison", trainer_progress={})
        stamp = f"{int(time.time())}_{os.getpid()}"
        code = self.launch(["-m", "reconstruction.controller_local", "--controller-dir", str(self.directory), "--summarize"],
            log_name=f"comparison_performance_{stamp}.log", gpu=False, container_name=f"cxr-e2-compare-perf-{stamp}")
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested()); return self.state
        if code or not base._read_json(self.directory / "comparison/report.json").get("complete"):
            raise RuntimeError("Paired comparison did not complete")
        self.verify_frozen()
        self.publish("completed", current_job=None, comparison_report=str(self.directory / "comparison/REPORT.md"))
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs=2, type=Path, required=True)
    parser.add_argument("--execution-profile", type=Path, required=True)
    parser.add_argument("--execution-profile-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--resume-run", action="append", type=Path, default=[])
    args = parser.parse_args()
    directory = base._inside(args.controller_dir, local.RUN_ROOT / "E2_20261104")
    binding = profile_binding(args.execution_profile, args.execution_profile_sha256)
    plan = base._read_json(directory / "frozen_plan.json")
    if [str(p.resolve()) for p in args.configs] != [j["config_path"] for j in plan["jobs"]]:
        raise ValueError("Performance continuation must preserve the original configs and queue order")
    if not args.execute:
        print(json.dumps({"preview_only": True, "execution_profile": binding, "original_plan": str(directory / "frozen_plan.json")})); return
    if not args.resume_controller:
        parser.error("Loader migration requires explicit --resume-controller")
    with (directory / ".controller.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = base._read_json(directory / "STATUS.json")
        if base._running_container(previous.get("active_container")) or (directory / "STOP").exists():
            raise RuntimeError("Prior container active or controller STOP has not been explicitly archived")
        running = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True, timeout=20)
        if any(name.startswith("cxr-e2-") for name in running.stdout.splitlines()):
            raise RuntimeError("An E2 container is already running; refusing duplicate execution")
        base.verify_plan(plan)
        path = directory / "performance_plan.json"
        performance = base._read_json(path) if path.exists() else prepare_performance_plan(directory, plan, binding)
        if performance["execution_profile"] != binding:
            raise ValueError("An existing performance plan uses a different execution profile")
        verify_performance_plan(plan, performance)
        controller = PerformanceController(directory, plan, performance, resume_runs=args.resume_run)
        handlers = {sig: signal.signal(sig, controller.signal_stop) for sig in [signal.SIGTERM, signal.SIGINT]}
        try:
            result = controller.run()
        except BaseException as error:
            controller.publish("failed", error=f"{type(error).__name__}: {error}"); raise
        finally:
            for sig, handler in handlers.items(): signal.signal(sig, handler)
        if result["state"] != "completed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
