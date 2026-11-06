"""Explicit E0 -> E1 research queue; no confirmation/test/deployment actions.

Run on the host. Each trainer runs in the pinned runtime, one at a time. STOP
requests an epoch-boundary pause and waits for the current container to exit.
Restarting this controller never silently resumes or adopts a training run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from reconstruction import runtime
from training_compliance import read_events

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT.parent / "reconstruction_runs_20261103"
SUCCESS_STATES = {"completed", "early_stopped"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))


def _history(path, *, strict=True):
    path = Path(path)
    if not path.exists() and not strict:
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if strict:
            raise ValueError("Incomplete history tail; cannot certify completion")
        data = data.rsplit(b"\n", 1)[0] if b"\n" in data else b""
    return [json.loads(row, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite history: {x}")))
            for row in data.splitlines() if row]


def _finite_auc(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be inside isolated research output: {path}")
    return path


def source_files():
    return [ROOT / "reconstruction" / name for name in
        ("__init__.py", "train.py", "data.py", "model.py", "evaluate.py", "provenance.py",
         "runtime.py", "splits.py", "controller.py")] + [ROOT / name for name in
        ("model_overnight.py", "runtime_compat.py", "training_hooks.py", "training_compliance.py", "audit_trail.py")]


def build_plan(config_paths, controller_dir):
    """Inspect only explicit new configs and their registered train/dev inputs."""
    controller_dir = _inside(controller_dir, RUN_ROOT)
    if len(config_paths) != 2:
        raise ValueError("Register exactly E0 mean and then E1 attention configs")
    jobs = []
    for index, supplied in enumerate(config_paths):
        path = Path(supplied).resolve(strict=True)
        config = _read_json(path)
        expected = "mean" if index == 0 else "attention"
        if config.get("aggregation") != expected or config.get("use_local") is not False:
            raise ValueError("This queue only runs global E0 mean followed by global E1 attention")
        if config.get("smoke_studies") != 0:
            raise ValueError("Mechanics-only smoke runs cannot enter the research queue")
        output = _inside(config["output"], RUN_ROOT)
        if output == controller_dir or output in controller_dir.parents or controller_dir in output.parents:
            raise ValueError("Controller evidence and training outputs must be separate directories")
        split_plan = Path(config["split_plan"]).resolve(strict=True)
        role_plan = _read_json(split_plan)
        inputs = {str(split_plan): sha(split_plan)}
        for role, name in [("train", "train_manifest"), ("dev", "dev_manifest")]:
            actual = Path(config[name]).resolve(strict=True)
            declared = role_plan["roles"][role]
            if actual != Path(declared["path"]).resolve() or sha(actual) != declared["sha256"]:
                raise ValueError(f"Config is not bound to its registered {role} manifest")
            inputs[str(actual)] = declared["sha256"]
        jobs.append({"name": f"E{index}", "config_path": str(path), "config_sha256": sha(path),
                     "config": config, "output": str(output), "inputs_sha256": inputs})
    if jobs[0]["output"] == jobs[1]["output"]:
        raise ValueError("E0 and E1 require distinct output directories")
    keys = set(jobs[0]["config"]) | set(jobs[1]["config"])
    for key in sorted(keys - {"aggregation", "output"}):
        if jobs[0]["config"].get(key) != jobs[1]["config"].get(key):
            raise ValueError(f"Matched E0/E1 comparison differs at {key}")
    return {"schema": "explicit_reconstruction_e0_e1_queue_v1", "created_at": now(),
            "controller_dir": str(controller_dir), "jobs": jobs,
            "source_sha256": {str(p.resolve()): sha(p) for p in source_files()},
            "runtime_image": runtime.IMAGE, "confirmation_allowed": False,
            "official_valid_allowed": False, "test_allowed": False, "deployment_allowed": False}


def verify_plan(plan):
    for path, expected in plan["source_sha256"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if plan["runtime_image"] != runtime.IMAGE:
        raise ValueError("Pinned runtime changed")
    for job in plan["jobs"]:
        if sha(job["config_path"]) != job["config_sha256"]:
            raise ValueError(f"Registered config changed: {job['name']}")
        for path, expected in job["inputs_sha256"].items():
            if sha(path) != expected:
                raise ValueError(f"Registered train/dev input changed: {path}")


def _closed_sessions(run_dir):
    """Validate hash chain and require a genuine end record for every invocation."""
    run_dir = Path(run_dir).resolve()
    events = read_events(run_dir / "compliance/events.jsonl")
    if not events:
        raise ValueError("Missing training lifecycle events")
    sessions = {}
    for event in events:
        if event.get("demonstration_only") or event.get("trainer") != "reconstruction.train":
            raise ValueError("Lifecycle belongs to a demonstration or another trainer")
        if Path(event.get("run_dir", "")).resolve() != run_dir:
            raise ValueError("Lifecycle run identity differs")
        sessions.setdefault(event["session_id"], []).append(event)
    for name, rows in sessions.items():
        starts = [r for r in rows if r["event_type"] == "training_started"]
        ends = [r for r in rows if r["event_type"] == "training_ended"]
        if len(starts) != 1 or len(ends) != 1 or rows[0] != starts[0] or rows[-1] != ends[0]:
            raise ValueError(f"Unclosed or malformed training session: {name}")
        if starts[0].get("previous_unclosed_sessions"):
            raise ValueError("Run resumed after an unclosed session; explicit evidence repair is required")
        evidence = run_dir / "compliance" / starts[0]["session_evidence"]
        for field, filename in [("configuration_sha256", "configuration.json"),
                                ("source_manifest_sha256", "source_manifest.json"),
                                ("environment_sha256", "environment.json"), ("git_sha256", "git.json")]:
            if sha(evidence / filename) != starts[0][field]:
                raise ValueError(f"Lifecycle evidence hash differs: {name}/{filename}")
        if _read_json(evidence / "end.json") != ends[0]:
            raise ValueError("Session end file differs from authoritative end event")
        for archived in _read_json(evidence / "source_manifest.json"):
            snapshot = (run_dir / "compliance" / archived["archived_path"]).resolve()
            if (run_dir / "compliance").resolve() not in snapshot.parents or sha(snapshot) != archived["sha256"]:
                raise ValueError("Archived training source differs from its bound hash")
    compliance = _read_json(run_dir / "compliance/status.json")
    if (events[-1]["event_type"] != "training_ended"
            or compliance.get("last_event_sha256") != events[-1]["record_sha256"]
            or compliance.get("state") != events[-1]["state"]):
        raise ValueError("Latest compliance status does not match the final end event")
    return events


def verify_success(job, *, allowed_states=SUCCESS_STATES):
    """A zero process exit alone is never treated as a successful experiment."""
    output = Path(job["output"])
    status = _read_json(output / "status.json")
    if status.get("state") not in allowed_states:
        raise ValueError(f"Trainer did not finish successfully: {status.get('state')}")
    if status.get("confirmation_evaluated") is not False or status.get("official_valid_evaluated") is not False:
        raise ValueError("Training status lacks explicit isolation declarations")
    evidence = _read_json(output / "configuration.json")
    if any(evidence.get(k) != v for k, v in job["config"].items()):
        raise ValueError("Actual training configuration differs from registered config")
    for relative, expected in evidence.get("source_sha256", {}).items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Actual training source differs from current frozen queue source")
    history = _history(output / "history.jsonl")
    if not history or [r.get("epoch") for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Training history has missing or duplicate epochs")
    if any(not _finite_auc(r.get("macro_auc")) for r in history):
        raise ValueError("Missing or nonfinite inner-development AUC in run history")
    if status["state"] == "completed" and len(history) != job["config"]["epochs"]:
        raise ValueError("Completed state lacks the registered epoch budget")
    best_epoch = status.get("best_epoch")
    if type(best_epoch) is not int or best_epoch < 1 or best_epoch > len(history):
        raise ValueError("Invalid best epoch")
    best_auc = status.get("best_macro_auc")
    if not _finite_auc(best_auc) or best_auc != max(r["macro_auc"] for r in history):
        raise ValueError("Best AUC is not supported by complete epoch history")
    if history[best_epoch - 1]["macro_auc"] != best_auc:
        raise ValueError("Best epoch and AUC disagree")
    events = _closed_sessions(output)
    if events[-1]["state"] != status["state"]:
        raise ValueError("Trainer and compliance terminal states disagree")
    epoch_events = [e for e in events if e["event_type"] == "epoch_completed"]
    if len(epoch_events) != len(history):
        raise ValueError("History is not covered by exactly one lifecycle event per epoch")
    for row, event in zip(history, epoch_events):
        if row != event.get("metrics") or row["epoch"] != event.get("epoch"):
            raise ValueError("Epoch history differs from its lifecycle event")
    for name, epoch, role in [("last.pt", len(history), "last_full_state"),
                              ("best.pt", best_epoch, "best_inner_ema")]:
        candidates = [e for e in events if e["event_type"] == "checkpoint_saved"
                      and e.get("role") == role and e.get("epoch") == epoch]
        if not candidates or Path(candidates[-1]["checkpoint_path"]).resolve() != (output / name).resolve():
            raise ValueError(f"Missing bound checkpoint event for {name}")
        if sha(output / name) != candidates[-1]["checkpoint_sha256"]:
            raise ValueError(f"Actual checkpoint hash differs: {name}")
    folder = output / f"epoch_{best_epoch:02d}"
    report = _read_json(folder / "report.json")
    split = _read_json(job["config"]["split_plan"])
    if (report.get("scope") != "inner_validation" or report.get("macro_auc") != best_auc
            or report.get("num_studies") != split["roles"]["dev"]["studies"]
            or any(report.get(k) is not False for k in ["test_accessed", "official_validation_accessed", "holdout_H_accessed"])
            or len(report.get("per_class_auc", [])) != 10
            or not all(_finite_auc(v) for v in report["per_class_auc"])):
        raise ValueError("Best-epoch report lacks complete, finite internal-development evidence")
    if sha(folder / "predictions.npz") != report.get("predictions_sha256"):
        raise ValueError("Best-epoch predictions differ from their report hash")
    return {"name": job["name"], "state": status["state"], "epochs": len(history),
            "best_epoch": best_epoch, "best_macro_auc": best_auc,
            "prediction_path": str(folder / "predictions.npz"),
            "prediction_sha256": report["predictions_sha256"],
            "report_sha256": sha(folder / "report.json"), "ledger_sha256": sha(output / "compliance/events.jsonl"),
            "history_sha256": sha(output / "history.jsonl"), "verified_at": now()}


def verify_resume(job):
    """Accept an explicit resume only with closed, truthful prior lifecycle evidence."""
    output = Path(job["output"])
    status = _read_json(output / "status.json")
    if status.get("state") not in {"paused", "failed", "interrupted"}:
        raise ValueError("Explicit resume needs paused/failed/interrupted new-run state")
    events = _closed_sessions(output)
    if events[-1]["state"] != status["state"]:
        raise ValueError("Resume status and lifecycle disagree")
    if any(_read_json(output / "configuration.json").get(k) != v for k, v in job["config"].items()):
        raise ValueError("Resume configuration differs from registered new run")
    checkpoints = [e for e in events if e["event_type"] == "checkpoint_saved" and e.get("role") == "last_full_state"]
    if not checkpoints or sha(output / "last.pt") != checkpoints[-1]["checkpoint_sha256"]:
        raise ValueError("Resume requires a verified complete new-run checkpoint")
    history = _history(output / "history.jsonl")
    if not history or history[-1].get("epoch") != checkpoints[-1]["epoch"]:
        raise ValueError("Resume checkpoint and complete history disagree")


def _running_container(name):
    if not name:
        return False
    check = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                           capture_output=True, text=True, timeout=15)
    if check.returncode == 0:
        return check.stdout.strip() == "true"
    if "No such" in check.stderr or "not found" in check.stderr.lower():
        return False
    raise RuntimeError(f"Cannot verify whether prior container is still active: {check.stderr.strip()[:300]}")


class Controller:
    def __init__(self, directory, plan, *, resume_runs=(), poll_seconds=30):
        self.directory, self.plan = Path(directory), plan
        self.resume_runs = {str(Path(p).resolve()) for p in resume_runs}
        if not self.resume_runs <= {j["config_path"] for j in plan["jobs"]}:
            raise ValueError("Resume request refers to a config outside this explicit queue")
        self.poll_seconds = poll_seconds
        self.stop_reason = None
        self.child = None
        self.active_job = None
        self.state = {"state": "starting", "started_at": now(), "controller_pid": os.getpid(),
                      "completed_jobs": [], "current_job": None, "active_container": None,
                      "test_accessed": False, "confirmation_evaluated": False, "official_valid_evaluated": False}

    def signal_stop(self, signum, _frame):
        self.stop_reason = f"signal_{signal.Signals(signum).name}"

    def stop_requested(self):
        return self.stop_reason or ((self.directory / "STOP").exists() and "controller_STOP")

    def publish(self, state=None, **updates):
        if state:
            self.state["state"] = state
        self.state.update(updates, updated_at=now())
        active = self.active_job
        history = []
        if active:
            output = Path(active["output"])
            monitoring_errors = []
            try:
                if (output / "progress.json").is_file():
                    self.state["trainer_progress"] = _read_json(output / "progress.json")
            except (OSError, ValueError) as error:
                monitoring_errors.append(f"progress: {error}")
            try:
                history = _history(output / "history.jsonl", strict=False)
                if any(not _finite_auc(row.get("macro_auc")) for row in history):
                    monitoring_errors.append("Nonfinite or missing inner-development AUC appeared in history")
            except (OSError, ValueError) as error:
                monitoring_errors.append(f"history: {error}")
            self.state["monitor_error"] = "; ".join(monitoring_errors) if monitoring_errors else None
            self.state["actual_epoch_auc"] = [{"epoch": row["epoch"], "macro_auc": row.get("macro_auc")}
                                                for row in history]
            self.state.setdefault("job_histories", {})[active["name"]] = self.state["actual_epoch_auc"]
            progress = self.state.get("trainer_progress", {})
            estimate = progress.get("estimated_epoch_seconds")
            elapsed = progress.get("elapsed_seconds")
            self.state["current_epoch_eta"] = (datetime.now(ZoneInfo("Asia/Hong_Kong")) +
                timedelta(seconds=max(0, estimate - elapsed))).isoformat() if (
                    type(estimate) in (int, float) and type(elapsed) in (int, float)
                    and math.isfinite(estimate) and math.isfinite(elapsed) and progress.get("state") == "training") else None
        write_json(self.directory / "STATUS.json", self.state)
        lines = ["# Reconstruction E0 / E1 progress", "", f"Updated (Hong Kong): {self.state['updated_at']}", "",
                 f"State: **{self.state['state']}**; current job: {self.state.get('current_job') or 'none'}.",
                 f"Current epoch ETA: {self.state.get('current_epoch_eta') or 'unavailable'}.", "",
                 "ETA extrapolates the current trainer epoch only; early stopping and later jobs are not predicted.", "",
                 "| Job | Completed epoch | Inner-development Macro-AUC |", "| --- | --- | ---: |"]
        lines += [f"| {job} | {r['epoch']} | {r.get('macro_auc')} |"
                  for job, rows in self.state.get("job_histories", {}).items() for r in rows]
        if self.state.get("error"):
            lines += ["", f"Error: {self.state['error']}"]
        lines += ["", "H, official validation, test and deployment remain unopened.", ""]
        target = self.directory / "PROGRESS.md"
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, target)

    def request_current_stop(self, reason):
        if self.active_job:
            stop = Path(self.active_job["output"]) / "STOP"
            stop.parent.mkdir(parents=True, exist_ok=True)
            if not stop.exists():
                with stop.open("x") as stream:
                    stream.write(f"Controller requests epoch-boundary stop: {reason}; {now()}\n")
        self.publish("stopping", stop_reason=reason)

    def wait_child(self):
        while self.child.poll() is None:
            reason = self.stop_requested()
            if reason:
                if self.active_job:
                    self.request_current_stop(reason)
                else:
                    # Summarization is CPU-only and has no training checkpoint to preserve.
                    self.child.terminate()
            else:
                verify_plan(self.plan)
                self.publish("training" if self.active_job else "summarizing")
                if self.state.get("monitor_error"):
                    raise ValueError(self.state["monitor_error"])
            try:
                return self.child.wait(timeout=self.poll_seconds)
            except subprocess.TimeoutExpired:
                pass
        return self.child.wait()

    def launch(self, arguments, *, log_name, gpu, container_name):
        log_path = self.directory / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = runtime.command(arguments, gpu=gpu, name=container_name)
        self.publish(active_container=container_name, command=command, log_path=str(log_path))
        with log_path.open("xb") as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            self.publish(child_pid=self.child.pid)
            try:
                exit_code = self.wait_child()
            except BaseException as error:
                self.stop_reason = self.stop_reason or f"controller_error:{type(error).__name__}"
                if self.active_job and self.child.poll() is None:
                    self.request_current_stop(self.stop_reason)
                    # Preserve the failure as primary while waiting for the actual job.
                    while self.child.poll() is None:
                        try:
                            self.child.wait(timeout=self.poll_seconds)
                        except subprocess.TimeoutExpired:
                            self.publish("stopping", error=f"{type(error).__name__}: {error}")
                elif self.child.poll() is None:
                    self.child.terminate()
                    self.child.wait(timeout=60)
                self.wait_container_closed(container_name, self.child.returncode)
                raise
            finally:
                log.flush()
                os.fsync(log.fileno())
        self.child = None
        self.wait_container_closed(container_name, exit_code)
        self.publish(child_pid=None, active_container=None, last_exit_code=exit_code)
        return exit_code

    def wait_container_closed(self, container_name, exit_code):
        if _running_container(container_name):
            # Docker client exit does not prove the engine-side process exited.
            # Keep its identity visible and request a safe stop before waiting.
            if self.active_job:
                self.stop_reason = self.stop_reason or "docker_client_exited_while_container_running"
                self.request_current_stop(self.stop_reason)
            else:
                subprocess.run(["docker", "stop", "--time", "10", container_name],
                               capture_output=True, text=True, timeout=30, check=True)
            while _running_container(container_name):
                self.publish("stopping", docker_client_exit_code=exit_code)
                time.sleep(self.poll_seconds)

    def run(self):
        verified = []
        for job in self.plan["jobs"]:
            verify_plan(self.plan)
            if self.stop_requested():
                self.publish("paused", stop_reason=self.stop_requested())
                return self.state
            self.active_job = job
            self.publish("checking", current_job=job["name"], trainer_progress={}, actual_epoch_auc=[])
            output = Path(job["output"])
            existing = output.exists() and any(output.iterdir())
            if existing and (output / "status.json").is_file() and _read_json(output / "status.json").get("state") in SUCCESS_STATES:
                result = verify_success(job)
            else:
                resume = job["config_path"] in self.resume_runs
                if existing and not resume:
                    raise ValueError(f"Existing unfinished output requires explicit --resume-run: {output}")
                if resume:
                    verify_resume(job)
                    if (output / "STOP").exists():
                        # Preserve the exact historical request; explicit resume authorizes clearing it.
                        archive = self.directory / "resume_stop_markers"
                        archive.mkdir(exist_ok=True)
                        os.replace(output / "STOP", archive / f"{job['name']}_{time.time_ns()}.STOP")
                stamp = f"{int(time.time())}_{os.getpid()}"
                arguments = ["-m", "reconstruction.train", "--config", job["config_path"], "--execute"]
                if resume:
                    arguments.append("--resume")
                self.publish("training")
                exit_code = self.launch(arguments, log_name=f"{job['name']}_{stamp}.log", gpu=True,
                                        container_name=f"cxr-reconstruct-{job['name'].lower()}-{stamp}")
                if self.stop_requested():
                    self.publish("paused", stop_reason=self.stop_requested(), trainer_exit_code=exit_code)
                    return self.state
                if exit_code:
                    raise RuntimeError(f"{job['name']} trainer exited {exit_code}; see its preserved log")
                result = verify_success(job)
            verified.append(result)
            write_json(self.directory / f"{job['name']}_verified.json", result)
            self.publish("verified", completed_jobs=list(verified))
            self.active_job = None
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested())
            return self.state
        verify_plan(self.plan)
        self.publish("summarizing", current_job="paired_comparison", trainer_progress={}, actual_epoch_auc=[])
        stamp = f"{int(time.time())}_{os.getpid()}"
        code = self.launch(["-m", "reconstruction.controller", "--controller-dir", str(self.directory), "--summarize"],
                           log_name=f"comparison_{stamp}.log", gpu=False,
                           container_name=f"cxr-reconstruct-compare-{stamp}")
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested())
            return self.state
        if code:
            raise RuntimeError(f"CPU comparison exited {code}")
        summary = _read_json(self.directory / "comparison/report.json")
        if not summary.get("complete"):
            raise ValueError("Comparison did not produce complete paired evidence")
        self.publish("completed", current_job=None, comparison_report=str(self.directory / "comparison/report.json"))
        return self.state


def run_controller(directory, config_paths, *, resume_controller=False, resume_runs=(), poll_seconds=30):
    directory = _inside(directory, RUN_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".controller.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller already owns this explicit queue") from None
        plan_file = directory / "frozen_plan.json"
        if resume_controller:
            plan = _read_json(plan_file)
            if [str(Path(p).resolve()) for p in config_paths] != [j["config_path"] for j in plan["jobs"]]:
                raise ValueError("Controller resume must name the original configs in their original order")
            previous = _read_json(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {}
            if _running_container(previous.get("active_container")):
                raise RuntimeError("Prior recorded container is still active; refusing duplicate launch or silent adoption")
            verify_plan(plan)
            if (directory / "STOP").exists():
                raise ValueError("Controller STOP remains present; explicitly archive/remove it before resuming")
        else:
            if any(p.name != ".controller.lock" for p in directory.iterdir()):
                raise FileExistsError("Controller output already contains evidence; use explicit --resume-controller")
            plan = build_plan(config_paths, directory)
            for job in plan["jobs"]:
                output = Path(job["output"])
                if output.exists() and any(output.iterdir()) and job["config_path"] not in {str(Path(p).resolve()) for p in resume_runs}:
                    raise FileExistsError(f"Fresh queue requires fresh outputs: {output}")
            write_json(plan_file, plan)
        controller = Controller(directory, plan, resume_runs=resume_runs, poll_seconds=poll_seconds)
        previous_handlers = {sig: signal.signal(sig, controller.signal_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            return controller.run()
        except BaseException as error:
            controller.publish("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def summarize(directory):
    """CPU official-runtime comparison of E1 versus E0; no fresh inference."""
    import numpy as np
    from reconstruction.evaluate import save_evaluation

    directory = Path(directory)
    plan = _read_json(directory / "frozen_plan.json")
    verify_plan(plan)
    certificates = [verify_success(job) for job in plan["jobs"]]
    loaded = []
    for result in certificates:
        with np.load(result["prediction_path"], allow_pickle=False) as data:
            loaded.append({k: data[k].copy() for k in data.files})
    baseline, candidate = loaded
    if len(set(candidate["study_keys"].tolist())) != len(candidate["study_keys"]):
        raise ValueError("Duplicate candidate Study predictions")
    index = {key: i for i, key in enumerate(candidate["study_keys"].tolist())}
    if set(index) != set(baseline["study_keys"].tolist()):
        raise ValueError("Paired comparison lacks identical Study coverage")
    order = np.asarray([index[k] for k in baseline["study_keys"].tolist()])
    for name in ("targets", "patient_ids", "views_json", "image_counts"):
        if not np.array_equal(baseline[name], candidate[name][order]):
            raise ValueError(f"Paired Study metadata mismatch: {name}")
    comparison = directory / "comparison"
    report = save_evaluation(comparison, candidate["probabilities"][order], baseline["targets"],
        baseline["study_keys"], baseline["patient_ids"], [json.loads(v) for v in baseline["views_json"]],
        reference_probabilities=baseline["probabilities"], bootstrap_repeats=1000, seed=20261103,
        scope="inner_validation", metadata={"candidate": certificates[1], "reference": certificates[0],
            "selection_caveat": "Both checkpoints selected on this inner development split; intervals do not remove selection bias"})
    interval = report["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"]
    if interval is None:
        decision = "insufficient_class_support_for_paired_interval"
    elif interval[0] > 0:
        decision = "E1_improves_selected_inner_development_predictions; independent_confirmation_not_opened"
    elif interval[1] < 0:
        decision = "E1_worse_on_selected_inner_development_predictions; retain_E0_reference"
    else:
        decision = "difference_inconclusive_on_inner_development; retain_both_evidence_sets"
    report.update(complete=True, decision=decision, selection_bias_corrected=False,
                  confirmation_authorized=False, deployment_selected=False)
    write_json(comparison / "report.json", report)
    lines = ["# E1 versus E0: internal development comparison", "",
             "Both checkpoints were selected on this same split. Patient intervals are exploratory and do not remove selection bias.", "",
             "| Slice | E0 Macro-AUC | E1 Macro-AUC | E1−E0 | Patient 95% CI |", "| --- | ---: | ---: | ---: | --- |"]
    for name, result in report["slices"].items():
        ci = result.get("patient_bootstrap", {}).get("paired_delta_macro_auc", {}).get("ci95")
        lines.append(f"| {name} | {result['reference']['macro_auc']} | {result['macro_auc']} | {result['macro_auc_delta']} | {ci} |")
    lines += ["", f"Decision: {decision}", "", "No H, official-validation, test, CSV, submission or deployment action was performed.", ""]
    (comparison / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--resume-run", action="append", default=[], type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        if args.execute or args.resume_controller or args.resume_run:
            parser.error("Summarization cannot launch or resume trainers")
        result = summarize(args.controller_dir)
        print(json.dumps({"complete": result["complete"], "decision": result["decision"]}), flush=True)
        return
    if not args.configs:
        parser.error("Explicit --configs E0.json E1.json are required")
    if not args.execute:
        plan = build_plan(args.configs, args.controller_dir)
        print(json.dumps({"preview_only": True, "jobs": plan["jobs"], "runtime_image": plan["runtime_image"]}, indent=2))
        return
    result = run_controller(args.controller_dir, args.configs,
        resume_controller=args.resume_controller, resume_runs=args.resume_run)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["state"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
