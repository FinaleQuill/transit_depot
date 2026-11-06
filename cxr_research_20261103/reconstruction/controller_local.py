"""Explicit E2 global continuation -> global/local -> paired CPU review queue.

Never resumes E0/E1, opens H/test, retries failed jobs, or expands to E3/E4.
Epoch zero is an evaluated parent baseline and may remain the selected best.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from reconstruction import controller as base
from reconstruction import runtime_local
from training_compliance import read_events

ROOT, RUN_ROOT = base.ROOT, base.RUN_ROOT


def source_files():
    names = ["__init__.py", "train.py", "train_local.py", "model.py", "local_model.py",
             "data.py", "local_data.py", "local_compliance.py", "evaluate.py", "diagnostics.py", "provenance.py",
             "splits.py", "controller.py", "controller_local.py", "runtime.py", "runtime_local.py"]
    return [ROOT / "reconstruction" / n for n in names] + [ROOT / n for n in
            ["training_hooks.py", "training_compliance.py", "audit_trail.py", "model_overnight.py", "runtime_compat.py"]]


def build_plan(configs):
    if len(configs) != 2:
        raise ValueError("Exactly a global control and a local candidate are required")
    jobs = []
    for path in configs:
        path = Path(path).resolve(strict=True)
        config = base._read_json(path)
        if config["smoke_studies"] or config["smoke_high_view"] or not config["fixed_budget"]:
            raise ValueError("Only full fixed-budget research jobs may enter this queue")
        if (config["epochs"] != 6 or config["eval_epoch_zero"] is not True
                or config["head_warmup_epochs"] != 0 or config["eval_pack_images"] != 1
                or config["global_aux_weight"] != 0):
            raise ValueError("Configuration violates the registered E2 comparison")
        output = base._inside(config["output"], RUN_ROOT / "E2_20261104")
        inputs = [config[k] for k in ["train_manifest", "dev_manifest", "split_plan", "provenance_path", "parent_controller_plan"]]
        inputs.append(str(runtime_local.ADDON_MANIFEST))
        jobs.append({"name": config["experiment_arm"], "config_path": str(path),
                     "config_sha256": base.sha(path), "config": config, "output": str(output),
                     "inputs_sha256": {str(Path(p).resolve()): base.sha(p) for p in inputs}})
    a, b = [j["config"] for j in jobs]
    if a["experiment_arm"] != "global_control" or b["experiment_arm"] != "global_local":
        raise ValueError("E2 queue order must be global_control then global_local")
    allowed = {"output", "experiment_arm", "use_local", "reference_epoch_zero"}
    if set(a) != set(b) or any(a[k] != b[k] for k in set(a) - allowed):
        raise ValueError("Matched arms differ outside the experimental intervention")
    if (a["use_local"] is not False or b["use_local"] is not True or a["reference_epoch_zero"] is not None
            or Path(b["reference_epoch_zero"]).resolve() != Path(a["output"]) / "epoch_00"):
        raise ValueError("Local/control inputs or epoch-zero reference differ")
    if jobs[0]["output"] == jobs[1]["output"]:
        raise ValueError("Arms must have separate outputs")
    return {"schema": "E2_matched_global_local_v1", "created_at": base.now(), "jobs": jobs,
            "runtime_image": runtime_local.IMAGE,
            "source_sha256": {str(p): base.sha(p) for p in source_files()},
            "selection": "each arm best EMA among epoch 0 and 1..6; fixed attempted-update budget",
            "later_experiments_authorized_for_this_queue": [], "holdout_H_accessed": False,
            "official_validation_accessed": False, "test_accessed": False}


def closed_sessions(output):
    output = Path(output).resolve()
    events = read_events(output / "compliance/events.jsonl")
    if not events:
        raise ValueError("Missing training lifecycle")
    sessions = {}
    for event in events:
        if (event.get("trainer") != "reconstruction.train_local" or event.get("demonstration_only")
                or Path(event.get("run_dir", "")).resolve() != output):
            raise ValueError("Lifecycle identity differs")
        sessions.setdefault(event["session_id"], []).append(event)
    for rows in sessions.values():
        starts = [e for e in rows if e["event_type"] == "training_started"]
        ends = [e for e in rows if e["event_type"] == "training_ended"]
        if (len(starts) != 1 or len(ends) != 1 or rows[0] != starts[0] or rows[-1] != ends[0]
                or starts[0].get("previous_unclosed_sessions")):
            raise ValueError("Incomplete or malformed lifecycle session")
        evidence = output / "compliance" / starts[0]["session_evidence"]
        for field, filename in [("configuration_sha256", "configuration.json"),
                                ("source_manifest_sha256", "source_manifest.json"),
                                ("environment_sha256", "environment.json"), ("git_sha256", "git.json")]:
            if base.sha(evidence / filename) != starts[0][field]:
                raise ValueError("Lifecycle evidence changed")
        if base._read_json(evidence / "end.json") != ends[0]:
            raise ValueError("Lifecycle end evidence differs")
        for item in base._read_json(evidence / "source_manifest.json"):
            snapshot = (output / "compliance" / item["archived_path"]).resolve()
            if (output / "compliance").resolve() not in snapshot.parents or base.sha(snapshot) != item["sha256"]:
                raise ValueError("Archived training source differs")
    status = base._read_json(output / "compliance/status.json")
    if (events[-1]["event_type"] != "training_ended" or status.get("last_event_sha256") != events[-1]["record_sha256"]
            or status["state"] != events[-1]["state"]):
        raise ValueError("Lifecycle status does not match its end event")
    return events


def selection_from_history(history, baseline, *, epochs, require_complete):
    if [r.get("epoch") for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("History contains missing or duplicate epochs")
    if len(history) > epochs or (require_complete and len(history) != epochs):
        raise ValueError("Fixed epoch budget has not completed")
    points = [{"epoch": 0, "macro_auc": baseline["macro_auc"]}, *history]
    if any(not base._finite_auc(r.get("macro_auc")) for r in points):
        raise ValueError("Nonfinite full-development selection metric")
    return max(points, key=lambda r: r["macro_auc"])


def verify_run(job, *, require_complete=True):
    output = Path(job["output"])
    config = job["config"]
    status = base._read_json(output / "status.json")
    allowed = {"completed"} if require_complete else {"paused", "failed", "interrupted"}
    if status["state"] not in allowed:
        raise ValueError(f"Unacceptable trainer state: {status['state']}")
    if require_complete and any(status.get(k) is not False for k in ["confirmation_evaluated", "official_valid_evaluated"]):
        raise ValueError("Missing training isolation declaration")
    evidence = base._read_json(output / "configuration.json")
    if any(evidence.get(k) != v for k, v in config.items()):
        raise ValueError("Actual configuration differs from registered arm")
    for relative, expected in evidence["source_sha256"].items():
        if base.sha(ROOT / relative) != expected:
            raise ValueError("Training source changed since registration")
    events = closed_sessions(output)
    if events[-1]["state"] != status["state"]:
        raise ValueError("Trainer and lifecycle states disagree")
    history = base._history(output / "history.jsonl", strict=require_complete)
    baseline = base._read_json(output / "epoch_00/report.json")
    best = selection_from_history(history, baseline, epochs=config["epochs"], require_complete=require_complete)
    if require_complete and (status.get("best_epoch") != best["epoch"] or status.get("best_macro_auc") != best["macro_auc"]):
        raise ValueError("Final best does not match epoch-zero-plus-training selection")
    epoch_events = [e for e in events if e["event_type"] == "epoch_completed"]
    if [e["metrics"] for e in epoch_events] != history:
        raise ValueError("Complete history differs from lifecycle events")
    split = base._read_json(config["split_plan"])
    for row in history:
        if (row["studies"] != split["roles"]["train"]["studies"]
                or row["images"] != split["roles"]["train"]["images"]
                or row["attempted_updates"] != row["epoch"] * math.ceil(row["studies"] / config["effective_studies"])):
            raise ValueError("Full Study coverage or matched attempted-update budget differs")
    checkpoint_hashes = {}
    for filename, epoch, role in [("best.pt", best["epoch"], "best_inner_ema"),
                                  ("last.pt", len(history), "last_full_state")]:
        saved = [e for e in events if e["event_type"] == "checkpoint_saved"
                 and e.get("epoch") == epoch and e.get("role") == role]
        actual = base.sha(output / filename)
        if (not saved or Path(saved[-1]["checkpoint_path"]).resolve() != output / filename
                or saved[-1]["checkpoint_sha256"] != actual):
            raise ValueError(f"Checkpoint identity differs: {filename}")
        if epoch == 0 and saved[-1].get("stage") != "pre_training_baseline":
            raise ValueError("Epoch zero must be recorded explicitly as an untrained baseline")
        checkpoint_hashes[filename] = actual
    for epoch in {0, best["epoch"]}:
        folder = output / f"epoch_{epoch:02d}"
        report = base._read_json(folder / "report.json")
        if (report.get("scope") != "inner_validation" or report.get("num_studies") != split["roles"]["dev"]["studies"]
                or report.get("metadata", {}).get("epoch") != epoch
                or any(report.get(k) is not False for k in ["test_accessed", "official_validation_accessed", "holdout_H_accessed"])
                or len(report.get("per_class_auc", [])) != 10
                or not all(base._finite_auc(x) for x in report["per_class_auc"])
                or base.sha(folder / "predictions.npz") != report["predictions_sha256"]):
            raise ValueError("Evaluation lacks complete bound development evidence")
        if epoch == best["epoch"] and report["macro_auc"] != best["macro_auc"]:
            raise ValueError("Best report and history disagree")
    return {"name": job["name"], "state": status["state"], "epochs": len(history),
            "best_epoch": best["epoch"], "best_macro_auc": best["macro_auc"],
            "baseline_macro_auc": baseline["macro_auc"], "checkpoint_sha256": checkpoint_hashes,
            "history_sha256": base.sha(output / "history.jsonl"),
            "ledger_sha256": base.sha(output / "compliance/events.jsonl"),
            "prediction_path": str(output / f"epoch_{best['epoch']:02d}" / "predictions.npz"),
            "verified_at": base.now()}


class LocalController(base.Controller):
    def publish(self, state=None, **updates):
        super().publish(state, **updates)
        path = self.directory / "PROGRESS.md"
        text = path.read_text().replace("Reconstruction E0 / E1 progress", "E2 global / local progress")
        text = text.replace("early stopping and later jobs are not predicted", "validation and later jobs are not predicted")
        path.write_text(text)

    def launch(self, arguments, *, log_name, gpu, container_name):
        log_path = self.directory / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = runtime_local.command(arguments, gpu=gpu, name=container_name)
        self.publish(active_container=container_name, command=command, log_path=str(log_path))
        with log_path.open("xb") as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.publish(child_pid=self.child.pid)
            try:
                exit_code = self.wait_child()
            except BaseException as error:
                self.stop_reason = self.stop_reason or f"controller_error:{type(error).__name__}"
                if self.active_job and self.child.poll() is None:
                    self.request_current_stop(self.stop_reason)
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

    def run(self):
        verified = []
        for job in self.plan["jobs"]:
            base.verify_plan(self.plan)
            if self.stop_requested():
                self.publish("paused", stop_reason=self.stop_requested())
                return self.state
            self.active_job = job
            self.publish("checking", current_job=job["name"], trainer_progress={}, actual_epoch_auc=[])
            output = Path(job["output"])
            existing = output.exists() and any(output.iterdir())
            if existing and (output / "status.json").exists() and base._read_json(output / "status.json")["state"] == "completed":
                result = verify_run(job)
            else:
                resume = job["config_path"] in self.resume_runs
                if existing and not resume:
                    raise ValueError("Existing unfinished output needs an explicit --resume-run")
                if resume:
                    verify_run(job, require_complete=False)
                    if (output / "STOP").exists():
                        archive = self.directory / "resume_stop_markers"
                        archive.mkdir(exist_ok=True)
                        os.replace(output / "STOP", archive / f"{job['name']}_{time.time_ns()}.STOP")
                stamp = f"{int(time.time())}_{os.getpid()}"
                args = ["-m", "reconstruction.train_local", "--config", job["config_path"], "--execute"]
                if resume:
                    args.append("--resume")
                code = self.launch(args, log_name=f"{job['name']}_{stamp}.log", gpu=True,
                                   container_name=f"cxr-e2-{job['name'].replace('_','-')}-{stamp}")
                if self.stop_requested():
                    self.publish("paused", stop_reason=self.stop_requested(), trainer_exit_code=code)
                    return self.state
                if code:
                    raise RuntimeError(f"Trainer exited {code}; failure preserved; no automatic retry")
                result = verify_run(job)
            verified.append(result)
            base.write_json(self.directory / f"{job['name']}_verified.json", result)
            self.publish("verified", completed_jobs=list(verified))
            self.active_job = None
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested())
            return self.state
        self.publish("summarizing", current_job="paired_comparison", trainer_progress={})
        stamp = f"{int(time.time())}_{os.getpid()}"
        code = self.launch(["-m", "reconstruction.controller_local", "--controller-dir", str(self.directory), "--summarize"],
                           log_name=f"comparison_{stamp}.log", gpu=False, container_name=f"cxr-e2-compare-{stamp}")
        if self.stop_requested():
            self.publish("paused", stop_reason=self.stop_requested())
            return self.state
        if code or not base._read_json(self.directory / "comparison/report.json").get("complete"):
            raise RuntimeError("Paired comparison did not complete")
        self.publish("completed", current_job=None, comparison_report=str(self.directory / "comparison/REPORT.md"))
        return self.state


def summarize(directory):
    import numpy as np
    from reconstruction.diagnostics import align_reference, load_bound_predictions
    from reconstruction.evaluate import save_evaluation
    directory = Path(directory)
    plan = base._read_json(directory / "frozen_plan.json")
    base.verify_plan(plan)
    certificates = [verify_run(j) for j in plan["jobs"]]
    selected = [load_bound_predictions(Path(c["prediction_path"]).parent) for c in certificates]
    initial = [load_bound_predictions(Path(j["output"]) / "epoch_00") for j in plan["jobs"]]
    difference = np.abs(align_reference(initial[0], initial[1]) - initial[0]["probabilities"])
    if float(difference.max()) > plan["jobs"][0]["config"]["epoch_zero_max_abs_tolerance"]:
        raise ValueError("Initial arm predictions were not equivalent within the preregistered tolerance")
    summaries = {}
    for name, a, b in [("selected_local_vs_control", selected[0], selected[1]),
                       ("local_vs_initial", initial[1], selected[1]),
                       ("control_vs_initial", initial[0], selected[0])]:
        report = save_evaluation(directory / "comparison" / name, align_reference(a, b), a["targets"],
                                 a["study_keys"], a["patient_ids"], a["views"],
                                 reference_probabilities=a["probabilities"], bootstrap_repeats=1000,
                                 seed=plan["jobs"][0]["config"]["seed"], scope="inner_validation",
                                 metadata={"comparison": name, "reference_binding": a["binding"], "candidate_binding": b["binding"]})
        summaries[name] = report
    primary = summaries["selected_local_vs_control"]
    report = {**primary, "complete": True, "certificates": certificates,
              "comparisons": summaries, "epoch_zero_max_abs_delta": float(difference.max()),
              "epoch_zero_mean_abs_delta": float(difference.mean()),
              "selection_caveat": "Best of 0..6 on the same dev; no seed uncertainty or selection-bias correction",
              "interpretation": "Tests the local-input plus extra-head package; not isolated pixel or lesion causality",
              "followup_training_launched": False, "deployment_selected": False}
    shutil.copyfile(directory / "comparison/selected_local_vs_control/predictions.npz",
                    directory / "comparison/predictions.npz")
    if base.sha(directory / "comparison/predictions.npz") != report["predictions_sha256"]:
        raise ValueError("Primary report/prediction identity differs")
    base.write_json(directory / "comparison/report.json", report)
    rows = ["# E2 global/local paired review", "", "Two fixed six-epoch arms; best may remain at untrained epoch zero.", "",
            "| Comparison | Reference AUC | Candidate AUC | Delta | Patient 95% CI |", "| --- | ---: | ---: | ---: | --- |"]
    for name, item in summaries.items():
        s = item["overall"]
        rows.append(f"| {name} | {s['reference']['macro_auc']:.8f} | {s['macro_auc']:.8f} | {s['macro_auc_delta']:+.8f} | {s['patient_bootstrap']['paired_delta_macro_auc']['ci95']} |")
    rows += ["", "Full per-class/AP/single/multiple-image metrics and intervals are in report.json.",
             "These are exploratory internal-development results, not independent confirmation.",
             "No H/official valid/test, E3/E4, submission, or deployment action was performed.", ""]
    (directory / "comparison/REPORT.md").write_text("\n".join(rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs=2, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--resume-run", action="append", type=Path, default=[])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    directory = base._inside(args.controller_dir, RUN_ROOT / "E2_20261104")
    if args.summarize:
        if args.execute or args.resume_controller or args.resume_run:
            parser.error("CPU summary cannot launch trainers")
        summarize(directory)
        return
    if not args.configs:
        parser.error("Explicit matched configs are required")
    if not args.execute:
        print(build_plan(args.configs))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".controller.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = directory / "frozen_plan.json"
        if args.resume_controller:
            plan = base._read_json(plan_path)
            if [str(p.resolve()) for p in args.configs] != [j["config_path"] for j in plan["jobs"]]:
                raise ValueError("Resume must preserve the exact original queue")
            previous = base._read_json(directory / "STATUS.json")
            if base._running_container(previous.get("active_container")) or (directory / "STOP").exists():
                raise RuntimeError("Prior container active or original STOP still present")
            base.verify_plan(plan)
        else:
            if any(p.name != ".controller.lock" for p in directory.iterdir()):
                raise FileExistsError("Existing queue evidence requires explicit resume")
            plan = build_plan(args.configs)
            for job in plan["jobs"]:
                path = Path(job["output"])
                if path.exists() and any(path.iterdir()):
                    raise FileExistsError("Fresh E2 queue requires fresh arm output directories")
            base.write_json(plan_path, plan)
        controller = LocalController(directory, plan, resume_runs=args.resume_run)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, controller.signal_stop)
        try:
            result = controller.run()
        except BaseException as error:
            controller.publish("failed", error=f"{type(error).__name__}: {error}")
            raise
        if result["state"] != "completed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
