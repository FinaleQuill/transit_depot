"""Host-stdlib controller for the registered four-run tight-crop A/B study.

No training starts without --execute. Use --detach --execute for a background
controller, --status for progress, and --resume --execute after an interruption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "docs/TIGHT_CROP_AB_PLAN_20261102.json"
WRAPPER = ROOT / "run_tight_crop_environment.sh"
RUN_ROOT = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102")
ARTIFACTS = ROOT / "artifacts/tight_crop_ab_20261102"
JOBS = (("A", "tiny"), ("B", "tiny"), ("A", "base"), ("B", "base"))
SOURCE_NAMES = (
    "run_tight_crop_ab_pipeline.py", "run_tight_crop_environment.sh",
    "tight_crop_ab_training.py", "tight_crop_ab_data.py", "tight_crop_ab_preprocess.py",
    "tight_crop_ab_evaluate.py", "tight_crop.py", "study_student.py",
    "study_packing.py", "student_experiments.py", "train_distill.py",
    "train_overnight.py", "distill_data.py", "distill_tta_data.py",
    "model_overnight.py", "dataset_overnight.py", "dataset.py", "cv_splits.py",
    "kd_common.py", "runtime_compat.py", "audit_trail.py",
    "training_evidence_monitor.py", "deploy_kd.py", "inference_inputs.py",
    "student_ensemble.py", "student_experiment_evaluation.py",
    "student_fixed_pair.py", "student_draft_pair.py", "fast_lanczos.py",
    "inference_runtime.py",
)
SMALL_NAMES = {"status.json", "config.json", "history.jsonl", "best_metrics.json",
               "pretrained_provenance.json", "checkpoint_audit.jsonl"}
HEARTBEAT_SECONDS = 15


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def frozen_paths():
    paths = [PLAN, *(ROOT / name for name in SOURCE_NAMES)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    return paths


def validate_plan():
    plan = read_json(PLAN)
    if not isinstance(plan, dict):
        raise ValueError("Registered A/B plan is missing or invalid")
    actual_order = [f"{arm}_{member}" for arm, member in JOBS]
    if (plan.get("paired_design", {}).get("job_order") != actual_order
            or plan.get("paired_design", {}).get("epochs") != 2
            or plan.get("paired_design", {}).get("checkpoint")
                != "last.pt final E2 EMA only, never best.pt"
            or plan.get("automation", {}).get("run_root") != str(RUN_ROOT)
            or plan.get("automation", {}).get("max_checkpoint_retry") != 1
            or plan.get("preprocessing", {}).get("original_crop_sha256")
                != sha256(ROOT / "tight_crop.py")
            or plan.get("preprocessing", {}).get("shared_module_sha256")
                != sha256(ROOT / "tight_crop_ab_preprocess.py")):
        raise ValueError("Registered plan disagrees with fixed A/B execution")


def check_frozen(expected):
    for path in frozen_paths():
        key = str(path.relative_to(ROOT))
        if sha256(path) != expected.get(key):
            raise RuntimeError(f"Frozen source or plan changed: {key}")


def freeze_sources(resume):
    validate_plan()
    path = ARTIFACTS / "pipeline_sources.json"
    existing = read_json(path)
    if existing is not None:
        if not resume:
            raise FileExistsError("Pipeline already registered; use --resume")
        hashes = existing["sha256"]
        check_frozen(hashes)
        return hashes
    if resume:
        raise FileNotFoundError("No registered pipeline source set to resume")
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in frozen_paths()}
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    write_json(path, {"registered_utc": utc_now(), "git_head": git_head,
                      "sha256": hashes,
                      "claim": "Checked before every sequential stage; no source changes allowed."})
    return hashes


def job_dir(arm, member):
    return RUN_ROOT / f"{arm}_{member}"


def mirror_dir(source, target, *, small_only=False):
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(source)
        if small_only and path.name not in SMALL_NAMES and "evidence" not in relative.parts:
            continue
        if small_only and path.stat().st_size > 10 * 1024 * 1024:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (destination.is_file() and destination.stat().st_size == path.stat().st_size
                and destination.stat().st_mtime_ns == path.stat().st_mtime_ns):
            continue
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        shutil.copy2(path, temporary)
        temporary.replace(destination)


def mirror_job(arm, member, *, full=False):
    source = job_dir(arm, member)
    target = ARTIFACTS / "runs" / f"{arm}_{member}"
    mirror_dir(source, target, small_only=not full)
    return target


def mirror_epoch_checkpoint(arm, member, record, hashes):
    """Persist each committed epoch once, after status and checkpoint hashes agree."""
    folder = job_dir(arm, member)
    status = read_json(folder / "status.json", {})
    epoch = status.get("completed_epochs", 0)
    if (type(epoch) is not int or epoch not in (1, 2)
            or epoch <= record.get("mirrored_epoch", 0)):
        mirror_job(arm, member)
        return
    history = history_rows(folder / "history.jsonl")
    if (not history or history[-1].get("epoch") != epoch
            or status.get("last_checkpoint_sha256") != history[-1].get("last_checkpoint_sha256")
            or status.get("best_checkpoint_sha256") != history[-1].get("best_checkpoint_sha256")):
        mirror_job(arm, member)
        return
    try:
        verified = verify_checkpoint(folder, complete=epoch == 2,
                                     arm=arm, member=member, hashes=hashes)
    except ValueError:
        # A checkpoint may be replaced between observing status and hashing it.
        mirror_job(arm, member)
        return
    if verified["completed_epochs"] != epoch:
        mirror_job(arm, member)
        return
    target = mirror_job(arm, member, full=True)
    copied = verify_checkpoint(target, complete=epoch == 2,
                               arm=arm, member=member, hashes=hashes)
    if copied != verified:
        raise RuntimeError(f"Persistent checkpoint copy changed for {arm}_{member} E{epoch}")
    record["mirrored_epoch"] = epoch


def restore_job(arm, member):
    source = ARTIFACTS / "runs" / f"{arm}_{member}"
    target = job_dir(arm, member)
    if target.exists() or not source.is_dir():
        return
    mirror_dir(source, target)


def history_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_config(folder, arm, member, hashes):
    config = read_json(folder / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"Missing run configuration: {folder}")
    if (config.get("arm") != arm or config.get("member") != member
            or config.get("epochs") != 2 or config.get("seed") != 48
            or config.get("ab_plan_sha256") != hashes["docs/TIGHT_CROP_AB_PLAN_20261102.json"]
            or config.get("ab_preprocess_sha256") != hashes["tight_crop_ab_preprocess.py"]
            or config.get("ab_crop_sha256") != hashes["tight_crop.py"]):
        raise ValueError(f"Run configuration differs from registration: {folder}")


def verify_checkpoint(folder, *, complete, arm, member, hashes):
    verify_config(folder, arm, member, hashes)
    rows = history_rows(folder / "history.jsonl")
    epochs = [row.get("epoch") for row in rows]
    if epochs not in ([1], [1, 2]):
        raise ValueError(f"Unexpected epoch history: {folder}: {epochs}")
    status = read_json(folder / "status.json", {})
    latest = rows[-1]
    for name, key in (("last.pt", "last_checkpoint_sha256"),
                      ("best.pt", "best_checkpoint_sha256")):
        path = folder / name
        expected = latest.get(key)
        if not path.is_file() or not expected or sha256(path) != expected:
            raise ValueError(f"Missing or mismatched {name}: {folder}")
    if complete:
        if (epochs != [1, 2] or status.get("state") != "completed"
                or status.get("completed_epochs") != 2
                or status.get("validated_epochs") != 2
                or status.get("last_checkpoint_sha256") != latest["last_checkpoint_sha256"]
                or status.get("best_checkpoint_sha256") != latest["best_checkpoint_sha256"]):
            raise ValueError(f"Fixed E2 completion evidence invalid: {folder}")
    return {"completed_epochs": epochs[-1],
            "last_sha256": latest["last_checkpoint_sha256"],
            "best_sha256": latest["best_checkpoint_sha256"]}


def completed_job(folder, arm, member, hashes):
    status = read_json(folder / "status.json", {})
    if status.get("state") != "completed":
        return None
    return verify_checkpoint(folder, complete=True, arm=arm, member=member, hashes=hashes)


def write_status(state):
    state["updated_utc"] = utc_now()
    state["elapsed_seconds"] = round(time.monotonic() - state["started_monotonic"], 1)
    persistent = {key: value for key, value in state.items() if key != "started_monotonic"}
    write_json(ARTIFACTS / "status.json", persistent)
    lines = ["# Tight-crop A/B pipeline", "",
             f"State: {state['state']}", f"Stage: {state['stage']}",
             f"PID: {state['pid']}", f"Updated UTC: {state['updated_utc']}",
             f"Elapsed: {state['elapsed_seconds']} seconds", "",
             "Four fixed E2 runs, followed by paired valid and historical 044 evaluation.",
             "`last.pt` E2 EMA is the treatment; `best.pt` is diagnostic only.", ""]
    for arm, member in JOBS:
        name = f"{arm}_{member}"
        job = state["jobs"].get(name, {})
        source = read_json(job_dir(arm, member) / "status.json", {})
        lines.append(f"- {name}: {job.get('state', 'pending')}; epochs "
                     f"{source.get('completed_epochs', job.get('completed_epochs', 0))}/2; "
                     f"retries {job.get('retries', 0)}/1")
    if state.get("error"):
        lines.extend(("", f"Error: {state['error']}"))
    lines.extend(("", "Read `status.json` for machine-readable progress and `logs/` for command output.", ""))
    (ARTIFACTS / "STATUS.md").write_text("\n".join(lines))


def run_command(command, log_path, state, hashes, *, mirror=None):
    check_frozen(hashes)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(map(str, command))}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        state["child_pid"] = process.pid
        write_status(state)
        while process.poll() is None:
            if mirror is not None:
                mirror()
            write_status(state)
            time.sleep(HEARTBEAT_SECONDS)
        code = process.wait()
        state["child_pid"] = None
        if mirror is not None:
            mirror()
        check_frozen(hashes)
        write_status(state)
        return code


def run_job(arm, member, state, hashes, *, resume):
    name = f"{arm}_{member}"
    folder = job_dir(arm, member)
    record = state["jobs"].setdefault(name, {"state": "pending", "retries": 0})
    state["stage"] = f"train_{name}"
    check_frozen(hashes)
    if resume:
        restore_job(arm, member)
    if folder.exists():
        done = completed_job(folder, arm, member, hashes)
        if done:
            mirror_job(arm, member, full=True)
            persistent = ARTIFACTS / "runs" / name
            verify_checkpoint(persistent, complete=True, arm=arm, member=member, hashes=hashes)
            record.update(state="completed", **done)
            write_status(state)
            return
        if not resume:
            raise FileExistsError(f"Existing incomplete run requires --resume: {folder}")
        if record["retries"] >= 1:
            raise RuntimeError(f"Retry already consumed for {name}")
        safe = verify_checkpoint(folder, complete=False, arm=arm, member=member, hashes=hashes)
        record["retries"] += 1
        record["resumed_from_epoch"] = safe["completed_epochs"]
        resume_path = folder / "last.pt"
    else:
        resume_path = None
    while True:
        command = ["bash", str(WRAPPER), "tight_crop_ab_training.py",
                   "--arm", arm, "--member", member, "--output", str(folder)]
        if resume_path is not None:
            command.extend(("--resume", str(resume_path)))
        command.append("--execute")
        record["state"] = "running"
        write_status(state)
        code = run_command(command, ARTIFACTS / "logs" / f"{name}.log", state,
                           hashes, mirror=lambda: mirror_epoch_checkpoint(
                               arm, member, record, hashes))
        if code == 0:
            done = completed_job(folder, arm, member, hashes)
            if done:
                mirror_job(arm, member, full=True)
                persistent = ARTIFACTS / "runs" / name
                verify_checkpoint(persistent, complete=True, arm=arm, member=member, hashes=hashes)
                record.update(state="completed", **done)
                write_status(state)
                return
        record["last_exit_code"] = code
        mirror_job(arm, member)
        if record["retries"] >= 1:
            raise RuntimeError(f"{name} failed after its single safe retry; see persistent log")
        safe = verify_checkpoint(folder, complete=False, arm=arm, member=member, hashes=hashes)
        if (folder / "STOP").exists():
            raise RuntimeError(f"{name} has STOP marker; refusing automatic retry")
        mirror_job(arm, member, full=True)
        record["retries"] += 1
        record["resumed_from_epoch"] = safe["completed_epochs"]
        resume_path = folder / "last.pt"
        write_status(state)


def evaluation_complete(action, output, hashes):
    if action == "paired":
        metrics = read_json(output / "metrics.json")
        if not isinstance(metrics, dict) or not (output / "REPORT.md").is_file():
            return False
        if (metrics.get("plan_sha256") != hashes["docs/TIGHT_CROP_AB_PLAN_20261102.json"]
                or metrics.get("preprocess_sha256") != hashes["tight_crop_ab_preprocess.py"]
                or metrics.get("crop_sha256") != hashes["tight_crop.py"]
                or not all((output / "exports" / f"{a}_{m}" / "state_fp16.pt").is_file()
                           for a, m in JOBS)):
            raise ValueError("Paired evaluation artifacts differ from frozen sources")
        return True
    summary = read_json(output / "historical_044/historical_summary.json")
    if not isinstance(summary, dict):
        return False
    report = output / "historical_044/validation_report.json"
    if (summary.get("reference") != "historical_044" or not report.is_file()
            or sha256(report) != summary.get("report_sha256")):
        raise ValueError("Historical reference summary/report mismatch")
    return True


def write_auc_plot(output):
    metrics = read_json(output / "metrics.json")
    if not isinstance(metrics, dict):
        raise FileNotFoundError("Paired metrics missing for plot")
    names = ("all", "ap_only", "non_ap_only", "single_image", "multiple_image")
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                             "width": "900", "height": "310", "viewBox": "0 0 900 310"})
    ET.SubElement(svg, "rect", {"width": "900", "height": "310", "fill": "#ffffff"})
    title = ET.SubElement(svg, "text", {"x": "25", "y": "28", "font-size": "19",
                                       "font-family": "sans-serif", "fill": "#202830"})
    title.text = "Fixed E2 EMA valid macro-AUC by Study slice"
    for i, name in enumerate(names):
        y = 65 + i * 45
        label = ET.SubElement(svg, "text", {"x": "25", "y": str(y + 17),
                                           "font-size": "13", "font-family": "sans-serif"})
        label.text = name.replace("_", " ")
        for j, arm in enumerate(("A", "B")):
            value = metrics["arms"][arm][name]["macro_auc"]
            if value is None:
                continue
            width = max(0, min(500, round(float(value) * 500)))
            yy = y + j * 18
            ET.SubElement(svg, "rect", {"x": "180", "y": str(yy), "width": str(width),
                                        "height": "14", "fill": "#168d8d" if arm == "A" else "#c75837"})
            number = ET.SubElement(svg, "text", {"x": str(190 + width), "y": str(yy + 12),
                                                 "font-size": "12", "font-family": "sans-serif"})
            number.text = f"{arm} {value:.6f}"
    ET.ElementTree(svg).write(output / "auc_slices.svg", encoding="utf-8", xml_declaration=True)


def run_evaluation(action, state, hashes, *, resume):
    output = RUN_ROOT / "evaluation"
    state["stage"] = f"evaluate_{action}"
    check_frozen(hashes)
    if evaluation_complete(action, output, hashes):
        if not resume:
            raise FileExistsError("Existing evaluation requires --resume")
    else:
        if action == "paired" and any((output / x).exists() for x in
                                      ("exports", "A", "B", "metrics.json", "REPORT.md")):
            raise RuntimeError("Partial paired evaluation exists; preserve and inspect before retry")
        if action == "historical" and (output / "historical_044").exists():
            raise RuntimeError("Partial historical evaluation exists; preserve and inspect before retry")
        command = ["bash", str(WRAPPER), "tight_crop_ab_evaluate.py", "--action", action,
                   "--plan", str(PLAN), "--run-root", str(RUN_ROOT), "--output", str(output)]
        code = run_command(command, ARTIFACTS / "logs" / f"evaluate_{action}.log",
                           state, hashes)
        if code != 0 or not evaluation_complete(action, output, hashes):
            raise RuntimeError(f"{action} evaluation failed or incomplete; see persistent log")
    if action == "paired":
        write_auc_plot(output)
    mirror_dir(output, ARTIFACTS / "evaluation")
    state["evaluations"][action] = "completed"
    write_status(state)


def pipeline(resume):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / ".pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = freeze_sources(resume)
        prior = read_json(ARTIFACTS / "status.json", {}) if resume else {}
        prior_child = prior.get("child_pid")
        if isinstance(prior_child, int):
            try:
                command_line = Path(f"/proc/{prior_child}/cmdline").read_bytes()
            except FileNotFoundError:
                command_line = b""
            belongs_to_run = str(RUN_ROOT).encode() in command_line and any(
                marker in command_line for marker in
                (b"tight_crop_ab_training.py", b"tight_crop_ab_evaluate.py",
                 b"run_tight_crop_environment.sh"))
            if belongs_to_run:
                raise RuntimeError(f"Previous stage launcher PID {prior_child} is still active")
        state = {"state": "running", "stage": "startup", "pid": os.getpid(),
                 "child_pid": None, "started_utc": prior.get("started_utc", utc_now()),
                 "started_monotonic": time.monotonic(), "jobs": prior.get("jobs", {}),
                 "evaluations": prior.get("evaluations", {}), "error": None,
                 "source_manifest": str(ARTIFACTS / "pipeline_sources.json"),
                 "run_root": str(RUN_ROOT), "test_predicted": False}
        write_status(state)
        try:
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            for arm, member in JOBS:
                run_job(arm, member, state, hashes, resume=resume)
            run_evaluation("paired", state, hashes, resume=resume)
            run_evaluation("historical", state, hashes, resume=resume)
            state.update(state="completed", stage="done")
            write_status(state)
            print(f"Completed; see {ARTIFACTS / 'evaluation/REPORT.md'}", flush=True)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            write_status(state)
            with (ARTIFACTS / "logs" / "controller_error.log").open("a") as stream:
                traceback.print_exc(file=stream)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_json(ARTIFACTS / "status.json", {"state": "not_started"}), indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"dry_run": True, "jobs": [f"{a}_{m}" for a, m in JOBS],
                          "run_root": str(RUN_ROOT), "artifacts": str(ARTIFACTS),
                          "plan_exists": PLAN.is_file(), "resume": args.resume}, indent=2))
        return 0
    if args.detach:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "logs").mkdir(parents=True, exist_ok=True)
        with (ARTIFACTS / "logs" / "controller.log").open("a") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute"]
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f"Detached controller PID {process.pid}; status: {ARTIFACTS / 'STATUS.md'}")
        return 0
    pipeline(args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
