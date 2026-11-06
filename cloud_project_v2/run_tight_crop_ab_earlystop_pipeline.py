"""Host controller for AP early stopping, max 25 epochs, A runs before B runs.

The original two-epoch source is never edited. This controller waits for its
graceful E1 STOP and controller exit, then migrates into a new run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import run_tight_crop_ab_pipeline as old


ROOT = old.ROOT
ARTIFACTS = old.ARTIFACTS
WORK = ARTIFACTS / "earlystop25"
RUN_ROOT = old.RUN_ROOT / "earlystop25"
PLAN = ROOT / "docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json"
TRAINER = ROOT / "tight_crop_ab_earlystop_training.py"
EVALUATOR = ROOT / "tight_crop_ab_earlystop_evaluate.py"
WRAPPER = old.WRAPPER
JOBS = (("A", "tiny"), ("A", "base"), ("B", "tiny"), ("B", "base"))
POLL_SECONDS = 15
SOURCE_KEYS = (PLAN, TRAINER, EVALUATOR, Path(__file__).resolve())


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].strip().split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        return {"state": stat[0], "start_ticks": int(stat[19]), "command": command}
    except (FileNotFoundError, IndexError, ValueError):
        return None


def live_process(pid, ticks, marker):
    identity = process_identity(pid)
    return bool(identity and identity["state"] not in ("Z", "X")
                and identity["start_ticks"] == ticks and marker in identity["command"])


def docker_container_for_old_a_tiny():
    result = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Command}}"],
                            capture_output=True, text=True, check=True, timeout=20)
    matches = [line.split(" ", 1)[0] for line in result.stdout.splitlines()
               if "tight_crop_ab_training.py --arm A --member tiny" in line
               and str(old.job_dir("A", "tiny")) in line]
    if len(matches) > 1:
        raise RuntimeError("Multiple live old A_tiny containers")
    return matches[0] if matches else None


def old_controller_finished(plan):
    pid = plan["orchestration"]["original_controller_pid"]
    identity = process_identity(pid)
    return (identity is None or identity["state"] in ("Z", "X")
            or "run_tight_crop_ab_pipeline.py --execute" not in identity["command"])


def container_running(container_id):
    if container_id is None:
        return False
    result = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", container_id],
                            capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and result.stdout.strip() == "true"


def validate_plan():
    plan = old.read_json(PLAN)
    if not isinstance(plan, dict):
        raise FileNotFoundError("Early-stop amendment missing")
    train = plan.get("training", {})
    orchestration = plan.get("orchestration", {})
    if (plan.get("original_plan_sha256") != old.sha256(old.PLAN)
            or plan.get("job_order") != [f"{a}_{m}" for a, m in JOBS]
            or train.get("max_epochs") != 25 or train.get("patience") != 5
            or train.get("min_delta") != 0.0001 or train.get("monitor") != "ap_only_macro_auc"
            or orchestration.get("run_root") != str(RUN_ROOT)
            or orchestration.get("failure_retry_max") != 1
            or orchestration.get("adopt_docker_pid") != 28156
            or orchestration.get("adopt_docker_start_ticks") != 1758452):
        raise ValueError("Early-stop amendment/controller mismatch")
    return plan


def freeze_sources(resume):
    plan = validate_plan()
    original_receipt = old.read_json(ARTIFACTS / "pipeline_sources.json")
    if not isinstance(original_receipt, dict):
        raise FileNotFoundError("Original two-epoch source receipt missing")
    original_hashes = original_receipt["sha256"]
    old.check_frozen(original_hashes)
    if any(not path.is_file() for path in SOURCE_KEYS):
        raise FileNotFoundError("Amended controller/trainer/evaluator must exist")
    hashes = {str(path.relative_to(ROOT)): old.sha256(path) for path in SOURCE_KEYS}
    receipt = WORK / "sources.json"
    existing = old.read_json(receipt)
    if existing is not None:
        if not resume or existing.get("sha256") != hashes:
            raise RuntimeError("Amended sources differ from registered receipt")
    elif resume:
        raise FileNotFoundError("No amended source receipt to resume")
    else:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        old.write_json(receipt, {"registered_utc": now(), "git_head": git_head,
                                 "original_receipt": str(ARTIFACTS / "pipeline_sources.json"),
                                 "sha256": hashes})
    return plan, original_hashes, hashes


def check_sources(original_hashes, hashes):
    old.check_frozen(original_hashes)
    for relative, digest in hashes.items():
        if old.sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Frozen amended source changed: {relative}")


def run_dir(arm, member):
    return RUN_ROOT / f"{arm}_{member}"


def persistent_run(arm, member):
    return WORK / "runs" / f"{arm}_{member}"


def write_status(state):
    state["updated_utc"] = now()
    state["elapsed_seconds"] = round(time.monotonic() - state["started_monotonic"], 1)
    payload = {key: value for key, value in state.items() if key != "started_monotonic"}
    old.write_json(WORK / "status.json", payload)
    lines = ["# Tight-crop A/B, AP early stopping", "",
             f"State: {state['state']}", f"Stage: {state['stage']}",
             f"PID: {state['pid']}", f"Updated UTC: {state['updated_utc']}",
             f"Elapsed: {state['elapsed_seconds']} seconds", "",
             "Order: A_tiny, A_base, B_tiny, B_base. Max 25 epochs; AP-only patience 5.",
             "Old A_tiny E1 is migrated with optimizer, EMA, scaler and RNG state.",
             "Historical 044 is separate; no test prediction or production timing claim.", ""]
    for arm, member in JOBS:
        name = f"{arm}_{member}"
        record = state["jobs"].get(name, {})
        native = old.read_json(run_dir(arm, member) / "status.json", {})
        lines.append(f"- {name}: {record.get('state', 'pending')}; "
                     f"epoch {native.get('completed_epochs', 0)}/25; "
                     f"best AP {native.get('best_auc', 'pending')}; "
                     f"failure retry {record.get('failure_retries', 0)}/1")
    if state.get("error"):
        lines.extend(("", f"Error: {state['error']}"))
    lines.extend(("", "Details: `earlystop25/status.json`; logs: `earlystop25/logs/`.", ""))
    (WORK / "STATUS.md").write_text("\n".join(lines))
    if state.get("owns_original_lock"):
        old.write_json(ARTIFACTS / "status.json", payload)
        (ARTIFACTS / "STATUS.md").write_text("\n".join(lines))


def verify_native_run(arm, member, *, terminal):
    folder = run_dir(arm, member)
    status = old.read_json(folder / "status.json", {})
    progress = status.get("completed_epochs")
    if type(progress) is not int or progress < 1 or progress > 25:
        raise ValueError(f"Missing committed checkpoint for {arm}_{member}")
    if status.get("validated_epochs") != progress:
        raise ValueError(f"Unvalidated epoch for {arm}_{member}")
    if terminal and (status.get("state") not in ("completed", "early_stopped")
                     or status.get("selected_checkpoint") != "best.pt"):
        raise ValueError(f"A/B AP early-stop terminal status invalid: {arm}_{member}")
    history = old.history_rows(folder / "history.jsonl")
    if [row.get("epoch") for row in history] != list(range(1, progress + 1)):
        raise ValueError(f"A/B epoch history gap: {arm}_{member}")
    latest = history[-1]
    expected_last = status.get("last_checkpoint_sha256") or latest.get("last_checkpoint_sha256")
    expected_best = status.get("best_checkpoint_sha256") or latest.get("best_checkpoint_sha256")
    if (latest.get("last_checkpoint_sha256") != expected_last
            or latest.get("best_checkpoint_sha256") != expected_best
            or old.sha256(folder / "last.pt") != expected_last
            or old.sha256(folder / "best.pt") != expected_best):
        raise ValueError(f"A/B checkpoint/hash mismatch: {arm}_{member}")
    config = old.read_json(folder / "config.json", {})
    if (config.get("arm") != arm or config.get("member") != member
            or config.get("epochs") != 25 or config.get("patience") != 5
            or config.get("min_delta") != 0.0001):
        raise ValueError(f"A/B amended config mismatch: {arm}_{member}")
    return {"completed_epochs": progress, "last_sha256": expected_last,
            "best_sha256": expected_best, "best_epoch": status.get("best_epoch"),
            "best_auc": status.get("best_auc")}


def mirror_run(arm, member, record):
    folder = run_dir(arm, member)
    status = old.read_json(folder / "status.json", {})
    progress = status.get("completed_epochs", 0)
    if type(progress) is int and progress > record.get("mirrored_epoch", 0):
        try:
            verified = verify_native_run(arm, member, terminal=False)
        except (ValueError, FileNotFoundError):
            old.mirror_dir(folder, persistent_run(arm, member), small_only=True)
            return
        old.mirror_dir(folder, persistent_run(arm, member))
        copied = old.sha256(persistent_run(arm, member) / "last.pt")
        if copied != verified["last_sha256"]:
            raise RuntimeError(f"Persistent E{progress} last.pt copy changed for {arm}_{member}")
        record["mirrored_epoch"] = progress
    else:
        old.mirror_dir(folder, persistent_run(arm, member), small_only=True)


def run_command(command, log_name, state, original_hashes, hashes, *, job=None):
    check_sources(original_hashes, hashes)
    log_path = WORK / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] command: {' '.join(map(str, command))}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        identity = process_identity(child.pid)
        state["child_pid"] = child.pid
        state["child_start_ticks"] = identity["start_ticks"] if identity else None
        write_status(state)
        while child.poll() is None:
            if job:
                arm, member = job
                mirror_run(arm, member, state["jobs"][f"{arm}_{member}"])
            write_status(state)
            time.sleep(POLL_SECONDS)
        code = child.wait()
        state["child_pid"] = None
        state["child_start_ticks"] = None
        if job:
            arm, member = job
            mirror_run(arm, member, state["jobs"][f"{arm}_{member}"])
        check_sources(original_hashes, hashes)
        write_status(state)
        return code


def adopt_original_e1(state, plan, original_hashes, hashes):
    source = old.job_dir("A", "tiny")
    target = WORK / "migration" / "original_E1" / "last.pt"
    state["stage"] = "adopt_old_A_tiny_E1"
    write_status(state)
    if target.is_file():
        receipt = old.read_json(WORK / "migration/original_E1_receipt.json")
        if not isinstance(receipt, dict):
            raise RuntimeError("Saved original E1 migration snapshot differs from receipt")
        if old.sha256(target) != receipt.get("sha256"):
            raise RuntimeError("Saved original E1 migration snapshot SHA differs from receipt")
        return target
    pid = plan["orchestration"]["adopt_docker_pid"]
    ticks = plan["orchestration"]["adopt_docker_start_ticks"]
    identity = process_identity(pid)
    if identity and identity["start_ticks"] != ticks:
        raise RuntimeError("Original Docker PID reused")
    container_id = docker_container_for_old_a_tiny()
    if container_id and not live_process(pid, ticks, str(source)):
        raise RuntimeError("Original A_tiny container lacks registered Docker client")
    stop = source / "STOP"
    if not stop.exists():
        raise RuntimeError("Old A_tiny E1 STOP is missing; refusing uncontrolled continuation")
    state["adoption"] = {"docker_pid": pid, "docker_start_ticks": ticks,
                         "container_id": container_id, "stop_path": str(stop)}
    write_status(state)
    while live_process(pid, ticks, str(source)) or container_running(container_id):
        check_sources(original_hashes, hashes)
        old.mirror_epoch_checkpoint("A", "tiny", state["jobs"].setdefault("A_tiny", {}),
                                    original_hashes)
        write_status(state)
        time.sleep(POLL_SECONDS)
    status = old.read_json(source / "status.json", {})
    if (status.get("state") != "paused" or status.get("completed_epochs") != 1
            or status.get("validated_epochs") != 1):
        raise RuntimeError("Original A_tiny did not stop after complete E1 checkpoint")
    verified = old.verify_checkpoint(source, complete=False, arm="A", member="tiny",
                                     hashes=original_hashes)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    shutil.copy2(source / "last.pt", temporary)
    if old.sha256(temporary) != verified["last_sha256"]:
        raise RuntimeError("Original E1 checkpoint changed during immutable copy")
    temporary.replace(target)
    old.mirror_dir(source, WORK / "migration/original_A_tiny_run")
    old.write_json(WORK / "migration/original_E1_receipt.json", {
        "captured_utc": now(), "source": str(source / "last.pt"),
        "immutable_copy": str(target), "sha256": verified["last_sha256"],
        "completed_epochs": 1, "original_source_receipt": str(ARTIFACTS / "pipeline_sources.json")})
    return target


def score_original_e1(snapshot, state, original_hashes, hashes):
    output = RUN_ROOT / "migration" / "A_tiny_E1_monitor"
    score = old.read_json(output / "monitor.json")
    if score is None:
        if output.exists():
            raise RuntimeError("Partial original E1 score exists; inspect before retry")
        state["stage"] = "score_original_A_tiny_E1_AP"
        write_status(state)
        command = ["bash", str(WRAPPER), TRAINER.name,
                   "--arm", "A", "--member", "tiny", "--output", str(output),
                   "--monitor-source", str(snapshot), "--execute"]
        code = run_command(command, "migration_A_tiny_E1_score.log", state,
                           original_hashes, hashes)
        if code != 0:
            raise RuntimeError("Original E1 AP score failed")
        score = old.read_json(output / "monitor.json")
    if (not isinstance(score, dict) or score.get("arm") != "A"
            or score.get("member") != "tiny"
            or score.get("source_sha256") != old.sha256(snapshot)
            or score.get("monitor_protocol")
                != "native_fp32_ema_fp16_autocast_batch4_hflip2"
            or score.get("earlystop_plan_sha256") != old.sha256(PLAN)
            or type(score.get("ap_only_macro_auc")) not in (int, float)):
        raise ValueError("Original E1 AP score identity invalid")
    old.mirror_dir(output, WORK / "migration/A_tiny_E1_monitor")
    return output / "monitor.json"


def completed_job(arm, member):
    status = old.read_json(run_dir(arm, member) / "status.json", {})
    if status.get("state") not in ("completed", "early_stopped"):
        return None
    return verify_native_run(arm, member, terminal=True)


def run_job(arm, member, state, original_hashes, hashes, *, migrate=None, ap_score=None,
            resume=False):
    name = f"{arm}_{member}"
    folder = run_dir(arm, member)
    record = state["jobs"].setdefault(name, {"state": "pending", "failure_retries": 0})
    if resume and not folder.exists() and persistent_run(arm, member).is_dir():
        old.mirror_dir(persistent_run(arm, member), folder)
    done = completed_job(arm, member) if folder.is_dir() else None
    if done:
        old.mirror_dir(folder, persistent_run(arm, member))
        if old.sha256(persistent_run(arm, member) / "best.pt") != done["best_sha256"]:
            raise RuntimeError(f"Persistent AP-best checkpoint mismatch: {name}")
        record.update(state="completed", **done)
        write_status(state)
        return
    if folder.exists() and not resume:
        raise FileExistsError(f"Existing incomplete run needs explicit --resume: {folder}")
    if not folder.exists() and migrate is not None:
        if ap_score is None:
            raise ValueError("A_tiny migration requires native E1 AP monitor receipt")
        state["stage"] = "migrate_A_tiny_E1_full_state"
        write_status(state)
        command = ["bash", str(WRAPPER), TRAINER.name, "--arm", arm, "--member", member,
                   "--output", str(folder), "--migrate-from", str(migrate),
                   "--migrate-ap-score", str(ap_score), "--execute"]
        code = run_command(command, f"migrate_{name}_E1.log", state, original_hashes, hashes)
        if code != 0:
            raise RuntimeError("A_tiny full-state E1 migration failed")
        migrated = verify_native_run(arm, member, terminal=False)
        if migrated["completed_epochs"] != 1:
            raise RuntimeError("A_tiny migration did not preserve completed E1")
        old.mirror_dir(folder, persistent_run(arm, member))
        record["migration_sha256"] = old.sha256(folder / "migration.json")
        resume_checkpoint = folder / "last.pt"
        migrate = None
        ap_score = None
        write_status(state)
    if folder.exists() and migrate is not None:
        migrate = None
        ap_score = None
    if folder.exists():
        if not (folder / "last.pt").is_file():
            raise RuntimeError(f"No safe checkpoint for existing incomplete {name}")
        verify_native_run(arm, member, terminal=False)
        resume_checkpoint = folder / "last.pt"
    else:
        resume_checkpoint = None
    while True:
        state["stage"] = f"train_{name}"
        record["state"] = "running"
        write_status(state)
        command = ["bash", str(WRAPPER), TRAINER.name, "--arm", arm, "--member", member,
                   "--output", str(folder)]
        if resume_checkpoint is not None:
            command.extend(("--resume", str(resume_checkpoint)))
        command.append("--execute")
        code = run_command(command, f"train_{name}.log", state, original_hashes,
                           hashes, job=(arm, member))
        done = completed_job(arm, member) if folder.is_dir() else None
        if code == 0 and done:
            old.mirror_dir(folder, persistent_run(arm, member))
            if old.sha256(persistent_run(arm, member) / "best.pt") != done["best_sha256"]:
                raise RuntimeError(f"Persistent AP-best checkpoint mismatch: {name}")
            record.update(state="completed", **done)
            write_status(state)
            return
        if record["failure_retries"] >= 1:
            raise RuntimeError(f"{name} failed after one safe checkpoint retry")
        safe = verify_native_run(arm, member, terminal=False)
        if (folder / "STOP").exists():
            raise RuntimeError(f"{name} has STOP marker; refusing automatic retry")
        old.mirror_dir(folder, persistent_run(arm, member))
        record["failure_retries"] += 1
        record["failure_resume_epoch"] = safe["completed_epochs"]
        resume_checkpoint = folder / "last.pt"
        migrate = None
        ap_score = None
        write_status(state)


def score_best(arm, member, state, original_hashes, hashes):
    name = f"{arm}_{member}"
    source = run_dir(arm, member) / "best.pt"
    score_dir = RUN_ROOT / "scores" / name
    score = old.read_json(score_dir / "score.json")
    if score is None:
        if score_dir.exists():
            raise RuntimeError(f"Partial best AP score exists: {score_dir}")
        state["stage"] = f"score_best_{name}"
        write_status(state)
        command = ["bash", str(WRAPPER), EVALUATOR.name, "--action", "score",
                   "--arm", arm, "--member", member,
                   "--source", str(source),
                   "--output", str(score_dir)]
        code = run_command(command, f"score_best_{name}.log", state, original_hashes, hashes)
        if code != 0:
            raise RuntimeError(f"AP-best full-valid score failed: {name}")
        score = old.read_json(score_dir / "score.json")
    if (not isinstance(score, dict) or score.get("arm") != arm
            or score.get("member") != member
            or score.get("source_sha256") != old.sha256(source)
            or type(score.get("ap_only_macro_auc")) not in (int, float)):
        raise ValueError(f"AP-best score identity invalid: {name}")
    old.mirror_dir(score_dir, WORK / "scores" / name)
    state["jobs"][name]["best_score_dir"] = str(score_dir)
    write_status(state)


def final_evaluation(state, original_hashes, hashes):
    selected = {f"{arm}_{member}": state["jobs"][f"{arm}_{member}"]["best_score_dir"]
                for arm, member in JOBS}
    selection = RUN_ROOT / "selection.json"
    old.write_json(selection, {"selected": selected, "amendment_sha256": old.sha256(PLAN),
                               "selection_basis": "native AP-only early-stopped best.pt",
                               "test_used": False})
    old.write_json(WORK / "selection.json", old.read_json(selection))
    output = RUN_ROOT / "evaluation"
    paired = old.read_json(output / "metrics.json")
    if paired is None:
        if output.exists():
            raise FileExistsError(f"Partial final evaluation needs inspection: {output}")
        state["stage"] = "final_paired_valid"
        write_status(state)
        command = ["bash", str(WRAPPER), EVALUATOR.name, "--action", "final",
                   "--selection", str(selection), "--output", str(output)]
        code = run_command(command, "final_paired_valid.log", state, original_hashes, hashes)
        if code != 0:
            raise RuntimeError("Final AP early-stop paired valid evaluation failed")
        paired = old.read_json(output / "metrics.json")
    if (not isinstance(paired, dict) or not (output / "REPORT.md").is_file()
            or paired.get("earlystop_plan_sha256") != old.sha256(PLAN)):
        raise ValueError("Final paired evaluation output/plan identity invalid")
    historical = old.read_json(output / "historical_044/historical_summary.json")
    if historical is None:
        if (output / "historical_044").exists():
            raise FileExistsError("Partial historical 044 reference needs inspection")
        state["stage"] = "historical_044_reference"
        write_status(state)
        command = ["bash", str(WRAPPER), "tight_crop_ab_evaluate.py", "--action", "historical",
                   "--plan", str(old.PLAN), "--run-root", str(RUN_ROOT), "--output", str(output)]
        code = run_command(command, "historical_044.log", state, original_hashes, hashes)
        if code != 0:
            raise RuntimeError("Separate historical 044 reference failed")
        historical = old.read_json(output / "historical_044/historical_summary.json")
    report = output / "historical_044/validation_report.json"
    if (not isinstance(historical, dict) or historical.get("reference") != "historical_044"
            or not report.is_file() or historical.get("report_sha256") != old.sha256(report)):
        raise ValueError("Historical 044 reference identity invalid")
    old.mirror_dir(output, WORK / "evaluation")


def pipeline(resume):
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    with (WORK / ".controller.lock").open("a+") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, original_hashes, hashes = freeze_sources(resume)
        existing = old.read_json(WORK / "status.json", {}) if resume else {}
        prior_child = existing.get("child_pid")
        prior_ticks = existing.get("child_start_ticks")
        if isinstance(prior_child, int) and isinstance(prior_ticks, int):
            identity = process_identity(prior_child)
            if (identity and identity["state"] not in ("Z", "X")
                    and identity["start_ticks"] == prior_ticks
                    and str(RUN_ROOT) in identity["command"]):
                raise RuntimeError(f"Previous amended child PID {prior_child} is still active")
        state = {"state": "running", "stage": "startup", "pid": os.getpid(),
                 "child_pid": None, "child_start_ticks": None,
                 "started_utc": existing.get("started_utc", now()),
                 "started_monotonic": time.monotonic(),
                 "jobs": existing.get("jobs", {}), "adoption": existing.get("adoption"),
                 "source_manifest": str(WORK / "sources.json"), "error": None,
                 "test_predicted": False, "owns_original_lock": False}
        write_status(state)
        try:
            state["stage"] = "wait_old_A_tiny_E1_and_controller"
            while not old_controller_finished(plan):
                check_sources(original_hashes, hashes)
                write_status(state)
                time.sleep(POLL_SECONDS)
            with (ARTIFACTS / ".pipeline.lock").open("a+") as old_lock:
                fcntl.flock(old_lock, fcntl.LOCK_EX)
                state["owns_original_lock"] = True
                if not resume and (ARTIFACTS / "status.json").is_file():
                    shutil.copy2(ARTIFACTS / "status.json", WORK / "original_controller_status.json")
                    if (ARTIFACTS / "STATUS.md").is_file():
                        shutil.copy2(ARTIFACTS / "STATUS.md", WORK / "ORIGINAL_CONTROLLER_STATUS.md")
                write_status(state)
                RUN_ROOT.mkdir(parents=True, exist_ok=True)
                snapshot = adopt_original_e1(state, plan, original_hashes, hashes)
                score = score_original_e1(snapshot, state, original_hashes, hashes)
                for arm, member in JOBS:
                    if arm == "A" and member == "tiny":
                        run_job(arm, member, state, original_hashes, hashes,
                                migrate=snapshot, ap_score=score, resume=resume)
                    else:
                        run_job(arm, member, state, original_hashes, hashes, resume=resume)
                    score_best(arm, member, state, original_hashes, hashes)
                final_evaluation(state, original_hashes, hashes)
                state.update(state="completed", stage="done")
                write_status(state)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            write_status(state)
            with (WORK / "logs/controller_error.log").open("a") as stream:
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
        print(json.dumps(old.read_json(WORK / "status.json", {"state": "not_started"}), indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"dry_run": True, "order": [f"{a}_{m}" for a, m in JOBS],
                          "max_epochs": 25, "patience": 5, "min_delta": 0.0001,
                          "plan_exists": PLAN.is_file(), "trainer_exists": TRAINER.is_file(),
                          "evaluator_exists": EVALUATOR.is_file()}, indent=2))
        return 0
    if args.detach:
        (WORK / "logs").mkdir(parents=True, exist_ok=True)
        with (WORK / "logs/controller.log").open("a") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute"]
            if args.resume:
                command.append("--resume")
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Detached PID {child.pid}; status: {WORK / 'STATUS.md'}")
        return 0
    pipeline(args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
