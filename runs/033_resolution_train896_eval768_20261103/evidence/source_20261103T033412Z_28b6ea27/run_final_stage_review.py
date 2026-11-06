"""Finish the user-stopped A/B run from its committed B_base checkpoint.

This controller never starts training. Its timing review uses the frozen legacy
100-image serial protocol and is not a Study-level official cost result.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_tight_crop_ab_pipeline as old
import run_tight_crop_ab_earlystop_pipeline as amended
import run_tight_crop_postreview as post


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "artifacts/new_study_cost_20261102/finalization"
WORK = amended.WORK
RUN_ROOT = amended.RUN_ROOT
NAMES = ("A_tiny", "A_base", "B_tiny", "B_base")
REVIEW_OUTPUT = RUN_ROOT / "final_stage_postreview"
REVIEW_COPY = ROOT / "artifacts/new_study_cost_20261102/ab_review"
RECEIPTS = (old.ARTIFACTS / "pipeline_sources.json", WORK / "sources.json",
            post.CONTROL / "sources.json")
POLL_SECONDS = 15


def publish(state):
    state["updated_utc"] = post.utc_now()
    old.write_json(CONTROL / "status.json", state)
    lines = ["# User-stopped A/B finalization", "",
             f"State: {state['state']}", f"Stage: {state['stage']}",
             f"PID: {state['pid']}", f"Updated UTC: {state['updated_utc']}", "",
             "B_base STOP is a user termination after a fully validated epoch.",
             "The old training controller remains failed; no training is restarted.",
             "Six-candidate timing uses the historical 100-image serial batch-1 proxy, not the new Study-level official rule.",
             f"Review: {REVIEW_COPY}", f"Logs: {CONTROL / 'logs'}"]
    if state.get("gpu_pids"):
        lines.append(f"Active GPU compute PIDs: {state['gpu_pids']}")
    if state.get("error"):
        lines.append(f"Error: {state['error']}")
    (CONTROL / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sources():
    expected_sets = (
        {"docs/TIGHT_CROP_AB_PLAN_20261102.json", *old.SOURCE_NAMES},
        {str(path.relative_to(ROOT)) for path in amended.SOURCE_KEYS},
        {str(path.relative_to(ROOT)) for path in post.SOURCE_PATHS},
    )
    records = []
    for receipt, expected in zip(RECEIPTS, expected_sets):
        row = old.read_json(receipt)
        hashes = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(hashes, dict) or set(hashes) != expected:
            raise RuntimeError(f"Frozen source receipt missing or incomplete: {receipt}")
        for relative, digest in hashes.items():
            if old.sha256(ROOT / relative) != digest:
                raise RuntimeError(f"Frozen source changed: {relative}")
        records.append({"path": str(receipt), "sha256": old.sha256(receipt),
                        "file_count": len(hashes)})
    return records


def active_registered_process(status, marker):
    pid, ticks = status.get("pid"), status.get("start_ticks")
    if not isinstance(pid, int):
        return False
    identity = amended.process_identity(pid)
    return bool(identity and identity["state"] not in ("Z", "X")
                and (ticks is None or identity["start_ticks"] == ticks)
                and marker in identity["command"])


def ready_after_stop():
    training = old.read_json(WORK / "status.json", {})
    prior_review = old.read_json(post.CONTROL / "status.json", {})
    child = training.get("child_pid")
    if isinstance(child, int):
        identity = amended.process_identity(child)
        if (identity and identity["state"] not in ("Z", "X")
                and identity["start_ticks"] == training.get("child_start_ticks")):
            return False
    review_child = prior_review.get("child_pid")
    if isinstance(review_child, int):
        identity = amended.process_identity(review_child)
        if (identity and identity["state"] not in ("Z", "X")
                and identity["start_ticks"] == prior_review.get("child_start_ticks")):
            return False
    return not (active_registered_process(training, "run_tight_crop_ab_earlystop_pipeline.py")
                or active_registered_process(prior_review, "run_tight_crop_postreview.py"))


def stopped_checkpoint():
    training = old.read_json(WORK / "status.json", {})
    prior_review = old.read_json(post.CONTROL / "status.json", {})
    if (training.get("state") != "failed" or "B_base" not in str(training.get("error"))
            or "STOP" not in str(training.get("error"))):
        raise RuntimeError("Old controller did not report the expected B_base STOP")
    if (prior_review.get("state") != "blocked"
            or prior_review.get("stage") != "training_failed"):
        raise RuntimeError("Old postreview did not block on the stopped training")
    folder = amended.run_dir("B", "base")
    status = old.read_json(folder / "status.json", {})
    if (not (folder / "STOP").is_file()
            or status.get("state") not in ("paused", "completed", "early_stopped")
            or status.get("validated_epochs") != status.get("completed_epochs")):
        raise RuntimeError("B_base STOP or validated terminal checkpoint is missing")
    verified = amended.verify_native_run("B", "base", terminal=False)
    return {"user_stop": True, "training_controller_state": training["state"],
            "training_controller_error": training["error"],
            "postreview_controller_state": prior_review["state"],
            "B_base_state": status["state"], "stop_path": str(folder / "STOP"),
            "selection_basis": "best.pt among completed, validated epochs",
            "no_new_selection_logic": True, **verified}


def run_child(command, log_name, state):
    verify_sources()
    path = CONTROL / "logs" / log_name
    with path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{post.utc_now()}] command: {' '.join(map(str, command))}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        state["child_pid"] = child.pid
        publish(state)
        while child.poll() is None:
            publish(state)
            time.sleep(POLL_SECONDS)
        code = child.wait()
        state["child_pid"] = None
        publish(state)
    verify_sources()
    if code:
        raise RuntimeError(f"Stage {state['stage']} exited {code}; see {path}")


def score_receipt(name):
    arm, member = name.split("_")
    path = RUN_ROOT / "scores" / name / "score.json"
    score = old.read_json(path)
    if (not isinstance(score, dict) or score.get("arm") != arm
            or score.get("member") != member
            or score.get("source_kind") != "ap_selected_best"
            or score.get("source_sha256") != old.sha256(amended.run_dir(arm, member) / "best.pt")
            or score.get("test_used") is not False
            or type(score.get("ap_only_macro_auc")) not in (int, float)):
        raise RuntimeError(f"Selected score invalid: {name}")
    return score


def write_once(path, value):
    previous = old.read_json(path)
    if previous is None:
        if path.exists():
            raise RuntimeError(f"Existing non-JSON stage file: {path}")
        old.write_json(path, value)
    elif previous != value:
        raise RuntimeError(f"Existing stage receipt differs: {path}")


def pipeline():
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "logs").mkdir(exist_ok=True)
    with (CONTROL / ".finalization.lock").open("a+") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / "status.json").exists():
            raise RuntimeError("Finalization already started; inspect status before manual recovery")
        state = {"state": "running", "stage": "wait_old_controllers", "pid": os.getpid(),
                 "started_utc": post.utc_now(), "child_pid": None,
                 "gpu_pids": [], "error": None, "test_predicted": False}
        publish(state)
        try:
            receipts = verify_sources()
            write_once(CONTROL / "sources.json", {
                "registered_utc": post.utc_now(), "runner": str(Path(__file__).resolve()),
                "runner_sha256": old.sha256(__file__), "frozen_receipts": receipts})
            lock_paths = (old.ARTIFACTS / ".pipeline.lock",
                          WORK / ".controller.lock", post.CONTROL / ".review.lock")
            while True:
                verify_sources()
                if ready_after_stop():
                    break
                publish(state)
                time.sleep(POLL_SECONDS)
            with open(lock_paths[0], "a+") as first, open(lock_paths[1], "a+") as second, \
                    open(lock_paths[2], "a+") as third:
                locks = (first, second, third)
                while True:
                    acquired = []
                    try:
                        for lock in locks:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            acquired.append(lock)
                    except BlockingIOError:
                        for lock in reversed(acquired):
                            fcntl.flock(lock, fcntl.LOCK_UN)
                        publish(state)
                        time.sleep(POLL_SECONDS)
                        continue
                    break
                if not ready_after_stop():
                    raise RuntimeError("Old controller or child still active after lock acquisition")
                state["stage"] = "wait_gpu_idle"
                while (pids := post.active_gpu_pids()):
                    state["gpu_pids"] = pids
                    publish(state)
                    time.sleep(POLL_SECONDS)
                state["gpu_pids"] = []
                verify_sources()
                stop = stopped_checkpoint()
                write_once(CONTROL / "stop_amendment.json", {
                    "registered_utc": post.utc_now(), "reason": "user-requested STOP after validated B_base epoch",
                    "original_status_preserved": str(WORK / "status.json"), **stop})
                state["B_base"] = stop
                state["stage"] = "mirror_B_base_checkpoint"
                publish(state)
                old.mirror_dir(amended.run_dir("B", "base"), amended.persistent_run("B", "base"))
                for filename, digest in (("last.pt", stop["last_sha256"]),
                                         ("best.pt", stop["best_sha256"])):
                    if old.sha256(amended.persistent_run("B", "base") / filename) != digest:
                        raise RuntimeError(f"Persistent B_base {filename} differs")
                for name in NAMES[:3]:
                    amended.verify_native_run(*name.split("_"), terminal=True)
                    score_receipt(name)
                name = "B_base"
                score_dir = RUN_ROOT / "scores" / name
                state["stage"] = "score_B_base_best"
                publish(state)
                if not score_dir.exists():
                    run_child(["bash", str(amended.WRAPPER), amended.EVALUATOR.name,
                               "--action", "score", "--arm", "B", "--member", "base",
                               "--source", str(amended.run_dir("B", "base") / "best.pt"),
                               "--output", str(score_dir)], "score_B_base.log", state)
                score_receipt(name)
                old.mirror_dir(score_dir, WORK / "scores" / name)
                selected = {candidate: str(RUN_ROOT / "scores" / candidate) for candidate in NAMES}
                selection = {"selected": selected, "amendment_sha256": old.sha256(amended.PLAN),
                             "selection_basis": "user-stopped B_base; AP-best among completed validated epochs; no new selection",
                             "test_used": False}
                state["stage"] = "final_paired_valid"
                publish(state)
                write_once(RUN_ROOT / "selection.json", selection)
                write_once(WORK / "selection.json", selection)
                output = RUN_ROOT / "evaluation"
                if not output.exists():
                    run_child(["bash", str(amended.WRAPPER), amended.EVALUATOR.name,
                               "--action", "final", "--selection", str(RUN_ROOT / "selection.json"),
                               "--output", str(output)], "final_paired_valid.log", state)
                metrics = old.read_json(output / "metrics.json")
                if (not isinstance(metrics, dict) or not (output / "REPORT.md").is_file()
                        or metrics.get("earlystop_plan_sha256") != old.sha256(amended.PLAN)
                        or metrics.get("test_used") is not False):
                    raise RuntimeError("Final paired valid evaluation incomplete or unbound")
                old.mirror_dir(output, WORK / "evaluation")
                state["stage"] = "historical_proxy_review"
                publish(state)
                if not REVIEW_OUTPUT.exists():
                    run_child(["bash", str(amended.WRAPPER), post.REVIEW.name,
                               "--run-root", str(RUN_ROOT), "--output", str(REVIEW_OUTPUT),
                               "--benchmark", "--timing-inputs", str(post.TIMING_INPUTS)],
                              "historical_proxy_review.log", state)
                if not post.review_complete(REVIEW_OUTPUT):
                    raise RuntimeError("Six-candidate historical timing review incomplete")
                post.mirror_directory(REVIEW_OUTPUT, REVIEW_COPY)
                if not post.review_complete(REVIEW_COPY):
                    raise RuntimeError("Persistent six-candidate review incomplete")
                state.update(state="completed", stage="done", completed_utc=post.utc_now())
                publish(state)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            publish(state)
            with (CONTROL / "logs/controller_error.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(old.read_json(CONTROL / "status.json", {"state": "not_started"}), indent=2))
        return
    if not args.execute:
        print(json.dumps({"dry_run": True, "output": str(CONTROL),
                          "training_state": old.read_json(WORK / "status.json", {}).get("state"),
                          "b_base_state": old.read_json(amended.run_dir("B", "base") / "status.json", {}).get("state"),
                          "legacy_timing_only": True}, indent=2))
        return
    if args.detach:
        (CONTROL / "logs").mkdir(parents=True, exist_ok=True)
        with (CONTROL / "logs/controller.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--execute"],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Detached PID {child.pid}; status: {CONTROL / 'STATUS.md'}")
        return
    pipeline()


if __name__ == "__main__":
    main()
