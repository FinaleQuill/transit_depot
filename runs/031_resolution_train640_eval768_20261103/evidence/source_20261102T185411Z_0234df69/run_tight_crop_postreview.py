"""Detached host controller for the authorized post-training review and timing.

Waits for the four-member A/B pipeline and its final evaluation to finish. It
never starts training, terminates a GPU process, or treats partial review output
as complete. Run with --execute --detach after the review/render sources freeze.
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


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts/tight_crop_ab_20261102/earlystop25"
CONTROL = ARTIFACTS / "postreview_controller"
RUN_ROOT = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/earlystop25")
RUN_OUTPUT = RUN_ROOT / "postreview"
PERSISTENT_OUTPUT = ARTIFACTS / "postreview"
TRAINING_STATUS = ARTIFACTS / "status.json"
ORIGINAL_LOCK = ROOT / "artifacts/tight_crop_ab_20261102/.pipeline.lock"
PLAN = ROOT / "docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json"
REVIEW = ROOT / "review_tight_crop_ab_results.py"
RENDER = ROOT / "render_tight_crop_strategy_review.py"
WRAPPER = ROOT / "run_tight_crop_environment.sh"
TIMING_INPUTS = ROOT.parent / "deliveries/034_dual768_proxy_timing_20261029/bundle/inputs"
SOURCE_PATHS = (Path(__file__).resolve(), PLAN, REVIEW, RENDER, WRAPPER,
                *(ROOT / name for name in (
                    "benchmark_local_resources.py", "cost.py", "deploy_kd.py",
                    "inference_inputs.py", "inference_reproducibility.py",
                    "model_overnight.py", "tight_crop.py", "tight_crop_ab_preprocess.py",
                    "tight_crop_ab_evaluate.py", "tight_crop_ab_earlystop_evaluate.py")))
POLL_SECONDS = 60


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_hashes():
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Postreview source missing: {missing}")
    return {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}


def freeze_sources(resume):
    hashes = source_hashes()
    receipt = CONTROL / "sources.json"
    previous = read_json(receipt)
    if previous is not None:
        if not resume or previous.get("sha256") != hashes:
            raise RuntimeError("Postreview source receipt exists or changed; use --resume with frozen sources")
    elif resume:
        raise FileNotFoundError("No postreview source receipt to resume")
    else:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        write_json(receipt, {"registered_utc": utc_now(), "git_head": head,
                             "sha256": hashes, "training_source_receipt":
                             str(ARTIFACTS / "sources.json")})
    return hashes


def check_sources(hashes):
    if source_hashes() != hashes:
        raise RuntimeError("Frozen postreview source changed")


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].strip().split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        return {"state": fields[0], "start_ticks": int(fields[19]), "command": command}
    except (FileNotFoundError, ValueError, IndexError):
        return None


def active_gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"GPU process query failed: {result.stderr.strip()}")
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value or value in ("[Not Found]", "N/A"):
            continue
        try:
            pids.append(int(value))
        except ValueError as error:
            raise RuntimeError(f"Unrecognized nvidia-smi compute PID: {value}") from error
    return pids


def publish(state):
    state["updated_utc"] = utc_now()
    state["elapsed_seconds"] = round(time.monotonic() - state["started_monotonic"], 1)
    payload = {key: value for key, value in state.items() if key != "started_monotonic"}
    write_json(CONTROL / "status.json", payload)
    lines = ["# Post-training A/B review", "",
             f"State: {state['state']}", f"Stage: {state['stage']}",
             f"PID: {state['pid']}", f"Updated UTC: {state['updated_utc']}",
             f"Elapsed: {state['elapsed_seconds']} seconds", "",
             "Runs only after the A/B training, final paired valid evaluation and historical 044 reference complete.",
             "Six fixed candidates receive serial batch-1 end-to-end timing; no training or test prediction.", ""]
    if state.get("gpu_pids"):
        lines.append(f"Waiting for active GPU compute PIDs: {state['gpu_pids']}")
    if state.get("error"):
        lines.append(f"Error: {state['error']}")
    lines.extend(("", f"Review output: {PERSISTENT_OUTPUT}",
                  f"Strategy report: {ARTIFACTS / 'STRATEGY_REVIEW.md'}",
                  f"Logs: {CONTROL / 'logs'}", ""))
    (CONTROL / "STATUS.md").write_text("\n".join(lines))


def wait_for_training(state, hashes):
    while True:
        check_sources(hashes)
        training = read_json(TRAINING_STATUS, {})
        status = training.get("state", "pending")
        if status == "failed":
            state.update(state="blocked", stage="training_failed",
                         error=f"Training controller failed: {training.get('error')}")
            publish(state)
            return False
        if status == "completed":
            if training.get("stage") != "done" or training.get("child_pid") is not None:
                raise RuntimeError("Training completion status retains an active stage or child")
            if not (ARTIFACTS / "evaluation/metrics.json").is_file():
                raise FileNotFoundError("Training marked complete without persistent paired evaluation")
            if not (ARTIFACTS / "evaluation/historical_044/historical_summary.json").is_file():
                raise FileNotFoundError("Training marked complete without historical 044 reference")
            state["training_controller_pid"] = training.get("pid")
            state["training_finished_utc"] = training.get("updated_utc")
            return True
        state["stage"] = f"wait_training_{training.get('stage', status)}"
        publish(state)
        time.sleep(POLL_SECONDS)


def run_child(command, log_path, state, hashes):
    check_sources(hashes)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(map(str, command))}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        identity = process_identity(child.pid)
        state["child_pid"] = child.pid
        state["child_start_ticks"] = identity["start_ticks"] if identity else None
        publish(state)
        source_error = None
        while child.poll() is None:
            try:
                check_sources(hashes)
            except Exception as error:
                source_error = error
            publish(state)
            time.sleep(POLL_SECONDS)
        code = child.wait()
        state["child_pid"] = None
        state["child_start_ticks"] = None
        publish(state)
        if source_error is not None:
            raise source_error
        check_sources(hashes)
        return code


def mirror_directory(source, target):
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256(destination) == sha256(path):
            continue
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        shutil.copy2(path, temporary)
        if sha256(temporary) != sha256(path):
            raise RuntimeError(f"Postreview copy changed: {path}")
        temporary.replace(destination)


def review_complete(output):
    review = read_json(output / "review.json")
    timing = read_json(output / "timing_summary.json")
    results = read_json(output / "results.json")
    if not all(isinstance(item, dict) for item in (review, timing, results)):
        return False
    if not (output / "REPORT.md").is_file():
        raise ValueError("Postreview metrics exist without REPORT.md")
    expected = {"A_tiny", "A_base", "B_tiny", "B_base", "A_fused", "B_fused"}
    if set(review.get("candidate_scores", {})) != expected or set(results.get("candidates", {})) != expected:
        raise ValueError("Postreview candidate scores lack one of six candidates")
    if set(timing) != expected:
        raise ValueError("Postreview timing summary lacks one of six candidates")
    if set(results.get("timing", {})) != expected:
        raise ValueError("Postreview results lack six candidate timings")
    for name in expected:
        row = timing[name]
        if not isinstance(row, dict) or not isinstance(row.get("mean_ms"), (int, float)):
            raise ValueError(f"Postreview timing missing mean for {name}")
        if not (output / "timing" / name / "latency_report.json").is_file():
            raise ValueError(f"Postreview timing report missing for {name}")
    return True


def strategy_complete(path):
    receipt_path = path.with_suffix(".sources.json")
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError("Strategy review and its source receipt must both exist")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("report_sha256") != sha256(path):
        raise ValueError("Strategy review hash differs from its receipt")
    sources = receipt.get("sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Strategy review source receipt is incomplete")
    for source, expected in sources.items():
        if not Path(source).is_file() or sha256(source) != expected:
            raise ValueError(f"Strategy review input changed: {source}")
    return True


def pipeline(resume):
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "logs").mkdir(exist_ok=True)
    with (CONTROL / ".review.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = freeze_sources(resume)
        previous = read_json(CONTROL / "status.json", {}) if resume else {}
        prior_child = previous.get("child_pid")
        prior_ticks = previous.get("child_start_ticks")
        if isinstance(prior_child, int) and isinstance(prior_ticks, int):
            identity = process_identity(prior_child)
            if (identity and identity["state"] not in ("Z", "X")
                    and identity["start_ticks"] == prior_ticks):
                raise RuntimeError(f"Previous postreview child PID {prior_child} is still active")
        state = {"state": "running", "stage": "startup", "pid": os.getpid(),
                 "started_utc": previous.get("started_utc", utc_now()),
                 "started_monotonic": time.monotonic(), "child_pid": None,
                 "child_start_ticks": None, "gpu_pids": [], "error": None,
                 "source_receipt": str(CONTROL / "sources.json"),
                 "test_predicted": False}
        publish(state)
        try:
            if not wait_for_training(state, hashes):
                return
            with ORIGINAL_LOCK.open("a+") as training_lock:
                state["stage"] = "wait_training_lock_release"
                publish(state)
                fcntl.flock(training_lock, fcntl.LOCK_EX)
                training = read_json(TRAINING_STATUS, {})
                if training.get("state") != "completed" or training.get("child_pid") is not None:
                    raise RuntimeError("Training status changed after obtaining pipeline lock")
                while True:
                    state["stage"] = "wait_gpu_idle"
                    state["gpu_pids"] = active_gpu_pids()
                    publish(state)
                    if not state["gpu_pids"]:
                        break
                    time.sleep(POLL_SECONDS)
                state["stage"] = "review_and_benchmark"
                state["gpu_pids"] = []
                publish(state)
                if not review_complete(RUN_OUTPUT):
                    if RUN_OUTPUT.exists():
                        raise FileExistsError(f"Partial postreview exists; preserve for inspection: {RUN_OUTPUT}")
                    command = ["bash", str(WRAPPER), REVIEW.name,
                               "--run-root", str(RUN_ROOT), "--output", str(RUN_OUTPUT),
                               "--benchmark", "--timing-inputs", str(TIMING_INPUTS)]
                    code = run_child(command, CONTROL / "logs/review_and_benchmark.log", state, hashes)
                    if code != 0 or not review_complete(RUN_OUTPUT):
                        raise RuntimeError("Postreview benchmark failed or incomplete")
                mirror_directory(RUN_OUTPUT, PERSISTENT_OUTPUT)
                if not review_complete(PERSISTENT_OUTPUT):
                    raise RuntimeError("Persistent postreview copy is incomplete")
                state["stage"] = "render_strategy_review"
                publish(state)
                strategy = ARTIFACTS / "STRATEGY_REVIEW.md"
                if not strategy_complete(strategy):
                    command = [sys.executable, str(RENDER), "--review-dir", str(PERSISTENT_OUTPUT),
                               "--output", str(strategy)]
                    code = run_child(command, CONTROL / "logs/render_strategy_review.log", state, hashes)
                    if code != 0 or not strategy_complete(strategy):
                        raise RuntimeError("Strategy review rendering failed")
                state.update(state="completed", stage="done", completed_utc=utc_now())
                publish(state)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            publish(state)
            if RUN_OUTPUT.exists():
                mirror_directory(RUN_OUTPUT, PERSISTENT_OUTPUT)
            with (CONTROL / "logs/controller_error.log").open("a") as log:
                traceback.print_exc(file=log)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_json(CONTROL / "status.json", {"state": "not_started"}), indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"dry_run": True, "training_state": read_json(TRAINING_STATUS, {}).get("state"),
                          "review_source_exists": REVIEW.is_file(), "render_source_exists": RENDER.is_file(),
                          "timing_inputs_exist": TIMING_INPUTS.is_dir(),
                          "output": str(PERSISTENT_OUTPUT)}, indent=2))
        return 0
    if args.detach:
        (CONTROL / "logs").mkdir(parents=True, exist_ok=True)
        with (CONTROL / "logs/controller.log").open("a") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute"]
            if args.resume:
                command.append("--resume")
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Detached postreview PID {child.pid}; status: {CONTROL / 'STATUS.md'}")
        return 0
    pipeline(args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
