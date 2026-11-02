"""Wait for finalization, then run fixed 043/044/045 Study cost previews.

This queue never trains or predicts the official test split. Default invocation
is a CPU-only dry run. --execute --detach starts the one-shot background queue.
"""
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone

import run_tight_crop_ab_earlystop_pipeline as amended
import run_tight_crop_postreview as post


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "artifacts/new_study_cost_20261102/study_preview"
FINAL = ROOT / "artifacts/new_study_cost_20261102/finalization"
LOCAL = Path("/tmp/cloud_student_runs_20261030/study_batch_preview_20261102")
SAMPLE = LOCAL / "cost_data_test"
SAMPLE_RECEIPT = ROOT / "artifacts/new_study_cost_20261102/SAMPLE_LOCAL_COPY.json"
PREVIEW = ROOT / "benchmark_study_batch_preview.py"
SUMMARIZER = ROOT / "artifacts/new_study_cost_20261102/summarize_study_batch_previews.py"
WRAPPER = ROOT / "run_tight_crop_environment.sh"
MODELS = {
    "043": (ROOT / "artifacts/kd_v1/bounded_finetune_20261101/reviewed_models/full_decode/B",
            "043_base_B", "ec8d155fdf1b4a8ca5de7c914ab098efa5d75c449f3a986e950a811d0c632011"),
    "044": (ROOT.parent / "deliveries/044_tiny_base_class1B_draft768_20261031/model",
            "044_A_B_draft768", "f722b9c3e0e62ac0a675c49163dfac58bc1b52481ded8f2e9e05c1602c2fe92d"),
    "045": (ROOT / "artifacts/kd_v1/bounded_finetune_20261101/reviewed_models/full_decode",
            "045_X1_B_full_decode", "09f2c96debb6afbb83e5a99c4c1e80b5fcbb6a2a87c343eb4c70c32659ec563d"),
}
POLL_SECONDS = 15


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def publish(state):
    state["updated_utc"] = now()
    write_json(CONTROL / "status.json", state)


def source_hashes():
    paths = [Path(__file__).resolve(), PREVIEW, SUMMARIZER, WRAPPER, SAMPLE_RECEIPT,
             FINAL / "sources.json"]
    paths += [directory / "config.json" for directory, _, _ in MODELS.values()]
    hashes = {str(path): digest(path) for path in paths}
    for label, (directory, _, expected) in MODELS.items():
        if hashes[str(directory / "config.json")] != expected:
            raise RuntimeError(f"Frozen {label} config SHA changed")
    return hashes


def verify_sources(expected):
    for name, sha in expected.items():
        if digest(name) != sha:
            raise RuntimeError(f"Queue source changed: {name}")


def verify_samples():
    original = read_json(SAMPLE_RECEIPT)
    local_receipt = read_json(LOCAL / "sample_receipt.json")
    if (original != local_receipt or original.get("count") != 28
            or original.get("local_root") != str(LOCAL)):
        raise RuntimeError("Local sample receipt differs from its archived source")
    expected = {}
    for entry in original["files"]:
        relative = Path(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[:1] != ("cost_data_test",)
                or relative.suffix.lower() not in (".jpg", ".jpeg", ".png")):
            raise RuntimeError("Unsafe or unexpected sample path")
        if relative in expected:
            raise RuntimeError("Duplicate sample path")
        expected[relative] = entry
    actual = {path.relative_to(LOCAL) for path in SAMPLE.rglob("*") if path.is_file()}
    if set(expected) != actual or len(expected) != 28:
        raise RuntimeError("Local sample files differ from the 28-file receipt")
    for relative, entry in expected.items():
        path = LOCAL / relative
        if (path.is_symlink() or path.stat().st_size != entry["bytes"]
                or digest(path) != entry["sha256"]):
            raise RuntimeError(f"Local sample SHA/size mismatch: {relative}")
    return {"files": len(expected), "bytes": sum(entry["bytes"] for entry in expected.values()),
            "receipt_sha256": digest(SAMPLE_RECEIPT)}


def finalizer_exited(status):
    pid = status.get("pid")
    if not isinstance(pid, int) or pid != 83862:
        raise RuntimeError("Finalization PID differs from the registered controller 83862")
    identity = amended.process_identity(pid)
    return not (identity and identity["state"] not in ("Z", "X")
                and "run_final_stage_review.py" in identity["command"])


def finalizer_lock_free():
    lock_path = FINAL / ".finalization.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock, fcntl.LOCK_UN)
    return True


def wait_until_ready(state):
    while True:
        status = read_json(FINAL / "status.json")
        if status.get("state") == "failed":
            raise RuntimeError(f"Finalization failed: {status.get('error')}")
        if status.get("state") not in ("running", "completed"):
            raise RuntimeError(f"Unexpected finalization state: {status.get('state')}")
        if (status["state"] == "completed"
                and (status.get("stage") != "done" or status.get("test_predicted") is not False)):
            raise RuntimeError("Finalization completed without the expected done/no-test receipt")
        if (status["state"] == "completed" and status.get("stage") == "done"
                and status.get("test_predicted") is False
                and finalizer_exited(status) and finalizer_lock_free()):
            pids = post.active_gpu_pids()
            state["gpu_pids"] = pids
            if not pids:
                state["gpu_pids"] = []
                publish(state)
                return
        publish(state)
        time.sleep(POLL_SECONDS)


def wait_gpu_idle(state):
    state["stage"] = "wait_gpu_idle"
    while True:
        pids = post.active_gpu_pids()
        state["gpu_pids"] = pids
        publish(state)
        if not pids:
            return
        time.sleep(POLL_SECONDS)


def verify_preview_output(label, output):
    report_path = output / "preview.json"
    report = read_json(report_path)
    _, expected_model, expected_config = MODELS[label]
    routes = report.get("routes", {})
    if (report.get("model") != expected_model
            or report.get("config_sha256") != expected_config
            or report.get("rounds") != 3
            or report.get("official_cost_certification") is not False
            or len(report.get("comparisons", [])) != 3
            or any(len(routes.get(route, [])) != 3 for route in
                   ("serial_image_reference", "whole_study_batch"))):
        raise RuntimeError(f"Incomplete or wrong {label} preview report")
    files = {report_path}
    for route in ("serial_image_reference", "whole_study_batch"):
        for item in routes[route]:
            if (item.get("studies") != 10 or item.get("images") != 18
                    or len(item.get("per_study", [])) != 10):
                raise RuntimeError(f"Incomplete {label} Study timing coverage")
            for field in ("study_csv", "image_csv"):
                path = Path(item[field]).resolve(strict=True)
                if path.parent != output.resolve() or path.suffix != ".csv":
                    raise RuntimeError(f"Unexpected {label} output path: {path}")
                files.add(path)
    if len(files) != 13 or {p for p in output.iterdir() if p.is_file()} != files:
        raise RuntimeError(f"Unexpected or missing {label} preview files")
    return sorted(files)


def mirror_preview(label, output):
    files = verify_preview_output(label, output)
    destination = CONTROL / label
    destination.mkdir(exist_ok=False)
    entries = []
    for source in files:
        target = destination / source.name
        shutil.copyfile(source, target)
        sha = digest(source)
        if digest(target) != sha:
            raise RuntimeError(f"Mirrored preview differs: {source}")
        entries.append({"file": source.name, "bytes": source.stat().st_size, "sha256": sha})
    write_json(destination / "receipt.json", {"model": label, "files": entries,
                                               "source": str(output), "mirrored_utc": now()})
    return destination


def run_preview(label, state, hashes):
    directory, _, _ = MODELS[label]
    output = LOCAL / label
    if output.exists() or (CONTROL / label).exists():
        raise FileExistsError(f"Existing {label} preview; no automatic retry")
    verify_sources(hashes)
    before = verify_samples()
    wait_gpu_idle(state)
    state["stage"] = f"preview_{label}"
    state["sample_before"] = before
    publish(state)
    command = ["bash", str(WRAPPER), PREVIEW.name, "--model-dir", str(directory),
               "--pictures-100", str(SAMPLE / "pictures_100"),
               "--pictures-5", str(SAMPLE / "pictures_5"),
               "--output-dir", str(output), "--execute"]
    log_path = CONTROL / "logs" / f"preview_{label}.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"[{now()}] command: {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        state["child_pid"] = child.pid
        publish(state)
        while child.poll() is None:
            time.sleep(POLL_SECONDS)
            publish(state)
        code = child.wait()
    state["child_pid"] = None
    publish(state)
    after = verify_samples()
    if after != before:
        raise RuntimeError(f"Sample receipt changed across {label} preview")
    verify_sources(hashes)
    if code:
        raise RuntimeError(f"{label} preview exited {code}; see {log_path}")
    mirror = mirror_preview(label, output)
    state["outputs"][label] = {"local": str(output), "mirrored": str(mirror),
                               "preview_sha256": digest(mirror / "preview.json"),
                               "sample_before_after": before}
    publish(state)


def pipeline():
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "logs").mkdir(exist_ok=True)
    with (CONTROL / ".queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status_path = CONTROL / "status.json"
        if status_path.exists():
            prior = read_json(status_path)
            if prior.get("state") != "launching" or prior.get("pid") != os.getpid():
                raise RuntimeError("Study preview queue already started; no automatic retry")
        state = {"state": "running", "stage": "wait_finalization", "pid": os.getpid(),
                 "started_utc": now(), "child_pid": None, "gpu_pids": [],
                 "outputs": {}, "error": None, "test_predicted": False}
        publish(state)
        try:
            if any((LOCAL / label).exists() or (CONTROL / label).exists() for label in MODELS):
                raise FileExistsError("An output directory already exists; no automatic retry")
            hashes = source_hashes()
            write_json(CONTROL / "sources.json", {"created_utc": now(), "sha256": hashes})
            state["source_receipt_sha256"] = digest(CONTROL / "sources.json")
            state["sample_before_queue"] = verify_samples()
            publish(state)
            wait_until_ready(state)
            for label in MODELS:
                run_preview(label, state, hashes)
            state["stage"] = "summarize"
            publish(state)
            command = [sys.executable, str(SUMMARIZER)]
            for label in MODELS:
                command += [f"--preview-{label}", str(CONTROL / label / "preview.json")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f"Summary failed: {result.stderr.strip()}")
            summary = json.loads(result.stdout)
            write_json(CONTROL / "SUMMARY.json", summary)
            state["summary_sha256"] = digest(CONTROL / "SUMMARY.json")
            state["sample_after_queue"] = verify_samples()
            if state["sample_after_queue"] != state["sample_before_queue"]:
                raise RuntimeError("Sample receipt changed across queue")
            verify_sources(hashes)
            state.update(state="completed", stage="done", completed_utc=now())
            publish(state)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            publish(state)
            with (CONTROL / "logs/controller_error.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            raise


def detach():
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "logs").mkdir(exist_ok=True)
    with (CONTROL / ".queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / "status.json").exists():
            raise RuntimeError("Study preview queue already started; inspect status")
        with (CONTROL / "logs/queue.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--execute"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        publish({"state": "launching", "stage": "waiting_for_child", "pid": child.pid,
                 "started_utc": now(), "error": None})
    print(json.dumps({"detached_pid": child.pid,
                      "status": str(CONTROL / "status.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_json(CONTROL / "status.json") if
                         (CONTROL / "status.json").is_file() else
                         {"state": "not_started"}, indent=2))
    elif not args.execute:
        if args.detach:
            parser.error("--detach requires --execute")
        print(json.dumps({"dry_run": True, "finalization": str(FINAL / "status.json"),
                          "models": list(MODELS), "sample": str(SAMPLE),
                          "output": str(CONTROL), "will_start_gpu": False}, indent=2))
    elif args.detach:
        detach()
    else:
        pipeline()


if __name__ == "__main__":
    main()
