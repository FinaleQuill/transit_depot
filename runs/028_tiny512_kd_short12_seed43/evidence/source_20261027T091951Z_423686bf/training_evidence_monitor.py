"""Preserve training evidence without loading checkpoints, models, or images.

watch observes the already-running teacher from small status files. run captures
future command output from process start. Neither command starts training unless
the caller explicitly supplies a training command to run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from audit_trail import read_checkpoint_audit, record_checkpoint

PROJECT = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def snapshot_source(project, output):
    """Archive code, requirements and package metadata, never medical images."""
    project, output = Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (list(project.glob("*.py")) + list(project.glob("*.sh"))
             + list(project.glob("requirements*.txt")) + [project / "AGENTS.md"])
    files = []
    for source in sorted(set(paths)):
        if not source.is_file():
            continue
        data = source.read_bytes()
        target = output / source.name
        target.write_bytes(data)
        files.append({"path": source.name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "source_mtime_ns": source.stat().st_mtime_ns})
    versions = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                       if d.metadata.get("Name")}, key=str.lower)
    (output / "environment_lock.txt").write_text("\n".join(versions) + "\n")
    write_json(output / "source_manifest.json", {
        "captured_at": utc_now(), "python_executable": sys.executable,
        "python_version": sys.version, "files": files,
        "claim": "On-disk source and environment captured at the declared time."})
    return output


def observe_teacher_checkpoints(run_dir, source_version=None):
    """Use the trainer's completed-save hash; only stat the checkpoint itself."""
    run_dir = Path(run_dir)
    observations = []
    for fold in range(5):
        output = run_dir / f"fold_{fold}"
        status_path = output / "status.json"
        if not status_path.exists():
            continue
        before = status_path.read_bytes()
        status = json.loads(before)
        epoch = status.get("completed_epochs", 0)
        digest = status.get("last_checkpoint_sha256")
        checkpoint = output / "last.pt"
        if not epoch or not digest or not checkpoint.exists():
            continue
        stat = checkpoint.stat()
        committed = datetime.fromisoformat(status["updated_utc"]).timestamp()
        # During the next atomic save last.pt can be newer than the old status.
        # Do not bind that new file's metadata to the old completed epoch/hash.
        if stat.st_mtime > committed + 0.001 or status_path.read_bytes() != before:
            continue
        row = record_checkpoint(checkpoint, digest, epoch=epoch, role="last",
            source_version=source_version, evidence_source="external_status_observer",
            observed_status_at=status["updated_utc"],
            checkpoint_metadata={"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        observations.append({"fold": fold, "epoch": epoch, "sha256": digest,
                             "record_sha256": row["record_sha256"]})
        if status.get("training_complete") and status.get("best_checkpoint_sha256"):
            best = output / "best.pt"
            best_stat = best.stat() if best.exists() else None
            if (best_stat is not None and best_stat.st_mtime <= committed + 0.001
                    and status_path.read_bytes() == before):
                record_checkpoint(best, status["best_checkpoint_sha256"], epoch=epoch,
                    role="best", source_version=source_version,
                    evidence_source="external_status_observer",
                    observed_status_at=status["updated_utc"],
                    checkpoint_metadata={"size_bytes": best_stat.st_size,
                                         "mtime_ns": best_stat.st_mtime_ns})
    return observations


def runner_alive(run_dir):
    lock = Path(run_dir) / ".teacher_runner.lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return any(Path(p.decode(errors="replace")).name == "run_teachers.py" for p in cmdline if p)
    except (ValueError, FileNotFoundError, ProcessLookupError):
        return False


def checkpoint_coverage(run_dir):
    result = {}
    for fold in range(5):
        output = Path(run_dir) / f"fold_{fold}"
        history = output / "history.jsonl"
        if not history.exists():
            continue
        completed = [json.loads(line)["epoch"] for line in history.read_text().splitlines() if line.strip()]
        ledger = output / "checkpoint_audit.jsonl"
        rows = read_checkpoint_audit(ledger) if ledger.exists() else []
        recorded = {r["epoch"] for r in rows if r.get("event_type") == "checkpoint_saved"
                    and r.get("role") == "last"}
        result[str(fold)] = {"completed_epochs": completed, "recorded_epochs": sorted(recorded),
                            "missing_epochs": sorted(set(completed) - recorded)}
    return result


def refresh_manifest(run_dir):
    """Refresh completion and discover native log receipts; preserve provenance."""
    run_dir = Path(run_dir)
    path = run_dir / "evidence/evidence_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    original = json.dumps(manifest, sort_keys=True)
    archives = {item["id"]: item for item in manifest.get("source_archives", [])}
    logs = {item["id"]: item for item in manifest.get("console_logs", [])}
    for item in manifest.get("runs", []):
        folder = Path(item["config"]).parent
        status_path = folder / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            complete = (status.get("training_complete") is True if item["role"] == "teacher"
                        else status.get("state") in ("completed", "early_stopped"))
            item["complete"] = complete
            item["completed_epochs"] = status.get("completed_epochs", 0)
            if complete and item["role"] == "student":
                item["expected_epochs"] = status["completed_epochs"]
        receipts = list((folder / "evidence").glob("command_*.json"))
        if item["role"] == "teacher":
            for parent_receipt in (run_dir / "evidence").glob("command_*.json"):
                parent_command = json.loads(parent_receipt.read_text()).get("command", [])
                if any(Path(arg).name == "run_teachers.py" for arg in parent_command):
                    receipts.append(parent_receipt)
        for receipt_path in sorted(receipts, key=lambda p: json.loads(p.read_text()).get("started_at", "")):
            receipt = json.loads(receipt_path.read_text())
            archive_path = Path(receipt["source_archive"])
            if not archive_path.is_absolute():
                archive_path = Path(receipt["cwd"]) / archive_path
            archive_id = archive_path.name
            archives.setdefault(archive_id, {
                "id": archive_id, "path": str(archive_path),
                "captured_at": receipt["started_at"],
                "description": "Captured immediately before this explicit CLI invocation."})
            log_id = receipt_path.stem
            log_path = Path(receipt["console_log"])
            if not log_path.is_absolute():
                log_path = Path(receipt["cwd"]) / log_path
            logs.setdefault(log_id, {"id": log_id, "path": str(log_path),
                "capture_method": "subprocess_stdout_stderr_tee",
                "coverage_from_start": "complete", "run_ids": [item["id"]],
                "command_receipt": str(receipt_path)})
            if item["id"] not in logs[log_id]["run_ids"]:
                logs[log_id]["run_ids"].append(item["id"])
            item["source_archive_id"] = archive_id
            timeline = manifest.setdefault("source_version_timeline", [])
            if not any(t.get("run_id") == item["id"] and t.get("source_archive_id") == archive_id
                       for t in timeline):
                timeline.append({"run_id": item["id"], "source_archive_id": archive_id,
                    "applies_to": "future_processes", "command_receipt": str(receipt_path),
                    "note": "Actual command-start source snapshot; preserved alongside earlier versions."})
            coverage = {c["run_id"]: c for c in manifest.get("console_coverage", [])}
            prior = coverage.get(item["id"], {})
            # A resume receipt only covers that invocation. It cannot fill a gap
            # in an earlier invocation's logs. Legacy receipts lacking this new
            # field are treated as unknown rather than complete-from-first-run.
            starts_fresh = receipt.get("preexisting_run_artifacts") == []
            full = prior.get("coverage_from_start") == "complete" or starts_fresh
            explanation = prior.get("coverage_explanation", "")
            receipt_note = "Native receipts cover each recorded invocation; an earlier gap is not erased."
            if receipt_note not in explanation:
                explanation = (explanation + " " + receipt_note).strip()
            coverage[item["id"]] = {
                "run_id": item["id"], "coverage_from_start": "complete" if full else "partial",
                "log_ids": sorted(set(prior.get("log_ids", []) + [log_id])),
                "coverage_explanation": explanation}
            manifest["console_coverage"] = list(coverage.values())
    manifest["source_archives"] = list(archives.values())
    manifest["console_logs"] = list(logs.values())
    if json.dumps(manifest, sort_keys=True) != original:
        manifest["updated_at_utc"] = utc_now()
        write_json(path, manifest)


def watch(args):
    evidence = args.run_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    with (evidence / ".monitor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (evidence / "monitor.pid").write_text(str(os.getpid()) + "\n")
        source = {"source_archive_id": args.source_archive_id,
                  "claim": "Source archive described in evidence_manifest.json; no checkpoint bytes read."}
        if args.pane:
            capture = json.loads((evidence / "console_capture.json").read_text())
            if not capture.get("attached_at"):
                raise ValueError("The live console capture must be attached before starting its monitor")
        failures = 0
        previous_summary = None
        last_heartbeat = 0.0
        while True:
            try:
                observed = observe_teacher_checkpoints(args.run_dir, source)
                coverage = checkpoint_coverage(args.run_dir)
                alive = runner_alive(args.run_dir)
                refresh_manifest(args.run_dir)
                status = {
                    "pid": os.getpid(), "heartbeat_utc": utc_now(),
                    "runner_alive": alive, "observations": observed,
                    "checkpoint_coverage": coverage, "consecutive_errors": 0,
                    "checkpoint_bytes_read": 0, "gpu_used": False}
                summary = json.dumps([observed, coverage, alive], sort_keys=True)
                if summary != previous_summary or time.monotonic() - last_heartbeat >= 30 or not alive:
                    write_json(evidence / "monitor_status.json", status)
                    previous_summary = summary
                    last_heartbeat = time.monotonic()
                failures = 0
                if args.once or not alive:
                    return 0
            except Exception as error:
                failures += 1
                write_json(evidence / "monitor_status.json", {
                    "pid": os.getpid(), "heartbeat_utc": utc_now(),
                    "consecutive_errors": failures, "error": f"{type(error).__name__}: {error}"})
                print(f"{utc_now()} evidence monitor error: {error}", flush=True)
                if args.once:
                    raise
            time.sleep(args.interval)


def run_logged(args):
    """Mirror one explicit command to a durable file and return its exit code."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("An explicit command after -- is required")
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    preexisting = sorted(p.name for p in evidence.parent.iterdir() if p.name != "evidence")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    archive = snapshot_source(PROJECT, evidence / ("source_" + stamp))
    log_path = evidence / ("console_" + stamp + ".log")
    receipt_path = evidence / ("command_" + stamp + ".json")
    receipt = {"started_at": utc_now(), "command": command, "cwd": str(Path.cwd()),
               "source_archive": str(archive), "console_log": str(log_path),
               "preexisting_run_artifacts": preexisting,
               "coverage_from_start": "complete", "exit_code": None}
    write_json(receipt_path, receipt)
    child = None
    handlers = {}
    try:
        with log_path.open("xb", buffering=0) as log:
            child_env = dict(os.environ, KD_TRAINING_LOG_ACTIVE="1", PYTHONUNBUFFERED="1")
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=None, bufsize=0, env=child_env)
            receipt["child_pid"] = child.pid
            write_json(receipt_path, receipt)
            # Keep this recorder alive to drain output, forwarding direct signals
            # as well. Repeated graceful-stop requests are safe in the trainers.
            def keep_recording(_signum, _frame):
                if child.poll() is None:
                    child.send_signal(_signum)
            handlers = {s: signal.signal(s, keep_recording) for s in (signal.SIGINT, signal.SIGTERM)}
            while True:
                block = os.read(child.stdout.fileno(), 65536)
                if not block:
                    break
                log.write(block)
                try:
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    pass  # Preserve the evidence even if the viewer disconnects.
            code = child.wait()
            os.fsync(log.fileno())
        receipt.update({"ended_at": utc_now(), "exit_code": code,
                        "console_bytes": log_path.stat().st_size})
        write_json(receipt_path, receipt)
        return code if code >= 0 else 128 - code
    finally:
        if child is not None and child.stdout is not None:
            child.stdout.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    observer = subs.add_parser("watch")
    observer.add_argument("--run-dir", type=Path, required=True)
    observer.add_argument("--source-archive-id", default="source_before_patch")
    observer.add_argument("--pane")
    observer.add_argument("--interval", type=float, default=2.0)
    observer.add_argument("--once", action="store_true")
    recorder = subs.add_parser("run")
    recorder.add_argument("--evidence-dir", type=Path, required=True)
    recorder.add_argument("command", nargs=argparse.REMAINDER)
    refresher = subs.add_parser("refresh")
    refresher.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "watch":
        if args.interval < 1:
            parser.error("Observer interval must be at least one second")
        return watch(args)
    if args.mode == "refresh":
        refresh_manifest(args.run_dir)
        return 0
    return run_logged(args)


if __name__ == "__main__":
    raise SystemExit(main())
