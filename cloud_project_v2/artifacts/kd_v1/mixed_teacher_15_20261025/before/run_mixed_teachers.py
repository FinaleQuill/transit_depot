"""Execute the approved 20/12/12/12/12 teacher transition without restarting fold0.

The retired scheduler reaches a planned handoff guard before starting fold1.
Only after that scheduler and its GPU child exit do validation and the new run
start. Default invocation is a read-only preview; --execute is explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from teacher_collection import read_teacher_collection
from training_evidence_monitor import refresh_manifest, run_logged, write_json

PROJECT = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/plan.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return {"pid": int(pid), "state": stat[0], "ppid": int(stat[1]),
                "start_ticks": int(stat[19]),
                "cmdline_sha256": hashlib.sha256(b"\0".join(args)).hexdigest()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(reference):
    actual = identity(reference["pid"])
    return bool(actual and actual["state"] != "Z"
                and actual["start_ticks"] == reference["start_ticks"]
                and actual["cmdline_sha256"] == reference["cmdline_sha256"])


def read_plan(path):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_text())
    if plan.get("artifact_type") != "kd_mixed_teacher_transition" or plan.get("schema_version") != 1:
        raise ValueError("Unsupported mixed teacher plan")
    if plan.get("fixed_epochs") != {"0":20,"1":12,"2":12,"3":12,"4":12}:
        raise ValueError("Only the approved frozen 20/12/12/12/12 schedule is accepted")
    original = Path(plan["original_run"])
    remaining = Path(plan["remaining_run"])
    if original.resolve() == remaining.resolve():
        raise ValueError("The twelve-epoch run must have its own directory")
    remaining_metadata = json.loads((remaining / "teacher_run.json").read_text())
    command = plan["remaining_command"]
    if (remaining_metadata.get("epochs") != 12
            or command[command.index("--epochs") + 1] != "12"
            or Path(command[command.index("--run-dir") + 1]).resolve() != remaining.resolve()
            or command[command.index("--folds") + 1:command.index("--execute")] != ["1","2","3","4"]):
        raise ValueError("The remaining four teachers must retain their separately frozen twelve-epoch budget")
    if hashlib.sha256((original / "teacher_run.json").read_bytes()).hexdigest() != plan["original_run_metadata_sha256"]:
        raise ValueError("Original fixed twenty-epoch protocol changed")
    collection = read_teacher_collection(plan["teacher_collection"], require_checkpoints=False)
    if [entry["epochs"] for entry in collection["teachers"]] != [20,12,12,12,12]:
        raise ValueError("Teacher collection differs from the approved budget")
    marker = json.loads((original / "evidence/planned_handoff.json").read_text())
    if (marker.get("artifact_type") != "kd_planned_handoff"
            or Path(marker["original_run"]).resolve() != original.resolve()
            or marker.get("blocked_folds") != [1,2,3,4]):
        raise ValueError("The old scheduler's planned handoff guard is not installed")
    if hashlib.sha256((original / "evidence/planned_handoff.json").read_bytes()).hexdigest() != plan["handoff_marker_sha256"]:
        raise ValueError("The planned handoff notice changed")
    retained = Path(plan["retained_epoch13_checkpoint"])
    receipt = json.loads(Path(plan["retained_epoch13_receipt"]).read_text())
    if receipt.get("epoch") != 13 or retained.stat().st_size != receipt["source_stat"]["size_bytes"]:
        raise ValueError("Retained epoch13 checkpoint receipt is inconsistent")
    return plan


def validate_comparison(report, plan):
    """A report file alone must never authorize the next training stage."""
    if (report.get("complete") is not True or report.get("diagnostic_only") is not True
            or report.get("checkpoint_selection_performed") is not False
            or report.get("thresholds_fitted") is not False
            or report.get("manifest_sha256") != plan["valid_manifest_sha256"]
            or report.get("fold_plan_sha256") != plan["fold_plan_sha256"]):
        raise ValueError("Teacher validation report is incomplete or bound to different data")
    results, reports = report.get("results", []), report.get("reports", [])
    if len(results) != 2 or len(reports) != 2 or [r.get("epoch") for r in results] != [13,20]:
        raise ValueError("The comparison must contain exactly the retained epoch13 and final epoch20")
    status = json.loads((Path(plan["original_run"]) / "fold_0/status.json").read_text())
    expected_hashes = [plan["retained_epoch13_sha256"], status.get("best_checkpoint_sha256")]
    expected_paths = [Path(plan["retained_epoch13_checkpoint"]).resolve(),
                      (Path(plan["original_run"]) / "fold_0/best.pt").resolve()]
    for index, result in enumerate(results):
        detail = reports[index]
        auc = result.get("macro_auc")
        if (not expected_hashes[index] or result.get("checkpoint_sha256") != expected_hashes[index]
                or Path(result.get("checkpoint", "")).resolve() != expected_paths[index]
                or type(auc) not in (int,float) or not math.isfinite(auc) or not 0 <= auc <= 1
                or detail.get("complete") is not True or detail.get("fold") != 0
                or detail.get("num_images") != 1422 or detail.get("num_studies") != 859
                or detail.get("image_size") != 768):
            raise ValueError("Teacher validation checkpoint, coverage or score is inconsistent")
    return report


def register_evidence(root):
    """Attribute this controller's complete log/source receipt to future folds."""
    root = Path(root)
    manifest_path = root / "evidence/evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in sorted((root / "evidence").glob("command_*.json")):
        receipt = json.loads(path.read_text())
        if not any(Path(arg).name == Path(__file__).name for arg in receipt.get("command", [])):
            continue
        archive_id = Path(receipt["source_archive"]).name
        if not any(x["id"] == archive_id for x in manifest["source_archives"]):
            manifest["source_archives"].append({"id":archive_id,"path":receipt["source_archive"],
                "captured_at":receipt["started_at"],"description":"Source at approved transition controller start"})
        log_id = path.stem
        if not any(x["id"] == log_id for x in manifest["console_logs"]):
            manifest["console_logs"].append({"id":log_id,"path":receipt["console_log"],
                "capture_method":"subprocess_stdout_stderr_tee","coverage_from_start":"complete",
                "run_ids":[f"teacher_fold_{i}" for i in range(1,5)]})
        for item in manifest["runs"]:
            if item["id"] not in {f"teacher_fold_{i}" for i in range(1,5)}:
                continue
            item["source_archive_id"] = archive_id
            if not any(t.get("run_id") == item["id"] and t.get("source_archive_id") == archive_id
                       for t in manifest["source_version_timeline"]):
                manifest["source_version_timeline"].append({"run_id":item["id"],
                    "source_archive_id":archive_id,"applies_to":"future_processes",
                    "note":"Frozen twelve-epoch training after original fold0; full controller log and command receipt retained."})
    write_json(manifest_path, manifest)


def execute(plan):
    root = Path(plan["collection_root"])
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    state_path = root / "transition_status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("phase") == "completed":
        read_teacher_collection(plan["teacher_collection"], require_checkpoints=True)
        print("Approved mixed teacher collection is already complete.", flush=True)
        return 0
    with (evidence / ".transition.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (evidence / "transition.pid").write_text(str(os.getpid()) + "\n")
        register_evidence(root)

        def update(phase, **values):
            state.update(phase=phase, updated_at_utc=now(), controller_pid=os.getpid(), **values)
            write_json(state_path, state)

        def child_command(stage, command):
            prior = state.get("command") or {}
            reference = prior.get("process")
            if prior.get("stage") == stage and reference and same_process(reference):
                print(f"Reattached to existing {stage} process {reference['pid']}.", flush=True)
                while same_process(reference):
                    time.sleep(5)
                return None
            print("Starting " + stage + ": " + repr(command), flush=True)
            update(stage, command={"stage":stage,"args":command,"process":None,
                                   "spawn_intent_at_utc":now()})
            # Keep exclusive ownership in a live stage even if this controller
            # dies between spawn and recording its PID. A restart cannot launch
            # a second GPU stage while the orphaned original stage is active.
            process = subprocess.Popen(command, cwd=PROJECT, pass_fds=(lock.fileno(),))
            reference = identity(process.pid)
            update(stage, command={"stage":stage,"args":command,"process":reference})
            code = process.wait()
            state["command"]["exit_code"] = code
            write_json(state_path, state)
            if code:
                raise RuntimeError(f"{stage} exited with code {code}; later stages were not started")
            return code

        original = Path(plan["original_run"])
        status_path = original / "fold_0/status.json"
        try:
            if not state.get("original_scheduler_released"):
                update("waiting_for_fold0_twenty_epochs")
                print("Current fold0 continues unchanged. Waiting for twenty epochs and the planned scheduler handoff.", flush=True)
                while True:
                    status = json.loads(status_path.read_text())
                    alive = same_process(plan["original_scheduler"])
                    if not alive:
                        break
                    update("waiting_for_fold0_twenty_epochs", original_completed_epochs=status.get("completed_epochs",0))
                    time.sleep(15)
                status = json.loads(status_path.read_text())
                if status.get("training_complete") is not True or status.get("completed_epochs") != 20:
                    raise RuntimeError("The original fold0 ended before its declared twenty epochs")
                if same_process(plan["original_trainer"]):
                    raise RuntimeError("The old GPU training child is still alive; refusing overlapping GPU work")
                receipt_path = original / "evidence/planned_handoff_receipt.json"
                if not receipt_path.is_file():
                    raise RuntimeError("Original scheduler exited without the planned handoff receipt; inspect before continuing")
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("expected_exit_code") != 75
                        or receipt.get("blocked_fold") != 1
                        or receipt.get("marker_sha256") != plan["handoff_marker_sha256"]):
                    raise RuntimeError("Unexpected scheduler handoff receipt")
                update("original_scheduler_released", original_scheduler_released=True)

            chosen_output = Path(state.get("validation_output", plan["validation_output"]))
            comparison = chosen_output / "comparison_report.json"
            if not comparison.is_file():
                prior_command = state.get("command") or {}
                reattaching = (prior_command.get("stage") == "validating_epoch13_and_epoch20"
                               and prior_command.get("process") and same_process(prior_command["process"]))
                if chosen_output.exists() and not reattaching:
                    chosen_output = chosen_output.with_name(chosen_output.name + "_retry_" +
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                    comparison = chosen_output / "comparison_report.json"
                command = list(plan["validation_command"])
                command[command.index("--output-dir") + 1] = str(chosen_output)
                update("preparing_teacher_validation", validation_output=str(chosen_output))
                child_command("validating_epoch13_and_epoch20", command)
            if not comparison.is_file():
                raise RuntimeError("Validation ended without a comparison report")
            validate_comparison(json.loads(comparison.read_text()), plan)
            update("teacher_comparison_ready", comparison_report=str(comparison))

            remaining = Path(plan["remaining_run"])
            runner_status = remaining / "runner_status.json"
            ready = runner_status.is_file() and json.loads(runner_status.read_text()).get("all_requested_complete") is True
            if not ready:
                child_command("training_folds_1_to_4_twelve_epochs", plan["remaining_command"])
            if not runner_status.is_file() or json.loads(runner_status.read_text()).get("all_requested_complete") is not True:
                raise RuntimeError("The twelve-epoch scheduler did not complete every requested fold")
            read_teacher_collection(plan["teacher_collection"], require_checkpoints=True)
            refresh_manifest(root)
            update("completed", all_teachers_complete=True,
                   next_step="Run strict OOF extraction with --teacher-collection; student training was not started")
            print("Mixed teacher collection complete. OOF extraction and student training remain manual next steps.", flush=True)
            return 0
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    if args.status:
        path = Path(plan["collection_root"]) / "transition_status.json"
        print(path.read_text() if path.exists() else json.dumps({"phase":"prepared_not_started"}))
        return 0
    if not args.execute:
        print(json.dumps({"execute":False,"fixed_epochs":plan["fixed_epochs"],
                          "retained_epoch13_checkpoint":plan["retained_epoch13_checkpoint"],
                          "teacher_collection":plan["teacher_collection"],
                          "validation_command":plan["validation_command"],
                          "remaining_command":plan["remaining_command"]}, ensure_ascii=False, indent=2))
        return 0
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        return run_logged(argparse.Namespace(evidence_dir=Path(plan["collection_root"]) / "evidence",
            command=[sys.executable,"-u",str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    return execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
