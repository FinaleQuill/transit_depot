#!/usr/bin/env python3
"""Pause the exploration controller after the resolution-only report is durable."""
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE = Path(__file__).resolve().parents[1]
STATUS = BASE / "STATUS.json"
HOLD = BASE / "queue" / "resolution_analysis_hold.json"
CONTROLLER_PID = int(sys.argv[1])

while True:
    try:
        status = json.loads(STATUS.read_text())
    except (OSError, json.JSONDecodeError):
        time.sleep(1)
        continue

    completed = set(status.get("completed_jobs", []))
    if ("report_resolution" in completed
            and status.get("current_job") == "roi"
            and status.get("phase") == "checking_preregistration"):
        try:
            cmdline = Path(f"/proc/{CONTROLLER_PID}/cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            break
        if b"run_exploration_queue.py" not in cmdline:
            break
        HOLD.write_text(json.dumps({
            "state": "PAUSING_BEFORE_ROI_TRAINING",
            "paused_controller_pid": CONTROLLER_PID,
            "paused_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_jobs": status.get("completed_jobs", []),
            "plan_sha256": status.get("plan_sha256"),
            "next_training_not_started": "roi",
            "test_accessed": False
        }, indent=2) + "\n")
        os.kill(CONTROLLER_PID, signal.SIGSTOP)
        break

    if status.get("state") in {"FAILED", "COMPLETED"}:
        break
    time.sleep(0.5)
