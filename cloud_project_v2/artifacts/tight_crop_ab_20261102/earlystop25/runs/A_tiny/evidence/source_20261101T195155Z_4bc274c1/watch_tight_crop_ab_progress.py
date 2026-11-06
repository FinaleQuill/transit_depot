"""Read mirrored A/B epoch metrics and publish a small live AUC ledger.

This host-stdlib watcher never opens images, checkpoints, or model weights.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/tight_crop_ab_20261102/earlystop25"
JOBS = ("A_tiny", "A_base", "B_tiny", "B_base")
FIELDS = ("job", "epoch", "all_study_macro_auc", "ap_only_macro_auc", "is_best",
          *(f"all_class_{class_id}_auc" for class_id in range(1, 11)),
          *(f"ap_class_{class_id}_auc" for class_id in range(1, 11)))


def now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path):
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # A writer may still be appending this final line.
            raise
    return rows


def value_text(value):
    return "" if value is None else f"{float(value):.9f}"


def auc_vector(validation, key):
    values = validation.get(key)
    if values is None:
        return [""] * 10
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError(f"Expected ten AUC values in {key}")
    return [value_text(value) for value in values]


def collect():
    rows = []
    summary = {}
    for job in JOBS:
        folder = OUTPUT / "runs" / job
        config = read_json(folder / "config.json", {})
        status = read_json(folder / "status.json", {})
        history = read_history(folder / "history.jsonl")
        if history and (config.get("arm"), config.get("member")) != tuple(job.split("_", 1)):
            raise ValueError(f"Mirrored run config does not match {job}")
        if history and [item.get("epoch") for item in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Mirrored epoch history has a gap for {job}")
        for item in history:
            validation = item.get("validation", {})
            if not isinstance(validation, dict):
                raise ValueError(f"Missing validation metrics for {job} E{item.get('epoch')}")
            row = {"job": job, "epoch": item["epoch"],
                   "all_study_macro_auc": value_text(validation.get("all_study_macro_auc")),
                   "ap_only_macro_auc": value_text(validation.get("ap_only_macro_auc")),
                   "is_best": str(bool(item.get("is_best"))).lower()}
            for index, auc in enumerate(auc_vector(validation, "all_study_per_class_auc"), 1):
                row[f"all_class_{index}_auc"] = auc
            for index, auc in enumerate(auc_vector(validation, "ap_only_per_class_auc"), 1):
                row[f"ap_class_{index}_auc"] = auc
            rows.append(row)
        latest = rows[-1] if history else None
        best = max((item for item in rows if item["job"] == job and item["ap_only_macro_auc"]),
                   key=lambda item: float(item["ap_only_macro_auc"]), default=None)
        summary[job] = {"latest": latest, "best": best,
                        "state": status.get("state", "pending"),
                        "best_epoch_status": status.get("best_epoch")}
    return rows, summary


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def publish():
    rows, summary = collect()
    controller = read_json(OUTPUT / "status.json", {})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUTPUT / "auc_progress.csv", buffer.getvalue())
    lines = ["# Tight-crop A/B AUC progress", "",
             f"Updated Hong Kong: {now()}",
             f"Controller: {controller.get('state', 'pending')} / {controller.get('stage', 'pending')}",
             "", "The overall column is all-Study valid Macro-AUC. AP-only controls native early stopping.",
             "", "| Job | Run state | Latest epoch | Overall AUC | AP-only AUC | Best AP epoch | Best AP AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for job in JOBS:
        item = summary[job]
        latest, best = item["latest"], item["best"]
        lines.append(f"| {job} | {item['state']} | {latest['epoch'] if latest else '-'} | "
                     f"{latest['all_study_macro_auc'] if latest else '-'} | "
                     f"{latest['ap_only_macro_auc'] if latest else '-'} | "
                     f"{best['epoch'] if best else '-'} | "
                     f"{best['ap_only_macro_auc'] if best else '-'} |")
    monitor = read_json(OUTPUT / "migration/A_tiny_E1_monitor/monitor.json")
    if monitor and not summary["A_tiny"]["latest"]:
        lines.extend(("", "Original two-epoch A_tiny E1 migration monitor (separate from new run history): "
                      f"overall {value_text(monitor.get('all_study_macro_auc'))}, "
                      f"AP-only {value_text(monitor.get('ap_only_macro_auc'))}."))
    elif not summary["A_tiny"]["latest"]:
        original_status = read_json(OUTPUT.parent / "runs/A_tiny/status.json", {})
        if original_status.get("completed_epochs") == 1:
            lines.extend(("", "Original two-epoch A_tiny E1 checkpoint exists; "
                          "the separate native AP monitor is pending."))
    lines.extend(("", "Per-class valid AUC for classes 1-10 is in `auc_progress.csv`. "
                  "No test data or checkpoint weights are read by this watcher.", ""))
    atomic_text(OUTPUT / "AUC_PROGRESS.md", "\n".join(lines))
    return controller.get("state", "pending"), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".auc_progress.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state, count = publish()
            print(f"{now()} {state}: {count} committed epoch rows", flush=True)
            if args.once or state in ("completed", "failed"):
                return 0
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
