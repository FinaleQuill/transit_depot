"""Execute one registered stage and retain its final EMA with compact evidence.

The host launcher uses only the standard library. Large full checkpoints remain
on local disk; final FP16 inference weights and source/log/lineage evidence are
retained in work. No teacher is trained and no test data is accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "docs/BOUNDED_STUDY_FINETUNE_20261101.json"
PLAN_SHA = "c73aef018b7841bf3d47cd0e88b79f79d4c32aef1555ec8cdadd119ece76ee93"
PACKING = ROOT / "docs/BOUNDED_STUDY_PACKING_20261101.json"
PACKING_SHA = "d6ff67eb13776b66e1737bd953f9929919d82039108151edc158625e81cb5471"
EVIDENCE = ROOT / "artifacts/kd_v1/bounded_finetune_20261101"
LOCAL = Path("/tmp/cloud_student_runs_20261030/bounded_finetune_20261101")
BASE_SOURCE = ROOT / "artifacts/kd_v1/student_upper_bound_20261030"
WRAPPER = ROOT / "run_student_environment.sh"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def stamp():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def registered(stage):
    if sha(PLAN) != PLAN_SHA:
        raise ValueError("Registered plan identity changed")
    if sha(PACKING) != PACKING_SHA:
        raise ValueError("Registered execution-packing addendum changed")
    plan = json.loads(PLAN.read_text())
    recipe = plan["training"][stage]
    source_label = "A" if stage == "X1" else "B"
    snapshot = BASE_SOURCE / "models" / source_label / "best_ema_fp32.pt"
    if sha(snapshot) != recipe["init_sha256"]:
        raise ValueError("Registered initialization EMA identity changed")
    return plan, recipe, snapshot


def run_directory(stage):
    return LOCAL / (stage + '_packed')


def command(stage):
    plan, recipe, snapshot = registered(stage)
    tiny = stage == "X1"
    provenance = (ROOT.parent / "runs/030_tiny768_kd_seed44/pretrained_provenance.json"
                  if tiny else BASE_SOURCE / "base_original_pretrained_provenance.json")
    pretrain = ROOT.parent / "deliveries/038_dual768_optimized_project_20261030/pretrain_model"
    pretrain /= ("timm__convnext_tiny.fb_in22k_ft_in1k_384_a5cbe10cfa2e" if tiny
                 else "timm__convnext_base.fb_in22k_ft_in1k_384_bc0cfafc7d77")
    arguments = {
        "fold-plan": ROOT / "data/cv5_overnight/folds.json",
        "soft-labels": ROOT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2",
        "output": run_directory(stage),
        "image-root": "/home/featurize/data/cxr_first",
        "pretrained-provenance-path": provenance,
        "pretrained-weights-path": pretrain / "model.safetensors",
        "init-student-ema": snapshot,
        "init-student-ema-sha256": recipe["init_sha256"],
        "backbone-name": recipe["backbone"],
        "epochs": recipe["epochs"], "seed": recipe["seed"],
        "backbone-lr": recipe["backbone_lr"], "head-lr": recipe["head_lr"],
        "alpha": recipe["alpha"], "drop-path": recipe["drop_path"],
        "warmup-epochs": plan["common"]["warmup_epochs"],
        "kd-warmup-epochs": plan["common"]["kd_warmup_epochs"],
        "min-lr-ratio": plan["common"]["min_lr_ratio"],
        "patience": 0,
        "pack-images": 8,
    }
    result = ["bash", str(WRAPPER), "-u", str(ROOT / "study_student.py")]
    for name, value in arguments.items():
        result.extend(["--" + name, str(value)])
    return result


def retain_small_files(stage):
    run, destination = run_directory(stage), EVIDENCE / "training" / (stage + '_packed')
    candidates = [run / name for name in (
        "config.json", "status.json", "history.jsonl", "checkpoint_audit.jsonl",
        "best_metrics.json", "pretrained_provenance.json", "student_warm_start.json")]
    if (run / "evidence").is_dir():
        candidates.extend(p for p in (run / "evidence").rglob("*")
                          if p.suffix in {".py", ".sh", ".json", ".jsonl", ".log", ".txt", ".md"})
    for source in candidates:
        if not source.is_file() or source.is_symlink():
            continue
        if source.stat().st_size > 20 * 1024**2:
            raise ValueError("Unexpectedly large training evidence file")
        target = destination / source.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)


def train(stage):
    plan, recipe, _ = registered(stage)
    args = command(stage)
    run = run_directory(stage)
    if run.exists():
        raise FileExistsError("Fresh registered runs never overwrite existing training state")
    deadline = datetime.fromisoformat(plan["budget"]["deadline_hong_kong"]).timestamp() - 3600
    cap_seconds = min((4 if stage == "X1" else 6) * 3600, deadline - time.time())
    if cap_seconds < 3600:
        raise RuntimeError("Insufficient budget for training and final validation")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    receipt = {"stage": stage, "registered_plan_sha256": PLAN_SHA,
        "started_hong_kong": stamp(), "command": [*args, "--execute"],
        "wall_seconds_cap": cap_seconds, "final_ema_epoch": recipe["epochs"],
        "test_used": False, "exact_resume": False, "state": "starting",
        "packing_addendum_sha256": PACKING_SHA}
    record = EVIDENCE / f"{stage}_packed.execution.json"
    if record.exists():
        raise FileExistsError(record)
    write(record, receipt)
    container = f"codex-bounded-study-{stage.lower()}-packed-20261101"
    environment = {**os.environ, "STUDENT_RUNTIME_MODE": "training",
                   "STUDENT_CONTAINER_NAME": container}
    receipt["container_name"] = container
    with (EVIDENCE / f"{stage}_packed.training.log").open("x") as log:
        process = subprocess.Popen([*args, "--execute"], cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(state="training", host_launcher_pid=os.getpid(), child_pid=process.pid)
        write(record, receipt)
        started, stop_sent = time.monotonic(), None
        while process.poll() is None:
            retain_small_files(stage)
            elapsed = time.monotonic() - started
            if elapsed >= cap_seconds and stop_sent is None:
                # The trainer handles TERM by finishing the current epoch and
                # saving state. A capped partial run is never called final E4.
                subprocess.run(["docker", "kill", "--signal=TERM", container],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                stop_sent = time.monotonic()
                receipt.update(state="budget_stop_requested", stopped_hong_kong=stamp())
                write(record, receipt)
            if stop_sent is not None and time.monotonic() - stop_sent > 600:
                subprocess.run(["docker", "kill", container],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            time.sleep(15)
        code = process.wait()
    retain_small_files(stage)
    status_path = run / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    complete = code == 0 and status.get("state") == "completed" and status.get("completed_epochs") == recipe["epochs"]
    receipt.update(state="training_complete" if complete else "incomplete_or_failed",
        exit_code=code, finished_hong_kong=stamp(), training_status=status,
        console_sha256=sha(EVIDENCE / f"{stage}_packed.training.log"))
    write(record, receipt)
    if not complete:
        raise RuntimeError("Stage did not complete its registered horizon; no final candidate exported")
    inference_env = {**os.environ, "STUDENT_RUNTIME_MODE": "inference"}
    subprocess.run(["bash", str(WRAPPER), str(Path(__file__).resolve()),
                    "--stage", stage, "--action", "export", "--execute"],
                   cwd=ROOT, env=inference_env, check=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def export_final(stage):
    import torch
    from student_experiment_evaluation import export_experimental_checkpoint
    plan, recipe, _ = registered(stage)
    source = run_directory(stage) / "last.pt"
    before = sha(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    warm = checkpoint.get("student_warm_start", {})
    if (checkpoint.get("epoch") != recipe["epochs"]
            or checkpoint.get("progress", {}).get("completed_epochs") != recipe["epochs"]
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("model_config", {}).get("backbone_name") != recipe["backbone"]
            or checkpoint.get("training_config", {}).get("seed") != recipe["seed"]
            or checkpoint.get("training_config", {}).get("pack_images") != 8
            or warm.get("source_snapshot_sha256") != recipe["init_sha256"]):
        raise ValueError("Final checkpoint differs from its registered horizon/EMA/initialization")
    summary = {"registered_plan_sha256": PLAN_SHA, "packing_addendum_sha256": PACKING_SHA, "stage": stage,
        "source_checkpoint_sha256": before, "source_epoch": checkpoint["epoch"],
        "checkpoint_selection": "fixed_final_epoch_EMA", "student_warm_start": warm,
        "training_config": checkpoint["training_config"], "progress": checkpoint["progress"],
        "validation_diagnostic": checkpoint["validation"], "resume_state_persisted_in_work": False,
        "full_local_checkpoint": str(source), "test_used": False}
    del checkpoint
    if sha(source) != before:
        raise RuntimeError("Final checkpoint changed during verification")
    output = EVIDENCE / "models" / stage
    needed = (60 if stage == "X1" else 178) * 1024**2 + 20 * 1024**2
    if shutil.disk_usage(EVIDENCE).free < needed:
        raise OSError("Not enough work space to retain final FP16 model and evidence reserve")
    config = export_experimental_checkpoint(source, output)
    if config.get("student_warm_start") != warm:
        raise ValueError("Export lost the EMA initialization ancestry")
    summary.update(weights_sha256=config["weights_sha256"], model_dir=str(output),
        export_config_sha256=sha(output / "config.json"), persisted_hong_kong=stamp())
    write(EVIDENCE / f"{stage}.final_ema.json", summary)
    print(json.dumps({"stage": stage, "persisted_final_model": str(output),
                      "weights_sha256": config["weights_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("X1", "X2"), required=True)
    parser.add_argument("--action", choices=("train", "export"), default="train")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"dry_run": True, "command": command(args.stage),
            "action": args.action, "plan_sha256": PLAN_SHA}, indent=2))
    elif args.action == "train":
        train(args.stage)
    else:
        export_final(args.stage)


if __name__ == "__main__":
    main()
