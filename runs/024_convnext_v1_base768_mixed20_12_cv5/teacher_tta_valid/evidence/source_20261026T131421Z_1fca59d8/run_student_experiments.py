"""Preview staged full-train student experiments; --execute explicitly trains.

compare trains only seed43 at512/768. Additional seeds require a separately
selected stage. Completed teachers and a valid strict-OOF cache are required.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
STAGES = {"compare": [(512, 43), (768, 43)], "extra512": [(512, 47)],
          "extra768": [(768, 47)], "four512": [(512, s) for s in (43, 47, 53, 59)]}


def experiment_commands(*, stage, output_root, soft_labels, fold_plan, image_root,
                        python=sys.executable, epochs=25, num_workers=8):
    if stage not in STAGES or type(epochs) is not int or epochs < 1:
        raise ValueError("Unknown stage or invalid epoch budget")
    root = Path(output_root).resolve()
    jobs = []
    for size, seed in STAGES[stage]:
        output = root / f"tiny{size}_seed{seed}"
        batch = 8 if size == 512 else 4
        command = [str(python), "-u", str(PROJECT / "train_distill.py"),
                   "--fold-plan", str(Path(fold_plan).resolve()), "--soft-labels", str(Path(soft_labels).resolve()),
                   "--image-root", str(Path(image_root).resolve()), "--output", str(output),
                   "--image-size", str(size), "--validation-tta", "hflip2", "--seed", str(seed),
                   "--batch-size", str(batch), "--accumulation-steps", str(32 // batch),
                   "--epochs", str(epochs), "--num-workers", str(num_workers), "--execute"]
        if (output / "last.pt").is_file():
            command += ["--resume", str(output / "last.pt")]
        elif (output / "best.pt").is_file():
            # Existing student crash reconciliation also handles first best-before-last.
            command += ["--resume", str(output / "last.pt")]
        jobs.append({"image_size": size, "seed": seed, "output": str(output), "command": command})
    return jobs


def run_experiments(args):
    jobs = experiment_commands(stage=args.stage, output_root=args.output_root, soft_labels=args.soft_labels,
        fold_plan=args.fold_plan, image_root=args.image_root, python=args.python,
        epochs=args.epochs, num_workers=args.num_workers)
    if not args.execute:
        print(json.dumps({"execute": False, "stage": args.stage, "jobs": jobs,
                          "note": "Full-train seed models; these are not student CV folds."}, indent=2))
        for job in jobs:
            print(shlex.join(job["command"]))
        return 0
    from teacher_collection import read_teacher_collection
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from distill_tta_data import load_soft_labels
    from distill_data import sha256_file
    collection = read_teacher_collection(args.teacher_collection, fold_plan_path=args.fold_plan,
                                         require_checkpoints=True)
    plan = read_fold_plan(args.fold_plan)
    records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    cache = load_soft_labels(args.soft_labels, args.fold_plan, records)
    if cache.metadata.get("tta") != "identity_hflip":
        raise ValueError("This experiment plan requires the approved dual-view OOF cache")
    if [t["checkpoint_sha256"] for t in cache.metadata["teachers"]] != [
            sha256_file(t["checkpoint"]) for t in collection["teachers"]]:
        raise ValueError("OOF cache does not correspond to the frozen teacher collection")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("A GPU task is already running; refuse to overlap student experiments")
    root = args.output_root.resolve()
    image_root = args.image_root.resolve(strict=True)
    if root == image_root or root.is_relative_to(image_root) or image_root.is_relative_to(root):
        raise ValueError("Experiment output cannot overlap the original image directory")
    root.mkdir(parents=True, exist_ok=True)
    stopped = False
    current = None
    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        (root / "STOP").touch()
        if current is not None:
            current.mkdir(parents=True, exist_ok=True)
            (current / "STOP").touch()
        print("Stop requested: finish the active student's epoch; do not start another student.", flush=True)
    previous = {}
    with (root / ".experiments.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for job in jobs:
                if stopped or (root / "STOP").exists():
                    return 2
                current = Path(job["output"])
                print(shlex.join(job["command"]), flush=True)
                student_env = os.environ.copy()
                student_env.pop("KD_TRAINING_LOG_ACTIVE", None)
                process = subprocess.Popen(job["command"], cwd=PROJECT, env=student_env,
                                           pass_fds=(lock.fileno(),))
                code = process.wait()
                if code:
                    return code
                status = json.loads((current / "status.json").read_text())
                if status.get("state") not in {"completed", "early_stopped", "already_terminal"}:
                    print("Student paused. Later students were not started.", flush=True)
                    return 2
            return 0
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="compare")
    parser.add_argument("--output-root", type=Path, default=PROJECT.parent / "runs/025_tiny_multires_kd")
    parser.add_argument("--soft-labels", type=Path, default=PROJECT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2")
    parser.add_argument("--teacher-collection", type=Path, default=PROJECT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json")
    parser.add_argument("--fold-plan", type=Path, default=PROJECT / "data/cv5_overnight/folds.json")
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return run_experiments(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
