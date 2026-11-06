"""五折合规教师调度。默认仅展示命令，只有 --execute 才启动训练。

每折固定训练轮数，自动从原目录 last.pt 续训；不会启动软标签、学生或推理。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT.parent / "runs" / "021_convnext_v1_base768_teacher_cv5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict OOF V1 Base 768 teachers; dry-run by default")
    parser.add_argument("--execute", action="store_true", help="Explicitly start teacher training")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fold-plan", type=Path, default=PROJECT / "data/cv5_overnight/folds.json")
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-epochs-this-call", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.batch_size <= 0 or args.effective_batch_size <= 0
            or args.effective_batch_size % args.batch_size):
        parser.error("batch-size must be positive and divide effective-batch-size")
    if args.epochs < 2 or args.max_epochs_this_call < 0 or not 0 <= args.num_workers <= 12:
        parser.error("epochs >= 2, max-epochs-this-call >= 0 and 0 <= num-workers <= 12 are required")
    if args.prefetch_factor not in (1, 2) or not 0 <= args.seed < 2**32:
        parser.error("prefetch-factor must be 1/2 and seed must be in [0, 2**32)")
    if len(set(args.folds)) != len(args.folds) or any(i not in range(5) for i in args.folds):
        parser.error("folds must be unique values from 0 through 4")
    args.run_dir = args.run_dir.resolve()
    args.fold_plan = args.fold_plan.resolve()
    args.image_root = args.image_root.resolve()
    args.python = args.python.absolute()
    if (args.run_dir == args.image_root or args.run_dir.is_relative_to(args.image_root)
            or args.image_root.is_relative_to(args.run_dir)):
        parser.error("run-dir must not overlap the read-only image root")
    return args


def training_command(args: argparse.Namespace, fold: int) -> list[str]:
    output = args.run_dir / f"fold_{fold}"
    command = [str(args.python), "-u", str(PROJECT / "train_teacher.py"), "--execute",
        "--fold-plan", str(args.fold_plan), "--fold", str(fold), "--output", str(output),
        "--image-root", str(args.image_root), "--batch-size", str(args.batch_size),
        "--accumulation-steps", str(args.effective_batch_size // args.batch_size),
        "--epochs", str(args.epochs), "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor), "--seed", str(args.seed),
        "--amp-dtype", args.amp_dtype, "--stop-file", str(args.run_dir / "STOP"),
        "--max-epochs-this-call", str(args.max_epochs_this_call)]
    if (output / "last.pt").exists():
        command += ["--resume", str(output / "last.pt")]
    elif (output / "config.json").exists():
        command.append("--restart-empty")
    return command


def frozen_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {"artifact_type": "kd_teacher_run", "schema_version": 6,
        "backbone_name": "convnext_base.fb_in22k_ft_in1k_384", "image_size": 768,
        "selection_protocol": "fixed_epoch_final_ema", "epochs": args.epochs,
        "seed": args.seed, "effective_batch_size": args.effective_batch_size,
        "fold_plan": str(args.fold_plan), "fold_plan_sha256": sha256(args.fold_plan),
        "image_root": str(args.image_root), "official_valid_used_for_training": False,
        "outer_fold_used_for_selection": False}


def run(args: argparse.Namespace) -> int:
    if not args.fold_plan.is_file():
        raise FileNotFoundError(f"Fold plan does not exist: {args.fold_plan}; create/validate it with cv_splits.py first")
    frozen = frozen_configuration(args)
    print(json.dumps({"execute": args.execute, "run_dir": str(args.run_dir), **frozen}, indent=2), flush=True)
    if not args.execute:
        for fold in args.folds:
            print(shlex.join(training_command(args, fold)), flush=True)
        print("Dry-run only: no model, image loading, weight download or training was started.", flush=True)
        return 0
    if not args.python.is_file() or not args.image_root.is_dir():
        raise FileNotFoundError("Training interpreter and extracted image-root must exist")
    metadata = args.run_dir / "teacher_run.json"
    # The command recorder creates only evidence/ before the first real run.
    unexpected = args.run_dir.exists() and any(
        p.name != "evidence" or not p.is_dir() or p.is_symlink() for p in args.run_dir.iterdir())
    if unexpected and not metadata.exists():
        raise FileExistsError("Nonempty run-dir has no V1 teacher protocol marker; choose a new directory")
    if metadata.exists() and json.loads(metadata.read_text()) != frozen:
        raise ValueError("Teacher run configuration changed; do not mix legacy V2 or another training horizon")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = args.run_dir / ".teacher_runner.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            if pid <= 0:
                raise ValueError("Invalid lock PID")
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()  # Only an absent process permits automatic stale-lock recovery.
        except (ValueError, PermissionError):
            raise RuntimeError("Teacher runner lock cannot be verified; inspect it before restarting")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    previous_handlers = {}
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        write_json(metadata, frozen)
        def request_stop(_signum, _frame):
            # Parent and child may share a terminal process group. Keep waiting for
            # the child so its epoch-boundary checkpoint and the runner lock agree.
            (args.run_dir / "STOP").touch(exist_ok=True)
            print("Teacher runner stop requested; waiting for the active epoch checkpoint.", flush=True)
        previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        statuses = {}
        for fold in args.folds:
            if (args.run_dir / "STOP").exists():
                print("STOP exists; no next fold will be started.", flush=True)
                break
            command = training_command(args, fold)
            print(shlex.join(command), flush=True)
            completed = subprocess.run(command, cwd=PROJECT, check=False)
            status_path = args.run_dir / f"fold_{fold}" / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "missing"}
            statuses[str(fold)] = status
            write_json(args.run_dir / "runner_status.json", {"schema_version": 6,
                "folds": statuses, "updated_utc": datetime.now(timezone.utc).isoformat(),
                "all_requested_complete": all(statuses.get(str(i), {}).get("training_complete") is True for i in args.folds)})
            if completed.returncode:
                return completed.returncode
            if status.get("training_complete") is not True:
                print("Teacher paused before its fixed horizon; resume this command to continue.", flush=True)
                break
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
