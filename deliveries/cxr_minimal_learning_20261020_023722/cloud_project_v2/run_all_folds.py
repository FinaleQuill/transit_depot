"""Sequential, resumable round-robin CV. Only main() can launch child jobs.

--dry-run validates metadata and prints commands; it never constructs a model,
downloads weights, starts a training/inference process or writes run artifacts.
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
import re
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent
WORK = PROJECT.parent
RUN_NAME = "020_convnextv2_base768_posbce_cv5_4090"
# 源码指纹只覆盖本学习包实际存在的九个模块，删除旧版本的文件清单依赖。
SOURCE_FILES = ("run_all_folds.py", "train_overnight.py", "model_overnight.py",
                "dataset.py", "dataset_overnight.py", "cv_splits.py",
                "inference_overnight.py", "metrics_overnight.py", "runtime_compat.py")
TERMINAL = {"completed", "early_stopped"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hardware() -> dict:
    # A read-only query; honor the first numeric CUDA_VISIBLE_DEVICES entry.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    query = subprocess.run(["nvidia-smi", "-i", visible, "--query-gpu=name,memory.total",
                            "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    name, memory = query.stdout.strip().rsplit(",", 1)
    return {"name": name.strip(), "memory_mib": int(memory.strip())}


# microbatch 决定每次显存里处理的图片数，effective 决定每次更新累计多少图片。
# 32/2=16 次；OOM 后降到 1，累积改成 32，继续保持有效批次。
def choose_batch(memory_mib: int, requested: int, effective: int) -> tuple[int, int]:
    batch = requested or (2 if memory_mib >= 14000 else 1)
    if batch < 1 or effective < 1 or effective % batch:
        raise ValueError("Microbatch must be a positive divisor of effective batch size")
    return batch, effective // batch


def smaller_batch(batch: int, effective: int) -> int:
    """Preserve the effective batch even for non-power-of-two custom settings."""
    return next(value for value in range(max(1, batch // 2), 0, -1) if effective % value == 0)


def next_fold(statuses: dict[int, dict], excluded: set[int]) -> int | None:
    # 已完成/早停的折不再训练；发生非可恢复错误的折排除，仍保留已有产物。
    available = [fold for fold in range(5) if fold not in excluded and statuses[fold].get("state") not in TERMINAL]
    if not available:
        return None
    # 选择完成 epoch 最少的折，平手按折号：形成每折一轮、再下一圈的轮转。
    return min(available, key=lambda fold: (statuses[fold].get("completed_epochs", 0), fold))


def training_command(args: argparse.Namespace, fold: int, batch: int, valid_batch: int) -> list[str]:
    output = args.run_dir / f"fold_{fold}"
    # 每次子进程只完成一个 epoch，但完整学习率计划仍按总 epochs 计算。
    # accumulation 根据当前实际微批动态相除，不写死为 16。
    command = [str(args.python), "-u", str(PROJECT / "train_overnight.py"),
        "--fold-plan", str(args.fold_plan), "--fold", str(fold), "--output", str(output),
        "--image-root", str(args.image_root), "--batch-size", str(batch),
        "--accumulation-steps", str(args.effective_batch_size // batch),
        "--valid-batch-size", str(valid_batch), "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(getattr(args, "prefetch_factor", 2)), "--epochs", str(args.epochs), "--max-epochs-this-call", "1",
        "--patience", str(args.patience), "--seed", str(args.seed), "--amp-dtype", "auto",
        "--stop-file", str(args.run_dir / "STOP")]
    # 优先恢复 last；若中断在 best 写完而 last 未写时，完整 best 也能接续。
    # 训练器还会协调 best/last 的时间与指标，保护已完成进度和历史最佳。
    if (output / "last.pt").is_file():
        command += ["--resume", str(output / "last.pt")]
    elif (output / "best.pt").is_file():
        # An interruption after best was committed but before the first last.pt.
        # Schema 5 best also contains complete optimizer/scaler/RNG state.
        command += ["--resume", str(output / "best.pt")]
    elif output.exists() and any(output.iterdir()):
        # The trainer refuses this if even one checkpoint/history entry exists.
        command += ["--restart-empty"]
    return command


def run_child(command: list[str], log_path: Path) -> tuple[int, str]:
    print("\n=> " + shlex.join(command), flush=True)
    tail = ""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
        log.flush()
        with subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail = (tail + line)[-16000:]
            code = process.wait()
        log.write(f"[{utc_now()}] exit_code={code}\n")
    return code, tail


def is_oom(status: dict, tail: str) -> bool:
    text = (str(status.get("error", "")) + "\n" + tail).lower()
    return bool(status.get("oom")) or "cuda out of memory" in text or "cuda error: out of memory" in text


# 冻结数据/源码哈希、有效批次和 TTA 等策略，同名运行不能悄悄换协议。
# 教学副本添加了注释，文件字节哈希会变化；不要用它覆盖运行中的原代码。
def frozen_config(args: argparse.Namespace, plan: dict) -> dict:
    return {"schema_version": 1, "model": "convnextv2_base.fcmae_ft_in1k", "run_name": args.run_name,
        "fold_plan": str(args.fold_plan), "fold_plan_sha256": sha256(args.fold_plan),
        "source_train_manifest_sha256": plan["source_manifest_sha256"],
        "official_valid_manifest": str(args.valid_manifest),
        "official_valid_manifest_sha256": sha256(args.valid_manifest),
        "test_manifest": str(args.test_manifest), "test_manifest_sha256": sha256(args.test_manifest),
        "sample_submission": str(args.sample_submission), "sample_submission_sha256": sha256(args.sample_submission),
        "image_root": str(args.image_root), "epochs": args.epochs, "patience": args.patience,
        "seed": args.seed, "effective_batch_size": args.effective_batch_size,
        "input_size": 768, "loss": "study_equal_BCEWithLogitsLoss_fold_train_neg_over_pos",
        "num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor,
        "round_robin_epochs_per_visit": 1, "tta": {"scales": [768], "hflip": True},
        "ensemble_weighting": "uniform_probability_mean", "holdout_used_for_fold_selection": False,
        "code_sha256": {name: sha256(PROJECT / name) for name in SOURCE_FILES}}


def build_ensemble_plan(args: argparse.Namespace, config: dict) -> dict:
    # 各折只选内部验证 AUC 最佳的 EMA best；不根据官方 valid 再选模型或拟合折权重。
    checkpoints = []
    for fold in range(5):
        path = args.run_dir / f"fold_{fold}" / "best.pt"
        if path.is_file():
            checkpoints.append({"fold": fold, "path": str(path), "sha256": sha256(path), "weight": 1.0})
    if not checkpoints:
        raise RuntimeError("No validated best.pt exists; inspect runner_status.json and logs")
    # 不足五个可用 best 时明确标记部分集成，不能把计划五折误称为五折已经完成。
    return {"schema_version": 1, "fold_plan": str(args.fold_plan),
        "fold_plan_sha256": config["fold_plan_sha256"], "checkpoints": checkpoints,
        "tta": config["tta"], "partial_cv_ensemble": len(checkpoints) != 5,
        **{key: config[key] for key in ("official_valid_manifest_sha256", "test_manifest_sha256", "sample_submission_sha256")}}


def inference_command(args: argparse.Namespace, plan_path: Path, destination: Path, batch: int) -> list[str]:
    return [str(args.python), "-u", str(PROJECT / "inference_overnight.py"), "--ensemble-plan", str(plan_path),
        "--output", str(destination), "--image-root", str(args.image_root), "--work-root", str(WORK),
        "--valid-manifest", str(args.valid_manifest), "--test-manifest", str(args.test_manifest),
        "--sample-submission", str(args.sample_submission), "--batch-size", str(batch),
        "--num-workers", str(args.num_workers), "--prefetch-factor", str(getattr(args, "prefetch_factor", 2)), "--device", "cuda"]


def orchestrate(args: argparse.Namespace, config: dict, gpu: dict, batch: int, valid_batch: int) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Keep this descriptor open for the entire orchestration; children cannot inherit it.
    # 非阻塞文件锁避免两个调度器同时写一份运行目录；仅在训练入口实际执行时使用。
    with (args.run_dir / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner is already using this run directory") from exc
        config_path = args.run_dir / "run_config.json"
        # 旧运行绑定的代码/数据/关键设置不一致时拒绝混用，保留已有权重。
        if config_path.exists():
            if read_json(config_path) != config:
                raise ValueError("Existing run binds different code/data/settings. Use a new --run-name; existing weights are preserved.")
        elif any(args.run_dir.glob("fold_*")):
            raise ValueError("Refusing a populated run without its bound run_config.json")
        else:
            write_json(config_path, config)
        log_dir = args.run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        started = time.monotonic()
        deadline = started + args.hours * 3600 if args.hours else math.inf
        statuses = {fold: read_json(args.run_dir / f"fold_{fold}" / "status.json") for fold in range(5)}
        previous = read_json(args.run_dir / "runner_status.json")
        # Retain successful OOM adaptations between invocations.
        # 记住各折的 OOM 降档设置，下一次调用不自动升回曾经失败的微批。
        batches = {fold: min(batch, int(previous.get("microbatches", {}).get(str(fold), batch))) for fold in range(5)}
        valid_batches = {fold: min(valid_batch, int(previous.get("valid_microbatches", {}).get(str(fold), valid_batch))) for fold in range(5)}
        excluded: set[int] = set()
        failures: dict[str, str] = {}
        stop_reason = "all_folds_terminal"

        def status(state: str, **extra) -> None:
            write_json(args.run_dir / "runner_status.json", {"state": state, "updated_utc": utc_now(),
                "gpu": gpu, "folds": {str(k): v for k, v in statuses.items()}, "failures": failures,
                "microbatches": {str(k): v for k, v in batches.items()},
                "valid_microbatches": {str(k): v for k, v in valid_batches.items()},
                "elapsed_seconds": time.monotonic() - started, "hours_limit": args.hours, **extra})

        status("training")
        # 同一时刻只有一个折的训练子进程，五折不会同时争占显存。
        while (fold := next_fold(statuses, excluded)) is not None:
            # 在安排下一轮前检查停止/时间边界；子任务负责完整 epoch 的验证和保存。
            if (args.run_dir / "STOP").exists() or time.monotonic() >= deadline:
                stop_reason = "STOP_file" if (args.run_dir / "STOP").exists() else "time_budget_at_epoch_boundary"
                break
            # Two full checkpoint files and an atomic replacement need headroom.
            if shutil.disk_usage(args.run_dir).free < 3 * 2**30:
                failures["disk"] = "Less than 3 GiB free; retained checkpoints and stopped before next epoch"
                stop_reason = "low_disk_space"
                break
            before = statuses[fold].get("completed_epochs", 0) if statuses[fold].get("state") != "failed" else -1
            command = training_command(args, fold, batches[fold], valid_batches[fold])
            code, tail = run_child(command, log_dir / f"fold_{fold}.log")
            statuses[fold] = read_json(args.run_dir / f"fold_{fold}" / "status.json")
            current = statuses[fold]
            if code:
                # OOM 后减小微批，下一次命令自动增大累积来保持 effective batch。
                # 从完整 epoch 恢复，不把失败轮尚未落盘的半截梯度带入新轮。
                if is_oom(current, tail) and (batches[fold] > 1 or valid_batches[fold] > 1):
                    batches[fold] = smaller_batch(batches[fold], args.effective_batch_size)
                    valid_batches[fold] = max(1, valid_batches[fold] // 2)
                    print(f"=> OOM fold={fold}: retry microbatch={batches[fold]}, effective batch={args.effective_batch_size}", flush=True)
                    status("training", last_event=f"fold_{fold}_OOM_retry")
                    continue
                excluded.add(fold)
                failures[str(fold)] = f"exit={code}; {current.get('error', tail[-1500:])}"
                print(f"=> Fold {fold} failed; retaining its checkpoints and proceeding with other folds", flush=True)
            elif (current.get("state") not in TERMINAL and current.get("completed_epochs", 0) <= before
                  and not (args.run_dir / "STOP").exists()):
                excluded.add(fold)
                failures[str(fold)] = "Child exited without completing another epoch"
            status("training")
        status("preparing_ensemble", stop_reason=stop_reason)
        try:
            plan = build_ensemble_plan(args, config)
        except RuntimeError as exc:
            status("failed", stop_reason=stop_reason, error=str(exc))
            return 1
        plan_path = args.run_dir / "ensemble_plan.json"
        write_json(plan_path, plan)
        # --skip-inference 留下权重与冻结计划；最终官方 valid 阈值搜索不会在此发生。
        if args.skip_inference:
            status("checkpoints_ready", stop_reason=stop_reason, ensemble_plan=str(plan_path))
            return 0
        # A failed or previous delivery is retained; inference requires a fresh directory.
        destination = args.delivery_dir
        if destination.exists():
            old_status = read_json(destination / "status.json")
            if old_status.get("status") == "complete" and old_status.get("ensemble_plan_sha256") == sha256(plan_path):
                submission = destination / "submission_final.csv"
                provenance = read_json(destination / "provenance.json")
                if submission.is_file() and sha256(submission) == provenance.get("submission_validation", {}).get("sha256"):
                    status("complete", stop_reason=stop_reason, submission=str(submission), fold_count=len(plan["checkpoints"]))
                    print(f"=> Existing verified submission: {submission}", flush=True)
                    return 0
            destination = args.delivery_dir.with_name(args.delivery_dir.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
        infer_batch = valid_batch
        while True:
            status("inference", stop_reason=stop_reason, delivery=str(destination))
            # 推理先用固定集成评价官方 valid 并拟合 F1 阈值，再用相同协议预测 test。
            # 阈值只影响二值表；提交始终保留各 Study 完整十类的连续概率。
            code, tail = run_child(inference_command(args, plan_path, destination, infer_batch), log_dir / "inference.log")
            if not code:
                submission = destination / "submission_final.csv"
                if not submission.is_file() or read_json(destination / "status.json").get("status") != "complete":
                    status("inference_failed", error="Inference exited without complete submission", stop_reason=stop_reason)
                    return 1
                write_json(args.run_dir / "latest_submission.json", {"path": str(submission), "sha256": sha256(submission),
                    "fold_count": len(plan["checkpoints"]), "ensemble_plan_sha256": sha256(plan_path)})
                status("complete", stop_reason=stop_reason, submission=str(submission), fold_count=len(plan["checkpoints"]))
                print(f"=> FINAL SUBMISSION: {submission}", flush=True)
                return 0
            if is_oom({}, tail) and infer_batch > 1:
                infer_batch = max(1, infer_batch // 2)
                destination = destination.with_name(destination.name + f"_retry_b{infer_batch}")
                continue
            status("inference_failed", stop_reason=stop_reason, error=tail[-3000:], ensemble_plan=str(plan_path))
            return 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--fold-plan", type=Path, default=PROJECT / "data/cv5_overnight/folds.json")
    parser.add_argument("--valid-manifest", type=Path, default=PROJECT / "data/clean_v1_reviewed/valid_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=PROJECT / "data/clean_v1_reviewed/test_manifest.csv")
    parser.add_argument("--sample-submission", type=Path, default=WORK / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=0, help="0: measured default 2 on 16/24GB; OOM fallback enabled")
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hours", type=float, default=0,
                        help="0: unlimited. Positive: stop scheduling training at next epoch boundary, then infer (extra time).")
    parser.add_argument("--skip-inference", action="store_true", help="Leave best checkpoints and frozen ensemble plan only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name must be a simple directory name")
    if (args.epochs < 2 or args.patience < 0 or not 0 <= args.num_workers <= 12 or args.batch_size < 0
            or args.effective_batch_size < 1 or not math.isfinite(args.hours) or args.hours < 0):
        parser.error("Invalid epochs/patience/workers/batch/hours")
    for key in ("python", "image_root", "fold_plan", "valid_manifest", "test_manifest", "sample_submission"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.run_dir = WORK / "runs" / args.run_name
    args.delivery_dir = WORK / "deliveries" / args.run_name
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    from cv_splits import read_fold_plan, load_fold_records
    from dataset import read_manifest
    from inference_overnight import read_submission_template

    plan = read_fold_plan(args.fold_plan)
    if plan["n_splits"] != 5 or plan["seed"] != args.seed:
        raise ValueError("Fold plan split count/seed differs from runner configuration")
    if sha256(args.valid_manifest) != plan["official_holdout"]["sha256"]:
        raise ValueError("Official holdout is not the one audited by the fold plan")
    for fold in range(5):
        load_fold_records(args.fold_plan, fold, "valid")
    read_submission_template(args.sample_submission, read_manifest(args.test_manifest, split="test"))
    gpu = hardware()
    batch, _ = choose_batch(gpu["memory_mib"], args.batch_size, args.effective_batch_size)
    valid_batch = max(1, min(batch * 2, 8))
    config = frozen_config(args, plan)
    print(json.dumps({"hardware": gpu, "microbatch": batch, "accumulation_steps": args.effective_batch_size // batch,
        "workers": args.num_workers, "fold_plan": str(args.fold_plan), "run_dir": str(args.run_dir),
        "hours_limit": args.hours, "strategy": "one epoch per fold per round; complete resume; independent early stopping"}, indent=2), flush=True)
    if args.dry_run:
        for fold in range(5):
            print(shlex.join(training_command(args, fold, batch, valid_batch)))
        print(shlex.join(inference_command(args, args.run_dir / "ensemble_plan.json", args.delivery_dir, valid_batch)))
        print("DRY RUN ONLY: no model constructed, no training/inference subprocess started, no run files written.")
        return 0
    return orchestrate(args, config, gpu, batch, valid_batch)


if __name__ == "__main__":
    raise SystemExit(main())
