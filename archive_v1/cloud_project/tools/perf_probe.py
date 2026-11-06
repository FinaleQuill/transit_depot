#!/usr/bin/env python3
"""Bounded train-only performance probe; no epoch metrics or weight files.

Each invocation owns one configuration and one fresh CUDA process. gpu reuses a
real train batch (optionally repeats views for a declared capacity test); loader
and e2e consume the actual Study loader. All image/manifest reads stay read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from zoneinfo import ZoneInfo

# Set before numpy/torch and propagate the same limits to spawn workers.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gpu", "loader", "e2e"), required=True)
    parser.add_argument("--batch-size", type=int, default=32, help="Studies, not images")
    parser.add_argument("--view-chunk-size", type=int, default=16)
    parser.add_argument("--view-pad-multiple", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--min-views", type=int, default=0,
                        help="gpu mode only: duplicate real CPU views, with unchanged Study labels")
    parser.add_argument("--largest-bags", action="store_true",
                        help="gpu mode only: use the train Studies with the most complete views")
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--train-manifest", type=Path,
                        default=ROOT / "data/clean_v1_reviewed/train_clean.csv")
    parser.add_argument("--worker-timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("--batch-size must be an even Study count >=2")
    if (args.steps < 1 or args.warmup_steps < 0 or args.view_chunk_size < 1
            or args.num_workers < 0 or args.prefetch_factor < 1 or args.min_views < 0):
        parser.error("invalid nonpositive steps/chunk/prefetch or negative workers/warmup/min-views")
    if args.view_pad_multiple < 0 or (args.view_pad_multiple and args.view_chunk_size % args.view_pad_multiple):
        parser.error("--view-pad-multiple must be 0 or a positive divisor of --view-chunk-size")
    if args.mode != "gpu" and args.min_views:
        parser.error("--min-views is restricted to the declared gpu capacity probe")
    if args.mode != "gpu" and args.largest_bags:
        parser.error("--largest-bags is restricted to gpu mode; e2e uses the normal class-aware sampler")
    if args.output.exists():
        parser.error("--output already exists; use a new file to preserve prior evidence")
    return args


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {}
    seconds = sum(row["total_seconds"] for row in samples)
    result = {
        "measured_steps": len(samples), "seconds": seconds,
        "studies": sum(row["studies"] for row in samples),
        "images": sum(row["images"] for row in samples),
        "max_views_observed": max(row["images"] for row in samples),
        "processed_images": sum(row["processed_images"] for row in samples),
    }
    result["studies_per_second"] = result["studies"] / seconds
    result["images_per_second"] = result["images"] / seconds
    result["padding_images"] = result["processed_images"] - result["images"]
    result["padding_fraction"] = result["padding_images"] / result["processed_images"]
    for name in ("loader_wait_seconds", "compute_seconds", "optimizer_seconds", "ema_seconds"):
        result[name] = sum(row.get(name, 0.0) for row in samples)
        result["mean_" + name] = result[name] / len(samples)
    result["loader_wait_fraction"] = result["loader_wait_seconds"] / seconds
    return result


def probe(args: argparse.Namespace, result: dict) -> None:
    import cv2
    import numpy as np
    import torch

    from dataset_highres import make_study_loader

    result.setdefault("cleanup_errors", [])
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.mode != "loader" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe never falls back to CPU training")
    device = torch.device("cuda", 0)
    if args.mode != "loader":
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": properties.name, "total_gib": properties.total_memory / 2**30}

    # Only this manifest is opened. In particular, no valid/test Dataset exists.
    manifest = args.train_manifest.resolve()
    result["train_manifest_sha256_before"] = digest(manifest)
    startup_begin = time.perf_counter()
    effective_workers = 0 if args.mode == "gpu" else args.num_workers
    result["effective_num_workers"] = effective_workers
    loader = make_study_loader(
        manifest, split="train", batch_size=args.batch_size, num_workers=effective_workers,
        prefetch_factor=args.prefetch_factor, balanced_probability=0.5,
        batches_per_epoch=(max(args.warmup_steps + args.steps, 2) if args.mode == "gpu"
                           else args.warmup_steps + args.steps),
        training=True, timeout=args.worker_timeout, seed=args.seed,
        raw_root=args.image_root, pin_memory=args.mode != "loader")
    loader.batch_sampler.set_epoch(2)
    loader.dataset.reseed((args.seed + 2 * 1009) % 2**32)
    iterator = iter(loader)
    fixed_batch = None
    if args.mode == "gpu":
        if args.largest_bags:
            from dataset_highres import collate_studies
            largest = sorted(range(len(loader.dataset)),
                             key=lambda index: len(loader.dataset.groups[index]), reverse=True)
            fixed_batch = collate_studies([loader.dataset[index] for index in largest[:args.batch_size]])
        else:
            fixed_batch = next(iterator)
        original_views = int(fixed_batch["images"].shape[0])
        target_views = max(args.min_views, args.view_chunk_size, original_views)
        if target_views > original_views:
            source_indices = torch.arange(target_views) % original_views
            fixed_batch["images"] = fixed_batch["images"].index_select(0, source_indices)
            fixed_batch["view_study_indices"] = fixed_batch["view_study_indices"].index_select(0, source_indices)
            fixed_batch["view_counts"] = torch.bincount(
                fixed_batch["view_study_indices"], minlength=args.batch_size)
        fixed_batch["images"] = fixed_batch["images"].pin_memory()
        result["capacity_batch"] = {
            "unique_real_studies": args.batch_size, "original_real_views": original_views,
            "processed_views": target_views, "repeated_views": target_views - original_views,
            "fixed_batch_reused": True,
            "largest_real_train_bags": args.largest_bags,
            "meaning": "capacity/compute benchmark only; repeated views are not additional data or training results",
        }
        assert target_views >= args.view_chunk_size

    model = optimizer = scaler = ema = criterion = anchor = None
    updates = {"optimizer": 0, "ema": 0, "backward": 0, "amp_skipped": 0}
    result["updates"] = updates
    hook = None
    if args.mode != "loader":
        from timm.utils import ModelEmaV3
        from model_highres import StudyBCERankLoss, build_highres_model
        from train_highres import (
            HighResConfig, build_optimizer, checkpointed_view_forward,
            configure_phase, study_logits_from_views,
        )
        config = HighResConfig(output=args.output.parent, batch_size=args.batch_size,
            view_chunk_size=args.view_chunk_size, num_workers=effective_workers,
            view_pad_multiple=args.view_pad_multiple, prefetch_factor=args.prefetch_factor,
            freeze_epochs=0, accumulation_steps=1)
        config.validate()
        model = build_highres_model(pretrained=True, **config.model_config())
        ema = ModelEmaV3(model, decay=config.ema_decay, device="cpu", use_warmup=True,
                        warmup_power=0.75, exclude_buffers=True, foreach=False)
        ema.module.requires_grad_(False)
        model.to(device)
        configure_phase(model, 1, config)
        assert all(parameter.requires_grad for parameter in model.parameters())
        optimizer = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        criterion = StudyBCERankLoss(rank_weight=0.2).to(device)
        anchor = torch.empty(0, device=device)
        result["training_policy"] = {
            "phase": "full_finetune", "all_parameters_trainable": True,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "amp": "float16", "inner_activation_checkpoint": True,
            "outer_cpu_view_checkpoint": True, "rank_weight": 0.2,
            "loss": "StudyBCE+PairwiseLogisticRank", "optimizer": "AdamW",
            "adamw_foreach": False, "adamw_fused": False, "ema_device": "cpu",
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "pretrained": True, "image_size": 384,
            "warmup_excluded_from_throughput": args.warmup_steps,
            "view_pad_multiple": args.view_pad_multiple,
            "padding_outputs_excluded_from_study_and_loss": True,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "backend_environment": {name: os.environ.get(name) for name in
                ("TORCH_CUDNN_V8_API_DISABLED", "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
                 "CUBLAS_WORKSPACE_CONFIG", "CUDA_MODULE_LOADING")},
            "code_sha256": {name: digest(ROOT / name) for name in
                ("train_highres.py", "model_highres.py", "dataset_highres.py", "tools/perf_probe.py")},
        }

        def count_update(_optimizer, _args, _kwargs):
            updates["optimizer"] += 1

        hook = optimizer.register_step_post_hook(count_update)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    result["startup_seconds"] = time.perf_counter() - startup_begin
    result["steps"] = []
    result["warmup"] = []
    measured_start_updates = None
    naturally_exhausted = False
    try:
        for step in range(args.warmup_steps + args.steps):
            if step == args.warmup_steps:
                measured_start_updates = dict(updates)
                result["successful_warmup_updates"] = updates["optimizer"]
                result["adamw_state_ready_before_measured_steps"] = (
                    bool(optimizer.state) if optimizer is not None else None)
            start = time.perf_counter()
            batch = fixed_batch if fixed_batch is not None else next(iterator)
            loaded = time.perf_counter()
            row = {"index": step + 1, "studies": int(batch["targets"].shape[0]),
                   "images": int(batch["images"].shape[0]),
                   "actual_images_chunk_max": min(args.view_chunk_size, int(batch["images"].shape[0])),
                   "loader_wait_seconds": loaded - start if fixed_batch is None else 0.0}
            tail = row["images"] % args.view_chunk_size
            padding = ((-tail) % args.view_pad_multiple
                       if args.view_pad_multiple and tail and args.mode != "loader" else 0)
            row["processed_images"] = row["images"] + padding
            row["padding_images"] = padding
            row["padding_fraction"] = padding / row["processed_images"]
            row["real_images_chunk_max"] = row["actual_images_chunk_max"]
            row["actual_images_chunk_max"] = min(args.view_chunk_size, row["processed_images"])
            if step == 0:
                result["first_batch_wait_seconds"] = row["loader_wait_seconds"]
            if args.mode != "loader":
                optimizer.zero_grad(set_to_none=True)
                view_logits = checkpointed_view_forward(model, batch["images"], device, anchor,
                    chunk_size=args.view_chunk_size, pad_multiple=args.view_pad_multiple)
                study_logits = study_logits_from_views(
                    view_logits, batch["view_study_indices"], batch["view_counts"])
                targets = batch["targets"].to(device, dtype=torch.float32, non_blocking=True)
                loss, bce, rank = criterion(study_logits, targets, rank_weight=0.2)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                scaler.scale(loss).backward()
                updates["backward"] += 1
                row["loss"] = float(loss.detach())
                del loss, bce, rank, view_logits, study_logits, targets
                torch.cuda.synchronize(device)
                computed = time.perf_counter()
                row["compute_seconds"] = computed - loaded
                before_updates = updates["optimizer"]
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                           error_if_nonfinite=False)
                row["finite_gradient_norm"] = bool(torch.isfinite(grad_norm))
                row["grad_scaler_scale_before"] = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                optimized = time.perf_counter()
                row["optimizer_seconds"] = optimized - computed
                row["optimizer_updated"] = updates["optimizer"] > before_updates
                if row["optimizer_updated"]:
                    ema.update(model, step=updates["optimizer"])
                    updates["ema"] += 1
                else:
                    updates["amp_skipped"] += 1
                torch.cuda.synchronize(device)
                row["ema_seconds"] = time.perf_counter() - optimized
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            row["total_seconds"] = time.perf_counter() - start
            partition = "warmup" if step < args.warmup_steps else "steps"
            result[partition].append(row)
            print(json.dumps({"phase": partition, **row}, ensure_ascii=False, allow_nan=False), flush=True)
        result["summary"] = summarize(result["steps"])
        result["measured_updates"] = {key: updates[key] - measured_start_updates[key] for key in updates}
        result["warmup_excluded_steps"] = args.warmup_steps
        if fixed_batch is None:
            # Consume the exact bounded iterator to StopIteration. There are no
            # additional prefetched batches for workers/pin threads to abandon.
            try:
                next(iterator)
            except StopIteration:
                naturally_exhausted = True
                result["loader_naturally_exhausted"] = True
            else:
                raise RuntimeError("loader yielded more than the configured bounded probe steps")
        if args.mode != "loader":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            result["adamw_state_parameters"] = len(optimizer.state)
            if updates["optimizer"] < 1 or not optimizer.state:
                raise RuntimeError("No actual AdamW update/state: cannot accept this as a successful capacity test")
            if result["measured_updates"]["optimizer"] < 1:
                raise RuntimeError("No successful optimizer update in measured steps")
        result["status"] = "ok"
    finally:
        primary_exception_active = sys.exc_info()[0] is not None

        def cleanup_action(label, action):
            try:
                action()
            except BaseException as cleanup_exc:
                result["cleanup_errors"].append({"action": label,
                    "error_type": type(cleanup_exc).__name__, "error": str(cleanup_exc),
                    "traceback": traceback.format_exc()})
                print(f"cleanup error ({label}): {type(cleanup_exc).__name__}: {cleanup_exc}",
                      file=sys.stderr, flush=True)

        if hook is not None:
            cleanup_action("optimizer_hook_remove", hook.remove)
        # Failure cleanup is best-effort and must never replace the original
        # exception, especially a CUDA OOM that must retain exit code 42.
        if (not naturally_exhausted and not getattr(iterator, "_shutdown", False)
                and hasattr(iterator, "_shutdown_workers")):
            cleanup_action("loader_shutdown_workers", iterator._shutdown_workers)

        def verify_manifest():
            result["train_manifest_sha256_after"] = digest(manifest)
            result["train_manifest_unchanged"] = (
                result["train_manifest_sha256_before"] == result["train_manifest_sha256_after"])
            if not result["train_manifest_unchanged"]:
                raise RuntimeError("train manifest changed during probe")

        cleanup_action("verify_train_manifest", verify_manifest)
        result["cleanup_status"] = "error" if result["cleanup_errors"] else "ok"
        if result["cleanup_errors"] and not primary_exception_active:
            result["status"] = "error"
            raise RuntimeError("probe cleanup failed; see cleanup_errors; this run is not accepted as ok")


def main() -> int:
    args = arguments()
    result = {"schema_version": 1, "status": "starting", "mode": args.mode, "cleanup_errors": [],
        "created_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "data_policy": "train whitelist only; raw images and labels read-only; no valid/test reads; no saved weights/epoch metrics",
        "script_sha256": digest(Path(__file__))}
    code = 0
    try:
        probe(args, result)
    except BaseException as exc:
        trace = traceback.format_exc()
        traceback.print_exc()
        oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        result.update(status="oom" if oom else "error", error_type=type(exc).__name__,
                      error=str(exc), traceback=trace)
        torch = sys.modules.get("torch")
        if torch is not None and args.mode != "loader" and torch.cuda.is_initialized():
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        code = 42 if oom else 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents accidentally overwriting a concurrent probe.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "exit_code": code}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
