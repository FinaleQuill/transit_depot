"""Short synthetic training mechanics benchmark; no scientific validation data.

Run in the pinned runtime. Every trial includes AMP, backward, AdamW and EMA,
with initialization from the approved public weights. Timings exclude startup
and real-image loading; these are training-throughput estimates, never cost.py
or inference-compliance measurements. Each requested variant is a fresh process
so OOM/allocator history cannot contaminate subsequent variants.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import torch

from reconstruction.model import StudyModel, study_bce_loss
from reconstruction.train import update_ema, write_json


def run_benchmark(args):
    torch.set_num_threads(1)
    torch.manual_seed(918)
    torch.cuda.manual_seed_all(918)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = StudyModel("convnext_base.fb_in22k_ft_in1k_384", args.pretrained_path,
        drop_rate=.2, drop_path=.15, gradient_checkpointing=not args.no_checkpoint,
        checkpoint_mode=args.checkpoint_mode, channels_last=args.channels_last,
        view_chunk_size=args.chunk_size).cuda().train()
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    pixels = torch.randn(args.views, 3, 768, 768, device="cuda")
    if args.channels_last:
        pixels = pixels.contiguous(memory_format=torch.channels_last)
    # Pack complete two-image Studies; an odd final Study retains its one image.
    studies = list(pixels.split(2)) if args.packed else [pixels]
    targets = [(torch.arange(10, device="cuda") % 3 == i % 3).float() for i in range(len(studies))]
    weights = torch.ones(10, device="cuda")
    attempted, successful = 0, 0

    def step():
        nonlocal attempted, successful
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model.forward_studies(studies)
            loss = torch.stack([study_bce_loss(z, y, weights) for z, y in zip(outputs, targets)]).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempted += 1
        if scaler.get_scale() >= old_scale:
            successful += 1
            update_ema(ema, model, successful)
        return float(loss.detach())

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    seconds, losses = [], []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
    result = {"kind": "synthetic_training_mechanics_only", "gpu": torch.cuda.get_device_name(),
        "views": args.views, "studies_per_update": len(studies), "image_size": 768,
        "channels_last": args.channels_last, "checkpoint_mode": "none" if args.no_checkpoint else args.checkpoint_mode,
        "view_chunk_size": args.chunk_size, "warmup": args.warmup, "repeats": args.repeats,
        "seconds": seconds, "mean_seconds": statistics.mean(seconds),
        "images_per_second": args.views / statistics.mean(seconds),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "attempted_optimizer_steps": attempted, "successful_optimizer_steps": successful,
        "losses": losses, "includes": ["forward", "backward", "AdamW", "EMA"],
        "excludes": ["source image loading", "initialization", "validation", "checkpoint writing"],
        "data": "synthetic random tensors only"}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-mode", choices=["nested", "chunks"], default="nested")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.views, args.chunk_size, args.warmup, args.repeats) < 1:
        parser.error("Positive views/chunk/warmup/repeats required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Benchmark evidence already exists")
    try:
        run_benchmark(args)
    except Exception as error:
        write_json(args.output, {"state": "failed", "error": f"{type(error).__name__}: {error}",
            "views": args.views, "channels_last": args.channels_last,
            "checkpoint_mode": args.checkpoint_mode, "chunk_size": args.chunk_size,
            "data": "synthetic random tensors only"})
        raise


if __name__ == "__main__":
    main()
