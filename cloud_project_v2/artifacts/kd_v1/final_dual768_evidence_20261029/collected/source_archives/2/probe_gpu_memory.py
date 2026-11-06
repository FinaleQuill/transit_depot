"""768px synthetic memory preflight; NEVER starts training or updates weights.

Each batch runs in a fresh process with GPU EMA, FP32 Adam moments and existing
gradients resident throughout BF16/checkpointed accumulation microbatches.
No real image is sent to a model, optimizer.step is never called, and no model
checkpoint is saved. CSV data only supplies the actual class weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parent


def single_probe(batch: int, steps: int, manifest: Path) -> dict:
    import torch
    from timm.utils import ModelEmaV3
    from dataset import read_manifest
    from model_overnight import build_overnight_model
    from train_overnight import build_weighted_bce, compute_loss_config

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 AMP are required")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = {"batch_size": batch, "input_shape": [batch, 3, 768, 768],
              "amp": "bfloat16", "gradient_checkpointing": True,
              "pretrained": False, "optimizer_steps": 0, "real_images": 0,
              "gpu": torch.cuda.get_device_name(0), "status": "error",
              "total_device_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    stop = threading.Event()
    samples = []

    def monitor() -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        while not stop.is_set():
            try:
                query = subprocess.run(["nvidia-smi", "-i", visible,
                    "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True)
                samples.append(tuple(float(v) for v in query.stdout.strip().split(",")))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        loss_config = compute_loss_config(read_manifest(manifest, split="train"),
            source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), fold=None)
        criterion = build_weighted_bce(loss_config, device=torch.device("cuda"))
        result["loss_config"] = loss_config
        model = build_overnight_model(pretrained=False).to(
            "cuda", memory_format=torch.channels_last).train()
        if not all(stage.grad_checkpointing for stage in model.stages):
            raise RuntimeError("Not all stages enabled gradient checkpointing")
        result["parameters"] = sum(p.numel() for p in model.parameters())
        ema = ModelEmaV3(model, decay=0.999, device="cuda", use_warmup=True,
                         warmup_power=0.75, exclude_buffers=True, foreach=True)
        ema.module.requires_grad_(False)
        moments = [(torch.zeros_like(p), torch.zeros_like(p))
                   for p in model.parameters() if p.requires_grad]
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        result["resident_adam_moment_gib"] = sum(
            t.numel() * t.element_size() for pair in moments for t in pair) / 2**30
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result["baseline_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
        host_images = torch.randn(batch, 3, 768, 768).pin_memory()
        targets = torch.randint(0, 2, (batch, 10), device="cuda").float()
        times = []
        accumulation_steps = 32 // batch
        result["accumulation_steps"] = accumulation_steps
        for index in range(steps + 2):
            if index == 2:
                result["cold_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                result["cold_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            before = time.perf_counter()
            images = host_images.to("cuda", memory_format=torch.channels_last, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            if logits.shape != (batch, 10):
                raise RuntimeError("Model must return ten logits")
            loss = criterion(logits.float(), targets).mean() / accumulation_steps
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite weighted BCE")
            loss.backward()
            torch.cuda.synchronize()
            if index >= 2:
                times.append(time.perf_counter() - before)
            result["loss"] = float(loss.detach()) * accumulation_steps
            result["logits_dtype"] = str(logits.dtype)
            del images, logits, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
        if not torch.stack(finite).all().item():
            raise FloatingPointError("Nonfinite accumulated gradients")
        result.update(status="ok", gradients_finite=True, warmup_microbatches=2,
            measured_microbatches=steps, median_microbatch_seconds=statistics.median(times),
            synthetic_images_per_second=batch / statistics.median(times))
        result["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["training_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        valid_batch = min(2 * batch, 8)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            valid_images = torch.randn(valid_batch, 3, 768, 768, device="cuda").contiguous(
                memory_format=torch.channels_last)
            valid_logits = ema.module(valid_images)
            if valid_logits.shape != (valid_batch, 10) or not torch.isfinite(valid_logits).all():
                raise FloatingPointError("Synthetic EMA evaluation failed")
        torch.cuda.synchronize()
        result["synthetic_valid_batch_size"] = valid_batch
        result["synthetic_valid_finite"] = True
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        torch.cuda.synchronize()
        result.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            seconds=round(time.monotonic() - started, 3))
        stop.set()
        thread.join(timeout=6)
        if samples:
            result["sampled_peak_device_used_gib"] = max(s[0] for s in samples) / 1024
            result["sampled_peak_gpu_utilization_percent"] = max(s[1] for s in samples)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--batches", type=int, choices=(1, 2, 4), nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/clean_v1_reviewed/train_clean.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/upgrade_768/gpu_probe_768.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.single:
        result = single_probe(args.batch_size, args.steps, args.manifest)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "ok" else 2
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NO_ALBUMENTATIONS_UPDATE": "1"}
    report = {"created_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "python": sys.executable, "synthetic_only": True,
              "real_training_started": False, "optimizer_steps": 0,
              "versions": {name: importlib.metadata.version(name)
                           for name in ("torch", "torchvision", "timm", "albumentations")},
              "results": []}
    for batch in args.batches:
        print(f"768px BF16 + checkpoint + EMA/Adam resident probe: batch={batch}", flush=True)
        try:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single",
                "--batch-size", str(batch), "--steps", str(args.steps), "--manifest", str(args.manifest)],
                env=env, text=True, capture_output=True, timeout=600)
            try:
                result = json.loads(child.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                result = {"batch_size": batch, "status": "error", "stdout": child.stdout, "stderr": child.stderr}
            result["exit_code"] = child.returncode
            if child.stderr:
                print(child.stderr, file=sys.stderr, end="", flush=True)
        except subprocess.TimeoutExpired:
            result = {"batch_size": batch, "status": "timeout"}
        report["results"].append(result)
        safe = [r for r in report["results"] if r["status"] == "ok"
            and max(r["peak_reserved_gib"], r.get("sampled_peak_device_used_gib", 0))
                <= min(r["total_device_gib"] * 0.8, r["total_device_gib"] - 3)]
        report["largest_safe_batch_size"] = max((r["batch_size"] for r in safe), default=None)
        report["recommended_batch_size"] = max(safe, key=lambda r: r["synthetic_images_per_second"])["batch_size"] if safe else None
        report["selection_policy"] = "fastest measured synthetic throughput among safe tested batches"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}; recommended batch={report['recommended_batch_size']}", flush=True)
    return int(report["recommended_batch_size"] is None or any(r["status"] not in ("ok", "oom") for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
