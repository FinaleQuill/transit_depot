#!/usr/bin/env python3
"""Train-only CPU loader upper bound; no model, CUDA, or saved pixels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import torch
    from dataset_highres import make_study_loader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    torch.set_num_threads(1)
    results = []
    for workers in (8, 12, 16):
        started = time.perf_counter()
        loader = make_study_loader(batch_size=64, num_workers=workers,
            batches_per_epoch=40, prefetch_factor=1, pin_memory=False,
            seed=43, raw_root="/home/featurize/data/cxr_first")
        loader.batch_sampler.set_epoch(2)
        iterator = iter(loader)
        first = None
        rows = []
        try:
            for index in range(40):
                begin = time.perf_counter()
                batch = next(iterator)
                elapsed = time.perf_counter() - begin
                if first is None:
                    first = time.perf_counter() - started
                if index >= 8:
                    rows.append({"images": len(batch["image_ids"]), "seconds": elapsed})
                del batch
            # Exhaust the finite sampler before joining; no unfinished prefetch
            # transfers should be terminated during a successful measurement.
            try:
                next(iterator)
                raise RuntimeError("unexpected extra batch")
            except StopIteration:
                pass
        finally:
            iterator._shutdown_workers()
        images = sum(row["images"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        result = {"workers": workers, "batch_studies": 64,
            "prefetch_factor": 1, "warmup_steps": 8, "measured_steps": 32,
            "first_batch_ready_seconds": first, "images": images,
            "seconds": seconds, "images_per_second": images / seconds,
            "pin_memory": False, "model_or_gpu_used": False,
            "data_policy": "train whitelist only; CPU loader upper bound, not training throughput"}
        results.append(result)
        print(json.dumps(result), flush=True)
        with args.output.open("w") as stream:
            json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
