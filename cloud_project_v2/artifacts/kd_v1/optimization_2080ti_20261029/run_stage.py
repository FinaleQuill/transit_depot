"""Seeded, uninstrumented validation/benchmark launcher; never trains."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "valid"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    assert not args.output_dir.exists(), "Preserve all previous runs"
    assert not any(key.startswith("CUDA_MPS_") for key in os.environ), "No MPS"
    assert sys.version_info[:2] == (3, 12), sys.version
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch

    assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    props = torch.cuda.get_device_properties(0)
    source_files = {}
    for path in sorted(args.source_root.glob("*.py")):
        source_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv, "source_root": str(args.source_root),
        "output_dir": str(args.output_dir), "seed": 43,
        "source_sha256": source_files, "gpu": props.name,
        "sm": props.multi_processor_count, "allocator_gib": 8,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "timm", "numpy", "Pillow")},
        "unmodified_model_dir": str(args.model_dir), "training": False,
        "uninstrumented": True,
    }
    started = time.perf_counter()
    code = 1
    try:
        if args.stage == "benchmark":
            assert args.bundle_root is not None
            from benchmark_local_resources import run_benchmark
            report = run_benchmark(
                model_dir=args.model_dir,
                pictures_100=args.bundle_root / "inputs/pictures_100",
                pictures_5=args.bundle_root / "inputs/pictures_5",
                selection_json=args.bundle_root / "inputs/selection.portable.json",
                output_dir=args.output_dir, profile=args.output_dir.name,
                baseline_sm_count=props.multi_processor_count,
                memory_limit_gib=8, execute=True)
            code = 0 if report["comparison"]["status"] == "matched" else 2
        else:
            assert args.image_root is not None and args.manifest is not None
            torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / props.total_memory)
            from validate_student import validate_student
            report = validate_student(model_dir=args.model_dir,
                                      image_root=args.image_root,
                                      manifest=args.manifest,
                                      output_dir=args.output_dir)
            code = 0 if report["complete"] else 2
            print(json.dumps({key: report[key] for key in
                              ("num_images", "num_studies", "macro_auc",
                               "per_class_auc", "macro_f1_fixed_0_5")}, indent=2))
    except BaseException:
        receipt["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        receipt.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       wall_seconds=time.perf_counter()-started, exit_code=code)
        path = args.output_dir.with_name(args.output_dir.name + ".receipt.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
