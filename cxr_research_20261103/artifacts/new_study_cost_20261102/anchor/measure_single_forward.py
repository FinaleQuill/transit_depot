from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from deploy_kd import sha256_file
from student_experiment_evaluation import ExperimentalPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    model_dir = Path(args.model_dir).resolve(strict=True)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    weight_path = model_dir / config["weights_file"]
    predictor = ExperimentalPredictor(model_dir, tta="identity", device="cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = predictor.model
    model.eval().requires_grad_(False)
    image_size = int(config["image_size"])
    x = torch.zeros((1, 3, image_size, image_size), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()
        samples_ms = []
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(samples_ms)
    result = {
        "label": "near-official / 2080Ti reference; not a competition Gate",
        "name": args.name,
        "backbone": config["model_config"]["backbone_name"],
        "image_size": image_size,
        "precision": "fp16",
        "forward_views": 1,
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "weights_path": str(weight_path),
        "weights_sha256": sha256_file(weight_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "median_ms": statistics.median(samples_ms),
        "p95_ms_nearest_rank": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
