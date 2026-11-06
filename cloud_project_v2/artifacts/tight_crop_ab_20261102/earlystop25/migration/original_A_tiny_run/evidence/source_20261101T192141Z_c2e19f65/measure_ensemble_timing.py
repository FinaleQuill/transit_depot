"""Measure a fixed student ensemble end to end with the shared 100-image protocol.

The timing loop is exactly ``cost.time_round``: batch-1 serial read/decode,
all member forwards/views, Study aggregation, actual CSV write and flush.  Five
additional images warm the decoder/model path; three measured rounds must agree
bit-for-bit.  This is a 2080Ti development diagnostic, not a T4 certificate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from benchmark_local_resources import validate_inputs, validate_round_output
from deploy_kd import sha256_file
from inference_reproducibility import configure_inference
from student_ensemble import StudentEnsemble
from student_experiment_evaluation import write_new_json
import cost


def measure(*, model_dir, inputs, manifest, output_dir):
    inputs, output_dir = Path(inputs), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new ensemble timing output directory")
    selection_path = inputs / "selection.portable.json"
    if not selection_path.is_file():
        selection_path = inputs / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Timing selection is not this official valid manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    relocated = {**selection, "manifest_path": str(Path(manifest).resolve())}
    relocated_path = output_dir / "selection.relocated.json"
    write_new_json(relocated_path, relocated)
    measured, warmup, coverage = validate_inputs(inputs / "pictures_100",
                                                 inputs / "pictures_5", relocated_path)
    configure_inference()
    predictor = StudentEnsemble(model_dir)
    environment = cost.timing_environment(predictor)
    for record in warmup:
        predictor.predict_image(record["path"])
    predictor.synchronize()
    durations, outputs, comparisons, first = [], [], [], None
    for index in range(3):
        path = output_dir / f"round_{index + 1}.csv"
        times, probabilities = cost.time_round(predictor, measured, path)
        durations.append(times)
        outputs.append(validate_round_output(measured, path, probabilities))
        if first is None:
            first = probabilities
        comparisons.append(cost.compare_predictions(probabilities, first, atol=0))
    rounds, per_image, mean_ms = cost.summarize_timings(durations)
    if not all(item["match"] for item in comparisons):
        raise RuntimeError("Timing rounds changed deterministic predictions")
    result = {
        "T4_certified": False,
        "development_diagnostic_only": True,
        "model_dir": str(Path(model_dir).resolve(strict=True)),
        "config_sha256": sha256_file(Path(model_dir) / "config.json"),
        "protocol": {"images": 100, "warmup": 5, "rounds": 3, "batch": 1,
                     "execution": "serial", "timing": "read_file_to_actual_CSV_write",
                     "model_load_excluded": True},
        "coverage": coverage,
        "mean_ms": mean_ms,
        "round_mean_ms": rounds.mean(axis=1).tolist(),
        "per_image_round_ms": rounds.T.tolist(),
        "per_image_mean_ms": per_image.tolist(),
        "p95_ms": float(np.percentile(per_image, 95)),
        "environment": environment,
        "round_outputs": outputs,
        "round_comparisons": comparisons,
    }
    write_new_json(output_dir / "latency_report.json", result)
    del predictor
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "deliveries/034_dual768_proxy_timing_20261029/bundle/inputs")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "data/clean_v1_reviewed/valid_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = measure(**vars(args))
    print(json.dumps({"mean_ms": result["mean_ms"], "round_mean_ms": result["round_mean_ms"],
                      "T4_certified": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
