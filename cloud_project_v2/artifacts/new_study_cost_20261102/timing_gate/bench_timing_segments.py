"""Profile Tiny/Base single-pass inference by complete valid Study.

This isolated diagnostic does not train, read labels/test files, or modify the
deployment sources. It benchmarks a Study-local batch of 1-3 images with the
frozen decoder, letterbox, FP16 model, sigmoid, and fixed FP64 Study mean.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_VALID = Path("/home/featurize/data/cxr_first/valid")
DEFAULT_TINY = Path("/home/featurize/work/deliveries/044_tiny_base_class1B_draft768_20261031/model/A")
DEFAULT_BASE = Path("/home/featurize/work/deliveries/044_tiny_base_class1B_draft768_20261031/model/B")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ROUNDS = 3
WARMUP_STUDIES = 5
ANCHOR_WARMUP = 20
ANCHOR_RUNS = 100


def _runtime_versions(runtime_label: str, image_digest: str | None) -> dict:
    import torch
    import fast_lanczos
    props = torch.cuda.get_device_properties("cuda")
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10).splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        driver = None
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "pillow": PILLOW_VERSION, "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "driver": driver,
        "resize_implementation": fast_lanczos.implementation(),
        "runtime_label": runtime_label,
        "official_image_digest": image_digest,
        "gate_eligible": False,
    }


def _discover_valid_studies(valid_root: Path) -> dict[int, list[dict]]:
    valid_root = valid_root.resolve(strict=True)
    if valid_root.name != "valid" or "test" in valid_root.parts:
        raise ValueError("Only an explicit valid/ root is accepted")
    groups: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for subject_dir in sorted(valid_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.is_symlink():
            continue
        for study_dir in sorted(subject_dir.iterdir()):
            if not study_dir.is_dir() or study_dir.is_symlink():
                continue
            images = sorted(path for path in study_dir.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if len(images) in groups:
                if any(not path.resolve().is_relative_to(valid_root) for path in images):
                    raise ValueError("A valid image path escapes the valid/ root")
                groups[len(images)].append({
                    "study_key": f"{subject_dir.name}/{study_dir.name}",
                    "subject_id": subject_dir.name, "study_id": study_dir.name,
                    "paths": images,
                })
    return groups


def _choose_studies(groups: dict[int, list[dict]], per_count: int, seed: int):
    rng = random.Random(seed)
    remaining = {}
    for count in (1, 2, 3):
        values = sorted(groups[count], key=lambda row: row["study_key"])
        rng.shuffle(values)
        if len(values) < per_count + 2:
            raise ValueError(f"Only {len(values)} valid Studies have {count} images")
        remaining[count] = values
    warmup = []
    warmup.extend(remaining[1][:2])
    warmup.extend(remaining[2][:2])
    warmup.extend(remaining[3][:1])
    measured = {count: remaining[count][(2 if count < 3 else 1):
                                              (2 if count < 3 else 1) + per_count]
                for count in (1, 2, 3)}
    return warmup, measured


def _decode(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {"I", "F"} or source.mode.startswith("I;"):
            raise ValueError(f"Unsupported non-8-bit source: {path.name}")
        return source.copy() if source.mode == "L" else source.convert("RGB")


def _event_pair(torch):
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_ms(pairs) -> float:
    return float(sum(start.elapsed_time(end) for start, end in pairs))


def _stage_input(predictor, lookup, pixels):
    """Return a deployment-equivalent tensor and separate host prep/H2D events."""
    torch = predictor.torch
    import torch.nn.functional as F
    host_prepare_ms = 0.0
    h2d_pairs, normalize_pairs = [], []
    if pixels.dtype != np.uint8:
        raise ValueError("Expected uint8 pixels")
    if pixels.ndim == 2 and pixels.shape in lookup.staging:
        host, writable, device_pixels = lookup.staging[pixels.shape]
        started = time.perf_counter()
        np.copyto(writable, pixels)
        host_prepare_ms += (time.perf_counter() - started) * 1000
        h2d_start, h2d_end = _event_pair(torch)
        h2d_start.record()
        device_pixels.copy_(host, non_blocking=True)
        h2d_end.record()
        h2d_pairs.append((h2d_start, h2d_end))
        normalize_start, normalize_end = _event_pair(torch)
        normalize_start.record()
        indices = device_pixels.to(dtype=torch.int32)
        hwc = F.embedding(indices, lookup.table_fp16)
        tensor = hwc.unsqueeze(0).permute(0, 3, 1, 2)
        normalize_end.record()
        normalize_pairs.append((normalize_start, normalize_end))
        return tensor, host_prepare_ms, h2d_pairs, normalize_pairs

    from inference_inputs import normalize_pixels
    started = time.perf_counter()
    array = normalize_pixels(pixels, lookup.mean, lookup.std)
    host_prepare_ms += (time.perf_counter() - started) * 1000
    host = torch.from_numpy(array).unsqueeze(0)
    h2d_start, h2d_end = _event_pair(torch)
    h2d_start.record()
    tensor = host.to(predictor.device, dtype=torch.float16,
                     memory_format=torch.channels_last)
    h2d_end.record()
    h2d_pairs.append((h2d_start, h2d_end))
    return tensor, host_prepare_ms, h2d_pairs, normalize_pairs


def _anchor(predictor) -> dict:
    torch, model = predictor.torch, predictor.model
    value = torch.empty((1, 3, 768, 768), device=predictor.device,
                        dtype=torch.float16, memory_format=torch.channels_last).zero_()
    with torch.inference_mode():
        for _ in range(ANCHOR_WARMUP):
            model(value)
        torch.cuda.synchronize(predictor.device)
        pairs = []
        for _ in range(ANCHOR_RUNS):
            start, end = _event_pair(torch)
            start.record()
            output = model(value)
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize(predictor.device)
        event_ms = np.asarray([start.elapsed_time(end) for start, end in pairs])
    del output, value
    torch.cuda.empty_cache()
    return {"warmup_forwards": ANCHOR_WARMUP, "measured_forwards": ANCHOR_RUNS,
            "cuda_event_ms": {"p50": float(np.percentile(event_ms, 50)),
                              "p95": float(np.percentile(event_ms, 95)),
                              "mean": float(event_ms.mean())},
            "measurement": "CUDA Event around model(x), FP16, batch=1, no sigmoid/TTA/H2D"}


def _run_study(predictor, lookups, study: dict, study_writer, image_writer):
    torch = predictor.torch
    torch.cuda.synchronize(predictor.device)
    started = time.perf_counter()
    cpu_started = time.process_time()

    decode_started = time.perf_counter()
    images = [_decode(path) for path in study["paths"]]
    decode_ms = (time.perf_counter() - decode_started) * 1000

    resize_started = time.perf_counter()
    from deploy_kd import _letterbox_image
    pixels = [np.asarray(_letterbox_image(image, 768, image.mode)) for image in images]
    resize_ms = (time.perf_counter() - resize_started) * 1000

    tensors, h2d_pairs, normalize_pairs = [], [], []
    normalize_cpu_ms = 0.0
    for lookup, array in zip(lookups, pixels):
        tensor, host_ms, h2d, normalize = _stage_input(predictor, lookup, array)
        tensors.append(tensor)
        normalize_cpu_ms += host_ms
        h2d_pairs.extend(h2d)
        normalize_pairs.extend(normalize)
    if len(tensors) > 1:
        start, end = _event_pair(torch)
        start.record()
        batch = torch.cat(tensors, dim=0)
        end.record()
        normalize_pairs.append((start, end))
    else:
        batch = tensors[0]

    forward_start, forward_end = _event_pair(torch)
    with torch.inference_mode():
        forward_start.record()
        logits = predictor.model(batch)
        forward_end.record()
        if tuple(logits.shape) != (len(pixels), 10):
            raise FloatingPointError("Unexpected model output shape")
        finite = torch.isfinite(logits).all()
        post_start_wall = time.perf_counter()
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        if not bool(finite.item()):
            raise FloatingPointError("Nonfinite model logits")
        postprocess_ms = (time.perf_counter() - post_start_wall) * 1000

    torch.cuda.synchronize(predictor.device)
    h2d_ms = _event_ms(h2d_pairs)
    normalize_gpu_ms = _event_ms(normalize_pairs)
    forward_ms = float(forward_start.elapsed_time(forward_end))
    probabilities = [np.asarray(value, dtype=np.float64) for value in probs]

    aggregate_started = time.perf_counter()
    study_probability = np.mean(np.stack(probabilities, axis=0), axis=0, dtype=np.float64)
    aggregate_ms = (time.perf_counter() - aggregate_started) * 1000

    write_started = time.perf_counter()
    for path, probability in zip(study["paths"], probabilities):
        image_writer.writerows([[path.name, study["subject_id"], study["study_id"],
                                 class_id + 1, repr(float(value))]
                                for class_id, value in enumerate(probability)])
    study_writer.writerows([[study["subject_id"], study["study_id"], class_id + 1,
                             repr(float(value))]
                            for class_id, value in enumerate(study_probability)])
    image_writer.stream.flush()
    study_writer.stream.flush()
    write_ms = (time.perf_counter() - write_started) * 1000
    torch.cuda.synchronize(predictor.device)
    total_ms = (time.perf_counter() - started) * 1000
    cpu_ms = (time.process_time() - cpu_started) * 1000
    return {
        "study_key": study["study_key"], "images": len(images),
        "decode_ms": decode_ms, "resize_ms": resize_ms,
        "normalize_ms": normalize_cpu_ms + normalize_gpu_ms,
        "normalize_host_component_ms": normalize_cpu_ms,
        "normalize_device_component_ms": normalize_gpu_ms,
        "h2d_ms": h2d_ms, "forward_ms": forward_ms,
        "postprocess_ms": postprocess_ms, "aggregate_ms": aggregate_ms,
        "write_ms": write_ms, "total_ms": total_ms,
        "cpu_process_ms": cpu_ms,
        "cpu_utilization_one_core_percent": (cpu_ms / total_ms * 100 if total_ms else 0),
        "study_probability": study_probability.tolist(),
    }


def _summary(rows: list[dict]) -> dict:
    fields = ("decode_ms", "resize_ms", "normalize_ms", "h2d_ms", "forward_ms",
              "postprocess_ms", "aggregate_ms", "write_ms", "total_ms",
              "cpu_utilization_one_core_percent")
    result = {"studies": len(rows)}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        result[field] = {"p50": float(np.percentile(values, 50)),
                         "p95": float(np.percentile(values, 95)),
                         "mean": float(values.mean())}
    return result


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "round", "study_key", "image_count", "decode_ms",
                         "resize_ms", "normalize_ms", "h2d_ms", "forward_ms",
                         "postprocess_ms", "aggregate_ms", "write_ms", "total_ms",
                         "cpu_utilization_one_core_percent"])
        for row in rows:
            writer.writerow([row["model"], row["round"], row["study_key"], row["images"],
                             *[row[field] for field in ("decode_ms", "resize_ms", "normalize_ms",
                               "h2d_ms", "forward_ms", "postprocess_ms", "aggregate_ms",
                               "write_ms", "total_ms", "cpu_utilization_one_core_percent")]])


def run(args) -> dict:
    sys.path.insert(0, str(PROJECT))
    import torch
    from deploy_kd import CSV_HEADER, IMAGE_CSV_HEADER
    from inference_reproducibility import configure_inference
    from inference_inputs import ExactU8Input
    from student_experiment_evaluation import ExperimentalPredictor
    from deploy_kd import MEAN, STD, sha256_file

    valid_root = Path(args.valid_root).resolve(strict=True)
    groups = _discover_valid_studies(valid_root)
    warmup, measured = _choose_studies(groups, args.studies_per_count, args.seed)
    if len({row["study_key"] for row in warmup} &
           {row["study_key"] for values in measured.values() for row in values}):
        raise ValueError("Warmup and measured Study identities overlap")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    if not output.is_relative_to((PROJECT / "artifacts").resolve()):
        raise ValueError("Outputs must remain under ignored project artifacts/")
    output.mkdir(parents=True)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    reproducibility = configure_inference()

    model_dirs = {"Tiny768": Path(args.tiny_model).resolve(strict=True),
                  "Base768": Path(args.base_model).resolve(strict=True)}
    results, all_rows = {}, []
    for model_name, model_dir in model_dirs.items():
        predictor = ExperimentalPredictor(model_dir, tta="identity")
        if predictor.config.get("image_size") != 768:
            raise ValueError("Expected a 768 model")
        max_images = 3
        lookups = [ExactU8Input(torch, predictor.device, MEAN, STD, image_sizes=[768])
                   for _ in range(max_images)]
        anchor = _anchor(predictor)

        for study in warmup:
            # Warm-up uses the same decode/resize/input/forward/result path, but
            # does not open output CSVs or contribute to percentiles.
            _run_study(predictor, lookups, study,
                       _NullWriter(), _NullWriter())
        predictor.synchronize()

        model_rows = []
        for round_index in range(1, ROUNDS + 1):
            round_rows = []
            study_rows = [study for count in (1, 2, 3) for study in measured[count]]
            random.Random(args.seed + round_index).shuffle(study_rows)
            study_path = output / f"{model_name}_round_{round_index}_studies.csv"
            image_path = output / f"{model_name}_round_{round_index}_images.csv"
            with study_path.open("x", newline="", encoding="utf-8") as study_stream, \
                    image_path.open("x", newline="", encoding="utf-8") as image_stream:
                study_writer = _CSVWriter(study_stream, CSV_HEADER)
                image_writer = _CSVWriter(image_stream, IMAGE_CSV_HEADER)
                for study in study_rows:
                    row = _run_study(predictor, lookups, study, study_writer, image_writer)
                    row.update(model=model_name, round=round_index)
                    round_rows.append(row)
            model_rows.extend(round_rows)
            all_rows.extend(round_rows)
        grouped = {str(count): _summary([row for row in model_rows if row["images"] == count])
                   for count in (1, 2, 3)}
        results[model_name] = {
            "model_dir": str(model_dir),
            "config_sha256": sha256_file(model_dir / "config.json"),
            "weights_sha256": sha256_file(model_dir / "student_fp16.pt"),
            "single_forward_anchor": anchor,
            "study_batches": grouped,
            "measurement_count_per_image_count": args.studies_per_count * ROUNDS,
            "raw_rows": len(model_rows),
        }
        del predictor, lookups
        gc_collect = getattr(torch, "cuda", None)
        if gc_collect is not None:
            torch.cuda.empty_cache()

    _write_rows(output / "per_study_timings.csv", all_rows)
    report = {
        "purpose": "2080Ti near-official local screening only; not official T4 timing certification",
        "protocol": {
            "split": "valid only; image paths/bytes only; no labels read",
            "study_counts": {str(k): len(v) for k, v in measured.items()},
            "studies_per_image_count_per_round": args.studies_per_count,
            "rounds": ROUNDS, "warmup_studies_per_model": WARMUP_STUDIES,
            "study_batch": "all images from one Study in one FP16 model forward; no cross-Study batching",
            "tta": "identity (single forward)", "model_load_in_timing": False,
            "timed_scope": "decode through Study probability mean and flushed image/Study CSV rows",
            "max_concurrent_studies": 1,
            "test_accessed": False, "labels_accessed": False,
        },
        "environment": _runtime_versions(args.runtime_label, args.image_digest),
        "inference_settings": reproducibility,
        "models": results,
        "stage_timing_boundary": (
            "H2D and Tiny/Base forward are CUDA Event durations. Decode/resize/CPU normalize, "
            "postprocess D2H, aggregation, and CSV are wall durations. Stage values are diagnostic "
            "components and are not summed to replace the separately measured Study total. "
            "This script does not insert a CUDA sync between every component."
        ),
        "known_limits": [
            "Current host is an RTX 2080 Ti, not the official T4 timing host; these results cannot certify T4 P95.",
            "Study batching has not been cleared against the registered probability/AUC equivalence gates.",
            "Per-image batch-vs-serial probability deltas and valid AUC deltas are not calculated here.",
            "CSV writing uses a local two-file image/Study output; official runner overhead may differ.",
        ],
        "outputs": {"per_study_timings_csv": str(output / "per_study_timings.csv"),
                    "output_dir": str(output)},
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


class _CSVWriter:
    def __init__(self, stream, header):
        self.stream = stream
        self.writer = csv.writer(stream)
        self.writer.writerow(header)
        stream.flush()

    def writerows(self, rows):
        self.writer.writerows(rows)


class _NullStream:
    def flush(self):
        return None


class _NullWriter:
    """Minimal writer for the five unmeasured warm-up Studies."""
    stream = _NullStream()

    def writerows(self, _rows):
        return None

    def flush(self):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-root", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tiny-model", type=Path, default=DEFAULT_TINY)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--studies-per-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--runtime-label", default="near-official")
    parser.add_argument("--image-digest", default=None)
    args = parser.parse_args()
    if args.studies_per_count < 30:
        parser.error("The timing gate requires at least 30 Studies per image-count stratum")
    report = run(args)
    print(json.dumps({"output_dir": report["outputs"]["output_dir"],
                      "environment": report["environment"],
                      "three_image_p95_ms": {
                          name: result["study_batches"]["3"]["total_ms"]["p95"]
                          for name, result in report["models"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
