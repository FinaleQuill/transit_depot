"""Official 100-image, batch=1, 5-warmup, three-round end-to-end timing.

Usage: python cost.py submission.csv pictures_100 pictures_5 --model-dir MODEL
       --manifest FULL_ORIGINAL_TEST_MANIFEST --output-dir NEW_REPORT_DIRECTORY

Submission is Study-level. Certification requires the original manifest proving
complete selected Studies and the saved predict.py image-probability sidecar for
all 100 image comparisons. Otherwise timing still runs, comparison remains
pending, and the command exits 2 rather than claiming compliance. The sidecar
defaults to submission.image_predictions.csv beside submission.csv; override its
location with --image-submission.
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, Predictor, StudyAccumulator, discover_images,
                       checked_probabilities, match_timing_manifest, read_manifest, read_submission, sha256_file,
                       write_json, write_prediction, write_image_prediction, deployment_members)
from inference_reproducibility import compare_probability_outputs, configure_inference


def time_round(predictor: Any, records: list[dict], output: Path, *,
               clock: Callable[[], float] = time.perf_counter) -> tuple[list[float], dict]:
    """The identical read/model/Study aggregation/CSV writer used by predict.py."""
    accumulator = StudyAccumulator(records)
    durations = []
    results = {}
    gates = predictor.config.get("gates")
    image_output = output.with_suffix(".image_predictions.csv")
    with output.open("x", encoding="utf-8", newline="") as stream, \
            image_output.open("x", encoding="utf-8", newline="") as image_stream:
        writer = csv.writer(stream)
        image_writer = csv.writer(image_stream)
        writer.writerow(CSV_HEADER + (["Gate"] if gates is not None else []))
        image_writer.writerow(IMAGE_CSV_HEADER)
        stream.flush()
        image_stream.flush()
        for record in records:
            predictor.synchronize()
            started = clock()
            probability = predictor.predict_image(record["path"])
            write_image_prediction(image_writer, record, probability)
            image_stream.flush()
            result = accumulator.add(record, probability)
            if result is not None:
                subject, study, mean = result
                write_prediction(writer, subject, study, mean, gates=gates)
                stream.flush()
                results[(subject, study)] = mean.copy()
            # Includes GPU completion, transfer, aggregation and the actual CSV write.
            predictor.synchronize()
            durations.append((clock() - started) * 1000)
    if not accumulator.complete():
        raise RuntimeError("Incomplete measured-image coverage")
    return durations, results


def compare_predictions(results: dict, submitted: dict, *, atol: float = 1e-7,
                        gates=None, expected_keys=None,
                        allow_reference_superset: bool = False) -> dict:
    return compare_probability_outputs(
        results, submitted, atol=atol, gates=gates, expected_keys=expected_keys,
        allow_reference_superset=allow_reference_superset)


def read_image_probabilities(path: str | Path) -> dict:
    """Read the existing predict.py sidecar, validating every image's ten classes."""
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(IMAGE_CSV_HEADER).issubset(reader.fieldnames or []):
            raise ValueError("Image reference must use predict.py's image-probability sidecar format")
        for row in reader:
            key = (row["Subject_id"], row["Study_id"], row["Image_id"])
            if any(not value for value in key):
                raise ValueError("Missing image reference identity")
            class_id = int(row["Predict_class"])
            values = rows.setdefault(key, {})
            if class_id not in range(1, 11) or class_id in values:
                raise ValueError("Duplicate or invalid image reference class")
            values[class_id] = float(row["Probability"])
    if not rows or any(set(values) != set(range(1, 11)) for values in rows.values()):
        raise ValueError("Each image reference must contain all ten classes")
    return {key: checked_probabilities([values[c] for c in range(1, 11)]) for key, values in rows.items()}


def source_image_ids(records: list[dict], manifest: str | Path | None) -> dict:
    """Map original identities, without reading any image outside the timing set."""
    if manifest is None:
        return {(r["subject_id"], r["study_id"], r["path"].name): r["image_id"] for r in records}
    identities = {}
    for row in read_manifest(manifest):
        key = (row["subject_id"], row["study_id"], Path(row["relative_path"]).name)
        if key in identities:
            raise ValueError("Original manifest has duplicate file identities")
        identities[key] = row["image_id"]
    return identities


def compare_image_output(records: list[dict], output: Path, reference: dict,
                         original_ids: dict, *, gates=None) -> dict:
    """Compare each timed image to its saved final-inference output, outside timing."""
    actual_rows = read_image_probabilities(output)
    actual, expected, expected_keys, consumed = {}, {}, set(), set()
    for record in records:
        subject, study, filename = record["subject_id"], record["study_id"], record["path"].name
        key = (subject, f"{study}/{filename}")
        expected_keys.add(key)
        actual_key = (subject, study, record["image_id"])
        if actual_key in actual_rows:
            actual[key] = actual_rows[actual_key]
            consumed.add(actual_key)
        canonical = original_ids.get((subject, study, filename))
        candidates = {(subject, study, image_id) for image_id in (canonical, record["image_id"])
                      if image_id is not None and (subject, study, image_id) in reference}
        if len(candidates) > 1:
            raise ValueError("Ambiguous final image sidecar identity")
        if candidates:
            expected[key] = reference[next(iter(candidates))]
    # Do not silently discard unexpected image rows produced by the timing run.
    for subject, study, image_id in set(actual_rows) - consumed:
        actual[(subject, f"unexpected:{study}/{image_id}")] = actual_rows[(subject, study, image_id)]
    report = compare_predictions(actual, expected, gates=gates, expected_keys=expected_keys)
    report.update(level="image", compared_images=report["compared_studies"],
                  expected_images=len(records), mismatched_image_count=report["mismatched_study_count"],
                  reference_mapping="official manifest image_id, with original discovery-id compatibility")
    return report


def timing_environment(predictor: Any) -> dict:
    torch = predictor.torch
    props = torch.cuda.get_device_properties(predictor.device)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10).splitlines()[0].strip()
        driver_compatible = tuple(int(p) for p in driver.split(".")[:2]) >= (570, 117)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        driver, driver_compatible = None, False
    environment_compatible = (
        "T4" in props.name and platform.python_version_tuple()[:2] == ("3", "12")
        and torch.__version__.split("+")[0] == "2.10.0"
        and torch.version.cuda == "12.8"
        and 90_000 <= (torch.backends.cudnn.version() or 0) < 100_000
        and driver_compatible)
    import fast_lanczos

    preprocessing = {"grayscale_resize": fast_lanczos.implementation(),
                     "native_available": fast_lanczos.available(),
                     "native_threads_max": fast_lanczos.default_threads(),
                     "uint8_lookup_input": hasattr(predictor, "input_lookup"),
                     "serial_images": True}
    return {"gpu_name": props.name, "gpu_memory_bytes": props.total_memory,
            "gpu_compute_capability": [props.major, props.minor],
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "precision": "float16", "device": str(predictor.device),
            "driver": driver, "official_gpu": "T4" in props.name,
            "official_environment_compatible": environment_compatible,
            "preprocessing": preprocessing}


def summarize_timings(rounds: list[list[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.asarray(rounds, dtype=np.float64)
    if times.shape != (3, 100) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Expected three complete rounds of 100 finite nonnegative timings")
    per_image_ms = times.mean(axis=0)
    return times, per_image_ms, float(per_image_ms.mean())


def run_cost(*, submission: str | Path, pictures_100: str | Path, pictures_5: str | Path,
             model_dir: str | Path, output_dir: str | Path,
             manifest: str | Path | None = None,
             image_submission: str | Path | None = None,
             predictor_factory=None) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new timing output directory")
    for image_dir in (pictures_100, pictures_5):
        if output_dir.is_relative_to(Path(image_dir).resolve()):
            raise ValueError("Timing outputs must not be inside image directories")
    measured, warmup = discover_images(pictures_100), discover_images(pictures_5)
    if len(measured) != 100 or len(warmup) != 5:
        raise ValueError(f"Official protocol requires exactly 100 measured and 5 warmup images; got {len(measured)}/{len(warmup)}")
    coverage = match_timing_manifest(measured, manifest)
    submitted = read_submission(submission)
    image_reference_path = (Path(image_submission) if image_submission is not None
                            else Path(submission).with_suffix(".image_predictions.csv"))
    image_reference = read_image_probabilities(image_reference_path) if image_reference_path.is_file() else None
    original_ids = source_image_ids(measured, manifest)
    reproducibility = configure_inference()
    predictor = (predictor_factory or Predictor)(model_dir)  # Initialization is not timed.
    members = getattr(predictor, 'members', None)
    if members is None:
        members = deployment_members(predictor.config)
    environment = timing_environment(predictor)
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in warmup:
        predictor.predict_image(record["path"])
    predictor.synchronize()
    rounds, comparisons, image_comparisons, outputs = [], [], [], []
    for index in range(3):
        output = output_dir / f"round_{index + 1}.csv"
        durations, result = time_round(predictor, measured, output)
        rounds.append(durations)
        comparisons.append(compare_predictions(
            result, submitted, gates=predictor.config.get("gates"),
            expected_keys={(record["subject_id"], record["study_id"]) for record in measured},
            allow_reference_superset=True))
        image_output = output.with_suffix(".image_predictions.csv")
        image_comparisons.append(
            compare_image_output(measured, image_output, image_reference, original_ids,
                                 gates=predictor.config.get("gates"))
            if image_reference is not None else {
                "match": False, "level": "image", "expected_images": len(measured),
                "status": "pending_missing_final_image_sidecar",
                "reason": "Final predict.py image sidecar is required to verify each timed image independently"})
        outputs.append({"path": output.name, "sha256": sha256_file(output),
                        "image_predictions": image_output.name,
                        "image_predictions_sha256": sha256_file(image_output)})
    times, per_image_ms, mean_ms = summarize_timings(rounds)
    comparison_proven = (coverage["complete_studies_proven"]
                         and all(c["match"] for c in comparisons)
                         and all(c["match"] for c in image_comparisons))
    latency_pass = mean_ms <= 100.0
    official_pass = (environment["official_environment_compatible"] and predictor.inventory["size_pass"]
                     and comparison_proven and latency_pass)
    report = {"protocol": {"measured_images": 100, "warmup_images": 5, "rounds": 3,
                            "batch_size": 1, "execution": "serial",
                            "timing": "read_file_to_prediction_CSV_write_including_preprocess_postprocess",
                            "aggregation": "per_image_three_round_mean_then_100_image_mean"},
              "environment": environment, "reproducibility": reproducibility,
              "inference_files": predictor.inventory,
              "deployment_config_sha256": sha256_file(Path(model_dir) / "config.json"),
              "deployment": {"member_count": len(members),
                             "image_sizes": sorted({m["image_size"] for m in members}),
                             "tta": predictor.config["tta"], "ensemble": predictor.config["ensemble"],
                             "aggregation": "fixed_weighted_mean_probability_then_study_mean"},
              "per_image_round_ms": times.T.tolist(), "per_image_mean_ms": per_image_ms.tolist(),
              "image_order": [str(r["path"].relative_to(Path(pictures_100).resolve())) for r in measured],
              "mean_ms": mean_ms, "latency_limit_ms": 100.0,
              "latency_pass_on_current_hardware": latency_pass,
              "comparison": {"status": "matched" if comparison_proven else "pending_or_failed",
                             "coverage": coverage, "rounds": comparisons,
                             "image_rounds": image_comparisons,
                             "image_submission": str(image_reference_path),
                             "image_submission_sha256": sha256_file(image_reference_path) if image_reference is not None else None},
              "official_compliance_pass": official_pass,
              "status": "local_T4_checks_passed_official_retest_required" if official_pass else "not_certified",
              "submission_sha256": sha256_file(submission), "round_outputs": outputs,
              "note": "Final compliance is determined by the organizer's reproduction. Non-T4 timing is diagnostic only."}
    write_json(output_dir / "cost_report.json", report)
    print(f"[CONCLUSION] 100 images, batch=1 serial, 3 rounds: {mean_ms:.3f} ms; "
          f"FP16; CSV agreement={comparison_proven}; GPU={environment['gpu_name']}; "
          f"local T4 checks passed={official_pass}; official retest required.")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-submission", type=Path,
                        help="Final predict.py image sidecar; defaults beside submission.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from submission_runtime import load_submission_predictor
    report = run_cost(**vars(args), predictor_factory=load_submission_predictor)
    return 0 if report["official_compliance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
