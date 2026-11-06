"""Local FP16 resource stress timing; never certifies T4 latency.

The default is a CPU metadata preview. --execute explicitly enables CUDA/model
work. Run each MPS profile in a fresh process with its environment set before
Python starts. No deployment or official timing implementation is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import cost
from deploy_kd import (Predictor, checked_probabilities, deployment_inventory,
                       deployment_members, discover_images, read_json,
                       read_submission, sha256_file, write_json)

MPS_ENVIRONMENT = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                   "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                   "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING",
                   "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "CUDA_VISIBLE_DEVICES")


def validate_inputs(pictures_100: Path, pictures_5: Path, selection_json: Path) -> tuple[list, list, dict]:
    """Prove identity, byte hashes and Study closure against the source manifest."""
    selection = read_json(selection_json)
    if selection.get("schema_version") != 1 or selection.get("origin_split") != "valid":
        raise ValueError("Require a schema1 original-valid selection")
    manifest = Path(selection["manifest_path"]).resolve(strict=True)
    if sha256_file(manifest) != selection.get("manifest_sha256"):
        raise ValueError("Original manifest SHA-256 mismatch")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    originals = {}
    original_ids = set()
    full_counts: Counter = Counter()
    for row in rows:
        subject, study = row["subject_id"], row["study_id"]
        relative = Path(row["relative_path"])
        if (row["split"] != "valid" or row["study_key"] != f"{subject}/{study}"
                or relative.parts[:3] != ("valid", subject, study)
                or len(relative.parts) != 4 or ".." in relative.parts):
            raise ValueError("Malformed original-valid manifest identity")
        key = (subject, study, relative.name)
        if key in originals or not row["image_id"] or row["image_id"] in original_ids:
            raise ValueError("Duplicate or missing source image identity")
        original_ids.add(row["image_id"])
        originals[key] = row
        full_counts[row["study_key"]] += 1
    if not rows or dict(full_counts) != selection.get("full_study_image_counts"):
        raise ValueError("Independent original Study image counts disagree")
    groups, evidence, paths, inodes, patients = [], {}, [], [], []
    for name, directory, expected in (("measured", pictures_100, 100), ("warmup", pictures_5, 5)):
        records = discover_images(directory)
        declarations = selection.get(name)
        if len(records) != expected or not isinstance(declarations, list) or len(declarations) != expected:
            raise ValueError("Require exactly 100 measured and 5 warmup images")
        declared = {}
        for item in declarations:
            relative = Path(item["relative_path"])
            key = (item["subject_id"], item["study_id"], relative.name)
            if key in declared:
                raise ValueError("Duplicate selection image identity")
            declared[key] = item
        selected_counts: Counter = Counter()
        image_ids = set()
        for record in records:
            key = (record["subject_id"], record["study_id"], record["path"].name)
            item, original = declared.get(key), originals.get(key)
            if item is None or original is None:
                raise ValueError("Supplied image is absent from selection/source manifest")
            for field in ("image_id", "subject_id", "study_id", "study_key", "relative_path"):
                if item.get(field) != original[field]:
                    raise ValueError(f"Selection/source identity mismatch: {field}")
            digest = sha256_file(record["path"])
            if not digest or digest != item.get("sha256") or digest != item.get("file_sha256") or digest != original["file_sha256"]:
                raise ValueError("Selected image SHA-256 mismatch")
            if item["image_id"] in image_ids:
                raise ValueError("Duplicate original image ID")
            image_ids.add(item["image_id"])
            selected_counts[record["study_key"]] += 1
        if any(full_counts[key] != count for key, count in selected_counts.items()):
            raise ValueError("Selection omits an original Study view")
        group_paths = {r["path"] for r in records}
        paths.append(group_paths)
        inodes.append({(p.stat().st_dev, p.stat().st_ino) for p in group_paths})
        patients.append({r["subject_id"] for r in records})
        groups.append(records)
        evidence[name] = {"images": len(records), "studies": len(selected_counts),
                          "patients": len(patients[-1]), "complete_studies_proven": True}
    counts = selection.get("counts", {})
    for name, group in evidence.items():
        for field in ("images", "studies", "patients"):
            if counts.get(f"{name}_{field}") != group[field]:
                raise ValueError("Declared selection counts disagree with independent counts")
    if paths[0] & paths[1] or inodes[0] & inodes[1] or patients[0] & patients[1]:
        raise ValueError("Measured/warmup files and patients must be disjoint")
    evidence.update(selection_sha256=sha256_file(selection_json), manifest_sha256=sha256_file(manifest),
                    origin_split="valid", hashes_verified=True, patient_disjoint=True)
    return groups[0], groups[1], evidence


def prepare_cuda(torch: Any, *, requested_thread_percentage: int | None,
                 baseline_sm_count: int, memory_limit_gib: float) -> dict:
    """Verify the fresh client's MPS quota before constructing a model."""
    environment = {key: os.environ.get(key) for key in MPS_ENVIRONMENT}
    requested = requested_thread_percentage
    active = environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]
    if requested is not None and active is not None and active != str(requested):
        raise RuntimeError("Requested MPS thread percentage differs from client environment")
    if requested is not None and requested < 100:
        if not environment["CUDA_MPS_PIPE_DIRECTORY"] or active != str(requested):
            raise RuntimeError("Restricted profile requires matching MPS pipe and active thread percentage")
    if not torch.cuda.is_available():
        raise RuntimeError("Local FP16 stress execution requires CUDA")
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    actual_sm_count = props.multi_processor_count
    if actual_sm_count <= 0:
        raise RuntimeError("Invalid CUDA-visible multiprocessor count")
    if requested is not None and requested < 100 and actual_sm_count >= baseline_sm_count:
        raise RuntimeError("MPS restriction ineffective: CUDA-visible SM count is not below native baseline")
    requested_bytes = int(memory_limit_gib * 1024 ** 3)
    fraction = requested_bytes / props.total_memory
    if not 0 < fraction <= 1:
        raise ValueError("Allocator limit must be positive and no larger than physical device memory")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    print(f"[ALLOCATOR] requested={memory_limit_gib:g} GiB ({requested_bytes} bytes); "
          f"applied_fraction={fraction:.12f}; device_total_bytes={props.total_memory}")
    return {"requested_thread_percentage": requested, "baseline_sm_count": baseline_sm_count,
            "cuda_visible_sm_count": actual_sm_count,
            "native_baseline_sm_count": actual_sm_count if requested in (None, 100) and not active else baseline_sm_count,
            "mode": "mps_client" if environment["CUDA_MPS_PIPE_DIRECTORY"] else "native",
            "environment": environment, "physical_device_name": props.name,
            "allocator": {"requested_limit_gib": memory_limit_gib,
                          "requested_limit_bytes": requested_bytes,
                          "applied_fraction": fraction, "device_total_bytes": props.total_memory,
                          "scope": "PyTorch caching allocator only; no bandwidth or hardware-capacity equivalence"}}


def validate_round_output(records: list, output: Path, results: dict) -> dict:
    expected = {(r["subject_id"], r["study_id"]) for r in records}
    if set(results) != expected:
        raise ValueError("Round result Study coverage differs from inputs")
    for values in results.values():
        checked_probabilities(values)
    submitted = read_submission(output)  # Requires each Study's ten classes exactly once.
    if set(submitted) != expected or not cost.compare_predictions(results, submitted, atol=1e-7)["match"]:
        raise ValueError("Round CSV differs from its complete Study predictions")
    image_output = output.with_suffix(".image_predictions.csv")
    seen: dict[str, set] = {}
    identities = {r["image_id"]: (r["subject_id"], r["study_id"]) for r in records}
    with image_output.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id, class_id = row["Image_id"], int(row["Predict_class"])
            values = seen.setdefault(image_id, set())
            if (identities.get(image_id) != (row["Subject_id"], row["Study_id"])
                    or class_id not in range(1, 11) or class_id in values
                    or not math.isfinite(float(row["Probability"]))
                    or not 0 <= float(row["Probability"]) <= 1):
                raise ValueError("Invalid image CSV identity/class/probability")
            values.add(class_id)
    if set(seen) != set(identities) or any(value != set(range(1, 11)) for value in seen.values()):
        raise ValueError("Image CSV requires every image and all ten classes")
    return {"path": output.name, "sha256": sha256_file(output),
            "image_predictions": image_output.name,
            "image_predictions_sha256": sha256_file(image_output),
            "studies": len(expected), "study_classes": 10, "image_classes": 10}


def run_benchmark(*, model_dir: str | Path, pictures_100: str | Path,
                  pictures_5: str | Path, selection_json: str | Path,
                  output_dir: str | Path, profile: str,
                  requested_thread_percentage: int | None = None,
                  baseline_sm_count: int = 128, memory_limit_gib: float = 16,
                  execute: bool = False) -> dict:
    if (requested_thread_percentage is not None
            and (type(requested_thread_percentage) is not int or not 1 <= requested_thread_percentage <= 100)):
        raise ValueError("Requested thread percentage must be an integer in 1..100")
    if type(baseline_sm_count) is not int or baseline_sm_count <= 0:
        raise ValueError("Native baseline SM count must be positive")
    if not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
        raise ValueError("Memory limit must be positive and finite")
    model_dir, pictures_100, pictures_5, selection_json, output_dir = [
        Path(value).resolve() for value in (model_dir, pictures_100, pictures_5, selection_json, output_dir)]
    if output_dir.exists():
        raise FileExistsError("Choose a new local stress output directory")
    if any(output_dir.is_relative_to(path) for path in (model_dir, pictures_100, pictures_5)):
        raise ValueError("Stress outputs must be outside model and input directories")
    measured, warmup, coverage = validate_inputs(pictures_100, pictures_5, selection_json)
    config = read_json(model_dir / "config.json")
    inventory = deployment_inventory(model_dir, config)
    members = deployment_members(config)
    report = {"profile": profile, "executed": False, "local_stress_test": True,
              "T4_latency_certified": False, "official_compliance_pass": False,
              "T4_elapsed_ms": None, "status": "metadata_preview",
              "protocol": {"measured_images": 100, "warmup_images": 5, "rounds": 3,
                           "precision": "float16", "batch_size": 1, "execution": "serial",
                           "timing": "read_file_to_prediction_CSV_write_including_preprocess_postprocess",
                           "model_initialization_timed": False,
                           "aggregation": "per_image_three_round_mean_then_100_image_mean"},
              "coverage": coverage, "inference_files": inventory,
              "deployment_config_sha256": sha256_file(model_dir / "config.json"),
              "deployment": {"member_count": len(members), "tta": config.get("tta"),
                             "ensemble": config.get("ensemble"),
                             "image_sizes": sorted({m["image_size"] for m in members}),
                             "members": [{field: m.get(field) for field in (
                                 "weights_file", "weights_sha256", "weights_source", "training_mode",
                                 "source_checkpoint_sha256", "source_checkpoint_schema_version",
                                 "seed", "image_size", "tta")} for m in members]},
              "requested_resources": {"thread_percentage": requested_thread_percentage,
                                      "baseline_sm_count": baseline_sm_count,
                                      "memory_limit_gib": memory_limit_gib},
              "note": "Dedicated local stress test. MPS and an allocator cap do not emulate T4 architecture, bandwidth, memory or latency. Actual T4 retest required."}
    if not execute:
        return report
    import torch
    resources = prepare_cuda(torch, requested_thread_percentage=requested_thread_percentage,
                             baseline_sm_count=baseline_sm_count, memory_limit_gib=memory_limit_gib)
    predictor = Predictor(model_dir)  # Shared FP16 predictor; initialization excluded from timing.
    report["cuda_graph_execution_plan"] = predictor.cuda_graph_metadata
    report["protocol"]["extra_synthetic_graph_setup"] = "explicitly_recorded_not_a_substitute_for5_real_warmup_images"
    report["protocol"]["official_initialization_warmup_interpretation"] = "pending_review"
    environment = cost.timing_environment(predictor)
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in warmup:
        predictor.predict_image(record["path"])
    predictor.synchronize()
    rounds, comparisons, outputs, first = [], [], [], None
    for index in range(3):
        output = output_dir / f"round_{index + 1}.csv"
        durations, results = cost.time_round(predictor, measured, output)
        rounds.append(durations)
        outputs.append(validate_round_output(measured, output, results))
        if first is None:
            first = {key: np.asarray(values).copy() for key, values in results.items()}
        comparisons.append(cost.compare_predictions(results, first, atol=1e-7))
    times, per_image, mean = cost.summarize_timings(rounds)
    reproducible = all(item["match"] for item in comparisons)
    report.update(executed=True, status="local_stress_completed" if reproducible else "local_stress_prediction_mismatch",
                  environment=environment, resources=resources,
                  per_image_round_ms=times.T.tolist(), per_image_mean_ms=per_image.tolist(),
                  image_order=[str(r["path"].relative_to(pictures_100)) for r in measured],
                  mean_ms=mean, median_ms=float(np.median(per_image)),
                  p95_ms=float(np.percentile(per_image, 95)), max_ms=float(per_image.max()),
                  round_mean_ms=times.mean(axis=1).tolist(),
                  latency_limit_ms=100.0, latency_pass_on_current_hardware=mean <= 100.0,
                  comparison={"status": "matched" if reproducible else "failed", "rounds": comparisons},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(0), round_outputs=outputs)
    write_json(output_dir / "local_resource_report.json", report)
    print(f"[LOCAL STRESS] {profile}: {mean:.3f} ms/image; "
          f"visible SMs={resources['cuda_visible_sm_count']}; T4 certified=False")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "pictures-100", "pictures-5", "selection-json", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--requested-thread-percentage", type=int)
    parser.add_argument("--baseline-sm-count", type=int, default=128)
    parser.add_argument("--memory-limit-gib", type=float, default=16)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable CUDA/model work")
    report = run_benchmark(**vars(parser.parse_args(argv)))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if not report["executed"] or report["comparison"]["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
