"""Optional instrumented diagnostic: one round, never the primary benchmark.

This wrapper preserves the frozen production inference and writer. CUDA events
span root-module forwards and may include host submission gaps. They are not
pure kernel busy time. Run only after the uninstrumented three-round benchmark.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Diagnostic requires a new output directory")
    if output.is_relative_to(bundle):
        raise ValueError("Diagnostic output must be outside the frozen bundle")
    output.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    wall_started = time.perf_counter()
    entries = []
    handles = []
    report = {
        "diagnostic_only": True,
        "instrumented": True,
        "replaces_primary_benchmark": False,
        "T4_latency_certified": False,
        "official_compliance_pass": False,
        "protocol": {"warmup_images": 5, "measured_images": 100, "rounds": 1,
                     "batch_size": 1, "seed": 43, "allocator_limit_gib": 8},
        "cuda_event_interpretation": "Root forward CUDA event spans may include host submission gaps; not pure kernel busy time.",
        "started_utc": started,
    }
    exit_code = 1
    try:
        if any(key.startswith("CUDA_MPS_") for key in os.environ):
            raise RuntimeError("Unexpected CUDA_MPS environment")
        assert sys.version_info[:2] == (3, 12), sys.version
        import numpy as np
        import torch

        assert torch.__version__ == "2.10.0+cu128", torch.__version__
        assert torch.version.cuda == "12.8", torch.version.cuda
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        random.seed(43)
        np.random.seed(43)
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
        sys.path.insert(0, str(bundle))
        import benchmark_local_resources as benchmark
        import cost
        import deploy_kd

        measured, warmup, coverage = benchmark.validate_inputs(
            bundle / "inputs/pictures_100", bundle / "inputs/pictures_5",
            bundle / "inputs/selection.portable.json")
        config = deploy_kd.read_json(bundle / "model/033/config.json")
        assert deploy_kd.sha256_file(bundle / "model/033/config.json") == "9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590"
        members = deploy_kd.deployment_members(config)
        assert len(members) == 2
        assert [member["seed"] for member in members] == [43, 44]
        assert all(member["image_size"] == 768 and member["tta"] == "hflip2" for member in members)
        assert config["model_weights"] == [[0.25] * 10, [0.75] * 10]
        actual_sm = torch.cuda.get_device_properties(0).multi_processor_count
        report["resources"] = benchmark.prepare_cuda(
            torch, requested_thread_percentage=None,
            baseline_sm_count=actual_sm, memory_limit_gib=8)
        predictor = deploy_kd.Predictor(bundle / "model/033")
        report["environment"] = cost.timing_environment(predictor)
        report["coverage"] = coverage
        report["inventory"] = predictor.inventory
        for record in warmup:
            predictor.predict_image(record["path"])
        predictor.synchronize()

        # Install instrumentation only after the unchanged five-image warmup.
        current = None
        original_load = deploy_kd.load_image_arrays
        original_predict = predictor.predict_image

        def timed_load(*load_args, **load_kwargs):
            before = time.perf_counter()
            try:
                return original_load(*load_args, **load_kwargs)
            finally:
                if current is not None:
                    current["cpu_preprocess_ms"] += (time.perf_counter() - before) * 1000
                    current["load_calls"] += 1

        deploy_kd.load_image_arrays = timed_load

        def pre_hook(member_index):
            def hook(module, inputs):
                if current is None:
                    raise RuntimeError("Forward outside active measured image")
                index = len(current["events"])
                assert index < 4 and member_index == index // 2
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                current["events"].append((member_index, start_event, end_event))
                start_event.record()
            return hook

        def post_hook(member_index):
            def hook(module, inputs, result):
                assert current is not None
                item = current["events"][-1]
                assert item[0] == member_index
                item[2].record()
            return hook

        for index, model in enumerate(predictor.models):
            handles.append(model.register_forward_pre_hook(pre_hook(index)))
            handles.append(model.register_forward_hook(post_hook(index)))

        def timed_predict(path):
            nonlocal current
            entry = {"path": str(path), "cpu_preprocess_ms": 0.0,
                     "load_calls": 0, "events": [], "prediction_completed": False}
            entries.append(entry)
            current = entry
            try:
                result = original_predict(path)
                entry["prediction_completed"] = True
                assert entry["load_calls"] == 1 and len(entry["events"]) == 4
                return result
            except BaseException:
                entry["exception"] = traceback.format_exc()
                raise
            finally:
                current = None

        predictor.predict_image = timed_predict
        try:
            durations, results = cost.time_round(predictor, measured, output / "round_1.csv")
        finally:
            predictor.predict_image = original_predict
            deploy_kd.load_image_arrays = original_load
            for handle in handles:
                handle.remove()
            handles.clear()
        predictor.synchronize()
        report["round_output"] = benchmark.validate_round_output(
            measured, output / "round_1.csv", results)
        assert len(durations) == len(entries) == 100
        for entry, record, duration in zip(entries, measured, durations):
            assert Path(entry["path"]) == record["path"]
            entry.update(image_id=record["image_id"], subject_id=record["subject_id"],
                         study_id=record["study_id"], e2e_ms=duration)
            spans = [start.elapsed_time(end) for _, start, end in entry["events"]]
            assert len(spans) == 4 and all(math.isfinite(v) and v >= 0 for v in spans)
            entry["forward_event_spans_ms"] = spans
            entry["forward_event_sum_ms"] = sum(spans)
        report.update(status="completed", measured_images_completed=100,
                      e2e_mean_ms=statistics.mean(durations), e2e_max_ms=max(durations),
                      cpu_preprocess_mean_ms=statistics.mean(e["cpu_preprocess_ms"] for e in entries),
                      forward_event_sum_mean_ms=statistics.mean(e["forward_event_sum_ms"] for e in entries),
                      per_forward_event_mean_ms=[statistics.mean(e["forward_event_spans_ms"][i] for e in entries) for i in range(4)],
                      e2e_p95_ms=float(np.percentile(durations, 95)),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        exit_code = 0
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        print(report["exception"], file=sys.stderr)
    finally:
        for entry in entries:
            entry.pop("events", None)
        report["entries_recorded"] = len(entries)
        report["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter() - wall_started
        report["exit_code"] = exit_code
        write_json(output / "diagnostic_report.json", report)
        write_json(output / "per_image_segments.json", {"images": entries})
        (output / "exit_code.txt").write_text(str(exit_code) + "\n", encoding="utf-8")
        if entries and all("e2e_ms" in entry for entry in entries):
            with (output / "per_image_segments.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["image_id", "subject_id", "study_id", "path", "cpu_preprocess_ms",
                          "forward_0_identity_event_ms", "forward_0_hflip_event_ms",
                          "forward_1_identity_event_ms", "forward_1_hflip_event_ms",
                          "forward_event_sum_ms", "e2e_ms"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for entry in entries:
                    row = {k: entry[k] for k in fields if k in entry}
                    row.update(zip(fields[5:9], entry["forward_event_spans_ms"]))
                    writer.writerow(row)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
