"""Independent current-source diagnostic, never a replacement for three-round E2E.

No project source is modified. CPU staged decoding uses bytes in memory after an
explicit file read, so its stages describe that diagnostic path rather than an
exclusive breakdown of Pillow's normal lazy file-reading path. GPU forward CUDA
Event spans can contain host launch gaps. A separate five-image profiler capture
reports leaf GPU kernel activities and is not an E2E timing result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time
import traceback


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


def summary(values):
    import numpy as np
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": float(np.percentile(values, 95)), "max": max(values)}


def cpu_stages(path, deploy):
    import numpy as np
    from PIL import Image, ImageOps

    stages = {}
    total_start = time.perf_counter()
    start = time.perf_counter()
    raw = path.read_bytes()
    stages["raw_file_read_ms"] = elapsed_ms(start)
    start = time.perf_counter()
    source = Image.open(io.BytesIO(raw))
    stages["memory_bytes_header_open_ms"] = elapsed_ms(start)
    try:
        start = time.perf_counter()
        source.load()
        stages["memory_bytes_decode_ms"] = elapsed_ms(start)
        original_mode, original_size = source.mode, list(source.size)
        start = time.perf_counter()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {"I", "F"} or source.mode.startswith("I;"):
            raise ValueError("Non-8-bit input requires an explicit intensity policy")
        image = source if source.mode == "L" else source.convert("RGB")
        stages["exif_and_color_ms"] = elapsed_ms(start)
        grayscale = image.mode == "L"
        start = time.perf_counter()
        scale = 768 / max(image.size)
        size = tuple(max(1, min(768, round(length * scale))) for length in image.size)
        resized = image.resize(size, Image.Resampling.LANCZOS)
        stages["lanczos_resize_ms"] = elapsed_ms(start)
        start = time.perf_counter()
        canvas = Image.new("L" if grayscale else "RGB", (768, 768), 0)
        canvas.paste(resized, ((768 - size[0]) // 2, (768 - size[1]) // 2))
        stages["letterbox_pad_ms"] = elapsed_ms(start)
        start = time.perf_counter()
        pixels = np.asarray(canvas)
        stages["numpy_uint8_view_ms"] = elapsed_ms(start)
        start = time.perf_counter()
        actual = np.empty((3, 768, 768), dtype=np.float32)
        offset = np.asarray(deploy.MEAN, dtype=np.float32) * np.float32(255.0)
        denominator = np.asarray(deploy.STD, dtype=np.float32) * np.float32(255.0)
        for channel in range(3):
            np.copyto(actual[channel], pixels if grayscale else pixels[:, :, channel])
            np.subtract(actual[channel], offset[channel], out=actual[channel])
            np.divide(actual[channel], denominator[channel], out=actual[channel])
        stages["float32_chw_normalize_ms"] = elapsed_ms(start)
        stages["staged_path_total_ms"] = elapsed_ms(total_start)
    finally:
        source.close()
    # Validation is deliberately outside all reported staged intervals.
    start = time.perf_counter()
    expected = deploy.load_image_arrays(path, [768])[768]
    reference_load_ms = elapsed_ms(start)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise AssertionError("Staged CPU path is not byte-identical to current source")
    return {"path": str(path), "bytes": len(raw), "source_mode": original_mode,
            "source_size": original_size, "post_exif_size": list(image.size),
            "byte_parity": True, "reference_current_file_path_load_ms": reference_load_ms,
            **stages}


def kernel_summary(trace):
    """Count only leaf device kernel X events, never aggregated CPU operators."""
    events = trace.get("traceEvents", [])
    kernels = [event for event in events
               if event.get("ph") == "X"
               and "kernel" in str(event.get("cat", "")).split(",")
               and isinstance(event.get("dur"), (float, int))]
    if not kernels:
        return {"status": "no_cuda_kernel_activities", "kernel_count": 0,
                "trace_categories": sorted({str(event.get("cat", "")) for event in events})[:50]}
    totals, counts = defaultdict(float), Counter()
    intervals = []
    for event in kernels:
        totals[event["name"]] += event["dur"]
        counts[event["name"]] += 1
        intervals.append((event["ts"], event["ts"] + event["dur"]))
    intervals.sort()
    start, end = intervals[0]
    busy = 0.0
    for left, right in intervals[1:]:
        if left > end:
            busy += end - start
            start, end = left, right
        else:
            end = max(end, right)
    busy += end - start
    total_us = sum(totals.values())
    return {"status": "completed", "profiled_images": 5, "forward_calls": 20,
            "kernel_count": len(kernels), "leaf_kernel_duration_sum_ms": total_us / 1000,
            "leaf_kernel_sum_per_image_ms": total_us / 1000 / 5,
            "device_kernel_busy_union_ms": busy / 1000,
            "device_kernel_busy_union_per_image_ms": busy / 1000 / 5,
            "first_kernel_to_last_kernel_ms": (max(right for _, right in intervals) - intervals[0][0]) / 1000,
            "excludes": ["H2D", "D2H", "TTA construction", "sigmoid", "TTA/member averaging", "CSV"],
            "top_kernels": [{"name": name, "calls": counts[name], "total_ms": duration / 1000}
                            for name, duration in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:20]]}


def profile_forwards(predictor, records, deploy):
    torch = predictor.torch
    temporary = Path(tempfile.mkdtemp(prefix="dual768_forward_profile_"))
    trace_path = temporary / "forward_trace.json"
    result = {"diagnostic_only": True, "trace_path": str(trace_path),
              "scope": "first five fixed measured images; resident FP16 tensors; four serial batch1 forwards per image; no postprocessing"}
    try:
        resident = []
        with torch.inference_mode():
            for record in records[:5]:
                array = deploy.load_image_arrays(record["path"], [768])[768]
                image = torch.from_numpy(array).unsqueeze(0).to(
                    predictor.device, dtype=torch.float16, memory_format=torch.channels_last)
                flipped = torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)
                resident.append((image, flipped))
            predictor.synchronize()  # One setup boundary, outside the profiled loop.
            outputs = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False,
                                        with_stack=False) as profiler:
                for index, views in enumerate(resident):
                    for member_index, model in enumerate(predictor.models):
                        for view_index, image in enumerate(views):
                            with torch.profiler.record_function(
                                    f"diagnostic_forward_image{index}_member{member_index}_view{view_index}"):
                                outputs.append(model(image))
            # Profiler collection has completed; validation does not add operators to the trace.
            if any(tuple(logits.shape) != (1, 10) for logits in outputs):
                raise AssertionError("Unexpected profiled logits shape")
            if not bool(torch.stack([torch.isfinite(logits).all() for logits in outputs]).all()):
                raise FloatingPointError("Nonfinite profiled forward logits")
            profiler.export_chrome_trace(str(trace_path))
        result.update(kernel_summary(json.loads(trace_path.read_text())))
        result["trace_bytes"] = trace_path.stat().st_size
    except Exception:
        result.update(status="unavailable_or_failed", exception=traceback.format_exc())
    return result


def copy_and_sync_summary(trace):
    """Summarize device memcpy activities and CUDA API ranges.

    Each CUDA runtime/driver API event is counted once. Runtime and driver totals
    are kept separate because a runtime call can contain a driver call. Their
    sums therefore must not be added together as exclusive CPU elapsed time.
    """
    events = trace.get("traceEvents", [])
    activities = [event for event in events if event.get("ph") == "X"
                  and isinstance(event.get("dur"), (int, float))]
    prediction_ranges = [(event["ts"], event["ts"] + event["dur"], event.get("tid"))
                         for event in activities
                         if str(event.get("name", "")).startswith("diagnostic_predictor_image")]

    def inside_prediction_range(event):
        if not prediction_ranges:
            return True
        left = event.get("ts")
        if not isinstance(left, (int, float)):
            return False
        right = left + event["dur"]
        return any(start <= left and right <= end and thread == event.get("tid")
                   for start, end, thread in prediction_ranges)

    prediction_apis = [event for event in activities if inside_prediction_range(event)]
    groups = {}
    unclassified = Counter()
    for event in activities:
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        if "gpu_memcpy" not in category.split(","):
            continue
        compact_name = "".join(character.lower() for character in name if character.isalnum())
        if "htod" in compact_name or "hosttodevice" in compact_name:
            direction = "HtoD"
        elif "dtoh" in compact_name or "devicetohost" in compact_name:
            direction = "DtoH"
        elif "dtod" in compact_name or "devicetodevice" in compact_name:
            direction = "DtoD"
        else:
            direction = "other"
            unclassified[name] += 1
        item = groups.setdefault(direction, {"activities": 0, "total_ms": 0.0,
                                             "known_bytes_sum": 0,
                                             "activities_with_known_bytes": 0,
                                             "activity_names": Counter()})
        item["activities"] += 1
        item["total_ms"] += event["dur"] / 1000
        item["activity_names"][name] += 1
        byte_count = None
        for key, value in event.get("args", {}).items():
            normalized = "".join(character.lower() for character in str(key) if character.isalnum())
            if normalized in {"bytes", "sizebytes", "numbytes", "bytecount"}:
                try:
                    byte_count = int(value)
                except (ValueError, TypeError):
                    pass
                if byte_count is not None:
                    break
        if byte_count is not None:
            item["known_bytes_sum"] += byte_count
            item["activities_with_known_bytes"] += 1
    for item in groups.values():
        item["activity_names"] = dict(item["activity_names"])
        item["activities_without_byte_metadata"] = item["activities"] - item["activities_with_known_bytes"]
        item["bytes_per_profiled_image_if_complete"] = (
            item["known_bytes_sum"] / 5 if not item["activities_without_byte_metadata"] else None)
    api = {}
    for api_category in ("cuda_runtime", "cuda_driver"):
        totals, counts = defaultdict(float), Counter()
        for event in prediction_apis:
            if api_category not in str(event.get("cat", "")).split(","):
                continue
            name = str(event.get("name", ""))
            if "synchronize" in name.lower():
                totals[name] += event["dur"] / 1000
                counts[name] += 1
        api[api_category] = {
            "call_count": sum(counts.values()), "inclusive_cpu_api_total_ms": sum(totals.values()),
            "top_calls": [{"name": name, "calls": counts[name], "inclusive_cpu_ms": duration}
                          for name, duration in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:15]]}
    copy_api_totals, copy_api_counts = defaultdict(float), Counter()
    for event in prediction_apis:
        if "cuda_runtime" not in str(event.get("cat", "")).split(","):
            continue
        name = str(event.get("name", ""))
        if "memcpy" in name.lower():
            copy_api_totals[name] += event["dur"] / 1000
            copy_api_counts[name] += 1
    return {"device_memcpy": groups, "unclassified_memcpy_names": dict(unclassified),
            "cpu_cuda_synchronization_apis": api,
            "cpu_cuda_api_scope": "only calls contained in same-thread diagnostic_predictor_image ranges; profiler setup/teardown excluded" if prediction_ranges else "entire capture; no diagnostic_predictor_image annotation ranges found",
            "diagnostic_predictor_image_annotation_count": len(prediction_ranges),
            "cpu_cuda_runtime_copy_apis": [
                {"name": name, "calls": copy_api_counts[name], "inclusive_cpu_ms": duration}
                for name, duration in sorted(copy_api_totals.items(), key=lambda pair: pair[1], reverse=True)[:15]],
            "api_time_interpretation": "Inclusive CPU CUDA API durations under profiler; runtime/driver may nest and are not added. Memcpy activity durations are device timeline durations, not CPU wait. Byte totals count only events exposing byte metadata."}


def profile_predictor_path(predictor, records, deploy, reference_probabilities):
    """Profile five original predictor calls with preprocessing outside capture."""
    import numpy as np

    torch = predictor.torch
    temporary = Path(tempfile.mkdtemp(prefix="dual768_predictor_profile_"))
    trace_path = temporary / "predictor_trace.json"
    result = {"diagnostic_only": True, "trace_path": str(trace_path),
              "profiled_images": 5, "expected_forward_calls": 20,
              "scope": "first five fixed measured images; original predict_image including input conversion/H2D, four serial forwards, finite check, D2H and member fusion; CPU image preprocessing precomputed outside capture"}
    original_loader = deploy.load_image_arrays
    try:
        selected = records[:5]
        arrays = {record["path"]: original_loader(record["path"], [768]) for record in selected}

        def precomputed_array_loader(path, sizes):
            assert set(sizes) == {768}
            return arrays[Path(path)]

        predictor.synchronize()  # Setup boundary before the profiler and its loop.
        probabilities = []
        deploy.load_image_arrays = precomputed_array_loader
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False,
                                        with_stack=False) as profiler:
                for index, record in enumerate(selected):
                    with torch.profiler.record_function(f"diagnostic_predictor_image{index}"):
                        probabilities.append(predictor.predict_image(record["path"]))
        finally:
            deploy.load_image_arrays = original_loader
        for record, actual in zip(selected, probabilities):
            expected = reference_probabilities[record["image_id"]]
            if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise AssertionError("Profiled original predictor differs from earlier GPU-path prediction")
        result["prediction_byte_parity_against_gpu_path"] = True
        profiler.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        result.update(copy_and_sync_summary(trace))
        # Unlike profile_forwards, this includes all predictor GPU operations.
        all_kernels = kernel_summary(trace)
        all_kernels.pop("excludes", None)
        all_kernels["scope"] = "all original predictor GPU kernels, including conversion/TTA/finite/probability kernels"
        result["all_predictor_kernel_activities"] = all_kernels
        result.update(status="completed", trace_bytes=trace_path.stat().st_size)
    except Exception:
        result.update(status="unavailable_or_failed", exception=traceback.format_exc())
    finally:
        deploy.load_image_arrays = original_loader
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "model-dir", "inputs-root", "selection-json", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError("Use a new diagnostic output directory")
    args.output_dir.mkdir(parents=True)
    report = {"diagnostic_only": True, "replaces_primary_benchmark": False,
              "T4_latency_certified": False, "source_root": str(args.source_root),
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "argv": sys.argv, "training": False, "production_source_changed": False,
              "seed": 43, "allocator_gib": 8, "warmup_images": 5,
              "measured_images": 100,
              "cpu_interpretation": "Each staged file read precedes BytesIO decoding. This is not an exclusive decomposition of the normal lazy file path. The subsequent reference file-path load is cache-warmed and separate.",
              "gpu_event_interpretation": "CUDA Event spans include possible host submission gaps; they are not pure kernel busy time. CPU preprocessing is outside GPU-path wall timing. Original Predictor.predict_image handles H2D, all forwards, original finite check, D2H and fusion."}
    code = 1
    handles = []
    started = time.perf_counter()
    try:
        if any(key.startswith("CUDA_MPS_") for key in os.environ):
            raise RuntimeError("Unexpected MPS environment")
        assert sys.version_info[:2] == (3, 12), sys.version
        sys.path.insert(0, str(args.source_root))
        import numpy as np
        import torch
        from PIL import __version__ as pillow_version
        import benchmark_local_resources as benchmark
        import cost
        import deploy_kd as deploy

        assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
        assert torch.version.cuda == "12.8", torch.version.cuda
        random.seed(43)
        np.random.seed(43)
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
        measured, warmup, coverage = benchmark.validate_inputs(
            args.inputs_root / "pictures_100", args.inputs_root / "pictures_5", args.selection_json)
        report["coverage"] = coverage
        config = deploy.read_json(args.model_dir / "config.json")
        members = deploy.deployment_members(config)
        assert len(members) == 2 and [member["seed"] for member in members] == [43, 44]
        assert all(member["image_size"] == 768 and member["tta"] == "hflip2" for member in members)
        assert config["model_weights"] == [[0.25] * 10, [0.75] * 10]
        report["source_sha256"] = {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest()
                                   for name in ("deploy_kd.py", "model_overnight.py", "cost.py", "benchmark_local_resources.py")}
        report["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "Pillow": pillow_version, "torch": torch.__version__}
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        cpu_rows = []
        for record in measured:
            row = cpu_stages(record["path"], deploy)
            row["image_id"] = record["image_id"]
            cpu_rows.append(row)
        write_json(args.output_dir / "cpu_per_image.json", {"images": cpu_rows})
        report["cpu_stage_summary_ms"] = {key: summary([row[key] for row in cpu_rows])
                                           for key in cpu_rows[0] if key.endswith("_ms")}
        report["cpu_all_100_float32_byte_parity"] = all(row["byte_parity"] for row in cpu_rows)

        props = torch.cuda.get_device_properties(0)
        report["resources"] = benchmark.prepare_cuda(
            torch, requested_thread_percentage=None, baseline_sm_count=props.multi_processor_count,
            memory_limit_gib=8)
        predictor = deploy.Predictor(args.model_dir)
        report["environment"] = cost.timing_environment(predictor)
        for record in warmup:
            predictor.predict_image(record["path"])
        predictor.synchronize()
        current = None
        # Allocate all events before the measured GPU diagnostic loop.
        event_sets = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(4)] for _ in measured]

        def pre_hook(member_index):
            def hook(module, inputs):
                assert current is not None
                slot = current["forward_calls"]
                assert slot < 4 and member_index == slot // 2
                current["forward_calls"] += 1
                current["events"][slot][0].record()
            return hook

        def post_hook(member_index):
            def hook(module, inputs, result):
                assert current is not None
                slot = current["forward_calls"] - 1
                assert member_index == slot // 2
                current["events"][slot][1].record()
            return hook

        for member_index, model in enumerate(predictor.models):
            handles.append(model.register_forward_pre_hook(pre_hook(member_index)))
            handles.append(model.register_forward_hook(post_hook(member_index)))
        original_loader = deploy.load_image_arrays
        gpu_rows = []
        reference_probabilities = {}
        accumulator = deploy.StudyAccumulator(measured)
        with (args.output_dir / "gpu_path.image_predictions.csv").open("x", newline="", encoding="utf-8") as image_stream, \
                (args.output_dir / "gpu_path.csv").open("x", newline="", encoding="utf-8") as study_stream:
            image_writer, study_writer = csv.writer(image_stream), csv.writer(study_stream)
            image_writer.writerow(deploy.IMAGE_CSV_HEADER)
            study_writer.writerow(deploy.CSV_HEADER)
            for index, record in enumerate(measured):
                arrays = original_loader(record["path"], [768])  # Outside GPU-path wall timing.
                current = {"image_id": record["image_id"], "path": str(record["path"]),
                           "forward_calls": 0, "events": event_sets[index]}

                def current_array_loader(path, sizes):
                    assert Path(path) == record["path"] and set(sizes) == {768}
                    return arrays

                deploy.load_image_arrays = current_array_loader
                start = time.perf_counter()
                try:
                    probability = predictor.predict_image(record["path"])
                    current["gpu_path_wall_ms"] = elapsed_ms(start)
                finally:
                    deploy.load_image_arrays = original_loader
                assert current["forward_calls"] == 4
                reference_probabilities[record["image_id"]] = probability.copy()
                gpu_rows.append(current)
                current = None
                deploy.write_image_prediction(image_writer, record, probability)
                result = accumulator.add(record, probability)
                if result is not None:
                    subject, study, mean = result
                    deploy.write_prediction(study_writer, subject, study, mean)
        # All original predict_image D2H boundaries already completed. One final
        # boundary precedes Event queries; there is no added per-forward sync.
        predictor.synchronize()
        for handle in handles:
            handle.remove()
        handles.clear()
        if not accumulator.complete():
            raise RuntimeError("Incomplete diagnostic Study coverage")
        for row in gpu_rows:
            spans = [start_event.elapsed_time(end_event) for start_event, end_event in row.pop("events")]
            assert len(spans) == 4 and all(math.isfinite(value) and value >= 0 for value in spans)
            row["forward_event_spans_ms"] = spans
            row["four_forward_event_sum_ms"] = sum(spans)
        write_json(args.output_dir / "gpu_per_image.json", {"images": gpu_rows})
        report["gpu_path_wall_summary_ms"] = summary([row["gpu_path_wall_ms"] for row in gpu_rows])
        report["four_forward_event_sum_summary_ms"] = summary([row["four_forward_event_sum_ms"] for row in gpu_rows])
        report["per_forward_event_mean_ms"] = [statistics.mean(row["forward_event_spans_ms"][index] for row in gpu_rows)
                                                for index in range(4)]
        report["gpu_probabilities_csv_sha256"] = deploy.sha256_file(args.output_dir / "gpu_path.image_predictions.csv")
        report["profiler"] = profile_forwards(predictor, measured, deploy)
        report["predictor_path_profiler"] = profile_predictor_path(
            predictor, measured, deploy, reference_probabilities)
        report["status"] = "completed"
        code = 0
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        report.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - started, exit_code=code)
        write_json(args.output_dir / "diagnostic_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
