"""Study-batch preview for the frozen 043, 044 or 045 exports.

Without --execute this inspects paths and model metadata only. With --execute it
compares the original serial Predictor against one forward per complete Study,
using the original decode, resize, normalization, TTA and probability fusion.
This is a local research preview, not the organizer's cost certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, MEAN, STD, StudyAccumulator,
                       _letterbox_image, aggregate_member_probabilities,
                       discover_images, sha256_file, write_image_prediction,
                       write_prediction)
from inference_reproducibility import compare_probability_outputs


SAMPLE_ROOT = Path(__file__).resolve().parent / "artifacts/new_study_cost_20261102/cost_data_test"
MODEL_045 = Path(__file__).resolve().parent / "artifacts/kd_v1/bounded_finetune_20261101/reviewed_models/full_decode"
MODEL_044 = Path(__file__).resolve().parent.parent / "deliveries/044_tiny_base_class1B_draft768_20261031/model"
CONFIG_SHA_045 = "09f2c96debb6afbb83e5a99c4c1e80b5fcbb6a2a87c343eb4c70c32659ec563d"
CONFIG_SHA_044 = "f722b9c3e0e62ac0a675c49163dfac58bc1b52481ded8f2e9e05c1602c2fe92d"
CONFIG_SHA_043 = "ec8d155fdf1b4a8ca5de7c914ab098efa5d75c449f3a986e950a811d0c632011"
ROUNDS = 3  # A local preview convention; the new announcement does not prescribe rounds.


def study_groups(records):
    groups = []
    seen = set()
    for record in records:
        key = (record["subject_id"], record["study_id"])
        if key not in seen:
            groups.append((key, []))
            seen.add(key)
        if groups[-1][0] != key:
            raise ValueError("Images of one Study are not contiguous")
        groups[-1][1].append(record)
    return groups


def inspect_inputs(model_dir, pictures_100, pictures_5):
    model_dir = Path(model_dir).resolve(strict=True)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = sha256_file(config_path)
    if digest == CONFIG_SHA_045:
        if (config.get("artifact_type") != "fixed_student_ensemble"
                or config.get("decoder_candidate") is not None
                or [m.get("directory") for m in config.get("members", [])] != ["X1", "B"]):
            raise ValueError("045 config no longer declares the fixed full-decode X1+B pair")
        model_label = "045_X1_B_full_decode"
    elif digest == CONFIG_SHA_044:
        if (config.get("artifact_type") != "fixed_classwise_student_pair"
                or config.get("decoder_candidate") != "JPEG_draft_target768"
                or config.get("candidate") != "AB_B100_class1"
                or config.get("view_candidate") is not None
                or [m.get("directory") for m in config.get("source_members", [])] != ["A", "B"]):
            raise ValueError("044 config no longer declares the fixed A+B draft pair")
        model_label = "044_A_B_draft768"
    elif digest == CONFIG_SHA_043:
        if (config.get("artifact_type") != "experimental_student_fp16"
                or config.get("backbone_name") != "convnext_base.fb_in22k_ft_in1k_384"):
            raise ValueError("043 config no longer declares the original Base model")
        model_label = "043_base_B"
    else:
        raise ValueError("Only the original 043, 044 or 045 config is supported")
    if config.get("image_size") != 768 or config.get("tta") != "hflip2":
        raise ValueError("The preview requires the frozen 768/hflip2 configuration")
    measured = study_groups(discover_images(pictures_100))
    warmup = study_groups(discover_images(pictures_5))
    if not measured or not warmup:
        raise ValueError("Both image directories need at least one Study")
    measured_keys = {key for key, _ in measured}
    if measured_keys & {key for key, _ in warmup}:
        raise ValueError("Warmup and measured Studies overlap")
    summary = {
        "model": model_label,
        "model_dir": str(model_dir),
        "config_sha256": digest,
        "decoder": config.get("decoder_candidate") or "full_decode_exact_reference_v1",
        "pictures_100": str(Path(pictures_100).resolve(strict=True)),
        "pictures_5": str(Path(pictures_5).resolve(strict=True)),
        "measured_studies": len(measured),
        "measured_images": sum(len(rows) for _, rows in measured),
        "measured_source_bytes": sum(row["path"].stat().st_size
                                     for _, rows in measured for row in rows),
        "warmup_studies": len(warmup),
        "warmup_images": sum(len(rows) for _, rows in warmup),
        "warmup_source_bytes": sum(row["path"].stat().st_size
                                   for _, rows in warmup for row in rows),
        "measured_images_per_study": dict(sorted(Counter(len(rows) for _, rows in measured).items())),
        "warmup_images_per_study": dict(sorted(Counter(len(rows) for _, rows in warmup).items())),
        "batch_unit": "one Study; all its pictures in one model forward per TTA view",
        "rounds": ROUNDS,
        "official_cost_certification": False,
    }
    return config, measured, warmup, summary


def make_batch_predictor(predictor, max_study_images):
    from inference_inputs import ExactU8Input

    if predictor.config.get("decoder_candidate") == "JPEG_draft_target768":
        from student_draft_pair import draft_image_pixels

        def decode(path):
            return draft_image_pixels(path, [768])[768]
    elif predictor.config.get("decoder_candidate") is None:
        from inference_inputs import load_image_pixels

        def decode(path):
            return load_image_pixels(path, [768], _letterbox_image)[768]
    else:
        raise ValueError("Unknown frozen decoder candidate")

    # Separate pinned staging per image: the original batch-1 buffer cannot be
    # reused before an asynchronous H2D copy from the previous image completes.
    lookups = [predictor.input_lookup]
    for _ in range(max_study_images - 1):
        lookups.append(ExactU8Input(predictor.torch, predictor.device, MEAN, STD,
                                    image_sizes=[768]))

    def predict_study(rows):
        torch = predictor.torch
        if len(rows) > len(lookups):
            raise ValueError("Study exceeds prepared input staging slots")
        with torch.inference_mode():
            images = [lookup.to_tensor(decode(row["path"]))
                      for lookup, row in zip(lookups, rows)]
            batch = images[0] if len(images) == 1 else torch.cat(images, dim=0)
            flipped = torch.flip(batch, dims=(-1,)).contiguous(
                memory_format=torch.channels_last)
            member_means, finite_checks = [], []
            for member, model in zip(predictor.members, predictor.models):
                if member["tta"] != "hflip2":
                    raise ValueError("Unexpected member TTA")
                probabilities = []
                for inputs in (batch, flipped):
                    logits = model(inputs)
                    if tuple(logits.shape) != (len(rows), 10):
                        raise FloatingPointError("Invalid batched student logits")
                    finite_checks.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float()))
                member_means.append(torch.stack(probabilities).mean(0))
            payload = torch.cat((torch.stack(member_means).reshape(-1),
                                 torch.stack(finite_checks).all().reshape(1).float())).cpu().numpy()
        if payload[-1] != np.float32(1.0):
            raise FloatingPointError("Invalid batched student logits")
        member_values = payload[:-1].reshape(len(predictor.models), len(rows), 10)
        return [aggregate_member_probabilities(member_values[:, i], predictor.model_weights)
                for i in range(len(rows))]

    return predict_study


def warmup_model(predictor, groups, batch_predict):
    for _, rows in groups:
        if batch_predict is None:
            for row in rows:
                predictor.predict_image(row["path"])
        else:
            batch_predict(rows)
    predictor.synchronize()


def run_round(predictor, groups, output, batch_predict):
    records = [record for _, rows in groups for record in rows]
    accumulator = StudyAccumulator(records)
    images, studies, timings = {}, {}, []
    image_output = output.with_suffix(".image_predictions.csv")
    with output.open("x", encoding="utf-8", newline="") as study_stream, \
            image_output.open("x", encoding="utf-8", newline="") as image_stream:
        study_writer, image_writer = csv.writer(study_stream), csv.writer(image_stream)
        study_writer.writerow(CSV_HEADER)
        image_writer.writerow(IMAGE_CSV_HEADER)
        for key, rows in groups:
            predictor.synchronize()
            start = time.perf_counter()
            probabilities = ([predictor.predict_image(row["path"]) for row in rows]
                             if batch_predict is None else batch_predict(rows))
            if len(probabilities) != len(rows):
                raise RuntimeError("Missing batched image output")
            for record, probability in zip(rows, probabilities):
                write_image_prediction(image_writer, record, probability)
                image_key = (record["subject_id"],
                             f"{record['study_id']}/{record['path'].name}")
                images[image_key] = probability.copy()
                result = accumulator.add(record, probability)
                if result is not None:
                    subject, study, mean = result
                    write_prediction(study_writer, subject, study, mean)
                    studies[(subject, study)] = mean.copy()
            image_stream.flush()
            study_stream.flush()
            predictor.synchronize()
            timings.append({"study": "/".join(key), "images": len(rows),
                            "milliseconds": (time.perf_counter() - start) * 1000})
    if not accumulator.complete() or len(studies) != len(groups) or len(images) != len(records):
        raise RuntimeError("Incomplete Study or image coverage")
    return {"images": images, "studies": studies, "timings": timings,
            "study_csv": str(output), "image_csv": str(image_output)}


def compare_round(reference, candidate):
    return {
        "image": compare_probability_outputs(candidate["images"], reference["images"]),
        "study": compare_probability_outputs(candidate["studies"], reference["studies"]),
    }


def summarize_round(result):
    durations = result["timings"]
    by_size = {}
    for size in sorted({row["images"] for row in durations}):
        selected = [row["milliseconds"] for row in durations if row["images"] == size]
        by_size[str(size)] = {"studies": len(selected),
                              "mean_ms_per_study": float(np.mean(selected))}
    return {"studies": len(durations), "images": sum(row["images"] for row in durations),
            "total_ms": float(sum(row["milliseconds"] for row in durations)),
            "mean_ms_per_study": float(np.mean([row["milliseconds"] for row in durations])),
            "by_images_per_study": by_size, "per_study": durations,
            "study_csv": result["study_csv"], "image_csv": result["image_csv"]}


def runtime_environment(predictor):
    torch = predictor.torch
    props = torch.cuda.get_device_properties(predictor.device)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10).splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        driver = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": props.name, "gpu_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor], "driver": driver,
            "model_inventory_bytes": predictor.inventory["total_bytes"],
            "precision": "float16", "image_size": 768, "tta": "hflip2"}


def execute(measured, warmup, summary, output_dir):
    from inference_reproducibility import configure_inference
    from submission_runtime import load_submission_predictor

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("Preview outputs must be under /tmp")
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    configure_inference()
    predictor = load_submission_predictor(summary["model_dir"])
    if (len(predictor.models) != (1 if summary["model"] == "043_base_B" else 2)
            or any(member.get("tta") != "hflip2" or member.get("image_size") != 768
                   for member in predictor.members)):
        raise ValueError("Loaded member configuration differs from the frozen preview")
    batch_predict = make_batch_predictor(
        predictor, max(len(rows) for _, rows in measured + warmup))
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {**summary, "output_dir": str(output_dir),
              "environment": runtime_environment(predictor),
              "measurement_scope": "local Study preview, not official cost certification",
              "warmup_protocol": "5 supplied Studies before each route; local convention",
              "routes": {"serial_image_reference": [], "whole_study_batch": []},
              "comparisons": []}
    for round_index in range(ROUNDS):
        warmup_model(predictor, warmup, None)
        serial = run_round(predictor, measured,
                           output_dir / f"round_{round_index + 1}_serial.csv", None)
        warmup_model(predictor, warmup, batch_predict)
        batched = run_round(predictor, measured,
                            output_dir / f"round_{round_index + 1}_study_batch.csv", batch_predict)
        report["routes"]["serial_image_reference"].append(summarize_round(serial))
        report["routes"]["whole_study_batch"].append(summarize_round(batched))
        report["comparisons"].append(compare_round(serial, batched))
        report_path = output_dir / "preview.json"
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_045,
                        help=f"Frozen model directory; 044 draft: {MODEL_044}")
    parser.add_argument("--pictures-100", type=Path, default=SAMPLE_ROOT / "pictures_100")
    parser.add_argument("--pictures-5", type=Path, default=SAMPLE_ROOT / "pictures_5")
    parser.add_argument("--output-dir", type=Path, help="New /tmp directory for CSV and JSON results")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable CUDA model inference")
    args = parser.parse_args(argv)
    _, measured, warmup, summary = inspect_inputs(
        args.model_dir, args.pictures_100, args.pictures_5)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires --output-dir")
        report = execute(measured, warmup, summary, args.output_dir)
        print(json.dumps({"output_dir": report["output_dir"],
                          "comparisons": report["comparisons"]}, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
