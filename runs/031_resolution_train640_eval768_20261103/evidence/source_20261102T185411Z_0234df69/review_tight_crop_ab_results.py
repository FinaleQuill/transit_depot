"""Post-run official-valid review of the four frozen AP-selected A/B models.

Reads only completed score artifacts. --benchmark additionally measures six fixed
candidates on the registered 100-image valid selection; it does not train or tune.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from benchmark_local_resources import validate_inputs, validate_round_output
import cost
from deploy_kd import MEAN, STD, StudyAccumulator, aggregate_member_probabilities, sha256_file
from inference_inputs import ExactU8Input, member_results_to_cpu
from tight_crop_ab_evaluate import (IMAGE_ROOT, VALID, WEIGHTS, paired_bootstrap,
                                    score_arm, validation_records)
from tight_crop_ab_earlystop_evaluate import _identities

ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_20261102/earlystop25")
TIMING_INPUTS = ROOT.parent / "deliveries/034_dual768_proxy_timing_20261029/bundle/inputs"
NAMES = ("A_tiny", "A_base", "B_tiny", "B_base")
PAIRS = (("B_tiny", "A_tiny"), ("B_base", "A_base"),
         ("B_fused", "A_fused"), ("A_base", "A_tiny"),
         ("B_base", "B_tiny"), ("A_fused", "A_tiny"),
         ("B_fused", "B_tiny"), ("A_fused", "A_base"),
         ("B_fused", "B_base"))
SLICES = ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image",
          "non_ap_only", "single_image", "multiple_image")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_sources(run_root, records):
    identities = _identities()
    expected_ids = [row["image_id"] for row in records]
    scores, images, exports = {}, {}, {}
    for name in NAMES:
        directory = run_root / "scores" / name
        receipt = json.loads((directory / "score.json").read_text(encoding="utf-8"))
        arm, member = name.split("_", 1)
        if (receipt.get("arm") != arm or receipt.get("member") != member
                or receipt.get("source_kind") != "ap_selected_best"
                or any(receipt.get(key) != value for key, value in identities.items())):
            raise ValueError(f"Unbound AP-selected score: {name}")
        source = run_root / name / "best.pt"
        if sha256_file(source) != receipt.get("source_sha256"):
            raise ValueError(f"Selected checkpoint changed: {name}")
        probability_path = directory / "image_probs.npz"
        if sha256_file(probability_path) != receipt.get("image_probs_sha256"):
            raise ValueError(f"Image probabilities changed: {name}")
        export_dir = directory / "export"
        export = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
        if (export.get("source_kind") != "ap_selected_best"
                or export.get("source_sha256") != receipt["source_sha256"]
                or export.get("source_epoch") != receipt["epoch"]
                or export.get("arm") != arm or export.get("member") != member
                or export.get("precision") != "float16" or export.get("tta") != "hflip2"
                or export.get("weights_sha256") != receipt.get("export_weights_sha256")
                or sha256_file(export_dir / "state_fp16.pt") != receipt.get("export_weights_sha256")):
            raise ValueError(f"FP16 export changed: {name}")
        with np.load(probability_path, allow_pickle=False) as data:
            ids, values = data["image_ids"].tolist(), data["probabilities"]
        if (ids != expected_ids or values.shape != (len(records), 10)
                or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
            raise ValueError(f"Official-valid image coverage differs: {name}")
        scores[name], images[name], exports[name] = receipt, values, export_dir
    return scores, images, exports


def study_predictions(records, images):
    candidates = {name: values for name, values in images.items()}
    for arm in ("A", "B"):
        candidates[f"{arm}_fused"] = np.stack([
            aggregate_member_probabilities([tiny, base], WEIGHTS)
            for tiny, base in zip(images[f"{arm}_tiny"], images[f"{arm}_base"])])
    predictions = {}
    for name, values in candidates.items():
        accumulator = StudyAccumulator(records)
        studies = {}
        for record, probability in zip(records, values):
            completed = accumulator.add(record, probability)
            if completed is not None:
                subject, study, result = completed
                studies[f"{subject}/{study}"] = result
        if not accumulator.complete() or len(studies) != 859:
            raise RuntimeError(f"Incomplete Study coverage: {name}")
        predictions[name] = studies
    return predictions


def review(run_root, output, *, benchmark=False, timing_inputs=TIMING_INPUTS,
           image_root=IMAGE_ROOT, bootstrap_samples=1000):
    run_root, output = Path(run_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    status = json.loads((run_root / "selection.json").read_text(encoding="utf-8"))
    if set(status.get("selected", {})) != set(NAMES):
        raise ValueError("Four completed selected members are required")
    final_path = run_root / "evaluation" / "metrics.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (final.get("earlystop_plan_sha256") != sha256_file(
            ROOT / "docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json")
            or final.get("test_used") is not False
            or not (run_root / "evaluation" / "REPORT.md").is_file()):
        raise ValueError("Completed official-valid final evaluation is required")
    records, labels, views, subjects = validation_records(VALID, image_root)
    receipts, images, exports = load_sources(run_root, records)
    if any(Path(status["selected"][name]).resolve() != (run_root / "scores" / name).resolve()
           for name in NAMES):
        raise ValueError("Final selection differs from scored members")
    predictions = study_predictions(records, images)
    keys = sorted(labels)
    scores = {name: score_arm(keys, labels, views, values)
              for name, values in predictions.items()}
    for arm in ("A", "B"):
        for subset in SLICES:
            if scores[f"{arm}_fused"][subset]["macro_auc"] != final["arms"][arm][subset]["macro_auc"]:
                raise ValueError("Recomputed fusion differs from final paired evaluation")
    comparisons = {}
    for candidate, reference in PAIRS:
        key = f"{candidate}_minus_{reference}"
        interval = (final["patient_bootstrap"] if (candidate, reference) ==
                    ("B_fused", "A_fused") and bootstrap_samples == 1000 else
                    paired_bootstrap(keys, labels, subjects, views,
                                     {"A": predictions[reference], "B": predictions[candidate]},
                                     samples=bootstrap_samples))
        comparisons[key] = {
            "candidate": candidate, "reference": reference,
            "delta_macro_auc": {
                subset: None if scores[candidate][subset]["macro_auc"] is None
                or scores[reference][subset]["macro_auc"] is None else
                scores[candidate][subset]["macro_auc"] - scores[reference][subset]["macro_auc"]
                for subset in SLICES},
            "patient_bootstrap": interval}
    output.mkdir(parents=True)
    report = {"source_root": str(run_root.resolve()), "source_receipts": receipts,
              "valid_manifest_sha256": sha256_file(VALID),
              "num_images": len(records), "num_studies": len(keys),
              "num_patients": len(set(subjects.values())),
              "candidate_scores": scores, "comparisons": comparisons,
              "fusion_weights": WEIGHTS, "bootstrap_samples": bootstrap_samples,
              "selection_bias_corrected": False, "test_used": False,
              "timing_certified": False}
    write_new(output / "review.json", report)
    lines = ["# AP-selected A/B post-run review", "",
             "Official valid, FP16 batch-1 hflip2 saved probabilities. AP-only valid selected the epochs, so intervals are conditional on selection.", "",
             "| Candidate | Epoch | Overall | AP-only | AP single | AP multiple | Non-AP |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*NAMES, "A_fused", "B_fused"):
        row = scores[name]
        epoch = str(receipts[name]["epoch"]) if name in receipts else "fixed fusion"
        values = [row[s]["macro_auc"] for s in SLICES[:5]]
        lines.append(f"| {name} | {epoch} | " + " | ".join(
            "NA" if value is None else f"{value:.6f}" for value in values) + " |")
    lines += ["", "| Comparison | Overall delta | AP delta | AP patient CI |",
              "| --- | ---: | ---: | ---: |"]
    for key, row in comparisons.items():
        ci = row["patient_bootstrap"]["slices"]["ap_only"]["intervals"]["B_minus_A"]["ci_95"]
        overall, ap = row["delta_macro_auc"]["all"], row["delta_macro_auc"]["ap_only"]
        lines.append(f"| {key} | {overall:+.6f} | "
                     f"{ap:+.6f} | "
                     f"{ci if ci is not None else 'NA'} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    timing = None
    if benchmark:
        timing = benchmark_candidates(exports, images, records, output / "timing",
                                      timing_inputs, image_root)
        write_new(output / "timing_summary.json", timing)
    write_new(output / "results.json", {
        "candidates": scores, "paired_tests": comparisons, "timing": timing,
        "selected_epochs": {name: receipts[name]["epoch"] for name in NAMES},
        "source_receipts": receipts, "valid_manifest_sha256": sha256_file(VALID),
        "num_images": len(records), "num_studies": len(keys),
        "num_patients": len(set(subjects.values())),
        "bootstrap_samples": bootstrap_samples, "selection_bias_corrected": False,
        "T4_certified": False, "test_used": False})
    return report


class FixedPredictor:
    def __init__(self, arm, members, exports):
        import torch
        from inference_reproducibility import configure_inference
        from model_overnight import build_overnight_model

        configure_inference(torch_module=torch)
        if not torch.cuda.is_available():
            raise RuntimeError("Actual FP16 timing requires CUDA")
        self.torch, self.device, self.arm = torch, torch.device("cuda:0"), arm
        self.config = {"gates": None}
        self.input_lookup = ExactU8Input(torch, self.device, MEAN, STD, image_sizes=[])
        self.models = []
        for member in members:
            export = exports[f"{arm}_{member}"]
            metadata = json.loads((export / "config.json").read_text())
            weight_path = export / "state_fp16.pt"
            if sha256_file(weight_path) != metadata["weights_sha256"]:
                raise ValueError("FP16 timing export changed")
            model = build_overnight_model(pretrained=False, **metadata["model_config"])
            model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
            model.eval().requires_grad_(False).to(self.device, dtype=torch.float16,
                                                  memory_format=torch.channels_last)
            self.models.append(model)

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def predict_image(self, path):
        from tight_crop_ab_preprocess import load_pixels

        pixels = load_pixels(Path(path).read_bytes(), self.arm)
        if pixels.dtype != np.uint8 or pixels.shape not in ((768, 768), (768, 768, 3)):
            raise ValueError("Unexpected timing preprocessing output")
        torch = self.torch
        with torch.inference_mode():
            image = self.input_lookup.to_tensor(pixels)
            flipped = torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)
            probabilities, guards = [], []
            for model in self.models:
                logits = [model(image), model(flipped)]
                guards.extend(torch.isfinite(item).all() for item in logits)
                probabilities.append(torch.stack([
                    torch.sigmoid(item.float())[0] for item in logits]).mean(0))
            values = member_results_to_cpu(torch, probabilities, guards)
        return values[0] if len(values) == 1 else aggregate_member_probabilities(values, WEIGHTS)


def local_timing_inputs(inputs, destination):
    """Verify source bundle and copy the 105 fixed valid images to local disk."""
    inputs, destination = Path(inputs), Path(destination)
    selection = json.loads((inputs / "selection.portable.json").read_text(encoding="utf-8"))
    if selection.get("manifest_sha256") != sha256_file(VALID):
        raise ValueError("Timing selection differs from official valid")
    source_selection = destination / "source_selection.json"
    write_new(source_selection, {**selection, "manifest_path": str(VALID.resolve(strict=True))})
    measured, warmup, source_coverage = validate_inputs(inputs / "pictures_100",
                                                        inputs / "pictures_5", source_selection)
    for name, rows in (("pictures_100", measured), ("pictures_5", warmup)):
        root = inputs / name
        for row in rows:
            source = row["path"]
            relative = source.relative_to(root.resolve(strict=True))
            target = destination / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(source) != sha256_file(target):
                raise RuntimeError("Local timing image copy differs from registered source")
    measured, warmup, local_coverage = validate_inputs(destination / "pictures_100",
                                                       destination / "pictures_5", source_selection)
    if local_coverage["measured"] != source_coverage["measured"] or local_coverage["warmup"] != source_coverage["warmup"]:
        raise RuntimeError("Local timing Study coverage changed")
    return measured, warmup, local_coverage, selection


def benchmark_candidates(exports, images, valid_records, output, inputs, image_root):
    inputs, output = Path(inputs), Path(output)
    output.mkdir()
    with tempfile.TemporaryDirectory(prefix="tight_crop_ab_timing_", dir="/tmp") as local_directory:
        measured, warmup, coverage, selection = local_timing_inputs(inputs, local_directory)
        return _benchmark_local(exports, images, valid_records, output, measured,
                                warmup, coverage, selection)


def _benchmark_local(exports, images, valid_records, output, measured, warmup, coverage, selection):
    import torch

    by_id = {row["image_id"]: i for i, row in enumerate(valid_records)}
    declared = {(row["subject_id"], row["study_id"], Path(row["relative_path"]).name):
                row["image_id"] for row in selection["measured"]}
    summaries = {}
    for arm in ("A", "B"):
        for members in (("tiny",), ("base",), ("tiny", "base")):
            name = f"{arm}_{members[0]}" if len(members) == 1 else f"{arm}_fused"
            predictor = FixedPredictor(arm, members, exports)
            destination = output / name
            destination.mkdir()
            for row in warmup:
                predictor.predict_image(row["path"])
            predictor.synchronize()
            duration, first = [], None
            round_receipts, reference_images = [], {}
            reference_accumulator = StudyAccumulator(measured)
            reference_studies = {}
            for row in measured:
                key = (row["subject_id"], f"{row['study_id']}/{row['image_id']}")
                original_id = declared[(row["subject_id"], row["study_id"], row["path"].name)]
                index = by_id[original_id]
                expected = (images[name][index] if name in images else
                    aggregate_member_probabilities([images[f"{arm}_tiny"][index],
                                                    images[f"{arm}_base"][index]], WEIGHTS))
                reference_images[key] = expected
                completed = reference_accumulator.add(row, expected)
                if completed is not None:
                    subject, study, probability = completed
                    reference_studies[(subject, study)] = probability
            if not reference_accumulator.complete():
                raise RuntimeError("Fixed timing selection omits a Study view")
            for index in range(3):
                path = destination / f"round_{index + 1}.csv"
                times, values = cost.time_round(predictor, measured, path)
                round_receipts.append(validate_round_output(measured, path, values))
                from cost import read_image_probabilities
                timed_images = {
                    (subject, f"{study}/{image_id}"): probability
                    for (subject, study, image_id), probability in
                    read_image_probabilities(path.with_suffix(".image_predictions.csv")).items()}
                if not cost.compare_predictions(timed_images, reference_images, atol=0)["match"]:
                    raise ValueError(f"Timed image probabilities differ exactly from full-valid cache: {name}")
                if not cost.compare_predictions(values, reference_studies, atol=0)["match"]:
                    raise ValueError(f"Timed Study probabilities differ exactly from full-valid cache: {name}")
                if first is None:
                    first = values
                elif not cost.compare_predictions(values, first, atol=0)["match"]:
                    raise RuntimeError(f"Timing predictions changed across rounds: {name}")
                duration.append(times)
            rounds, per_image, mean_ms = cost.summarize_timings(duration)
            files = []
            for member in members:
                export = exports[f"{arm}_{member}"]
                for filename in ("config.json", "state_fp16.pt"):
                    source = export / filename
                    files.append({"member": member, "path": str(source),
                                  "bytes": source.stat().st_size, "sha256": sha256_file(source)})
            environment = cost.timing_environment(predictor)
            environment["actual_native_resize_threads"] = 1
            environment["actual_opencv_threads"] = 1
            result = {"mean_ms": mean_ms, "round_mean_ms": rounds.mean(axis=1).tolist(),
                      "per_image_round_ms": rounds.T.tolist(),
                      "per_image_mean_ms": per_image.tolist(),
                      "p95_ms": float(np.percentile(per_image, 95)),
                      "exact_full_valid_probability_match": True,
                      "source_files": files,
                      "round_outputs": round_receipts,
                      "coverage": coverage, "environment": environment,
                      "protocol": "100 SHA-verified local valid images, 5 warmup, serial batch1, 3 rounds, read-through-CSV-flush",
                      "T4_certified": False}
            write_new(destination / "latency_report.json", result)
            summaries[name] = result
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--timing-inputs", type=Path, default=TIMING_INPUTS)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    result = review(args.run_root, args.output, benchmark=args.benchmark,
                    timing_inputs=args.timing_inputs, image_root=args.image_root,
                    bootstrap_samples=args.bootstrap_samples)
    print(json.dumps({name: value["all"]["macro_auc"]
                      for name, value in result["candidate_scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
