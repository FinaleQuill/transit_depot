"""Fixed E2 A/B evaluation on official valid, with a separate historical 044 action."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from deploy_kd import (CLASS_IDS, CSV_HEADER, IMAGE_CSV_HEADER, MEAN, STD,
                       StudyAccumulator, aggregate_member_probabilities,
                       prediction_records, sha256_file, write_image_prediction,
                       write_prediction)
from inference_inputs import ExactU8Input, member_results_to_cpu
from model_overnight import (BASE_BACKBONE_NAME, TINY_BACKBONE_NAME,
                             build_overnight_model, validate_model_config)

ROOT = Path(__file__).resolve().parent
VALID = ROOT / "data/clean_v1_reviewed/valid_manifest.csv"
IMAGE_ROOT = Path("/home/featurize/data/cxr_first")
PREPROCESS = ROOT / "tight_crop_ab_preprocess.py"
CROP = ROOT / "tight_crop.py"
MEMBERS = (("A", "tiny"), ("A", "base"), ("B", "tiny"), ("B", "base"))
WEIGHTS = [[0.0 if c == 0 else 0.5 for c in range(10)],
           [1.0 if c == 0 else 0.5 for c in range(10)]]


def write_new_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checkpoint_identity(checkpoint: dict, *, arm: str, member: str,
                        plan_sha: str, preprocess_sha: str, crop_sha: str,
                        valid_sha: str) -> dict:
    if checkpoint.get("artifact_type") != "kd_student" or checkpoint.get("epoch") != 2:
        raise ValueError("A/B treatment requires the fixed final E2 student checkpoint")
    progress = checkpoint.get("progress")
    if (not isinstance(progress, dict) or progress.get("completed_epochs") != 2
            or progress.get("validated_epochs") != 2):
        raise ValueError("A/B treatment requires two completed, validated epochs")
    if (checkpoint.get("weights_source") != "ema" or
            checkpoint.get("validation_complete") is not True or
            checkpoint.get("inference_ready") is not True or
            checkpoint.get("official_valid_manifest_sha256") != valid_sha or
            checkpoint.get("class_ids") != CLASS_IDS or
            checkpoint.get("image_size") != 768):
        raise ValueError("A/B checkpoint lacks complete EMA/valid identity")
    training, preprocessing = checkpoint.get("training_config"), checkpoint.get("preprocessing")
    if not isinstance(training, dict) or not isinstance(preprocessing, dict):
        raise ValueError("A/B configuration metadata is missing")
    expected = {"ab_plan_sha256": plan_sha, "ab_preprocess_sha256": preprocess_sha,
                "ab_crop_sha256": crop_sha}
    for key, value in expected.items():
        if training.get(key) != value or preprocessing.get(key) != value:
            raise ValueError(f"A/B checkpoint {key} differs from the frozen run")
    if (training.get("arm") != arm or training.get("member") != member or
            preprocessing.get("ab_arm") != arm or preprocessing.get("ab_member") != member):
        raise ValueError("A/B checkpoint arm/member differs from the frozen run")
    if (training.get("seed") != 48 or training.get("epochs") != 2 or
            preprocessing.get("decoder_candidate") != "full_decode_exact_reference_v1" or
            preprocessing.get("validation_tta") != "hflip2" or
            preprocessing.get("crop_policy") != ("none" if arm == "A" else "tight_crop.detect_tight_crop")):
        raise ValueError("A/B seed, schedule, decoder, or TTA changed")
    if training.get("erase_probability") != (0.5 if arm == "B" else 0.0):
        raise ValueError("A/B random erasing assignment changed")
    from tight_crop_ab_training import INIT_SHA
    if checkpoint.get("student_warm_start", {}).get("source_snapshot_sha256") != INIT_SHA[member]:
        raise ValueError("A/B E2 student warm start differs from registered FP32 EMA")
    model_config = validate_model_config(checkpoint.get("model_config", {}))
    expected_backbone = TINY_BACKBONE_NAME if member == "tiny" else BASE_BACKBONE_NAME
    if model_config["backbone_name"] != expected_backbone:
        raise ValueError("A/B member backbone changed")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing final EMA tensors")
    return {**model_config, "gradient_checkpointing": False}


def export_e2(checkpoint_path: Path, output: Path, *, arm: str, member: str,
              plan_sha: str, preprocess_sha: str, crop_sha: str, valid_sha: str) -> dict:
    import torch
    from tight_crop_ab_training import load_final_ema
    if output.exists():
        raise FileExistsError(output)
    source_sha = sha256_file(checkpoint_path)
    checkpoint = load_final_ema(checkpoint_path, arm=arm, member=member)
    model_config = checkpoint_identity(checkpoint, arm=arm, member=member,
        plan_sha=plan_sha, preprocess_sha=preprocess_sha, crop_sha=crop_sha,
        valid_sha=valid_sha)
    half_state = {}
    for key, tensor in checkpoint["model_state_dict"].items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Invalid EMA tensor state")
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite E2 EMA tensor")
            tensor = tensor.half()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("E2 EMA overflows FP16")
        half_state[key] = tensor
    if sha256_file(checkpoint_path) != source_sha:
        raise RuntimeError("E2 checkpoint changed during export")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(half_state, output / "state_fp16.pt")
    config = {"schema_version": 1, "artifact_type": "tight_crop_ab_e2_fp16",
              "arm": arm, "member": member, "model_config": model_config,
              "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": source_sha,
              "source_epoch": 2, "weights_source": "final_ema", "precision": "float16",
              "weights_sha256": sha256_file(output / "state_fp16.pt"),
              "plan_sha256": plan_sha, "preprocess_sha256": preprocess_sha,
              "crop_sha256": crop_sha, "valid_manifest_sha256": valid_sha,
              "image_size": 768, "tta": "hflip2", "class_ids": CLASS_IDS,
              "decoder": "full_decode_exact_reference_v1"}
    write_new_json(output / "config.json", config)
    del checkpoint, half_state
    gc.collect()
    return config


class ABPredictor:
    def __init__(self, exports: dict[str, Path], arm: str, *, device="cuda"):
        import torch
        from inference_reproducibility import configure_inference
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Actual A/B FP16 evaluation requires CUDA")
        configure_inference(torch_module=torch)
        torch.set_num_threads(1)
        self.arm = arm
        self.models = []
        self.input_lookup = ExactU8Input(torch, self.device, MEAN, STD, image_sizes=[])
        for member in ("tiny", "base"):
            directory = exports[member]
            config = json.loads((directory / "config.json").read_text())
            if config.get("artifact_type") != "tight_crop_ab_e2_fp16" or config.get("arm") != arm or config.get("member") != member:
                raise ValueError("Export arm/member mismatch")
            weights = directory / "state_fp16.pt"
            if sha256_file(weights) != config["weights_sha256"]:
                raise ValueError("Exported weights changed")
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model = build_overnight_model(pretrained=False, **config["model_config"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False).to(self.device, dtype=torch.float16,
                                                   memory_format=torch.channels_last)
            self.models.append(model)
            del state
            gc.collect()

    def predict_image(self, path: Path) -> np.ndarray:
        from tight_crop_ab_preprocess import load_pixels
        pixels = load_pixels(path, self.arm)
        if pixels.dtype != np.uint8 or pixels.shape not in ((768, 768), (768, 768, 3)):
            raise ValueError("Shared A/B preprocessing must return uint8 768 pixels")
        torch = self.torch
        results, guards = [], []
        with torch.inference_mode():
            image = self.input_lookup.to_tensor(pixels)
            flipped = torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)
            for model in self.models:
                probabilities = []
                for input_tensor in (image, flipped):
                    logits = model(input_tensor)
                    if tuple(logits.shape) != (1, 10):
                        raise ValueError("Invalid A/B model output shape")
                    guards.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                results.append(torch.stack(probabilities).mean(0))
            member_values = member_results_to_cpu(torch, results, guards)
        return aggregate_member_probabilities(member_values, WEIGHTS)


def validation_records(manifest: Path, image_root: Path):
    records = prediction_records(image_root, manifest, split="valid")
    if len(records) != 1422:
        raise ValueError("Official valid must cover exactly 1422 images")
    labels, views, subjects = {}, {}, {}
    for row in records:
        key = row["study_key"]
        target = row["targets"] if isinstance(row["targets"], list) else json.loads(row["targets"])
        if len(target) != 10 or any(v not in (0, 1) for v in target):
            raise ValueError("Invalid ten-class official valid label")
        if key in labels and labels[key] != target:
            raise ValueError("Inconsistent labels within a Study")
        labels[key] = target
        views.setdefault(key, []).append(row["view_position"].upper())
        subjects[key] = row["subject_id"]
    if len(labels) != 859:
        raise ValueError("Official valid must cover exactly 859 Studies")
    return records, labels, views, subjects


def score_slice(keys, labels, predictions, *, include):
    from sklearn.metrics import roc_auc_score
    selected = [key for key in keys if include(key)]
    targets = np.asarray([labels[key] for key in selected], dtype=np.int8).reshape(-1, 10)
    values = np.asarray([predictions[key] for key in selected], dtype=np.float64).reshape(-1, 10)
    scores, counts = [], []
    for c in range(10):
        positive = int(targets[:, c].sum())
        negative = len(selected) - positive
        eligible = positive > 0 and negative > 0
        scores.append(float(roc_auc_score(targets[:, c], values[:, c])) if eligible else None)
        counts.append({"positive": positive, "negative": negative, "eligible": eligible})
    available = [x for x in scores if x is not None]
    return {"num_studies": len(selected), "eligible_classes": len(available),
            "class_counts": counts, "per_class_auc": scores,
            "macro_auc": float(np.mean(available)) if available else None}


def score_arm(keys, labels, views, predictions):
    slices = {"all": lambda key: True,
              "ap_only": lambda key: all(view == "AP" for view in views[key]),
              "ap_only_single_image": lambda key: len(views[key]) == 1 and views[key][0] == "AP",
              "ap_only_multiple_image": lambda key: len(views[key]) > 1 and all(view == "AP" for view in views[key]),
              "non_ap_only": lambda key: not all(view == "AP" for view in views[key]),
              "single_image": lambda key: len(views[key]) == 1,
              "multiple_image": lambda key: len(views[key]) > 1}
    return {name: score_slice(keys, labels, predictions, include=predicate)
            for name, predicate in slices.items()}


def fmt_auc(value):
    return "NA" if value is None else f"{value:.6f}"


def fmt_interval(value):
    if value is None:
        return "NA"
    return f"[{value[0]:.6f}, {value[1]:.6f}]"


def paired_report(metrics):
    a, b = metrics["arms"]["A"], metrics["arms"]["B"]
    lines = ["# Tight-crop A/B official-valid evaluation", "",
             f"Images: {metrics['num_images']}; Studies: {metrics['num_studies']}; patients: {metrics['num_patients']}.",
             "Fixed E2 final EMA, full decode, FP16, 768, hflip2. Class 1 uses Base; other classes use equal Tiny/Base probabilities.",
             "AP-only requires every image in the Study to be AP; non-AP includes Studies with any other view.", "",
             "| Study slice | Studies | Eligible classes | A macro-AUC | B macro-AUC | B - A |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image",
                 "non_ap_only", "single_image", "multiple_image"):
        lines.append(f"| {name} | {a[name]['num_studies']} | {a[name]['eligible_classes']} | {fmt_auc(a[name]['macro_auc'])} | {fmt_auc(b[name]['macro_auc'])} | {fmt_auc(metrics['B_minus_A_macro_auc'][name])} |")
    for slice_name in ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image"):
        lines += ["", f"## Per-class AUC: {slice_name}", "",
                  "| Class | Positive / negative Studies | A | B | B - A |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for c in range(10):
            count = a[slice_name]["class_counts"][c]
            av, bv = a[slice_name]["per_class_auc"][c], b[slice_name]["per_class_auc"][c]
            difference = None if av is None or bv is None else bv - av
            lines.append(f"| {c+1} | {count['positive']} / {count['negative']} | {fmt_auc(av)} | {fmt_auc(bv)} | {fmt_auc(difference)} |")
    lines += ["", "## Patient-cluster bootstrap", "",
              "1,000 paired patient resamples; seed 20261102. Each interval uses classes eligible in the original slice.", "",
              "| Slice | Eligible classes | A 95% CI | B 95% CI | B - A 95% CI | Valid paired resamples |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image"):
        entry = metrics["patient_bootstrap"]["slices"][name]
        intervals = entry["intervals"]
        lines.append(f"| {name} | {entry['eligible_classes']} | {fmt_interval(intervals['A']['ci_95'])} | {fmt_interval(intervals['B']['ci_95'])} | {fmt_interval(intervals['B_minus_A']['ci_95'])} | {intervals['B_minus_A']['valid_resamples']} |")
    lines += ["", "Historical 044 is reported separately and is not a paired treatment arm.",
              "Validation comparisons are exploratory; no test selection or T4 certification.", ""]
    return "\n".join(lines)


def paired_bootstrap(keys, labels, subjects, views, predictions, *, samples=1000, seed=20261102):
    grouped = {}
    for key in keys:
        grouped.setdefault(subjects[key], []).append(key)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    result = {}
    for subset, predicate in (("all", lambda key: True),
                              ("ap_only", lambda key: all(v == "AP" for v in views[key])),
                              ("ap_only_single_image", lambda key: len(views[key]) == 1 and views[key][0] == "AP"),
                              ("ap_only_multiple_image", lambda key: len(views[key]) > 1 and all(v == "AP" for v in views[key]))):
        original = score_slice(keys, labels, predictions["A"], include=predicate)
        eligible = [c for c, row in enumerate(original["class_counts"]) if row["eligible"]]
        draws = {"A": [], "B": [], "B_minus_A": []}
        for _ in range(samples):
            selected = [key for i in rng.integers(0, len(patients), len(patients))
                        for key in grouped[patients[i]] if predicate(key)]
            scored = {arm: score_slice(selected, labels, prediction, include=lambda _: True)
                      for arm, prediction in predictions.items()}
            scores = {arm: (float(np.mean([result["per_class_auc"][c] for c in eligible]))
                            if eligible and all(result["per_class_auc"][c] is not None for c in eligible)
                            else None) for arm, result in scored.items()}
            for arm in ("A", "B"):
                if scores[arm] is not None:
                    draws[arm].append(scores[arm])
            if scores["A"] is not None and scores["B"] is not None:
                draws["B_minus_A"].append(scores["B"] - scores["A"])
        result[subset] = {"eligible_classes": len(eligible),
                          "intervals": {name: {"valid_resamples": len(values),
                                               "ci_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None}
                                        for name, values in draws.items()}}
    return {"samples": samples, "seed": seed, "sampling_unit": "patient_with_all_Studies",
            "paired": True, "slices": result}


def evaluate_arm(predictor, records, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    accumulator = StudyAccumulator(records)
    studies = {}
    with (destination / "image_predictions.csv").open("x", newline="", encoding="utf-8") as image_stream, \
         (destination / "study_predictions.csv").open("x", newline="", encoding="utf-8") as study_stream:
        image_writer, study_writer = csv.writer(image_stream), csv.writer(study_stream)
        image_writer.writerow(IMAGE_CSV_HEADER)
        study_writer.writerow(CSV_HEADER)
        for index, row in enumerate(records, 1):
            source = row["path"].read_bytes()
            if hashlib.sha256(source).hexdigest() != row["file_sha256"]:
                raise ValueError("Official valid image changed: " + row["image_id"])
            probabilities = predictor.predict_image(row["path"])
            write_image_prediction(image_writer, row, probabilities)
            result = accumulator.add(row, probabilities)
            if result is not None:
                subject, study, probability = result
                write_prediction(study_writer, subject, study, probability)
                studies[f"{subject}/{study}"] = probability.copy()
            if index % 100 == 0 or index == len(records):
                print(f"[{predictor.arm} VALID] {index}/{len(records)}", flush=True)
    if not accumulator.complete() or len(studies) != 859:
        raise RuntimeError("A/B official valid Study coverage incomplete")
    return studies


def paired(*, plan: Path, run_root: Path, output: Path, manifest: Path = VALID,
           image_root: Path = IMAGE_ROOT):
    import torch
    plan_sha, preprocess_sha, crop_sha = (sha256_file(path) for path in (plan, PREPROCESS, CROP))
    valid_sha = sha256_file(manifest)
    records, labels, views, subjects = validation_records(manifest, image_root)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("exports", "A", "B", "metrics.json", "REPORT.md")):
        raise FileExistsError("Paired evaluation already has output artifacts")
    exports = output / "exports"
    exports.mkdir()
    configs = {}
    for arm, member in MEMBERS:
        checkpoint = run_root / f"{arm}_{member}" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        configs[f"{arm}_{member}"] = export_e2(checkpoint, exports / f"{arm}_{member}",
            arm=arm, member=member, plan_sha=plan_sha, preprocess_sha=preprocess_sha,
            crop_sha=crop_sha, valid_sha=valid_sha)
    predictions = {}
    for arm in ("A", "B"):
        predictor = ABPredictor({member: exports / f"{arm}_{member}" for member in ("tiny", "base")}, arm)
        predictions[arm] = evaluate_arm(predictor, records, output / arm)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    keys = sorted(labels)
    scores = {arm: score_arm(keys, labels, views, predictions[arm]) for arm in ("A", "B")}
    intervals = paired_bootstrap(keys, labels, subjects, views, predictions)
    deltas = {name: None if scores["A"][name]["macro_auc"] is None or scores["B"][name]["macro_auc"] is None
              else scores["B"][name]["macro_auc"] - scores["A"][name]["macro_auc"]
              for name in scores["A"]}
    metrics = {"plan_sha256": plan_sha, "preprocess_sha256": preprocess_sha,
               "crop_sha256": crop_sha, "manifest_sha256": valid_sha,
               "num_images": len(records), "num_studies": len(keys),
               "num_patients": len(set(subjects.values())), "members": configs,
               "arms": scores, "B_minus_A_macro_auc": deltas,
               "patient_bootstrap": intervals, "precision": "float16",
               "tta": "hflip2", "decoder": "full_decode_exact_reference_v1",
               "class_weights": WEIGHTS, "test_used": False,
               "selection_bias_corrected": False, "T4_certified": False}
    write_new_json(output / "metrics.json", metrics)
    (output / "REPORT.md").write_text(paired_report(metrics), encoding="utf-8")
    return metrics


def historical(*, output: Path, manifest: Path = VALID, image_root: Path = IMAGE_ROOT):
    from deploy_kd import read_submission
    from student_ensemble import StudentEnsemble
    from student_experiment_evaluation import evaluate_predictor
    model = ROOT.parent / "deliveries/044_tiny_base_class1B_draft768_20261031/model"
    destination = output / "historical_044"
    predictor = StudentEnsemble(model, device="cuda")
    report = evaluate_predictor(predictor, image_root=image_root, manifest=manifest,
                                output_dir=destination)
    records, labels, views, subjects = validation_records(manifest, image_root)
    submitted = read_submission(destination / "valid_predictions.csv")
    predictions = {f"{subject}/{study}": value for (subject, study), value in submitted.items()}
    if set(predictions) != set(labels):
        raise ValueError("Historical 044 valid Study coverage differs")
    scores = score_arm(sorted(labels), labels, views, predictions)
    bootstrap = paired_bootstrap(sorted(labels), labels, subjects, views,
                                 {"A": predictions, "B": predictions})
    intervals = {name: {"eligible_classes": entry["eligible_classes"],
                        "valid_resamples": entry["intervals"]["A"]["valid_resamples"],
                        "ci_95": entry["intervals"]["A"]["ci_95"]}
                 for name, entry in bootstrap["slices"].items()}
    summary = {"reference": "historical_044", "paired_treatment_arm": False,
               "decoder": "JPEG_draft_target768", "macro_auc": report["macro_auc"],
               "slices": scores, "patient_bootstrap": {"samples": 1000,
                   "seed": 20261102, "sampling_unit": "patient_with_all_Studies",
                   "slices": intervals},
               "manifest_sha256": sha256_file(manifest),
               "report_sha256": sha256_file(destination / "validation_report.json")}
    write_new_json(destination / "historical_summary.json", summary)
    lines = ["# Historical 044 official-valid reference", "",
             "Draft-768 decoder, FP16, hflip2. This is a separate historical reference, not a paired A/B treatment arm.", "",
             "| Slice | Studies | Eligible classes | Macro-AUC | 95% patient CI |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("all", "ap_only", "ap_only_single_image", "ap_only_multiple_image",
                 "non_ap_only", "single_image", "multiple_image"):
        score = scores[name]
        ci = intervals.get(name, {}).get("ci_95")
        lines.append(f"| {name} | {score['num_studies']} | {score['eligible_classes']} | {fmt_auc(score['macro_auc'])} | {fmt_interval(ci)} |")
    lines += ["", "| Class | All AUC | AP-only AUC | AP-only single | AP-only multiple |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for c in range(10):
        lines.append(f"| {c+1} | {fmt_auc(scores['all']['per_class_auc'][c])} | {fmt_auc(scores['ap_only']['per_class_auc'][c])} | {fmt_auc(scores['ap_only_single_image']['per_class_auc'][c])} | {fmt_auc(scores['ap_only_multiple_image']['per_class_auc'][c])} |")
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("historical", "paired"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=VALID)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    args = parser.parse_args(argv)
    if not args.plan.is_file():
        raise FileNotFoundError(args.plan)
    if args.action == "historical":
        args.output.mkdir(parents=True, exist_ok=True)
        result = historical(output=args.output, manifest=args.manifest, image_root=args.image_root)
    else:
        result = paired(plan=args.plan, run_root=args.run_root, output=args.output,
                        manifest=args.manifest, image_root=args.image_root)
    print(json.dumps({"action": args.action, "result": result.get("macro_auc", result.get("B_minus_A_macro_auc"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
