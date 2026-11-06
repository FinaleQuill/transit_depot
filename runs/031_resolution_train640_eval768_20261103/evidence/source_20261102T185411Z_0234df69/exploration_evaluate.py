"""Train/valid-only exploration evaluation, fixed probability fusion and timing.

Imports do not construct models or inspect data. ``evaluate`` explicitly runs a
validated EMA in FP16 at 768; ``compare`` is CPU-only unless --timing-fusions is
supplied. No command accepts a test split or a replacement validation manifest.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
VALID = ROOT / "data/clean_v1_reviewed/valid_manifest.csv"
TRAIN = ROOT / "data/clean_v1_reviewed/train_clean.csv"
IMAGE_ROOT = Path("/home/featurize/data/cxr_first")
CLASS_IDS = list(range(1, 11))
MODES = ("identity", "hflip2")
ROI_VERSION = "body_envelope_occupancy85_margin_v1"
MIL_VERSION = "class_attention_v1"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_probabilities(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape or not np.isfinite(result).all()
            or (result < 0).any() or (result > 1).any()):
        raise ValueError("Incomplete, nonfinite or out-of-range probabilities")
    return result


def aggregate_predictions(records, per_view):
    """FP32 TTA mean per image, then FP64 arithmetic mean within each Study."""
    values = checked_probabilities(per_view, (len(records), 2, 10))
    groups = OrderedDict()
    image_ids = set()
    for i, row in enumerate(records):
        if row.get("split") not in {"train", "valid"}:
            raise ValueError("Only original train/valid records are accepted")
        key = str(row["study_key"])
        if key != f"{row['subject_id']}/{row['study_id']}":
            raise ValueError("Study key must include its original patient identity")
        if row["image_id"] in image_ids:
            raise ValueError("Duplicate image identity")
        image_ids.add(row["image_id"])
        target = np.asarray(row["targets"], dtype=np.int8)
        if target.shape != (10,) or not np.isin(target, (0, 1)).all():
            raise ValueError("Original ten-class binary targets are required")
        if key not in groups:
            groups[key] = {"indices": [], "targets": target, "subject": str(row["subject_id"]),
                           "study": str(row["study_id"]), "views": []}
        if not np.array_equal(groups[key]["targets"], target):
            raise ValueError("Labels differ within a Study")
        groups[key]["indices"].append(i)
        groups[key]["views"].append(str(row.get("view_position", "")).upper())
    # Match torch.float32 mean used in deployment; do not introduce float64 TTA.
    image_mean = values.astype(np.float32).mean(axis=1, dtype=np.float32).astype(np.float64)
    result = {"study_keys": np.asarray(list(groups)),
              "subject_ids": np.asarray([g["subject"] for g in groups.values()]),
              "study_ids": np.asarray([g["study"] for g in groups.values()]),
              "targets": np.stack([g["targets"] for g in groups.values()]),
              "ap_only": np.asarray([all(v == "AP" for v in g["views"]) for g in groups.values()]),
              "image_counts": np.asarray([len(g["indices"]) for g in groups.values()]),
              "identity": np.stack([values[g["indices"], 0].mean(axis=0) for g in groups.values()]),
              "hflip2": np.stack([image_mean[g["indices"]].mean(axis=0) for g in groups.values()]),
              "image_ids": np.asarray([str(r["image_id"]) for r in records]),
              "image_study_keys": np.asarray([str(r["study_key"]) for r in records]),
              "image_view_positions": np.asarray([str(r.get("view_position", "")) for r in records]),
              "per_view_probabilities": values, "image_probabilities": image_mean}
    return result


def _auc_plan(labels, probabilities):
    order = np.argsort(probabilities, kind="stable")
    p = np.asarray(probabilities)[order]
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    return order, starts, np.asarray(labels, dtype=np.float64)[order]


def _weighted_auc(plan, weights):
    order, starts, positive = plan
    weight = np.asarray(weights, dtype=np.float64)[order]
    pos = np.add.reduceat(weight * positive, starts)
    neg = np.add.reduceat(weight * (1 - positive), starts)
    denominator = pos.sum() * neg.sum()
    if denominator == 0:
        return None
    return float(np.dot(pos, np.cumsum(neg) - .5 * neg) / denominator)


def class_aucs(targets, probabilities):
    y, p = np.asarray(targets), checked_probabilities(probabilities)
    if y.ndim != 2 or y.shape[1] != 10 or y.shape != p.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("AUC arrays need matching N by 10 binary labels/probabilities")
    if not len(y):
        return [None] * 10
    return [_weighted_auc(_auc_plan(y[:, c], p[:, c]), np.ones(len(y))) for c in range(10)]


def score_predictions(bundle, mode="hflip2"):
    scores = {}
    n = len(bundle["targets"])
    for name, mask in (("all", np.ones(n, dtype=bool)), ("ap_only", bundle["ap_only"]),
                       ("ap_only_single_image", bundle["ap_only"] & (bundle["image_counts"] == 1)),
                       ("ap_only_multiple_image", bundle["ap_only"] & (bundle["image_counts"] > 1))):
        y, p = bundle["targets"][mask], bundle[mode][mask]
        aucs = class_aucs(y, p)
        scores[name] = {"num_studies": len(y), "positive_studies": y.sum(axis=0).astype(int).tolist(),
                        "negative_studies": (len(y) - y.sum(axis=0)).astype(int).tolist(),
                        "per_class_auc": aucs, "eligible_classes": sum(x is not None for x in aucs),
                        "macro_auc": float(np.mean(aucs)) if all(x is not None for x in aucs) else None}
    return scores


def _interval(values):
    finite = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return {"ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "valid_resamples": int(len(finite))}


def paired_bootstrap(bundle, reference, candidate, *, samples=1000, seed=20261103):
    """Pair patient multiplicities, retaining all their Studies and tie credit."""
    if samples < 1:
        raise ValueError("At least one patient bootstrap resample is required")
    y = bundle["targets"]
    a, b = (checked_probabilities(p, y.shape) for p in (reference, candidate))
    patients, inverse = np.unique(bundle["subject_ids"], return_inverse=True)
    if not len(patients):
        raise ValueError("Patient bootstrap requires nonempty patients")
    # Identical patient draws for all slices and all contrasts at a fixed seed.
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(patients), size=len(patients)),
                                   minlength=len(patients))[inverse] for _ in range(samples)])
    slices = {}
    for name, mask in (("all", np.ones(len(y), dtype=bool)), ("ap_only", bundle["ap_only"])):
        yy, aa, bb = y[mask], a[mask], b[mask]
        ac, bc = class_aucs(yy, aa), class_aucs(yy, bb)
        eligible = [c for c in range(10) if ac[c] is not None and bc[c] is not None]
        point = [None if c not in eligible else bc[c] - ac[c] for c in range(10)]
        draws_a, draws_b, draws_delta = [], [], []
        per_class = [[] for _ in range(10)]
        if len(yy):
            plans_a = [_auc_plan(yy[:, c], aa[:, c]) for c in range(10)]
            plans_b = [_auc_plan(yy[:, c], bb[:, c]) for c in range(10)]
            for weight in weights[:, mask]:
                av = [_weighted_auc(plans_a[c], weight) for c in range(10)]
                bv = [_weighted_auc(plans_b[c], weight) for c in range(10)]
                for c in eligible:
                    if av[c] is not None and bv[c] is not None:
                        per_class[c].append(bv[c] - av[c])
                if eligible and all(av[c] is not None and bv[c] is not None for c in eligible):
                    ma, mb = (float(np.mean([v[c] for c in eligible])) for v in (av, bv))
                    draws_a.append(ma)
                    draws_b.append(mb)
                    draws_delta.append(mb - ma)
        slices[name] = {"reference_macro_auc": float(np.mean([ac[c] for c in eligible])) if eligible else None,
                        "candidate_macro_auc": float(np.mean([bc[c] for c in eligible])) if eligible else None,
                        "macro_auc_delta": float(np.mean([point[c] for c in eligible])) if eligible else None,
                        "reference_macro": _interval(draws_a), "candidate_macro": _interval(draws_b),
                        "delta_macro": _interval(draws_delta), "eligible_classes": [c + 1 for c in eligible],
                        "per_class": [{"class_id": c + 1, "reference_auc": ac[c], "candidate_auc": bc[c],
                                       "delta": point[c], "positive_studies": int(yy[:, c].sum()),
                                       "negative_studies": int(len(yy) - yy[:, c].sum()),
                                       **_interval(per_class[c])} for c in range(10)]}
    return {"sampling_unit": "patient_with_all_Studies", "paired": True, "samples": samples,
            "seed": seed, "slices": slices,
            "limitation": "Conditional on chosen checkpoints; does not remove repeated-valid selection or training-seed uncertainty."}


def sensitivity_map(high_comparison=None, low_comparison=None):
    """Claims concern TRAIN size with common inference768, never disease names."""
    result = []
    for c in range(10):
        high = high_comparison["slices"]["all"]["per_class"][c] if high_comparison else None
        low = low_comparison["slices"]["all"]["per_class"][c] if low_comparison else None
        positive = (high or low or {}).get("positive_studies")
        required = .03 if positive is not None and positive < 30 else .01
        high_supported = bool(high and high["delta"] is not None and high["delta"] >= required
                              and high["ci95"] and high["ci95"][0] > 0)
        low_noninferior = bool(low and low["ci95"] and low["ci95"][0] >= -.01)
        result.append({"class_id": c + 1, "positive_studies": positive,
                       "high_resolution_required_delta": required,
                       "train896_benefit_at_infer768": high_supported,
                       "train640_noninferior_at_infer768_margin_0_01": low_noninferior,
                       "status": ("higher_training_resolution_supported" if high_supported else
                                  "lower_training_resolution_noninferior_within_margin" if low_noninferior else
                                  "inconclusive_or_noise"),
                       "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"})
    return result


def validate_checkpoint_metadata(checkpoint, *, valid_sha, roi=None, mil=None):
    """Reject incomplete EMA or silent changes to the trained preprocessing/head."""
    if (checkpoint.get("artifact_type") not in {"kd_student", "student_ema_snapshot"}
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True
            or checkpoint.get("class_ids") != CLASS_IDS
            or checkpoint.get("official_valid_manifest_sha256") != valid_sha
            or checkpoint.get("image_size") not in {640, 768, 896}
            or type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1):
        raise ValueError("A validated Tiny EMA with exact official-valid binding is required")
    experiment = checkpoint.get("experiment", {})
    training = checkpoint.get("training_config", {})
    declared_roi = bool(experiment.get("roi", training.get("roi", False)))
    declared_mil = bool(experiment.get("mil", training.get("mil", False)))
    if (roi is not None and roi != declared_roi or mil is not None and mil != declared_mil):
        raise ValueError("CLI ROI/MIL differs from checkpoint metadata")
    if declared_roi and experiment.get("roi_version") != ROI_VERSION:
        raise ValueError("Unrecognized or missing trained ROI version")
    if declared_mil and experiment.get("spatial_pooling") != MIL_VERSION:
        raise ValueError("Unrecognized or missing trained spatial pooling version")
    validation_size = experiment.get("inference_size", training.get("validation_size") or
                                      checkpoint.get("preprocessing", {}).get("validation_size", 768))
    if validation_size != 768:
        raise ValueError("This exploration protocol fixes inference to 768")
    model_config = checkpoint.get("model_config", {})
    if model_config.get("backbone_name") != "convnext_tiny.fb_in22k_ft_in1k_384":
        raise ValueError("Exploration evaluator only accepts declared ConvNeXt V1 Tiny")
    if not isinstance(checkpoint.get("model_state_dict"), dict) or not checkpoint["model_state_dict"]:
        raise ValueError("Missing EMA tensor state")
    return {"roi": declared_roi, "mil": declared_mil, "roi_version": ROI_VERSION if declared_roi else None,
            "spatial_pooling": MIL_VERSION if declared_mil else "gap", "inference_size": 768,
            "tta": "hflip2", "training_size": checkpoint["image_size"]}


class ExplorationPredictor:
    """FP16, serial image forwards; share decoded image and same-transform inputs."""
    def __init__(self, checkpoint_path, *, expected_sha, roi=None, mil=None):
        import torch
        from inference_reproducibility import configure_inference
        from inference_inputs import ExactU8Input
        from model_overnight import build_overnight_model, validate_model_config
        self.torch = torch
        self.path = Path(checkpoint_path).resolve(strict=True)
        before = sha256(self.path)
        if expected_sha != before:
            raise ValueError("Checkpoint SHA differs from the registered expected identity")
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        self.protocol = validate_checkpoint_metadata(checkpoint, valid_sha=sha256(VALID), roi=roi, mil=mil)
        if sha256(self.path) != before:
            raise RuntimeError("Checkpoint changed while loading")
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit FP16 deployment evaluation requires CUDA")
        configure_inference(torch_module=torch)
        torch.set_num_threads(1)
        config = {**validate_model_config(checkpoint["model_config"]), "gradient_checkpointing": False}
        model = build_overnight_model(pretrained=False, **config)
        if self.protocol["mil"]:
            from spatial_mil import install_spatial_mil
            install_spatial_mil(model)
        state = {}
        for key, value in checkpoint["model_state_dict"].items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("EMA state must contain tensors only")
            value = value.detach().cpu()
            if value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite source EMA tensor")
                value = value.half()
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Source EMA overflows FP16")
            state[key] = value
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False).to("cuda:0", dtype=torch.float16,
                                                              memory_format=torch.channels_last)
        self.lookup = ExactU8Input(torch, "cuda:0", (.485, .456, .406), (.229, .224, .225), image_sizes=[768])
        self.state = state
        kind = "initialization_reference" if checkpoint.get("artifact_type") == "student_ema_snapshot" else (
            f"fixed_final_E{checkpoint['epoch']}" if self.path.name == "last.pt" else "valid_selected_best")
        self.metadata = {"source_checkpoint": str(self.path), "source_checkpoint_sha256": before,
                         "source_epoch": checkpoint["epoch"], "selection_kind": kind,
                         "model_config": config, "experiment": self.protocol, "precision": "float16",
                         "valid_manifest_sha256": sha256(VALID), "class_ids": CLASS_IDS,
                         "train_manifest_sha256": checkpoint.get("source_manifest_sha256"),
                         "fold_plan_sha256": checkpoint.get("fold_plan_sha256"),
                         "student_warm_start": checkpoint.get("student_warm_start"),
                         "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                         "script_sha256": sha256(Path(__file__)), "official_gate": False,
                         "test_accessed": False, "trained_weights_modified": False}
        self.metadata["source_code_sha256"] = {str(ROOT / name): sha256(ROOT / name) for name in
            ["dataset.py", "inference_inputs.py", "model_overnight.py"] +
            (["roi_scale.py"] if self.protocol["roi"] else []) + (["spatial_mil.py"] if self.protocol["mil"] else [])}
        ledger = self.path.parent / "compliance/events.jsonl"
        if ledger.is_file():
            from training_compliance import read_events
            events = read_events(ledger)
            self.metadata["training_compliance"] = {"events_path": str(ledger), "events_sha256": sha256(ledger),
                                                     "event_count": len(events), "chain_verified": True,
                                                     "last_event": events[-1]["event_type"] if events else None}
        else:
            self.metadata["training_compliance"] = {"status": "historical checkpoint; no lifecycle hook ledger in its directory"}
        del checkpoint
        gc.collect()

    @property
    def preprocessing_key(self):
        return (self.protocol["roi"], self.protocol["roi_version"], 768)

    def prepare(self, image):
        from dataset import letterbox_rgb
        if self.protocol["roi"]:
            from roi_scale import normalize_roi
            pixels = normalize_roi(image, image_size=768)
            if isinstance(pixels, tuple):
                pixels = pixels[0]
            pixels = np.asarray(pixels, dtype=np.uint8)
        else:
            pixels = np.asarray(letterbox_rgb(image.convert("RGB"), 768), dtype=np.uint8)
        if pixels.shape not in ((768, 768), (768, 768, 3)):
            raise ValueError("Preprocessing must return an aspect-preserving 768 uint8 canvas")
        return self.lookup.to_tensor(pixels)

    def forward(self, tensor, *, tta=True):
        torch = self.torch
        tensors = [tensor]
        if tta:
            tensors.append(torch.flip(tensor, dims=(-1,)).contiguous(memory_format=torch.channels_last))
        predictions = []
        with torch.inference_mode():
            for item in tensors:
                logits = self.model(item)
                if tuple(logits.shape) != (1, 10) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite or wrong-shaped ten-class model output")
                predictions.append(torch.sigmoid(logits.float())[0])
            return torch.stack(predictions).cpu().numpy().astype(np.float64)

    def export(self, output):
        self.torch.save(self.state, output / "state_fp16.pt")
        self.metadata["export_weights_sha256"] = sha256(output / "state_fp16.pt")
        self.metadata["export_weights_bytes"] = (output / "state_fp16.pt").stat().st_size
        del self.state


def decode_record(row):
    from PIL import Image, ImageOps
    source = Path(row["path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != row["file_sha256"]:
        raise ValueError("Original image SHA differs: " + row["image_id"])
    with Image.open(BytesIO(source)) as opened:
        opened.load()
        upright = ImageOps.exif_transpose(opened)
        if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
            raise ValueError("Non-8-bit image requires a registered intensity policy")
        return upright.copy()


def predict_image_members(row, predictors, *, tta=True):
    image = decode_record(row)
    prepared = {}
    values = []
    for predictor in predictors:
        key = predictor.preprocessing_key
        if key not in prepared:
            prepared[key] = predictor.prepare(image)
        values.append(predictor.forward(prepared[key], tta=tta))
    return values


def official_records(image_root, *, split="valid"):
    if split not in ("train", "valid"):
        raise ValueError("No test split is supported")
    from deploy_kd import prediction_records
    records = prediction_records(image_root, VALID if split == "valid" else TRAIN, split=split)
    if split == "valid" and (len(records) != 1422 or len({r["study_key"] for r in records}) != 859
                             or len({r["subject_id"] for r in records}) != 228):
        raise ValueError("Official valid must retain 1422 images / 859 Studies / 228 patients")
    return records


def save_predictions(output, records, per_view):
    bundle = aggregate_predictions(records, per_view)
    np.savez_compressed(output / "predictions.npz", **bundle)
    with (output / "image_view_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id", "Image_id", "TTA_view", "Predict_class", "Probability"])
        for row, values in zip(records, per_view):
            for name, probability in zip(("identity", "hflip"), values):
                for c, p in enumerate(probability, 1):
                    writer.writerow([row["subject_id"], row["study_id"], row["image_id"], name, c, repr(float(p))])
    with (output / "study_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id", "Mode", "Predict_class", "Probability"])
        for i in range(len(bundle["study_keys"])):
            for mode in MODES:
                for c, p in enumerate(bundle[mode][i], 1):
                    writer.writerow([bundle["subject_ids"][i], bundle["study_ids"][i], mode, c, repr(float(p))])
    return bundle


def timing_selection(records, *, count=30, seed=20261103):
    grouped = OrderedDict()
    for row in records:
        if row.get("split") != "valid":
            raise ValueError("Study timing accepts official valid only")
        grouped.setdefault(row["study_key"], []).append(row)
    eligible = [key for key, rows in grouped.items() if len(rows) == 3]
    eligible.sort(key=lambda key: hashlib.sha256(f"{seed}/{key}".encode()).hexdigest())
    if len(eligible) < count + 5:
        raise ValueError("Need 30 distinct three-image valid Studies plus five disjoint warmups")
    return [grouped[k] for k in eligible[5:count + 5]], [grouped[k] for k in eligible[:5]]


def time_predictors(predictors, records, output, *, modes=("hflip2",), rounds=3):
    """Serial 3-image Study read through CSV flush, no cross-Study prefetch."""
    output.mkdir(parents=True, exist_ok=False)
    measured, warmups = timing_selection(records)
    torch = predictors[0].torch

    def study_prediction(rows, tta):
        image_values = []
        for row in rows:
            members = predict_image_members(row, predictors, tta=tta)
            member_means = [v.astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64) for v in members]
            image_values.append(np.mean(member_means, axis=0))
        return np.mean(image_values, axis=0)

    results = {}
    samples = []
    for mode in modes:
        for rows in warmups:
            study_prediction(rows, mode == "hflip2")
        torch.cuda.synchronize()
        observed = []
        with (output / f"{mode}_study_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", "Subject_id", "Study_id", "Predict_class", "Probability"])
            stream.flush()
            for round_index in range(1, rounds + 1):
                for rows in measured:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    probability = study_prediction(rows, mode == "hflip2")
                    for c, value in enumerate(probability, 1):
                        writer.writerow([round_index, rows[0]["subject_id"], rows[0]["study_id"], c, repr(float(value))])
                    stream.flush()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    observed.append(elapsed)
                    samples.append({"mode": mode, "round": round_index, "study_key": rows[0]["study_key"],
                                    "num_images": 3, "milliseconds": elapsed})
        results[mode] = {"mean_ms": float(np.mean(observed)), "p50_ms": float(np.quantile(observed, .5)),
                         "p95_ms": float(np.quantile(observed, .95)), "max_ms": float(np.max(observed)),
                         "observations": len(observed), "above_100ms": int(np.count_nonzero(np.asarray(observed) > 100)),
                         "image_model_forwards_per_study": len(predictors) * 3 * (2 if mode == "hflip2" else 1)}
    report = {"protocol": "30 distinct official-valid three-image Studies x 3 rounds; serial images; read/hash/decode/preprocess/H2D/forward/mean/CSV flush",
              "gpu_name": torch.cuda.get_device_name(0), "official_gate": False,
              "warmup_studies": [r[0]["study_key"] for r in warmups],
              "measured_studies": [r[0]["study_key"] for r in measured],
              "shared_decode": True, "shared_preprocessing_for_equal_policy": True,
              "model_count": len(predictors), "probability_weights": [1 / len(predictors)] * len(predictors),
              "modes": results, "samples": samples,
              "limits": "2080Ti developer timing is not T4 certification; max is retained for strict per-Study constraint."}
    write_json(output / "timing.json", report)
    return report


def history_note(checkpoint):
    history = Path(checkpoint).parent / "history.jsonl"
    if not history.is_file():
        return {"available": False, "train_valid_auc_gap": None}
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    return {"available": True, "history_path": str(history), "history_sha256": sha256(history),
            "epochs": [{"epoch": row.get("epoch"), "training": row.get("training"),
                        "validation": row.get("validation")} for row in rows],
            "train_valid_auc_gap": None,
            "warning": "Training weighted BCE/KL loss is not comparable to validation AUC or unweighted BCE; no AUC gap inferred from loss."}


def evaluate(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output must be new; never overwrite prior scores")
    records = official_records(args.image_root)
    expected_sha = args.checkpoint_sha256 or sha256(args.checkpoint)
    # When SHA is omitted, explicitly record that the source was bound at invocation.
    predictor = ExplorationPredictor(args.checkpoint, expected_sha=expected_sha, roi=args.roi, mil=args.mil)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "evaluation_started.json", {"started_utc": utc_now(), "command": sys.argv,
        "source_checkpoint_sha256": expected_sha, "sha_supplied_by_caller": args.checkpoint_sha256 is not None,
        "test_accessed": False})
    try:
        predictor.export(output)
        per_view = []
        for i, row in enumerate(records, 1):
            per_view.append(predict_image_members(row, [predictor])[0])
            if i % 100 == 0 or i == len(records):
                print(f"[exploration valid] {i}/{len(records)}", flush=True)
        bundle = save_predictions(output, records, np.stack(per_view))
        report = {"complete": True, "num_images": len(records), "num_studies": len(bundle["study_keys"]),
                  "num_patients": len(np.unique(bundle["subject_ids"])),
                  "scores": {mode: score_predictions(bundle, mode) for mode in MODES},
                  "history": history_note(args.checkpoint), "sampled_train_gap": None,
                  "timing": None, "test_accessed": False, "official_gate": False}
        if args.train_sample_studies:
            train_records = official_records(args.image_root, split="train")
            if sha256(TRAIN) != predictor.metadata["train_manifest_sha256"]:
                raise ValueError("Train manifest differs from checkpoint ancestry")
            keys = sorted({r["study_key"] for r in train_records},
                          key=lambda k: hashlib.sha256(("sample512/20261103/" + k).encode()).hexdigest())
            selected = set(keys[:args.train_sample_studies])
            sampled = [r for r in train_records if r["study_key"] in selected]
            train_views = [predict_image_members(row, [predictor])[0] for row in sampled]
            train_bundle = aggregate_predictions(sampled, np.stack(train_views))
            train_scores = score_predictions(train_bundle)
            full_score = report["scores"]["hflip2"]["all"]["macro_auc"]
            sample_score = train_scores["all"]["macro_auc"]
            report["sampled_train_gap"] = {"sampling": "deterministic Study hash; all selected Study images",
                "studies": len(selected), "image_count": len(sampled), "scores": train_scores,
                "sampled_train_minus_full_valid_macro_auc": sample_score - full_score if sample_score is not None else None,
                "warning": "Descriptive sampled-train AUC gap, not full-train gap; training subjects were seen during fitting."}
            np.savez_compressed(output / "sampled_train_predictions.npz", **train_bundle)
        if args.timing:
            report["timing"] = time_predictors([predictor], records, output / "timing")
        predictor.metadata["prediction_files_sha256"] = {name: sha256(output / name) for name in
            ["predictions.npz", "image_view_predictions.csv", "study_predictions.csv"]}
        write_json(output / "metadata.json", predictor.metadata)
        report["metadata_sha256"] = sha256(output / "metadata.json")
        report["completed_utc"] = utc_now()
        write_json(output / "report.json", report)
        print(json.dumps({"output": str(output), "scores": report["scores"], "timing": report["timing"] and report["timing"]["modes"]}, indent=2))
        return report
    except BaseException as error:
        write_json(output / "evaluation_failed.json", {"failed_utc": utc_now(), "error": f"{type(error).__name__}: {error}",
                                                       "test_accessed": False, "complete": False})
        raise


def load_evaluation(path):
    path = Path(path).resolve(strict=True)
    report = json.loads((path / "report.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    if not report.get("complete") or report.get("metadata_sha256") != sha256(path / "metadata.json"):
        raise ValueError("Evaluation metadata identity/complete status differs")
    for name, digest in metadata["prediction_files_sha256"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError("Saved predictions changed")
    with np.load(path / "predictions.npz", allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    for mode in MODES:
        checked_probabilities(bundle[mode], bundle["targets"].shape)
    return bundle, metadata, report


def aligned_bundles(bundles):
    names = list(bundles)
    reference = bundles[names[0]]
    fields = ("study_keys", "subject_ids", "study_ids", "targets", "ap_only", "image_counts", "image_ids", "image_study_keys")
    for name in names[1:]:
        for field in fields:
            if not np.array_equal(reference[field], bundles[name][field]):
                raise ValueError(f"Candidate {name} differs in original valid identity/order: {field}")
    return reference


def _markdown(report):
    lines = ["# Exploration comparison", "", "All metrics are official-valid exploration results; no test access or platform submission.", "",
             "| Candidate | Overall AUC | AP-only AUC | Three-image P50/P95 ms |", "| --- | ---: | ---: | --- |"]
    for name, value in report["candidates"].items():
        score = value["scores"]
        timing = value.get("timing")
        ms = f"{timing['p50_ms']:.2f}/{timing['p95_ms']:.2f}" if timing else "unmeasured"
        def fmt(v):
            return "NA" if v is None else f"{v:.6f}"
        lines.append(f"| {name} | {fmt(score['all']['macro_auc'])} | {fmt(score['ap_only']['macro_auc'])} | {ms} |")
    lines += ["", "Intervals resample patients with all their Studies. They are conditional on selected checkpoints and do not remove repeated-validation selection bias.",
              "", "Resolution evidence concerns training size at common 768 inference; disease-specific inference-size recommendations remain untested.", ""]
    for name, comparison in report["comparisons"].items():
        s = comparison["slices"]["all"]
        lines.append(f"- {name}: delta {s['macro_auc_delta']:+.6f}; patient 95% CI {s['delta_macro']['ci95']}.")
    return "\n".join(lines) + "\n"


def compare(args):
    inputs = OrderedDict()
    for item in args.inputs:
        if "=" not in item:
            raise ValueError("Inputs use name=evaluation_directory")
        name, path = item.split("=", 1)
        if not name or name in inputs or not all(c.isalnum() or c == "_" for c in name):
            raise ValueError("Candidate names must be unique alphanumeric/underscore names")
        inputs[name] = Path(path)
    if len(inputs) < 2:
        raise ValueError("At least two completed evaluations are needed")
    bundles, metadata, original = {}, {}, {}
    for name, path in inputs.items():
        bundles[name], metadata[name], original[name] = load_evaluation(path)
    reference = aligned_bundles(bundles)
    if any(m["valid_manifest_sha256"] != next(iter(metadata.values()))["valid_manifest_sha256"] for m in metadata.values()):
        raise ValueError("Candidates bind different official-valid manifests")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Comparison output must be new")
    output.mkdir(parents=True)
    candidates = {}
    arrays = {}
    for name, bundle in bundles.items():
        for mode in MODES:
            key = name + "_" + mode
            arrays[key] = bundle[mode]
            timing = (original[name].get("timing") or {}).get("modes", {}).get(mode)
            candidates[key] = {"scores": score_predictions(bundle, mode), "members": [name], "probability_weights": [1.],
                               "mode": mode, "timing": timing,
                               "selection_kind": metadata[name]["selection_kind"]}
    pairs = [("res768", "res896"), ("res768", "roi")]
    fusions = []
    for left, right in pairs:
        if left in bundles and right in bundles:
            fusion_name = f"{left}_{right}_equal"
            fusions.append((fusion_name, left, right))
            for mode in MODES:
                key = fusion_name + "_" + mode
                # Study mean and fixed model probability mean commute in FP64.
                arrays[key] = .5 * (bundles[left][mode] + bundles[right][mode])
                candidates[key] = {"scores": score_predictions({**reference, mode: arrays[key]}, mode),
                    "members": [left, right], "probability_weights": [.5, .5], "mode": mode, "timing": None,
                    "selection_kind": [metadata[left]["selection_kind"], metadata[right]["selection_kind"]]}
    contrasts = OrderedDict()
    baseline = "res768" if "res768" in bundles else next(iter(bundles))
    baseline_key = baseline + "_hflip2"
    for name in arrays:
        if name != baseline_key:
            contrasts[name + "_minus_" + baseline_key] = (baseline_key, name)
    for name in bundles:
        contrasts[name + "_hflip2_minus_identity"] = (name + "_identity", name + "_hflip2")
    for fusion_name, left, right in fusions:
        for mode in MODES:
            for member in (left, right):
                contrasts[fusion_name + "_" + mode + "_minus_" + member + "_" + mode] = (member + "_" + mode, fusion_name + "_" + mode)
    comparisons = {name: paired_bootstrap(reference, arrays[a], arrays[b], samples=args.bootstrap, seed=args.seed)
                   for name, (a, b) in contrasts.items()}
    high = comparisons.get("res896_hflip2_minus_res768_hflip2")
    low = comparisons.get("res640_hflip2_minus_res768_hflip2")
    if args.timing_fusions:
        records = official_records(args.image_root)
        for fusion_name, left, right in fusions:
            predictors = [ExplorationPredictor(metadata[n]["source_checkpoint"],
                expected_sha=metadata[n]["source_checkpoint_sha256"]) for n in (left, right)]
            timing = time_predictors(predictors, records, output / ("timing_" + fusion_name), modes=MODES)
            for mode in MODES:
                candidates[fusion_name + "_" + mode]["timing"] = timing["modes"][mode]
            del predictors
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    report = {"created_utc": utc_now(), "complete": True, "baseline": baseline_key,
              "inputs": {name: {"directory": str(inputs[name].resolve()), "metadata_sha256": sha256(inputs[name] / "metadata.json"),
                                 "source_checkpoint_sha256": metadata[name]["source_checkpoint_sha256"]} for name in inputs},
              "num_images": len(reference["image_ids"]), "num_studies": len(reference["study_keys"]),
              "num_patients": len(np.unique(reference["subject_ids"])), "candidates": candidates,
              "comparisons": comparisons, "resolution_sensitivity": sensitivity_map(high, low),
              "fusion_policy": "Only fixed equal res768+res896 and res768+roi; no coefficient fitting or candidate-weight search",
              "test_accessed": False, "platform_submission": False, "official_gate": False,
              "script_sha256": sha256(Path(__file__)),
              "limits": ["Inference resolution fixed768; this evaluates effects of training resolution.",
                         "CI includes0 means inconclusive/noise, not equivalence.",
                         "640 noninferiority is relative to a declared 0.01 AUC margin, not exact equivalence.",
                         "Use fixed-final-E3 contrasts for attribution; best checkpoints are supplementary valid-selected candidates."]}
    np.savez_compressed(output / "candidate_study_probabilities.npz", **arrays)
    report["candidate_probabilities_sha256"] = sha256(output / "candidate_study_probabilities.npz")
    write_json(output / "comparison.json", report)
    (output / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "candidate_auc": {name: c["scores"]["all"]["macro_auc"] for name, c in candidates.items()}}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="FP16 full-valid inference at 768, identity+hflip cached")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint-sha256", help="Expected immutable SHA; recommended for every scheduled invocation")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    evaluate_parser.add_argument("--roi", action="store_true", default=None, help="Require checkpoint-declared ROI")
    evaluate_parser.add_argument("--mil", action="store_true", default=None, help="Require checkpoint-declared MIL")
    evaluate_parser.add_argument("--timing", action="store_true", help="Also time 30 three-image valid Studies x3, hflip2")
    evaluate_parser.add_argument("--train-sample-studies", type=int, choices=(0, 512), default=0)
    compare_parser = sub.add_parser("compare", help="CPU paired patient bootstrap and fixed fusions")
    compare_parser.add_argument("--inputs", nargs="+", required=True, help="name=directory; canonical names res640,res768,res896,roi,mil,champion041")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--bootstrap", type=int, default=1000)
    compare_parser.add_argument("--seed", type=int, default=20261103)
    compare_parser.add_argument("--timing-fusions", action="store_true", help="Explicit GPU timing of fixed pairs; otherwise CPU only")
    compare_parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    args = parser.parse_args(argv)
    return evaluate(args) if args.action == "evaluate" else compare(args)


if __name__ == "__main__":
    main()
