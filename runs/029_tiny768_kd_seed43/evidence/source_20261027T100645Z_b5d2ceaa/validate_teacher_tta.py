"""Compare five frozen final-EMA teachers with identity and fixed hflip TTA.

Official valid only; never creates train soft labels, selects a teacher checkpoint,
changes weights, or runs without --execute. Models are loaded one at a time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from distill_data import sha256_file
from teacher_collection import read_teacher_collection, validate_collection_checkpoint
from validate_teacher import (active_training_processes, official_binding, write_json)

PROJECT = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT / "data/cv5_overnight/folds.json"
DEFAULT_MANIFEST = PROJECT / "data/clean_v1_reviewed/valid_manifest.csv"
DEFAULT_COLLECTION = PROJECT.parent / "runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_collection.json"
MODES = ("identity", "hflip2")


def validation_binding(manifest: str | Path, fold_plan: str | Path) -> dict[str, Any]:
    """Require both exact registered bytes and the actual official-holdout path."""
    binding = official_binding(manifest, fold_plan)
    declared = Path(binding["plan"]["official_holdout"]["path"]).resolve(strict=True)
    if Path(binding["manifest"]).resolve(strict=True) != declared:
        raise ValueError("Teacher TTA validation only accepts the fold plan's official holdout path")
    return binding


def completed_teacher_collection(path: str | Path, fold_plan: str | Path) -> dict[str, Any]:
    """Small-file completion guard, before any checkpoint or CUDA initialization."""
    collection = read_teacher_collection(path, fold_plan_path=fold_plan, require_checkpoints=True)
    for entry in collection["teachers"]:
        status_file = Path(entry["checkpoint"]).parent / "status.json"
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if (status.get("state") != "completed" or status.get("training_complete") is not True
                or status.get("inference_ready") is not True
                or status.get("completed_epochs") != entry["epochs"]):
            raise ValueError(f"Teacher fold {entry['fold_index']} has not completed its fixed horizon")
    transition = Path(collection["collection_path"]).parent / "transition_status.json"
    if transition.is_file():
        state = json.loads(transition.read_text(encoding="utf-8"))
        if state.get("phase") != "completed" or state.get("all_teachers_complete") is not True:
            raise ValueError("Teacher collection controller has not finished; wait before GPU diagnostics")
    return collection


def require_idle_gpu() -> None:
    """Fail closed; CPU-stage trainers and any existing compute process block validation."""
    if active_training_processes():
        raise RuntimeError("Teacher TTA validation refuses to overlap an active training process")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify that the GPU is idle; validation was not started") from exc
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    if pids:
        raise RuntimeError("GPU already has compute processes; validation was not started: " + ", ".join(pids))


def probability_sets(records: Sequence[dict[str, Any]], image_ids: Sequence[str],
                     raw_views: np.ndarray) -> dict[str, dict[str, Any]]:
    """Pure CPU math: T=1 raw sigmoid, view mean, then equal image mean per Study."""
    from dataset_overnight import aggregate_study_predictions

    raw_views = np.asarray(raw_views)
    if raw_views.shape != (len(image_ids), 2, 10) or not np.isfinite(raw_views).all():
        raise ValueError("Teacher comparison requires finite Nx2x10 raw view logits")
    probabilities = np.exp(-np.logaddexp(0., -raw_views.astype(np.float64)))
    return {
        "identity": aggregate_study_predictions(records, image_ids, probabilities[:, 0, :]),
        "hflip2": aggregate_study_predictions(records, image_ids, probabilities.mean(axis=1)),
    }


def equal_teacher_average(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("Teacher ensemble comparison requires exactly five teachers")
    first = values[0]
    for item in values[1:]:
        if (item["study_keys"] != first["study_keys"]
                or item["subject_ids"] != first["subject_ids"]
                or item["study_ids"] != first["study_ids"]
                or not np.array_equal(item["targets"], first["targets"])):
            raise ValueError("Teacher predictions do not have identical official-valid Study coverage")
    stacked = np.stack([np.asarray(item["probabilities"], dtype=np.float64) for item in values])
    if not np.isfinite(stacked).all() or (stacked < 0).any() or (stacked > 1).any():
        raise ValueError("Teacher ensemble members must contain finite probabilities in [0,1]")
    return {**first, "probabilities": stacked.mean(axis=0)}


def auc_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    targets, probabilities = np.asarray(values["targets"]), np.asarray(values["probabilities"])
    if (targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != 10
            or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or any(len(np.unique(targets[:, c])) != 2 for c in range(10))):
        raise ValueError("Official valid comparison requires all ten binary classes with both outcomes")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Official valid labels must remain binary")
    per_class = [float(roc_auc_score(targets[:, c], probabilities[:, c])) for c in range(10)]
    return {"macro_auc": float(np.mean(per_class)), "per_class_auc": per_class}


def save_predictions(output: Path, stem: str, values: Mapping[str, Any]) -> dict[str, str]:
    path = output / (stem + ".csv")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability"])
        for subject, study, probabilities in zip(values["subject_ids"], values["study_ids"], values["probabilities"]):
            for class_id, value in enumerate(probabilities, 1):
                writer.writerow([subject, study, class_id, format(float(value), ".17g")])
    npz = output / (stem + ".npz")
    with npz.open("xb") as stream:
        np.savez_compressed(stream, study_keys=np.asarray(values["study_keys"], dtype=str),
            subject_ids=np.asarray(values["subject_ids"], dtype=str),
            study_ids=np.asarray(values["study_ids"], dtype=str),
            targets=np.asarray(values["targets"], dtype=np.float32),
            probabilities=np.asarray(values["probabilities"], dtype=np.float64))
    return {"csv": path.name, "csv_sha256": sha256_file(path),
            "npz": npz.name, "npz_sha256": sha256_file(npz)}


def checked_output(output_dir: str | Path, image_root: str | Path) -> tuple[Path, Path]:
    """Apply read-only-data and no-overwrite guards before CLI logging can mkdir."""
    output, raw_root = Path(output_dir).resolve(), Path(image_root).resolve(strict=True)
    if output == raw_root or output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError("Validation output must not overlap the read-only image root")
    from audit_trail import training_output_has_artifacts
    if training_output_has_artifacts(output):
        raise FileExistsError("Choose a new teacher TTA validation output directory")
    return output, raw_root


def evaluate_teachers(*, teacher_collection: str | Path, fold_plan: str | Path,
                      manifest: str | Path, image_root: str | Path, output_dir: str | Path,
                      batch_size: int = 4, num_workers: int = 8, amp_dtype: str = "auto") -> dict[str, Any]:
    if (type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int
            or not 0 <= num_workers <= 12 or amp_dtype not in {"auto", "bfloat16", "float16"}):
        raise ValueError("Invalid teacher diagnostic batch/workers/precision")
    binding = validation_binding(manifest, fold_plan)
    collection = completed_teacher_collection(teacher_collection, fold_plan)
    require_idle_gpu()
    output, raw_root = checked_output(output_dir, image_root)
    # Validate every final checkpoint on CPU before touching the CUDA API.
    import torch
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from dataset_overnight import make_overnight_loader
    from extract_soft_labels_tta import _forward_raw_view_logits
    from model_overnight import build_overnight_model
    from kd_common import require_training_runtime

    require_training_runtime()
    plan = read_fold_plan(fold_plan)
    train_records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    records = read_manifest(manifest, split="valid")
    for field in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        if ({r.get(field) for r in train_records} - {None, ""}) & ({r.get(field) for r in records} - {None, ""}):
            raise ValueError("Teacher validation train/holdout leakage: " + field)
    sources = []
    for entry in collection["teachers"]:
        path = Path(entry["checkpoint"])
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        validate_collection_checkpoint(checkpoint, entry, plan=plan,
                                      plan_hash=binding["fold_plan_sha256"], records=train_records)
        if sha256_file(path) != digest:
            raise RuntimeError("Teacher checkpoint changed during CPU prevalidation")
        sources.append({"fold": entry["fold_index"], "checkpoint": str(path),
            "checkpoint_sha256": digest, "epoch": checkpoint["epoch"],
            "weights_source": "ema", "training_complete": True,
            "pretrained_provenance": checkpoint["pretrained_provenance"],
            "teacher_pos_weight": checkpoint["loss_config"]["pos_weight"]})
        del checkpoint
    require_idle_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher TTA validation requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    bf16 = torch.cuda.is_bf16_supported()
    if amp_dtype == "bfloat16" and not bf16:
        raise ValueError("This GPU does not support BF16")
    precision = "bfloat16" if amp_dtype == "bfloat16" or amp_dtype == "auto" and bf16 else "float16"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    output.mkdir(parents=True, exist_ok=True)
    from training_evidence_monitor import snapshot_source
    source_archive = snapshot_source(PROJECT, output / "evidence" / ("source_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    by_mode = {mode: [] for mode in MODES}
    results = []
    for entry, source in zip(collection["teachers"], sources):
        path = Path(entry["checkpoint"])
        if sha256_file(path) != source["checkpoint_sha256"]:
            raise RuntimeError("Teacher checkpoint changed after prevalidation")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        model = build_overnight_model(pretrained=False,
            **{**checkpoint["model_config"], "gradient_checkpointing": False})
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
        loader = make_overnight_loader(records, raw_root, training=False, image_size=768,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=1, seed=43)
        image_ids, raw_values = [], []
        try:
            with torch.inference_mode():
                for batch in loader:
                    if batch["split"] != "valid":
                        raise ValueError("Teacher TTA diagnostics only accept official-valid images")
                    inputs = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
                    raw_values.append(_forward_raw_view_logits(model, inputs, tta="identity_hflip",
                        context_factory=lambda: torch.autocast("cuda", dtype=dtype)))
                    image_ids.extend(batch["image_ids"])
            predictions = probability_sets(records, image_ids, np.concatenate(raw_values))
        finally:
            del loader, model
            gc.collect()
            torch.cuda.empty_cache()
        if sha256_file(path) != source["checkpoint_sha256"]:
            raise RuntimeError("Teacher checkpoint changed during validation")
        for mode in MODES:
            values = predictions[mode]
            by_mode[mode].append(values)
            result = {"member": f"fold_{entry['fold_index']}", "fold": entry["fold_index"],
                "tta": mode, **auc_metrics(values), "num_studies": len(values["study_keys"]),
                "predictions": save_predictions(output, f"fold_{entry['fold_index']}_{mode}", values)}
            results.append(result)
            print(f"Teacher fold={entry['fold_index']} tta={mode} MacroAUC={result['macro_auc']:.8f}", flush=True)
    for mode in MODES:
        values = equal_teacher_average(by_mode[mode])
        result = {"member": "five_teacher_equal_probability_average", "tta": mode,
            **auc_metrics(values), "num_studies": len(values["study_keys"]),
            "predictions": save_predictions(output, f"ensemble5_{mode}", values)}
        results.append(result)
        print(f"Teacher ensemble=5 tta={mode} MacroAUC={result['macro_auc']:.8f}", flush=True)
    if (sha256_file(manifest) != binding["manifest_sha256"]
            or sha256_file(plan["source_train_manifest"]["path"]) != plan["source_manifest_sha256"]
            or sha256_file(fold_plan) != binding["fold_plan_sha256"]
            or sha256_file(collection["collection_path"]) != collection["collection_sha256"]):
        raise RuntimeError("Teacher collection or registered validation inputs changed")
    report = {"complete": True, "diagnostic_only": True, "image_size": 768,
        "precision": precision, "class_ids": list(range(1, 11)), "views": ["identity", "hflip"],
        "probability_space": "sigmoid(raw_logits) at T=1; no pos_weight correction",
        "aggregation": "view_probability_mean_then_Study_image_mean_then_fixed_equal_teacher_mean",
        "num_images": len(records), "manifest_sha256": binding["manifest_sha256"],
        "fold_plan_sha256": binding["fold_plan_sha256"],
        "teacher_collection_sha256": collection["collection_sha256"], "teachers": sources,
        "source_archive": str(source_archive), "teacher_ensemble_weights": [0.2] * 5,
        "environment": {"gpu_name": torch.cuda.get_device_name(device), "torch": torch.__version__, "seed": 43},
        "results": results, "checkpoint_selection_performed": False, "tta_selection_performed": False,
        "thresholds_fitted": False, "test_used": False, "train_soft_labels_generated": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Independent official-valid diagnostics only. Five-teacher averaging is not valid strict-OOF train labeling."}
    write_json(output / "teacher_tta_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--fold-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        binding = validation_binding(args.manifest, args.fold_plan)
        collection = read_teacher_collection(args.teacher_collection, fold_plan_path=args.fold_plan)
        print(json.dumps({"dry_run": True, "checkpoints_loaded": 0, "gpu_used": False,
            "manifest": binding["manifest"], "manifest_sha256": binding["manifest_sha256"],
            "teachers": collection["teachers"], "comparison": ["each_fold_identity_vs_hflip2", "ensemble5_identity_vs_hflip2"],
            "output_dir": str(args.output_dir), "next": "Use --execute only after all five teachers finish and the GPU is idle."}, indent=2))
        return 0
    # Refuse active training or an incomplete collection before even creating a log directory.
    validation_binding(args.manifest, args.fold_plan)
    completed_teacher_collection(args.teacher_collection, args.fold_plan)
    require_idle_gpu()
    checked_output(args.output_dir, args.image_root)
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=args.output_dir / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])]))
    settings = vars(args)
    settings.pop("execute")
    print(json.dumps(evaluate_teachers(**settings), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
