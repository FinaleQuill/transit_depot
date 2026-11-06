"""Save six teachers' image/view raw logits and diagnose E13 versus E20.

This is official-valid diagnostics only. It never selects a checkpoint, changes
the frozen teacher collection, or creates train soft labels. Default is preview;
--execute runs GPU inference. --replay-dir plus --execute is CPU-only replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from distill_data import sha256_file
from teacher_collection import read_teacher_collection, validate_collection_checkpoint
from validate_teacher import validate_teacher_metadata, write_json
from validate_teacher_tta import (DEFAULT_COLLECTION, DEFAULT_MANIFEST, DEFAULT_PLAN,
    PROJECT, auc_metrics, checked_output, completed_teacher_collection,
    equal_teacher_average, require_idle_gpu, save_predictions, validation_binding)

DEFAULT_EPOCH13 = PROJECT.parent / "runs/021_convnext_v1_base768_teacher_cv5/fold_0/retained/epoch_013_last.pt"
RAW_SCHEMA = "cxr_official_valid_image_view_logits_v1"
VIEWS = ["identity", "hflip"]
POLICIES = ("raw_probability", "train_pos_weight_corrected_probability")
ARRAY_KEYS = {"image_ids", "subject_ids", "study_ids", "study_keys", "relative_paths",
              "source_image_sha256", "targets", "logits"}


def validate_raw_source(source: Mapping[str, Any]) -> None:
    name = source.get("id")
    expected = {"fold0_e20": (0, 20, 20, True), "fold0_e13": (0, 13, 20, False),
                **{f"fold{i}_e12": (i, 12, 12, True) for i in range(1, 5)}}
    if name not in expected:
        raise ValueError("Unknown raw teacher diagnostic source")
    fold, epoch, horizon, complete = expected[name]
    weights = np.asarray(source.get("teacher_pos_weight", []), dtype=np.float64)
    digest = source.get("checkpoint_sha256")
    if (source.get("fold") != fold or source.get("epoch") != epoch
            or source.get("fixed_horizon") != horizon or source.get("training_complete") is not complete
            or source.get("inference_ready") is not complete or source.get("weights_source") != "ema"
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or weights.shape != (10,) or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Raw teacher source epoch, completion, EMA, SHA or class weights differ")


def source_plan(collection: Mapping[str, Any], epoch13_checkpoint: str | Path) -> list[dict[str, Any]]:
    """Declare the real budgets; there is deliberately no invented all-E13 set."""
    entries = [{"id": f"fold{t['fold_index']}_e{t['epochs']}", "checkpoint": t["checkpoint"],
                "fold": t["fold_index"], "epoch": t["epochs"], "kind": "frozen_final",
                "declaration": dict(t)} for t in collection["teachers"]]
    if [t["epoch"] for t in entries] != [20, 12, 12, 12, 12]:
        raise ValueError("This diagnostic compares the frozen 20/12/12/12/12 collection")
    entries.append({"id": "fold0_e13", "checkpoint": str(Path(epoch13_checkpoint).resolve(strict=True)),
                    "fold": 0, "epoch": 13, "kind": "intermediate_diagnostic_only"})
    if len({str(Path(t["checkpoint"]).resolve()) for t in entries}) != 6:
        raise ValueError("E13 must be a separate retained checkpoint")
    return entries


def validate_epoch13(checkpoint: Mapping[str, Any], final: Mapping[str, Any], *,
                     plan: Mapping[str, Any], plan_hash: str, train_records: Sequence[dict]) -> dict:
    """Accept the true incomplete E13 metadata, never relabel it as final E20."""
    result = validate_teacher_metadata(checkpoint, plan=plan, plan_sha256=plan_hash, train_records=train_records)
    if (result["fold"] != 0 or result["epoch"] != 13 or result["fixed_horizon"] != 20
            or result["training_complete"] is not False or result["inference_ready"] is not False):
        raise ValueError("Retained source must be fold0 epoch13 of the original twenty-epoch run")
    invariant = ("epochs", "seed", "backbone_lr", "head_lr", "warmup_epochs", "min_lr_ratio",
                 "weight_decay", "dropout", "drop_path", "ema_decay", "grad_clip", "patience")
    first, second = checkpoint["training_config"], final["training_config"]
    if any(first.get(k) != second.get(k) for k in invariant):
        raise ValueError("E13 and E20 must come from the same frozen training configuration")
    if (first.get("batch_size", 0) * first.get("accumulation_steps", 0)
            != second.get("batch_size", 0) * second.get("accumulation_steps", 0)):
        raise ValueError("E13 and E20 effective batch sizes differ")
    for field in ("model_config", "preprocessing", "loss_config"):
        if checkpoint.get(field) != final.get(field):
            raise ValueError("E13 and E20 model, preprocessing or training weights differ")
    for field in ("weights_sha256", "revision", "repo_id"):
        if checkpoint["pretrained_provenance"].get(field) != final["pretrained_provenance"].get(field):
            raise ValueError("E13 and E20 have different original initialization")
    return result


def checked_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    ids = [r.get("image_id") for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("Raw diagnostic records require unique image identities")
    for row in rows:
        target = row.get("targets")
        digest = row.get("file_sha256")
        if (row.get("split") != "valid" or row.get("study_key") != f"{row.get('subject_id')}/{row.get('study_id')}"
                or not isinstance(row.get("relative_path"), str)
                or not row["relative_path"].startswith("valid/")
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or target is None or len(target) != 10 or any(v not in (0, 1) for v in target)):
            raise ValueError("Raw diagnostic archive requires original valid identities, hashes and labels")
    return rows


def save_raw_logits(output: Path, source: Mapping[str, Any], records: Sequence[dict], logits: np.ndarray,
                    *, binding: Mapping[str, Any], precision: str, collection_sha256: str) -> dict:
    rows = checked_records(records)
    validate_raw_source(source)
    raw = np.asarray(logits)
    if raw.shape != (len(rows), 2, 10) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Raw archive logits must be finite FP32 Nx2x10")
    folder = output / "raw_logits"
    folder.mkdir(exist_ok=True)
    name = source["id"]
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Source identifier must be a plain filename stem")
    path, metadata_path = folder / f"{name}.npz", folder / f"{name}.json"
    arrays = {key: np.asarray([r[field] for r in rows], dtype=str) for key, field in (
        ("image_ids", "image_id"), ("subject_ids", "subject_id"), ("study_ids", "study_id"),
        ("study_keys", "study_key"), ("relative_paths", "relative_path"), ("source_image_sha256", "file_sha256"))}
    arrays.update(targets=np.asarray([r["targets"] for r in rows], dtype=np.float32), logits=raw)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata = {"schema": RAW_SCHEMA, "source_split": "valid", "diagnostic_only": True,
        "complete": True, "output_type": "per_image_per_view_raw_logits", "class_ids": list(range(1, 11)),
        "views": VIEWS, "image_size": 768, "num_images": len(rows), "precision": precision,
        "temperature_applied": False, "pos_weight_correction_applied": False,
        "manifest_sha256": binding["manifest_sha256"], "fold_plan_sha256": binding["fold_plan_sha256"],
        "teacher_collection_sha256": collection_sha256, "source": dict(source),
        "npz_file": path.name, "npz_sha256": sha256_file(path)}
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"id": name, "npz": str(path.relative_to(output)), "metadata": str(metadata_path.relative_to(output)),
            "npz_sha256": metadata["npz_sha256"], "metadata_sha256": sha256_file(metadata_path)}


def load_raw_logits(directory: Path, entry: Mapping[str, Any]) -> tuple[list[dict], np.ndarray, dict]:
    root = directory.resolve(strict=True)
    paths = []
    for key in ("npz", "metadata"):
        value = Path(entry[key])
        path = root / value
        if value.is_absolute() or ".." in value.parts or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Raw archive reference escapes its diagnostic directory")
        if sha256_file(path) != entry[key + "_sha256"]:
            raise ValueError("Raw diagnostic file hash differs from the sealed index")
        paths.append(path)
    metadata = json.loads(paths[1].read_text())
    if (metadata.get("schema") != RAW_SCHEMA or metadata.get("source_split") != "valid"
            or metadata.get("diagnostic_only") is not True or metadata.get("complete") is not True
            or metadata.get("views") != VIEWS or metadata.get("output_type") != "per_image_per_view_raw_logits"
            or metadata.get("image_size") != 768 or metadata.get("class_ids") != list(range(1, 11))
            or metadata.get("precision") not in {"bfloat16", "float16"}
            or type(metadata.get("num_images")) is not int or metadata["num_images"] < 1
            or metadata.get("temperature_applied") is not False or metadata.get("pos_weight_correction_applied") is not False
            or metadata.get("npz_sha256") != entry["npz_sha256"] or metadata.get("source", {}).get("id") != entry["id"]):
        raise ValueError("Raw diagnostic metadata or view semantics differ")
    validate_raw_source(metadata["source"])
    with np.load(paths[0], allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError("Raw diagnostic array inventory differs")
        arrays = {name: archive[name] for name in ARRAY_KEYS}
    n = metadata["num_images"]
    if any(arrays[key].shape != (n,) or arrays[key].dtype.kind != "U" for key in ARRAY_KEYS - {"logits", "targets"}):
        raise ValueError("Raw diagnostic identity arrays must be one-dimensional strings")
    if arrays["targets"].shape != (n, 10) or not np.isin(arrays["targets"], [0, 1]).all():
        raise ValueError("Raw diagnostic targets must be original binary labels")
    logits = arrays["logits"]
    if logits.shape != (n, 2, 10) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Raw diagnostic logits must retain finite FP32 Nx2x10")
    rows = [{field: str(arrays[key][i]) for key, field in (
        ("image_ids", "image_id"), ("subject_ids", "subject_id"), ("study_ids", "study_id"),
        ("study_keys", "study_key"), ("relative_paths", "relative_path"), ("source_image_sha256", "file_sha256"))}
        for i in range(n)]
    for i, row in enumerate(rows):
        row.update(split="valid", targets=arrays["targets"][i].tolist())
    return checked_records(rows), logits, metadata


def probabilities_from_raw(records: Sequence[dict], logits: np.ndarray, pos_weights: Any,
                           *, correction: bool) -> dict[str, dict]:
    from dataset_overnight import aggregate_study_predictions
    rows = checked_records(records)
    raw = np.asarray(logits, dtype=np.float64)
    weights = np.asarray(pos_weights, dtype=np.float64)
    if (raw.shape != (len(rows), 2, 10) or not np.isfinite(raw).all()
            or weights.shape != (10,) or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Expected finite Nx2x10 logits and ten positive training class weights")
    if correction:
        raw = raw - np.log(weights)[None, None, :]
    probability = np.exp(-np.logaddexp(0., -raw))
    ids = [r["image_id"] for r in rows]
    return {"identity": aggregate_study_predictions(rows, ids, probability[:, 0]),
            "hflip2": aggregate_study_predictions(rows, ids, probability.mean(1))}


def replay_diagnostic(input_dir: str | Path, output_dir: str | Path) -> dict:
    """CPU-only deterministic re-aggregation; never touches image files or CUDA."""
    root, output = Path(input_dir).resolve(strict=True), Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Replay output must be outside its sealed input directory")
    if output.exists():
        raise FileExistsError("Choose a new replay output directory")
    index = json.loads((root / "raw_index.json").read_text())
    if index.get("image_root"):
        original = Path(index["image_root"]).resolve()
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("Replay output must not overlap the read-only image root")
    return report_from_raw(root, output, index, create_output=True)


def report_from_raw(input_dir: Path, output: Path, index: Mapping[str, Any], *, create_output=False) -> dict:
    expected_ids = ["fold0_e20", "fold1_e12", "fold2_e12", "fold3_e12", "fold4_e12", "fold0_e13"]
    if (index.get("schema") != RAW_SCHEMA or index.get("complete") is not True
            or [e.get("id") for e in index.get("sources", [])] != expected_ids):
        raise ValueError("Replay requires all six sealed original-valid raw-logit sources")
    if create_output:
        output.mkdir(parents=True)
    by_policy = {policy: {} for policy in POLICIES}
    reference_rows = None
    sources, results = [], []
    for entry in index["sources"]:
        rows, logits, metadata = load_raw_logits(input_dir, entry)
        if any(metadata.get(k) != index.get(k) for k in ("manifest_sha256", "fold_plan_sha256", "teacher_collection_sha256", "precision")):
            raise ValueError("Raw sources differ from the diagnostic input binding")
        if reference_rows is None:
            reference_rows = rows
        elif rows != reference_rows:
            raise ValueError("Raw sources have different image identities, hashes or labels")
        source = metadata["source"]
        sources.append(source)
        for policy in POLICIES:
            modes = probabilities_from_raw(rows, logits, source["teacher_pos_weight"],
                                           correction=policy != "raw_probability")
            by_policy[policy][entry["id"]] = modes
            for mode, values in modes.items():
                result = {"member": entry["id"], "policy": policy, "tta": mode,
                          **auc_metrics(values), "num_studies": len(values["study_keys"])}
                result["predictions"] = save_predictions(output, f"{entry['id']}_{policy}_{mode}", values)
                results.append(result)
    comparisons = []
    for policy, members in by_policy.items():
        for mode in ("identity", "hflip2"):
            ensembles = {}
            for name, first in (("A_20_12_12_12_12", "fold0_e20"), ("B_13_12_12_12_12", "fold0_e13")):
                ids = [first] + expected_ids[1:5]
                values = equal_teacher_average([members[member][mode] for member in ids])
                metrics = auc_metrics(values)
                ensembles[name] = metrics
                results.append({"member": name, "members": ids, "policy": policy, "tta": mode,
                    **metrics, "num_studies": len(values["study_keys"]),
                    "predictions": save_predictions(output, f"{name}_{policy}_{mode}", values)})
            a, b = ensembles["A_20_12_12_12_12"], ensembles["B_13_12_12_12_12"]
            comparisons.append({"policy": policy, "tta": mode,
                "A_macro_auc": a["macro_auc"], "B_macro_auc": b["macro_auc"],
                "B_minus_A_macro_auc": b["macro_auc"] - a["macro_auc"],
                "B_minus_A_per_class_auc": (np.asarray(b["per_class_auc"]) - a["per_class_auc"]).tolist()})
    report = {"complete": True, "diagnostic_only": True, "raw_schema": RAW_SCHEMA,
        "raw_index_sha256": sha256_file(input_dir / "raw_index.json"),
        "manifest_sha256": index["manifest_sha256"], "fold_plan_sha256": index["fold_plan_sha256"],
        "teacher_collection_sha256": index["teacher_collection_sha256"], "num_images": len(reference_rows),
        "precision": index["precision"], "views": VIEWS, "temperature": 1,
        "aggregation": "per_view_sigmoid_then_view_mean_then_Study_image_mean_then_equal_teacher_mean",
        "prior_correction": "candidate: subtract log(training pos_weight) per image/view before sigmoid; approximate calibration, no valid fitting",
        "sources": sources, "results": results, "A_vs_B": comparisons,
        "checkpoint_selection_performed": False, "teacher_collection_modified": False,
        "weights_modified": False, "train_soft_labels_generated": False, "thresholds_fitted": False,
        "test_used": False, "raw_logits_preserved": True,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "teacher_epoch_diagnostic_report.json", report)
    return report


def evaluate(*, teacher_collection: str | Path, epoch13_checkpoint: str | Path,
             fold_plan: str | Path, manifest: str | Path, image_root: str | Path,
             output_dir: str | Path, batch_size=4, num_workers=8, amp_dtype="auto") -> dict:
    if (type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int
            or not 0 <= num_workers <= 12 or amp_dtype not in {"auto", "bfloat16", "float16"}):
        raise ValueError("Invalid diagnostic batch/workers/precision")
    binding = validation_binding(manifest, fold_plan)
    collection = completed_teacher_collection(teacher_collection, fold_plan)
    entries = source_plan(collection, epoch13_checkpoint)
    require_idle_gpu()
    output, raw_root = checked_output(output_dir, image_root)
    import torch
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from dataset_overnight import make_overnight_loader
    from extract_soft_labels_tta import _forward_raw_view_logits
    from kd_common import require_training_runtime
    from model_overnight import build_overnight_model
    require_training_runtime()
    plan = read_fold_plan(fold_plan)
    training = read_manifest(plan["source_train_manifest"]["path"], split="train")
    records = checked_records(read_manifest(manifest, split="valid"))
    for key in ("subject_id", "study_key", "image_id", "file_sha256", "pixel_sha256"):
        if ({r.get(key) for r in training} - {None, ""}) & ({r.get(key) for r in records} - {None, ""}):
            raise ValueError("Teacher diagnostic train/valid leakage: " + key)
    sources, final0 = [], None
    for entry in entries:
        path = Path(entry["checkpoint"])
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if entry["kind"] == "frozen_final":
            validate_collection_checkpoint(checkpoint, entry["declaration"], plan=plan,
                plan_hash=binding["fold_plan_sha256"], records=training)
            metadata = validate_teacher_metadata(checkpoint, plan=plan,
                plan_sha256=binding["fold_plan_sha256"], train_records=training)
            if entry["fold"] == 0:
                final0 = {k: deepcopy(checkpoint[k]) for k in ("training_config", "model_config", "preprocessing", "loss_config", "pretrained_provenance")}
        else:
            metadata = validate_epoch13(checkpoint, final0, plan=plan,
                plan_hash=binding["fold_plan_sha256"], train_records=training)
        if sha256_file(path) != digest:
            raise RuntimeError("Teacher checkpoint changed during CPU validation")
        sources.append({"id": entry["id"], **metadata, "checkpoint": str(path.resolve()),
            "checkpoint_sha256": digest, "kind": entry["kind"], "model_config": checkpoint["model_config"],
            "preprocessing": checkpoint["preprocessing"], "pretrained_provenance": checkpoint["pretrained_provenance"],
            "teacher_pos_weight": checkpoint["loss_config"]["pos_weight"]})
        del checkpoint
    require_idle_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher epoch diagnostics requires CUDA")
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
    index = {"schema": RAW_SCHEMA, "complete": False, "diagnostic_only": True,
        "manifest_sha256": binding["manifest_sha256"], "fold_plan_sha256": binding["fold_plan_sha256"],
        "teacher_collection_sha256": collection["collection_sha256"], "precision": precision,
        "source_archive": str(source_archive), "image_root": str(raw_root), "sources": []}
    position = {r["image_id"]: i for i, r in enumerate(records)}
    for source in sources:
        path = Path(source["checkpoint"])
        if sha256_file(path) != source["checkpoint_sha256"]:
            raise RuntimeError("Teacher source changed after CPU prevalidation")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        model = build_overnight_model(pretrained=False,
            **{**checkpoint["model_config"], "gradient_checkpointing": False})
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
        loader = make_overnight_loader(records, raw_root, training=False, image_size=768,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=1, seed=43)
        values = np.empty((len(records), 2, 10), dtype=np.float32)
        seen = set()
        try:
            with torch.inference_mode():
                for batch in loader:
                    ids = batch["image_ids"]
                    if (batch["split"] != "valid" or len(set(ids)) != len(ids)
                            or set(ids) & seen or not set(ids) <= set(position)):
                        raise ValueError("Diagnostic batches must cover original valid exactly once")
                    inputs = batch["images"].to(device, memory_format=torch.channels_last, non_blocking=True)
                    raw = _forward_raw_view_logits(model, inputs, tta="identity_hflip",
                        context_factory=lambda: torch.autocast("cuda", dtype=dtype))
                    for i, image_id in enumerate(ids):
                        values[position[image_id]] = raw[i]
                    seen.update(ids)
            if seen != set(position):
                raise RuntimeError("Teacher diagnostic image coverage is incomplete")
        finally:
            del loader, model
            gc.collect()
            torch.cuda.empty_cache()
        if sha256_file(path) != source["checkpoint_sha256"]:
            raise RuntimeError("Teacher checkpoint changed during inference")
        index["sources"].append(save_raw_logits(output, source, records, values, binding=binding,
            precision=precision, collection_sha256=collection["collection_sha256"]))
        write_json(output / "raw_index.json", index)
        print(f"Saved raw logits source={source['id']} shape={values.shape} original_complete={source['training_complete']}", flush=True)
    for path, digest in ((manifest, binding["manifest_sha256"]), (fold_plan, binding["fold_plan_sha256"]),
                         (plan["source_train_manifest"]["path"], plan["source_manifest_sha256"]),
                         (collection["collection_path"], collection["collection_sha256"])):
        if sha256_file(path) != digest:
            raise RuntimeError("A bound teacher diagnostic input changed")
    index["complete"] = True
    write_json(output / "raw_index.json", index)
    return report_from_raw(output, output, index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--epoch13-checkpoint", type=Path, default=DEFAULT_EPOCH13)
    parser.add_argument("--fold-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--replay-dir", type=Path, help="Read saved raw logits using CPU only; no models or images")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.replay_dir is not None:
        if not args.execute:
            print(json.dumps({"dry_run": True, "mode": "CPU_replay", "input": str(args.replay_dir),
                              "output": str(args.output_dir), "gpu_used": False}, indent=2))
            return 0
        checked_output(args.output_dir, args.image_root)
        print(json.dumps(replay_diagnostic(args.replay_dir, args.output_dir), ensure_ascii=False, indent=2))
        return 0
    binding = validation_binding(args.manifest, args.fold_plan)
    collection = read_teacher_collection(args.teacher_collection, fold_plan_path=args.fold_plan)
    sources = source_plan(collection, args.epoch13_checkpoint)
    if not args.execute:
        print(json.dumps({"dry_run": True, "checkpoints_loaded": 0, "gpu_used": False,
            "manifest_sha256": binding["manifest_sha256"], "sources": sources,
            "A_epochs": [20, 12, 12, 12, 12], "B_epochs": [13, 12, 12, 12, 12],
            "raw_output": "raw_logits/*.npz (FP32 Nx2x10) plus metadata and sealed index",
            "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    completed_teacher_collection(args.teacher_collection, args.fold_plan)
    require_idle_gpu()
    checked_output(args.output_dir, args.image_root)
    if os.environ.get("KD_TRAINING_LOG_ACTIVE") != "1":
        from training_evidence_monitor import run_logged
        return run_logged(argparse.Namespace(evidence_dir=args.output_dir / "evidence",
            command=[sys.executable, "-u", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]))
    settings = vars(args)
    settings.pop("execute")
    settings.pop("replay_dir")
    report = evaluate(**settings)
    print("Teacher epoch/raw-logit diagnostic complete:", args.output_dir, flush=True)
    for result in report["A_vs_B"]:
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
