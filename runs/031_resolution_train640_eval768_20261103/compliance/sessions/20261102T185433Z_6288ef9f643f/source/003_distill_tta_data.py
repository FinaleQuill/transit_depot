"""Strict patient-OOF cache of each deterministic teacher view's raw logits.

This version intentionally leaves distill_data.py unchanged while teachers run.
No probabilities or temperature are baked into this cache: students first align
class weights per view, divide by their temperature, then average sigmoid values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import distill_data as legacy
from distill_data import (ARRAY_KEYS, CLASS_IDS, SELECTION_PROTOCOL, SoftLabelCache,
                          sha256_file, teacher_evidence, validate_teacher_checkpoint)

CACHE_SCHEMA = "cxr_strict_oof_view_logits_v2"
OUTPUT_TYPE = "per_view_raw_logits"
TARGET_AGGREGATION = "mean_sigmoid_after_pos_weight_alignment_and_temperature"


def canonical_tta(tta: str) -> str:
    if tta == "hflip2":
        return "identity_hflip"
    if tta not in {"identity", "identity_hflip"}:
        raise ValueError("Teacher TTA must be identity or identity_hflip")
    return tta


def normalize_tta(tta: str) -> tuple[str, ...]:
    return ("identity",) if canonical_tta(tta) == "identity" else ("identity", "hflip")


def _read_validated_plan(path: str | Path) -> dict[str, Any]:
    return legacy._read_validated_plan(path)


def validate_cache_arrays(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *,
                          plan: Mapping[str, Any], plan_hash: str,
                          records: Sequence[dict[str, Any]]) -> SoftLabelCache:
    """Retain every legacy provenance/leakage check separately for each view."""
    tta = metadata.get("tta")
    if tta not in {"identity", "identity_hflip"}:
        raise ValueError("TTA cache metadata must declare its canonical deterministic views")
    views = normalize_tta(tta)
    if (metadata.get("schema") != CACHE_SCHEMA
            or metadata.get("output_type") != OUTPUT_TYPE
            or metadata.get("views") != list(views)
            or metadata.get("view_axis") != 1
            or metadata.get("target_aggregation") != TARGET_AGGREGATION
            or metadata.get("temperature_applied") is not False
            or metadata.get("class_weight_alignment_applied") is not False
            or metadata.get("view_transform") != "horizontal_flip_after_letterbox_normalization"
            or metadata.get("original_train_only") is not True
            or metadata.get("official_valid_used") is not False
            or metadata.get("test_used") is not False):
        raise ValueError("TTA cache must contain declared per-view raw logits without temperature/alignment")
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Soft-label NPZ contains unexpected/missing arrays")
    logits = arrays["logits"]
    if (logits.shape != (len(records), len(views), 10)
            or logits.dtype != np.float32 or not np.isfinite(logits).all()):
        raise ValueError("TTA cache logits must be finite FP32 NxVx10 arrays")
    delegated_metadata = dict(metadata, schema=legacy.CACHE_SCHEMA,
                              output_type="raw_logits", tta="identity")
    checked = None
    for view_index in range(len(views)):
        delegated_arrays = dict(arrays, logits=logits[:, view_index, :])
        checked = legacy.validate_cache_arrays(delegated_metadata, delegated_arrays,
                    plan=plan, plan_hash=plan_hash, records=records)
    assert checked is not None
    logits.setflags(write=False)
    return SoftLabelCache(checked.image_ids, logits, checked.teacher_pos_weights,
                          checked.index, dict(metadata))


def load_soft_labels(cache_dir: str | Path, fold_plan_path: str | Path,
                     records: Sequence[dict[str, Any]]) -> SoftLabelCache:
    directory = Path(cache_dir).resolve(strict=True)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    npz = directory / "soft_labels.npz"
    if metadata.get("soft_labels_sha256") != sha256_file(npz):
        raise ValueError("Soft-label NPZ fingerprint changed")
    plan = _read_validated_plan(fold_plan_path)
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return validate_cache_arrays(metadata, arrays, plan=plan,
                                plan_hash=sha256_file(fold_plan_path), records=records)


def write_soft_labels(cache_dir: str | Path, fold_plan_path: str | Path,
                      records: Sequence[dict[str, Any]], logits: np.ndarray,
                      teachers: Sequence[dict[str, Any]], *,
                      tta: str = "identity_hflip") -> SoftLabelCache:
    """Validate before exclusively creating a new cache; never overwrite history."""
    tta = canonical_tta(tta)
    plan = _read_validated_plan(fold_plan_path)
    plan_hash = sha256_file(fold_plan_path)
    teacher_map = {entry["fold_index"]: entry for entry in teachers}
    if len(teachers) != 5 or set(teacher_map) != set(range(5)):
        raise ValueError("Exactly five unique teacher folds are required")
    folds = [plan["study_to_fold"][row["study_key"]] for row in records]
    arrays = {
        "image_ids": np.asarray([r["image_id"] for r in records], dtype=str),
        "subject_ids": np.asarray([r["subject_id"] for r in records], dtype=str),
        "study_keys": np.asarray([r["study_key"] for r in records], dtype=str),
        "relative_paths": np.asarray([r["relative_path"] for r in records], dtype=str),
        "source_image_sha256": np.asarray([r["file_sha256"] for r in records], dtype=str),
        "fold_indices": np.asarray(folds, dtype=np.int16),
        "logits": np.asarray(logits, dtype=np.float32),
        "teacher_pos_weights": np.asarray([teacher_map[f]["loss_config"]["pos_weight"]
                                            for f in folds], dtype=np.float32),
    }
    metadata = {
        "schema": CACHE_SCHEMA, "class_ids": CLASS_IDS, "source_split": "train",
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "fold_plan_sha256": plan_hash, "image_size": 768, "prediction_unit": "image",
        "output_type": OUTPUT_TYPE, "tta": tta, "views": list(normalize_tta(tta)),
        "view_axis": 1, "view_transform": "horizontal_flip_after_letterbox_normalization",
        "target_aggregation": TARGET_AGGREGATION, "temperature_applied": False,
        "class_weight_alignment_applied": False, "num_images": len(records),
        "selection_protocol": SELECTION_PROTOCOL, "complete": True,
        "teachers": list(teachers), "original_train_only": True,
        "official_valid_used": False, "test_used": False,
        "notes": "Each image uses one fixed-epoch teacher that excluded its patient; each view remains raw.",
    }
    validate_cache_arrays(metadata, arrays, plan=plan, plan_hash=plan_hash, records=records)
    destination = Path(cache_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    npz = destination / "soft_labels.npz"
    temporary_npz = destination / "soft_labels.npz.tmp"
    with temporary_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_npz.replace(npz)
    metadata["soft_labels_sha256"] = sha256_file(npz)
    temporary_metadata = destination / "metadata.json.tmp"
    with temporary_metadata.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_metadata.replace(destination / "metadata.json")
    return load_soft_labels(destination, fold_plan_path, records)
