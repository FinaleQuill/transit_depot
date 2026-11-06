"""严格折外图像级教师缓存的读写与验收；导入时不读图片、不构建模型。

缓存只包含原始 train 的 logits。每行与图像、患者、原文件 SHA 及 fold
绑定，学生无需加载教师。NPZ 禁止 pickle，教师证据随 JSON 一起保留。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


CACHE_SCHEMA = "cxr_strict_oof_logits_v1"
TEACHER_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
TEACHER_MODEL_NAME = "convnext_v1_base_gap"
CLASS_IDS = list(range(1, 11))
SELECTION_PROTOCOL = "fixed_epoch_final_ema"
ARRAY_KEYS = {"image_ids", "subject_ids", "study_keys", "relative_paths",
              "source_image_sha256", "fold_indices", "logits", "teacher_pos_weights"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_validated_plan(path: str | Path) -> dict[str, Any]:
    from cv_splits import read_fold_plan

    return read_fold_plan(path)


def _record_map(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        image_id = row.get("image_id")
        if (not isinstance(image_id, str) or not image_id or image_id in result
                or row.get("split") != "train"
                or str(row.get("training_eligible", "")).lower() != "true"
                or not isinstance(row.get("subject_id"), str) or not row["subject_id"]
                or row.get("study_key") != f"{row['subject_id']}/{row.get('study_id')}"
                or not _sha256(row.get("file_sha256"))
                or row.get("targets") is None or len(row["targets"]) != 10
                or any(value not in (0, 1) for value in row["targets"])):
            raise ValueError("OOF cache requires unique, eligible original train image records")
        result[image_id] = row
    if not result:
        raise ValueError("OOF cache cannot be empty")
    return result


def _fold_records(records: Sequence[dict[str, Any]], plan: Mapping[str, Any],
                  fold: int, *, held_out: bool) -> list[dict[str, Any]]:
    mapping = plan["study_to_fold"]
    return [row for row in records if (mapping[row["study_key"]] == fold) == held_out]


def _pos_weights(records: Sequence[dict[str, Any]]) -> np.ndarray:
    studies: dict[str, tuple[int, ...]] = {}
    for row in records:
        labels = tuple(row["targets"])
        previous = studies.setdefault(row["study_key"], labels)
        if previous != labels:
            raise ValueError("Training labels disagree inside a Study")
    targets = np.asarray(list(studies.values()), dtype=np.float64)
    positives = targets.sum(0)
    negatives = len(targets) - positives
    if (positives <= 0).any() or (negatives <= 0).any():
        raise ValueError("Every teacher training class requires positive and negative Studies")
    return negatives / positives


def _validate_teacher_evidence(evidence: Mapping[str, Any], fold: int,
                               plan: Mapping[str, Any], plan_hash: str,
                               records: Sequence[dict[str, Any]]) -> None:
    """验收固定训练轮数协议，外折绝不参与早停或检查点选择。"""
    expected_train = _fold_records(records, plan, fold, held_out=False)
    expected_oof = _fold_records(records, plan, fold, held_out=True)
    train_subjects = sorted({row["subject_id"] for row in expected_train})
    oof_subjects = sorted({row["subject_id"] for row in expected_oof})
    train_images = sorted(row["image_id"] for row in expected_train)
    oof_images = sorted(row["image_id"] for row in expected_oof)
    config = evidence.get("model_config", {})
    epoch = evidence.get("epoch")
    horizon = evidence.get("training_config", {}).get("epochs")
    if (evidence.get("schema_version") != 6
            or evidence.get("artifact_type") != "kd_teacher"
            or evidence.get("model_name") != TEACHER_MODEL_NAME
            or config.get("backbone_name") != TEACHER_BACKBONE
            or evidence.get("class_ids") != CLASS_IDS or evidence.get("image_size") != 768
            or evidence.get("fold_index") != fold
            or type(evidence.get("fold_index")) is not int
            or evidence.get("fold_plan_sha256") != plan_hash
            or evidence.get("source_manifest_sha256") != plan["source_manifest_sha256"]
            or evidence.get("selection_protocol") != SELECTION_PROTOCOL
            or evidence.get("training_complete") is not True
            or evidence.get("inference_ready") is not True
            or evidence.get("weights_source") != "ema"
            or type(epoch) is not int or type(horizon) is not int or epoch < 1 or epoch != horizon
            or evidence.get("training_subject_ids") != train_subjects
            or evidence.get("oof_subject_ids") != oof_subjects
            or evidence.get("training_image_ids") != train_images
            or evidence.get("oof_image_ids") != oof_images
            or evidence.get("outer_fold_used_for_selection") is not False
            or evidence.get("official_valid_used_for_training") is not False
            or evidence.get("test_used_for_training") is not False
            or set(train_subjects) & set(oof_subjects)):
        raise ValueError(f"Teacher fold {fold} is not a complete, fixed-epoch strict-OOF V1 teacher")
    preprocessing = evidence.get("preprocessing", {})
    if (preprocessing.get("image_size") != 768
            or preprocessing.get("validation_size") != 768
            or preprocessing.get("resize") != "letterboxLANCZOS"
            or preprocessing.get("validation_tta") != "identity"
            or preprocessing.get("mean") != [0.485, 0.456, 0.406]
            or preprocessing.get("std") != [0.229, 0.224, 0.225]
            or preprocessing.get("color_mode") != "RGB"
            or preprocessing.get("exif_transpose") is not True
            or preprocessing.get("data_source") != "raw_readonly"):
        raise ValueError("Teacher preprocessing differs from identity 768 letterbox protocol")
    values = np.asarray(evidence.get("loss_config", {}).get("pos_weight", []), dtype=np.float64)
    if (values.shape != (10,) or not np.isfinite(values).all()
            or not np.allclose(values, _pos_weights(expected_train), rtol=1e-6, atol=1e-7)):
        raise ValueError("Teacher class weights were not fitted on its training Studies only")
    from kd_common import validate_pretrained_provenance

    validate_pretrained_provenance(evidence.get("pretrained_provenance"), TEACHER_BACKBONE)


def validate_teacher_checkpoint(checkpoint: Mapping[str, Any], *, fold: int,
                                plan: Mapping[str, Any], plan_hash: str,
                                records: Sequence[dict[str, Any]]) -> None:
    """显式提取前调用；不会构建模型或执行前向。"""
    _record_map(records)
    _validate_teacher_evidence(checkpoint, fold, plan, plan_hash, records)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Teacher EMA model_state_dict is absent")


def teacher_evidence(checkpoint: Mapping[str, Any], *, checkpoint_path: str | Path,
                     checkpoint_sha256: str, inference_precision: str) -> dict[str, Any]:
    if not _sha256(checkpoint_sha256) or inference_precision not in {"bfloat16", "float16"}:
        raise ValueError("Teacher checkpoint SHA or extraction precision is invalid")
    fields = ("schema_version", "artifact_type", "model_name", "model_config", "class_ids",
              "image_size", "fold_index", "fold_plan_sha256", "source_manifest_sha256",
              "selection_protocol", "training_complete", "inference_ready", "weights_source",
              "epoch", "training_subject_ids", "oof_subject_ids", "preprocessing",
              "training_image_ids", "oof_image_ids", "outer_fold_used_for_selection",
              "official_valid_used_for_training", "test_used_for_training",
              "loss_config", "pretrained_provenance")
    result = {key: checkpoint[key] for key in fields}
    result.update(checkpoint_path=str(Path(checkpoint_path).resolve()),
                  checkpoint_sha256=checkpoint_sha256,
                  extraction_precision=inference_precision,
                  training_config={"epochs": checkpoint["training_config"]["epochs"]})
    return result


@dataclass(frozen=True)
class SoftLabelCache:
    image_ids: tuple[str, ...]
    logits: np.ndarray
    teacher_pos_weights: np.ndarray
    index: dict[str, int]
    metadata: dict[str, Any]


def validate_cache_arrays(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *,
                          plan: Mapping[str, Any], plan_hash: str,
                          records: Sequence[dict[str, Any]]) -> SoftLabelCache:
    expected = _record_map(records)
    studies = {row["study_key"] for row in records}
    if (set(plan["study_to_fold"]) != studies
            or any(type(v) is not int or not 0 <= v < 5 for v in plan["study_to_fold"].values())
            or any(plan["study_to_patient"].get(row["study_key"]) != row["subject_id"] for row in records)):
        raise ValueError("Fold plan does not exactly cover original train Studies/patients")
    if (metadata.get("schema") != CACHE_SCHEMA
            or metadata.get("source_split") != "train"
            or metadata.get("class_ids") != CLASS_IDS
            or metadata.get("source_manifest_sha256") != plan["source_manifest_sha256"]
            or metadata.get("fold_plan_sha256") != plan_hash
            or metadata.get("image_size") != 768
            or metadata.get("prediction_unit") != "image"
            or metadata.get("output_type") != "raw_logits"
            or metadata.get("tta") != "identity"
            or metadata.get("selection_protocol") != SELECTION_PROTOCOL
            or metadata.get("num_images") != len(records)
            or metadata.get("complete") is not True):
        raise ValueError("Soft labels are not a complete original-train strict-OOF logits cache")
    teachers = metadata.get("teachers", [])
    if not isinstance(teachers, list) or len(teachers) != 5:
        raise ValueError("Exactly five teacher evidence entries are required")
    by_fold: dict[int, Mapping[str, Any]] = {}
    for evidence in teachers:
        fold = evidence.get("fold_index")
        if type(fold) is not int or fold in by_fold or not 0 <= fold < 5:
            raise ValueError("Teacher folds are duplicated or invalid")
        _validate_teacher_evidence(evidence, fold, plan, plan_hash, records)
        if not _sha256(evidence.get("checkpoint_sha256")):
            raise ValueError("Teacher checkpoint fingerprint is absent")
        by_fold[fold] = evidence
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Soft-label NPZ contains unexpected/missing arrays")
    n = len(records)
    for key in ARRAY_KEYS - {"logits", "teacher_pos_weights"}:
        if arrays[key].shape != (n,):
            raise ValueError(f"Soft-label {key} array must have shape N")
    for key in ("image_ids", "subject_ids", "study_keys", "relative_paths", "source_image_sha256"):
        if arrays[key].dtype.kind != "U":
            raise ValueError("Cache identities must be Unicode arrays, never pickle/object data")
    ids = tuple(str(value) for value in arrays["image_ids"])
    if len(set(ids)) != n or set(ids) != set(expected):
        raise ValueError("Soft labels must cover every train image exactly once")
    logits, weights = arrays["logits"], arrays["teacher_pos_weights"]
    if (logits.shape != (n, 10) or logits.dtype != np.float32 or not np.isfinite(logits).all()
            or weights.shape != (n, 10) or weights.dtype != np.float32
            or not np.isfinite(weights).all() or (weights <= 0).any()
            or arrays["fold_indices"].dtype.kind not in {"i", "u"}):
        raise ValueError("Cache logits/class weights must be finite FP32 Nx10 arrays")
    for index, image_id in enumerate(ids):
        row = expected[image_id]
        fold = plan["study_to_fold"][row["study_key"]]
        if (int(arrays["fold_indices"][index]) != fold
                or str(arrays["subject_ids"][index]) != row["subject_id"]
                or str(arrays["study_keys"][index]) != row["study_key"]
                or str(arrays["relative_paths"][index]) != row["relative_path"]
                or str(arrays["source_image_sha256"][index]) != row["file_sha256"]
                or row["subject_id"] in by_fold[fold]["training_subject_ids"]
                or not np.allclose(weights[index], by_fold[fold]["loss_config"]["pos_weight"],
                                   rtol=1e-6, atol=1e-7)):
            raise ValueError(f"Soft-label image identity/fold/teacher class weight mismatch: {image_id}")
    logits.setflags(write=False)
    weights.setflags(write=False)
    return SoftLabelCache(ids, logits, weights, {key: i for i, key in enumerate(ids)}, dict(metadata))


def load_soft_labels(cache_dir: str | Path, fold_plan_path: str | Path,
                     records: Sequence[dict[str, Any]]) -> SoftLabelCache:
    """离线验收缓存；不加载教师，不需要原教师权重仍位于同一路径。"""
    directory = Path(cache_dir).resolve(strict=True)
    metadata_path = directory / "metadata.json"
    npz_path = directory / "soft_labels.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("soft_labels_sha256") != sha256_file(npz_path):
        raise ValueError("Soft-label NPZ fingerprint changed")
    plan = _read_validated_plan(fold_plan_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return validate_cache_arrays(metadata, arrays, plan=plan,
                                 plan_hash=sha256_file(fold_plan_path), records=records)


def write_soft_labels(cache_dir: str | Path, fold_plan_path: str | Path,
                      records: Sequence[dict[str, Any]], logits: np.ndarray,
                      teachers: Sequence[dict[str, Any]]) -> SoftLabelCache:
    """完整提取后一次提交；独占创建目录，不覆盖已有软标签。"""
    plan = _read_validated_plan(fold_plan_path)
    plan_hash = sha256_file(fold_plan_path)
    teacher_map = {entry["fold_index"]: entry for entry in teachers}
    folds = [plan["study_to_fold"][row["study_key"]] for row in records]
    arrays = {
        "image_ids": np.asarray([row["image_id"] for row in records], dtype=str),
        "subject_ids": np.asarray([row["subject_id"] for row in records], dtype=str),
        "study_keys": np.asarray([row["study_key"] for row in records], dtype=str),
        "relative_paths": np.asarray([row["relative_path"] for row in records], dtype=str),
        "source_image_sha256": np.asarray([row["file_sha256"] for row in records], dtype=str),
        "fold_indices": np.asarray(folds, dtype=np.int16),
        "logits": np.asarray(logits, dtype=np.float32),
        "teacher_pos_weights": np.asarray([teacher_map[fold]["loss_config"]["pos_weight"]
                                            for fold in folds], dtype=np.float32),
    }
    metadata = {
        "schema": CACHE_SCHEMA, "class_ids": CLASS_IDS, "source_split": "train",
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "fold_plan_sha256": plan_hash, "image_size": 768, "prediction_unit": "image",
        "output_type": "raw_logits", "tta": "identity", "num_images": len(records),
        "selection_protocol": SELECTION_PROTOCOL, "complete": True,
        "teachers": list(teachers),
        "original_train_only": True, "official_valid_used": False, "test_used": False,
        "notes": "Each image uses exactly one fixed-epoch teacher from its held-out patient fold.",
    }
    validate_cache_arrays(metadata, arrays, plan=plan, plan_hash=plan_hash, records=records)
    destination = Path(cache_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    npz_path = destination / "soft_labels.npz"
    temporary_npz = destination / "soft_labels.npz.tmp"
    with temporary_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_npz.replace(npz_path)
    metadata["soft_labels_sha256"] = sha256_file(npz_path)
    temporary_metadata = destination / "metadata.json.tmp"
    with temporary_metadata.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary_metadata.replace(destination / "metadata.json")
    return load_soft_labels(destination, fold_plan_path, records)
