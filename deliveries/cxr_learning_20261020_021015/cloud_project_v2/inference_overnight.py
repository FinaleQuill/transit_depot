# 【学习导航】当前768五折推理主线：先逐图sigmoid，再Study视图均值，再TTA均值，最后fold概率加权均值；官方valid拟合阈值，test只输出。
# 本文件是独立学习副本；只增补解释性注释，运行语义保持源代码一致。
"""Explicit-only schema-5 CV ensemble inference; imports never run a model.

Each original image is decoded afresh for each requested resolution. Probabilities
are averaged over the original views of a Study, then over TTA transforms, then
over folds. Official validation is evaluated first with the exact same frozen
protocol; only its labels determine optional binary thresholds. Submission rows
always contain all ten continuous probabilities and follow the supplied sample.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dataset import CLASS_IDS, DEFAULT_TEST_MANIFEST, DEFAULT_VALID_MANIFEST, NUM_CLASSES, read_manifest


SAMPLE_HEADER = ["Subject_id", "Study_id", "Predict_class", "Probability"]
HK = timezone(timedelta(hours=8))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return result


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256: {field}")
    return value


def _resolve(value: Any, parent: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Plan file paths must be nonempty strings")
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


# 集成权重只在不同fold之间分配：默认每个fold权重相同，归一化后和为1。
# 不是对十种疾病做Softmax；十类概率仍互相独立。
def normalized_weights(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not 1 <= len(entries) <= 5:
        raise ValueError("An ensemble must contain one to five unique folds")
    weights = []
    folds: set[int] = set()
    for entry in entries:
        fold, weight = entry.get("fold"), entry.get("weight", 1.0)
        if type(fold) is not int or not 0 <= fold < 5 or fold in folds:
            raise ValueError("Fold indices must be unique integers in 0..4")
        folds.add(fold)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Ensemble weights must be finite, nonnegative numbers")
        weights.append(float(weight))
    # Scale before summing so even very large finite user weights stay finite.
    result = np.asarray(weights, dtype=np.float64)
    if result.max() <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    result /= result.max()
    return result / result.sum()


# 冻结集成计划绑定模型文件哈希、fold身份、分辨率和翻转开关，防止比较时悄悄换协议。
# 本函数只检查元数据；真正加载模型和计算在后续显式函数中。
def load_ensemble_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    source_hash = sha256(plan_path)
    plan = json_object(plan_path)
    if plan.get("schema_version") != 1 or type(plan.get("schema_version")) is not int:
        raise ValueError("Only ensemble plan schema_version=1 is supported")
    checkpoints = plan.get("checkpoints")
    if not isinstance(checkpoints, list) or any(not isinstance(item, dict) for item in checkpoints):
        raise ValueError("checkpoints must be a nonempty list of objects")
    weights = normalized_weights(checkpoints)
    tta = plan.get("tta")
    if not isinstance(tta, dict) or set(tta) != {"scales", "hflip"}:
        raise ValueError("TTA must explicitly define scales and hflip")
    scales = tta["scales"]
    if (not isinstance(scales, list) or not scales or any(type(x) is not int or x not in (384, 512, 768) for x in scales)
            or len(scales) != len(set(scales)) or type(tta["hflip"]) is not bool):
        raise ValueError("TTA scales must be distinct 384/512/768 values; hflip must be bool")
    fold_plan_path = _resolve(plan.get("fold_plan"), plan_path.parent)
    fold_plan_hash = _hash(plan.get("fold_plan_sha256"), "fold_plan_sha256")
    if sha256(fold_plan_path) != fold_plan_hash:
        raise ValueError("Fold plan hash differs from the frozen ensemble plan")
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    resolved_entries = []
    for entry, weight in zip(checkpoints, weights):
        checkpoint_path = _resolve(entry.get("path"), plan_path.parent)
        checkpoint_hash = _hash(entry.get("sha256"), "checkpoint sha256")
        if checkpoint_path in seen_paths or checkpoint_hash in seen_hashes:
            raise ValueError("The same checkpoint cannot masquerade as multiple folds")
        seen_paths.add(checkpoint_path)
        seen_hashes.add(checkpoint_hash)
        if sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
        resolved_entries.append({**entry, "path": str(checkpoint_path), "normalized_weight": float(weight)})
    if sha256(plan_path) != source_hash:
        raise RuntimeError("Ensemble plan changed while being read")
    return {**plan, "checkpoints": resolved_entries, "fold_plan": str(fold_plan_path),
            "plan_path": str(plan_path), "plan_sha256": source_hash}


def _check_probabilities(values: Any, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count, NUM_CLASSES):
        raise ValueError(f"Expected [{count},{NUM_CLASSES}] Study probabilities")
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    return result


# 读取提交样例的患者/Study/类别顺序，最终逐行匹配；不能因预测概率低就删某类。
def read_submission_template(path: str | Path, records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Check exact identity coverage without inspecting sample probabilities."""
    expected = {(str(row["subject_id"]), str(row["study_id"]), str(c)) for row in records for c in CLASS_IDS}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != SAMPLE_HEADER:
            raise ValueError(f"The template header must be exactly {SAMPLE_HEADER}")
        rows = list(reader)
    if any(len(row) != len(SAMPLE_HEADER) for row in rows):
        raise ValueError("Template has a malformed row")
    identities = [tuple(row[:3]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Template contains duplicate Subject/Study/class identities")
    if set(identities) != expected:
        raise ValueError(f"Template/test mismatch: missing={len(expected-set(identities))}, extra={len(set(identities)-expected)}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, header: Sequence[str], rows: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


# 多标签提交是长表：每个Study十行连续概率。二值阈值不参与此处的概率写出。
def write_matched_submission(path: str | Path, template_rows: Sequence[Sequence[str]],
                             result: Mapping[str, Any]) -> dict[str, Any]:
    """Write every template row, retaining identity text and order byte-for-byte."""
    path = Path(path)
    subjects, studies = result["subject_ids"], result["study_ids"]
    values = _check_probabilities(result["probabilities"], len(studies))
    if len(subjects) != len(studies) or len(set(zip(subjects, studies))) != len(studies):
        raise ValueError("Predicted Study identities are duplicated or malformed")
    indices = {(str(subject), str(study)): i for i, (subject, study) in enumerate(zip(subjects, studies))}
    expected = {(subject, study, str(c)) for subject, study in indices for c in CLASS_IDS}
    identities = [tuple(row[:3]) for row in template_rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Submission identities differ from the predicted Study coverage")
    generated = [[*row[:3], repr(float(values[indices[(row[0], row[1])], int(row[2])-1]))]
                 for row in template_rows]
    _write_csv(path, SAMPLE_HEADER, generated)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != SAMPLE_HEADER:
            raise RuntimeError("Written submission header changed")
        actual = list(reader)
    if len(actual) != len(template_rows) or [tuple(row[:3]) for row in actual] != identities:
        raise RuntimeError("Written submission row count or identity order changed")
    output_probabilities = np.asarray([float(row[3]) for row in actual])
    if (not np.isfinite(output_probabilities).all() or (output_probabilities < 0).any()
            or (output_probabilities > 1).any() or actual != generated):
        raise RuntimeError("Written submission probabilities failed round-trip validation")
    return {"header": SAMPLE_HEADER, "rows": len(actual), "studies": len(studies),
            "rows_per_study": NUM_CLASSES, "missing_studies": 0, "duplicate_identities": 0,
            "template_identity_order_equal": True, "all_probabilities_finite_in_0_1": True,
            "probability_min": float(output_probabilities.min()),
            "probability_max": float(output_probabilities.max()), "sha256": sha256(path)}


# checkpoint中同时可能存在线训练权重和EMA权重；这里明确读取验证同源的EMA。
# 还要核对768/历史512预处理、类别顺序和fold/hash，否则可能是另一套实验。
def _checkpoint(entry: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    from model_overnight import MODEL_NAME, validate_model_config

    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint changed after the ensemble plan was frozen")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha256(path) != entry["sha256"]:
        raise RuntimeError("Checkpoint changed while loading")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a metadata dictionary")
    expected = {"schema_version": 5, "model_name": MODEL_NAME, "num_classes": NUM_CLASSES,
                "class_ids": list(CLASS_IDS), "weights_source": "ema",
                "validation_complete": True, "aggregation": "study_mean_probability",
                "fold_index": entry["fold"], "fold_plan_sha256": plan["fold_plan_sha256"]}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch: {key}")
    if checkpoint.get("inference_ready", True) is not True:
        raise ValueError("Checkpoint is explicitly marked not ready for inference")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1:
        raise ValueError("A validated checkpoint must record a positive epoch")
    config = validate_model_config(checkpoint.get("model_config"))
    checkpoint["model_config"] = config
    if checkpoint.get("inference_precision") not in ("bfloat16", "float16"):
        raise ValueError("Schema 5 requires a recorded BF16/FP16 evaluation precision")
    from train_overnight import preprocessing_config
    image_size = checkpoint.get("image_size")
    if type(image_size) is not int or image_size not in (512, 768):
        raise ValueError("Schema 5 requires a recorded 512/768 input size")
    if checkpoint.get("preprocessing") != preprocessing_config(image_size=image_size):
        raise ValueError("Checkpoint preprocessing is not the schema-5 original-image protocol")
    fold_plan = json_object(plan["fold_plan"])
    if checkpoint.get("source_manifest_sha256") != fold_plan.get("source_manifest_sha256"):
        raise ValueError("Checkpoint source training manifest is not the frozen fold plan source")
    _hash(checkpoint.get("source_manifest_sha256"), "source_manifest_sha256")
    validation = checkpoint.get("validation")
    if (not isinstance(validation, dict) or type(validation.get("macro_auc")) not in (int, float)
            or not math.isfinite(validation["macro_auc"]) or not 0 <= validation["macro_auc"] <= 1):
        raise ValueError("Checkpoint lacks a complete finite internal-fold validation AUC")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("EMA model_state_dict must be a nonempty mapping of tensors")
    # Only validated EMA weights are used. Free optimizer/raw copies on the CPU.
    for field in ("raw_model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "ema_state_dict"):
        checkpoint.pop(field, None)
    return checkpoint


# 第一层平均：同一模型、同一TTA变换内，将同一Study多视图的sigmoid概率取均值。
# 第二层平均：把该模型在各分辨率/翻转下的Study概率再取均值。
def _predict_model(model: torch.nn.Module, records: Sequence[dict[str, Any]], *,
                   image_root: Path, scales: Sequence[int], hflip: bool, batch_size: int,
                   num_workers: int, prefetch_factor: int, device: torch.device,
                   precision: str) -> dict[str, Any]:
    from dataset_overnight import aggregate_study_predictions, make_overnight_loader

    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    total: np.ndarray | None = None
    reference: dict[str, Any] | None = None
    transforms = 0
    for scale in scales:
        # Each loader starts from the original file, never from a 512 thumbnail.
        loader = make_overnight_loader(records, image_root=image_root, training=False,
            image_size=scale, batch_size=batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor, seed=43)
        ids: list[str] = []
        variants: list[list[np.ndarray]] = [[], []] if hflip else [[]]
        try:
            with torch.inference_mode():
                for batch in loader:
                    images = batch["images"].to(device, non_blocking=device.type == "cuda")
                    if device.type == "cuda":
                        images = images.contiguous(memory_format=torch.channels_last)
                    ids.extend(batch["image_ids"])
                    for flip_index, values in enumerate(variants):
                        # 张量为[B,3,H,W]，最后一维是宽；flip(-1)就是水平翻转。
                        # 这是测试时增强TTA，没有随机亮度/旋转，也不会反向传播。
                        model_input = images.flip(-1) if flip_index else images
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                            logits = model(model_input)
                        if logits.shape != (len(batch["image_ids"]), NUM_CLASSES) or not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError("Invalid logits; no incomplete submission will be written")
                        # 模型给出[B,10]原始logit，独立sigmoid后才成为[0,1]概率。
                        # 先sigmoid再平均：sigmoid(mean(logit))一般不等于mean(sigmoid(logit))。
                        values.append(torch.sigmoid(logits.float()).cpu().numpy())
            for flip_index, values in enumerate(variants):
                # 检查每张图恰好预测一次，再按患者/Study键聚合；不把一张图当独立Study算分。
                result = aggregate_study_predictions(records, ids, np.concatenate(values))
                probabilities = _check_probabilities(result["probabilities"], len(result["study_keys"]))
                if reference is None:
                    reference = result
                    total = np.zeros_like(probabilities)
                elif (result["study_keys"] != reference["study_keys"] or result["subject_ids"] != reference["subject_ids"]
                        or result["study_ids"] != reference["study_ids"]):
                    raise RuntimeError("Study ordering differs between TTA transforms")
                total += probabilities
                transforms += 1
                print(f"TTA completed: resolution={scale}, hflip={bool(flip_index)}, studies={len(probabilities)}", flush=True)
        finally:
            del loader
            gc.collect()
    if reference is None or total is None or transforms == 0:
        raise RuntimeError("No complete TTA predictions")
    # 每个TTA变体在Study层面等权；当前新方案是768原图和水平翻转两个变体。
    return {**reference, "probabilities": total / transforms}


# 第三层平均：逐个加载fold的最佳EMA模型，把同一Study同一类别的概率按fold权重相加。
# 模型逐个加载可控制推理显存；这不同于先平均五套参数再做一次推理。
def _ensemble_phase(plan: Mapping[str, Any], records: Sequence[dict[str, Any]], *,
                    phase: str, image_root: Path, batch_size: int, num_workers: int,
                    prefetch_factor: int, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from model_overnight import build_overnight_model

    total: np.ndarray | None = None
    reference: dict[str, Any] | None = None
    metadata = []
    for entry in plan["checkpoints"]:
        if entry["normalized_weight"] == 0:
            continue
        checkpoint = _checkpoint(entry, plan)
        precision = checkpoint["inference_precision"]
        if device.type == "cuda" and precision == "bfloat16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Checkpoint requires BF16 but the selected GPU does not support it")
        model = build_overnight_model(pretrained=False, **checkpoint["model_config"])
        model.load_state_dict(checkpoint.pop("model_state_dict"), strict=True)
        model.eval().requires_grad_(False).to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        try:
            print(f"Ensemble {phase}: fold={entry['fold']}, epoch={checkpoint['epoch']}, weight={entry['normalized_weight']:.6f}", flush=True)
            result = _predict_model(model, records, image_root=image_root, scales=plan["tta"]["scales"],
                hflip=plan["tta"]["hflip"], batch_size=batch_size, num_workers=num_workers,
                prefetch_factor=prefetch_factor, device=device, precision=precision)
            if reference is None:
                reference = result
                total = np.zeros_like(result["probabilities"], dtype=np.float64)
            elif (result["study_keys"] != reference["study_keys"] or result["subject_ids"] != reference["subject_ids"]
                    or result["study_ids"] != reference["study_ids"]):
                raise RuntimeError("Study ordering differs between folds")
            # 五折都存在且默认等权时，每折贡献1/5；类与类之间不做归一化。
            total += entry["normalized_weight"] * result["probabilities"]
            metadata.append({"fold": entry["fold"], "checkpoint": entry["path"], "sha256": entry["sha256"],
                "epoch": checkpoint["epoch"], "normalized_weight": entry["normalized_weight"],
                "internal_validation_macro_auc": checkpoint["validation"]["macro_auc"],
                "model_config": checkpoint["model_config"], "weights_source": "ema",
                "inference_precision": precision if device.type == "cuda" else "float32"})
        finally:
            del model, checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if reference is None or total is None:
        raise RuntimeError("No positive-weight complete models")
    # Bound only floating point round-off from convex averaging, not bad logits.
    total = np.clip(total, 0.0, 1.0)
    return {**reference, "probabilities": _check_probabilities(total, len(reference["study_keys"]))}, metadata


def _write_wide(path: Path, result: Mapping[str, Any], thresholds: Sequence[float] | None = None) -> None:
    probabilities = _check_probabilities(result["probabilities"], len(result["study_ids"]))
    values = probabilities if thresholds is None else (probabilities >= np.asarray(thresholds)[None, :]).astype(np.uint8)
    _write_csv(path, ["Subject_id", "Study_id", *(f"class_{c}" for c in CLASS_IDS)],
        ([subject, study, *[repr(float(v)) if thresholds is None else int(v) for v in row]]
         for subject, study, row in zip(result["subject_ids"], result["study_ids"], values)))


# 最终评估流程：固定模型/TTA -> 官方valid概率 -> 只在valid找每类F1阈值 -> test概率。
# 内部折用于训练选轮；官方valid用于整个冻结集成的评估和阈值拟合；test没有标签。
def run_ensemble_inference(*, ensemble_plan: str | Path, output_dir: str | Path,
        image_root: str | Path, valid_manifest: str | Path = DEFAULT_VALID_MANIFEST,
        test_manifest: str | Path = DEFAULT_TEST_MANIFEST,
        sample_submission: str | Path = "/home/featurize/work/submission.csv",
        batch_size: int = 8, num_workers: int = 4, prefetch_factor: int = 1,
        device: str = "cuda", work_root: str | Path | None = None) -> Path:
    """Only an explicit call performs inference; never called by imports/tests."""
    from train import calculate_metrics, search_class_thresholds

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(num_workers) is not int or not 0 <= num_workers <= 12:
        raise ValueError("Use 0..12 DataLoader workers on this 16-core host")
    if type(prefetch_factor) is not int or prefetch_factor not in (1, 2):
        raise ValueError("prefetch_factor must be 1 or 2")
    plan = load_ensemble_plan(ensemble_plan)
    from cv_splits import read_fold_plan
    fold_plan = read_fold_plan(plan["fold_plan"])
    image_root = Path(image_root).resolve()
    destination = Path(output_dir).resolve()
    if work_root is not None and not destination.is_relative_to(Path(work_root).resolve()):
        raise ValueError("Output must stay under the requested work root")
    if destination.is_relative_to(image_root) or image_root.is_relative_to(destination):
        raise ValueError("Output must not overlap the read-only original-image directory")
    if destination.exists():
        raise FileExistsError(f"Choose a new ensemble output directory: {destination}")
    valid_path, test_path, sample_path = (Path(value).resolve() for value in (valid_manifest, test_manifest, sample_submission))
    source_hashes = {"official_valid_manifest_sha256": sha256(valid_path),
                     "test_manifest_sha256": sha256(test_path), "sample_submission_sha256": sha256(sample_path)}
    for key, actual_hash in source_hashes.items():
        if key in plan and plan[key] != actual_hash:
            raise ValueError(f"Input differs from frozen ensemble plan: {key}")
    if fold_plan["official_holdout"]["sha256"] != source_hashes["official_valid_manifest_sha256"]:
        raise ValueError("Official validation must be the same holdout excluded by the fold plan")
    valid_records, test_records = read_manifest(valid_path, split="valid"), read_manifest(test_path, split="test")
    if {r["subject_id"] for r in valid_records} & {r["subject_id"] for r in test_records}:
        raise ValueError("Official valid/test subjects overlap")
    training_patients = set(fold_plan["study_to_patient"].values())
    if training_patients & {r["subject_id"] for r in (*valid_records, *test_records)}:
        raise ValueError("Training subjects overlap official validation/test subjects")
    template_rows = read_submission_template(sample_path, test_records)
    target_device = torch.device(device)
    if target_device.type not in ("cuda", "cpu"):
        raise ValueError("Only explicit cuda or cpu devices are supported")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    destination.mkdir(parents=True, exist_ok=False)
    protocol = {"tta": plan["tta"], "checkpoints": [{key: item[key] for key in ("fold", "sha256", "normalized_weight")}
        for item in plan["checkpoints"]], "aggregation": "sigmoid -> original Study view mean -> TTA mean -> fold weighted mean"}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_json(destination / "status.json", {"status": "valid_running", "ensemble_plan_sha256": plan["plan_sha256"]})
    try:
        # Weights and TTA are already frozen. Validation metrics do not select
        # or drop models; they are a held-out diagnostic of the whole ensemble.
        valid_result, metadata = _ensemble_phase(plan, valid_records, phase="official_valid", image_root=image_root,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, device=target_device)
        # 阈值搜索发生在这里，而不是train_overnight.py每轮里。
        # 输入已是最终集成的Study级概率，阈值须与这套模型/TTA协议绑定。
        thresholds = search_class_thresholds(valid_result["targets"], valid_result["probabilities"])
        metrics = calculate_metrics(valid_result["targets"], valid_result["probabilities"], thresholds)
        clipped = np.clip(valid_result["probabilities"], 1e-7, 1-1e-7)
        y = valid_result["targets"]
        metrics.update({"loss": float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
            "studies": len(valid_result["study_keys"]), "images": len(valid_records),
            "folds": len(metadata), "ensemble_protocol_sha256": protocol_hash,
            "evaluation_scope": "official valid held out from all CV training; thresholds fitted here", "complete": True})
        _write_json(destination / "valid_metrics.json", metrics)
        _write_wide(destination / "valid_study_probabilities.csv", valid_result)
        _write_json(destination / "thresholds.json", {"schema_version": 5, "class_ids": list(CLASS_IDS),
            "thresholds": thresholds, "source": "official_valid_Study_F1", "comparison": ">=",
            "ensemble_plan_sha256": plan["plan_sha256"], "ensemble_protocol_sha256": protocol_hash,
            "official_valid_manifest_sha256": source_hashes["official_valid_manifest_sha256"],
            "submission_filters_rows": False, "thresholds_affect_probability_submission": False})
        print(f"Official valid ensemble AUC={metrics['macro_auc']:.6f}; starting test with frozen protocol", flush=True)
        _write_json(destination / "status.json", {"status": "test_running", "ensemble_plan_sha256": plan["plan_sha256"]})
        # test沿用已经固定的模型、权重、TTA和阈值，不从test拟合任何参数。
        test_result, test_metadata = _ensemble_phase(plan, test_records, phase="test", image_root=image_root,
            batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, device=target_device)
        if metadata != test_metadata or test_result["targets"] is not None:
            raise RuntimeError("Validation/test inference protocol changed or test labels were encountered")
        for path, key in ((valid_path, "official_valid_manifest_sha256"), (test_path, "test_manifest_sha256"),
                          (sample_path, "sample_submission_sha256")):
            if sha256(path) != source_hashes[key]:
                raise RuntimeError("Source CSV changed during inference")
        if sha256(plan["plan_path"]) != plan["plan_sha256"] or sha256(plan["fold_plan"]) != plan["fold_plan_sha256"]:
            raise RuntimeError("Frozen plan changed during inference")
        if any(sha256(entry["path"]) != entry["sha256"] for entry in plan["checkpoints"]):
            raise RuntimeError("A checkpoint changed during inference")
        _write_wide(destination / "study_probabilities.csv", test_result)
        # 二值宽表用于看各类是否过阈值；正式submission_final.csv仍保存全部十类原概率。
        _write_wide(destination / "study_binary.csv", test_result, thresholds)
        submission = destination / "submission_final.csv"
        validation = write_matched_submission(submission, template_rows, test_result)
        _write_json(destination / "validation_report.json", validation)
        provenance = {"schema_version": 5, "created_at": datetime.now(HK).isoformat(timespec="seconds"),
            "ensemble_plan_sha256": plan["plan_sha256"], "fold_plan_sha256": plan["fold_plan_sha256"],
            **source_hashes, "protocol": protocol, "protocol_sha256": protocol_hash, "models": metadata,
            "fold_count": len(metadata), "partial_cv_ensemble": len(metadata) < 5,
            "model_weights_fitted_on_valid_or_test": False, "test_used_for_fitting": False,
            "threshold_source": "official valid only, same TTA and model probability ensemble",
            "device": str(target_device), "loader": {"batch_size": batch_size, "num_workers": num_workers,
                "prefetch_factor": prefetch_factor}, "submission_validation": validation,
            "files_sha256": {file.name: sha256(file) for file in destination.iterdir() if file.is_file() and file.name != "status.json"},
            "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                ("inference_overnight.py", "dataset_overnight.py", "model_overnight.py", "cv_splits.py")},
            "platform_upload_performed": False}
        _write_json(destination / "provenance.json", provenance)
        _write_json(destination / "status.json", {"status": "complete", "submission": str(submission),
            "ensemble_plan_sha256": plan["plan_sha256"], "fold_count": len(metadata)})
        return submission
    except BaseException as exc:
        _write_json(destination / "status.json", {"status": "failed", "exception": type(exc).__name__,
            "message": str(exc), "ensemble_plan_sha256": plan["plan_sha256"]})
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schema-5 original-image TTA CV probability ensemble")
    parser.add_argument("--ensemble-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/home/featurize/data/cxr_first"))
    parser.add_argument("--work-root", type=Path, default=Path("/home/featurize/work"))
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--sample-submission", type=Path, default=Path("/home/featurize/work/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    submission = run_ensemble_inference(ensemble_plan=args.ensemble_plan, output_dir=args.output,
        image_root=args.image_root, work_root=args.work_root, valid_manifest=args.valid_manifest,
        test_manifest=args.test_manifest, sample_submission=args.sample_submission,
        batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, device=args.device)
    print(f"Complete locally validated submission: {submission}", flush=True)


if __name__ == "__main__":
    main()
