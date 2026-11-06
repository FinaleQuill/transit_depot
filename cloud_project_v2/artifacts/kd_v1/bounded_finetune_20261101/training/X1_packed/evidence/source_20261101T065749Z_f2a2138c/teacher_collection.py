"""Declare five strict-OOF teachers from separately frozen training runs.

Collection parsing and metadata checks use only small JSON files. Checkpoint
weights are opened only by the explicit extraction command, never by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ARTIFACT_TYPE = "kd_teacher_collection"
BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
SELECTION_PROTOCOL = "fixed_epoch_final_ema"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Teacher collection paths must be nonempty strings")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def read_teacher_collection(path: str | Path, *, fold_plan_path: str | Path | None = None,
                            require_checkpoints: bool = False) -> dict[str, Any]:
    """Validate a frozen 0..4 collection, optionally checking completed-run files.

    Preview mode intentionally permits not-yet-created best.pt paths. It still
    verifies the exact fold-plan bytes and rejects incomplete/duplicate folds.
    """
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("artifact_type") != ARTIFACT_TYPE
            or value.get("image_size") != 768
            or value.get("selection_protocol") != SELECTION_PROTOCOL):
        raise ValueError("Unsupported fixed-epoch 768 teacher collection")
    for key in ("fold_plan_sha256", "source_manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))) is None:
            raise ValueError(f"Teacher collection requires exact {key}")
    declared_plan = _path(value.get("fold_plan"), path.parent)
    actual_plan = Path(fold_plan_path).resolve(strict=True) if fold_plan_path else declared_plan
    if (_sha256_file(actual_plan) != value["fold_plan_sha256"]
            or _sha256_file(declared_plan) != value["fold_plan_sha256"]):
        raise ValueError("Teacher collection fold-plan fingerprint changed")
    teachers = value.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != 5:
        raise ValueError("Teacher collection must contain all five patient folds")
    declarations, seen_folds, seen_paths = [], set(), set()
    for original in teachers:
        if not isinstance(original, dict):
            raise ValueError("Teacher collection entries must be objects")
        fold, epochs = original.get("fold_index"), original.get("epochs")
        if (type(fold) is not int or fold not in range(5) or fold in seen_folds
                or type(epochs) is not int or epochs < 2):
            raise ValueError("Teacher collection requires unique folds 0..4 and fixed epochs >=2")
        checkpoint = _path(original.get("checkpoint"), path.parent)
        if checkpoint in seen_paths:
            raise ValueError("Teacher collection cannot reuse one checkpoint for multiple folds")
        if checkpoint.parent.name != f"fold_{fold}" or checkpoint.name != "best.pt":
            raise ValueError("Teacher collection must point to each declared fold_N/best.pt")
        declaration = {**original, "checkpoint": str(checkpoint)}
        if require_checkpoints:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Teacher fold {fold} is not ready: {checkpoint}")
            _validate_frozen_run(declaration, value["fold_plan_sha256"],
                                 value["source_manifest_sha256"])
        declarations.append(declaration)
        seen_folds.add(fold)
        seen_paths.add(checkpoint)
    return {**value, "fold_plan": str(actual_plan),
            "teachers": sorted(declarations, key=lambda entry: entry["fold_index"]),
            "collection_path": str(path), "collection_sha256": _sha256_file(path)}


def _validate_frozen_run(declaration: Mapping[str, Any], plan_hash: str,
                         manifest_hash: str) -> dict[str, Any]:
    checkpoint = Path(declaration["checkpoint"])
    fold, epochs = declaration["fold_index"], declaration["epochs"]
    run_path = checkpoint.parent.parent / "teacher_run.json"
    config_path = checkpoint.parent / "config.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or not isinstance(config, dict):
        raise ValueError("Teacher frozen run and config must be JSON objects")
    if (run.get("artifact_type") != "kd_teacher_run" or run.get("schema_version") != 6
            or run.get("backbone_name") != BACKBONE or run.get("image_size") != 768
            or run.get("selection_protocol") != SELECTION_PROTOCOL
            or run.get("epochs") != epochs or run.get("fold_plan_sha256") != plan_hash
            or run.get("official_valid_used_for_training") is not False
            or run.get("outer_fold_used_for_selection") is not False):
        raise ValueError(f"Teacher fold {fold} differs from its frozen training-run declaration")
    binding = config.get("binding", {})
    if (config.get("fold") != fold or config.get("epochs") != epochs
            or config.get("seed") != run.get("seed")
            or config.get("model_config", {}).get("backbone_name") != BACKBONE
            or binding.get("fold_index") != fold
            or binding.get("fold_plan_sha256") != plan_hash
            or binding.get("source_manifest_sha256") != manifest_hash
            or binding.get("selection_protocol") != SELECTION_PROTOCOL
            or binding.get("outer_fold_used_for_selection") is not False):
        raise ValueError(f"Teacher fold {fold} config differs from its frozen collection")
    from kd_common import validate_pretrained_provenance

    validate_pretrained_provenance(config.get("pretrained_provenance"), BACKBONE)
    return config


def validate_collection_checkpoint(checkpoint: Mapping[str, Any], declaration: Mapping[str, Any], *,
                                   plan: Mapping[str, Any], plan_hash: str,
                                   records: Sequence[dict[str, Any]]) -> None:
    """Keep the existing source/patient-isolation checks and enforce this budget."""
    from distill_data import validate_teacher_checkpoint

    validate_teacher_checkpoint(checkpoint, fold=declaration["fold_index"], plan=plan,
                                plan_hash=plan_hash, records=records)
    if (checkpoint["epoch"] != declaration["epochs"]
            or checkpoint["training_config"]["epochs"] != declaration["epochs"]):
        raise ValueError("Teacher checkpoint does not complete its declared collection epochs")
    config = _validate_frozen_run(declaration, plan_hash, plan["source_manifest_sha256"])
    for key in ("fold", "epochs", "seed", "batch_size", "accumulation_steps"):
        if checkpoint["training_config"].get(key) != config.get(key):
            raise ValueError(f"Teacher checkpoint training configuration differs from recorded {key}")
    actual, recorded = checkpoint["pretrained_provenance"], config.get("pretrained_provenance", {})
    if any(actual.get(key) != recorded.get(key) for key in ("weights_sha256", "revision", "repo_id")):
        raise ValueError("Teacher checkpoint initialization differs from its recorded provenance")
