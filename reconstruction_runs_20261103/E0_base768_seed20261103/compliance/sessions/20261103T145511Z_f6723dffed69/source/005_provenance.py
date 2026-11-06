"""Fail-closed binding to the original, publicly released ImageNet Base weights.

This verifier hashes files but never deserializes a checkpoint or creates a model.
The trusted hashes are taken from the preserved original pretrained source record,
not from a caller-provided training checkpoint or its mutable metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_WEIGHTS_SHA256 = "bc0cfafc7d7755e8271f51343c4374725bf6409c0434e64028a863c4aca7d9fe"
EXPECTED_PROVENANCE_SHA256 = "bc3b95f32d79732da7b2e8553cbd0f79769cd5db902d2fd068e75ba53ceff4d8"
EXPECTED_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
EXPECTED_REPOSITORY = "timm/" + EXPECTED_BACKBONE
EXPECTED_REVISION = "2b2698d86523262ca8065d0957c7f80d963bb43f"
EXPECTED_WEIGHTS_URL = f"https://huggingface.co/{EXPECTED_REPOSITORY}/resolve/{EXPECTED_REVISION}/model.safetensors"
TRUSTED_WEIGHTS_PATH = Path("/tmp/cloud_student_runs_20261030/tight_crop_ab_assets/038_dual768_optimized_project_20261030/pretrain_model/timm__convnext_base.fb_in22k_ft_in1k_384_bc0cfafc7d77/model.safetensors")
TRUSTED_PROVENANCE_PATH = Path("/home/featurize/work/cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/base_original_pretrained_provenance.json")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_initialization(weights_path, provenance_path) -> dict:
    """Accept only exact original public weights and the preserved source record.

    Copies at different paths are accepted if their bytes are unchanged. A legacy
    task-trained checkpoint is rejected even if its metadata claims this backbone.
    Call immediately before loading the exact returned weights_path into the model.
    """
    weights_path, provenance_path = Path(weights_path).resolve(strict=True), Path(provenance_path).resolve(strict=True)
    if not weights_path.is_file() or not provenance_path.is_file():
        raise ValueError("Initialization requires two existing regular files")
    source_bytes = provenance_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != EXPECTED_PROVENANCE_SHA256:
        raise ValueError("Pretrained provenance is not the exact trusted original source record")
    source = json.loads(source_bytes)
    fields = {"repo_id": EXPECTED_REPOSITORY, "backbone_name": EXPECTED_BACKBONE,
              "revision": EXPECTED_REVISION, "weights_sha256": EXPECTED_WEIGHTS_SHA256,
              "weights_url": EXPECTED_WEIGHTS_URL, "license": "apache-2.0",
              "pretrain_dataset": "ImageNet-22k", "finetune_dataset": "ImageNet-1k"}
    for key, expected in fields.items():
        if source.get(key) != expected:
            raise ValueError(f"Unexpected original pretrained source field: {key}")
    materials = source.get("materials", {})
    for key in ["model_card", "apache_license", "upstream_license"]:
        record = materials.get(key, {})
        if not isinstance(record.get("text"), str) or not record.get("url"):
            raise ValueError(f"Missing original pretrained source material: {key}")
        if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record.get("sha256"):
            raise ValueError(f"Pretrained material text hash mismatch: {key}")
    actual_weights_sha = file_sha256(weights_path)
    if actual_weights_sha != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Initialization is not the original public ImageNet checkpoint; task-trained or changed weights are forbidden")
    return {"schema_version": 1, "kind": "original_public_nonmedical_pretrained_initialization",
            "verified": True, "weights_path": str(weights_path), "weights_bytes": weights_path.stat().st_size,
            "weights_sha256": actual_weights_sha, "provenance_path": str(provenance_path),
            "provenance_sha256": source_sha, **fields,
            "source_material_sha256": {k: materials[k]["sha256"] for k in ["model_card", "apache_license", "upstream_license"]},
            "legacy_task_checkpoint_used": False, "teacher_or_soft_label_initialization_used": False,
            "verification_scope": "File identity only; caller must load this exact path and enforce its train split ledger"}
