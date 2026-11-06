"""Export a validated single student's EMA weights to a fixed FP16 deployment.

Schema7 Tiny512 identity exports remain supported. Schema8 can declare 512/768
and fixed identity/hflip2. Choosing TTA never alters the trained checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from deploy_kd import (CLASS_IDS, IMAGE_SIZE, MAX_MODEL_BYTES, TINY_PARAMETERS_10_CLASSES,
                       TINY_BACKBONE, deployment_inventory, normalized_model_weights, sha256_file,
                       validate_deployment_config, validate_student_checkpoint, write_json)


def estimate_size() -> dict[str, Any]:
    return {"backbone_name": TINY_BACKBONE, "num_classes": 10,
            "estimated_parameters": TINY_PARAMETERS_10_CLASSES,
            "fp16_parameter_bytes": TINY_PARAMETERS_10_CLASSES * 2,
            "fp32_parameter_bytes": TINY_PARAMETERS_10_CLASSES * 4,
            "limit_bytes": MAX_MODEL_BYTES, "size_status": "estimate_only",
            "latency_status": "pending_T4_measurement",
            "note": "Each additional Tiny member adds this parameter count; actual serialization/configuration must be measured."}


def make_deployment_config(members: list[dict], *, source_manifest_sha256: str,
                           fold_plan_sha256: str, official_valid_manifest_sha256: str,
                           model_weights: Any = None) -> dict:
    """Canonical frozen configuration shared by single export and composition."""
    if not members:
        raise ValueError("A deployment requires student members")
    first = members[0]
    sizes = sorted({m["image_size"] for m in members})
    ttas = {m["tta"] for m in members}
    weights = normalized_model_weights(model_weights if model_weights is not None else [1.0] * len(members), len(members))
    config = {"schema_version": 2, "artifact_type": "kd_student_deployment",
              "backbone_name": TINY_BACKBONE, "class_ids": CLASS_IDS,
              "precision": "float16", "batch_size": 1,
              "ensemble": len(members) > 1,
              "tta": next(iter(ttas)) if len(ttas) == 1 else "mixed",
              "image_size": sizes[0] if len(sizes) == 1 else None, "image_sizes": sizes,
              "aggregation": "study_mean_probability", "member_aggregation": "fixed_weighted_mean_probability",
              "tta_aggregation": "mean_sigmoid_probability", "members": members,
              "model_weights": weights.tolist(),
              "source_manifest_sha256": source_manifest_sha256,
              "fold_plan_sha256": fold_plan_sha256,
              "official_valid_manifest_sha256": official_valid_manifest_sha256,
              "inference_files": [m["weights_file"] for m in members] + ["config.json"],
              "gates": None, "weights_source": "ema",
              "latency_status": "pending_T4_measurement",
              "fp16_validation_status": "pending_same_deployment_validation",
              # Representative source retained for readers of legacy package metadata.
              "pretrained_provenance": first["pretrained_provenance"],
              "teacher_provenance": first["teacher_provenance"],
              "distillation": first.get("distillation", {}),
              "source_validation": first.get("source_validation", {})}
    validate_deployment_config(config)
    return config


def export_student(checkpoint_path: str | Path, output: str | Path, *, tta: str | None = None) -> dict[str, Any]:
    import torch
    from model_overnight import build_overnight_model

    if tta not in {None, "identity", "hflip2"}:
        raise ValueError("TTA must be fixed identity or hflip2")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory; existing files are never overwritten")
    source_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_student_checkpoint(checkpoint)
    if sha256_file(checkpoint_path) != source_sha256:
        raise RuntimeError("Source checkpoint changed while loading")
    raw = checkpoint.get("model_state_dict")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Missing EMA model_state_dict")
    state = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Malformed EMA tensor state")
        if value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Cannot export nonfinite EMA weights")
            value = value.detach().cpu().half()
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Weight overflows FP16")
        state[name] = value
    model_config = {**checkpoint["model_config"], "gradient_checkpointing": False}
    model = build_overnight_model(pretrained=False, **model_config)
    model.load_state_dict(state, strict=True)  # Shape/key verification only; no forward.
    del model
    selected_tta = tta or checkpoint.get("deployment_tta", "identity")
    common = {"model_config": model_config, "image_size": checkpoint["image_size"],
              "tta": selected_tta, "preprocessing": checkpoint["preprocessing"],
              "pretrained_provenance": checkpoint["pretrained_provenance"],
              "teacher_provenance": checkpoint["teacher_provenance"],
              "distillation": checkpoint.get("distillation", {}),
              "source_checkpoint_sha256": source_sha256,
              "source_validation": checkpoint.get("validation", {}),
              "weights_source": "ema", "seed": checkpoint.get("training_config", {}).get("seed")}
    output.mkdir(parents=True, exist_ok=False)
    torch.save(state, output / "student_fp16.pt")
    common["weights_sha256"] = sha256_file(output / "student_fp16.pt")
    if checkpoint["schema_version"] == 7 and tta is None:
        config = {**common, "schema_version": 1, "artifact_type": "kd_student_deployment",
                  "backbone_name": TINY_BACKBONE, "class_ids": CLASS_IDS,
                  "precision": "float16", "batch_size": 1, "ensemble": False,
                  "aggregation": "study_mean_probability", "gates": None,
                  "source_manifest_sha256": checkpoint["source_manifest_sha256"],
                  "fold_plan_sha256": checkpoint["fold_plan_sha256"],
                  "official_valid_manifest_sha256": checkpoint["official_valid_manifest_sha256"],
                  "inference_files": ["student_fp16.pt", "config.json"],
                  "latency_status": "pending_T4_measurement",
                  "fp16_validation_status": "pending_same_deployment_validation"}
    else:
        config = make_deployment_config([{**common, "weights_file": "student_fp16.pt"}],
            source_manifest_sha256=checkpoint["source_manifest_sha256"],
            fold_plan_sha256=checkpoint["fold_plan_sha256"],
            official_valid_manifest_sha256=checkpoint["official_valid_manifest_sha256"])
    validate_deployment_config(config)
    write_json(output / "config.json", config)
    inventory = deployment_inventory(output, config)
    report = {**estimate_size(), **inventory,
              "source_checkpoint_sha256": source_sha256,
              "export_directory": str(output), "size_status": "measured",
              "image_size": checkpoint["image_size"], "tta": selected_tta, "member_count": 1,
              "fp16_validation_status": "pending_same_deployment_validation"}
    write_json(output / "export_report.json", report)
    if not inventory["size_pass"]:
        raise ValueError("Actual deployment files exceed the 500MB limit")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tta", choices=("identity", "hflip2"),
                        help="Fixed inference views; omitted uses the checkpoint deployment declaration")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    import json
    if args.estimate_only:
        print(json.dumps(estimate_size(), ensure_ascii=False, indent=2))
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required unless --estimate-only")
    print(json.dumps(export_student(args.checkpoint, args.output, tta=args.tta), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
