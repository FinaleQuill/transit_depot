"""Export only the approved student's EMA to a new FP16 deployment directory."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from deploy_kd import (CLASS_IDS, IMAGE_SIZE, MAX_MODEL_BYTES, TINY_PARAMETERS_10_CLASSES,
                       TINY_BACKBONE, deployment_inventory, sha256_file,
                       validate_deployment_config, validate_student_checkpoint, write_json)


def estimate_size() -> dict[str, Any]:
    return {"backbone_name": TINY_BACKBONE, "num_classes": 10,
            "estimated_parameters": TINY_PARAMETERS_10_CLASSES,
            "fp16_parameter_bytes": TINY_PARAMETERS_10_CLASSES * 2,
            "fp32_parameter_bytes": TINY_PARAMETERS_10_CLASSES * 4,
            "limit_bytes": MAX_MODEL_BYTES, "size_status": "estimate_only",
            "latency_status": "pending_T4_measurement",
            "note": "Serialization/configuration overhead must be measured after actual export."}


def export_student(checkpoint_path: str | Path, output: str | Path) -> dict[str, Any]:
    import torch
    from model_overnight import build_overnight_model

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
    config = {"schema_version": 1, "artifact_type": "kd_student_deployment",
              "backbone_name": TINY_BACKBONE, "model_config": model_config,
              "image_size": IMAGE_SIZE, "class_ids": CLASS_IDS,
              "precision": "float16", "batch_size": 1,
              "tta": "identity", "ensemble": False,
              "aggregation": "study_mean_probability",
              "preprocessing": checkpoint["preprocessing"],
              "pretrained_provenance": checkpoint["pretrained_provenance"],
              "teacher_provenance": checkpoint["teacher_provenance"],
              "distillation": checkpoint.get("distillation", {}),
              "source_checkpoint_sha256": source_sha256,
              "source_manifest_sha256": checkpoint["source_manifest_sha256"],
              "official_valid_manifest_sha256": checkpoint["official_valid_manifest_sha256"],
              "source_validation": checkpoint.get("validation", {}),
              "weights_source": "ema", "gates": None,
              "inference_files": ["student_fp16.pt", "config.json"],
              "latency_status": "pending_T4_measurement",
              "fp16_validation_status": "pending_same_deployment_validation"}
    output.mkdir(parents=True, exist_ok=False)
    torch.save(state, output / "student_fp16.pt")
    config["weights_sha256"] = sha256_file(output / "student_fp16.pt")
    validate_deployment_config(config)
    write_json(output / "config.json", config)
    inventory = deployment_inventory(output, config)
    report = {**estimate_size(), **inventory,
              "source_checkpoint_sha256": source_sha256,
              "export_directory": str(output), "size_status": "measured",
              "fp16_validation_status": "pending_same_deployment_validation"}
    write_json(output / "export_report.json", report)
    if not inventory["size_pass"]:
        raise ValueError("Actual deployment files exceed the 500MB limit")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    import json
    if args.estimate_only:
        print(json.dumps(estimate_size(), ensure_ascii=False, indent=2))
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required unless --estimate-only")
    print(json.dumps(export_student(args.checkpoint, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
