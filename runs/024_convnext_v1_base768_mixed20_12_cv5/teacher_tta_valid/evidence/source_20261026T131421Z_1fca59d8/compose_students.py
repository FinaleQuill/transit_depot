"""Compose fixed single-student FP16 exports into an auditable probability ensemble.

Default is a read-only plan. --execute copies verified FP16 weights and writes a
new deployment config; it never loads a model, runs inference or starts training.
Weights/TTA must be chosen using train/validation and frozen before test use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from deploy_kd import (MAX_MODEL_BYTES, deployment_inventory, deployment_members, read_json,
                       sha256_file, validate_deployment_config, write_json)
from export_student import make_deployment_config


def composition_plan(model_dirs: Sequence[str | Path], *, output: str | Path,
                     tta: str = "hflip2", weights: Any = None) -> dict:
    if not 1 <= len(model_dirs) <= 4:
        raise ValueError("Choose one to four single-student exports")
    if tta not in {"identity", "hflip2"}:
        raise ValueError("TTA must be fixed identity or hflip2")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Composition output must be a new directory")
    members, sources, shared = [], [], None
    for index, directory in enumerate(model_dirs):
        root = Path(directory).resolve(strict=True)
        if destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError("Output must not overlap its source export directories")
        config = read_json(root / "config.json")
        validate_deployment_config(config)
        inventory = deployment_inventory(root, config)
        if not inventory["size_pass"]:
            raise ValueError("A source export already exceeds the deployment size budget")
        existing = deployment_members(config)
        if len(existing) != 1:
            raise ValueError("Compose independent single-student exports; nested ensembles are refused")
        member = dict(existing[0])
        weight_path = root / member["weights_file"]
        if sha256_file(weight_path) != member["weights_sha256"]:
            raise ValueError("Source FP16 weight checksum mismatch")
        declared = {"source_manifest_sha256": config.get("source_manifest_sha256"),
                    "fold_plan_sha256": config.get("fold_plan_sha256") or
                        member["teacher_provenance"][0].get("fold_plan_sha256"),
                    "official_valid_manifest_sha256": config.get("official_valid_manifest_sha256")}
        if shared is None:
            shared = declared
        elif declared != shared:
            raise ValueError("Students must share original train/fold plan and untouched official valid")
        target_name = f"student_{index:02d}_fp16.pt"
        source_config_hash = sha256_file(root / "config.json")
        member.update(weights_file=target_name, tta=tta,
                      source_export_config_sha256=source_config_hash)
        # A deployment member stores the student's actual training validation, even
        # when the chosen fixed inference TTA differs from that validation mode.
        for key in ("inference_files", "members", "model_weights", "ensemble", "image_sizes",
                    "gates", "schema_version", "artifact_type", "latency_status",
                    "fp16_validation_status"):
            member.pop(key, None)
        members.append(member)
        sources.append({"model_dir": str(root), "source_config_sha256": source_config_hash,
                        "source_weights": str(weight_path), "target_weights": target_name,
                        "weights_sha256": member["weights_sha256"], "bytes": weight_path.stat().st_size})
    config = make_deployment_config(members, **shared, model_weights=weights)
    config_bytes = len((json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    total = sum(s["bytes"] for s in sources) + config_bytes
    if total > MAX_MODEL_BYTES:
        raise ValueError(f"Planned actual weight/config bytes {total} exceed {MAX_MODEL_BYTES}")
    return {"output": str(destination), "sources": sources, "config": config,
            "planned_inference_bytes": total, "limit_bytes": MAX_MODEL_BYTES,
            "size_pass": True, "latency_status": "pending_T4_measurement",
            "fp16_validation_status": "pending_same_deployment_validation"}


def compose_students(model_dirs: Sequence[str | Path], *, output: str | Path,
                     tta: str = "hflip2", weights: Any = None, execute: bool = False) -> dict:
    plan = composition_plan(model_dirs, output=output, tta=tta, weights=weights)
    if not execute:
        return {"execute": False, **plan}
    destination = Path(plan["output"])
    destination.mkdir(parents=True, exist_ok=False)
    for source in plan["sources"]:
        root = Path(source["model_dir"])
        if sha256_file(root / "config.json") != source["source_config_sha256"]:
            raise RuntimeError("Source deployment config changed during composition")
        target = destination / source["target_weights"]
        shutil.copyfile(source["source_weights"], target)
        if (sha256_file(target) != source["weights_sha256"]
                or sha256_file(source["source_weights"]) != source["weights_sha256"]):
            raise RuntimeError("Source or copied FP16 weights changed during composition")
    write_json(destination / "config.json", plan["config"])
    inventory = deployment_inventory(destination, plan["config"])
    report = {"execute": True, "export_directory": str(destination),
              "member_count": len(plan["config"]["members"]),
              "image_sizes": plan["config"]["image_sizes"], "tta": plan["config"]["tta"],
              "sources": plan["sources"], **inventory,
              "latency_status": "pending_T4_measurement",
              "fp16_validation_status": "pending_same_deployment_validation"}
    write_json(destination / "composition_report.json", report)
    if not inventory["size_pass"]:
        raise ValueError("Actual inference files exceed 500MB")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tta", choices=("identity", "hflip2"), default="hflip2")
    weight_source = parser.add_mutually_exclusive_group()
    weight_source.add_argument("--weights", type=float, nargs="+", help="One nonnegative fixed scalar per student")
    weight_source.add_argument("--weights-json", type=Path, help="JSON array of M scalar weights or Mx10 fixed class weights")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    weights = json.loads(args.weights_json.read_text()) if args.weights_json else args.weights
    report = compose_students(args.model_dirs, output=args.output, tta=args.tta,
                              weights=weights, execute=args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
