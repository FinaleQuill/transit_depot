"""Assemble a fixed 2-3 member student ensemble from audited FP16 exports.

The class-wise weights and member views must already be recorded in a
pre-registration plan before this script runs.  The script only copies and
hash-binds immutable exports; it never fits weights on validation or test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy_kd import (CLASS_IDS, MAX_MODEL_BYTES, normalized_model_weights,
                       sha256_file)
from student_experiment_evaluation import write_new_json

ALLOWED_BACKBONES = {
    "convnext_tiny.fb_in22k_ft_in1k_384",
    "convnext_base.fb_in22k_ft_in1k_384",
}
ALLOWED_VIEWS = ("identity", "hflip2")
ALLOWED_DECODERS = (None, "JPEG_draft_target768")


def _read_json(path):
    return json.loads(Path(path).read_text())


def _member_identity(root, expected_backbone=None):
    root = Path(root).resolve(strict=True)
    config = _read_json(root / "config.json")
    backbone = config.get("backbone_name")
    weights_file = config.get("weights_file")
    if (config.get("artifact_type") != "experimental_student_fp16"
            or config.get("precision") != "float16"
            or config.get("image_size") != 768
            or config.get("class_ids") != CLASS_IDS
            or backbone not in ALLOWED_BACKBONES
            or not isinstance(weights_file, str)
            or Path(weights_file).name != weights_file
            or not (root / weights_file).is_file()
            or sha256_file(root / weights_file) != config.get("weights_sha256")):
        raise ValueError(f"Invalid experimental student export: {root}")
    if expected_backbone is not None and backbone != expected_backbone:
        raise ValueError(f"Member backbone differs from declaration: {root}")
    return root, config


def make_ensemble(*, members, names, ttas, weights, decoder, selection_plan, output):
    member_dirs = [Path(p) for p in members]
    names = list(names)
    ttas = list(ttas) if ttas is not None else ["hflip2"] * len(member_dirs)
    if not 2 <= len(member_dirs) <= 3 or len(names) != len(member_dirs) or len(ttas) != len(member_dirs):
        raise ValueError("Require 2-3 members with one name and one TTA each")
    if len(set(names)) != len(names) or any(Path(n).name != n or n in {"", ".", ".."} for n in names):
        raise ValueError("Member names must be unique plain directory names")
    if any(tta not in ALLOWED_VIEWS for tta in ttas):
        raise ValueError("Member TTA must be identity or hflip2")
    if decoder not in ALLOWED_DECODERS:
        raise ValueError("Unknown decoder candidate")
    normalized_model_weights(weights, len(member_dirs))
    plan = _read_json(selection_plan)
    if (plan.get("test_used") is not False
            or plan.get("registered_before_new_model_evaluation") is not True
            or plan.get("valid_used_for_rule_registration") is not False):
        raise ValueError("Selection plan must be pre-registered without test/valid rule fitting")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new ensemble output directory")

    identities = []
    for name, root, tta in zip(names, member_dirs, ttas):
        source, config = _member_identity(root)
        identities.append((name, source, config, tta))

    output.mkdir(parents=True)
    members_config = []
    for name, source, config, tta in identities:
        destination = output / name
        destination.mkdir()
        config_bytes = (source / "config.json").read_bytes()
        weights_bytes = (source / config["weights_file"]).read_bytes()
        (destination / "config.json").write_bytes(config_bytes)
        (destination / config["weights_file"]).write_bytes(weights_bytes)
        if (sha256_file(destination / "config.json") != sha256_file(source / "config.json")
                or sha256_file(destination / config["weights_file"]) != config["weights_sha256"]):
            raise RuntimeError("Copied member bytes differ from their source")
        members_config.append({
            "directory": name,
            "backbone": config["backbone_name"],
            "config_sha256": sha256_file(destination / "config.json"),
            "weights_file": config["weights_file"],
            "weights_sha256": config["weights_sha256"],
            "tta": tta,
            "source_checkpoint_sha256": config.get("source_checkpoint_sha256"),
            "source_epoch": config.get("source_epoch"),
            "official_valid_manifest_sha256": config.get("official_valid_manifest_sha256"),
        })
    top_tta = "hflip2" if all(item["tta"] == "hflip2" for item in members_config) else "mixed"
    config = {
        "schema_version": 1,
        "artifact_type": "fixed_student_ensemble",
        "class_ids": CLASS_IDS,
        "precision": "float16",
        "batch_size": 1,
        "image_size": 768,
        "tta": top_tta,
        "ensemble": True,
        "gates": None,
        "decoder_candidate": decoder,
        "members": members_config,
        "member_count": len(members_config),
        "model_weights": weights,
        "selection_plan": plan,
        "selection_plan_sha256": sha256_file(selection_plan),
        "test_used_for_selection": False,
        "formal_T4_certification": False,
        "aggregation": "fixed class-wise image probability blend, then original Study probability mean",
    }
    write_new_json(output / "config.json", config)
    files = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
              "sha256": sha256_file(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    total = sum(item["bytes"] for item in files)
    if total > MAX_MODEL_BYTES:
        raise ValueError("Actual ensemble inference assets exceed 500,000,000 bytes")
    report = {"total_inference_bytes": total, "size_limit_bytes": MAX_MODEL_BYTES,
              "size_pass": True, "files": files, "model_weights": weights,
              "config_sha256": sha256_file(output / "config.json"),
              "selection_plan_sha256": config["selection_plan_sha256"],
              "test_used_for_selection": False}
    write_new_json(output / "make_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--ttas", nargs="+", choices=ALLOWED_VIEWS)
    parser.add_argument("--weights-json", type=Path, required=True,
                        help="Pre-registered M or Mx10 fixed member weights")
    parser.add_argument("--decoder", choices=["none", "JPEG_draft_target768"], default="none")
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    weights = json.loads(args.weights_json.read_text())
    decoder = None if args.decoder == "none" else args.decoder
    print(json.dumps(make_ensemble(members=args.members, names=args.names, ttas=args.ttas,
                                   weights=weights, decoder=decoder,
                                   selection_plan=args.selection_plan,
                                   output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
