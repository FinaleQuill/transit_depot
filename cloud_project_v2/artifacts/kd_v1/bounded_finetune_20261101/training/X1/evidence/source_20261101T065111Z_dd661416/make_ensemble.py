"""Assemble a fixed 2-3 member student ensemble from audited FP16 exports.

The class-wise weights and member views must already be recorded in a
pre-registration plan before this script runs.  The script only copies and
hash-binds immutable exports; it never fits weights on validation or test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from deploy_kd import (CLASS_IDS, MAX_MODEL_BYTES, normalized_model_weights,
                       sha256_file)
from student_experiment_evaluation import write_new_json

ALLOWED_BACKBONES = {
    "convnext_tiny.fb_in22k_ft_in1k_384",
    "convnext_base.fb_in22k_ft_in1k_384",
}
ALLOWED_VIEWS = ("identity", "hflip2")
ALLOWED_DECODERS = (None, "JPEG_draft_target768")
TINY_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
BASE_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def registered_ensemble_rule(plan, *, rule_name, names, ttas, weights, decoder,
                             allow_latency_fallback=False):
    """Bind actual member-indexed arguments to one explicitly named plan rule."""
    if (plan.get("test_used") is not False
            or plan.get("registered_before_new_model_evaluation") is not True):
        raise ValueError("Selection plan must be pre-registered without test use")
    plan_id = plan.get("plan_id")
    if plan_id == "auc820_20261031":
        if plan.get("valid_used_for_rule_registration") is not False:
            raise ValueError("Legacy plan registration declaration changed")
        primary_decoder, fallback_decoder = None, None
    elif plan_id == "bounded_study_finetune_20261101":
        evaluation = plan.get("evaluation", {})
        if (evaluation.get("tta") != "hflip2"
                or evaluation.get("decoder_primary") != "original full decode"
                or not _is_sha256(plan.get("manifest_sha256"))):
            raise ValueError("Fine-tuning plan must explicitly bind hflip2, full decode and valid SHA256")
        primary_decoder = None
        fallback_decoder = ("JPEG_draft_target768" if evaluation.get("decoder_bounded_fallback")
            == "JPEG draft768 if selected full decode model exceeds preferred80ms; revalidate fully" else None)
    else:
        raise ValueError("Unsupported selection-plan schema")
    rules = plan.get("rules")
    if not isinstance(rules, list) or not rules or any(not isinstance(rule, dict) for rule in rules):
        raise ValueError("Selection plan must contain explicit registered rules")
    rule_names = [rule.get("name") for rule in rules]
    if (any(not isinstance(name, str) or not name for name in rule_names)
            or len(set(rule_names)) != len(rule_names)):
        raise ValueError("Registered rule names must be unique and nonempty")
    selected_name = rule_name
    expected_ttas = {name: "hflip2" for name in names}
    view_fallback = False
    if (plan_id == "auc820_20261031" and rule_name == "R2b_A_identity"
            and plan.get("latency_fallback_rule", {}).get("name") == rule_name):
        if not allow_latency_fallback:
            raise ValueError("The registered identity fallback requires explicit --allow-latency-fallback")
        selected_name = "R2_B_B2_A_class1_base"
        expected_ttas["A"] = "identity"
        view_fallback = True
    if selected_name not in rule_names:
        raise ValueError("Rule name is absent from the supplied pre-registration plan")
    rule = rules[rule_names.index(selected_name)]
    registered_names = rule.get("members")
    declared_members = plan.get("members")
    if (not isinstance(registered_names, list) or not 2 <= len(registered_names) <= 3
            or any(not isinstance(name, str) for name in registered_names)
            or len(set(registered_names)) != len(registered_names)
            or set(registered_names) != set(names)
            or not isinstance(declared_members, list)
            or not set(registered_names) <= set(declared_members)):
        raise ValueError("Actual named members differ from the selected registered rule")
    registered_weights = normalized_model_weights(rule.get("weights"), len(registered_names))
    by_name = {name: row for name, row in zip(registered_names, registered_weights)}
    actual_weights = normalized_model_weights(weights, len(names))
    expected_weights = np.asarray([by_name[name] for name in names], dtype=np.float64)
    if not np.array_equal(actual_weights, expected_weights):
        raise ValueError("Actual per-member class weights differ from the selected registered rule")
    if ttas != [expected_ttas[name] for name in names]:
        raise ValueError("Actual member TTAs differ from the registered hflip2 rule/fallback")
    decoder_fallback = decoder != primary_decoder
    if decoder_fallback and not (allow_latency_fallback and fallback_decoder is not None
                                 and decoder == fallback_decoder):
        raise ValueError("Decoder differs from the registered primary; no explicit registered fallback applies")
    expected_identities = {}
    for name in names:
        if name in ("A", "B", "B2"):
            expected_identities[name] = {"backbone": TINY_BACKBONE if name == "A" else BASE_BACKBONE,
                                         "epoch": {"A": 9, "B": 10, "B2": 12}[name]}
        elif plan_id == "bounded_study_finetune_20261101" and name in ("X1", "X2"):
            training = plan.get("training", {}).get(name, {})
            backbone, epoch = training.get("backbone"), training.get("epochs")
            if backbone not in ALLOWED_BACKBONES or type(epoch) is not int or epoch < 1:
                raise ValueError("Registered new member lacks its actual backbone/final epoch")
            expected_identities[name] = {"backbone": backbone, "epoch": epoch}
        else:
            raise ValueError("Member identity is not declared in a supported selection plan")
    return {"rule_name": rule_name, "probability_rule_name": selected_name,
            "member_order": names, "normalized_weights": expected_weights.tolist(),
            "ttas": ttas, "decoder_candidate": decoder,
            "latency_fallback": view_fallback or decoder_fallback,
            "requires_new_validation": view_fallback or decoder_fallback,
            "expected_member_identities": expected_identities}


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
            or not _is_sha256(config.get("source_checkpoint_sha256"))
            or not _is_sha256(config.get("official_valid_manifest_sha256"))
            or type(config.get("source_epoch")) is not int or config["source_epoch"] < 1
            or not isinstance(weights_file, str)
            or Path(weights_file).name != weights_file
            or not (root / weights_file).is_file()
            or sha256_file(root / weights_file) != config.get("weights_sha256")):
        raise ValueError(f"Invalid experimental student export: {root}")
    if expected_backbone is not None and backbone != expected_backbone:
        raise ValueError(f"Member backbone differs from declaration: {root}")
    return root, config


def make_ensemble(*, members, names, ttas, weights, decoder, selection_plan, rule_name, output,
                  allow_latency_fallback=False, selection_plan_sha256=None):
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
    plan_hash = sha256_file(selection_plan)
    if selection_plan_sha256 is not None and plan_hash != selection_plan_sha256:
        raise ValueError("Selection-plan SHA256 differs from its frozen declaration")
    plan = _read_json(selection_plan)
    registration = registered_ensemble_rule(plan, rule_name=rule_name, names=names, ttas=ttas,
        weights=weights, decoder=decoder, allow_latency_fallback=allow_latency_fallback)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new ensemble output directory")

    identities = []
    input_hashes = {str(Path(selection_plan).resolve()): plan_hash}
    for name, root, tta in zip(names, member_dirs, ttas):
        expected = registration["expected_member_identities"][name]
        source, config = _member_identity(root, expected_backbone=expected["backbone"])
        if config["source_epoch"] != expected["epoch"]:
            raise ValueError(f"Member is not the registered source/final EMA epoch: {name}")
        if output == source or output.is_relative_to(source):
            raise ValueError("Ensemble output cannot be inside an immutable member export")
        input_hashes[str(source / "config.json")] = sha256_file(source / "config.json")
        input_hashes[str(source / config["weights_file"])] = config["weights_sha256"]
        identities.append((name, source, config, tta))
    valid_hashes = {config["official_valid_manifest_sha256"] for _, _, config, _ in identities}
    if len(valid_hashes) != 1:
        raise ValueError("All ensemble members must bind the same official valid manifest")
    valid_hash = next(iter(valid_hashes))
    if plan.get("manifest_sha256", valid_hash) != valid_hash:
        raise ValueError("Ensemble members disagree with the registered valid manifest SHA256")
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("An ensemble source changed during preflight")

    output.mkdir(parents=True)
    members_config = []
    for name, source, config, tta in identities:
        destination = output / name
        destination.mkdir()
        shutil.copyfile(source / "config.json", destination / "config.json")
        shutil.copyfile(source / config["weights_file"], destination / config["weights_file"])
        if (sha256_file(destination / "config.json") != input_hashes[str(source / "config.json")]
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
        "selection_plan_sha256": plan_hash,
        "registered_rule": registration,
        "rule_name": rule_name,
        "official_valid_manifest_sha256": valid_hash,
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
              "registered_rule": registration,
              "source_inputs_sha256": input_hashes,
              "source_member_bytes_and_identity_preserved": True,
              "test_used_for_selection": False}
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("An ensemble source changed while assembling the registered candidate")
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
    parser.add_argument("--selection-plan-sha256", help="Optional independently frozen plan identity")
    parser.add_argument("--rule-name", required=True, help="Exact probability/fallback rule name recorded in selection plan")
    parser.add_argument("--allow-latency-fallback", action="store_true",
                        help="Explicitly use only the selected plan's registered view/decoder fallback; full revalidation is required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    weights = json.loads(args.weights_json.read_text())
    decoder = None if args.decoder == "none" else args.decoder
    print(json.dumps(make_ensemble(members=args.members, names=args.names, ttas=args.ttas,
                                   weights=weights, decoder=decoder,
                                   selection_plan=args.selection_plan,
                                   selection_plan_sha256=args.selection_plan_sha256,
                                   rule_name=args.rule_name,
                                   allow_latency_fallback=args.allow_latency_fallback,
                                   output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
