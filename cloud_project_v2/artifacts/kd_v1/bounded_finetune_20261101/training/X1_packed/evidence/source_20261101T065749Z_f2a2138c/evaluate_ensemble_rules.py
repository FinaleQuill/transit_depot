"""Evaluate only the pre-registered fixed ensemble rules on the official valid.

Rules and member weights are frozen in a JSON plan before the new member is
evaluated.  This script refuses test-selected plans, builds the fixed class-wise
combinations, and delegates AUC/slice/patient-bootstrap arithmetic to the
existing CPU comparison tool.  No threshold or weight is fitted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from deploy_kd import sha256_file
from compare_student_candidates import compare, normalized_weights
from student_experiment_evaluation import write_new_json


def _read_json(path):
    return json.loads(Path(path).read_text())


def _member_map(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Members must be NAME=VALIDATION_DIR")
        name, directory = item.split("=", 1)
        if not name or not directory or name in result:
            raise ValueError("Member names must be unique and nonempty")
        result[name] = Path(directory)
    return result


def _parse_rule(rule, ensemble_members, root_names):
    if not isinstance(rule, dict) or not isinstance(rule.get("name"), str) or not rule["name"]:
        raise ValueError("Every rule needs a unique nonempty name")
    names = rule.get("members", list(ensemble_members))
    weights = rule.get("weights")
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(name not in ensemble_members for name in names)
            or not isinstance(weights, list) or len(weights) != len(names)):
        raise ValueError(f"Malformed registered rule: {rule.get('name')}")
    normalized_weights(weights, len(names))
    by_name = dict(zip(names, weights))
    rows = []
    for root in root_names:
        rows.append(by_name.get(root, [0.0] * 10))
    normalized_weights(rows, len(root_names))
    return rule["name"], rows


def evaluate_rules(*, members, extra, rules, manifest, output, bootstrap, seed, reference):
    ensembles = _member_map(members)
    extras = _member_map(extra) if extra else {}
    overlap = set(ensembles) & set(extras)
    if overlap:
        raise ValueError("Member and extra names must be disjoint: " + ", ".join(sorted(overlap)))
    plan = _read_json(rules)
    if (plan.get("registered_before_new_model_evaluation") is not True
            or plan.get("test_used") is not False
            or plan.get("valid_used_for_rule_registration") is not False):
        raise ValueError("Rules must be pre-registered without test/valid fitting")
    plan_members = plan.get("members")
    if plan_members != list(ensembles):
        raise ValueError("Rules plan member order differs from the supplied ensemble members")
    rules_list = plan.get("rules")
    if not isinstance(rules_list, list) or not rules_list:
        raise ValueError("No registered rules")
    root_names = list(ensembles) + list(extras)
    combinations = {}
    for rule in rules_list:
        name, rows = _parse_rule(rule, list(ensembles), root_names)
        if name in combinations:
            raise ValueError("Duplicate registered rule name")
        combinations[name] = rows
    reference = reference or plan.get("reference") or list(ensembles)[0]
    if reference not in root_names:
        raise ValueError("Reference must be one of the evaluated candidates")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new rule-evaluation output directory")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(combinations, stream)
        combinations_path = Path(stream.name)
    try:
        result = compare(
            validation_dirs=list(ensembles.values()) + list(extras.values()),
            names=root_names,
            manifest=manifest,
            output=output,
            combinations_json=combinations_path,
            bootstrap=bootstrap,
            seed=seed,
            reference=reference,
        )
    finally:
        combinations_path.unlink(missing_ok=True)
    registered = {name: result["candidates"][name] for name in combinations}
    receipt = {
        "rules_sha256": sha256_file(rules),
        "manifest_sha256": result["manifest_sha256"],
        "members": {name: str(path.resolve(strict=True)) for name, path in ensembles.items()},
        "extra_candidates": {name: str(path.resolve(strict=True)) for name, path in extras.items()},
        "registered_rules": plan,
        "registered_scores": {name: {"macro_auc": score["macro_auc"],
                                     "macro_auc_delta_from_reference": score.get("macro_auc_delta_from_reference"),
                                     "per_class_auc": score["per_class_auc"]}
                              for name, score in registered.items()},
        "reference": reference,
        "bootstrap": bootstrap,
        "bootstrap_seed": seed,
        "weights_fitted": False,
        "thresholds_fitted": False,
        "test_used": False,
        "note": "Only the pre-registered rules are decision candidates; the tool's automatic equal_mean remains diagnostic.",
    }
    write_new_json(output / "registered_rules.json", plan)
    write_new_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True, metavar="NAME=VALIDATION_DIR")
    parser.add_argument("--extra", nargs="*", default=[], metavar="NAME=VALIDATION_DIR")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "data/clean_v1_reviewed/valid_manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--reference")
    args = parser.parse_args(argv)
    receipt = evaluate_rules(members=args.members, extra=args.extra, rules=args.rules,
                             manifest=args.manifest, output=args.output,
                             bootstrap=args.bootstrap, seed=args.seed,
                             reference=args.reference)
    print(json.dumps(receipt["registered_scores"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
