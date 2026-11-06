"""Patient/content-isolated five-fold plans from accepted TRAIN records only.

Official validation stays an independent holdout.  Test is never read here.
Only CSV metadata is accessed: no image decode, model, or training is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

from dataset import (CLASS_IDS, DEFAULT_TRAIN_MANIFEST, DEFAULT_VALID_MANIFEST,
                     NUM_CLASSES, PROJECT_ROOT, read_manifest)


DEFAULT_FOLD_PLAN = PROJECT_ROOT / "data" / "cv5_overnight" / "folds.json"
SCHEMA = "cxr_patient_content_cv_v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _study_records(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    studies: dict[str, dict[str, Any]] = {}
    for row in records:
        if row["split"] != "train" or row["targets"] is None:
            raise ValueError("Fold strata and class counts may use original train only")
        subject = row.get("subject_id")
        if not subject or row["study_key"] != f"{subject}/{row.get('study_id')}":
            raise ValueError("Study/patient identity is malformed")
        if len(row["targets"]) != NUM_CLASSES or any(value not in (0, 1) for value in row["targets"]):
            raise ValueError("Study labels must be unchanged ten-class binary labels")
        key = row["study_key"]
        labels = tuple(int(value) for value in row["targets"])
        if key not in studies:
            studies[key] = {"patient": subject, "labels": labels, "images": 0}
        elif studies[key]["patient"] != subject or studies[key]["labels"] != labels:
            raise ValueError("Study has inconsistent patient or labels")
        studies[key]["images"] += 1
    return studies


def _content_groups(records: Sequence[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    patients = sorted({row["subject_id"] for row in records})
    union = _UnionFind(patients)
    edges = {"file_sha256": 0, "pixel_sha256": 0}
    for field in edges:
        seen: dict[str, str] = {}
        for row in records:
            digest = row.get(field, "")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Every train image requires its accepted {field}")
            previous = seen.setdefault(digest, row["subject_id"])
            if previous != row["subject_id"]:
                union.union(previous, row["subject_id"])
                edges[field] += 1
    return {patient: union.find(patient) for patient in patients}, edges


def _check_holdout(records: Sequence[dict[str, Any]], holdout: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # Used solely to enforce isolation; no holdout label statistics are fitted.
    if {row["subject_id"] for row in records} & {row["subject_id"] for row in holdout}:
        raise ValueError("Official holdout and train share patients")
    if {row["study_key"] for row in records} & {row["study_key"] for row in holdout}:
        raise ValueError("Official holdout and train share Studies")
    for field in ("file_sha256", "pixel_sha256"):
        train_hashes = {row[field] for row in records}
        if train_hashes & {row[field] for row in holdout}:
            raise ValueError(f"Official holdout/train contain exact duplicate {field}; resolve leakage first")
    return {"patients_disjoint": True, "studies_disjoint": True,
            "file_content_disjoint": True, "pixel_content_disjoint": True,
            "holdout_used_for_stratification": False, "test_read": False}


def _statistics(records: Sequence[dict[str, Any]], studies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = {row["study_key"] for row in records}
    positives = [sum(studies[key]["labels"][c] for key in keys) for c in range(NUM_CLASSES)]
    return {"images": len(records), "studies": len(keys),
            "patients": len({row["subject_id"] for row in records}),
            "positive_studies": positives,
            "negative_studies": [len(keys) - value for value in positives]}


def build_fold_plan(train_manifest: str | Path = DEFAULT_TRAIN_MANIFEST, *,
                    official_valid_manifest: str | Path = DEFAULT_VALID_MANIFEST,
                    n_splits: int = 5, seed: int = 43) -> dict[str, Any]:
    if type(n_splits) is not int or n_splits != 5:
        raise ValueError("This overnight protocol requires exactly five folds")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    train_manifest = Path(train_manifest).resolve()
    official_valid_manifest = Path(official_valid_manifest).resolve()
    records = read_manifest(train_manifest, split="train")
    holdout = read_manifest(official_valid_manifest, split="valid")
    isolation = _check_holdout(records, holdout)
    studies = _study_records(records)
    patient_groups, duplicate_edges = _content_groups(records)
    groups: dict[str, dict[str, Any]] = {}
    for key, study in studies.items():
        group_id = patient_groups[study["patient"]]
        group = groups.setdefault(group_id, {"studies": [], "patients": set(),
                                             "positives": [0] * NUM_CLASSES, "images": 0})
        group["studies"].append(key)
        group["patients"].add(study["patient"])
        group["images"] += study["images"]
        for c in range(NUM_CLASSES):
            group["positives"][c] += study["labels"][c]
    if len(groups) < n_splits:
        raise ValueError("Too few independent patient/content groups")
    total = _statistics(records, studies)
    if min(total["positive_studies"] + total["negative_studies"]) < n_splits:
        raise ValueError("Each class needs at least five positive and five negative Studies")
    rng = random.Random(seed)
    ordered = sorted(groups)
    rng.shuffle(ordered)
    # Large groups containing rare labels are assigned first.  The objective
    # balances study-level positives AND negatives, then study/patient/image load.
    ordered.sort(key=lambda key: (
        sum(groups[key]["positives"][c] / total["positive_studies"][c]
            for c in range(NUM_CLASSES)), len(groups[key]["studies"])), reverse=True)
    targets = ([value / n_splits for value in total["positive_studies"]]
               + [value / n_splits for value in total["negative_studies"]]
               + [total["studies"] / n_splits, total["patients"] / n_splits,
                  total["images"] / n_splits])
    dimension_weights = [1.0] * (2 * NUM_CLASSES) + [0.5, 0.25, 0.25]
    loads = [[0.0] * len(targets) for _ in range(n_splits)]
    group_to_fold: dict[str, int] = {}
    for group_id in ordered:
        group = groups[group_id]
        count = len(group["studies"])
        vector = (group["positives"] + [count - value for value in group["positives"]]
                  + [count, len(group["patients"]), group["images"]])
        candidates = list(range(n_splits))
        rng.shuffle(candidates)
        def cost(fold: int) -> float:
            return sum(weight * ((loads[fold][d] + value) ** 2 - loads[fold][d] ** 2) / targets[d] ** 2
                       for d, (value, weight) in enumerate(zip(vector, dimension_weights)))
        fold = min(candidates, key=cost)
        group_to_fold[group_id] = fold
        loads[fold] = [old + value for old, value in zip(loads[fold], vector)]
    study_to_fold = {key: group_to_fold[patient_groups[study["patient"]]]
                     for key, study in studies.items()}
    folds = []
    for fold in range(n_splits):
        validation = [row for row in records if study_to_fold[row["study_key"]] == fold]
        training = [row for row in records if study_to_fold[row["study_key"]] != fold]
        valid_stats, train_stats = _statistics(validation, studies), _statistics(training, studies)
        if min(valid_stats["positive_studies"] + valid_stats["negative_studies"]
               + train_stats["positive_studies"] + train_stats["negative_studies"]) <= 0:
            raise ValueError("A fold lacks class-positive or class-negative Studies; change the seed")
        folds.append({"fold": fold, "train": train_stats, "valid": valid_stats})
    digest = sha256_file(train_manifest)
    plan = {"schema": SCHEMA, "n_splits": n_splits, "seed": seed,
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "algorithm": "deterministic rare-label-first patient/content-group greedy multilabel balance",
            "split_policy": "train-only CV; official valid independent holdout; test excluded",
            "class_ids": list(CLASS_IDS), "source_manifest_sha256": digest,
            "train_manifest_sha256": digest,
            "source_train_manifest": {"path": str(train_manifest), "sha256": digest},
            "official_holdout": {"path": str(official_valid_manifest),
                                  "sha256": sha256_file(official_valid_manifest)},
            "source_code_sha256": sha256_file(__file__), "isolation_audit": isolation,
            "source_train_statistics": total, "duplicate_patient_edges": duplicate_edges,
            "independent_patient_content_groups": len(groups),
            "patient_to_group": patient_groups, "group_to_fold": group_to_fold,
            "study_to_patient": {key: value["patient"] for key, value in studies.items()},
            "study_to_fold": study_to_fold, "folds": folds}
    return plan


def read_fold_plan(plan_path: str | Path = DEFAULT_FOLD_PLAN) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA or plan.get("n_splits") != 5:
        raise ValueError("Unsupported fold-plan schema or fold count")
    for field in ("source_train_manifest", "official_holdout"):
        source = plan[field]
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"Accepted manifest changed after split creation: {field}")
    if (plan["source_manifest_sha256"] != plan["source_train_manifest"]["sha256"]
            or plan["train_manifest_sha256"] != plan["source_manifest_sha256"]):
        raise ValueError("Fold plan contains conflicting source fingerprints")
    return plan


def load_fold_records(plan_path: str | Path, fold: int, role: str) -> list[dict[str, Any]]:
    """Fold roles select rows; original record['split'] remains 'train'."""
    if type(fold) is not int or not 0 <= fold < 5 or role not in {"train", "valid"}:
        raise ValueError("fold must be 0..4 and role train/valid")
    plan = read_fold_plan(plan_path)
    records = read_manifest(plan["source_train_manifest"]["path"], split="train")
    studies = _study_records(records)
    mapping = plan["study_to_fold"]
    if set(mapping) != set(studies) or any(type(value) is not int or not 0 <= value < 5
                                         for value in mapping.values()):
        raise ValueError("Fold mapping does not exactly cover accepted train Studies")
    patients, _ = _content_groups(records)
    if patients != plan["patient_to_group"]:
        raise ValueError("Patient/content grouping changed")
    group_folds: dict[str, int] = {}
    for key, item in studies.items():
        if plan["study_to_patient"].get(key) != item["patient"]:
            raise ValueError("Study patient identity changed")
        group_id = patients[item["patient"]]
        previous = group_folds.setdefault(group_id, mapping[key])
        if previous != mapping[key] or plan["group_to_fold"].get(group_id) != mapping[key]:
            raise ValueError("A patient/content group leaks across folds")
    selected = [row for row in records if (mapping[row["study_key"]] == fold) == (role == "valid")]
    if _statistics(selected, studies) != plan["folds"][fold][role]:
        raise ValueError("Fold record counts or original labels differ from the split plan")
    return selected


def ensure_fold_plan(plan_path: str | Path = DEFAULT_FOLD_PLAN, *,
                     train_manifest: str | Path = DEFAULT_TRAIN_MANIFEST,
                     official_valid_manifest: str | Path = DEFAULT_VALID_MANIFEST,
                     n_splits: int = 5, seed: int = 43) -> dict[str, Any]:
    """Create a new plan, or strictly validate an identical existing one."""
    target = Path(plan_path).resolve()
    if target.exists():
        plan = read_fold_plan(target)
        if (plan["seed"] != seed or plan["n_splits"] != n_splits
                or Path(plan["source_train_manifest"]["path"]).resolve() != Path(train_manifest).resolve()
                or Path(plan["official_holdout"]["path"]).resolve() != Path(official_valid_manifest).resolve()):
            raise ValueError("Existing fold plan uses another source, seed, or split count; use a new path")
        for fold in range(5):
            load_fold_records(target, fold, "valid")
        return plan
    plan = build_fold_plan(train_manifest, official_valid_manifest=official_valid_manifest,
                           n_splits=n_splits, seed=seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an accepted plan.
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_FOLD_PLAN)
    parser.add_argument("--seed", type=int, default=43)
    arguments = parser.parse_args()
    plan = ensure_fold_plan(arguments.output, train_manifest=arguments.train_manifest,
                            official_valid_manifest=arguments.valid_manifest, seed=arguments.seed)
    print(json.dumps({"plan": str(arguments.output.resolve()),
                      "sha256": sha256_file(arguments.output), "folds": plan["folds"],
                      "isolation_audit": plan["isolation_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
