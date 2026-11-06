"""Prepare a reproducible compliance directory without downloading or running models.

All source files are read-only. Original pretraining weights are copied for
license/source verification, not loaded by inference and not part of model/.
Missing submission/timing/FP16-validation evidence leaves the package pending.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
import zipfile

from deploy_kd import (deployment_inventory, read_json, read_manifest, read_submission,
                       sha256_file, validate_deployment_config, write_json)
from inference_runtime import ADDONS_ARCHIVE_SHA256

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "train_teacher.py", "run_teachers.py", "run_overnight.sh", "train_distill.py", "extract_soft_labels.py",
    "audit_trail.py", "training_evidence_monitor.py",
    "kd_common.py", "distill_data.py", "model_overnight.py", "train_overnight.py",
    "dataset.py", "dataset_overnight.py", "cv_splits.py", "runtime_compat.py",
    "deploy_kd.py", "predict.py", "cost.py", "export_student.py", "validate_student.py",
    "package_kd_submission.py", "requirements_kd.txt", "requirements_inference.txt",
    "planned_handoff.py", "teacher_collection.py", "validate_teacher.py",
    "run_mixed_teachers.py", "run_mixed_teachers.sh",
    "distill_tta_data.py", "extract_soft_labels_tta.py", "compose_students.py",
    "run_student_experiments.py", "compare_student_candidates.py", "build_training_evidence.py",
    "validate_teacher_tta.py",
    "diagnose_teacher_epochs.py",
    "inference_reproducibility.py", "inference_runtime.py", "inference_inputs.py",
    "fast_lanczos.py", "prepare_inference.py",
)
NATIVE_SOURCE_FILES = (
    "native_lanczos/__init__.py", "native_lanczos/resample.c", "native_lanczos/setup.py",
    "native_lanczos/README.md", "native_lanczos/upstream/LICENSE",
    "native_lanczos/upstream/PROVENANCE.json", "native_lanczos/upstream/Resample.c",
    "native_lanczos/upstream/Imaging.h",
)
ADDONS_FILES = ("competition_addons_cp312_linux_x86_64.tar.gz",
                "competition_addons_archive_manifest.json")
ADDONS_SOURCE = "artifacts/kd_v1/optimization_2080ti_20261029/environment"
MANIFEST_FILES = (
    "data/clean_v1_reviewed/train_clean.csv",
    "data/clean_v1_reviewed/valid_manifest.csv",
    "data/clean_v1_reviewed/test_manifest.csv",
)
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024


def inference_support_files(source: Path) -> list[tuple[Path, Path]]:
    """Explicit runtime allowlist; never include build/dist/cache directories."""
    files = [(source / name, Path("code") / name) for name in NATIVE_SOURCE_FILES]
    extension = source / "native_lanczos/_core.cpython-312-x86_64-linux-gnu.so"
    files.append((extension, Path("code/native_lanczos") / extension.name))
    vendor = source / "vendor"
    if not all((vendor / name).is_file() for name in ADDONS_FILES):
        vendor = source / ADDONS_SOURCE
    archive, manifest = [vendor / name for name in ADDONS_FILES]
    for path, _ in files:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Prepared native inference source or CPython3.12 extension missing: {path}")
    if not archive.is_file() or not manifest.is_file():
        raise FileNotFoundError("Pinned public-addon archive and manifest are required for an offline project")
    if (archive.is_symlink() or manifest.is_symlink()
            or sha256_file(archive) != ADDONS_ARCHIVE_SHA256
            or read_json(manifest).get("archive_sha256") != ADDONS_ARCHIVE_SHA256):
        raise ValueError("Pinned public-addon archive or manifest SHA binding differs")
    files.extend((path, Path("code/vendor") / path.name) for path in (archive, manifest))
    return files


def _cost_semantics_complete(cost: Mapping[str, Any], submission: Mapping[str, Any] | None) -> bool:
    """Do not upgrade an old boolean-only cost report into current evidence."""
    protocol, comparison = cost.get("protocol", {}), cost.get("comparison", {})
    if (cost.get("reproducibility", {}).get("seed") != 43
            or any(protocol.get(key) != value for key, value in {
                "measured_images": 100, "warmup_images": 5, "rounds": 3,
                "batch_size": 1, "execution": "serial"}.items())
            or comparison.get("status") != "matched"
            or comparison.get("coverage", {}).get("complete_studies_proven") is not True):
        return False
    studies, images = comparison.get("rounds"), comparison.get("image_rounds")
    if not isinstance(studies, list) or not isinstance(images, list) or len(studies) != 3 or len(images) != 3:
        return False
    for report in studies + images:
        error = report.get("max_absolute_error")
        if (report.get("match") is not True or report.get("semantic_match") is not True
                or report.get("identity_coverage_match") is not True
                or report.get("mismatched_study_count") != 0
                or report.get("semantic_mismatched_study_count") != 0
                or not isinstance(error, (int, float)) or not math.isfinite(error) or error > 1e-7):
            return False
    if any(report.get("compared_images") != 100 or report.get("expected_images") != 100
           or report.get("mismatched_image_count") != 0 for report in images):
        return False
    return (submission is not None and bool(submission.get("image_predictions_sha256"))
            and comparison.get("image_submission_sha256") == submission["image_predictions_sha256"])


def deployment_members(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return actual inference members; preserve the original single-model format."""
    return list(config["members"]) if "members" in config else [config]


def ancestry_runs(config: Mapping[str, Any]) -> tuple[dict, dict]:
    members = deployment_members(config)
    students = {"student" if len(members) == 1 else f"student_{i}": member
                for i, member in enumerate(members)}
    teachers = {}
    for member in members:
        for teacher in member.get("teacher_provenance", []):
            role = f"teacher_fold_{teacher['fold_index']}"
            previous = teachers.setdefault(role, teacher)
            if previous.get("checkpoint_sha256") != teacher.get("checkpoint_sha256"):
                raise ValueError("This evidence bundle requires students from the same frozen teacher collection")
    return students, teachers


def _valid_evidence_time(value: Any) -> bool:
    try:
        return isinstance(value, str) and datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _evidence_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """Freeze a bounded small file; an appending log remains explicitly partial."""
    if source.is_symlink() or source.suffix.lower() in {".pt", ".pth", ".safetensors", ".ckpt", ".bin",
                                                      ".onnx", ".engine", ".jpg", ".jpeg", ".png"}:
        raise ValueError("Evidence must not reference model weights, images or symlinks")
    before = source.stat()
    if not source.is_file() or before.st_size > MAX_EVIDENCE_FILE_BYTES:
        raise ValueError("Training evidence must consist of small regular files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = before.st_size
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        opened = os.fstat(incoming.fileno())
        while remaining:
            block = incoming.read(min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError("Evidence was truncated during snapshot")
            outgoing.write(block)
            digest.update(block)
            remaining -= len(block)
    after = source.stat()
    if sha256_file(destination) != digest.hexdigest():
        raise ValueError("Training evidence snapshot failed SHA-256 verification")
    stable = ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
              == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
              and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino))
    return {"source_path": str(source), "bytes": before.st_size, "sha256": digest.hexdigest(),
            "stable_during_snapshot": stable, "source_mtime_ns": before.st_mtime_ns}


def _read_evidence_ledger(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Verify copied records without repairing or changing the source ledger."""
    raw = path.read_bytes()
    complete = not raw or raw.endswith(b"\n")
    rows, previous = [], None
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Checkpoint evidence contains a corrupt complete record") from exc
        if not isinstance(row, dict) or row.get("audit_schema_version") != 1:
            raise ValueError("Checkpoint evidence schema is unsupported")
        unhashed = {key: value for key, value in row.items() if key != "record_sha256"}
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        if (row.get("previous_record_sha256") != previous
                or row.get("record_sha256") != hashlib.sha256(canonical).hexdigest()):
            raise ValueError("Checkpoint evidence hash chain is inconsistent")
        previous = row["record_sha256"]
        if row.get("event_type") == "checkpoint_saved":
            if (type(row.get("epoch")) is not int or row["epoch"] < 1
                    or row.get("role") not in {"last", "best"}
                    or re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_sha256", ""))) is None):
                raise ValueError("Checkpoint evidence has invalid save metadata")
            rows.append(row)
    return rows, complete


def collect_training_evidence(root: str | Path | None, output: Path,
                              config: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot run evidence and distinguish current source from executed versions.

    External paths are explicit references to small run ledgers/source archives.
    Rolling checkpoint binaries are neither opened nor required by this check.
    """
    pending, copied = [], []
    if root is None:
        return {"complete": False, "pending_items": ["training logs, checkpoint ledger and source-version archives"],
                "files": [], "checkpoint_binaries_required": False}
    root = Path(root).resolve(strict=True)
    manifest_path = root / "evidence_manifest.json"
    if not manifest_path.is_file():
        return {"complete": False, "pending_items": ["training evidence_manifest.json"], "files": [],
                "checkpoint_binaries_required": False}
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "kd_training_evidence":
        raise ValueError("Training evidence manifest schema is unsupported")

    def reference(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Training-evidence references require a path")
        path = Path(value)
        return path if path.is_absolute() else root / path

    def snapshot(path, relative):
        if not path.is_file():
            pending.append(f"missing training evidence: {path}")
            return None
        target = output / relative
        item = {**_evidence_snapshot(path, target), "package_path": str(relative)}
        copied.append(item)
        if not item["stable_during_snapshot"]:
            pending.append(f"training evidence changed during snapshot: {path}")
        return target

    snapshot(manifest_path, Path("evidence_manifest.json"))
    archives = set()
    archive_sources = {}
    for index, entry in enumerate(manifest.get("source_archives", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] in archives:
            raise ValueError("Source archives require unique identifiers")
        directory = reference(entry.get("path"))
        if directory.is_symlink():
            raise ValueError("Source archive must not be a symlink")
        files = [] if not directory.is_dir() else [p for p in sorted(directory.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts]
        if not files or not any(p.suffix == ".py" for p in files):
            pending.append(f"missing Python source archive: {directory}")
            continue
        for path in files:
            snapshot(path, Path("source_archives") / str(index) / path.relative_to(directory))
        archives.add(entry["id"])
        archive_sources[entry["id"]] = {str(path.relative_to(directory)) for path in files}
    if not archives:
        pending.append("archived training source versions")
    logs = {}
    for index, entry in enumerate(manifest.get("console_logs", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] in logs:
            raise ValueError("Console logs require unique identifiers")
        path = reference(entry.get("path"))
        frozen_log = snapshot(path, Path("console_logs") / f"{index}_{path.name}")
        if frozen_log is not None and frozen_log.stat().st_size:
            logs[entry["id"]] = entry
        elif frozen_log is not None:
            pending.append(f"empty training console log: {path}")
    if not logs:
        pending.append("persistent training console logs")
    supporting_kinds = set()
    for index, entry in enumerate(manifest.get("supporting_files", [])):
        path = reference(entry.get("path"))
        if snapshot(path, Path("supporting_files") / f"{index}_{path.name}") is not None:
            supporting_kinds.add(entry.get("kind"))
    for kind in ("dependency_lock", "environment_record"):
        if kind not in supporting_kinds:
            pending.append(f"training {kind}")
    coverage = {item.get("run_id"): item for item in manifest.get("console_coverage", []) if isinstance(item, dict)}
    timeline = manifest.get("source_version_timeline", [])
    students, teachers = ancestry_runs(config)
    required_ids, found, runs = set(teachers) | set(students), set(), []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict) or not isinstance(run.get("id"), str) or run["id"] in found:
            raise ValueError("Training evidence requires unique run identifiers")
        run_id = run["id"]
        found.add(run_id)
        if run_id not in required_ids:
            continue
        config_path = run.get("config")
        if not config_path:
            pending.append(f"recorded initialization and seed for {run_id}")
        else:
            frozen_config = snapshot(reference(config_path), Path("runs") / run_id / "config.json")
            if frozen_config is not None:
                training_config = read_json(frozen_config)
                seed = training_config.get("seed", training_config.get("training_config", {}).get("seed"))
                if type(seed) is not int or not 0 <= seed < 2**32:
                    pending.append(f"fixed random seed for {run_id}")
                if run_id in students:
                    member = students[run_id]
                    if training_config.get("training_mode", "distillation") != member.get("training_mode", "distillation"):
                        raise ValueError(f"Training mode differs from deployed member {run_id}")
                    if member.get("seed") is not None and seed != member["seed"]:
                        raise ValueError(f"Training seed differs from deployed member {run_id}")
                    if training_config.get("image_size", 512) != member.get("image_size", 512):
                        raise ValueError(f"Training resolution differs from deployed member {run_id}")
                    if member.get("distillation") is not None and training_config.get("distillation") != member["distillation"]:
                        raise ValueError(f"Training distillation differs from deployed member {run_id}")
                from kd_common import validate_pretrained_provenance
                expected_provenance = (students[run_id].get("pretrained_provenance", {}) if run_id in students else
                                       teachers[run_id].get("pretrained_provenance", {}))
                initialization = training_config.get("pretrained_provenance")
                if initialization is None:
                    provenance_path = (reference(run["pretrained_provenance"]) if run.get("pretrained_provenance")
                                       else reference(config_path).parent / "pretrained_provenance.json")
                    provenance_copy = snapshot(provenance_path, Path("runs") / run_id / "pretrained_provenance.json")
                    initialization = None if provenance_copy is None else read_json(provenance_copy)
                if initialization is None:
                    pending.append(f"initialization provenance for {run_id}")
                else:
                    validate_pretrained_provenance(initialization, expected_provenance.get("backbone_name"))
                    if any(initialization.get(key) != expected_provenance.get(key)
                           for key in ("weights_sha256", "revision", "repo_id")):
                        raise ValueError(f"Training initialization differs from deployed ancestry for {run_id}")
        if (run.get("source_archive_id") not in archives
                or not any(isinstance(item, dict) and item.get("run_id") == run_id
                           and item.get("source_archive_id") == run.get("source_archive_id") for item in timeline)):
            pending.append(f"source archive/version relationship for {run_id}")
        required_sources = {"train_overnight.py", "model_overnight.py", "dataset_overnight.py",
                            "dataset.py", "kd_common.py", "cv_splits.py", "runtime_compat.py"}
        required_sources.add("train_distill.py" if run_id in students else "train_teacher.py")
        if run_id in students:
            if students[run_id].get("training_mode", "distillation") == "distillation":
                required_sources.add("distill_data.py")
            if (students[run_id].get("distillation", {}).get("teacher_tta") not in (None, "identity")
                    or "teacher_views" in students[run_id].get("distillation", {})):
                required_sources.add("distill_tta_data.py")
        if required_sources - archive_sources.get(run.get("source_archive_id"), set()):
            pending.append(f"complete training source archive for {run_id}")
        own_logs = [entry for entry in logs.values() if run_id in entry.get("run_ids", [])
                    and entry.get("coverage_from_start") == "complete"]
        group = coverage.get(run_id, {})
        group_ids = group.get("log_ids", [])
        grouped_complete = (group.get("coverage_from_start") == "complete"
                            and bool(group.get("coverage_explanation"))
                            and isinstance(group_ids, list) and bool(group_ids)
                            and all(log_id in logs for log_id in group_ids))
        if not own_logs and not grouped_complete:
            pending.append(f"complete console-log coverage for {run_id}")
        path = reference(run.get("checkpoint_audit"))
        frozen = snapshot(path, Path("runs") / run_id / "checkpoint_audit.jsonl")
        rows, ledger_complete = ([], False) if frozen is None else _read_evidence_ledger(frozen)
        if not ledger_complete:
            pending.append(f"complete checkpoint ledger for {run_id}")
        expected = run.get("expected_epochs")
        if run_id in teachers:
            teacher = teachers[run_id]
            expected = teacher.get("training_config", {}).get("epochs", teacher.get("epoch"))
            final_digest = teacher.get("checkpoint_sha256", teacher.get("sha256"))
        else:
            final_digest = students[run_id].get("source_checkpoint_sha256")
        if type(expected) is not int or expected < 1:
            pending.append(f"declared completed epoch coverage for {run_id}")
            expected = 0
        timed = [row for row in rows if _valid_evidence_time(row.get("saved_at_utc"))
                 or _valid_evidence_time(row.get("observed_status_at"))]
        epochs = {row["epoch"] for row in timed if row["role"] == "last"}
        missing_epochs = sorted(set(range(1, expected + 1)) - epochs)
        if missing_epochs:
            pending.append(f"checkpoint times/hashes for {run_id} epochs {missing_epochs}")
        if not final_digest or not any(row["checkpoint_sha256"] == final_digest for row in timed):
            pending.append(f"checkpoint ledger binding to deployed ancestry for {run_id}")
        if run.get("complete") is not True:
            pending.append(f"completed training evidence for {run_id}")
        runs.append({"id": run_id, "observed_epochs": sorted(epochs), "expected_epochs": expected,
                     "missing_epochs": missing_epochs, "ledger_records": len(rows),
                     "historically_recovered_records": sum(row.get("historical_recovery") is True for row in rows)})
    pending.extend(f"training evidence for {run_id}" for run_id in sorted(required_ids - found))
    historical = manifest.get("historical_missing_evidence", [])
    for gap in historical:
        if not isinstance(gap, dict) or gap.get("blocking", True):
            pending.append(f"unresolved historical evidence: {gap}")
    result = {"complete": not pending, "pending_items": list(dict.fromkeys(pending)), "files": copied, "runs": runs,
              "source_version_timeline": timeline, "historical_missing_evidence": historical,
              "checkpoint_binaries_required": False,
              "timestamp_policy": "saved_at_utc or accurately labelled saved-status observation time",
              "source_policy": "code/ is packaging-time source; executed versions are archived and attributed separately"}
    write_json(output / "training_evidence_status.json", result)
    return result


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_xlsx(path: str | Path, rows: Iterable[Sequence[Any]]) -> None:
    """Minimal OOXML workbook using inline strings, preserving IDs and SHA text."""
    workbook = ET.Element("workbook", xmlns=SHEET_NS, attrib={"xmlns:r": REL_NS})
    sheets = ET.SubElement(workbook, "sheets")
    ET.SubElement(sheets, "sheet", name="模型核验材料", sheetId="1", attrib={"r:id": "rId1"})
    worksheet = ET.Element("worksheet", xmlns=SHEET_NS)
    data = ET.SubElement(worksheet, "sheetData")
    for row_number, values in enumerate(rows, start=1):
        row = ET.SubElement(data, "row", r=str(row_number))
        for column, value in enumerate(values, start=1):
            cell = ET.SubElement(row, "c", r=f"{_column_name(column)}{row_number}", t="inlineStr")
            inline = ET.SubElement(cell, "is")
            text = ET.SubElement(inline, "t", attrib={"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
            text.text = "" if value is None else str(value)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>')
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>')
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", ET.tostring(workbook, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(worksheet, encoding="utf-8", xml_declaration=True))


def unique_pretrains(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate the original weights by exact repository and SHA, not filename."""
    from kd_common import validate_pretrained_provenance

    students, teachers = ancestry_runs(config)
    entries = [(role, member["pretrained_provenance"]) for role, member in students.items()]
    entries.extend((role, teacher["pretrained_provenance"]) for role, teacher in teachers.items())
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for role, provenance in entries:
        validate_pretrained_provenance(provenance, provenance["backbone_name"])
        key = (provenance["repo_id"], provenance["weights_sha256"])
        if key not in unique:
            unique[key] = {"provenance": provenance, "roles": []}
        unique[key]["roles"].append(role)
    return list(unique.values())


def locate_pretrain(provenance: Mapping[str, Any], cache: str | Path | None = None) -> Path:
    """Search explicit local cache layouts; never download missing source weights."""
    candidates = []
    if cache is not None:
        root = Path(cache).resolve(strict=True)
        repo = provenance["repo_id"]
        revision = provenance["revision"]
        candidates.extend([
            root / f"{provenance['weights_sha256']}.safetensors",
            root / repo.replace("/", "__") / "model.safetensors",
            root / repo / "model.safetensors",
            root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / "model.safetensors",
            root / provenance["backbone_name"] / "model.safetensors",
        ])
    if provenance.get("weights_path"):
        candidates.append(Path(provenance["weights_path"]))
    mismatches = []
    for candidate in candidates:
        if candidate.is_file():
            if sha256_file(candidate) == provenance["weights_sha256"]:
                return candidate.resolve(strict=True)
            mismatches.append(str(candidate))
    if mismatches:
        raise ValueError(f"Original pretraining SHA-256 mismatch: {mismatches}")
    raise FileNotFoundError(
        f"Original weights not found for {provenance['repo_id']}; provide --pretrain-cache. No download was attempted.")


def copy_verified(source: str | Path, destination: str | Path, expected_sha256: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
            outgoing.write(block)
    if digest.hexdigest() != expected_sha256:
        destination.unlink()
        raise ValueError(f"Source changed or copied data failed SHA-256 verification: {source}")


def validate_submission_material(path: str | Path, *, test_manifest: str | Path,
                                 expected_studies: int = 1414) -> dict[str, Any]:
    path = Path(path)
    if path.name != "submission.csv":
        raise ValueError("Submitted prediction material must be named submission.csv")
    values = read_submission(path)
    records = read_manifest(test_manifest)
    studies = {(r["subject_id"], r["study_id"]) for r in records}
    if len(studies) != expected_studies or len(values) != expected_studies:
        raise ValueError("Submission or original test manifest Study count differs from the declared dataset")
    if all(subject is None for subject, _ in values):
        if {study for _, study in values} != {study for _, study in studies}:
            raise ValueError("Submission does not cover all original test Studies")
    elif set(values) != studies:
        raise ValueError("Submission patient/Study coverage differs from the original test manifest")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if "Gate" in header and header[-1] != "Gate":
            raise ValueError("Gate must be the last column")
        for row in reader:
            if row.get("Gate", "").strip():
                gate = float(row["Gate"])
                if not math.isfinite(gate) or not 0 <= gate <= 1:
                    raise ValueError("Submission contains invalid Gate values")
    return {"studies": len(studies), "data_rows": len(studies) * 10,
            "sha256": sha256_file(path), "identity_coverage_complete": True}


def _same_inventory(reported: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    def entries(inventory):
        return {(f["path"], f["bytes"], f["sha256"]) for f in inventory.get("files", [])}
    return entries(reported) == entries(actual) and reported.get("total_bytes") == actual["total_bytes"]


def verify_reports(config: Mapping[str, Any], inventory: Mapping[str, Any], *,
                   submission: Mapping[str, Any] | None, cost: Mapping[str, Any] | None,
                   validation: Mapping[str, Any] | None,
                   training_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    missing = []
    config_hash = next((item["sha256"] for item in inventory.get("files", []) if item["path"] == "config.json"), None)
    if submission is None:
        missing.append("submission.csv")
    cost_pass = False
    if cost is None:
        missing.append("T4 cost_report.json")
    else:
        if not _same_inventory(cost.get("inference_files", {}), inventory):
            raise ValueError("Timing report was generated for another deployment")
        if config.get("schema_version") == 2 and cost.get("deployment_config_sha256") != config_hash:
            raise ValueError("Timing report does not bind the exact multi-member deployment config")
        if submission is not None and cost.get("submission_sha256") != submission["sha256"]:
            raise ValueError("Timing report does not correspond to the packaged submission")
        mean = cost.get("mean_ms")
        timing_valid = (cost.get("environment", {}).get("official_environment_compatible") is True
                        and "T4" in str(cost.get("environment", {}).get("gpu_name", ""))
                        and isinstance(mean, (int, float)) and math.isfinite(mean) and 0 <= mean <= 100)
        semantic_valid = _cost_semantics_complete(cost, submission)
        cost_pass = cost.get("official_compliance_pass") is True and timing_valid and semantic_valid
        if not semantic_valid:
            missing.append("seed43 and three-round 100-image semantic/sidecar-bound cost evidence")
        if not cost_pass:
            missing.append("passing local T4 timing and CSV agreement")
    if submission is not None and not submission.get("image_predictions_sha256"):
        missing.append("final predict.py image-probability sidecar")
    validation_pass = False
    if validation is None:
        missing.append("FP16 official-valid validation_report.json")
    else:
        if not _same_inventory(validation.get("deployment_inventory", {}), inventory):
            raise ValueError("FP16 validation report was generated for another deployment")
        if validation.get("manifest_sha256") != config["official_valid_manifest_sha256"]:
            raise ValueError("FP16 validation used another validation manifest")
        if config.get("schema_version") == 2:
            expected_members = [{k: member.get(k) for k in
                ("source_checkpoint_sha256", "weights_sha256", "image_size", "tta", "seed")}
                for member in config["members"]]
            if (validation.get("deployment_config_sha256") != config_hash
                    or validation.get("image_sizes") != config["image_sizes"]
                    or validation.get("members") != expected_members):
                raise ValueError("FP16 validation member declarations differ from deployment")
        auc = validation.get("macro_auc")
        validation_pass = (validation.get("complete") is True and validation.get("precision") == "float16"
                           and validation.get("image_size") == config.get("image_size", 512)
                           and validation.get("tta") == config.get("tta", "identity")
                           and validation.get("ensemble") is config.get("ensemble", False)
                           and ("members" not in config or validation.get("member_count") == len(config["members"]))
                           and isinstance(auc, (float, int))
                           and math.isfinite(auc) and 0 <= auc <= 1)
        if not validation_pass:
            missing.append("complete FP16 validation evidence")
    training_pass = training_evidence is not None and training_evidence.get("complete") is True
    if not training_pass:
        missing.extend((training_evidence or {}).get("pending_items") or ["complete training evidence"])
    ready = submission is not None and cost_pass and validation_pass and inventory["size_pass"] and training_pass
    return {"status": "ready_for_organizer_review" if ready else "pending",
            "ready_for_organizer_review": ready, "pending_items": missing,
            "final_organizer_retest_required": True}


def package_submission(*, model_dir: str | Path, output: str | Path,
                       submission: str | Path | None = None,
                       image_submission: str | Path | None = None,
                       cost_report: str | Path | None = None,
                       validation_report: str | Path | None = None,
                       pretrain_cache: str | Path | None = None,
                       training_evidence: str | Path | None = None,
                       project_root: str | Path = PROJECT_ROOT,
                       expected_studies: int = 1414) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve(strict=True)
    source = Path(project_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new verification-package directory")
    if output.is_relative_to(model_dir):
        raise ValueError("Package output cannot be inside its source deployment")
    config = read_json(model_dir / "config.json")
    validate_deployment_config(config)
    inventory = deployment_inventory(model_dir, config)
    if not inventory["size_pass"]:
        raise ValueError("Final inference files exceed 500,000,000 bytes")
    for member in deployment_members(config):
        if sha256_file(model_dir / member.get("weights_file", "student_fp16.pt")) != member["weights_sha256"]:
            raise ValueError("Deployment weights changed after export")
    originals = unique_pretrains(config)
    for item in originals:
        item["source_path"] = locate_pretrain(item["provenance"], pretrain_cache)
    for name in CODE_FILES + MANIFEST_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required reproducibility source is missing: {name}")
    support_files = inference_support_files(source)
    if sha256_file(source / MANIFEST_FILES[0]) != config["source_manifest_sha256"]:
        raise ValueError("Training manifest differs from deployment ancestry")
    if sha256_file(source / MANIFEST_FILES[1]) != config["official_valid_manifest_sha256"]:
        raise ValueError("Validation manifest differs from deployment ancestry")
    submission_info = None if submission is None else validate_submission_material(
        submission, test_manifest=source / MANIFEST_FILES[2], expected_studies=expected_studies)
    image_reference = (Path(image_submission) if image_submission is not None else
                       Path(submission).with_suffix(".image_predictions.csv") if submission is not None else None)
    if image_submission is not None and not image_reference.is_file():
        raise FileNotFoundError("The specified final image-probability sidecar is missing")
    if submission_info is not None and image_reference is not None and image_reference.is_file():
        # Validate format and all declared image identities, without rerunning a model.
        from cost import read_image_probabilities
        image_rows = read_image_probabilities(image_reference)
        image_manifest_rows = read_manifest(source / MANIFEST_FILES[2])
        original_identities = {(row["subject_id"], row["study_id"], row["image_id"]) for row in image_manifest_rows}
        discovered_identities = {(row["subject_id"], row["study_id"],
                                  f"{row['subject_id']}/{row['study_id']}/{Path(row['relative_path']).name}") for row in image_manifest_rows}
        if set(image_rows) not in (original_identities, discovered_identities):
            raise ValueError("Final image sidecar does not cover the exact original test images")
        submission_info["image_predictions_sha256"] = sha256_file(image_reference)
        submission_info["images"] = len(image_rows)
    cost = None if cost_report is None else read_json(cost_report)
    validation = None if validation_report is None else read_json(validation_report)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "package_status.json", {"status": "preparing", "ready_for_organizer_review": False})
    training_evidence_info = collect_training_evidence(training_evidence, output / "audit/training_evidence", config)
    readiness = verify_reports(config, inventory, submission=submission_info, cost=cost, validation=validation,
                               training_evidence=training_evidence_info)
    copied_sources = []
    sheet_rows: list[list[Any]] = [["序号", "名称", "规则依据", "内容或包内路径", "source_sha256", "code_sha256"]]
    for item in inventory["files"]:
        copy_verified(model_dir / item["path"], output / "model" / item["path"], item["sha256"])
    for name in CODE_FILES + MANIFEST_FILES:
        digest = sha256_file(source / name)
        target = output / "code" / name
        copy_verified(source / name, target, digest)
        copied_sources.append({"source": name, "package_path": str(target.relative_to(output)), "sha256": digest})
        sheet_rows.append([f"CODE-{len(copied_sources)}", "代码或清单", "附录A", str(target.relative_to(output)), digest, digest if name.endswith(".py") else ""])
    for path, relative in support_files:
        digest = sha256_file(path)
        copy_verified(path, output / relative, digest)
        copied_sources.append({"source": str(path), "package_path": str(relative), "sha256": digest})
        sheet_rows.append([f"CODE-{len(copied_sources)}", "原生推理源码/许可/编译产物或公开依赖归档", "第10条/附录A", str(relative), digest, ""])
    original_plan = source / "data/cv5_overnight/folds.json"
    if original_plan.is_file():
        digest = sha256_file(original_plan)
        _, teachers = ancestry_runs(config)
        expected_plans = {teacher["fold_plan_sha256"] for teacher in teachers.values()}
        if not teachers:
            expected_plans = {config.get("fold_plan_sha256")}
        if expected_plans != {digest}:
            raise ValueError("Local fold plan differs from the actual teachers' ancestry")
        copy_verified(original_plan, output / "code/data/cv5_overnight/folds.original.json", digest)
        copied_sources.append({"source": "data/cv5_overnight/folds.json", "package_path": "code/data/cv5_overnight/folds.original.json", "sha256": digest})
    pretrain_materials = []
    for index, item in enumerate(originals, start=1):
        provenance = item["provenance"]
        name = provenance["repo_id"].replace("/", "__") + "_" + provenance["weights_sha256"][:12]
        weight_relative = Path("pretrain_model") / name / "model.safetensors"
        copy_verified(item["source_path"], output / weight_relative, provenance["weights_sha256"])
        materials = {}
        for key, filename in (("apache_license", "Apache-2.0.txt"), ("upstream_license", "MIT.txt"),
                              ("model_card", "README.md")):
            relative = (Path("modelcards") / name / filename if key == "model_card"
                        else Path("licenses") / name / filename)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="") as stream:
                stream.write(provenance["materials"][key]["text"])
            if sha256_file(target) != provenance["materials"][key]["sha256"]:
                raise ValueError("Packaged license/model-card content changed")
            materials[key] = {"path": str(relative), "sha256": sha256_file(target),
                              "source_url": provenance["materials"][key].get("url")}
        pretrain_materials.append({"repo_id": provenance["repo_id"], "backbone_name": provenance["backbone_name"],
                                  "roles": item["roles"], "revision": provenance["revision"],
                                  "weights_url": provenance["weights_url"], "weights_sha256": provenance["weights_sha256"],
                                  "package_path": str(weight_relative), "materials": materials,
                                  "counted_as_inference_file": False})
        for key, label, value in (
            ("A1", "所用开源模型名称", provenance["backbone_name"]),
            ("A2", "权重公开链接", provenance["weights_url"]),
            ("A3", "许可证全文", f"{materials['apache_license']['path']}；{materials['upstream_license']['path']}"),
            ("A4", "上游训练数据说明", f"ImageNet-22k→ImageNet-1k；{materials['model_card']['path']}"),
            ("A5", "原预训练模型权重哈希", provenance["weights_sha256"]),
            ("A6", "推理加速精度", f"FP16；{len(deployment_members(config))}个Tiny；分辨率{[m.get('image_size', 512) for m in deployment_members(config)]}；TTA={config.get('tta', 'identity')}；集成={config.get('ensemble', False)}"),
        ):
            sheet_rows.append([f"{index}-{key}", label, "第1条/第7条/附录A", value, provenance["weights_sha256"], ""])
    for arg, relative in ((submission, "submission.csv"), (cost_report, "audit/cost_report.json"),
                          (validation_report, "audit/validation_report.json")):
        if arg is not None:
            copy_verified(arg, output / relative, sha256_file(arg))
    if submission_info is not None and submission_info.get("image_predictions_sha256"):
        copy_verified(image_reference, output / "submission.image_predictions.csv",
                      submission_info["image_predictions_sha256"])
    if (model_dir / "export_report.json").is_file():
        path = model_dir / "export_report.json"
        copy_verified(path, output / "audit/export_report.json", sha256_file(path))
    root_cost = ('"""Official cost entry; loads the same implementation as code/predict.py."""\n'
                 'from pathlib import Path\nimport sys\n'
                 'sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))\n'
                 'from cost import main\n'
                 'if __name__ == "__main__":\n    raise SystemExit(main())\n')
    (output / "cost.py").write_text(root_cost, encoding="utf-8")
    student_train = ('"""Student training entry; defaults to dry-run."""\n'
                     'from train_distill import main\n'
                     'if __name__ == "__main__":\n    raise SystemExit(main())\n')
    (output / "code/train.py").write_text(student_train, encoding="utf-8")
    (output / "code/model.py").write_text('"""Approved model builder alias."""\nfrom model_overnight import build_overnight_model, validate_model_config\n', encoding="utf-8")
    (output / "code/utils.py").write_text('"""Hash/provenance utilities alias."""\nfrom kd_common import sha256_file, validate_pretrained_provenance\n', encoding="utf-8")
    sheet_rows.extend([
        ["D1", "验证集使用声明", "第3条", "官方valid仅用于学生早停/选checkpoint及独立FP16验证，不加入梯度训练；教师外折不选模", "", ""],
        ["D2", "测试集使用声明", "第5/6条", "测试集仅固定推理及脚本运行验证；不生成训练伪标签、不拟合参数", "", ""],
        ["D3", "推理文件实际大小", "第8条", f"model/实际调用文件合计{inventory['total_bytes']} bytes；上限500000000 bytes", "", ""],
        ["D4", "材料状态", "第10条/附录A", readiness["status"] + "；" + "；".join(readiness["pending_items"]), "", ""],
        ["D5", "实际学生训练模式", "第1/2/10条", "；".join(member.get("training_mode", "distillation") for member in deployment_members(config)), "", ""],
    ])
    write_xlsx(output / "模型核验材料与数据申明清单.xlsx", sheet_rows)
    readme = f"""# 蒸馏模型核验材料

材料状态：{readiness['status']}。待补项：{'；'.join(readiness['pending_items']) or '无（仍须主办方复测）'}。
缺少最终CSV、同一部署的T4计时、FP16官方valid验证或完整训练备查证据时，本目录只是准备材料，不能宣称可正式提交。

model/含{len(deployment_members(config))}个V1 Tiny的实际FP16推理权重及必要config，实际合计{inventory['total_bytes']} bytes；固定TTA={config.get('tta', 'identity')}，集成={config.get('ensemble', False)}。
各成员训练模式：{[member.get('training_mode', 'distillation') for member in deployment_members(config)]}。纯监督成员不声明教师来源；蒸馏成员须保留完整的折外教师来源与证据。
pretrain_model/是核验用原始公开预训练权重，不被predict/cost加载；licenses/和modelcards/用于来源核验。
第8条500MB仅统计实际推理文件，本核验目录总大小可以超过500MB。没有打包比赛图像、旧V2权重或教师训练快照。

code/含打包时的训练、OOF提取、学生蒸馏和共享推理源码，代码与清单SHA见materials_manifest.json和XLSX。
执行期间使用的源码版本、修改时间线、完整终端日志和逐轮checkpoint哈希/时间记录单独归档在audit/training_evidence/；不能把打包时源码冒充所有已运行进程的原始版本。
训练断点采用滚动文件，仅留存每轮记录并不表示保存了每轮完整权重二进制；历史恢复证据和时间语义如实标注，缺失项仍列pending。
三份清洗清单原样复制在code/data/clean_v1_reviewed/；原始图像须由主办方在本机提供。
code/data/cv5_overnight/folds.original.json如存在，仅为原折表证据，可能含原机器绝对路径，不应直接用于异机续训。
在异机重放时，使用同seed=43重建新折表，并核对Study→fold映射与原表一致：

    python code/cv_splits.py --train-manifest code/data/clean_v1_reviewed/train_clean.csv --valid-manifest code/data/clean_v1_reviewed/valid_manifest.csv --output code/data/cv5_overnight/folds.json --seed 43

固定推理（输出目录需新建且文件不能已存在；路径按实际原图位置填写）：

    python code/prepare_inference.py --force-build --output /new/preparation.json

    python code/predict.py --model-dir model --image-root /path/to/cxr_first --manifest code/data/clean_v1_reviewed/test_manifest.csv --output /new/output/submission.csv

官方口径计时：

    python cost.py submission.csv /path/to/pictures_100 /path/to/pictures_5 --model-dir model --manifest code/data/clean_v1_reviewed/test_manifest.csv --output-dir /new/cost_report

code/prepare_inference.py在官方镜像内离线恢复code/vendor的固定公开附加包并重建原生CPU算子，不覆盖torch/torchvision/NumPy/Pillow/CUDA等核心依赖；准备必须在计时前完成。code/native_lanczos包含源码、上游许可/出处及本次CPython3.12编译产物，不能依赖其他机器的构建目录。
计时调用与predict相同的模型加载和单图预测，包含逐图十类sidecar写出和Study结果写出；两入口固定seed43。包根submission.image_predictions.csv原字节保留最终逐图输出，正式cost逐张检查概率、二值/Gate、排序和并列，再核Study聚合；缺sidecar或100图缺所选Study其他视图均保持pending，不得读取目录外预测补齐。
训练与推理依赖分别见code/requirements_kd.txt和code/requirements_inference.txt。推理沿用官方Py3.12、torch2.10.0、CUDA12.8/cuDNN9；不覆盖官方CUDA环境。
本脚本只复制和核验材料，不下载、不启动训练或模型前向。最终合规与评分以主办方复测为准。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    result = {**readiness, "inference_inventory": inventory, "submission": submission_info,
              "training_evidence": training_evidence_info,
              "pretraining_sources": pretrain_materials, "copied_sources": copied_sources,
              "runtime_weights": ["model/" + member.get("weights_file", "student_fp16.pt") for member in deployment_members(config)],
              "training_modes": [member.get("training_mode", "distillation") for member in deployment_members(config)],
              "pretrain_verification_weights_counted_toward_500MB": False,
              "competition_images_included": False, "models_executed": False,
              "downloads_performed": False,
              "files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)}
                        for path in sorted(output.rglob("*")) if path.is_file()
                        and path.name not in ("package_status.json", "materials_manifest.json")]}
    write_json(output / "materials_manifest.json", result)
    write_json(output / "package_status.json", readiness)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--image-submission", type=Path,
                        help="Saved final per-image sidecar; defaults beside submission.csv")
    parser.add_argument("--cost-report", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--pretrain-cache", type=Path)
    parser.add_argument("--training-evidence", type=Path,
                        help="Directory containing evidence_manifest.json and references to training evidence")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-studies", type=int, default=1414)
    args = parser.parse_args(argv)
    result = package_submission(**vars(args))
    print(json.dumps({k: result[k] for k in ("status", "ready_for_organizer_review", "pending_items", "inference_inventory")},
                     ensure_ascii=False, indent=2))
    return 0 if result["ready_for_organizer_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
