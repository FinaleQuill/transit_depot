"""Combine each deployed student's receipts with its required teacher evidence.

Default is a preview. This reads small records and checkpoint hashes only; it
never loads a model, downloads weights, edits a run, or claims missing evidence.
Independent supervised-only deployments require no teacher evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from deploy_kd import deployment_inventory, read_json, sha256_file, validate_deployment_config, write_json
from package_kd_submission import deployment_members


def build_evidence(*, model_dir, student_runs, output, teacher_evidence=None, execute=False):
    model_dir = Path(model_dir).resolve(strict=True)
    config = read_json(model_dir / "config.json")
    validate_deployment_config(config)
    deployment_inventory(model_dir, config)
    members = deployment_members(config)
    if len(student_runs) != len(members):
        raise ValueError("Provide one student run per deployed member, in deployment order")
    roots = [Path(p).resolve(strict=True) for p in student_runs]
    if len(set(roots)) != len(roots):
        raise ValueError("Each deployed student requires its own training run")
    needs_teachers = any(m.get("training_mode", "distillation") == "distillation" for m in members)
    if needs_teachers:
        if teacher_evidence is None:
            raise ValueError("Distilled members require the frozen teacher training evidence")
        teacher_root = Path(teacher_evidence).resolve(strict=True)
        manifest = deepcopy(read_json(teacher_root / "evidence_manifest.json"))
        if manifest.get("artifact_type") != "kd_training_evidence" or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported teacher evidence manifest")
    else:
        teacher_root = None
        manifest = {"schema_version": 1, "artifact_type": "kd_training_evidence",
                    "source_archives": [], "console_logs": [], "supporting_files": [],
                    "runs": [], "source_version_timeline": [], "console_coverage": []}
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new evidence directory")
    for root in [model_dir, *([teacher_root] if teacher_root is not None else []), *roots]:
        if destination == root or destination.is_relative_to(root):
            raise ValueError("Evidence output cannot be written into a source run or deployment")
    if not execute:
        return {"execute": False, "model_dir": str(model_dir), "teacher_evidence": str(teacher_root) if teacher_root else None,
                "student_runs_in_member_order": list(map(str, roots)), "output": str(destination)}

    def absolute(value, base):
        path = Path(value)
        return str(path if path.is_absolute() else base / path)

    for field in ("source_archives", "console_logs", "supporting_files"):
        for entry in manifest.get(field, []):
            entry["path"] = absolute(entry["path"], teacher_root)
            if entry.get("command_receipt"):
                entry["command_receipt"] = absolute(entry["command_receipt"], teacher_root)
    for run in manifest.get("runs", []):
        for field in ("config", "checkpoint_audit", "pretrained_provenance"):
            if run.get(field):
                run[field] = absolute(run[field], teacher_root)
    # Never silently retain an unrelated student already present in this manifest.
    if any(r.get("role") == "student" for r in manifest.get("runs", [])):
        raise ValueError("Teacher source manifest must contain teacher runs only")
    for index, (root, member) in enumerate(zip(roots, members)):
        role = "student" if len(members) == 1 else f"student_{index}"
        status = read_json(root / "status.json")
        run_config = read_json(root / "config.json")
        if status.get("state") not in {"completed", "early_stopped"}:
            raise ValueError(f"Student {index} has not finished")
        if sha256_file(root / "best.pt") != member.get("source_checkpoint_sha256"):
            raise ValueError(f"Student run {index} differs from the deployed checkpoint")
        if member.get("seed") is not None and run_config.get("seed") != member["seed"]:
            raise ValueError("Student initialization seed differs from deployment ancestry")
        if run_config.get("training_mode", "distillation") != member.get("training_mode", "distillation"):
            raise ValueError("Student training mode differs from deployment ancestry")
        if run_config.get("image_size", 512) != member.get("image_size", 512):
            raise ValueError("Student training resolution differs from deployment ancestry")
        if member.get("distillation") is not None and run_config.get("distillation") != member["distillation"]:
            raise ValueError("Student distillation protocol differs from deployment ancestry")
        receipts = sorted((root / "evidence").glob("command_*.json"),
                          key=lambda p: read_json(p).get("started_at", ""))
        if not receipts:
            raise ValueError(f"Student {index} has no native training command receipts")
        archive_ids = []
        log_ids = []
        starts_fresh = False
        invocation_coverage_complete = True
        for number, path in enumerate(receipts):
            receipt = read_json(path)
            cwd = Path(receipt["cwd"])
            archive = Path(absolute(receipt["source_archive"], cwd))
            console = Path(absolute(receipt["console_log"], cwd))
            if not archive.is_dir() or not console.is_file():
                raise FileNotFoundError("A recorded source archive or console log is missing")
            captured = (receipt.get("coverage_from_start") == "complete"
                        and receipt.get("ended_at") is not None
                        and type(receipt.get("exit_code")) is int
                        and receipt.get("console_bytes") == console.stat().st_size
                        and console.stat().st_size > 0)
            starts_fresh |= captured and receipt.get("preexisting_run_artifacts") == []
            invocation_coverage_complete &= captured
            archive_id, log_id = f"{role}_source_{number}", f"{role}_log_{number}"
            archive_ids.append(archive_id)
            log_ids.append(log_id)
            manifest.setdefault("source_archives", []).append({"id": archive_id, "path": str(archive),
                "captured_at": receipt["started_at"], "description": "Native pre-invocation source capture"})
            manifest.setdefault("console_logs", []).append({"id": log_id, "path": str(console),
                "capture_method": "subprocess_stdout_stderr_tee", "coverage_from_start": "invocation_only",
                "run_ids": [role], "command_receipt": str(path)})
            manifest.setdefault("source_version_timeline", []).append({"run_id": role,
                "source_archive_id": archive_id, "recorded_at": receipt["started_at"],
                "description": "Source version recorded by the student's actual CLI invocation"})
            for filename, kind in (("environment_lock.txt", "dependency_lock"),
                                   ("source_manifest.json", "environment_record")):
                manifest.setdefault("supporting_files", []).append({"kind": kind, "path": str(archive / filename)})
        manifest.setdefault("runs", []).append({"id": role, "role": "student", "complete": True,
            "config": str(root / "config.json"), "pretrained_provenance": str(root / "pretrained_provenance.json"),
            "checkpoint_audit": str(root / "checkpoint_audit.jsonl"),
            "source_archive_id": archive_ids[-1], "expected_epochs": status["completed_epochs"]})
        manifest.setdefault("console_coverage", []).append({"run_id": role, "log_ids": log_ids,
            "coverage_from_start": "complete" if starts_fresh and invocation_coverage_complete else "partial",
            "coverage_explanation":
            "Complete only with a native fresh-start receipt and finalized logs for every retained invocation; resume logs never erase an earlier gap."})
    destination.mkdir(parents=True)
    manifest["assembly"] = {"model_config_sha256": sha256_file(model_dir / "config.json"),
                            "teacher_evidence": str(teacher_root) if teacher_root else None, "student_runs": list(map(str, roots)),
                            "source_runs_modified": False, "models_loaded": False}
    write_json(destination / "evidence_manifest.json", manifest)
    return {"execute": True, "output": str(destination), "member_count": len(members),
            "note": "Packaging performs the final source/log/ledger completeness audit."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--teacher-evidence", type=Path,
                        help="Required only when at least one deployed member used distillation")
    parser.add_argument("--student-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(build_evidence(**vars(parser.parse_args(argv))), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
