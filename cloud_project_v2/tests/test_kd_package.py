"""Small temporary fixtures only; no real checkpoints, model, or image execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET
import zipfile
from tests.runtime_package_fixtures import add_runtime_support, passing_cost

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kd_common import APPROVED_PRETRAINS, text_sha256, BASE_BACKBONE, TINY_BACKBONE
from package_kd_submission import (copy_verified, locate_pretrain, unique_pretrains,
                                   verify_reports, write_xlsx, SHEET_NS, package_submission,
                                   CODE_FILES, MANIFEST_FILES, collect_training_evidence)


def provenance(backbone: str, content: bytes) -> dict:
    # These are explicitly synthetic unit-test materials, not official licenses.
    approved = APPROVED_PRETRAINS[backbone]
    revision = "1" * 40
    texts = {"model_card": "license: apache-2.0\nSynthetic unit fixture",
             "apache_license": "Apache License\nSynthetic unit fixture",
             "upstream_license": "MIT License\nSynthetic unit fixture"}
    return {**approved, "backbone_name": backbone, "revision": revision,
            "weights_sha256": hashlib.sha256(content).hexdigest(),
            "weights_url": f"https://huggingface.co/{approved['repo_id']}/resolve/{revision}/model.safetensors",
            "materials": {k: {"text": v, "sha256": text_sha256(v)} for k, v in texts.items()}}


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kd-package-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def training_evidence(self, *, missing_first_epoch=False):
        from audit_trail import record_checkpoint
        root = self.root / "training-evidence"
        root.mkdir()
        archive = self.root / "source-version-before-patch"
        archive.mkdir()
        for name in CODE_FILES:
            (archive / name).write_text("# original executed source fixture\n")
        for name in ("recovered.txt", "live.log"):
            (root / name).write_text("synthetic training terminal output\n")
        (root / "environment.json").write_text('{"python":"3.12.15"}')
        (root / "requirements.lock").write_text("fixture-package==1.0\n")
        base, tiny = provenance(BASE_BACKBONE, b"base"), provenance(TINY_BACKBONE, b"tiny")
        teachers = [{"fold_index": fold, "training_config": {"epochs": 2},
                     "checkpoint_sha256": hashlib.sha256(f"teacher{fold} best".encode()).hexdigest(),
                     "pretrained_provenance": base} for fold in range(5)]
        config = {"teacher_provenance": teachers, "pretrained_provenance": tiny,
                  "backbone_name": TINY_BACKBONE,
                  "source_checkpoint_sha256": hashlib.sha256(b"student best").hexdigest()}
        manifest = {"schema_version": 1, "artifact_type": "kd_training_evidence",
                    "source_archives": [{"id": "before", "path": str(archive),
                                         "captured_after_start": True}],
                    "source_version_timeline": [], "runs": [], "console_coverage": [],
                    "console_logs": [{"id": name, "path": name, "coverage_from_start": "partial"}
                                     for name in ("recovered.txt", "live.log")],
                    "supporting_files": [{"path": "requirements.lock", "kind": "dependency_lock"},
                                         {"path": "environment.json", "kind": "environment_record"}],
                    "historical_missing_evidence": [{"blocking": False, "detail": "rolling epoch binaries not retained"}]}
        for index, run_id in enumerate([f"teacher_fold_{fold}" for fold in range(5)] + ["student"]):
            directory = root / run_id
            directory.mkdir()
            initial = tiny if run_id == "student" else base
            (directory / "config.json").write_text(json.dumps({"seed": 43, "pretrained_provenance": initial}))
            digest = config["source_checkpoint_sha256"] if run_id == "student" else teachers[index]["checkpoint_sha256"]
            for epoch in (1, 2):
                if missing_first_epoch and run_id == "teacher_fold_0" and epoch == 1:
                    continue
                record_checkpoint(directory / "last.pt", hashlib.sha256(f"{run_id} {epoch}".encode()).hexdigest(),
                                  epoch=epoch, role="last", historical=True, evidence_source="synthetic fixture",
                                  observed_status_at="2026-10-25T08:00:00+00:00")
            record_checkpoint(directory / "best.pt", digest, epoch=2, role="best", historical=True,
                              evidence_source="synthetic fixture", observed_status_at="2026-10-25T08:00:00+00:00")
            manifest["runs"].append({"id": run_id, "expected_epochs": 2, "complete": True,
                                     "config": f"{run_id}/config.json", "source_archive_id": "before",
                                     "checkpoint_audit": f"{run_id}/checkpoint_audit.jsonl"})
            manifest["source_version_timeline"].append({"run_id": run_id, "source_archive_id": "before",
                                                        "applies_to": "already_running", "captured_after_start": True})
            manifest["console_coverage"].append({"run_id": run_id, "coverage_from_start": "complete",
                                                  "log_ids": ["recovered.txt", "live.log"],
                                                  "coverage_explanation": "synthetic joined prefix and live suffix"})
        (root / "evidence_manifest.json").write_text(json.dumps(manifest))
        return root, config, manifest

    def test_complete_training_evidence_preserves_history_and_no_checkpoint_binary_is_read(self):
        root, config, _ = self.training_evidence()
        result = collect_training_evidence(root, self.root / "frozen", config)
        self.assertTrue(result["complete"], result["pending_items"])
        self.assertFalse(result["checkpoint_binaries_required"])
        self.assertEqual(len(result["runs"]), 6)
        self.assertTrue(all(run["historically_recovered_records"] == 3 for run in result["runs"]))
        # All checkpoint paths are intentionally absent; observation timestamps suffice.
        self.assertFalse(list(root.rglob("*.pt")))
        self.assertEqual((self.root / "frozen/source_archives/0/train_teacher.py").read_text(),
                         "# original executed source fixture\n")
        for item in result["files"]:
            frozen = self.root / "frozen" / item["package_path"]
            self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), item["sha256"])

    def test_missing_checkpoint_epoch_remains_pending(self):
        root, config, _ = self.training_evidence(missing_first_epoch=True)
        result = collect_training_evidence(root, self.root / "frozen", config)
        self.assertFalse(result["complete"])
        self.assertTrue(any("teacher_fold_0 epochs [1]" in item for item in result["pending_items"]))

    def test_tampered_checkpoint_chain_is_rejected(self):
        root, config, _ = self.training_evidence()
        ledger = root / "teacher_fold_0/checkpoint_audit.jsonl"
        text = ledger.read_text()
        first = json.loads(text.splitlines()[0])["checkpoint_sha256"]
        ledger.write_text(text.replace(first, "f" * 64, 1))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            collect_training_evidence(root, self.root / "frozen", config)

    def test_missing_log_segment_and_environment_lock_stay_pending(self):
        root, config, manifest = self.training_evidence()
        (root / "live.log").unlink()
        manifest["supporting_files"] = []
        (root / "evidence_manifest.json").write_text(json.dumps(manifest))
        result = collect_training_evidence(root, self.root / "frozen", config)
        self.assertFalse(result["complete"])
        self.assertIn("training dependency_lock", result["pending_items"])
        self.assertTrue(any("complete console-log coverage" in item for item in result["pending_items"]))

    def test_final_checkpoint_ancestry_must_match_recorded_hash(self):
        root, config, _ = self.training_evidence()
        config["source_checkpoint_sha256"] = "f" * 64
        result = collect_training_evidence(root, self.root / "frozen", config)
        self.assertFalse(result["complete"])
        self.assertIn("checkpoint ledger binding to deployed ancestry for student", result["pending_items"])

    def test_actual_teacher_only_evidence_cannot_certify_unrun_student(self):
        root, config, manifest = self.training_evidence()
        manifest["runs"] = manifest["runs"][:1]
        manifest["runs"][0]["complete"] = False
        (root / "evidence_manifest.json").write_text(json.dumps(manifest))
        result = collect_training_evidence(root, self.root / "frozen", config)
        self.assertFalse(result["complete"])
        self.assertIn("training evidence for student", result["pending_items"])
        self.assertIn("completed training evidence for teacher_fold_0", result["pending_items"])

    def test_passing_deployment_reports_still_require_training_evidence(self):
        inventory = {"files": [{"path": "student_fp16.pt", "bytes": 10, "sha256": "a" * 64}],
                     "total_bytes": 10, "size_pass": True}
        config = {"official_valid_manifest_sha256": "b" * 64}
        submission = {"sha256": "c" * 64, "image_predictions_sha256": "d" * 64}
        cost = passing_cost(inventory, "c" * 64)
        validation = {"deployment_inventory": inventory, "manifest_sha256": "b" * 64,
                      "complete": True, "precision": "float16", "image_size": 512,
                      "tta": "identity", "ensemble": False, "macro_auc": 0.8}
        pending = verify_reports(config, inventory, submission=submission, cost=cost, validation=validation)
        self.assertFalse(pending["ready_for_organizer_review"])
        ready = verify_reports(config, inventory, submission=submission, cost=cost, validation=validation,
                               training_evidence={"complete": True, "pending_items": []})
        self.assertTrue(ready["ready_for_organizer_review"])

    def test_old_boolean_only_cost_cannot_be_upgraded_to_ready(self):
        inventory = {"files": [{"path": "student_fp16.pt", "bytes": 10, "sha256": "a" * 64}],
                     "total_bytes": 10, "size_pass": True}
        config = {"official_valid_manifest_sha256": "b" * 64}
        submission = {"sha256": "c" * 64, "image_predictions_sha256": "d" * 64}
        validation = {"deployment_inventory": inventory, "manifest_sha256": "b" * 64,
                      "complete": True, "precision": "float16", "image_size": 512,
                      "tta": "identity", "ensemble": False, "macro_auc": .8}
        cost = {"inference_files": inventory, "submission_sha256": "c" * 64,
                "official_compliance_pass": True}
        report = verify_reports(config, inventory, submission=submission, cost=cost, validation=validation,
                                training_evidence={"complete": True})
        self.assertFalse(report["ready_for_organizer_review"])
        self.assertTrue(any("semantic/sidecar" in reason for reason in report["pending_items"]))

    def test_cost_semantics_seed_coverage_and_sidecar_sha_are_required(self):
        import copy
        inventory = {"files": [{"path": "student_fp16.pt", "bytes": 10, "sha256": "a" * 64}],
                     "total_bytes": 10, "size_pass": True}
        config = {"official_valid_manifest_sha256": "b" * 64}
        submission = {"sha256": "c" * 64, "image_predictions_sha256": "d" * 64}
        validation = {"deployment_inventory": inventory, "manifest_sha256": "b" * 64,
                      "complete": True, "precision": "float16", "image_size": 512,
                      "tta": "identity", "ensemble": False, "macro_auc": .8}
        for change in ("seed", "count", "semantic", "sidecar", "timing"):
            cost = copy.deepcopy(passing_cost(inventory, "c" * 64))
            if change == "seed": cost["reproducibility"]["seed"] = 44
            if change == "count": cost["comparison"]["image_rounds"][1]["compared_images"] = 99
            if change == "semantic": cost["comparison"]["rounds"][0]["semantic_match"] = False
            if change == "sidecar": cost["comparison"]["image_submission_sha256"] = "0" * 64
            if change == "timing": cost["mean_ms"] = 101.
            report = verify_reports(config, inventory, submission=submission, cost=cost, validation=validation,
                                    training_evidence={"complete": True})
            self.assertFalse(report["ready_for_organizer_review"], change)

    def test_xlsx_is_valid_ooxml_and_sha_remains_text(self):
        path = self.root / "材料.xlsx"
        write_xlsx(path, [["名称", "哈希"], ["原模型 & <参数>", "0" * 64]])
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {"[Content_Types].xml", "_rels/.rels", "xl/workbook.xml",
                                                      "xl/_rels/workbook.xml.rels", "xl/worksheets/sheet1.xml"})
            for name in archive.namelist():
                ET.fromstring(archive.read(name))
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            values = [t.text for t in sheet.findall(f".//{{{SHEET_NS}}}t")]
            self.assertIn("原模型 & <参数>", values)
            self.assertIn("0" * 64, values)
        with self.assertRaises(FileExistsError):
            write_xlsx(path, [["cannot overwrite"]])

    def test_five_teachers_share_one_exact_original_pretrain(self):
        base = provenance(BASE_BACKBONE, b"base")
        tiny = provenance(TINY_BACKBONE, b"tiny")
        config = {"pretrained_provenance": tiny,
                  "teacher_provenance": [{"fold_index": fold, "pretrained_provenance": base} for fold in range(5)]}
        entries = unique_pretrains(config)
        self.assertEqual(len(entries), 2)
        self.assertEqual(sorted(len(item["roles"]) for item in entries), [1, 5])

    def test_source_weight_hash_mismatch_is_rejected_without_download(self):
        source = self.root / "original.safetensors"
        source.write_bytes(b"wrong")
        p = {**provenance(BASE_BACKBONE, b"expected"), "weights_path": str(source)}
        with self.assertRaises(ValueError):
            locate_pretrain(p)
        source.write_bytes(b"expected")
        self.assertEqual(locate_pretrain(p), source)

    def test_explicit_sha_cache_mapping_and_verified_copy(self):
        p = provenance(TINY_BACKBONE, b"small fixture")
        source = self.root / f"{p['weights_sha256']}.safetensors"
        source.write_bytes(b"small fixture")
        self.assertEqual(locate_pretrain(p, self.root), source)
        target = self.root / "output" / "model.safetensors"
        copy_verified(source, target, p["weights_sha256"])
        self.assertEqual(target.read_bytes(), source.read_bytes())
        with self.assertRaises(FileExistsError):
            copy_verified(source, target, p["weights_sha256"])
        bad = self.root / "bad-copy"
        with self.assertRaises(ValueError):
            copy_verified(source, bad, "0" * 64)
        self.assertFalse(bad.exists())

    def test_missing_evidence_remains_pending_and_wrong_model_report_is_rejected(self):
        inventory = {"files": [{"path": "student_fp16.pt", "bytes": 10, "sha256": "a" * 64}],
                     "total_bytes": 10, "size_pass": True}
        config = {"official_valid_manifest_sha256": "b" * 64}
        pending = verify_reports(config, inventory, submission=None, cost=None, validation=None)
        self.assertEqual(pending["status"], "pending")
        self.assertFalse(pending["ready_for_organizer_review"])
        with self.assertRaises(ValueError):
            verify_reports(config, inventory, submission=None,
                           cost={"inference_files": {"files": [], "total_bytes": 0}}, validation=None)

    def test_prepared_package_contains_reproducibility_materials_and_stays_pending(self):
        # Entirely synthetic byte fixtures: this does not export or read any model.
        from kd_common import student_preprocessing
        source = self.root / "source"
        source.mkdir()
        for name in CODE_FILES + MANIFEST_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source or manifest fixture\n", encoding="utf-8")
        add_runtime_support(source, self)
        model = self.root / "deployment"
        model.mkdir()
        (model / "student_fp16.pt").write_bytes(b"synthetic exported weight bytes")
        tiny = provenance(TINY_BACKBONE, b"tiny original")
        base = provenance(BASE_BACKBONE, b"base original")
        for p, content in ((tiny, b"tiny original"), (base, b"base original")):
            path = self.root / (p["weights_sha256"] + ".safetensors")
            path.write_bytes(content)
            p["weights_path"] = str(path)
        teachers = [{"fold_index": fold, "pretrained_provenance": base,
                     "checkpoint_sha256": "c" * 64, "selection_protocol": "fixed_epoch_final_ema",
                     "training_complete": True, "schema_version": 6, "artifact_type": "kd_teacher",
                     "model_name": "convnext_v1_base_gap", "weights_source": "ema",
                     "outer_fold_used_for_selection": False, "official_valid_used_for_training": False,
                     "test_used_for_training": False, "epoch": 20, "training_config": {"epochs": 20},
                     "training_subject_ids": ["train_patient"], "oof_subject_ids": ["heldout_patient"]}
                    for fold in range(5)]
        config = {"schema_version": 1, "artifact_type": "kd_student_deployment", "backbone_name": TINY_BACKBONE,
                  "precision": "float16", "image_size": 512, "class_ids": list(range(1, 11)),
                  "tta": "identity", "ensemble": False, "batch_size": 1,
                  "aggregation": "study_mean_probability", "preprocessing": student_preprocessing(),
                  "model_config": {"backbone_name": TINY_BACKBONE, "gradient_checkpointing": False},
                  "pretrained_provenance": tiny, "teacher_provenance": teachers,
                  "inference_files": ["student_fp16.pt", "config.json"],
                  "weights_sha256": hashlib.sha256((model / "student_fp16.pt").read_bytes()).hexdigest(),
                  "source_manifest_sha256": hashlib.sha256((source / MANIFEST_FILES[0]).read_bytes()).hexdigest(),
                  "official_valid_manifest_sha256": hashlib.sha256((source / MANIFEST_FILES[1]).read_bytes()).hexdigest()}
        (model / "config.json").write_text(json.dumps(config))
        output = self.root / "package"
        result = package_submission(model_dir=model, output=output, project_root=source)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(list((output / "pretrain_model").rglob("model.safetensors"))), 2)
        self.assertTrue((output / "code/data/clean_v1_reviewed/test_manifest.csv").is_file())
        self.assertTrue((output / "模型核验材料与数据申明清单.xlsx").is_file())
        self.assertTrue((output / "cost.py").is_file())
        self.assertFalse((output / "submission.csv").exists())
        self.assertEqual(set(path.name for path in (output / "model").iterdir()), {"config.json", "student_fp16.pt"})
        self.assertEqual(json.loads((output / "package_status.json").read_text())["status"], "pending")


if __name__ == "__main__":
    unittest.main()
