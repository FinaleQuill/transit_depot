"""Synthetic small-file ensemble packaging/evidence tests; no tensors or models."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET
import zipfile

from audit_trail import record_checkpoint
from build_training_evidence import build_evidence
from deploy_kd import (BASE_BACKBONE, TINY_BACKBONE, MEAN, STD,
                       deployment_inventory, sha256_file)
from export_student import make_deployment_config
from kd_common import APPROVED_PRETRAINS, text_sha256
from package_kd_submission import (CODE_FILES, MANIFEST_FILES, SHEET_NS,
                                   collect_training_evidence, package_submission,
                                   unique_pretrains, verify_reports)


def source_provenance(backbone, content, root):
    approved = APPROVED_PRETRAINS[backbone]
    revision = "a" * 40
    texts = {"model_card": "license: apache-2.0\nSynthetic ensemble package fixture",
             "apache_license": "Apache License\nSynthetic ensemble package fixture",
             "upstream_license": "MIT License\nSynthetic ensemble package fixture"}
    digest = hashlib.sha256(content).hexdigest()
    path = root / (digest + ".safetensors")
    path.write_bytes(content)
    return {**approved, "backbone_name": backbone, "revision": revision,
            "weights_sha256": digest, "weights_path": str(path),
            "weights_url": f"https://huggingface.co/{approved['repo_id']}/resolve/{revision}/model.safetensors",
            "materials": {k: {"text": t, "sha256": text_sha256(t)} for k, t in texts.items()}}


class EnsemblePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kd-ensemble-package-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in CODE_FILES + MANIFEST_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# Synthetic source/manifest fixture: {name}\n")
        plan = self.source / "data/cv5_overnight/folds.json"
        plan.parent.mkdir(parents=True)
        plan.write_text('{"synthetic_fold_evidence":true}\n')
        self.train_hash = sha256_file(self.source / MANIFEST_FILES[0])
        self.valid_hash = sha256_file(self.source / MANIFEST_FILES[1])
        self.fold_hash = sha256_file(plan)
        self.base = source_provenance(BASE_BACKBONE, b"synthetic base original", self.root)
        self.tiny = source_provenance(TINY_BACKBONE, b"synthetic tiny original", self.root)
        self.teachers = []
        for fold in range(5):
            self.teachers.append({"fold_index": fold, "pretrained_provenance": self.base,
                "checkpoint_sha256": hashlib.sha256(f"teacher {fold} final".encode()).hexdigest(),
                "selection_protocol": "fixed_epoch_final_ema", "training_complete": True,
                "schema_version": 6, "artifact_type": "kd_teacher", "model_name": "convnext_v1_base_gap",
                "weights_source": "ema", "outer_fold_used_for_selection": False,
                "official_valid_used_for_training": False, "test_used_for_training": False,
                "epoch": 2, "training_config": {"epochs": 2},
                "training_subject_ids": [f"training_{fold}"], "oof_subject_ids": [f"heldout_{fold}"],
                "source_manifest_sha256": self.train_hash, "fold_plan_sha256": self.fold_hash})
        self.model = self.root / "model"
        self.model.mkdir()
        self.members = []
        self.best_bytes = []
        for index, size in enumerate((512, 768)):
            weight = self.model / f"student_{index:02d}_fp16.pt"
            weight.write_bytes(f"synthetic runtime fp16 {index}".encode())
            best = f"synthetic saved student checkpoint {index}".encode()
            self.best_bytes.append(best)
            protocol = {"teacher_tta": "identity_hflip", "teacher_views": ["identity", "hflip"],
                        "student_training_size": size, "validation_tta": "hflip2",
                        "deployment_tta": "hflip2", "deployment_ensemble": False,
                        "soft_target_aggregation": "mean_sigmoid_after_pos_weight_alignment_and_temperature"}
            self.members.append({"weights_file": weight.name, "weights_sha256": sha256_file(weight),
                "source_checkpoint_sha256": hashlib.sha256(best).hexdigest(), "weights_source": "ema",
                "model_config": {"backbone_name": TINY_BACKBONE, "gradient_checkpointing": False},
                "image_size": size, "tta": "hflip2", "seed": 43 + index,
                "pretrained_provenance": self.tiny, "teacher_provenance": copy.deepcopy(self.teachers),
                "distillation": protocol,
                "preprocessing": {"image_size": size, "validation_size": size, "training_scales": [size],
                    "validation_tta": "hflip2", "resize": "letterboxLANCZOS", "color_mode": "RGB",
                    "exif_transpose": True, "mean": list(MEAN), "std": list(STD)}})
        self.config = make_deployment_config(self.members, source_manifest_sha256=self.train_hash,
            fold_plan_sha256=self.fold_hash, official_valid_manifest_sha256=self.valid_hash)
        (self.model / "config.json").write_text(json.dumps(self.config, ensure_ascii=False, indent=2) + "\n")

    def archive(self, root):
        root.mkdir()
        for name in CODE_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# Synthetic actual executed source fixture\n")
        (root / "environment_lock.txt").write_text("synthetic-dependency==1.0\n")
        (root / "source_manifest.json").write_text('{"synthetic_environment_record":true}')
        return root

    def ledger(self, root, role, final_hash):
        for epoch in (1, 2):
            record_checkpoint(root / "last.pt", hashlib.sha256(f"{role} epoch{epoch}".encode()).hexdigest(),
                epoch=epoch, role="last", historical=True, evidence_source="synthetic unit fixture",
                observed_status_at="2026-10-26T08:00:00+00:00")
        record_checkpoint(root / "best.pt", final_hash, epoch=2, role="best", historical=True,
            evidence_source="synthetic unit fixture", observed_status_at="2026-10-26T08:00:00+00:00")

    def evidence(self):
        teacher_root = self.root / "teacher_evidence"
        teacher_root.mkdir()
        archive = self.archive(self.root / "teacher_archive")
        log = teacher_root / "teacher.log"
        log.write_text("synthetic complete teacher training logs\n")
        manifest = {"schema_version": 1, "artifact_type": "kd_training_evidence",
            "source_archives": [{"id": "teacher_source", "path": str(archive)}],
            "source_version_timeline": [], "runs": [], "console_coverage": [],
            "console_logs": [{"id": "teacher_log", "path": str(log), "coverage_from_start": "complete",
                              "run_ids": [f"teacher_fold_{f}" for f in range(5)]}],
            "supporting_files": [{"path": str(archive / "environment_lock.txt"), "kind": "dependency_lock"},
                                 {"path": str(archive / "source_manifest.json"), "kind": "environment_record"}]}
        for fold, teacher in enumerate(self.teachers):
            role = f"teacher_fold_{fold}"
            run = teacher_root / role
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"seed": 43, "pretrained_provenance": self.base}))
            self.ledger(run, role, teacher["checkpoint_sha256"])
            manifest["runs"].append({"id": role, "role": "teacher", "complete": True, "expected_epochs": 2,
                "config": str(run / "config.json"), "checkpoint_audit": str(run / "checkpoint_audit.jsonl"),
                "source_archive_id": "teacher_source"})
            manifest["source_version_timeline"].append({"run_id": role, "source_archive_id": "teacher_source"})
        (teacher_root / "evidence_manifest.json").write_text(json.dumps(manifest))
        students = []
        for index, member in enumerate(self.members):
            run = self.root / f"student_run_{index}"
            run.mkdir()
            native = run / "evidence"
            native.mkdir()
            source = self.archive(self.root / f"student_archive_{index}")
            console = native / "console_native.log"
            console.write_text("synthetic complete student invocation logs\n")
            (run / "best.pt").write_bytes(self.best_bytes[index])
            (run / "status.json").write_text(json.dumps({"state": "completed", "completed_epochs": 2}))
            (run / "config.json").write_text(json.dumps({"seed": member["seed"], "image_size": member["image_size"],
                "distillation": member["distillation"]}))
            (run / "pretrained_provenance.json").write_text(json.dumps(self.tiny))
            self.ledger(run, f"student_{index}", member["source_checkpoint_sha256"])
            receipt = {"cwd": str(self.source), "source_archive": str(source), "console_log": str(console),
                "started_at": "2026-10-26T08:00:00+00:00", "ended_at": "2026-10-26T09:00:00+00:00",
                "exit_code": 0, "coverage_from_start": "complete", "console_bytes": console.stat().st_size,
                "preexisting_run_artifacts": []}
            (native / "command_native.json").write_text(json.dumps(receipt))
            students.append(run)
        return teacher_root, students

    def test_two_member_mixed_resolution_package_keeps_all_runtime_and_sources(self):
        originals = unique_pretrains(self.config)
        self.assertEqual(len(originals), 2)
        self.assertEqual(sorted(len(item["roles"]) for item in originals), [2, 5])
        teachers, students = self.evidence()
        output_evidence = self.root / "assembled_evidence"
        preview = build_evidence(model_dir=self.model, teacher_evidence=teachers,
            student_runs=students, output=output_evidence)
        self.assertFalse(preview["execute"])
        self.assertFalse(output_evidence.exists())
        build_evidence(model_dir=self.model, teacher_evidence=teachers,
            student_runs=students, output=output_evidence, execute=True)
        result = package_submission(model_dir=self.model, output=self.root / "package",
            project_root=self.source, training_evidence=output_evidence)
        self.assertEqual(result["status"], "pending")
        self.assertTrue(result["training_evidence"]["complete"], result["training_evidence"]["pending_items"])
        self.assertEqual({r["id"] for r in result["training_evidence"]["runs"]},
            {f"teacher_fold_{f}" for f in range(5)} | {"student_0", "student_1"})
        self.assertEqual(result["runtime_weights"], ["model/student_00_fp16.pt", "model/student_01_fp16.pt"])
        self.assertEqual({p.name for p in (self.root / "package/model").iterdir()},
            {"config.json", "student_00_fp16.pt", "student_01_fp16.pt"})
        self.assertEqual(result["inference_inventory"], deployment_inventory(self.model, self.config))
        self.assertEqual(len(result["pretraining_sources"]), 2)
        self.assertTrue(all((self.root / "package/code" / name).is_file() for name in CODE_FILES))
        with zipfile.ZipFile(self.root / "package/模型核验材料与数据申明清单.xlsx") as archive:
            xml = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        texts = [node.text or "" for node in xml.findall(f".//{{{SHEET_NS}}}t")]
        self.assertTrue(any("2个Tiny" in text and "[512, 768]" in text and "hflip2" in text and "True" in text for text in texts))

    def test_missing_one_student_evidence_prevents_ready_even_with_other_reports(self):
        teachers, students = self.evidence()
        assembled = self.root / "assembled"
        build_evidence(model_dir=self.model, teacher_evidence=teachers, student_runs=students,
                       output=assembled, execute=True)
        manifest_path = assembled / "evidence_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runs"] = [r for r in manifest["runs"] if r["id"] != "student_1"]
        manifest_path.write_text(json.dumps(manifest))
        evidence = collect_training_evidence(assembled, self.root / "frozen", self.config)
        self.assertFalse(evidence["complete"])
        self.assertIn("training evidence for student_1", evidence["pending_items"])
        inventory = deployment_inventory(self.model, self.config)
        source_sha = "9" * 64
        validation = {"deployment_inventory": inventory, "manifest_sha256": self.valid_hash,
            "complete": True, "precision": "float16", "image_size": None,
            "image_sizes": [512, 768], "tta": "hflip2", "ensemble": True,
            "member_count": 2, "members": [{k: m.get(k) for k in ("source_checkpoint_sha256", "weights_sha256", "image_size", "tta", "seed")} for m in self.members],
            "deployment_config_sha256": sha256_file(self.model / "config.json"),
            "macro_auc": .8, "per_class_auc": [.8] * 10}
        timing = {"inference_files": inventory, "submission_sha256": source_sha,
            "official_compliance_pass": True,
            "deployment_config_sha256": sha256_file(self.model / "config.json"),
            "deployment": {"member_count": 2, "image_sizes": [512, 768], "tta": "hflip2", "ensemble": True}}
        result = verify_reports(self.config, inventory, submission={"sha256": source_sha},
                                cost=timing, validation=validation, training_evidence=evidence)
        self.assertFalse(result["ready_for_organizer_review"])
        self.assertEqual(result["pending_items"], ["training evidence for student_1"])
        # All other reports are bound to the correct config; a tampered member
        # summary or timing-config hash must be rejected before readiness.
        wrong_member = copy.deepcopy(validation)
        wrong_member["members"][1]["image_size"] = 512
        with self.assertRaisesRegex(ValueError, "member declarations"):
            verify_reports(self.config, inventory, submission={"sha256": source_sha},
                cost=timing, validation=wrong_member, training_evidence={"complete": True})
        wrong_cost = {**timing, "deployment_config_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "exact multi-member deployment config"):
            verify_reports(self.config, inventory, submission={"sha256": source_sha},
                cost=wrong_cost, validation=validation, training_evidence={"complete": True})

    def test_direct_evidence_collection_checks_member_seed_and_resolution(self):
        teachers, students = self.evidence()
        assembled = self.root / "assembled"
        build_evidence(model_dir=self.model, teacher_evidence=teachers, student_runs=students,
                       output=assembled, execute=True)
        path = students[1] / "config.json"
        original = json.loads(path.read_text())
        for field, value, message in (("seed", 999, "Training seed"),
                                      ("image_size", 512, "Training resolution"),
                                      ("distillation", {}, "Training distillation")):
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaisesRegex(ValueError, message):
                    collect_training_evidence(assembled, self.root / ("frozen_" + field), self.config)
        path.write_text(json.dumps(original))

    def test_evidence_builder_rejects_mismatched_student_member(self):
        teachers, students = self.evidence()
        (students[1] / "best.pt").write_bytes(b"another synthetic checkpoint")
        with self.assertRaisesRegex(ValueError, "differs from the deployed checkpoint"):
            build_evidence(model_dir=self.model, teacher_evidence=teachers, student_runs=students,
                output=self.root / "wrong_evidence", execute=True)


if __name__ == "__main__":
    unittest.main()
