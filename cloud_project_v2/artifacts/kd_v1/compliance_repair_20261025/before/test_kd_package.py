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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kd_common import APPROVED_PRETRAINS, text_sha256, BASE_BACKBONE, TINY_BACKBONE
from package_kd_submission import (copy_verified, locate_pretrain, unique_pretrains,
                                   verify_reports, write_xlsx, SHEET_NS, package_submission,
                                   CODE_FILES, MANIFEST_FILES)


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
