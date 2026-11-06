"""Mixed fixed teacher budgets, using synthetic JSON and tiny fake weights only."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_kd_softlabels import synthetic_fixture
from distill_data import validate_cache_arrays
import extract_soft_labels
from teacher_collection import (BACKBONE, SELECTION_PROTOCOL, read_teacher_collection,
                                validate_collection_checkpoint)


class TeacherCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mixed-teachers-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records, self.plan, _, self.teachers, self.metadata, self.arrays = synthetic_fixture()
        self.plan_path = self.root / "folds.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.plan_hash = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.entries = []
        for fold, teacher in enumerate(self.teachers):
            epochs = 20 if fold == 0 else 12
            run = self.root / ("original20" if fold == 0 else "new12")
            folder = run / f"fold_{fold}"
            folder.mkdir(parents=True)
            (folder / "best.pt").write_bytes(b"SYNTHETIC NOT A MODEL")
            teacher.update(fold_plan_sha256=self.plan_hash, epoch=epochs,
                           model_state_dict={"synthetic": 0},
                           training_config={"epochs":epochs, "fold":fold, "seed":43,
                                            "batch_size":2, "accumulation_steps":16})
            run_metadata = {"schema_version":6, "artifact_type":"kd_teacher_run",
                "backbone_name":BACKBONE, "image_size":768, "epochs":epochs, "seed":43,
                "selection_protocol":SELECTION_PROTOCOL, "fold_plan_sha256":self.plan_hash,
                "official_valid_used_for_training":False, "outer_fold_used_for_selection":False}
            (run / "teacher_run.json").write_text(json.dumps(run_metadata))
            binding = {key:teacher[key] for key in ("fold_index","fold_plan_sha256",
                "source_manifest_sha256","selection_protocol","outer_fold_used_for_selection")}
            config = {**teacher["training_config"], "binding":binding,
                      "model_config":teacher["model_config"],
                      "pretrained_provenance":teacher["pretrained_provenance"]}
            (folder / "config.json").write_text(json.dumps(config))
            self.entries.append({"fold_index":fold, "checkpoint":str(folder / "best.pt"), "epochs":epochs})
        self.value = {"schema_version":1, "artifact_type":"kd_teacher_collection",
            "fold_plan":str(self.plan_path), "fold_plan_sha256":self.plan_hash,
            "source_manifest_sha256":self.plan["source_manifest_sha256"], "image_size":768,
            "selection_protocol":SELECTION_PROTOCOL, "teachers":self.entries}
        self.path = self.root / "collection.json"
        self.write(self.value)

    def write(self, value):
        self.path.write_text(json.dumps(value))

    def validate(self, fold=0, teacher=None, declaration=None):
        validate_collection_checkpoint(teacher if teacher is not None else self.teachers[fold],
            declaration if declaration is not None else self.entries[fold], plan=self.plan,
            plan_hash=self.plan_hash, records=self.records)

    def test_mixed_twenty_twelve_collection_and_cache_keep_strict_oof(self):
        value = read_teacher_collection(self.path, fold_plan_path=self.plan_path, require_checkpoints=True)
        self.assertEqual([entry["epochs"] for entry in value["teachers"]], [20,12,12,12,12])
        for fold in range(5):
            self.validate(fold)
        metadata = {**self.metadata, "fold_plan_sha256":self.plan_hash, "teachers":self.teachers}
        cache = validate_cache_arrays(metadata,self.arrays,plan=self.plan,plan_hash=self.plan_hash,
                                      records=self.records)
        self.assertEqual(len(cache.image_ids),len(self.records))

    def test_preview_allows_uncreated_final_weights_but_execution_requires_them(self):
        checkpoint = Path(self.entries[4]["checkpoint"])
        checkpoint.unlink()
        self.assertEqual(len(read_teacher_collection(self.path)["teachers"]),5)
        with self.assertRaises(FileNotFoundError):
            read_teacher_collection(self.path,require_checkpoints=True)

    def test_parsing_and_validation_do_not_open_checkpoint_bytes(self):
        original = Path.open
        def guarded(path,*args,**kwargs):
            if path.suffix == ".pt":
                raise AssertionError("Metadata validation must not read checkpoint weights")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",guarded):
            read_teacher_collection(self.path,require_checkpoints=True)
            self.validate()

    def test_missing_duplicate_and_bool_folds_are_rejected(self):
        for entries in (self.entries[:-1],self.entries[:4]+[self.entries[0]],
                        [{**self.entries[0],"fold_index":False}]+self.entries[1:]):
            self.write({**self.value,"teachers":entries})
            with self.assertRaises(ValueError):
                read_teacher_collection(self.path)

    def test_changed_plan_fingerprint_is_rejected(self):
        self.plan_path.write_text(json.dumps({**self.plan,"changed":True}))
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            read_teacher_collection(self.path)

    def test_declared_epochs_must_match_original_frozen_run(self):
        self.write({**self.value,"teachers":[{**self.entries[0],"epochs":12}]+self.entries[1:]})
        with self.assertRaisesRegex(ValueError,"frozen training-run"):
            read_teacher_collection(self.path,require_checkpoints=True)

    def test_unfinished_and_metadata_relabelled_checkpoint_are_rejected(self):
        broken = deepcopy(self.teachers[0]);broken["training_complete"] = False
        with self.assertRaisesRegex(ValueError,"complete, fixed-epoch"):
            self.validate(teacher=broken)
        broken = deepcopy(self.teachers[0]);broken["epoch"] = 12;broken["training_config"]["epochs"] = 12
        with self.assertRaisesRegex(ValueError,"declared collection epochs"):
            self.validate(teacher=broken)

    def test_old_v2_and_patient_leakage_remain_rejected(self):
        broken = deepcopy(self.teachers[1]);broken["model_config"]["backbone_name"] = "convnextv2_base.fcmae_ft_in1k"
        with self.assertRaisesRegex(ValueError,"V1 teacher"):
            self.validate(1,teacher=broken)
        broken = deepcopy(self.teachers[1]);broken["training_subject_ids"].append("p1")
        with self.assertRaisesRegex(ValueError,"strict-OOF"):
            self.validate(1,teacher=broken)

    def test_checkpoint_and_original_config_initialization_must_agree(self):
        config_path = Path(self.entries[0]["checkpoint"]).parent / "config.json"
        config = json.loads(config_path.read_text())
        config["pretrained_provenance"]["weights_sha256"] = "f" * 64
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,"initialization differs"):
            self.validate()

    def test_recorded_config_cannot_declare_nc_initialization(self):
        config_path = Path(self.entries[0]["checkpoint"]).parent / "config.json"
        config = json.loads(config_path.read_text())
        config["pretrained_provenance"]["license"] = "cc-by-nc-4.0"
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,"provenance"):
            read_teacher_collection(self.path,require_checkpoints=True)

    def test_collection_cli_preview_does_not_run_extract(self):
        stream = io.StringIO()
        with patch.object(extract_soft_labels,"extract") as execute,contextlib.redirect_stdout(stream):
            extract_soft_labels.main(["--fold-plan",str(self.plan_path),"--teacher-collection",str(self.path),
                                     "--output",str(self.root / "cache")])
        execute.assert_not_called()
        value = json.loads(stream.getvalue())
        self.assertFalse(value["execute"])
        self.assertEqual([entry["epochs"] for entry in value["teachers"]],[20,12,12,12,12])
        self.assertFalse((self.root / "cache").exists())

    def test_legacy_root_preview_still_works_without_weights(self):
        stream = io.StringIO()
        with patch.object(extract_soft_labels,"extract") as execute,contextlib.redirect_stdout(stream):
            extract_soft_labels.main(["--fold-plan",str(self.plan_path),"--teachers-root",str(self.root / "legacy"),
                                     "--output",str(self.root / "cache")])
        execute.assert_not_called()
        self.assertEqual(len(json.loads(stream.getvalue())["teachers"]),5)


if __name__ == "__main__":
    unittest.main()
