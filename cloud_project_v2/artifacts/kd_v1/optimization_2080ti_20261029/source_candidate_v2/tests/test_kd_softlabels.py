"""合成元数据与数组验收：不构建模型、不读原图、不训练、不下载。"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from distill_data import (ARRAY_KEYS, CACHE_SCHEMA, SELECTION_PROTOCOL, TEACHER_BACKBONE,
                          TEACHER_MODEL_NAME, load_soft_labels, sha256_file,
                          validate_cache_arrays, validate_teacher_checkpoint,
                          write_soft_labels)
from kd_common import text_sha256


def synthetic_fixture() -> tuple[list[dict], dict, str, list[dict], dict, dict]:
    records = []
    for fold in range(5):
        for study in range(2):
            subject, study_id = f"p{fold}", f"s{fold}{study}"
            image_id = f"i{fold}{study}"
            records.append({"image_id": image_id, "subject_id": subject, "study_id": study_id,
                "study_key": f"{subject}/{study_id}", "relative_path": f"train/{subject}/{study_id}/{image_id}.jpg",
                "file_sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                "split": "train", "training_eligible": True, "targets": [study] * 10})
    plan = {"source_manifest_sha256": "a" * 64,
        "study_to_fold": {r["study_key"]: int(r["subject_id"][1:]) for r in records},
        "study_to_patient": {r["study_key"]: r["subject_id"] for r in records}}
    plan_hash = "b" * 64
    contents = {"model_card": "---\nlicense: apache-2.0\n---\nSynthetic ImageNet fixture",
                "apache_license": "Apache License SYNTHETIC TEST ONLY",
                "upstream_license": "MIT License SYNTHETIC TEST ONLY"}
    revision = "c" * 40
    provenance = {"backbone_name": TEACHER_BACKBONE, "repo_id": f"timm/{TEACHER_BACKBONE}",
        "license": "apache-2.0", "pretrain_dataset": "ImageNet-22k", "finetune_dataset": "ImageNet-1k",
        "weights_sha256": "d" * 64, "revision": revision,
        "weights_url": f"https://huggingface.co/timm/{TEACHER_BACKBONE}/resolve/{revision}/model.safetensors",
        "materials": {name: {"text": text, "sha256": text_sha256(text)} for name, text in contents.items()}}
    teachers = []
    for fold in range(5):
        teachers.append({"schema_version": 6, "artifact_type": "kd_teacher",
            "model_name": TEACHER_MODEL_NAME, "model_config": {"backbone_name": TEACHER_BACKBONE},
            "class_ids": list(range(1, 11)), "image_size": 768, "fold_index": fold,
            "fold_plan_sha256": plan_hash, "source_manifest_sha256": "a" * 64,
            "selection_protocol": SELECTION_PROTOCOL, "training_complete": True,
            "inference_ready": True, "weights_source": "ema", "epoch": 20,
            "training_config": {"epochs": 20},
            "training_subject_ids": [f"p{i}" for i in range(5) if i != fold],
            "oof_subject_ids": [f"p{fold}"],
            "training_image_ids": sorted(r["image_id"] for r in records if r["subject_id"] != f"p{fold}"),
            "oof_image_ids": sorted(r["image_id"] for r in records if r["subject_id"] == f"p{fold}"),
            "outer_fold_used_for_selection": False, "official_valid_used_for_training": False,
            "test_used_for_training": False,
            "preprocessing": {"image_size": 768, "validation_size": 768,
                "resize": "letterboxLANCZOS", "validation_tta": "identity",
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                "color_mode": "RGB", "exif_transpose": True, "data_source": "raw_readonly"},
            "loss_config": {"pos_weight": [1.] * 10}, "pretrained_provenance": deepcopy(provenance),
            "checkpoint_sha256": f"{fold + 1:064x}"})
    metadata = {"schema": CACHE_SCHEMA, "source_split": "train", "class_ids": list(range(1, 11)),
        "source_manifest_sha256": "a" * 64, "fold_plan_sha256": plan_hash, "image_size": 768,
        "prediction_unit": "image", "output_type": "raw_logits", "tta": "identity",
        "selection_protocol": SELECTION_PROTOCOL, "num_images": len(records), "complete": True,
        "teachers": teachers}
    arrays = {"image_ids": np.asarray([r["image_id"] for r in records]),
        "subject_ids": np.asarray([r["subject_id"] for r in records]),
        "study_keys": np.asarray([r["study_key"] for r in records]),
        "relative_paths": np.asarray([r["relative_path"] for r in records]),
        "source_image_sha256": np.asarray([r["file_sha256"] for r in records]),
        "fold_indices": np.asarray([plan["study_to_fold"][r["study_key"]] for r in records], dtype=np.int16),
        "logits": np.arange(len(records) * 10, dtype=np.float32).reshape(len(records), 10) / 10,
        "teacher_pos_weights": np.ones((len(records), 10), dtype=np.float32)}
    assert set(arrays) == ARRAY_KEYS
    return records, plan, plan_hash, teachers, metadata, arrays


class SoftLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records, self.plan, self.plan_hash, self.teachers, self.metadata, self.arrays = synthetic_fixture()

    def validate(self, metadata=None, arrays=None, records=None):
        return validate_cache_arrays(metadata if metadata is not None else self.metadata,
            arrays if arrays is not None else self.arrays, plan=self.plan,
            plan_hash=self.plan_hash, records=records if records is not None else self.records)

    def test_complete_cache_has_stable_image_lookup_and_readonly_arrays(self):
        result = self.validate()
        self.assertEqual(result.index["i31"], 7)
        np.testing.assert_array_equal(result.logits[7], self.arrays["logits"][7])
        self.assertFalse(result.logits.flags.writeable)
        self.assertFalse(result.teacher_pos_weights.flags.writeable)

    def test_train_only_and_full_coverage(self):
        for split in ("valid", "test"):
            records = deepcopy(self.records)
            records[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "original train"):
                self.validate(records=records)
        arrays = deepcopy(self.arrays)
        arrays["image_ids"][0] = arrays["image_ids"][1]
        with self.assertRaisesRegex(ValueError, "every train image"):
            self.validate(arrays=arrays)

    def test_patient_teacher_leakage_is_rejected(self):
        metadata = deepcopy(self.metadata)
        metadata["teachers"][0]["training_subject_ids"].append("p0")
        with self.assertRaisesRegex(ValueError, "strict-OOF"):
            self.validate(metadata=metadata)

    def test_early_stopped_or_outer_fold_selected_teacher_is_rejected(self):
        for key, value in (("selection_protocol", "best_outer_auc"), ("training_complete", False),
                           ("inference_ready", False), ("epoch", 19)):
            metadata = deepcopy(self.metadata)
            metadata["teachers"][0][key] = value
            with self.assertRaisesRegex(ValueError, "fixed-epoch"):
                self.validate(metadata=metadata)

    def test_old_v2_and_nc_provenance_are_rejected(self):
        metadata = deepcopy(self.metadata)
        metadata["teachers"][0]["model_config"]["backbone_name"] = "convnextv2_base.fcmae_ft_in1k"
        with self.assertRaisesRegex(ValueError, "V1 teacher"):
            self.validate(metadata=metadata)
        metadata = deepcopy(self.metadata)
        metadata["teachers"][0]["pretrained_provenance"]["license"] = "cc-by-nc-4.0"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate(metadata=metadata)

    def test_mismatched_image_fingerprint_fold_path_and_weights_are_rejected(self):
        for field, value in (("source_image_sha256", "f" * 64), ("subject_ids", "p9"),
                             ("relative_paths", "test/p0/s00/i00.jpg"), ("fold_indices", 4)):
            arrays = deepcopy(self.arrays)
            arrays[field][0] = value
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.validate(arrays=arrays)
        arrays = deepcopy(self.arrays)
        arrays["teacher_pos_weights"][0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.validate(arrays=arrays)

    def test_nonfinite_logits_wrong_dtype_and_pickle_identity_are_rejected(self):
        arrays = deepcopy(self.arrays)
        arrays["logits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            self.validate(arrays=arrays)
        arrays = deepcopy(self.arrays)
        arrays["logits"] = arrays["logits"].astype(np.float16)
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            self.validate(arrays=arrays)
        arrays = deepcopy(self.arrays)
        arrays["image_ids"] = arrays["image_ids"].astype(object)
        with self.assertRaisesRegex(ValueError, "never pickle"):
            self.validate(arrays=arrays)

    def test_teacher_checkpoint_requires_ema_state(self):
        checkpoint = deepcopy(self.teachers[0])
        with self.assertRaisesRegex(ValueError, "state_dict"):
            validate_teacher_checkpoint(checkpoint, fold=0, plan=self.plan,
                                        plan_hash=self.plan_hash, records=self.records)
        checkpoint["model_state_dict"] = {"synthetic": 0}
        validate_teacher_checkpoint(checkpoint, fold=0, plan=self.plan,
                                    plan_hash=self.plan_hash, records=self.records)

    def test_cache_roundtrip_hash_binding_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "folds.json"
            plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
            plan_hash = sha256_file(plan_path)
            teachers = deepcopy(self.teachers)
            for teacher in teachers:
                teacher["fold_plan_sha256"] = plan_hash
            output = root / "cache"
            with patch("distill_data._read_validated_plan", return_value=self.plan):
                result = write_soft_labels(output, plan_path, self.records,
                                            self.arrays["logits"], teachers)
                self.assertEqual(result.image_ids, tuple(r["image_id"] for r in self.records))
                with self.assertRaises(FileExistsError):
                    write_soft_labels(output, plan_path, self.records,
                                      self.arrays["logits"], teachers)
                with (output / "soft_labels.npz").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    load_soft_labels(output, plan_path, self.records)

    def test_real_csv_plan_and_cache_integration_without_mocks(self):
        """真实读取合成CSV/划分/材料验证，不打开任何图片或调用模型。"""
        from cv_splits import ensure_fold_plan
        from dataset import read_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, valid, plan_path = root / "train.csv", root / "valid.csv", root / "folds.json"
            rows = []
            for original in self.records:
                row = dict(original)
                row.update(dicom_id=row["image_id"], targets=json.dumps(row["targets"]),
                    training_eligible="True", cache_relative_path="unused.png", status="ok",
                    pixel_sha256=hashlib.sha256((row["image_id"] + "pixel").encode()).hexdigest())
                rows.append(row)
            holdout = dict(rows[0])
            holdout.update(image_id="valid:unique", split="valid", subject_id="valid_patient",
                study_id="valid_study", study_key="valid_patient/valid_study", dicom_id="valid_image",
                relative_path="valid/valid_patient/valid_study/valid_image.jpg", training_eligible="False",
                file_sha256="8" * 64, pixel_sha256="9" * 64)
            for path, source in ((train, rows), (valid, [holdout])):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(source)
            plan = ensure_fold_plan(plan_path, train_manifest=train, official_valid_manifest=valid)
            records = read_manifest(train, split="train")
            teachers = deepcopy(self.teachers)
            for fold, teacher in enumerate(teachers):
                training = [r for r in records if plan["study_to_fold"][r["study_key"]] != fold]
                oof = [r for r in records if plan["study_to_fold"][r["study_key"]] == fold]
                teacher.update(fold_plan_sha256=sha256_file(plan_path),
                    source_manifest_sha256=sha256_file(train),
                    training_subject_ids=sorted({r["subject_id"] for r in training}),
                    oof_subject_ids=sorted({r["subject_id"] for r in oof}),
                    training_image_ids=sorted(r["image_id"] for r in training),
                    oof_image_ids=sorted(r["image_id"] for r in oof))
            output = root / "cache"
            result = write_soft_labels(output, plan_path, records, self.arrays["logits"], teachers)
            again = load_soft_labels(output, plan_path, records)
            self.assertEqual(result.index, again.index)
            batch_ids = [records[3]["image_id"], records[0]["image_id"]]
            batch = again.logits[[again.index[image_id] for image_id in batch_ids]]
            np.testing.assert_array_equal(batch, self.arrays["logits"][[3, 0]])


if __name__ == "__main__":
    unittest.main()
