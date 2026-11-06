"""CSV-only CV checks.  No image, model, GPU, or training operations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv_splits


FIELDS = ["image_id", "split", "subject_id", "study_id", "study_key", "dicom_id",
          "relative_path", "targets", "cache_relative_path", "status", "training_eligible",
          "file_sha256", "pixel_sha256"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def record(index: int, split: str = "train") -> dict[str, str]:
    patient, study, dicom = f"patient_{index}", f"study_{index}", f"image_{index}"
    labels = [int((index + c * 3) % (c % 4 + 3) == 0) for c in range(10)]
    return {"image_id": f"{split}:{index}", "split": split, "subject_id": patient,
            "study_id": study, "study_key": f"{patient}/{study}", "dicom_id": dicom,
            "relative_path": f"{split}/{patient}/{study}/{dicom}.jpg",
            "targets": json.dumps(labels) if split != "test" else "",
            "cache_relative_path": f"cache/{index}.png", "status": "ok",
            "training_eligible": str(split == "train"),
            "file_sha256": digest(f"file:{split}:{index}"),
            "pixel_sha256": digest(f"pixel:{split}:{index}")}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class OvernightFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.csv"
        self.valid = self.root / "valid.csv"
        self.test = self.root / "test.csv"
        self.plan = self.root / "cv" / "folds.json"
        self.rows = [record(index) for index in range(100)]
        # File equality connects patient0--patient1, pixel equality connects
        # patient1--patient2, and both constraints must be transitive.
        self.rows[1]["file_sha256"] = self.rows[0]["file_sha256"]
        self.rows[2]["pixel_sha256"] = self.rows[1]["pixel_sha256"]
        write_manifest(self.train, self.rows)
        write_manifest(self.valid, [record(200, "valid")])
        # This file deliberately fails CSV parsing if the CV path ever reads it.
        self.test.write_text("DO NOT READ TEST FOR SPLITTING", encoding="utf-8")

    def build(self, **kwargs):
        return cv_splits.build_fold_plan(self.train, official_valid_manifest=self.valid, **kwargs)

    def ensure(self):
        return cv_splits.ensure_fold_plan(self.plan, train_manifest=self.train,
                                          official_valid_manifest=self.valid)

    def test_deterministic_multilabel_and_content_isolation(self) -> None:
        first, second = self.build(), self.build()
        self.assertEqual(first["study_to_fold"], second["study_to_fold"])
        self.assertEqual(first["folds"], second["folds"])
        mapping = first["study_to_fold"]
        self.assertEqual(len({mapping[self.rows[index]["study_key"]] for index in range(3)}), 1)
        self.assertEqual(len({first["patient_to_group"][f"patient_{index}"] for index in range(3)}), 1)
        self.assertEqual(first["duplicate_patient_edges"], {"file_sha256": 1, "pixel_sha256": 1})
        for fold in first["folds"]:
            for role in ("train", "valid"):
                self.assertTrue(all(value > 0 for value in fold[role]["positive_studies"]))
                self.assertTrue(all(value > 0 for value in fold[role]["negative_studies"]))

    def test_original_csv_paths_and_labels_preserved_and_test_not_used(self) -> None:
        snapshots = {path: path.read_bytes() for path in (self.train, self.valid, self.test)}
        real_reader = cv_splits.read_manifest
        with patch.object(cv_splits, "read_manifest", wraps=real_reader) as spy:
            self.ensure()
            for call in spy.call_args_list:
                self.assertNotEqual(Path(call.args[0]), self.test)
                self.assertIn(call.kwargs["split"], {"train", "valid"})
        seen: set[str] = set()
        originals = {row["image_id"]: row for row in self.rows}
        for fold in range(5):
            training = cv_splits.load_fold_records(self.plan, fold, "train")
            validation = cv_splits.load_fold_records(self.plan, fold, "valid")
            self.assertFalse({row["subject_id"] for row in training}
                             & {row["subject_id"] for row in validation})
            for row in validation:
                self.assertNotIn(row["image_id"], seen)
                seen.add(row["image_id"])
                original = originals[row["image_id"]]
                self.assertEqual(row["split"], "train")
                self.assertEqual(row["relative_path"], original["relative_path"])
                self.assertEqual(list(row["targets"]), json.loads(original["targets"]))
        self.assertEqual(seen, set(originals))
        self.ensure()  # accepted plans are validated and reused, never replaced
        self.assertEqual(snapshots, {path: path.read_bytes() for path in snapshots})

    def test_holdout_patient_or_content_overlap_is_rejected(self) -> None:
        patient_overlap = record(0, "valid")
        write_manifest(self.valid, [patient_overlap])
        with self.assertRaisesRegex(ValueError, "share patients"):
            self.build()
        file_overlap = record(200, "valid")
        file_overlap["file_sha256"] = self.rows[0]["file_sha256"]
        write_manifest(self.valid, [file_overlap])
        with self.assertRaisesRegex(ValueError, "exact duplicate file_sha256"):
            self.build()
        pixel_overlap = record(200, "valid")
        pixel_overlap["pixel_sha256"] = self.rows[0]["pixel_sha256"]
        write_manifest(self.valid, [pixel_overlap])
        with self.assertRaisesRegex(ValueError, "exact duplicate pixel_sha256"):
            self.build()

    def test_plan_hash_and_group_tampering_rejected(self) -> None:
        plan = self.ensure()
        plan["study_to_fold"][self.rows[1]["study_key"]] = (
            plan["study_to_fold"][self.rows[0]["study_key"]] + 1) % 5
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "leaks across folds"):
            cv_splits.load_fold_records(self.plan, 0, "valid")
        self.train.write_bytes(self.train.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            cv_splits.read_fold_plan(self.plan)

    def test_insufficient_class_support_is_rejected(self) -> None:
        for row in self.rows:
            labels = json.loads(row["targets"])
            labels[0] = 0
            row["targets"] = json.dumps(labels)
        write_manifest(self.train, self.rows)
        with self.assertRaisesRegex(ValueError, "five positive and five negative"):
            self.build()


if __name__ == "__main__":
    unittest.main()
