"""Synthetic, CPU-only preparation tests; no model/GPU or real dataset loading."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_latency_inputs import prepare, sha256


class LatencyInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="latency-input-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.image_root = self.base / "raw"
        self.root = self.image_root / "cxr_first"
        self.manifest = self.base / "valid_manifest.csv"
        self.output = self.base / "prepared"
        self.rows = []
        # Multiple Studies per patient exercise patient-wide warmup exclusion.
        for patient in range(30):
            for study_number, size in enumerate((2, 3)):
                subject = f"subject_{patient}"
                study = f"study_{patient * 2 + study_number}"
                for image in range(size):
                    filename = f"image_{image}.jpg"
                    relative = f"valid/{subject}/{study}/{filename}"
                    data = f"synthetic-file-{relative}".encode()
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    self.rows.append({"image_id": f"valid:{len(self.rows):06d}",
                                      "split": "valid", "subject_id": subject,
                                      "study_id": study, "study_key": f"{subject}/{study}",
                                      "dicom_id": Path(filename).stem, "relative_path": relative,
                                      "file_sha256": hashlib.sha256(data).hexdigest(),
                                      "label_text": "unchanged original label",
                                      "review_flags": "arbitrary quality field"})
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def preview(self):
        return prepare(self.manifest, self.image_root, self.output)

    def test_preview_deterministic_exact_complete_and_patient_disjoint(self):
        first = self.preview()
        self.assertEqual(first, self.preview())
        self.assertFalse(self.output.exists())
        self.assertFalse(first["executed"])
        measured, warmup = first["measured"], first["warmup"]
        self.assertEqual((len(measured), len(warmup)), (100, 5))
        self.assertFalse({row["subject_id"] for row in measured}
                         & {row["subject_id"] for row in warmup})
        for group in (measured, warmup):
            for key, count in Counter(row["study_key"] for row in group).items():
                self.assertEqual(count, first["full_study_image_counts"][key])
        self.assertEqual(len(first["full_study_image_counts"]), 60)

    def test_selection_independent_of_labels_quality_and_csv_order(self):
        first = self.preview()
        for row in self.rows:
            row["label_text"] = "replacement values do not affect selection"
            row["review_flags"] = "completely different"
        self.rows.reverse()
        self.write_manifest()
        second = self.preview()
        self.assertEqual(first["measured"], second["measured"])
        self.assertEqual(first["warmup"], second["warmup"])
        changed_seed = prepare(self.manifest, self.image_root, self.output, seed=9)
        self.assertNotEqual(second["measured"], changed_seed["measured"])

    def test_execute_copies_bytes_preserves_sources_and_only_hashes_selected(self):
        plan = self.preview()
        selected = {row["image_id"] for row in plan["measured"] + plan["warmup"]}
        # Nonselected content is neither inspected nor used for selection.
        unselected = next(row for row in self.rows if row["image_id"] not in selected)
        (self.root / unselected["relative_path"]).write_bytes(b"not a selected input")
        manifest_before = self.manifest.read_bytes()
        original = {str(path): (sha256(path), path.stat().st_mtime_ns)
                    for path in self.root.rglob("*.jpg")}
        result = prepare(self.manifest, self.image_root, self.output, execute=True)
        self.assertEqual(result, json.loads((self.output / "selection.json").read_text()))
        self.assertTrue(result["executed"] and result["hashes_verified"])
        self.assertEqual(self.manifest.read_bytes(), manifest_before)
        self.assertEqual(original, {str(path): (sha256(path), path.stat().st_mtime_ns)
                                    for path in self.root.rglob("*.jpg")})
        copies = list(self.output.rglob("*.jpg"))
        self.assertEqual(len(copies), 105)
        for row in result["measured"] + result["warmup"]:
            source, target = Path(row["source_path"]), self.output / row["output_relative_path"]
            self.assertFalse(target.is_symlink())
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(sha256(target), row["file_sha256"])
            self.assertEqual(source.stat().st_mtime_ns, target.stat().st_mtime_ns)
        self.assertFalse(result["labels_rewritten"] or result["test_used"] or result["gpu_used"])
        self.assertEqual(result["origin_split"], "valid")
        self.assertFalse(list(self.output.glob("*.csv")))

    def test_hash_mismatch_rejected_before_output_creation(self):
        row = self.preview()["warmup"][0]
        Path(row["source_path"]).write_bytes(b"changed original")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare(self.manifest, self.image_root, self.output, execute=True)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_raw_manifest_overlap_rejected(self):
        self.output.mkdir()
        marker = self.output / "preserve"
        marker.write_text("original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.manifest, self.image_root, self.output, execute=True)
        self.assertEqual(marker.read_text(), "original")
        for output in (self.root / "new", self.image_root / "new", self.base / "absent" / ".."):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare(self.manifest, self.image_root, output, execute=True)
        dangling = self.base / "dangling"
        dangling.symlink_to(self.base / "missing-target")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.manifest, self.image_root, dangling)

    def test_invalid_paths_identity_split_and_duplicates_rejected_before_copy(self):
        original_rows = [dict(row) for row in self.rows]
        changes = [
            {"relative_path": "../outside.jpg"},
            {"relative_path": "/valid/subject_0/study_0/image_0.jpg"},
            {"relative_path": "valid/subject_0/../study_0/image_0.jpg"},
            {"relative_path": "valid//subject_0/study_0/image_0.jpg"},
            {"relative_path": "valid/subject_0/study_0/./image_0.jpg"},
            {"relative_path": "valid\\subject_0\\study_0\\image_0.jpg"},
            {"subject_id": "subject_999"}, {"study_id": "study_999"},
            {"study_key": "subject_0/study_999"}, {"dicom_id": "another_file"},
            {"split": "test"}, {"image_id": original_rows[1]["image_id"]},
            {"file_sha256": "bad-digest"},
        ]
        for update in changes:
            with self.subTest(update=update):
                self.rows = [dict(row) for row in original_rows]
                self.rows[0].update(update)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    prepare(self.manifest, self.image_root, self.output, execute=True)
                self.assertFalse(self.output.exists())
        self.rows = [dict(row) for row in original_rows]
        self.rows.append({**original_rows[0], "image_id": "unique-duplicate-path"})
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.preview()

    def test_source_symlink_escape_rejected(self):
        path = self.root / self.rows[0]["relative_path"]
        path.unlink()
        outside = self.base / "outside.jpg"
        outside.write_bytes(b"outside")
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.preview()

    def test_missing_original_study_view_rejected(self):
        plan = self.preview()
        source = Path(plan["warmup"][0]["source_path"])
        (source.parent / "unlisted-original.jpg").write_bytes(b"another view")
        with self.assertRaisesRegex(ValueError, "incomplete original Study"):
            prepare(self.manifest, self.image_root, self.output, execute=True)
        self.assertFalse(self.output.exists())

    def test_impossible_exact_complete_study_subset_rejected(self):
        # Two complete Studies of the same patient provide exactly five warmups,
        # leaving no eligible measured patient.
        self.rows = self.rows[:5]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            self.preview()

    def test_copy_failure_cleans_new_output_without_mutating_sources(self):
        from unittest.mock import patch
        manifest_before = self.manifest.read_bytes()
        with patch("prepare_latency_inputs.shutil.copy2", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                prepare(self.manifest, self.image_root, self.output, execute=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.manifest.read_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main()
