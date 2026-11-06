"""Filesystem-only checks; no model, dataset, training or inference imports."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import manage_living_compliance as living


class LivingPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "closeout"
        self.package = self.base / "package"
        self.package.mkdir(parents=True)
        payload = {
            living.WORKBOOK: b"unchanged-workbook-bytes",
            "A1_A5_FREEZE_RECORD.md": b"Frozen A1-A5",
            "licenses/license.txt": b"frozen license",
            "README.md": b"Historical package\n",
            "PACKAGE_STRUCTURE.md": b"Historical structure\n",
            "code/train_teacher.py": b"# historical source\n",
            "evidence/XLSX_frozen_cell_comparison.json": json.dumps({"cell_count": 62, "all_equal": True}).encode(),
        }
        for name, content in payload.items():
            p = self.package / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        living.refresh_manifest(self.package)
        self.archive = self.base / "review.zip"
        with zipfile.ZipFile(self.archive, "w") as z:
            for p in sorted(self.package.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(self.package).as_posix())
        info = living.file_info(self.archive)
        living.atomic_json(self.base / "DELIVERY_RECEIPT.json", {
            "zip_path": str(self.archive), "zip_sha256": info["sha256"], "zip_bytes": info["bytes"],
            "matrix": {"rows": 29, "satisfied": 19, "pending": 7, "not_applicable": 3},
        })
        self.snapshot = self.base / "snapshots" / living.SNAPSHOT_NAME

    def tearDown(self):
        for p in self.root.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
        self.tmp.cleanup()

    def test_snapshot_is_independent_readonly_and_repeatable(self):
        first = living.create_snapshot(self.base)
        before = {p.name: p.read_bytes() for p in self.snapshot.iterdir()}
        second = living.create_snapshot(self.base)
        self.assertEqual(first, second)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.snapshot.iterdir()})
        self.assertTrue(first["independent_copy"])
        self.assertTrue(first["read_only_mode_bits"])
        self.assertEqual(first["payload_files_verified"], 7)

    def test_original_receipt_drift_fails_closed(self):
        living.create_snapshot(self.base)
        self.archive.write_bytes(b"modified original")
        with self.assertRaisesRegex(ValueError, "receipt"):
            living.create_snapshot(self.base)

    def test_snapshot_write_permission_and_tampering_are_detected(self):
        living.create_snapshot(self.base)
        copy = self.snapshot / "review.zip"
        copy.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "writable"):
            living.create_snapshot(self.base)
        copy.write_bytes(b"tampered")
        copy.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            living.create_snapshot(self.base)

    def test_partial_snapshot_is_never_replaced(self):
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "partial.txt").write_text("keep for inspection")
        with self.assertRaisesRegex(ValueError, "Partial"):
            living.create_snapshot(self.base)
        self.assertEqual((self.snapshot / "partial.txt").read_text(), "keep for inspection")

    def test_upgrade_keeps_history_and_frozen_material(self):
        source = self.root / "train_teacher.py"
        source.write_text("# source with compliance lifecycle\n")
        archive_before = self.archive.read_bytes()
        report = living.update_package(self.base, self.root, [source.name])
        self.assertFalse(report["final_members_frozen"])
        self.assertEqual(self.archive.read_bytes(), archive_before)
        self.assertEqual(report["removed_paths"], [])
        self.assertIn("code/train_teacher.py", report["changed_paths"])
        state = json.loads((self.package / "LIVING_PACKAGE.json").read_text())
        old = state["source_sync"][source.name]["pre_living_file"]
        living.update_package(self.base, self.root, [source.name])
        state = json.loads((self.package / "LIVING_PACKAGE.json").read_text())
        self.assertEqual(old, state["source_sync"][source.name]["pre_living_file"])
        self.assertEqual(living.verify_manifest(self.package), report["living_payload_files"])

    def test_frozen_workbook_change_blocks_package_update(self):
        (self.package / living.WORKBOOK).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen material"):
            living.update_package(self.base, self.root, [])
        self.assertFalse((self.package / "LIVING_PACKAGE.json").exists())

    def test_final_member_freeze_is_not_allowed(self):
        living.update_package(self.base, self.root, [])
        state_path = self.package / "LIVING_PACKAGE.json"
        state = json.loads(state_path.read_text())
        state["final_members_frozen"] = True
        living.atomic_json(state_path, state)
        with self.assertRaisesRegex(ValueError, "cannot freeze"):
            living.update_package(self.base, self.root, [])

    def test_unsafe_sync_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe relative"):
            living.update_package(self.base, self.root, ["../outside.py"])


if __name__ == "__main__":
    unittest.main()
