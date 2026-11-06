"""CPU-only loader checks using generated images and temporary manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tight_crop_ab_data import (ABImageDataset, make_ab_study_loader,
                                random_erasing_ab_u8)


def record(root: Path, split: str):
    relative = Path(split) / "subject_00000" / "study_00000" / "pic_00001.png"
    path = root / relative
    path.parent.mkdir(parents=True)
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    pixels[8:73, 14:86] = 190
    Image.fromarray(pixels, "RGB").save(path)
    return {"image_id": f"{split}:000001", "split": split,
            "subject_id": "subject_00000", "study_id": "study_00000",
            "dicom_id": "pic_00001", "study_key": "subject_00000/study_00000",
            "relative_path": str(relative), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "targets": (0.,) * 10 if split != "test" else None,
            "training_eligible": "True"}


class ABLoaderTests(unittest.TestCase):
    def test_spawn_train_study_and_valid_no_erasing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = record(root, "train"), record(root, "valid")
            for arm in ("A", "B"):
                with self.subTest(arm=arm):
                    loader = make_ab_study_loader([train], root, arm=arm, image_size=768,
                        num_workers=1, prefetch_factor=1, seed=48, pin_memory=False)
                    batch = next(iter(loader))
                    self.assertEqual(batch["image_ids"], [train["image_id"]])
                    self.assertEqual(tuple(batch["images"].shape), (1, 3, 768, 768))
                    self.assertEqual(batch["split"], "train")
                    del loader

                    with patch("tight_crop_ab_data.random_erasing_ab_u8",
                               side_effect=AssertionError("Validation applied erasing")):
                        dataset = ABImageDataset([valid], root, training=False,
                                                 arm=arm, seed=48)
                        item = dataset[0]
                    self.assertEqual(item["split"], "valid")
                    self.assertEqual(tuple(item["image"].shape), (3, 768, 768))

    def test_test_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = record(root, "test")
            with self.assertRaisesRegex(ValueError, "train or valid"):
                ABImageDataset([test], root, training=False, arm="A")

    def test_edge_bias_and_augmentation_rng_separation(self):
        pixels = np.full((64, 64, 3), 190, dtype=np.uint8)
        rng = np.random.default_rng(48)
        records = [random_erasing_ab_u8(pixels, (0, 0, 64, 64),
                   training=True, rng=rng, p=0.5)[1] for _ in range(200)]
        triggered = [row for row in records if row["applied"]]
        edges = [row for row in triggered if row["selection"] == "outer_20pct_content_edge_band"]
        self.assertGreater(len(triggered), 60)
        self.assertLess(len(triggered), 140)
        self.assertGreater(len(edges), 20)
        self.assertLess(len(edges), 80)
        for row in edges:
            x0, y0, x1, y1 = row["box"]
            self.assertTrue((x0 + x1) / 2 <= 12.8 or (x0 + x1) / 2 >= 51.2
                            or (y0 + y1) / 2 <= 12.8 or (y0 + y1) / 2 >= 51.2)
        unchanged, no_erase = random_erasing_ab_u8(pixels, (0, 0, 64, 64), training=False)
        np.testing.assert_array_equal(unchanged, pixels)
        self.assertFalse(no_erase["applied"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = record(root, "train")
            a = ABImageDataset([train], root, training=True, arm="A", seed=48)
            b = ABImageDataset([train], root, training=True, arm="B", seed=48)
            a[0], b[0]
            self.assertEqual(a.geometry.py_random.getstate(), b.geometry.py_random.getstate())
            self.assertEqual(a.geometry.random_generator.bit_generator.state,
                             b.geometry.random_generator.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
