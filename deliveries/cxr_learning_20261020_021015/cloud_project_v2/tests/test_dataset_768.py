# 【学习导航】检查原图读取、等比例Pad、SHA和不同数据接口的一致性。
# 本次只打包与静态解析，没有执行此测试文件。
"""CPU image-path and geometry checks; no model, GPU, or training operations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import dataset
from dataset import ChestXrayDataset, CachedSampleError, IMAGENET_MEAN, IMAGENET_STD
from dataset_overnight import OvernightImageDataset, make_overnight_loader


class Dataset768Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.outer_root = self.root / "raw"
        self.raw_root = self.outer_root / "cxr_first"
        self.relative = "valid/patient_0/study_0/image_0.png"
        self.path = self.raw_root / self.relative
        self.path.parent.mkdir(parents=True)
        Image.new("RGB", (1024, 512), (255, 255, 255)).save(self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.record = {
            "image_id": "valid:0", "split": "valid", "subject_id": "patient_0",
            "study_id": "study_0", "study_key": "patient_0/study_0", "dicom_id": "image_0",
            "relative_path": self.relative, "targets": tuple([1] + [0] * 9),
            "cache_relative_path": "cache_that_does_not_exist.png", "status": "ok",
            "training_eligible": "False", "file_sha256": self.digest,
        }
        self.manifest = self.root / "valid.csv"
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.record))
            writer.writeheader()
            writer.writerow({**self.record, "targets": json.dumps(self.record["targets"])})

    @staticmethod
    def pixels(tensor: torch.Tensor) -> np.ndarray:
        mean = np.asarray(IMAGENET_MEAN)[None, None, :]
        std = np.asarray(IMAGENET_STD)[None, None, :]
        return np.rint((tensor.numpy().transpose(1, 2, 0) * std + mean) * 255).astype(np.uint8)

    def test_letterbox_preserves_landscape_portrait_and_full_image(self) -> None:
        for shape, expected_box in [((1000, 500), (0, 192, 768, 576)),
                                    ((500, 1000), (192, 0, 576, 768))]:
            with self.subTest(shape=shape):
                source = Image.new("RGB", shape, "white")
                actual = dataset.letterbox_rgb(source, 768)
                self.assertEqual(actual.size, (768, 768))
                self.assertEqual(actual.getbbox(), expected_box)
                self.assertEqual(np.asarray(actual)[384, 384].tolist(), [255, 255, 255])

    def test_dataset_768_reads_native_with_missing_cache_and_keeps_padding(self) -> None:
        # The default native-root branch must work even when no 300 cache exists.
        with patch.object(dataset, "DEFAULT_RAW_ROOT", self.outer_root):
            images = ChestXrayDataset(self.manifest, split="valid", image_size=768)
        before = self.path.read_bytes()
        sample = images[0]
        self.assertEqual(images.raw_root, self.raw_root)
        self.assertEqual(images.paths, [self.path])
        self.assertEqual(tuple(sample["image"].shape), (3, 768, 768))
        self.assertEqual(sample["targets"].tolist(), [1.0] + [0.0] * 9)
        pixels = self.pixels(sample["image"])
        self.assertTrue((pixels[:192] == 0).all())
        self.assertTrue((pixels[192:576] == 255).all())
        self.assertTrue((pixels[576:] == 0).all())
        self.assertEqual(before, self.path.read_bytes())

    def test_overnight_and_generic_dataset_share_native_768_geometry(self) -> None:
        generic = ChestXrayDataset(self.manifest, split="valid", image_size=768,
                                   raw_root=self.outer_root)
        overnight = OvernightImageDataset([self.record], self.outer_root,
                                          training=False, image_size=768)
        torch.testing.assert_close(generic[0]["image"], overnight[0]["image"], rtol=0, atol=0)
        # Constructing the new server configuration must not start worker processes.
        loader = make_overnight_loader([self.record], self.outer_root, training=False,
                                       image_size=768, batch_size=1, num_workers=12,
                                       prefetch_factor=2, pin_memory=False)
        self.assertEqual(loader.num_workers, 12)
        self.assertEqual(loader.prefetch_factor, 2)

    def test_native_sha_change_is_rejected_without_cache_fallback(self) -> None:
        generic = ChestXrayDataset(self.manifest, split="valid", image_size=768,
                                   raw_root=self.outer_root)
        overnight = OvernightImageDataset([self.record], self.outer_root,
                                          training=False, image_size=768)
        Image.new("RGB", (1024, 512), "black").save(self.path)
        for images in (generic, overnight):
            with self.subTest(dataset=type(images).__name__):
                with self.assertRaisesRegex(CachedSampleError, "image_id=valid:0.*SHA"):
                    images[0]

    def test_direct_transforms_preserve_aspect_ratio(self) -> None:
        transform = dataset.build_transforms("valid", image_size=768)
        pixels = self.pixels(transform(image=np.full((100, 200, 3), 255, dtype=np.uint8))["image"])
        self.assertTrue((pixels[:192] == 0).all())
        self.assertTrue((pixels[192:576] == 255).all())
        self.assertTrue((pixels[576:] == 0).all())


if __name__ == "__main__":
    unittest.main()
