"""Exact frozen-feature inputs and immutable split/image identity guards."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from frozen_head_cache import ExactFeatureImages
from deploy_kd import MEAN, STD, _letterbox_image
from inference_inputs import normalize_pixels


class FrozenFeatureInputTests(unittest.TestCase):
    def test_gray_and_rgb_pixels_match_independent_original_float32_then_half_path(self):
        for mode in ['L', 'RGB']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                file = root / 'train/subject/study/image.png'
                file.parent.mkdir(parents=True)
                shape = (29, 43) if mode == 'L' else (29, 43, 3)
                Image.fromarray(np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8), mode).save(file)
                records = [{'split': 'train', 'relative_path': 'train/subject/study/image.png',
                    'file_sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'image_id': 'image'}]
                index, actual = ExactFeatureImages(records, root)[0]
                with Image.open(file) as original:
                    expected = normalize_pixels(np.asarray(_letterbox_image(original, 768, mode)), MEAN, STD).astype(np.float16)
                self.assertEqual(index, 0)
                self.assertEqual(actual.dtype.__str__(), 'torch.float16')
                np.testing.assert_array_equal(actual.numpy().view(np.uint16), expected.view(np.uint16))

    def test_test_split_and_changed_original_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'train or valid'):
                ExactFeatureImages([{'split': 'test'}], root)
            file = root / 'train/s/t/image.png'
            file.parent.mkdir(parents=True)
            Image.new('L', (8, 8), 12).save(file)
            records = [{'split': 'train', 'relative_path': 'train/s/t/image.png',
                        'file_sha256': 'a'*64, 'image_id': 'image'}]
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                ExactFeatureImages(records, root)[0]


if __name__ == '__main__':
    unittest.main()
