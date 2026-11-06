"""Native full-box resampling must retain Pillow's uint8 intermediate rounding."""
import concurrent.futures
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import fast_lanczos


@unittest.skipUnless(fast_lanczos.available(), "Build native_lanczos for pinned Pillow 10.4 first")
class NativeLanczosTests(unittest.TestCase):
    def assert_parity(self, pixels, target, threads):
        image = Image.fromarray(pixels)
        before = image.tobytes()
        expected = image.resize(target, Image.Resampling.LANCZOS)
        actual = fast_lanczos.resize(image, target, threads=threads)
        self.assertEqual(actual.mode, "L")
        self.assertEqual(actual.size, target)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(image.tobytes(), before)

    def test_integer_boundaries_and_single_axis_resize(self):
        rng = np.random.default_rng(43)
        for shape in ((1, 1), (1, 19), (17, 1), (7, 13), (47, 31), (123, 157)):
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            for target in ((1, 1), (3, 11), (shape[1], shape[0]), (shape[1], 29), (27, shape[0])):
                for threads in (1, 2, 4, 8):
                    with self.subTest(shape=shape, target=target, threads=threads):
                        self.assert_parity(pixels, target, threads)

    def test_constants_edges_checkerboard_and_large_odd_sizes(self):
        y, x = np.indices((1025, 1537))
        fixtures = [np.full((1025, 1537), value, dtype=np.uint8) for value in (0, 1, 127, 254, 255)]
        fixtures.extend([(((x + y) % 2) * 255).astype(np.uint8),
                         ((x > 768) * 255).astype(np.uint8),
                         ((x * 79 + y * 53 + (x * y) % 251) % 256).astype(np.uint8)])
        for index, pixels in enumerate(fixtures):
            for target in ((768, 512), (511, 769), (13, 7)):
                for threads in (1, 8):
                    with self.subTest(fixture=index, target=target, threads=threads):
                        self.assert_parity(pixels, target, threads)

    def test_simultaneous_callers_do_not_mix_images(self):
        rng = np.random.default_rng(44)
        images = [Image.fromarray(rng.integers(0, 256, (513, 769), dtype=np.uint8)) for _ in range(4)]
        expected = [image.resize((177, 119), Image.Resampling.LANCZOS).tobytes() for image in images]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda image: fast_lanczos.resize(image, (177, 119)).tobytes(), images))
        self.assertEqual(actual, expected)

    def test_invalid_threads_fail_before_native_access(self):
        image = Image.new("L", (3, 3))
        for threads in (0, 9, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fast_lanczos.resize(image, (2, 2), threads=threads)

    def test_generic_fallback_and_cpu_dispatch_match(self):
        rng = np.random.default_rng(45)
        for shape in ((31, 47), (1025, 1537)):
            image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
            target = (177, 119)
            generic = Image.new("L", target)
            fast_lanczos._core.resize_l(image.getim(), generic.getim(), 8, False)
            expected = image.resize(target, Image.Resampling.LANCZOS).tobytes()
            self.assertEqual(generic.tobytes(), expected)
            self.assertEqual(fast_lanczos.resize(image, target).tobytes(), expected)
        self.assertIn(fast_lanczos.implementation(), ("generic", "avx2"))


class FallbackTests(unittest.TestCase):
    def test_missing_extension_and_rgb_keep_original_pillow(self):
        rng = np.random.default_rng(43)
        for shape in ((19, 31), (19, 31, 3)):
            image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
            expected = image.resize((11, 7), Image.Resampling.LANCZOS).tobytes()
            with patch.object(fast_lanczos, "_core", None):
                self.assertEqual(fast_lanczos.resize(image, (11, 7)).tobytes(), expected)
            self.assertEqual(fast_lanczos.resize(image, (11, 7)).tobytes(), expected)

    def test_pillow_abi_version_mismatch_uses_pillow(self):
        image = Image.new("L", (13, 19), 71)
        with patch.object(fast_lanczos, "PILLOW_VERSION", "future"):
            self.assertFalse(fast_lanczos.available())
            self.assertEqual(fast_lanczos.resize(image, (5, 7)).tobytes(),
                             image.resize((5, 7), Image.Resampling.LANCZOS).tobytes())


if __name__ == "__main__":
    unittest.main()
