"""Synthetic-only checks for the experimental tight-crop preview."""
from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from tight_crop import TightCropConfig, detect_tight_crop, letterbox, random_erasing_u8


class TightCropTests(unittest.TestCase):
    def test_otsu_uses_a_real_gray_level_split(self) -> None:
        # Repeating every intensity makes the histogram large; the old broken
        # formula was driven toward 254 by total pixel count.
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        result = detect_tight_crop(gray)
        reference, _ = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        self.assertEqual(result.diagnostics["otsu_threshold"], reference)
        self.assertGreater(reference, 110)
        self.assertLess(reference, 145)

    def test_wide_black_band_and_dark_lungs_remain_in_crop(self) -> None:
        gray = np.zeros((320, 420), dtype=np.uint8)
        gray[:185, 30:390] = 180
        gray[26:151, 77:155] = 8
        gray[26:151, 263:341] = 8
        # A low-contrast side of the body extends outside the bright Otsu seed.
        gray[42:160, 19:30] = 16
        result = detect_tight_crop(gray)
        self.assertTrue(result.diagnostics["applied"])
        x0, y0, x1, y1 = result.box
        self.assertEqual(y0, 0)
        self.assertLess(x0, 19)
        self.assertGreaterEqual(x1, 390)
        self.assertGreaterEqual(y1, 185)
        self.assertLess(y1, 240)
        self.assertEqual(gray[y0:y1, x0:x1][100-y0, 100-x0], 8)
        self.assertEqual(gray[y0:y1, x0:x1][100-y0, 300-x0], 8)
        self.assertIn("top", result.diagnostics["body_clipped_at_source_edges"])

    def test_blank_and_tiny_object_fall_back_to_exact_full_frame(self) -> None:
        for gray in (np.zeros((173, 251), dtype=np.uint8),
                     np.full((173, 251), 70, dtype=np.uint8)):
            with self.subTest(kind="uniform", value=int(gray[0, 0])):
                result = detect_tight_crop(gray)
                self.assertEqual(result.box, (0, 0, 251, 173))
                self.assertFalse(result.diagnostics["applied"])
        gray = np.zeros((173, 251), dtype=np.uint8)
        gray[75:95, 115:135] = 200
        result = detect_tight_crop(gray)
        self.assertEqual(result.box, (0, 0, 251, 173))
        self.assertFalse(result.diagnostics["applied"])

    def test_non_square_analysis_maps_to_half_open_original_and_margin(self) -> None:
        height, width = 301, 401
        gray = np.zeros((height, width), dtype=np.uint8)
        gray[31:221, 43:337] = 180
        config = TightCropConfig(analysis_long_edge=127, margin_min_original_px=0)
        result = detect_tight_crop(gray, config)
        self.assertTrue(result.diagnostics["applied"])
        self.assertEqual(result.diagnostics["analysis_size"], [127, 95])
        bx0, by0, bx1, by1 = result.body_box
        x0, y0, x1, y1 = result.box
        self.assertLessEqual(bx0, 43)
        self.assertLessEqual(by0, 31)
        self.assertGreaterEqual(bx1, 337)
        self.assertGreaterEqual(by1, 221)
        self.assertTrue(0 <= x0 < bx0 < bx1 < x1 <= width)
        self.assertTrue(0 <= y0 < by0 < by1 < y1 <= height)
        self.assertGreaterEqual(bx0-x0, .05*(bx1-bx0)-1)
        self.assertGreaterEqual(x1-bx1, .05*(bx1-bx0)-1)
        self.assertGreaterEqual(by0-y0, .05*(by1-by0)-1)
        self.assertGreaterEqual(y1-by1, .05*(by1-by0)-1)
        self.assertEqual(gray[30, 43], 0)
        self.assertEqual(gray[220, 336], 180)

    def test_letterbox_keeps_isotropic_scale_and_black_padding(self) -> None:
        for width, height in ((301, 101), (101, 301), (509, 307)):
            with self.subTest(source_size=(width, height)):
                result, meta = letterbox(Image.new("L", (width, height), 255))
                self.assertEqual(result.size, (768, 768))
                nw, nh = meta["resized_size"]
                self.assertLessEqual(abs(nw-width*meta["scale"]), .5)
                self.assertLessEqual(abs(nh-height*meta["scale"]), .5)
                self.assertLessEqual(abs(nw/width-nh/height),
                                     .5/min(width, height))
                self.assertEqual(result.getbbox(), meta["content_box"])
                pixels = np.asarray(result)
                x0, y0, x1, y1 = meta["content_box"]
                self.assertTrue((pixels[y0:y1, x0:x1] == 255).all())
                self.assertEqual(np.count_nonzero(pixels[:y0]), 0)
                self.assertEqual(np.count_nonzero(pixels[y1:]), 0)

    def test_random_erasing_is_train_only_bounded_and_replayable(self) -> None:
        image = np.full((768, 768, 3), 149, dtype=np.uint8)
        content = (84, 0, 684, 768)
        original = image.copy()
        evaluated, eval_record = random_erasing_u8(image, content, training=False)
        self.assertFalse(eval_record["applied"])
        self.assertTrue(np.array_equal(evaluated, image))
        self.assertIsNot(evaluated, image)
        triggers = 0
        for seed in range(100):
            result, record = random_erasing_u8(
                image, content, training=True, rng=np.random.default_rng(seed), p=.5,
            )
            replay, replay_record = random_erasing_u8(
                image, content, training=True, rng=np.random.default_rng(seed), p=.5,
            )
            self.assertEqual(record, replay_record)
            self.assertTrue(np.array_equal(result, replay))
            self.assertTrue(np.array_equal(image, original))
            self.assertTrue((result[:, :84] == 149).all())
            self.assertTrue((result[:, 684:] == 149).all())
            if record["applied"]:
                triggers += 1
                x0, y0, x1, y1 = record["box"]
                self.assertTrue(84 <= x0 < x1 <= 684)
                self.assertTrue(0 <= y0 < y1 <= 768)
                self.assertLessEqual((x1-x0)*(y1-y0), .02*600*768)
                self.assertTrue((result[y0:y1, x0:x1] == 0).all())
                self.assertEqual(np.count_nonzero(result != image),
                                 (x1-x0)*(y1-y0)*3)
            else:
                self.assertTrue(np.array_equal(result, image))
        self.assertGreater(triggers, 30)
        self.assertLess(triggers, 70)


if __name__ == "__main__":
    unittest.main()
