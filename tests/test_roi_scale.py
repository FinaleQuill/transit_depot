"""Meaningful geometry/safety checks for deterministic classical ROI scaling."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from roi_scale import normalize_roi, TARGET_OCCUPANCY
from tight_crop import letterbox


class RoiScaleTests(unittest.TestCase):
    def test_same_body_at_different_scales_converges_to_fixed_occupancy(self):
        occupancies = []
        for bounds in ((100, 80, 500, 520), (70, 50, 530, 550)):
            gray = np.zeros((600, 600), dtype=np.uint8)
            x0, y0, x1, y1 = bounds
            gray[y0:y1, x0:x1] = 170
            # Low-intensity bilateral spaces must stay inside the support envelope.
            gray[y0+40:y1-50, x0+25:x0+125] = 8
            gray[y0+40:y1-50, x1-125:x1-25] = 8
            output, meta = normalize_roi(Image.fromarray(gray), 768)
            self.assertTrue(meta["applied"], meta)
            self.assertEqual(output.size, (768, 768))
            self.assertLessEqual(meta["occupancy_after"], TARGET_OCCUPANCY)
            self.assertLess(TARGET_OCCUPANCY-meta["occupancy_after"], .002)
            self.assertTrue(meta["protected_margin_retained"])
            occupancies.append((meta["occupancy_before"], meta["occupancy_after"]))
        self.assertGreater(abs(occupancies[0][0]-occupancies[1][0]), .09)
        self.assertLess(abs(occupancies[0][1]-occupancies[1][1]), .002)

    def test_source_edge_contact_shrinks_and_pads_without_new_cut(self):
        gray = np.full((301, 401), 170, dtype=np.uint8)
        image = Image.fromarray(gray)
        output, meta = normalize_roi(image, 768)
        # Uniform data are unreliable, so exact full-frame fallback is required.
        baseline, _ = letterbox(image, 768)
        self.assertFalse(meta["applied"])
        self.assertEqual(output.tobytes(), baseline.tobytes())
        gray[30:250, 70:150] = 8
        gray[30:250, 250:330] = 8
        output, meta = normalize_roi(Image.fromarray(gray), 768)
        self.assertTrue(meta["applied"], meta)
        self.assertEqual(set(meta["source_edge_contact"]), {"left", "top", "right", "bottom"})
        self.assertEqual(meta["source_removed_area_fraction"], 0.)
        self.assertLess(meta["effective_magnification"], 1.)
        self.assertEqual(meta["window_box"][2]-meta["window_box"][0],
                         meta["window_box"][3]-meta["window_box"][1])

    def test_unsafe_margin_or_unreliable_component_falls_back_exactly(self):
        examples = [np.zeros((173, 251), dtype=np.uint8)]
        tiny = np.zeros((240, 240), dtype=np.uint8)
        tiny[100:140, 100:140] = 180
        examples.append(tiny)
        margin = np.zeros((240, 240), dtype=np.uint8)
        margin[45:195, 45:195] = 180
        examples.append(margin)
        for gray in examples:
            output, meta = normalize_roi(Image.fromarray(gray), 768)
            baseline, _ = letterbox(Image.fromarray(gray), 768)
            self.assertFalse(meta["applied"], meta)
            self.assertEqual(output.tobytes(), baseline.tobytes())

    def test_replay_keeps_original_and_aspect_ratio(self):
        gray = np.zeros((501, 701), dtype=np.uint8)
        gray[50:451, 100:601] = 180
        gray[120:250, 180:260] = 8
        source = Image.fromarray(gray)
        before = source.tobytes()
        a, meta_a = normalize_roi(source, 640)
        b, meta_b = normalize_roi(source, 640)
        self.assertTrue(meta_a["applied"], meta_a)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(meta_a, meta_b)
        self.assertEqual(source.tobytes(), before)
        bx0, by0, bx1, by1 = meta_a["body_box"]
        ox0, oy0, ox1, oy1 = meta_a["output_body_box"]
        self.assertAlmostEqual((bx1-bx0)/(by1-by0), (ox1-ox0)/(oy1-oy0))
        self.assertFalse(meta_a["medical_pretrained_model"])
        self.assertFalse(meta_a["random_erasing"])


if __name__ == "__main__":
    unittest.main()
