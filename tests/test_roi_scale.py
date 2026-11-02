"""Deterministic crop, magnification and full-frame fallback contracts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from roi_scale import VERSION, normalize_roi
from tight_crop import letterbox


def proposal(body_box, *, reason="body_envelope_candidate", edge_contact=()):
    return SimpleNamespace(body_box=body_box, diagnostics={
        "reason": reason, "body_clipped_at_source_edges": list(edge_contact),
        "medical_segmentation": False, "review_required": True,
    })


class RoiScaleTests(unittest.TestCase):
    def test_safe_body_crop_magnifies_without_distortion_and_is_replayable(self):
        source = Image.fromarray(np.full((600, 800), 100, dtype=np.uint8))
        source.putpixel((100, 100), 255)
        before = source.tobytes()
        detected = proposal((90, 35, 710, 565))
        with patch("roi_scale.detect_tight_crop", return_value=detected):
            first, meta = normalize_roi(source, 768)
            second, repeated = normalize_roi(source, 768)
        self.assertTrue(meta["applied"], meta)
        self.assertEqual(meta["version"], VERSION)
        self.assertEqual(first.size, (768, 768))
        self.assertTrue(meta["single_isotropic_scale"])
        self.assertGreaterEqual(meta["effective_magnification"], 1.01)
        self.assertLessEqual(meta["effective_magnification"], 1.25)
        self.assertGreaterEqual(meta["crop_short_long_ratio"], meta["source_short_long_ratio"])
        self.assertEqual(meta, repeated)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(source.tobytes(), before)
        self.assertFalse(meta["medical_pretrained_model"])
        self.assertFalse(meta["random_erasing"])
        self.assertEqual(meta["margin_original_px"], [13, 11])
        # Crop retains aspect and pads only the short axis.
        self.assertEqual(meta["output_content_box"][0], 0)
        self.assertEqual(meta["output_content_box"][2], 768)
        self.assertLess(meta["output_content_box"][3]-meta["output_content_box"][1], 768)

    def test_edge_contact_is_preserved_and_detector_failure_returns_exact_baseline(self):
        source = Image.fromarray(np.full((600, 800), 120, dtype=np.uint8))
        baseline, _ = letterbox(source, 768)
        cases = [
            proposal((0, 45, 650, 555), edge_contact=("left",)),
            proposal((20, 20, 780, 580), reason="broad_support_outside_candidate"),
        ]
        with patch("roi_scale.detect_tight_crop", side_effect=cases):
            edge, edge_meta = normalize_roi(source, 768)
            rejected, rejected_meta = normalize_roi(source, 768)
        self.assertTrue(edge_meta["applied"], edge_meta)
        self.assertEqual(edge_meta["reason"], "body_envelope_magnify_only")
        self.assertEqual(edge_meta["source_edge_contact"], ["left"])
        self.assertEqual(edge_meta["crop_box"][0], 0)
        self.assertTrue(edge_meta["crop_contains_detector_envelope"])
        self.assertFalse(edge_meta["crop_policy"]["reject_source_edge_clipped_body"])
        self.assertGreater(edge_meta["effective_magnification"], 1.)
        self.assertFalse(rejected_meta["applied"])
        self.assertEqual(rejected_meta["reason"], "detector_broad_support_outside_candidate")
        self.assertEqual(rejected.tobytes(), baseline.tobytes())
        self.assertNotEqual(edge.tobytes(), baseline.tobytes())

    def test_padding_aspect_extreme_crop_and_magnification_guards(self):
        source = Image.fromarray(np.full((600, 800), 120, dtype=np.uint8))
        baseline, _ = letterbox(source, 768)
        # This crop would worsen an already wide source's square padding.
        aspect = proposal((250, 40, 550, 560))
        # This crop exceeds the registered 1.25x magnification cap.
        magnification = proposal((200, 100, 600, 500))
        with patch("roi_scale.detect_tight_crop", side_effect=[aspect, magnification]):
            aspect_out, aspect_meta = normalize_roi(source, 768)
            magnification_out, magnification_meta = normalize_roi(source, 768)
        self.assertEqual(aspect_meta["reason"], "crop_aspect_worsens_padding")
        self.assertEqual(magnification_meta["reason"], "maximum_magnification_limit")
        self.assertEqual(aspect_out.tobytes(), baseline.tobytes())
        self.assertEqual(magnification_out.tobytes(), baseline.tobytes())
        self.assertGreater(magnification_meta["proposed_effective_magnification"], 1.25)

    def test_no_effective_gain_never_changes_original_letterbox(self):
        source = Image.fromarray(np.full((600, 800), 120, dtype=np.uint8))
        baseline, _ = letterbox(source, 768)
        # The crop remains almost the entire long axis and is rejected below 1.01x.
        detected = proposal((5, 5, 795, 595))
        with patch("roi_scale.detect_tight_crop", return_value=detected):
            output, meta = normalize_roi(source, 768)
        self.assertEqual(meta["reason"], "minimum_magnification_limit")
        self.assertFalse(meta["applied"])
        self.assertEqual(output.tobytes(), baseline.tobytes())

    def test_actual_classical_detector_fallback_is_safe(self):
        source = Image.fromarray(np.zeros((173, 251), dtype=np.uint8))
        baseline, _ = letterbox(source, 768)
        output, meta = normalize_roi(source, 768)
        self.assertFalse(meta["applied"])
        self.assertTrue(meta["reason"].startswith("detector_"))
        self.assertEqual(output.tobytes(), baseline.tobytes())


if __name__ == "__main__":
    unittest.main()
