"""Classical body-envelope crop that may magnify, but never shrink, an image.

The detector is an image-support heuristic rather than lung segmentation.  A
    proposal is used only when it is reliable, preserves the configured geometry
    envelope, and improves effective input scale within the magnification bounds.
Otherwise the exact full-frame letterbox is returned.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from tight_crop import detect_tight_crop, letterbox


VERSION = "body_envelope_magnify_only_v2"
MARGIN_FRACTION = .02
MIN_EFFECTIVE_MAGNIFICATION = 1.01
MAX_EFFECTIVE_MAGNIFICATION = 1.25
MIN_CROP_ASPECT_RELATIVE_TO_SOURCE = 1.0
MIN_CROP_SHORT_LONG_RATIO = .55


def normalize_roi(image: Image.Image, size: int) -> tuple[Image.Image, dict]:
    """Return an RGB square and JSON-serializable geometry diagnostics.

    Uses the existing classical OpenCV detector, then recomputes a conservative
    crop around its body envelope with about 2% margin per body-box axis. The
    source aspect ratio is kept. If the proposal does not magnify
    relative to full-frame letterboxing or violates a safety guard, it falls
    back pixel-exactly to that full-frame letterbox.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if type(size) is not int or size < 8:
        raise ValueError("size must be an integer of at least 8")
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    if min(width, height) < 1:
        raise ValueError("source image must be nonempty")
    base, base_meta = letterbox(source, size)
    base_scale = float(base_meta["scale"])
    metadata = {
        "version": VERSION, "method": "classical_body_envelope_crop_magnify_only",
        "output_size": size, "source_size": [width, height],
        "medical_pretrained_model": False, "medical_segmentation": False,
        "random_erasing": False, "single_isotropic_scale": True,
        "source_clipping_repaired": False,
        "text_or_device_removal": "not targeted or guaranteed",
        "resampling": "one Pillow LANCZOS resize after conservative crop",
        "padding_value_u8": 0, "base_scale": base_scale,
        "crop_policy": {"margin_fraction_of_body_box_axes": MARGIN_FRACTION,
                        "minimum_margin_original_px": 0,
                        "maximum_effective_magnification": MAX_EFFECTIVE_MAGNIFICATION,
                        "minimum_effective_magnification": MIN_EFFECTIVE_MAGNIFICATION,
                        "minimum_crop_aspect_relative_to_source": MIN_CROP_ASPECT_RELATIVE_TO_SOURCE,
                        "minimum_crop_short_long_ratio": MIN_CROP_SHORT_LONG_RATIO,
                        "reject_source_edge_clipped_body": True,
                        "detector_quality_and_removed_area_gates": "tight_crop.detect_tight_crop unchanged"},
    }

    def fallback(reason: str) -> tuple[Image.Image, dict]:
        proposed_box = metadata.get("proposed_crop_box")
        proposed_magnification = metadata.get("proposed_effective_magnification")
        metadata.update(applied=False, reason=reason, crop_box=[0, 0, width, height],
                        scale=base_scale, effective_magnification=1.,
                        crop_aspect_ratio=width/height,
                        source_removed_area_fraction=0., retained_area_fraction=1.,
                        protected_margin_retained=True,
                        output_content_box=list(base_meta["content_box"]))
        if proposed_box is not None:
            metadata["proposed_crop_box"] = proposed_box
        if proposed_magnification is not None:
            metadata["proposed_effective_magnification"] = proposed_magnification
        return base, metadata

    if min(width, height) < 8:
        return fallback("source_too_small_for_detector")
    gray = np.asarray(source.convert("L"), dtype=np.uint8)
    detected = detect_tight_crop(gray)
    metadata["detector"] = detected.diagnostics
    metadata["body_box"] = list(detected.body_box)
    edge_contact = detected.diagnostics.get("body_clipped_at_source_edges", [])
    metadata["source_edge_contact"] = edge_contact
    if detected.diagnostics["reason"] != "body_envelope_candidate":
        return fallback("detector_" + detected.diagnostics["reason"])
    if edge_contact:
        return fallback("body_envelope_clipped_at_source_edge")

    bx0, by0, bx1, by1 = detected.body_box
    margin_x = int(np.ceil(MARGIN_FRACTION*(bx1-bx0)))
    margin_y = int(np.ceil(MARGIN_FRACTION*(by1-by0)))
    box = (max(0, bx0-margin_x), max(0, by0-margin_y),
           min(width, bx1+margin_x), min(height, by1+margin_y))
    x0, y0, x1, y1 = box
    crop_width, crop_height = x1-x0, y1-y0
    crop_aspect = crop_width/crop_height
    crop_short_long_ratio = min(crop_width, crop_height)/max(crop_width, crop_height)
    source_short_long_ratio = min(width, height)/max(width, height)
    retained_area = crop_width*crop_height/(width*height)
    side_crops = {"left": x0/width, "top": y0/height,
                  "right": (width-x1)/width, "bottom": (height-y1)/height}
    crop_scale = size/max(crop_width, crop_height)
    magnification = crop_scale/base_scale
    metadata.update(proposed_crop_box=list(box), margin_original_px=[margin_x, margin_y],
                    proposed_crop_aspect_ratio=crop_aspect,
                    proposed_retained_area_fraction=retained_area,
                    proposed_source_removed_area_fraction=1-retained_area,
                    removed_fraction_by_side=side_crops,
                    crop_short_long_ratio=crop_short_long_ratio,
                    source_short_long_ratio=source_short_long_ratio,
                    proposed_effective_magnification=magnification,
                    source_clipping_repaired=False)

    checks = (
        (crop_width > 0 and crop_height > 0, "invalid_crop_box"),
        (crop_short_long_ratio >= source_short_long_ratio, "crop_aspect_worsens_padding"),
        (crop_short_long_ratio >= MIN_CROP_SHORT_LONG_RATIO, "crop_aspect_too_extreme"),
        (magnification <= MAX_EFFECTIVE_MAGNIFICATION, "maximum_magnification_limit"),
        (magnification >= MIN_EFFECTIVE_MAGNIFICATION, "minimum_magnification_limit"),
        (crop_scale >= base_scale*(1-1e-12), "would_shrink_full_frame"),
    )
    for accepted, reason in checks:
        if not accepted:
            return fallback(reason)

    output, resize_meta = letterbox(source.crop(box), size)
    metadata.update(applied=True, reason="body_envelope_magnify_only", crop_box=list(box),
                    crop_aspect_ratio=crop_aspect, retained_area_fraction=retained_area,
                    source_removed_area_fraction=1-retained_area,
                    effective_magnification=magnification,
                    scale=resize_meta["scale"], protected_margin_retained=True,
                    output_content_box=list(resize_meta["content_box"]))
    return output, metadata
