"""Classical, deterministic body-envelope scale normalization.

This is an image-support heuristic, not lung segmentation.  A reliable envelope
is mapped to 85% of the output long edge using one isotropic scale.  A virtual
square frame supplies padding; the detector's protective margin must remain
inside it.  Unreliable or unsafe proposals use the original full-frame letterbox.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageOps

from tight_crop import TightCropConfig, detect_tight_crop, letterbox


VERSION = "body_envelope_fixed_occupancy_v1"
TARGET_OCCUPANCY = .85
DETECTOR_CONFIG = TightCropConfig(margin_fraction=.05, margin_min_original_px=32)


def normalize_roi(image: Image.Image, size: int) -> tuple[Image.Image, dict]:
    """Return an RGB square and JSON-serializable geometric diagnostics.

    ``body_box`` is a conservative image-support envelope and does not establish
    that anatomy is complete in the source.  Source-edge contact is recorded.
    The original image is never modified, stretched, randomly erased, or fitted.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if type(size) is not int or size < 8:
        raise ValueError("size must be an integer of at least 8")
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    if min(width, height) < 1:
        raise ValueError("source image must be nonempty")
    metadata = {
        "version": VERSION, "method": "classical_body_envelope_fixed_occupancy",
        "target_occupancy": TARGET_OCCUPANCY, "output_size": size,
        "source_size": [width, height], "medical_pretrained_model": False,
        "medical_segmentation": False, "random_erasing": False,
        "single_isotropic_scale": True, "source_clipping_repaired": False,
        "text_or_device_removal": "not targeted or guaranteed",
        "resampling": "one Pillow LANCZOS resize after integer virtual-frame crop",
        "padding_value_u8": 0,
    }

    def fallback(reason: str) -> tuple[Image.Image, dict]:
        output, base = letterbox(source, size)
        metadata.update(applied=False, reason=reason,
                        window_box=[0, 0, width, height],
                        scale=base["scale"], effective_magnification=1.,
                        protected_margin_retained=True,
                        source_removed_area_fraction=0.,
                        source_edge_contact=metadata.get("source_edge_contact", []),
                        output_content_box=list(base["content_box"]))
        metadata["occupancy_after"] = metadata.get("occupancy_before")
        return output, metadata

    if min(width, height) < 8:
        return fallback("source_too_small_for_detector")
    gray = np.asarray(source.convert("L"))
    detected = detect_tight_crop(gray, DETECTOR_CONFIG)
    metadata["detector"] = detected.diagnostics
    metadata["body_box"] = list(detected.body_box)
    metadata["protected_box"] = list(detected.box)
    metadata["source_edge_contact"] = detected.diagnostics.get("body_clipped_at_source_edges", [])
    if detected.diagnostics["reason"] != "body_envelope_candidate":
        return fallback("detector_" + detected.diagnostics["reason"])

    bx0, by0, bx1, by1 = detected.body_box
    body_long = max(bx1-bx0, by1-by0)
    metadata["occupancy_before"] = body_long/max(width, height)
    # The virtual frame, not the crop's longest side, determines a fixed scale.
    # Ceiling guarantees that rounding never increases occupancy past the target.
    side = math.ceil(body_long/TARGET_OCCUPANCY)
    left = math.floor((bx0+bx1-side)/2)
    top = math.floor((by0+by1-side)/2)
    window = (left, top, left+side, top+side)
    px0, py0, px1, py1 = detected.box
    if not (left <= px0 < px1 <= left+side and top <= py0 < py1 <= top+side):
        return fallback("protected_margin_outside_normalized_frame")

    # The existing detector refuses broad support outside its protective box.
    # Its full mask must also fit this virtual frame after conservative mapping.
    ys, xs = np.nonzero(detected.mask)
    if len(xs):
        ah, aw = detected.mask.shape
        support = (math.floor(xs.min()*width/aw), math.floor(ys.min()*height/ah),
                   math.ceil((xs.max()+1)*width/aw), math.ceil((ys.max()+1)*height/ah))
        if not (left <= support[0] and top <= support[1] and
                support[2] <= left+side and support[3] <= top+side):
            return fallback("support_outside_normalized_frame")

    retained = (max(0, left), max(0, top), min(width, left+side), min(height, top+side))
    retained_area = max(0, retained[2]-retained[0])*max(0, retained[3]-retained[1])
    output = source.crop(window).resize((size, size), Image.Resampling.LANCZOS)
    scale = size/side
    metadata.update(
        applied=True, reason="fixed_body_envelope_occupancy", window_box=list(window),
        scale=scale, effective_magnification=max(width, height)/side,
        occupancy_after=body_long/side, protected_margin_retained=True,
        source_removed_area_fraction=1-retained_area/(width*height),
        output_body_box=[(bx0-left)*scale, (by0-top)*scale,
                         (bx1-left)*scale, (by1-top)*scale],
        output_content_box=[(retained[0]-left)*scale, (retained[1]-top)*scale,
                            (retained[2]-left)*scale, (retained[3]-top)*scale],
        virtual_padding_source_pixels={"left": max(0, -left), "top": max(0, -top),
                                       "right": max(0, left+side-width),
                                       "bottom": max(0, top+side-height)},
    )
    return output, metadata
