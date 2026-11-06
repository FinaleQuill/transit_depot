"""Experimental draft-pixel body-envelope detector; not production-integrated.

The JPEG draft image changes analysis pixels relative to the frozen full-decode
detector. Geometry and conservative decisions after the 512 analysis image are
kept in the same order as tight_crop.detect_tight_crop.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import cv2
import numpy as np
from PIL import Image

from tight_crop import TightCropConfig, TightCropResult, _central_component, _components, _external_fill


def analysis_from_draft(image: Image.Image, config: TightCropConfig | None = None) -> np.ndarray:
    config = config or TightCropConfig()
    gray = np.asarray(image if image.mode == "L" else image.convert("L"))
    h, w = gray.shape
    s = min(1., config.analysis_long_edge / max(w, h))
    aw, ah = max(1, round(w*s)), max(1, round(h*s))
    return cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)


def detect_prepared(analysis_gray: np.ndarray, original_size: tuple[int, int],
                    config: TightCropConfig | None = None,
                    *, original_range: int | None = None) -> TightCropResult:
    """Detect from prepared 512 pixels; original_range enables full-path replay."""
    config = config or TightCropConfig()
    if analysis_gray.ndim != 2 or analysis_gray.dtype != np.uint8 or min(analysis_gray.shape) < 8:
        raise ValueError("Expected uint8 grayscale analysis image")
    if not .05 <= config.margin_fraction <= .10:
        raise ValueError("Prototype margin must be 5%-10% per side")
    W, H = map(int, original_size)
    if W < 8 or H < 8:
        raise ValueError("Invalid original size")
    small = analysis_gray
    ah, aw = small.shape
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    otsu, high = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(3, round(min(aw, ah) * config.closing_fraction)) | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, open_kernel)
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, close_kernel)
    seed = _central_component(high)
    diagnostics = {
        "config": asdict(config), "opencv_version": cv2.__version__,
        "original_size": [W, H], "analysis_size": [aw, ah],
        "decode": "caller-prepared analysis; draft pixels may differ from full original",
        "otsu_threshold": float(otsu), "closing_kernel": k,
        "medical_segmentation": False, "review_required": True,
        "text_removal_guaranteed": False, "device_removal_guaranteed": False,
    }
    blank = np.zeros_like(small)

    def fallback(reason, mask=None):
        diagnostics.update(applied=False, reason=reason, body_clipped_at_source_edges=[])
        return TightCropResult((0, 0, W, H), (0, 0, W, H),
                               blank if mask is None else mask,
                               blank if seed is None else seed, small, diagnostics)

    value_range = original_range
    if value_range is None:
        value_range = int(small.max()) - int(small.min())
    if value_range < 2 or seed is None:
        return fallback("no_reliable_broad_seed")
    low_threshold = max(config.support_threshold_min,
                        min(config.support_threshold_max,
                            int(math.floor(otsu * config.support_threshold_fraction_of_otsu))))
    raw_support = np.uint8(blur > low_threshold) * 255
    support = cv2.morphologyEx(raw_support, cv2.MORPH_OPEN, open_kernel)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, close_kernel)
    support = _external_fill(support)
    n, labels, stats, _ = _components(support)
    overlaps = np.bincount(labels[seed > 0], minlength=n)
    overlaps[0] = 0
    if not overlaps.any():
        return fallback("no_support_overlapping_seed")
    selected = int(overlaps.argmax())
    body = np.uint8(labels == selected) * 255
    protected_labels = []
    for label in range(1, n):
        x, y, bw, bh, area = stats[label]
        if (label != selected and area >= config.secondary_support_area_fraction * aw*ah
                and bw >= .05*aw and bh >= .05*ah):
            body[labels == label] = 255
            protected_labels.append(label)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours, axis=0))
    envelope = np.zeros_like(body)
    cv2.drawContours(envelope, [hull], -1, 255, thickness=cv2.FILLED)
    bx, by, bw, bh = cv2.boundingRect(hull)
    body_area_fraction = float(np.count_nonzero(body) / body.size)
    seed_retention = float(np.count_nonzero((seed > 0) & (envelope > 0)) / np.count_nonzero(seed))
    diagnostics.update(low_support_threshold=low_threshold, protected_secondary_components=protected_labels,
                       body_area_fraction=body_area_fraction, seed_retention=seed_retention)
    if body_area_fraction < config.minimum_body_area_fraction:
        return fallback("support_too_small", envelope)
    if seed_retention < config.minimum_seed_retention:
        return fallback("seed_not_contained", envelope)
    body_float = (bx*W/aw, by*H/ah, (bx+bw)*W/aw, (by+bh)*H/ah)
    body_box = (math.floor(body_float[0]), math.floor(body_float[1]),
                math.ceil(body_float[2]), math.ceil(body_float[3]))
    mx = max(config.margin_min_original_px, config.margin_fraction * bw*W/aw)
    my = max(config.margin_min_original_px, config.margin_fraction * bh*H/ah)
    box = (max(0, math.floor(body_float[0]-mx)), max(0, math.floor(body_float[1]-my)),
           min(W, math.ceil(body_float[2]+mx)), min(H, math.ceil(body_float[3]+my)))
    x0, y0, x1, y1 = box
    removed = 1. - (x1-x0)*(y1-y0)/(W*H)
    if x1 <= x0 or y1 <= y0 or removed > config.maximum_removed_area_fraction:
        return fallback("implausibly_small_retained_frame", envelope)
    outside = raw_support.copy()
    outside[max(0, math.floor(y0*ah/H)):min(ah, math.ceil(y1*ah/H)),
            max(0, math.floor(x0*aw/W)):min(aw, math.ceil(x1*aw/W))] = 0
    n_out, _, stat_out, _ = _components(outside)
    broad_lost = [int(j) for j in range(1, n_out)
                  if stat_out[j, cv2.CC_STAT_AREA] >= .001*aw*ah
                  and stat_out[j, cv2.CC_STAT_WIDTH] >= .03*aw
                  and stat_out[j, cv2.CC_STAT_HEIGHT] >= .03*ah]
    diagnostics["outside_support_pixels"] = int(np.count_nonzero(outside))
    if broad_lost:
        return fallback("broad_support_outside_candidate", envelope)
    diagnostics.update(applied=box != (0, 0, W, H), reason="body_envelope_candidate",
                       margin_original_px=[mx, my], removed_area_fraction=removed,
                       body_clipped_at_source_edges=[side for side, yes in
                           (("left", bx == 0), ("top", by == 0),
                            ("right", bx+bw == aw), ("bottom", by+bh == ah)) if yes],
                       tiny_outside_support_requires_review=bool(np.count_nonzero(outside)))
    return TightCropResult(box, body_box, envelope, seed, small, diagnostics)


def draft_crop_box(box: tuple[int, int, int, int], original_size: tuple[int, int],
                   draft_size: tuple[int, int]) -> tuple[int, int, int, int]:
    W, H = original_size
    dw, dh = draft_size
    x0, y0, x1, y1 = box
    return (max(0, math.floor(x0 * dw / W)), max(0, math.floor(y0 * dh / H)),
            min(dw, math.ceil(x1 * dw / W)), min(dh, math.ceil(y1 * dh / H)))
