"""Experimental body-envelope crop. No dataset IO or production integration.

The mask describes image support, not a medically validated thorax/lung mask.
All returned boxes require visual review before any training experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TightCropConfig:
    version: str = "body_envelope_prototype_v1"
    analysis_long_edge: int = 512
    margin_fraction: float = 0.05  # per side, relative to detected body bbox
    margin_min_original_px: int = 32
    closing_fraction: float = 0.025
    support_threshold_fraction_of_otsu: float = 0.12
    support_threshold_min: int = 2
    support_threshold_max: int = 12
    minimum_body_area_fraction: float = 0.10
    maximum_removed_area_fraction: float = 0.75
    minimum_seed_retention: float = 0.98
    secondary_support_area_fraction: float = 0.005


@dataclass
class TightCropResult:
    box: tuple[int, int, int, int]
    body_box: tuple[int, int, int, int]
    mask: np.ndarray
    seed_mask: np.ndarray
    analysis_gray: np.ndarray
    diagnostics: dict


def _components(mask):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def _central_component(mask):
    """Pick a broad component intersecting the central rectangle, not a glyph."""
    h, w = mask.shape
    n, labels, stats, _ = _components(mask)
    central = labels[round(0.10*h):round(0.90*h), round(0.25*w):round(0.75*w)]
    candidates = []
    for label in np.unique(central):
        if label == 0:
            continue
        x, y, bw, bh, area = stats[label]
        if bw >= .15*w and bh >= .15*h:
            candidates.append((int(area), int(label)))
    if not candidates:
        return None
    label = max(candidates)[1]
    return np.uint8(labels == label) * 255


def _external_fill(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def detect_tight_crop(gray: np.ndarray, config: TightCropConfig | None = None) -> TightCropResult:
    config = config or TightCropConfig()
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 8:
        raise ValueError("Expected a nonempty original uint8 grayscale image")
    if not .05 <= config.margin_fraction <= .10:
        raise ValueError("Prototype margin must be 5%-10% per side")
    H, W = gray.shape
    s = min(1., config.analysis_long_edge / max(W, H))
    aw, ah = max(1, round(W*s)), max(1, round(H*s))
    small = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    otsu, high = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(3, round(min(aw, ah) * config.closing_fraction)) | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Bright seeds are not themselves lung masks: dark lungs need lower-level support.
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, open_kernel)
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, close_kernel)
    seed = _central_component(high)
    diagnostics = {
        "config": asdict(config), "opencv_version": cv2.__version__,
        "original_size": [W, H], "analysis_size": [aw, ah],
        "decode": "full grayscale; analysis INTER_AREA; no JPEG draft",
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

    if int(gray.max()) - int(gray.min()) < 2 or seed is None:
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
    # Protect other broad image-support components, rather than discarding dark side tissue.
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
    # Actual x/y analysis dimensions are used separately; all rounding expands retention.
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
    # Verify no broad low-intensity island was lost by opening/seed selection.
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


def letterbox(image: Image.Image, size: int = 768):
    if size < 1 or min(image.size) < 1:
        raise ValueError("Invalid image or output size")
    w, h = image.size
    scale = size/max(w, h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    image = image.convert("RGB")
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (size-nw)//2, (size-nh)//2
    output = Image.new("RGB", (size, size), (0, 0, 0))
    output.paste(resized, (left, top))
    return output, {"scale": scale, "resized_size": (nw, nh),
                    "content_box": (left, top, left+nw, top+nh),
                    "single_isotropic_scale": True, "padding_value_u8": 0}


def random_erasing_u8(image: np.ndarray, content_box, *, training=False,
                      rng=None, p=.5, area_range=(.005, .02)):
    """One small uniform-location black patch before normalization, train only.

    High intensity is not a nuisance label: we deliberately do not target opacity,
    pacemakers or lines. This is stochastic regularization, not device removal.
    """
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Erasing requires uint8 pixels before normalization")
    if not 0 <= p <= 1 or not 0 < area_range[0] <= area_range[1] <= .02:
        raise ValueError("Invalid probability or excessive erase area")
    out = image.copy()
    record = {"applied": False, "probability": p, "training": bool(training),
              "selection": "uniform_position_in_content_not_brightness", "box": None}
    if not training:
        return out, record
    if rng is None:
        raise ValueError("Training erasing requires an explicit worker RNG")
    if rng.random() >= p:
        return out, record
    x0, y0, x1, y1 = map(int, content_box)
    H, W = image.shape[:2]
    if not 0 <= x0 < x1 <= W or not 0 <= y0 < y1 <= H:
        raise ValueError("Content box outside image")
    cw, ch = x1-x0, y1-y0
    target_area = rng.uniform(*area_range) * cw*ch
    aspect = math.exp(rng.uniform(math.log(.5), math.log(2.)))
    ew = min(cw, max(1, math.floor(math.sqrt(target_area * aspect))))
    eh = min(ch, max(1, math.floor(math.sqrt(target_area / aspect))))
    ex = int(rng.integers(x0, x1-ew+1))
    ey = int(rng.integers(y0, y1-eh+1))
    out[ey:ey+eh, ex:ex+ew] = 0
    record.update(applied=True, box=[ex, ey, ex+ew, ey+eh],
                  area_fraction_of_content=ew*eh/(cw*ch), value_before_normalization=0)
    return out, record
