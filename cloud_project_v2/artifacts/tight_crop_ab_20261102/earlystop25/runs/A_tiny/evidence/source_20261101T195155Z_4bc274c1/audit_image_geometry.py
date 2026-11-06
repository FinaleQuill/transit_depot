#!/usr/bin/env python3
"""Read-only geometry audit for the cleaned train/valid chest X-rays.

This audit deliberately has no OpenCV dependency. It uses Pillow and NumPy
only: Otsu thresholding, row-run connected components, and Pillow FIND_EDGES.
It never writes source images or manifests and refuses the
test split because test geometry must not be used to choose a preprocessing
policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFESTS = {
    "train": PROJECT / "data" / "clean_v1_reviewed" / "train_clean.csv",
    "valid": PROJECT / "data" / "clean_v1_reviewed" / "valid_manifest.csv",
}
DEFAULT_IMAGE_ROOT = Path("/home/featurize/data/cxr_first")
DEFAULT_OUTPUT = Path("/tmp/geometry_audit_20261101")
ANALYSIS_SIZE = 512
BLACK_THRESHOLDS = (5, 12)
BLACK_ROW_COVERAGE = 0.98
SAFE_SIDE_LIMIT = 0.08


def iter_rows(path: Path, split: str):
    if split not in DEFAULT_MANIFESTS:
        raise ValueError("Only train and valid audits are permitted; test is forbidden")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        required = {"image_id", "split", "subject_id", "study_id", "study_key",
                    "relative_path", "view_position", "width", "height"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing fields: {sorted(missing)}")
        for row in reader:
            if row["split"] != split:
                raise ValueError(f"Manifest row split {row['split']!r} disagrees with {split!r}")
            yield row


def percentile(values: Iterable[float], p: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, p, method="linear"))


def view_group(value: str) -> str:
    value = (value or "").strip().upper()
    if value.startswith("AP"):
        return "AP"
    if value.startswith("PA"):
        return "PA"
    if value in {"LATERAL", "LL", "RL", "LAT"} or "LAT" in value:
        return "Lateral"
    return "Missing"


def otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    total = image.size
    cumulative = np.cumsum(histogram)
    cumulative_weight = np.cumsum(histogram * np.arange(256, dtype=np.float64))
    total_weight = cumulative_weight[-1]
    denominator = cumulative * (total - cumulative)
    between = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    between[valid] = ((total_weight * cumulative[valid] - cumulative_weight[valid]) ** 2
                      / denominator[valid])
    return int(np.argmax(between))


def row_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return inclusive/exclusive (row, start, end) runs of true pixels."""
    runs: list[tuple[int, int, int]] = []
    for row, values in enumerate(mask):
        padded = np.concatenate(([False], values, [False]))
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        runs.extend((row, int(start), int(end)) for start, end in zip(starts, ends))
    return runs


def largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Connected components from row runs, returning x0,y0,x1,y1,area.

    Components in adjacent rows are joined when their x-ranges overlap.  This
    avoids a Python flood-fill over all pixels while retaining 8-neighbour
    connectivity (one-pixel diagonal touching is joined).
    """
    runs = row_runs(mask)
    if not runs:
        return 0, 0, mask.shape[1], mask.shape[0], 0
    parent = list(range(len(runs)))
    area = [end - start for _, start, end in runs]
    x0 = [start for _, start, _ in runs]
    y0 = [row for row, _, _ in runs]
    x1 = [end for _, _, end in runs]
    y1 = [row + 1 for row, _, _ in runs]

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if area[left] < area[right]:
            left, right = right, left
        parent[right] = left
        area[left] += area[right]
        x0[left], y0[left] = min(x0[left], x0[right]), min(y0[left], y0[right])
        x1[left], y1[left] = max(x1[left], x1[right]), max(y1[left], y1[right])

    previous: list[int] = []
    cursor = 0
    while cursor < len(runs):
        row = runs[cursor][0]
        current: list[int] = []
        while cursor < len(runs) and runs[cursor][0] == row:
            current.append(cursor)
            cursor += 1
        first_possible = 0
        for current_index in current:
            _, start, end = runs[current_index]
            while first_possible < len(previous) and runs[previous[first_possible]][2] < start:
                first_possible += 1
            previous_cursor = first_possible
            while previous_cursor < len(previous):
                previous_index = previous[previous_cursor]
                _, pstart, pend = runs[previous_index]
                if pstart > end:
                    break
                union(current_index, previous_index)
                previous_cursor += 1
        previous = current
    roots = {find(index) for index in range(len(runs))}
    root = max(roots, key=lambda index: area[index])
    return int(x0[root]), int(y0[root]), int(x1[root]), int(y1[root]), int(area[root])


def find_edges(image: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, float, float, float, tuple[int, int, int, int]
]:
    """Return Pillow FIND_EDGES intensity and its adaptive binary edge map.

    This is a real Pillow edge-filter output, not Canny and not a Sobel
    approximation.  The binary mask is used only for geometry diagnostics.
    """
    edge_map = np.asarray(Image.fromarray(image, mode="L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    nonzero = edge_map[edge_map > 0]
    threshold = max(12, int(np.percentile(nonzero, 75))) if nonzero.size else 255
    edges = edge_map >= threshold
    h, w = image.shape
    outer = np.zeros_like(edges)
    border = max(1, int(round(min(h, w) * 0.15)))
    outer[:border] = True
    outer[-border:] = True
    outer[:, :border] = True
    outer[:, -border:] = True
    central = ~outer
    if edges.any():
        yy, xx = np.where(edges)
        bbox = (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))
    else:
        bbox = (0, 0, w, h)
    return edge_map, edges, float(edges.mean()), float(edges[outer].mean()), \
        float(edges[central].mean()) if central.any() else 0.0, bbox


def side_black_margins(image: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    mask = image <= threshold
    row_dark = mask.mean(axis=1) >= BLACK_ROW_COVERAGE
    col_dark = mask.mean(axis=0) >= BLACK_ROW_COVERAGE
    top = next((i for i, value in enumerate(row_dark) if not value), image.shape[0])
    bottom = next((i for i, value in enumerate(row_dark[::-1]) if not value), image.shape[0])
    left = next((i for i, value in enumerate(col_dark) if not value), image.shape[1])
    right = next((i for i, value in enumerate(col_dark[::-1]) if not value), image.shape[1])
    return top, bottom, left, right


def otsu_boundary_offsets(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find per-side intensity boundaries using Otsu on 1-D side projections."""
    profiles = (np.median(image, axis=1), np.median(image[::-1, :], axis=1),
                np.median(image, axis=0), np.median(image[:, ::-1], axis=0))
    offsets = []
    for profile in profiles:
        values = np.clip(np.rint(profile), 0, 255).astype(np.uint8)
        threshold = otsu_threshold(values.reshape(1, -1))
        active = np.flatnonzero(values > threshold)
        offsets.append(int(active[0]) if active.size else len(values))
    return tuple(offsets)


def safe_crop_frame(
    image: np.ndarray,
    black12: tuple[int, int, int, int],
    component: tuple[int, int, int, int, int],
    edge_map: np.ndarray,
    safe_margin_x: float,
    safe_margin_y: float,
    train_p1: float,
    train_p99: float,
) -> tuple[tuple[int, int, int, int], bool, str, dict[str, dict[str, float | bool]]]:
    """Crop only sides where near-black, Otsu and edge signals agree.

    The safety margins are already mapped from original-pixel units into this
    aspect-preserving 512-edge analysis frame.  Unconfirmed sides stay intact.
    """
    h, w = image.shape
    top, bottom, left, right = black12
    bx0, by0, bx1, by1, area = component
    otsu_top, otsu_bottom, otsu_left, otsu_right = otsu_boundary_offsets(image)
    debug: dict[str, dict[str, float | bool]] = {}
    frame = [0, 0, w, h]
    tol_x, tol_y = max(4, int(round(.02 * w))), max(4, int(round(.02 * h)))

    def line_edge_strength(side: str, offset: int, tolerance: int) -> float:
        if side == "top":
            coordinate = offset
            band = edge_map[max(0, coordinate - tolerance):min(h, coordinate + tolerance + 1), :]
        elif side == "bottom":
            coordinate = h - offset
            band = edge_map[max(0, coordinate - tolerance):min(h, coordinate + tolerance + 1), :]
        elif side == "left":
            coordinate = offset
            band = edge_map[:, max(0, coordinate - tolerance):min(w, coordinate + tolerance + 1)]
        else:
            coordinate = w - offset
            band = edge_map[:, max(0, coordinate - tolerance):min(w, coordinate + tolerance + 1)]
        return float(band.mean()) if band.size else 0.0

    sides = [
        ("top", top, otsu_top, h, safe_margin_y, tol_y),
        ("bottom", bottom, otsu_bottom, h, safe_margin_y, tol_y),
        ("left", left, otsu_left, w, safe_margin_x, tol_x),
        ("right", right, otsu_right, w, safe_margin_x, tol_x),
    ]
    reasons = []
    for side, black_width, otsu_offset, axis_length, safety, tolerance in sides:
        margin_fraction = black_width / max(1, axis_length)
        agreement = abs(black_width - otsu_offset) <= tolerance
        proposed_crop = max(0, int(round(black_width - safety)))
        if side == "top":
            outer = edge_map[:max(1, black_width - tolerance), :]
        elif side == "bottom":
            outer = edge_map[max(0, axis_length - black_width + tolerance):, :]
        elif side == "left":
            outer = edge_map[:, :max(1, black_width - tolerance)]
        else:
            outer = edge_map[:, max(0, axis_length - black_width + tolerance):]
        outer_density = float(outer.mean()) if outer.size else 0.0
        boundary_density = line_edge_strength(side, otsu_offset, tolerance)
        edge_support = boundary_density > max(8.0, outer_density + 2.0)
        eligible = (black_width > safety + 1 and margin_fraction <= SAFE_SIDE_LIMIT
                    and agreement and edge_support and proposed_crop / axis_length <= SAFE_SIDE_LIMIT)
        debug[side] = {
            "black_width_fraction": margin_fraction,
            "otsu_black_boundary_px": float(otsu_offset),
            "black_otsu_agreement": agreement,
            "edge_boundary_density": boundary_density,
            "edge_outer_density": outer_density,
            "edge_support": edge_support,
            "safe_margin_px": float(safety),
            "proposed_crop_fraction": proposed_crop / max(1, axis_length),
            "accepted": eligible,
        }
        if not eligible:
            continue
        reasons.append(side)
        if side == "top":
            frame[1] = proposed_crop
        elif side == "bottom":
            frame[3] = h - proposed_crop
        elif side == "left":
            frame[0] = proposed_crop
        else:
            frame[2] = w - proposed_crop

    x0, y0, x1, y1 = frame
    side_fractions = (x0 / w, y0 / h, (w - x1) / w, (h - y1) / h)
    ratio = (x1 - x0) / max(1, y1 - y0)
    if (x1 <= x0 or y1 <= y0 or max(side_fractions) > SAFE_SIDE_LIMIT
            or ratio < train_p1 or ratio > train_p99):
        return (0, 0, w, h), False, "full_frame_aspect_or_crop_limit", debug
    if not reasons:
        return (0, 0, w, h), False, "full_frame_signals_not_agreed", debug
    return tuple(frame), True, "verified:" + ",".join(reasons), debug


def analyze_row(row: dict[str, str], image_root: Path, train_p1: float,
                train_p99: float, manifest_sha256: str) -> dict[str, object]:
    source = (image_root / row["relative_path"]).resolve(strict=True)
    if not source.is_relative_to(image_root.resolve() / row["split"]):
        raise ValueError(f"Image path escapes split root: {source}")
    with Image.open(source) as opened:
        original_width, original_height = opened.size
        used_draft = False
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            opened.draft("L", (ANALYSIS_SIZE, ANALYSIS_SIZE))
            used_draft = True
        opened.load()
        gray = opened.convert("L")
        scale = ANALYSIS_SIZE / max(original_width, original_height)
        analysis_width = max(1, int(round(original_width * scale)))
        analysis_height = max(1, int(round(original_height * scale)))
        thumb = np.asarray(gray.resize((analysis_width, analysis_height), Image.Resampling.BILINEAR), dtype=np.uint8)
    threshold = otsu_threshold(thumb)
    foreground = thumb > threshold
    component = largest_component_bbox(foreground)
    edge_map, edge_binary, edge_density, outer_edge_density, central_edge_density, edge_bbox = find_edges(thumb)
    margins12 = side_black_margins(thumb, BLACK_THRESHOLDS[1])
    black5 = float((thumb <= BLACK_THRESHOLDS[0]).mean())
    black12 = float((thumb <= BLACK_THRESHOLDS[1]).mean())
    foreground_area = component[4] / float(analysis_width * analysis_height)
    foreground_bbox_area = ((component[2] - component[0]) * (component[3] - component[1])
                            / float(analysis_width * analysis_height))
    safe_margin_x = max(.02 * original_width, 32) * scale
    safe_margin_y = max(.02 * original_height, 32) * scale
    frame, safe, crop_reason, side_debug = safe_crop_frame(
        thumb, margins12, component, edge_map, safe_margin_x, safe_margin_y, train_p1, train_p99)
    crop_ratio = (frame[2] - frame[0]) / max(1, frame[3] - frame[1])
    max_black_margin = max(margins12[0] / analysis_height, margins12[1] / analysis_height,
                           margins12[2] / analysis_width, margins12[3] / analysis_width)
    irregularity = float(abs(foreground_bbox_area - foreground_area) +
                         abs(outer_edge_density - central_edge_density))
    text_like = float(outer_edge_density / max(central_edge_density, 1e-6))
    candidate = bool(max_black_margin >= 0.02 or safe or irregularity >= 0.20 or text_like >= 1.8)
    return {
        "image_id": row["image_id"], "subject_id": row["subject_id"],
        "study_id": row["study_id"], "study_key": row["study_key"],
        "split": row["split"], "view_position": row.get("view_position", ""),
        "view_group": view_group(row.get("view_position", "")),
        "relative_path": row["relative_path"], "source_path": str(source),
        "width": original_width, "height": original_height,
        "aspect_ratio": original_width / max(1, original_height),
        "analysis_width": analysis_width, "analysis_height": analysis_height,
        "analysis_longest_edge": ANALYSIS_SIZE, "analysis_scale": scale,
        "analysis_decode": "Pillow JPEG draft then bilinear resize" if used_draft else "Pillow full decode then bilinear resize",
        "manifest_sha256": manifest_sha256,
        "source_file_sha256_manifest": row.get("file_sha256", ""),
        "otsu_threshold": threshold,
        "black_fraction_le_5": black5, "black_fraction_le_12": black12,
        "black_margin_top": margins12[0] / analysis_height,
        "black_margin_bottom": margins12[1] / analysis_height,
        "black_margin_left": margins12[2] / analysis_width,
        "black_margin_right": margins12[3] / analysis_width,
        "black_margin_max": max_black_margin,
        "black_margin_top_original_px": margins12[0] / scale,
        "black_margin_bottom_original_px": margins12[1] / scale,
        "black_margin_left_original_px": margins12[2] / scale,
        "black_margin_right_original_px": margins12[3] / scale,
        "black_margin_max_original_px": max(margins12) / scale,
        "foreground_area": foreground_area,
        "foreground_bbox_area": foreground_bbox_area,
        "foreground_bbox_x0": component[0] / analysis_width,
        "foreground_bbox_y0": component[1] / analysis_height,
        "foreground_bbox_x1": component[2] / analysis_width,
        "foreground_bbox_y1": component[3] / analysis_height,
        "edge_density": edge_density, "outer_edge_density": outer_edge_density,
        "central_edge_density": central_edge_density, "edge_bbox_x0": edge_bbox[0] / analysis_width,
        "edge_bbox_y0": edge_bbox[1] / analysis_height,
        "edge_bbox_x1": edge_bbox[2] / analysis_width,
        "edge_bbox_y1": edge_bbox[3] / analysis_height,
        "edge_filter": "Pillow ImageFilter.FIND_EDGES; not Canny",
        "edge_consistency_ratio": outer_edge_density / max(central_edge_density, 1e-6),
        "irregularity_score": irregularity, "text_like_score": text_like,
        "crop_x0": frame[0] / analysis_width, "crop_y0": frame[1] / analysis_height,
        "crop_x1": frame[2] / analysis_width, "crop_y1": frame[3] / analysis_height,
        "crop_original_x0": int(round(frame[0] / scale)),
        "crop_original_y0": int(round(frame[1] / scale)),
        "crop_original_x1": int(round(frame[2] / scale)),
        "crop_original_y1": int(round(frame[3] / scale)),
        "crop_aspect_ratio": crop_ratio, "safe_crop": safe,
        "safe_margin_x_original_px": max(.02 * original_width, 32),
        "safe_margin_y_original_px": max(.02 * original_height, 32),
        "crop_reason": crop_reason, "side_validation": json.dumps(side_debug, sort_keys=True),
        "candidate": candidate,
        "candidate_reasons": ";".join(filter(None, [
            "black_band" if max_black_margin >= 0.02 else "",
            "safe_crop" if safe else "", "irregular_foreground" if irregularity >= 0.20 else "",
            "outer_edge_text_like" if text_like >= 1.8 else ""])),
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analysis_pixels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as opened:
        opened.load()
        gray = opened.convert("L")
        scale = ANALYSIS_SIZE / max(gray.size)
        size = (max(1, int(round(gray.width * scale))), max(1, int(round(gray.height * scale))))
        original = np.asarray(gray.resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)
    edge_map = np.asarray(Image.fromarray(original, mode="L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    return original, edge_map


def save_example(row: dict[str, object], output: Path, title: str) -> str:
    source = Path(str(row["source_path"]))
    original, edge_map = load_analysis_pixels(source)
    h, w = original.shape
    original_image = Image.fromarray(original, mode="L").convert("RGB")
    edge_image = Image.fromarray(edge_map, mode="L").convert("RGB")
    boxed = original_image.copy()
    draw = ImageDraw.Draw(boxed)

    def box(prefix: str, colour: tuple[int, int, int]) -> None:
        x0 = int(round(float(row[f"{prefix}_x0"]) * w))
        y0 = int(round(float(row[f"{prefix}_y0"]) * h))
        x1 = int(round(float(row[f"{prefix}_x1"]) * w))
        y1 = int(round(float(row[f"{prefix}_y1"]) * h))
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=2)

    box("crop", (255, 0, 0))
    box("foreground_bbox", (0, 220, 0))
    box("edge_bbox", (0, 100, 255))
    caption_h = 64
    panel_w = w
    panel_h = h + caption_h
    collage = Image.new("RGB", (panel_w * 3, panel_h), (18, 18, 18))
    for index, (panel, label) in enumerate(((original_image, "Original / 512 longest edge"),
                                            (edge_image, "Pillow FIND_EDGES (not Canny)"),
                                            (boxed, "Crop red / Otsu green / Edge blue"))):
        collage.paste(panel, (index * panel_w, caption_h))
        panel_draw = ImageDraw.Draw(collage)
        if index == 2:
            panel_draw.text((index * panel_w + 4, 3),
                            f"{title} | {row['image_id']} | {row['view_group']}", fill=(255, 255, 255))
            panel_draw.text((index * panel_w + 4, 20),
                            f"R crop {float(row['crop_x0']):.3f},{float(row['crop_y0']):.3f}-"
                            f"{float(row['crop_x1']):.3f},{float(row['crop_y1']):.3f}", fill=(255, 80, 80))
            panel_draw.text((index * panel_w + 4, 35),
                            f"G Otsu {float(row['foreground_bbox_x0']):.3f},{float(row['foreground_bbox_y0']):.3f}-"
                            f"{float(row['foreground_bbox_x1']):.3f},{float(row['foreground_bbox_y1']):.3f}", fill=(60, 245, 80))
            panel_draw.text((index * panel_w + 4, 50),
                            f"B edge {float(row['edge_bbox_x0']):.3f},{float(row['edge_bbox_y0']):.3f}-"
                            f"{float(row['edge_bbox_x1']):.3f},{float(row['edge_bbox_y1']):.3f}", fill=(80, 145, 255))
        else:
            panel_draw.text((index * panel_w + 4, 3), label, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output, format="PNG", optimize=True)
    return hash_file(source)


def score_example(row: dict[str, object], kind: str) -> tuple[float, ...]:
    if kind == "maximum_candidate":
        return (float(bool(row["candidate"])), float(row["black_margin_max"]),
                float(row["irregularity_score"]))
    if kind == "typical_text_like":
        return (float(row["text_like_score"]), float(row["outer_edge_density"]))
    return (float(row["irregularity_score"]), float(row["black_margin_max"]))


def update_examples(selected: dict[str, dict[str, dict[str, object]]], row: dict[str, object]) -> None:
    group = str(row["view_group"])
    current = selected.setdefault(group, {})
    for kind in ("maximum_candidate", "typical_text_like", "irregular_foreground"):
        old = current.get(kind)
        if old is None or score_example(row, kind) > score_example(old, kind):
            current[kind] = row


def run(args: argparse.Namespace) -> dict[str, object]:
    split = args.split
    canonical_manifest = DEFAULT_MANIFESTS[split].resolve(strict=True)
    manifest = Path(args.manifest).resolve(strict=True) if args.manifest else canonical_manifest
    if manifest != canonical_manifest:
        raise ValueError(f"--manifest must be the canonical {split} manifest: {canonical_manifest}")
    image_root = Path(args.image_root).resolve(strict=True)
    output = Path(args.output).resolve()
    # Geometry bounds are based on train manifest metadata only.  Valid is
    # diagnosed against the frozen train envelope and cannot influence it.
    train_manifest = DEFAULT_MANIFESTS["train"].resolve(strict=True)
    train_ratios = [float(r["width"]) / max(1, float(r["height"]))
                    for r in iter_rows(train_manifest, "train")]
    train_p1, train_p99 = percentile(train_ratios, 1), percentile(train_ratios, 99)
    manifest_sha256 = hash_file(manifest)
    images_declared = sum(1 for _ in iter_rows(manifest, split))
    split_output = output / split
    split_output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    csv_path = split_output / "geometry_candidates.csv"
    failure_path = split_output / "failures.csv"
    selected: dict[str, dict[str, dict[str, object]]] = {}
    by_view_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Counter[str] = Counter()
    aspect_values: list[float] = []
    failures_preview: list[dict[str, str]] = []
    scanned = 0
    with csv_path.open("w", encoding="utf-8", newline="") as csv_stream, \
            failure_path.open("w", encoding="utf-8", newline="") as failure_stream:
        writer = None
        failure_writer = csv.DictWriter(failure_stream, fieldnames=("image_id", "error"))
        failure_writer.writeheader()
        for index, row in enumerate(iter_rows(manifest, split), 1):
            try:
                analysis = analyze_row(row, image_root, train_p1, train_p99, manifest_sha256)
                if writer is None:
                    writer = csv.DictWriter(csv_stream, fieldnames=list(analysis))
                    writer.writeheader()
                writer.writerow(analysis)
                scanned += 1
                group = str(analysis["view_group"])
                counts[f"{group}:images"] += 1
                counts[f"{group}:candidate_images"] += bool(analysis["candidate"])
                counts[f"{group}:safe_crop_images"] += bool(analysis["safe_crop"])
                counts["candidate_images"] += bool(analysis["candidate"])
                counts["safe_crop_images"] += bool(analysis["safe_crop"])
                counts["black_margin_ge_2pct"] += float(analysis["black_margin_max"]) >= .02
                counts["black_margin_ge_5pct"] += float(analysis["black_margin_max"]) >= .05
                for cutoff in (1.5, 3.0, 5.0):
                    counts[f"{group}:black_margin_ge_{cutoff:g}pct"] += \
                        float(analysis["black_margin_max"]) >= cutoff / 100
                for metric in ("black_margin_max", "black_margin_max_original_px", "black_fraction_le_12",
                               "edge_density", "aspect_ratio", "irregularity_score", "text_like_score"):
                    by_view_values[group][metric].append(float(analysis[metric]))
                aspect_values.append(float(analysis["aspect_ratio"]))
                update_examples(selected, analysis)
            except Exception as exc:  # retain a row-level audit record, continue scanning
                failure = {"image_id": row.get("image_id", ""), "error": repr(exc)}
                failure_writer.writerow(failure)
                if len(failures_preview) < 20:
                    failures_preview.append(failure)
            if args.progress and (index == 1 or index % args.progress == 0 or index == images_declared):
                print(f"[{split}] {index}/{images_declared} images", flush=True)
    if scanned == 0:
        raise RuntimeError(f"No images scanned successfully; inspect {failure_path}")
    examples = []
    # Reopen only selected examples; no thumbnail is retained during the scan.
    for group in ("AP", "PA", "Lateral", "Missing"):
        kinds = selected.get(group, {})
        for kind, selected_row in kinds.items():
            name = f"{group.lower()}_{kind}.png"
            image_sha256 = save_example(selected_row, split_output / "examples" / name, kind)
            examples.append({"group": group, "kind": kind, "image_id": selected_row["image_id"],
                             "file": f"examples/{name}", "source_file_sha256": image_sha256,
                             "manifest_sha256": manifest_sha256,
                             "source_file_sha256_manifest": selected_row["source_file_sha256_manifest"],
                             "source_sha256_matches_manifest": image_sha256 == selected_row["source_file_sha256_manifest"],
                             "candidate": selected_row["candidate"],
                             "crop_reason": selected_row["crop_reason"],
                             "box_coordinates_normalized": {
                                 "crop_red": [selected_row[f"crop_{axis}"] for axis in ("x0", "y0", "x1", "y1")],
                                 "otsu_green": [selected_row[f"foreground_bbox_{axis}"] for axis in ("x0", "y0", "x1", "y1")],
                                 "edge_blue": [selected_row[f"edge_bbox_{axis}"] for axis in ("x0", "y0", "x1", "y1")],
                             }})
    elapsed = time.monotonic() - start
    by_view = {}
    for group in ("AP", "PA", "Lateral", "Missing"):
        values = by_view_values.get(group, {})
        by_view[group] = {
            "images": counts[f"{group}:images"],
            "candidate_images": counts[f"{group}:candidate_images"],
            "safe_crop_images": counts[f"{group}:safe_crop_images"],
            "candidate_rate": counts[f"{group}:candidate_images"] / max(1, counts[f"{group}:images"]),
            "safe_crop_rate": counts[f"{group}:safe_crop_images"] / max(1, counts[f"{group}:images"]),
            "black_margin_thresholds": {
                f"ge_{cutoff:g}pct": {
                    "count": counts[f"{group}:black_margin_ge_{cutoff:g}pct"],
                    "rate": counts[f"{group}:black_margin_ge_{cutoff:g}pct"] / max(1, counts[f"{group}:images"]),
                } for cutoff in (1.5, 3.0, 5.0)
            },
            "black_margin_max_original_px": max(values.get("black_margin_max_original_px", []), default=None),
            "black_margin_p95": percentile(values.get("black_margin_max", []), 95),
            "black_fraction_le_12_median": percentile(values.get("black_fraction_le_12", []), 50),
            "edge_density_median": percentile(values.get("edge_density", []), 50),
            "aspect_ratio_p01": percentile(values.get("aspect_ratio", []), 1),
            "aspect_ratio_p99": percentile(values.get("aspect_ratio", []), 99),
            "irregularity_p95": percentile(values.get("irregularity_score", []), 95),
            "text_like_p95": percentile(values.get("text_like_score", []), 95),
        }
    summary = {
        "schema_version": 1, "algorithm": {
            "library": "Pillow+NumPy", "longest_analysis_edge": ANALYSIS_SIZE,
            "aspect_ratio_preserved": True,
            "decode": "Pillow JPEG draft to <=512x512 request, then exact aspect-preserving bilinear resize; source dimensions captured before draft",
            "otsu": "global image threshold and per-side median-projection thresholds; between-class variance",
            "connected_components": "row-run 8-neighbour overlap; largest by pixel area",
            "otsu_boundary": "Otsu threshold on per-side median-intensity projections",
            "edge": "Pillow ImageFilter.FIND_EDGES intensity and adaptive binary edge map; not Canny",
            "black_band": {"thresholds": BLACK_THRESHOLDS, "row_coverage": BLACK_ROW_COVERAGE},
            "safe_margin": "max(2% of original side length, 32 original pixels), mapped to analysis coordinates",
        },
        "split": split, "manifest": str(manifest), "image_root": str(image_root),
        "manifest_sha256": manifest_sha256,
        "images_declared": images_declared, "images_scanned": scanned,
        "failure_count": images_declared - scanned, "failure_preview": failures_preview,
        "failure_csv": str(failure_path),
        "selected_sample_hash_audit": {
            "samples_hashed": len(examples),
            "manifest_sha256_bound_per_csv_row": True,
            "source_sha256_matches_manifest": sum(bool(item["source_sha256_matches_manifest"]) for item in examples),
        },
        "elapsed_seconds": elapsed, "train_aspect_p1": train_p1, "train_aspect_p99": train_p99,
        "aspect_ratio": {"p01": percentile(aspect_values, 1),
                         "median": percentile(aspect_values, 50),
                         "p99": percentile(aspect_values, 99)},
        "counts": {key: int(value) for key, value in counts.items()
                   if ":" not in key},
        "by_view": by_view, "examples": examples,
        "source_integrity": "read-only; no source or manifest writes",
    }
    (split_output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_preregistered(output: Path) -> None:
    payload = {
        "schema_version": 1, "purpose": "geometry preprocessing A/B audit",
        "test_access": "forbidden; no test manifest or image may be read",
        "A_original": {"name": "original_letterbox", "crop": "none", "resize": "longest_side_then_center_zero_pad", "geometry_invariant": True},
        "B_candidate": {"name": "review_only_safe_geometry_crop", "crop": "candidate coordinates only; not a deployment transform", "max_side_fraction": SAFE_SIDE_LIMIT, "safe_margin_original_px": "max(2% of source side, 32)", "aspect_envelope_source": "train metadata P1-P99", "geometry_invariant": True, "fallback": "A full frame when signals disagree or limits fail"},
        "selection": "no candidate crop is selected for deployment by this audit; any A/B inference requires separate authorization",
        "do_not_do": ["no automatic deletion", "no label edits", "no fixed global black threshold crop", "no test-derived threshold or selection"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "preregistered_ab.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid"), required=True,
                        help="audit train or valid; test is intentionally refused")
    parser.add_argument("--manifest", type=Path, help="optional matching clean manifest")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=int, default=2000, help="progress interval; 0 disables")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        canonical = DEFAULT_MANIFESTS[args.split].resolve(strict=True)
        try:
            supplied = Path(args.manifest).resolve(strict=True)
        except FileNotFoundError:
            parser.error("--manifest must identify the canonical manifest for the selected split")
        if supplied != canonical:
            parser.error(f"--manifest must identify the canonical {args.split} manifest")
    write_preregistered(Path(args.output).resolve())
    summary = run(args)
    print(json.dumps({"split": args.split, "summary": str(Path(args.output).resolve() / args.split / "summary.json"),
                      "scanned": summary["images_scanned"], "failures": summary["failure_count"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
