"""Exact full-box grayscale resize with a portable optional native implementation.

Only Pillow 10.4.0 L images use its pinned Imaging capsule ABI. All other modes
or environments keep Pillow's original implementation. No decoding, EXIF, color,
padding, precision or input/output-size policy is changed here.
"""
from __future__ import annotations

import os
from PIL import Image, __version__ as PILLOW_VERSION

try:
    from native_lanczos import _core
except ImportError:
    _core = None


def available() -> bool:
    return _core is not None and PILLOW_VERSION == "10.4.0"


def implementation() -> str:
    """Report CPU dispatch for project preparation and timing evidence."""
    return getattr(_core, "IMPLEMENTATION", "generic") if available() else "pillow"


def default_threads() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    return max(1, min(8, count))


def resize(image: Image.Image, size: tuple[int, int], *, threads: int | None = None) -> Image.Image:
    """Return the same uint8 pixels as Pillow LANCZOS for full-box resize."""
    if not available() or image.mode != "L":
        return image.resize(size, Image.Resampling.LANCZOS)
    if len(size) != 2 or any(type(value) is not int or value < 1 for value in size):
        raise ValueError("Expected two positive integer output dimensions")
    threads = default_threads() if threads is None else threads
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Native resampling threads must be in [1, 8]")
    image.load()
    output = Image.new("L", size)
    _core.resize_l(image.getim(), output.getim(), threads)
    return output
