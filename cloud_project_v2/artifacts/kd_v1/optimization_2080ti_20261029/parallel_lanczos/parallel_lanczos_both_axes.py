"""Artifact-only two-pass Lanczos: horizontal rows then vertical columns.

One at-most-four-worker executor serves one image. Both passes join all their
futures before the next pass/normalization/GPU; no work crosses source images.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def resize_single_image(image: Image.Image, size: tuple[int, int], *,
                        workers: int = 4, min_pixels: int = 1_000_000) -> Image.Image:
    if image.mode not in {'L', 'RGB'}:
        raise ValueError('Convert to post-EXIF L/RGB before parallel resizing')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Use at most four workers within one source image')
    if type(min_pixels) is not int or min_pixels < 0:
        raise ValueError('Nonnegative pixel threshold required')
    width, height = image.size
    if workers == 1 or width * height < min_pixels or image.size == size:
        return image.resize(size, Image.Resampling.LANCZOS)
    intermediate = image
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='single-image-lanczos') as pool:
        if width != size[0]:
            count = min(workers, height)
            rows = [(height * i // count, height * (i + 1) // count) for i in range(count)]
            horizontal = [pool.submit(image.crop((0, top, width, bottom)).resize,
                                      (size[0], bottom - top), Image.Resampling.LANCZOS)
                          for top, bottom in rows]
            bands = [future.result() for future in horizontal]
            intermediate = Image.new(image.mode, (size[0], height), 0)
            for (top, _), band in zip(rows, bands):
                intermediate.paste(band, (0, top))
        if height == size[1]:
            return intermediate
        count = min(workers, intermediate.width)
        columns = [(intermediate.width * i // count, intermediate.width * (i + 1) // count)
                   for i in range(count)]
        vertical = [pool.submit(intermediate.crop((left, 0, right, height)).resize,
                                (right - left, size[1]), Image.Resampling.LANCZOS)
                    for left, right in columns]
        bands = [future.result() for future in vertical]
        result = Image.new(image.mode, size, 0)
        for (left, _), band in zip(columns, bands):
            result.paste(band, (left, 0))
    return result


def letterbox(image: Image.Image, image_size: int, *, workers: int = 4,
              min_pixels: int = 1_000_000) -> Image.Image:
    scale = image_size / max(image.size)
    size = tuple(max(1, min(image_size, round(length * scale))) for length in image.size)
    resized = resize_single_image(image, size, workers=workers, min_pixels=min_pixels)
    canvas = Image.new(image.mode, (image_size, image_size), 0)
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return canvas


def image_array(image: Image.Image, image_size: int, *, workers: int = 4,
                min_pixels: int = 1_000_000) -> np.ndarray:
    pixels = np.asarray(letterbox(image, image_size, workers=workers, min_pixels=min_pixels))
    grayscale = image.mode == 'L'
    result = np.empty((3, image_size, image_size), dtype=np.float32)
    offset = np.asarray(MEAN, dtype=np.float32) * np.float32(255)
    denominator = np.asarray(STD, dtype=np.float32) * np.float32(255)
    for channel in range(3):
        np.copyto(result[channel], pixels if grayscale else pixels[:, :, channel])
        np.subtract(result[channel], offset[channel], out=result[channel])
        np.divide(result[channel], denominator[channel], out=result[channel])
    return result


def load_image_arrays(path: str | Path, image_sizes: Sequence[int], *, workers: int = 4,
                      min_pixels: int = 1_000_000) -> dict[int, np.ndarray]:
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {'I', 'F'} or source.mode.startswith('I;'):
            raise ValueError('Non-8-bit image requires an explicit intensity policy')
        image = source if source.mode == 'L' else source.convert('RGB')
        return {size: image_array(image, size, workers=workers, min_pixels=min_pixels)
                for size in sorted(set(image_sizes))}
