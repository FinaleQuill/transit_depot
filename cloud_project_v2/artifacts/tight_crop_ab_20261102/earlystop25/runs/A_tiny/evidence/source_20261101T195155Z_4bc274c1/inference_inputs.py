"""Exact uint8-to-FP16 inference inputs and a single checked result transfer.

Importing this module does not import Torch, initialize CUDA, read image files,
or construct a model. Each image is decoded and transformed inside its call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageOps


def normalization_table(mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Return 3 x 256 FP32 values with the original subtract/divide rounding."""
    table = np.empty((3, 256), dtype=np.float32)
    offset = np.asarray(mean, dtype=np.float32) * np.float32(255.0)
    denominator = np.asarray(std, dtype=np.float32) * np.float32(255.0)
    pixels = np.arange(256, dtype=np.uint8)
    for channel in range(3):
        np.copyto(table[channel], pixels)
        np.subtract(table[channel], offset[channel], out=table[channel])
        np.divide(table[channel], denominator[channel], out=table[channel])
    return table


def normalize_pixels(pixels: np.ndarray, mean, std) -> np.ndarray:
    """Keep the original CPU FP32 arithmetic for L and RGB pixels."""
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("Expected uint8 L or RGB pixels")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("Expected uint8 L or RGB pixels")
    result = np.empty((3, *pixels.shape[:2]), dtype=np.float32)
    offset = np.asarray(mean, dtype=np.float32) * np.float32(255.0)
    denominator = np.asarray(std, dtype=np.float32) * np.float32(255.0)
    for channel in range(3):
        np.copyto(result[channel], pixels if pixels.ndim == 2 else pixels[:, :, channel])
        np.subtract(result[channel], offset[channel], out=result[channel])
        np.divide(result[channel], denominator[channel], out=result[channel])
    return result


def load_image_pixels(path: str | Path, image_sizes: Sequence[int],
                      letterbox: Callable[[Image.Image, int, str], Any]) -> dict[int, np.ndarray]:
    """Decode/EXIF once and return the original post-letterbox uint8 pixels.

    The caller supplies the validated letterbox implementation. It can return
    a PIL image or an ndarray; interpolation, rounding and padding must match.
    """
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {"I", "F"} or source.mode.startswith("I;"):
            raise ValueError("Non-8-bit image requires an explicit intensity policy")
        image = source if source.mode == "L" else source.convert("RGB")
        mode = "L" if image.mode == "L" else "RGB"
        return {size: np.asarray(letterbox(image, size, mode))
                for size in sorted(set(image_sizes))}


class ExactU8Input:
    """Use a fixed lookup for gray inputs, retaining the original RGB path.

    Table entries use the original FP32-to-CUDA-FP16 conversion. Embedding writes
    HWC FP16 directly and a view supplies channels_last NCHW without arithmetic.
    Pinned staging is private to a batch1 serial Predictor. Its final result
    transfer must finish before the next image reuses the buffers. A caller that
    catches an earlier exception must call sync_after_error before reuse.
    """

    def __init__(self, torch, device, mean, std, *, image_sizes=(), pin_memory=True):
        self.torch = torch
        self.device = torch.device(device)
        self.mean, self.std = tuple(mean), tuple(std)
        table = normalization_table(mean, std).T.copy()
        self.table_fp16 = torch.from_numpy(table).to(self.device, dtype=torch.float16)
        self.staging = {}
        if pin_memory and self.device.type == "cuda":
            # Only allocate buffers here; image pixels are read during inference.
            for size in sorted(set(image_sizes)):
                host = torch.empty((size, size), dtype=torch.uint8, pin_memory=True)
                device_pixels = torch.empty((size, size), dtype=torch.uint8, device=self.device)
                self.staging[(size, size)] = (host, host.numpy(), device_pixels)

    def to_tensor(self, pixels: np.ndarray):
        torch = self.torch
        if pixels.dtype != np.uint8:
            raise ValueError("Expected uint8 pixels")
        if pixels.ndim != 2:
            array = normalize_pixels(pixels, self.mean, self.std)
            return torch.from_numpy(array).unsqueeze(0).to(
                self.device, dtype=torch.float16, memory_format=torch.channels_last)
        if pixels.shape in self.staging:
            host, writable_pixels, device_pixels = self.staging[pixels.shape]
            np.copyto(writable_pixels, pixels)
            device_pixels.copy_(host, non_blocking=True)
            indices = device_pixels.to(dtype=torch.int32)
        else:
            # PIL arrays can be read-only; expose an owned writable byte array.
            host = torch.from_numpy(np.array(pixels, dtype=np.uint8, order="C", copy=True))
            indices = host.to(self.device).to(dtype=torch.int32)
        hwc = torch.nn.functional.embedding(indices, self.table_fp16)
        return hwc.unsqueeze(0).permute(0, 3, 1, 2)

    def sync_after_error(self) -> None:
        """Fence only failed calls before their pinned source buffers are reused."""
        if self.staging:
            self.torch.cuda.synchronize(self.device)


def member_results_to_cpu(torch, results, finite_checks) -> np.ndarray:
    """Transfer member means and their raw-logit finite guard together once.

    Means are copied without recomputation; callers keep the existing FP64
    member aggregation. Checking original logits also rejects infinities hidden
    by sigmoid. The exact FP32 scalar appended to the payload is either 0 or 1.
    """
    means = torch.stack(results)
    valid = torch.stack(finite_checks).all().reshape(1).to(dtype=torch.float32)
    payload = torch.cat((means.reshape(-1), valid)).cpu().numpy()
    if payload[-1] != np.float32(1.0):
        raise FloatingPointError("Invalid single-image student logits")
    return payload[:-1].reshape(len(results), 10)
