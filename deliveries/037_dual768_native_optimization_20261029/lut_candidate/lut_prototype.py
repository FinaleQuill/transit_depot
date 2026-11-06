"""Isolated exact-input / one-D2H prototype; imports never initialize CUDA.

Production deployment files are deliberately not edited by this module. The
caller supplies the frozen deployment module and its already-loaded Predictor.
All pixel-dependent processing stays inside predict_image; no cross-image cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageOps


def normalization_table(mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """3 x 256 FP32, preserving the deployed subtract/divide operation sequence."""
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
    """Unchanged CPU fallback for RGB; also an independent LUT input oracle."""
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
    """Original decode/EXIF/color/geometry, returning only post-letterbox bytes.

    letterbox must preserve the deployed Pillow output. It may return a PIL
    image or a uint8 NumPy array, so a separately-validated native resize can be
    substituted without changing this loader's interface.
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
    """Generic Torch SM75-compatible lookup; CUDA is touched only at init/use.

    The table uses the same FP32 -> CUDA FP16 conversion as the old image path.
    Gray indices produce contiguous HWC FP16 directly via one embedding gather;
    the final permutation is a channels_last view, with no arithmetic rounding.
    RGB intentionally follows the old CPU FP32 path until independently tuned.
    """

    def __init__(self, torch, device, mean, std, *, image_sizes=(), pin_memory=True):
        self.torch = torch
        self.device = torch.device(device)
        self.mean, self.std = tuple(mean), tuple(std)
        table = normalization_table(mean, std).T.copy()  # 256 x 3, contiguous.
        self.table_fp16 = torch.from_numpy(table).to(self.device, dtype=torch.float16)
        self.staging = {}
        if pin_memory and self.device.type == "cuda":
            # Allocation only, no image pixels are read during initialization.
            # A required final D2H completes every successful predict_image, so
            # this source buffer is safe to reuse at the next serial image.
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
            # PIL arrays are normally read-only. Take one owned uint8 copy
            # rather than exposing a read-only array as a writable Torch tensor.
            host = torch.from_numpy(np.array(pixels, dtype=np.uint8, order="C", copy=True))
            indices = host.to(self.device).to(dtype=torch.int32)
        hwc = torch.nn.functional.embedding(indices, self.table_fp16)
        return hwc.unsqueeze(0).permute(0, 3, 1, 2)


def member_results_to_cpu(torch, results, finite_checks) -> np.ndarray:
    """One required blocking D2H, retaining the raw-logit finite guard.

    A shape or nonfinite-logit error is reported before member aggregation, just
    as in v1. The extra scalar is exactly 0 or 1 in FP32. Probability values are
    copied, not recomputed; CPU FP64 aggregation remains the caller's function.
    """
    means = torch.stack(results)
    valid = torch.stack(finite_checks).all().reshape(1).to(dtype=torch.float32)
    payload = torch.cat((means.reshape(-1), valid)).cpu().numpy()
    if payload[-1] != np.float32(1.0):
        raise FloatingPointError("Invalid single-image student logits")
    return payload[:-1].reshape(len(results), 10)


class CandidatePredictor:
    """Prototype wrapper sharing frozen models, with independently switchable parts."""

    def __init__(self, base, deploy_module, *, use_lut=True, merge_d2h=True,
                 pixel_loader: Callable | None = None):
        self.base = base
        self.deploy = deploy_module
        self.torch = base.torch
        self.device = base.device
        self.models = base.models
        self.members = base.members
        self.model_weights = base.model_weights
        self.config = base.config
        self.inventory = base.inventory
        self.use_lut, self.merge_d2h = use_lut, merge_d2h
        self.pixel_loader = pixel_loader or (
            lambda path, sizes: load_image_pixels(path, sizes, deploy_module._letterbox_image))
        self.input_lookup = (ExactU8Input(self.torch, self.device, deploy_module.MEAN, deploy_module.STD,
                                        image_sizes=[member["image_size"] for member in self.members])
                             if use_lut else None)

    def synchronize(self):
        self.base.synchronize()

    def predict_image(self, path):
        try:
            return self._predict_image(path)
        except BaseException as error:
            # An early shape/model exception can precede the normal final D2H.
            # Finish any queued H2D before the caller catches/retries and reuses
            # the pinned host buffer. There is no fence on the success path.
            if self.use_lut and self.input_lookup.staging:
                try:
                    self.synchronize()
                except Exception as sync_error:
                    error.add_note(f"Pending CUDA work also failed during staging cleanup: {sync_error}")
            raise

    def _predict_image(self, path):
        torch = self.torch
        sizes = [member["image_size"] for member in self.members]
        arrays = self.pixel_loader(path, sizes) if self.use_lut else self.deploy.load_image_arrays(path, sizes)
        results, finite_checks = [], []
        with torch.inference_mode():
            inputs = ({size: self.input_lookup.to_tensor(array) for size, array in arrays.items()}
                      if self.use_lut else {
                          size: torch.from_numpy(array).unsqueeze(0).to(
                              self.device, dtype=torch.float16, memory_format=torch.channels_last)
                          for size, array in arrays.items()})
            transformed_inputs = {(size, "identity"): image for size, image in inputs.items()}
            for member, model in zip(self.members, self.models):
                size = member["image_size"]
                views = ("identity", "hflip") if member["tta"] == "hflip2" else ("identity",)
                probabilities = []
                for view in views:
                    key = (size, view)
                    if key not in transformed_inputs:
                        transformed_inputs[key] = torch.flip(inputs[size], dims=(-1,)).contiguous(
                            memory_format=torch.channels_last)
                    logits = model(transformed_inputs[key])
                    if tuple(logits.shape) != (1, 10):
                        raise FloatingPointError("Invalid single-image student logits")
                    finite_checks.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                results.append(torch.stack(probabilities).mean(0))
            if self.merge_d2h:
                cpu_results = member_results_to_cpu(torch, results, finite_checks)
            else:
                if not bool(torch.stack(finite_checks).all()):
                    raise FloatingPointError("Invalid single-image student logits")
                cpu_results = torch.stack(results).cpu().numpy()
        return self.deploy.aggregate_member_probabilities(cpu_results, self.model_weights)
