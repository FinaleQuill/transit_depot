"""Small CPU parity checks for exact lookup inputs and raw-logit guards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from inference_inputs import (ExactU8Input, load_image_pixels, member_results_to_cpu,
                              normalization_table, normalize_pixels)


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def original_normalize(pixels):
    rgb = np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else pixels
    values = np.asarray(rgb, dtype=np.float32)
    offset = np.asarray(MEAN, dtype=np.float32) * np.float32(255.0)
    denominator = np.asarray(STD, dtype=np.float32) * np.float32(255.0)
    return np.ascontiguousarray(((values - offset) / denominator).transpose(2, 0, 1))


def letterbox(image, size, mode):
    scale = size / max(image.size)
    shape = tuple(max(1, min(size, round(length * scale))) for length in image.size)
    resized = image.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new(mode, (size, size), 0)
    canvas.paste(resized, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return canvas


class InferenceInputLookupTests(unittest.TestCase):
    def test_every_byte_retains_float32_operation_rounding(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        table = normalization_table(MEAN, STD)
        for pixels in (gray, np.stack((gray, gray[::-1], 255 - gray), axis=-1)):
            expected = original_normalize(pixels)
            self.assertEqual(normalize_pixels(pixels, MEAN, STD).tobytes(), expected.tobytes())
            for channel in range(3):
                values = pixels if pixels.ndim == 2 else pixels[..., channel]
                self.assertEqual(table[channel][values].tobytes(), expected[channel].tobytes())

    def test_cpu_lookup_retains_fp16_bits_and_channels_last(self):
        import torch

        lookup = ExactU8Input(torch, "cpu", MEAN, STD, image_sizes=[16])
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for pixels in (gray, np.stack((gray, gray[::-1], 255 - gray), axis=-1)):
            expected = torch.from_numpy(original_normalize(pixels)).unsqueeze(0).to(
                dtype=torch.float16, memory_format=torch.channels_last)
            actual = lookup.to_tensor(pixels)
            self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(actual.dtype, torch.float16)
            self.assertEqual(actual.numpy().tobytes(), expected.numpy().tobytes())
        self.assertEqual(lookup.staging, {})
        with patch.object(torch.cuda, "synchronize") as sync:
            lookup.sync_after_error()
            sync.assert_not_called()

    def test_packed_result_keeps_all_twenty_probabilities(self):
        import torch

        results = [torch.linspace(0, 1, 10), torch.linspace(1, 0, 10)]
        expected = torch.stack(results).numpy()
        actual = member_results_to_cpu(torch, results, [torch.tensor(True)] * 4)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_guard_checks_logits_even_when_sigmoid_hides_infinity(self):
        import torch

        for value in (float("inf"), float("-inf"), float("nan")):
            logits = torch.full((1, 10), value)
            finite = torch.isfinite(logits).all()
            means = [torch.sigmoid(logits.float())[0]] * 2
            with self.subTest(raw_logit=value), self.assertRaisesRegex(FloatingPointError, "student logits"):
                member_results_to_cpu(torch, means, [torch.tensor(True), finite])

    def test_pixels_keep_exif_color_geometry_and_unique_sizes(self):
        gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            image = Image.fromarray(gray)
            exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)
            before = path.read_bytes()
            actual = load_image_pixels(path, [16, 8, 16], letterbox)
            with Image.open(path) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                expected = {size: np.asarray(letterbox(upright, size, "L")) for size in (8, 16)}
            self.assertEqual(list(actual), [8, 16])
            for size in expected:
                self.assertEqual(actual[size].dtype, np.uint8)
                self.assertEqual(actual[size].tobytes(), expected[size].tobytes())
            self.assertEqual(path.read_bytes(), before)

    def test_non_eight_bit_images_keep_explicit_policy_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode, dtype in (("I", np.int32), ("F", np.float32), ("I;16", np.uint16)):
                path = Path(temporary) / "image.tiff"
                Image.fromarray(np.arange(35, dtype=dtype).reshape(5, 7), mode).save(path)
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "explicit intensity policy"):
                    load_image_pixels(path, [16], letterbox)


if __name__ == "__main__":
    unittest.main()
