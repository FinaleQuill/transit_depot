"""Exact input parity and per-image TTA sharing, using synthetic CPU fixtures only."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

import deploy_kd


def original_image_arrays(path, image_sizes):
    """Frozen pre-optimization pipeline; do not call the candidate's image helpers."""
    with Image.open(path) as source:
        source.load()
        upright = ImageOps.exif_transpose(source)
        if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
            raise ValueError("Non-8-bit image requires an explicit intensity policy")
        rgb = upright.convert("RGB")
        result = {}
        for image_size in sorted(set(image_sizes)):
            scale = image_size / max(rgb.size)
            size = tuple(max(1, min(image_size, round(length * scale))) for length in rgb.size)
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
            canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
            pixels = np.asarray(canvas, dtype=np.float32)
            denominator = np.asarray(deploy_kd.STD, dtype=np.float32) * np.float32(255.0)
            pixels = (pixels - np.asarray(deploy_kd.MEAN, dtype=np.float32) * np.float32(255.0)) / denominator
            result[image_size] = np.ascontiguousarray(pixels.transpose(2, 0, 1))
        return result


class FastInferenceInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fast-inference-inputs-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assert_original_parity(self, path):
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        expected = original_image_arrays(path, [768, 512, 768])
        actual = deploy_kd.load_image_arrays(path, [768, 512, 768])
        self.assertEqual(list(actual), [512, 768])
        for size in expected:
            self.assertEqual(actual[size].dtype, np.float32)
            self.assertTrue(actual[size].flags.c_contiguous)
            self.assertEqual(actual[size].shape, (3, size, size))
            np.testing.assert_array_equal(actual[size], expected[size])
            self.assertEqual(actual[size].tobytes(), expected[size].tobytes())
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    @staticmethod
    def fixture(mode, shape):
        height, width = shape
        y, x = np.indices(shape)
        gray = ((x * 79 + y * 53 + (x * y) % 251) % 256).astype(np.uint8)
        rgb = np.stack([gray, np.roll(gray, 1, 0), 255 - gray], axis=-1)
        if mode == "L":
            return Image.fromarray(gray)
        if mode == "RGB":
            return Image.fromarray(rgb)
        if mode == "1":
            return Image.fromarray(gray).convert("1", dither=Image.Dither.NONE)
        if mode == "P":
            image = Image.fromarray(gray).convert("P")
            image.putpalette([value for i in range(256) for value in (i, 255 - i, i // 2)])
            image.info["transparency"] = 17
            return image
        if mode == "LA":
            return Image.fromarray(np.stack([gray, 255 - gray], axis=-1), "LA")
        return Image.fromarray(np.concatenate([rgb, gray[..., None]], axis=-1), "RGBA")

    def test_modes_exif_and_odd_letterbox_geometry_are_byte_identical(self):
        # Opposite aspect ratios, odd padding, extreme aspect ratio and downsampling.
        shapes = [(37, 61), (61, 37), (1, 1027), (519, 1031)]
        for mode in ("L", "RGB", "P", "1", "LA", "RGBA"):
            for shape in shapes:
                for orientation in range(1, 9):
                    with self.subTest(mode=mode, shape=shape, orientation=orientation):
                        image = self.fixture(mode, shape)
                        exif = Image.Exif()
                        exif[274] = orientation
                        path = self.root / "fixture.png"
                        image.save(path, exif=exif)
                        self.assert_original_parity(path)

    def test_jpeg_l_and_rgb_decode_and_exif_are_byte_identical(self):
        for mode in ("L", "RGB"):
            for orientation in range(1, 9):
                with self.subTest(mode=mode, orientation=orientation):
                    exif = Image.Exif()
                    exif[274] = orientation
                    path = self.root / "fixture.jpg"
                    self.fixture(mode, (1079, 1297)).save(path, quality=91, exif=exif)
                    self.assert_original_parity(path)

    def test_non_eight_bit_modes_keep_the_original_rejection_and_file(self):
        for mode, dtype in (("I", np.int32), ("F", np.float32), ("I;16", np.uint16)):
            with self.subTest(mode=mode):
                path = self.root / "fixture.tiff"
                Image.fromarray(np.arange(35, dtype=dtype).reshape(5, 7), mode).save(path)
                before = (path.read_bytes(), path.stat().st_mtime_ns)
                for loader in (original_image_arrays, deploy_kd.load_image_arrays):
                    with self.assertRaisesRegex(ValueError, "explicit intensity policy"):
                        loader(path, [512, 768])
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_same_resolution_tta_shares_inputs_but_keeps_four_forwards_and_arithmetic(self):
        self.check_tta_sharing([768, 768])

    def test_different_resolutions_transform_each_size_once(self):
        self.check_tta_sharing([512, 768])

    def check_tta_sharing(self, sizes):
        import torch

        class FakeModel:
            def __init__(self, index):
                self.index, self.inputs = index, []
                self.outputs = []

            def __call__(self, tensor):
                self.inputs.append(tensor)
                # Distinct FP16 logits per model/view catch changes in TTA arithmetic.
                logits = (torch.arange(10, dtype=torch.float16).reshape(1, 10) * .37
                          + self.index * .61 + (len(self.inputs) % 2) * .29 - 3)
                self.outputs.append(logits)
                return logits

        path = self.root / "fixture.png"
        self.fixture("L", (17, 29)).save(path)
        predictor = deploy_kd.Predictor.__new__(deploy_kd.Predictor)
        predictor.torch = torch
        predictor.device = torch.device("cpu")
        predictor.members = [{"image_size": size, "tta": "hflip2"} for size in sizes]
        predictor.models = [FakeModel(0), FakeModel(1)]
        predictor.model_weights = [[.25] * 10, [.75] * 10]
        # All tensors remain CPU tensors; Predictor.__init__ and CUDA are never called.
        with patch.object(torch, "flip", wraps=torch.flip) as flipped, \
                patch.object(torch, "from_numpy", wraps=torch.from_numpy) as converted, \
                patch("deploy_kd.Image.open", wraps=Image.open) as opened:
            actual = predictor.predict_image(path)
            first_inputs = [list(model.inputs) for model in predictor.models]
            self.assertEqual(flipped.call_count, len(set(sizes)))
            self.assertEqual(converted.call_count, len(set(sizes)))
            self.assertEqual(opened.call_count, 1)
            second = predictor.predict_image(path)
            self.assertEqual(flipped.call_count, 2 * len(set(sizes)))
            self.assertEqual(converted.call_count, 2 * len(set(sizes)))
            self.assertEqual(opened.call_count, 2)
        expected_members = [torch.stack([torch.sigmoid(x.float())[0] for x in model.outputs[:2]])
                            .mean(0).cpu().numpy() for model in predictor.models]
        expected = deploy_kd.aggregate_member_probabilities(expected_members, predictor.model_weights)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second, expected)
        for index, model in enumerate(predictor.models):
            self.assertEqual(len(model.inputs), 4)  # Two views on each of two calls.
            identity, hflip = first_inputs[index]
            self.assertEqual(identity.device.type, "cpu")
            self.assertEqual(identity.dtype, torch.float16)
            self.assertTrue(identity.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(hflip.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.equal(hflip, torch.flip(identity, dims=(-1,))))
            self.assertIsNot(identity, model.inputs[2])
            self.assertIsNot(hflip, model.inputs[3])
        if sizes[0] == sizes[1]:
            for view in range(2):
                self.assertIs(first_inputs[0][view], first_inputs[1][view])
        self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
