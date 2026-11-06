"""CPU-only semantics checks for opt-in view padding; no images/GPU are used."""
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_highres import HighResConfig, checkpointed_view_forward


class SampleIndependentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Linear(4, 10)
        self.forward_sizes = []

    def forward(self, pixels):
        self.forward_sizes.append(int(pixels.shape[0]))
        features = torch.nn.functional.gelu(self.conv(pixels)).mean(dim=(2, 3))
        return self.head(features)


class ViewPaddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        # The real function requests CUDA autocast, while this isolated test
        # exercises its CPU checkpoint/slicing graph with double precision.
        self.autocast_patch = patch("train_highres.torch.autocast", side_effect=lambda *a, **k: nullcontext())
        self.autocast_patch.start()
        self.cuda_patch = patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test touched CUDA"))
        self.cuda_patch.start()

    def tearDown(self):
        self.cuda_patch.stop()
        self.autocast_patch.stop()

    def compare(self, count, multiple):
        source = torch.randn(count, 3, 5, 6, dtype=torch.float64)
        before = source.clone()
        target = torch.randn(count, 10, dtype=torch.float64)
        baseline = SampleIndependentModel().double()
        padded = copy.deepcopy(baseline)
        anchor = torch.empty(0)
        outputs = []
        for model, pad in ((baseline, 0), (padded, multiple)):
            logits = checkpointed_view_forward(model, source, torch.device("cpu"), anchor,
                                               chunk_size=4, pad_multiple=pad)
            self.assertEqual(tuple(logits.shape), (count, 10))
            ((logits - target).square().sum()).backward()
            outputs.append(logits.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-12, atol=1e-12)
        for left, right in zip(baseline.parameters(), padded.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(source, before, rtol=0, atol=0)
        expected_tail = count % 4
        expected_sizes = {4}
        if expected_tail:
            expected_sizes.add(((expected_tail + multiple - 1) // multiple) * multiple)
        self.assertEqual(set(padded.forward_sizes), expected_sizes)

    def test_partial_chunk_outputs_and_parameter_gradients_unchanged(self):
        for count, multiple in ((5, 4), (7, 2), (7, 4), (8, 4)):
            with self.subTest(count=count, multiple=multiple):
                self.compare(count, multiple)

    def test_default_matches_original_unpadded_chunks(self):
        source = torch.randn(7, 3, 5, 6, dtype=torch.float64)
        model = SampleIndependentModel().double()
        reference = copy.deepcopy(model)
        actual = checkpointed_view_forward(model, source, torch.device("cpu"), torch.empty(0), chunk_size=4)
        expected = torch.cat((reference(source[:4]), reference(source[4:])))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.forward_sizes, [4, 3])

    def test_invalid_multiples_rejected_before_model_execution(self):
        model = SampleIndependentModel().double()
        source = torch.zeros(3, 3, 5, 6, dtype=torch.float64)
        for multiple in (-1, 3, 8, True, 2.0):
            with self.subTest(multiple=multiple):
                with self.assertRaises(ValueError):
                    checkpointed_view_forward(model, source, torch.device("cpu"), torch.empty(0),
                                               chunk_size=4, pad_multiple=multiple)
                with self.assertRaises(ValueError):
                    HighResConfig(output=Path("unused"), view_chunk_size=4,
                                  view_pad_multiple=multiple).validate()
        self.assertEqual(model.forward_sizes, [])


if __name__ == "__main__":
    unittest.main()
