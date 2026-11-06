"""Bounded-loader lifecycle and exception preservation without images or CUDA."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.perf_probe import probe


class FakeIterator:
    def __init__(self, fail_on_read=False):
        self.reads = 0
        self.shutdown_calls = 0
        self.fail_on_read = fail_on_read

    def __next__(self):
        if self.fail_on_read:
            raise RuntimeError("CUDA out of memory: synthetic original failure")
        self.reads += 1
        if self.reads > 1:
            raise StopIteration
        return {"images": torch.zeros(2, 3, 4, 4), "targets": torch.zeros(2, 10)}

    def _shutdown_workers(self):
        self.shutdown_calls += 1
        raise RuntimeError("synthetic cleanup failure")


class FakeLoader:
    def __init__(self, iterator):
        self.iterator = iterator
        self.dataset = SimpleNamespace(reseed=lambda seed: None)
        self.batch_sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    def __iter__(self):
        return self.iterator


class ProbeCleanupTests(unittest.TestCase):
    def arguments(self, manifest):
        return argparse.Namespace(mode="loader", seed=43, train_manifest=manifest,
            batch_size=2, num_workers=2, prefetch_factor=1, warmup_steps=0, steps=1,
            worker_timeout=1, image_root=manifest.parent, view_chunk_size=4,
            view_pad_multiple=0)

    def test_exact_bounded_loader_naturally_exhausts_without_manual_shutdown(self):
        with TemporaryDirectory() as folder:
            manifest = Path(folder) / "train.csv"
            manifest.write_text("synthetic-only\n")
            iterator = FakeIterator()
            result = {}
            with patch("dataset_highres.make_study_loader", return_value=FakeLoader(iterator)) as factory:
                with redirect_stdout(StringIO()), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA used")):
                    probe(self.arguments(manifest), result)
            self.assertEqual(factory.call_args.kwargs["batches_per_epoch"], 1)
            self.assertEqual(iterator.shutdown_calls, 0)
            self.assertEqual(iterator.reads, 2)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["loader_naturally_exhausted"])
            self.assertEqual(result["cleanup_errors"], [])

    def test_worker_cleanup_error_does_not_replace_original_oom(self):
        with TemporaryDirectory() as folder:
            manifest = Path(folder) / "train.csv"
            manifest.write_text("synthetic-only\n")
            iterator = FakeIterator(fail_on_read=True)
            result = {}
            with patch("dataset_highres.make_study_loader", return_value=FakeLoader(iterator)):
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory: synthetic original failure"):
                    probe(self.arguments(manifest), result)
            self.assertEqual(iterator.shutdown_calls, 1)
            self.assertEqual(result["cleanup_status"], "error")
            self.assertEqual(result["cleanup_errors"][0]["action"], "loader_shutdown_workers")
            self.assertTrue(result["train_manifest_unchanged"])


if __name__ == "__main__":
    unittest.main()
