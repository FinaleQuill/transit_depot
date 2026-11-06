"""Exercise the real trainer on CPU with only its CUDA transport mocked out."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from reconstruction import train
from reconstruction.model import StudyModel, mean_probability_logits, study_bce_loss
from reconstruction.test_model import DummyBackbone


class SyntheticStudies:
    def __init__(self):
        self.batches, self.groups = [], []
        offset = 0
        for index, views in enumerate((1, 3, 2, 4, 1)):
            self.groups.append(list(range(offset, offset + views)))
            offset += views
            self.batches.append({"images": torch.randn(views, 3),
                "targets": ((torch.arange(10) + index) % 3 == 0).float(),
                "study_key": f"patient{index}/study{index}",
                "image_ids": [f"{index}_{view}" for view in range(views)]})

    def __len__(self):
        return len(self.groups)

    def set_epoch(self, epoch):
        self.epoch = epoch


class LinearStudy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 10)

    def forward(self, images, return_components=False):
        logits = mean_probability_logits(self.fc(images))
        return {"study_logits": logits, "global_study_logits": logits} if return_components else logits

    def forward_studies(self, images_list, return_components=False):
        logits = self.fc(torch.cat(images_list))
        output, offset = [], 0
        for images in images_list:
            pooled = mean_probability_logits(logits[offset:offset + len(images)])
            output.append({"study_logits": pooled, "global_study_logits": pooled} if return_components else pooled)
            offset += len(images)
        return output


class SkipFirstScaler:
    """One synthetic overflow; matches the scale-change signal used by GradScaler."""
    def __init__(self):
        self.updates = 0
        self.current_scale = 8.

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        if self.updates:
            optimizer.step()

    def update(self):
        if not self.updates:
            self.current_scale /= 2
        self.updates += 1


class TrainerSemanticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)
        self.dataset = SyntheticStudies()
        self.initial = LinearStudy()
        self.weights = torch.linspace(1., 2., 10)
        self.config = {"workers": 0, "seed": 41, "effective_studies": 3,
            "epochs": 4, "global_aux_weight": 0., "clip_grad": 1.e6, "ema_decay": .999}

    def optimizer(self, model):
        return torch.optim.SGD([{"params": list(model.parameters()), "lr": .1, "base_lr": .1}])

    def run_epoch_cpu(self, model, ema, optimizer, progress, epoch, scaler=None):
        # This runs train_epoch itself: the reductions, clipping, scheduler,
        # optimizer/EMA, coverage guards, and progress serialization are real.
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == "cuda":
                args = ("cpu", *args[1:])
            if kwargs.get("device") == "cuda":
                kwargs = {**kwargs, "device": "cpu"}
            return original_to(tensor, *args, **kwargs)
        if scaler is None:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(train, "make_loader", return_value=self.dataset.batches))
            stack.enter_context(patch.object(torch.Tensor, "to", cpu_to))
            stack.enter_context(patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()))
            stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats", return_value=None))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            return train.train_epoch(model, ema, self.dataset, optimizer, scaler, self.weights,
                self.config, epoch, progress, Path(directory))

    @staticmethod
    def progress():
        return {"attempted_updates": 0, "optimizer_updates": 0, "skipped_updates": 0}

    def reference_window(self, model, optimizer, batches, lr):
        optimizer.zero_grad(set_to_none=True)
        losses = [study_bce_loss(model(b["images"]), b["targets"], self.weights) for b in batches]
        torch.stack(losses).mean().backward()
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()

    def test_actual_epoch_full_and_tail_match_equal_study_batch_gradients(self):
        actual, reference = deepcopy(self.initial), deepcopy(self.initial)
        ema = deepcopy(actual).requires_grad_(False)
        progress = self.progress()
        result = self.run_epoch_cpu(actual, ema, self.optimizer(actual), progress, epoch=1)
        reference_optimizer = self.optimizer(reference)
        # A batch of three Studies, followed by a batch of TWO, irrespective of
        # their different image counts. The first epoch is two warmup updates.
        self.reference_window(reference, reference_optimizer, self.dataset.batches[:3], .05)
        self.reference_window(reference, reference_optimizer, self.dataset.batches[3:], .1)
        for left, right in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
        self.assertEqual(progress, {"attempted_updates": 2, "optimizer_updates": 2, "skipped_updates": 0})
        self.assertEqual((result["studies"], result["images"]), (5, 11))

    def test_amp_skip_does_not_update_ema_or_leak_window_gradients(self):
        actual, reference = deepcopy(self.initial), deepcopy(self.initial)
        ema = deepcopy(actual).requires_grad_(False)
        initial_ema = deepcopy(ema)
        progress = self.progress()
        self.run_epoch_cpu(actual, ema, self.optimizer(actual), progress, epoch=1, scaler=SkipFirstScaler())
        # First window was discarded. Attempted-update schedule still advances,
        # so the final two Studies use the second warmup LR, with no stale grads.
        self.reference_window(reference, self.optimizer(reference), self.dataset.batches[3:], .1)
        self.assertEqual(progress, {"attempted_updates": 2, "optimizer_updates": 1, "skipped_updates": 1})
        for left, right in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
        for observed, before, updated in zip(ema.parameters(), initial_ema.parameters(), reference.parameters()):
            expected = before * (2 / 11) + updated * (9 / 11)
            torch.testing.assert_close(observed, expected, atol=2e-7, rtol=2e-6)

    def test_packed_training_matches_single_studies_full_and_tail_updates(self):
        ordinary, packed = deepcopy(self.initial), deepcopy(self.initial)
        ema_ordinary, ema_packed = deepcopy(ordinary).requires_grad_(False), deepcopy(packed).requires_grad_(False)
        first_progress, packed_progress = self.progress(), self.progress()
        ordinary_result = self.run_epoch_cpu(ordinary, ema_ordinary, self.optimizer(ordinary), first_progress, 1)
        self.config["pack_images"] = 4
        packed_result = self.run_epoch_cpu(packed, ema_packed, self.optimizer(packed), packed_progress, 1)
        for left, right in zip(ordinary.parameters(), packed.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
        for left, right in zip(ema_ordinary.parameters(), ema_packed.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
        self.assertEqual(first_progress, packed_progress)
        self.assertEqual(packed_result["studies"], ordinary_result["studies"])
        self.assertEqual(packed_result["images"], 11)

    def test_packs_do_not_cross_update_windows_or_split_oversized_study(self):
        # The four-image Study is larger than the preferred cap of three;
        # it must still be emitted intact, with its final image retained.
        packs = list(train.study_packs(self.dataset.batches, max_images=3, effective_studies=3))
        collected, boundaries = [], []
        for batches, last_index, boundary in packs:
            ids = [int(b["study_key"].split("/")[0].removeprefix("patient")) for b in batches]
            self.assertEqual(len({index // 3 for index in ids}), 1)
            self.assertEqual(last_index, ids[-1])
            collected.extend(ids)
            if boundary:
                boundaries.append(last_index)
            if 3 in ids:
                self.assertEqual(len(batches), 1)
                self.assertEqual(len(batches[0]["image_ids"]), 4)
        self.assertEqual(collected, [0, 1, 2, 3, 4])
        self.assertEqual(boundaries, [2, 4])

    def test_epoch_boundary_restore_preserves_optimizer_ema_and_lr_progression(self):
        continuous = deepcopy(self.initial)
        ema_continuous = deepcopy(continuous).requires_grad_(False)
        optimizer_continuous = self.optimizer(continuous)
        progress_continuous = self.progress()
        self.run_epoch_cpu(continuous, ema_continuous, optimizer_continuous, progress_continuous, epoch=1)
        saved = deepcopy({"model": continuous.state_dict(), "ema": ema_continuous.state_dict(),
                          "optimizer": optimizer_continuous.state_dict(), "progress": progress_continuous})
        self.run_epoch_cpu(continuous, ema_continuous, optimizer_continuous, progress_continuous, epoch=2)
        resumed, ema_resumed = deepcopy(self.initial), deepcopy(self.initial).requires_grad_(False)
        optimizer_resumed = self.optimizer(resumed)
        resumed.load_state_dict(saved["model"])
        ema_resumed.load_state_dict(saved["ema"])
        optimizer_resumed.load_state_dict(saved["optimizer"])
        self.run_epoch_cpu(resumed, ema_resumed, optimizer_resumed, saved["progress"], epoch=2)
        for left, right in zip(continuous.parameters(), resumed.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for left, right in zip(ema_continuous.parameters(), ema_resumed.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(saved["progress"], progress_continuous)
        expected_last_lr = .1 * (.1 + .9 * .5 * (1 + math.cos(math.pi / 6)))
        self.assertAlmostEqual(optimizer_resumed.param_groups[0]["lr"], expected_last_lr)

    def test_ema_interpolates_floating_state_and_copies_integer_buffers(self):
        raw = nn.Module()
        raw.weight = nn.Parameter(torch.tensor([3.]))
        raw.register_buffer("counter", torch.tensor(8))
        ema = deepcopy(raw).requires_grad_(False)
        ema.weight.fill_(1.)
        ema.counter.fill_(2)
        actual_decay = train.update_ema(ema, raw, updates=5)
        self.assertEqual(actual_decay, .4)
        torch.testing.assert_close(ema.weight, torch.tensor([2.2]))
        self.assertEqual(int(ema.counter), 8)
        self.assertIsNone(raw.weight.grad)

    def test_optimizer_roles_and_decay_keep_classifier_at_head_rate(self):
        model = StudyModel.from_backbone(DummyBackbone(), aggregation="attention", attention_dim=12)
        config = {"backbone_lr": 2e-5, "head_lr": 2e-4, "weight_decay": .05}
        groups = train.parameter_groups(model, config)
        assignments = {}
        for group in groups:
            for parameter in group["params"]:
                self.assertNotIn(id(parameter), assignments)
                assignments[id(parameter)] = group
        self.assertEqual(len(assignments), len(list(model.parameters())))
        self.assertEqual(assignments[id(model.backbone.fc.weight)]["lr"], 2e-4)
        self.assertEqual(assignments[id(model.backbone.fc.bias)]["weight_decay"], 0.)
        self.assertEqual(assignments[id(model.backbone.norm.weight)]["lr"], 2e-5)
        self.assertEqual(assignments[id(model.backbone.norm.weight)]["weight_decay"], 0.)
        self.assertEqual(assignments[id(model.view_residual_weight)]["group_name"], "head")


if __name__ == "__main__":
    unittest.main()
