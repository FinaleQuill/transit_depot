"""Study-weighted BCE plus a bounded training-only pairwise ranking queue.

Every queue row is one complete Study, never a view or patch. Queue entries
are detached historical predictions, not extra examples with backpropagation.
The caller enqueues training predictions explicitly after computing the loss;
validation must use the model directly and never this objective.
"""
from __future__ import annotations

import hashlib
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_CLASSES = 10


class SprintLoss(nn.Module):
    def __init__(self, pos_weight: Tensor, *, ranking_weight=0.15,
                 class1_weight=2.0, queue_capacity=512):
        super().__init__()
        weights = torch.as_tensor(pos_weight, dtype=torch.float32).detach().clone()
        if weights.shape != (NUM_CLASSES,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("pos_weight must contain ten finite positive train-only weights")
        if not math.isfinite(ranking_weight) or ranking_weight < 0:
            raise ValueError("ranking_weight must be finite and nonnegative")
        if not math.isfinite(class1_weight) or class1_weight <= 0:
            raise ValueError("class1_weight must be finite and positive")
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 512:
            raise ValueError("queue_capacity must be in [1, 512]")
        self.ranking_weight = float(ranking_weight)
        self.queue_capacity = queue_capacity
        class_weight = torch.ones(NUM_CLASSES, dtype=torch.float32)
        # Original label class 1 is zero-based output index 0.
        class_weight[0] = float(class1_weight)
        self.register_buffer("pos_weight", weights)
        self.register_buffer("class_weight", class_weight)
        self.register_buffer("queue_logits", torch.zeros(queue_capacity, NUM_CLASSES))
        self.register_buffer("queue_targets", torch.zeros(queue_capacity, NUM_CLASSES))
        self.register_buffer("queue_patients", torch.full((queue_capacity,), -1, dtype=torch.long))
        self.register_buffer("queue_size", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_cursor", torch.zeros((), dtype=torch.long))

    def _checked(self, logits, targets):
        if not self.training:
            raise RuntimeError("SprintLoss is training-only; do not pass validation examples")
        if logits.ndim == 1:
            logits, targets = logits.unsqueeze(0), targets.unsqueeze(0)
        if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES or logits.shape != targets.shape or not len(logits):
            raise ValueError("Expected one ten-label target per complete Study")
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("Nonfinite Study logits or labels")
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("Original binary Study labels are required")
        return logits.float(), targets.to(device=logits.device, dtype=torch.float32)

    def _patient_hashes(self, patient_ids, count, device):
        if patient_ids is None:
            # Only for synthetic mechanics tests; the training entrypoint must
            # supply original patient identities on every call.
            return torch.full((count,), -1, dtype=torch.long, device=device)
        if (isinstance(patient_ids, str) or len(patient_ids) != count
                or any(not isinstance(value, str) or not value for value in patient_ids)):
            raise ValueError("Provide one nonempty original patient ID per Study")
        hashes = [int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
                  & ((1 << 63) - 1) for value in patient_ids]
        return torch.tensor(hashes, dtype=torch.long, device=device)

    def forward(self, logits: Tensor, targets: Tensor, *, patient_ids=None, return_components=False):
        z, y = self._checked(logits, targets)
        patients = self._patient_hashes(patient_ids, len(z), z.device)
        if z.device != self.pos_weight.device:
            raise ValueError("Move SprintLoss to the same device as the model")
        with torch.autocast(device_type=z.device.type, enabled=False):
            per_class = F.binary_cross_entropy_with_logits(z, y, pos_weight=self.pos_weight, reduction="none")
            bce = ((per_class * self.class_weight).sum(-1) / self.class_weight.sum()).mean()
            count = int(self.queue_size.item())
            ranking = z.sum() * 0.0
            if count:
                historical_z = self.queue_logits[:count].detach()
                historical_y = self.queue_targets[:count].detach()
                # [current Study, historical Study, class]. Positive current
                # rows seek larger logits; negative current rows seek smaller.
                opposite = (y[:, None, :] != historical_y[None, :, :])
                same_patient = ((patients[:, None] == self.queue_patients[None, :count])
                                & (patients[:, None] >= 0))
                opposite = opposite & ~same_patient[:, :, None]
                direction = (2.0 * y - 1.0)[:, None, :]
                pair_losses = F.softplus(direction * (historical_z[None, :, :] - z[:, None, :]))
                pairs = opposite.sum((0, 1))
                eligible = pairs > 0
                class_ranking = (pair_losses * opposite).sum((0, 1)) / pairs.clamp_min(1)
                active_weights = self.class_weight * eligible
                ranking = (class_ranking * active_weights).sum() / active_weights.sum().clamp_min(1.0)
            total = bce + self.ranking_weight * ranking
        if return_components:
            return {"loss": total, "bce": bce, "ranking": ranking}
        return total

    @torch.no_grad()
    def enqueue(self, logits: Tensor, targets: Tensor, *, patient_ids=None):
        z, y = self._checked(logits, targets)
        z, y = z.detach().to(self.queue_logits), y.detach().to(self.queue_targets)
        patients = self._patient_hashes(patient_ids, len(z), self.queue_patients.device)
        if len(z) >= self.queue_capacity:
            self.queue_logits.copy_(z[-self.queue_capacity:])
            self.queue_targets.copy_(y[-self.queue_capacity:])
            self.queue_patients.copy_(patients[-self.queue_capacity:])
            self.queue_size.fill_(self.queue_capacity)
            self.queue_cursor.zero_()
            return
        cursor = int(self.queue_cursor.item())
        indices = (torch.arange(len(z), device=z.device) + cursor) % self.queue_capacity
        self.queue_logits[indices] = z
        self.queue_targets[indices] = y
        self.queue_patients[indices] = patients
        self.queue_cursor.fill_((cursor + len(z)) % self.queue_capacity)
        self.queue_size.fill_(min(self.queue_capacity, int(self.queue_size.item()) + len(z)))

    @torch.no_grad()
    def reset_queue(self):
        self.queue_logits.zero_()
        self.queue_targets.zero_()
        self.queue_patients.fill_(-1)
        self.queue_size.zero_()
        self.queue_cursor.zero_()
