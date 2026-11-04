"""Explicit ten-class Study weights over the existing train-only ranking loss."""
from __future__ import annotations

import torch

from sprint.loss import SprintLoss


DEFAULT_CLASS_WEIGHTS = (1.5, 1., 1., 1.25, 1., 1., 1., 2., 1., 1.)


class EnsembleLoss(SprintLoss):
    def __init__(self, pos_weight, *, class_weights10=DEFAULT_CLASS_WEIGHTS,
                 ranking_weight=.2, queue_capacity=512):
        weights = torch.as_tensor(class_weights10, dtype=torch.float32).detach().clone()
        if (weights.shape != (10,) or not bool(torch.isfinite(weights).all())
                or not bool((weights > 0).all())):
            raise ValueError("class_weights10 must contain ten finite positive class weights")
        super().__init__(pos_weight, ranking_weight=ranking_weight,
                         class1_weight=1., queue_capacity=queue_capacity)
        self.class_weight.copy_(weights)
