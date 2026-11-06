"""CPU-only optimizer grouping checks for the spatial-MIL extension."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from exploration_train import build_mil_optimizer


class ModelWithSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Module()
        self.head.fc = nn.Linear(4, 10)
        self.head.attention = nn.Conv2d(4, 10, 1, bias=False)

    def get_classifier(self):
        return self.head.fc


class ExplorationTrainTests(unittest.TestCase):
    def test_attention_uses_head_lr_and_head_weight_decay(self):
        model = ModelWithSpatialAttention()
        config = SimpleNamespace(backbone_lr=1e-5, head_lr=1e-4, weight_decay=.05)
        optimizer = build_mil_optimizer(model, config, fused=False)
        attention = model.head.attention.weight
        fc_weight = model.head.fc.weight
        backbone_weight = model.backbone.weight

        groups = {id(parameter): group for group in optimizer.param_groups
                  for parameter in group['params']}
        attention_group = groups[id(attention)]
        self.assertIs(attention_group, groups[id(fc_weight)])
        self.assertEqual(attention_group['role'], 'head')
        self.assertEqual(attention_group['lr'], config.head_lr)
        self.assertEqual(attention_group['weight_decay'], config.weight_decay)
        self.assertEqual(groups[id(backbone_weight)]['role'], 'backbone')
        self.assertEqual(groups[id(backbone_weight)]['lr'], config.backbone_lr)
        self.assertEqual(sum(id(p) == id(attention) for group in optimizer.param_groups
                             for p in group['params']), 1)

    def test_missing_spatial_attention_is_rejected(self):
        model = nn.Linear(4, 10)
        model.get_classifier = lambda: model
        config = SimpleNamespace(backbone_lr=1e-5, head_lr=1e-4, weight_decay=.05)
        with self.assertRaisesRegex(ValueError, 'no spatial attention'):
            build_mil_optimizer(model, config, fused=False)


if __name__ == '__main__':
    unittest.main()
