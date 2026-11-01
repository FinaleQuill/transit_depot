"""EMA initialization rejects incompatible states and preserves resume ancestry."""
from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch import nn

from kd_common import TINY_BACKBONE
from study_student import StudyConfig
from train_distill import (DistillConfig, _configuration,
                           initialize_model_from_student_ema, resume_training_configuration)


def configuration():
    return DistillConfig(fold_plan=Path('/folds'), soft_labels=Path('/oof'),
                         output=Path('/new'), image_root=Path('/images'))


class StudentEmaWarmStartTests(unittest.TestCase):
    def test_exact_resume_and_ema_initialization_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            replace(configuration(), resume=Path('/new/last.pt'),
                    init_student_ema=Path('/old/ema.pt')).validate()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            replace(configuration(), init_student_ema_sha256='a' * 64).validate()

    def test_strict_load_retains_values_and_rejects_partial_or_wrong_shapes(self):
        model = nn.Linear(3, 2)
        source = {'weight': torch.full((2, 3), 7.), 'bias': torch.full((2,), -3.)}
        initialize_model_from_student_ema(model, source)
        self.assertTrue(torch.equal(model.weight.detach(), source['weight']))
        self.assertTrue(torch.equal(model.bias.detach(), source['bias']))
        with self.assertRaisesRegex(ValueError, 'names'):
            initialize_model_from_student_ema(model, {'weight': source['weight']})
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            initialize_model_from_student_ema(model, {**source, 'weight': torch.zeros(2, 4)})
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            initialize_model_from_student_ema(model, {k: v.half() for k, v in source.items()})

    def test_repeated_resume_without_original_snapshot_retains_lineage(self):
        initial = replace(configuration(), init_student_ema=Path('/no-longer-present/ema.pt'),
                          init_student_ema_sha256='a' * 64)
        lineage = {'kind': 'student_ema_warm_start', 'exact_resume': False,
            'source_snapshot_sha256': 'a' * 64, 'source_path': '/no-longer-present/ema.pt',
            'source_epoch': 9}
        current = replace(initial, init_student_ema=None, init_student_ema_sha256=None,
                          resume=Path('/new/last.pt'))
        for previous in [initial, current]:
            left, right = resume_training_configuration(current, {
                'training_config': _configuration(previous), 'student_warm_start': lineage})
            self.assertEqual(left, right)
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            resume_training_configuration(current, {'training_config': _configuration(current),
                'student_warm_start': {**lineage, 'source_snapshot_sha256': 'tampered'}})

    def test_tiny_study_recipe_has_explicit_registered_learning_rates_and_views(self):
        config = StudyConfig(fold_plan=Path('/folds'), soft_labels=Path('/oof'),
            output=Path('/new'), image_root=Path('/images'), backbone_name=TINY_BACKBONE,
            init_student_ema=Path('/old/ema.pt'), init_student_ema_sha256='a' * 64,
            epochs=4, patience=0, seed=46, backbone_lr=1e-5, head_lr=1e-4,
            drop_path=.1, alpha=.2, kd_warmup_epochs=0)
        config.validate()
        self.assertEqual(config.model_config()['backbone_name'], TINY_BACKBONE)
        self.assertIn('tiny', config.model_name())
        self.assertEqual(config.batch_size * config.accumulation_steps, 32)
        self.assertEqual(config.pos_weight_power, .5)


if __name__ == '__main__':
    unittest.main()
