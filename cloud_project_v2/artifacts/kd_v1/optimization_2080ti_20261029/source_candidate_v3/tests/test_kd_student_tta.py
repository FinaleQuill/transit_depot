"""CPU tensor/protocol checks only; no network, images, backward or training."""
from dataclasses import asdict, replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from train_distill import (DistillConfig, distillation_losses, distillation_protocol,
                           mean_validation_probabilities, student_preprocessing_config,
                           validate_student_resume)
import test_kd_student as legacy_student_tests
import train_distill


def config(**kwargs):
    return DistillConfig(Path("fold.json"), Path("cache"), Path("student"), Path("images"), **kwargs)


class MultiViewLossTests(unittest.TestCase):
    def test_alignment_and_temperature_precede_mean_probability(self):
        s = torch.linspace(-3, 4, 20).reshape(2, 10)
        teacher = torch.stack((torch.linspace(-5, 2, 20).reshape(2, 10),
                               torch.linspace(1, 7, 20).reshape(2, 10)), dim=1)
        wt = torch.linspace(1, 4, 20).reshape(2, 10)
        ws = torch.linspace(2, 5, 10)
        _, actual = distillation_losses(s, torch.zeros_like(s), teacher, wt, ws, temperature=2)
        z = (teacher.double().numpy() - np.log(wt.double().numpy())[:, None, :]
             + np.log(ws.double().numpy())[None, None, :]) / 2
        q = (1 / (1 + np.exp(-z))).mean(axis=1)
        p = 1 / (1 + np.exp(-s.double().numpy() / 2))
        expected = (q * np.log(q / p) + (1-q) * np.log((1-q) / (1-p))).mean(1) * 4
        np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=2e-6)
        _, wrong_mean_logits = distillation_losses(s, torch.zeros_like(s), teacher.mean(1), wt, ws)
        self.assertGreater(float((actual - wrong_mean_logits).abs().max()), .01)

    def test_single_view_cache_is_equivalent_to_legacy_raw_cache(self):
        z = torch.linspace(-5, 5, 30).reshape(3, 10)
        weights = torch.linspace(1, 4, 30).reshape(3, 10)
        args = (torch.zeros_like(z), torch.ones_like(z))
        old = distillation_losses(*args, z, weights, torch.ones(10))
        new = distillation_losses(*args, z[:, None, :], weights, torch.ones(10))
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-6)

    def test_saturated_views_and_student_remain_finite(self):
        s = torch.full((2, 10), 1000.)
        t = torch.full((2, 2, 10), -1000.)
        hard, soft = distillation_losses(s, torch.zeros_like(s), t, torch.ones_like(s), torch.ones(10))
        self.assertTrue(bool(torch.isfinite(hard).all() and torch.isfinite(soft).all()))
        self.assertTrue(bool((soft > 0).all()))
        t[:, 1, :] = 1000
        _, balanced = distillation_losses(torch.zeros_like(s), s*0, t, torch.ones_like(s), torch.ones(10))
        torch.testing.assert_close(balanced, torch.zeros(2), atol=1e-7, rtol=0)

    def test_invalid_view_counts_and_nonfinite_values_are_rejected(self):
        z = torch.zeros(2, 10)
        for bad in (torch.zeros(2, 0, 10), torch.zeros(2, 3, 10), torch.zeros(2, 2, 9)):
            with self.assertRaises(ValueError):
                distillation_losses(z, z, bad, torch.ones_like(z), torch.ones(10))
        with self.assertRaises(FloatingPointError):
            distillation_losses(z, z, torch.full((2, 2, 10), float('nan')),
                                torch.ones_like(z), torch.ones(10))


class StudentProtocolTests(unittest.TestCase):
    def test_resolution_and_fixed_validation_views_are_declared(self):
        for size in (512, 768):
            c = config(image_size=size)
            c.validate()
            p = student_preprocessing_config(c)
            self.assertEqual(p['training_scales'], [size])
            self.assertEqual(p['image_size'], size)
            self.assertEqual(p['validation_size'], size)
            self.assertEqual(p['validation_tta'], 'hflip2')
        for bad in (320, 1024, True):
            with self.assertRaises(ValueError):
                config(image_size=bad).validate()
        with self.assertRaises(ValueError):
            config(validation_tta='random_flip').validate()
        with self.assertRaises(ValueError):
            config(batch_size=8, accumulation_steps=2).validate()

    def test_view_cache_temperature_remains_a_student_loss_parameter(self):
        cache = SimpleNamespace(logits=np.zeros((3, 2, 10), np.float32),
                                metadata={'tta': 'identity_hflip', 'views': ['identity', 'hflip']})
        p = distillation_protocol(config(image_size=768), cache, soft_hash='a'*64, metadata_hash='b'*64)
        self.assertEqual(p['temperature'], 2)
        self.assertEqual(p['teacher_views'], ['identity', 'hflip'])
        self.assertEqual(p['teacher_tta'], 'identity_hflip')
        self.assertEqual(p['student_training_size'], 768)
        self.assertEqual(p['validation_tta'], 'hflip2')
        self.assertFalse(p['deployment_ensemble'])
        self.assertEqual(p['soft_target_aggregation'],
                         'mean_sigmoid_after_pos_weight_alignment_and_temperature')
        with self.assertRaises(ValueError):
            distillation_protocol(config(validation_tta='identity'), cache,
                                  soft_hash='a'*64, metadata_hash='b'*64, legacy=True)

    def test_declared_views_must_match_cached_view_axis(self):
        cache = SimpleNamespace(logits=np.zeros((3, 1, 10), np.float32),
                                metadata={'tta': 'identity_hflip', 'views': ['identity', 'hflip']})
        with self.assertRaises(ValueError):
            distillation_protocol(config(), cache, soft_hash='a'*64, metadata_hash='b'*64)

    def test_tta_averages_probabilities_without_thresholding(self):
        z = torch.full((2, 10), -2.)
        flipped = torch.full((2, 10), 6.)
        actual = mean_validation_probabilities([z, flipped])
        expected = (torch.sigmoid(z) + torch.sigmoid(flipped)) / 2
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, torch.sigmoid((z+flipped)/2)))
        self.assertTrue(bool(((actual > 0) & (actual < 1)).all()))

    def test_schema7_resume_defaults_missing_new_fields_to_identity512(self):
        c = config(validation_tta='identity')
        serialized = {k: str(v) if isinstance(v, Path) else v for k,v in asdict(c).items()}
        serialized.pop('image_size'); serialized.pop('validation_tta')
        checkpoint = {'schema_version': 7, 'artifact_type': 'kd_student', 'training_config': serialized,
            'epoch': 1, 'best_epoch': 1, 'best_auc': .8, 'model_state_dict': {},
            'raw_model_state_dict': {}, 'optimizer_state_dict': {}, 'scaler_state_dict': {},
            'rng_state': {}, 'progress': {'completed_epochs': 1, 'validated_epochs': 1,
                'optimizer_updates': 2, 'ema_updates': 2},
            'early_stopping': {'reference_auc': .8, 'bad_epochs': 0},
            'inference_ready': True, 'validation_complete': True, 'weights_source': 'ema',
            'pretrained_provenance': legacy_student_tests.SourceTests().provenance()}
        expected = {'schema_version': 7, 'artifact_type': 'kd_student'}
        validate_student_resume(checkpoint, c, expected)
        with self.assertRaises(ValueError):
            validate_student_resume(checkpoint, replace(c, validation_tta='hflip2'), expected)
        with self.assertRaises(ValueError):
            validate_student_resume(checkpoint, replace(c, image_size=768), expected)

    def test_execute_rejects_image_output_overlap_before_logger_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root/'images'
            images.mkdir()
            alias = root/'image_alias'
            alias.symlink_to(images, target_is_directory=True)
            for output in (images, images/'new_student', root, alias/'new_student'):
                for marker in ('0', '1'):
                    with self.subTest(output=output, recorder_marker=marker):
                        logged = Mock(side_effect=AssertionError('Recorder must not write'))
                        trained = Mock(side_effect=AssertionError('Training must not run'))
                        argv = ['--execute', '--fold-plan', 'unused_folds.json',
                                '--soft-labels', 'unused_cache', '--output', str(output),
                                '--image-root', str(images)]
                        with (patch.dict(os.environ, {'KD_TRAINING_LOG_ACTIVE': marker}),
                              patch.dict('sys.modules', {'training_evidence_monitor': SimpleNamespace(run_logged=logged)}),
                              patch.object(train_distill, 'run_training', trained)):
                            with self.assertRaisesRegex(ValueError, 'read-only image root'):
                                train_distill.main(argv)
                        logged.assert_not_called()
                        trained.assert_not_called()
            self.assertEqual(list(images.iterdir()), [])
            self.assertFalse((root/'evidence').exists())


if __name__ == '__main__':
    unittest.main()
