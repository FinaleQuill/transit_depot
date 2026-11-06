"""Independent BCE controls, source metadata and CLI; CPU tensors/mocks only."""
from dataclasses import replace
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

import train_distill as student
import test_kd_student as legacy_student_tests


def configuration(**changes):
    return replace(student.DistillConfig(Path('fold.json'), None, Path('out'), Path('images'),
                   training_mode='supervised_bce', alpha=0.), **changes)


class IndependentObjectiveTests(unittest.TestCase):
    def test_mode_is_explicit_and_teacher_inputs_are_rejected(self):
        configuration().validate()
        for changed in ({'alpha': .2}, {'soft_labels': Path('unused_cache')},
                        {'training_mode': 'distillation'}, {'training_mode': 'unknown'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                configuration(**changed).validate()
        kd = configuration(training_mode='distillation', soft_labels=Path('cache'), alpha=.2)
        kd.validate()
        self.assertEqual(kd.batch_size * kd.accumulation_steps, 32)
        self.assertEqual(kd.model_config(), configuration().model_config())

    def test_preparation_never_opens_cache_or_hashes_teacher_inputs(self):
        with (patch.object(student, '_load_soft_cache', side_effect=AssertionError('No cache')),
              patch.object(student, 'sha256_file', side_effect=AssertionError('No teacher hash'))):
            objective = student.prepare_student_objective(configuration(), [])
        self.assertIsNone(objective['cache'])
        self.assertEqual(objective['schema_version'], 9)
        self.assertEqual(objective['teacher_provenance'], [])
        self.assertEqual(objective['cache_bindings'], [])
        self.assertEqual(objective['protocol'], {
            'enabled': False, 'alpha': 0., 'loss': 'weighted_bce', 'class_reduction': 'mean',
            'study_equal_weighting': True, 'student_training_size': 512,
            'validation_tta': 'hflip2', 'deployment_tta': 'hflip2', 'deployment_ensemble': False})
        self.assertNotIn('temperature', objective['protocol'])
        self.assertNotIn('soft_labels_sha256', objective['protocol'])

    def test_supervised_loss_is_full_weighted_bce_without_any_kl(self):
        logits = torch.linspace(-3, 3, 20).reshape(2, 10).requires_grad_()
        targets = (torch.arange(20).reshape(2, 10) % 2).float()
        weights = torch.linspace(1, 10, 10)
        with patch.object(student, 'distillation_losses', side_effect=AssertionError('No KL')):
            hard, soft = student.student_batch_losses(configuration(), logits, targets, weights,
                                                      image_ids=['a', 'b'], cache=None)
        reference = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights,
                                                       reduction='none').mean(1)
        torch.testing.assert_close(hard, reference)
        torch.testing.assert_close(soft, torch.zeros(2))
        actual_gradient, = torch.autograd.grad(hard.sum(), logits)
        expected_gradient = ((1 - targets) * logits.detach().sigmoid()
                             + targets * weights * (logits.detach().sigmoid() - 1)) / 10
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertFalse(torch.allclose(hard, .8 * reference))
        with self.assertRaises(ValueError):
            student.student_batch_losses(configuration(), logits, targets, weights,
                                          image_ids=['a', 'b'], cache=object())

    def test_invalid_supervised_arrays_are_rejected(self):
        z, w = torch.zeros(2, 10), torch.ones(10)
        for logits, target, weights, exception in (
                (z[:, :9], z, w, ValueError), (z, z, torch.zeros(10), ValueError),
                (z, z+2, w, ValueError), (z*float('nan'), z, w, FloatingPointError)):
            with self.subTest(exception=exception), self.assertRaises(exception):
                student.supervised_bce_loss(logits, target, weights)

    def test_training_preflight_reaches_cuda_guard_without_teacher_cache(self):
        train = [{'subject_id': 'train', 'study_key': 'train/a', 'image_id': 'a',
                  'file_sha256': 'a'*64, 'pixel_sha256': 'a'*64}]
        valid = [{'subject_id': 'valid', 'study_key': 'valid/b', 'image_id': 'b',
                  'file_sha256': 'b'*64, 'pixel_sha256': 'b'*64}]
        plan = {'source_train_manifest': {'path': 'train.csv'}, 'source_manifest_sha256': 'a'*64,
                'official_holdout': {'path': 'valid.csv', 'sha256': 'b'*64}}
        with (patch.object(student, 'require_training_runtime'),
              patch.object(student, 'capture_source_version', return_value={}),
              patch.object(student, 'sha256_file', return_value='c'*64) as hashes,
              patch.object(student, '_load_soft_cache', side_effect=AssertionError('No teacher cache')),
              patch('cv_splits.read_fold_plan', return_value=plan),
              patch('dataset.read_manifest', side_effect=[train, valid]),
              patch('train_overnight.compute_loss_config', return_value={'pos_weight': [1.]*10}),
              patch.object(torch.cuda, 'is_available', return_value=False),
              patch('model_overnight.build_overnight_model', side_effect=AssertionError('No model')),
              patch.object(student, 'prepare_pretrained_provenance', side_effect=AssertionError('No download'))):
            with self.assertRaisesRegex(RuntimeError, 'requires a CUDA GPU'):
                student.run_training(configuration())
        hashes.assert_called_once_with(Path('fold.json'))


class SupervisedResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.checkpoint = {'schema_version': 9, 'artifact_type': 'kd_student',
            'training_mode': 'supervised_bce', 'teacher_provenance': [],
            'distillation': student.supervised_protocol(self.config),
            'training_config': student._configuration(self.config), 'epoch': 2, 'best_epoch': 1,
            'best_auc': .8, 'model_state_dict': {}, 'raw_model_state_dict': {},
            'optimizer_state_dict': {}, 'scaler_state_dict': {}, 'rng_state': {},
            'progress': {'completed_epochs': 2, 'validated_epochs': 2, 'optimizer_updates': 20, 'ema_updates': 20},
            'early_stopping': {'reference_auc': .8, 'bad_epochs': 1},
            'inference_ready': True, 'validation_complete': True, 'weights_source': 'ema',
            'pretrained_provenance': legacy_student_tests.SourceTests().provenance()}
        self.expected = {'schema_version': 9, 'artifact_type': 'kd_student'}

    def test_resume_preserves_independent_supervision(self):
        student.validate_student_resume(self.checkpoint, self.config, self.expected)
        bad_cases = [lambda c: c.update(training_mode='distillation'),
                     lambda c: c.update(teacher_provenance=[{'fold': 0}]),
                     lambda c: c['distillation'].update(enabled=True),
                     lambda c: c['training_config'].update(soft_labels='unexpected_cache'),
                     lambda c: c['training_config'].update(image_size=768)]
        for mutate in bad_cases:
            bad = copy.deepcopy(self.checkpoint); mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                student.validate_student_resume(bad, self.config, self.expected)

    def test_mode_switching_cannot_repurpose_saved_checkpoints(self):
        kd = configuration(training_mode='distillation', alpha=.2, soft_labels=Path('cache'))
        with self.assertRaises(ValueError):
            student.validate_student_resume(self.checkpoint, kd, self.expected)
        bad = copy.deepcopy(self.checkpoint)
        bad['schema_version'] = 8
        with self.assertRaises(ValueError):
            student.validate_student_resume(bad, self.config, {'schema_version': 8})

    def test_legacy_schema8_defaults_missing_mode_to_distillation(self):
        kd = configuration(training_mode='distillation', alpha=.2, soft_labels=Path('cache'))
        prior = copy.deepcopy(self.checkpoint)
        prior.pop('training_mode'); prior['schema_version'] = 8
        prior['training_config'] = student._configuration(kd)
        prior['training_config'].pop('training_mode')
        student.validate_student_resume(prior, kd, {'schema_version': 8})


class SupervisedCliTests(unittest.TestCase):
    def arguments(self, output):
        return ['--training-mode', 'supervised_bce', '--alpha', '0', '--fold-plan', 'unused_folds.json',
                '--output', str(output), '--image-size', '512', '--validation-tta', 'hflip2']

    def test_dry_run_requires_no_teacher_path_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new_student'
            stream = io.StringIO()
            logged = Mock(side_effect=AssertionError('No recorder'))
            with (patch.object(student, 'run_training', side_effect=AssertionError('No training')),
                  patch.dict('sys.modules', {'training_evidence_monitor': SimpleNamespace(run_logged=logged)}),
                  patch.object(torch.cuda, 'is_available', side_effect=AssertionError('No CUDA')),
                  redirect_stdout(stream)):
                self.assertEqual(student.main(self.arguments(output)), 0)
            result = json.loads(stream.getvalue())
            self.assertEqual(result['schema_version'], 9)
            self.assertFalse(result['teacher_cache_required'])
            self.assertIsNone(result['config']['soft_labels'])
            self.assertEqual(result['config']['training_mode'], 'supervised_bce')
            self.assertFalse(output.exists())
            logged.assert_not_called()

    def test_explicit_execute_records_original_teacher_free_command(self):
        args = self.arguments(Path('student_output')) + ['--execute']
        logged = Mock(return_value=7)
        with (patch.dict(os.environ, {'KD_TRAINING_LOG_ACTIVE': '0'}),
              patch.dict('sys.modules', {'training_evidence_monitor': SimpleNamespace(run_logged=logged)}),
              patch.object(student, 'run_training', side_effect=AssertionError('No training'))):
            self.assertEqual(student.main(args), 7)
        self.assertEqual(logged.call_args.args[0].command[3:], args)
        self.assertNotIn('--soft-labels', logged.call_args.args[0].command)


if __name__ == '__main__':
    unittest.main()
