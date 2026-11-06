"""CPU-only teacher TTA diagnostics guards, probability aggregation and artifacts."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import validate_teacher_tta as validation
from test_kd_softlabels import synthetic_fixture


class TeacherTtaMathTests(unittest.TestCase):
    def setUp(self):
        self.records = synthetic_fixture()[0]
        self.ids = [r['image_id'] for r in self.records]
        self.logits = np.zeros((len(self.ids), 2, 10), dtype=np.float32)
        for i, r in enumerate(self.records):
            self.logits[i] = 4. if r['targets'][0] else -4.

    def test_probabilities_average_views_then_images_in_each_study(self):
        extra = dict(self.records[0], image_id='extra_view')
        records = self.records + [extra]
        ids = self.ids + ['extra_view']
        logits = np.concatenate((self.logits, np.full((1, 2, 10), 2., dtype=np.float32)))
        logits[0, 0] = -2.
        logits[0, 1] = 6.
        values = validation.probability_sets(records, ids, logits)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        np.testing.assert_allclose(values['identity']['probabilities'][0], (sigmoid(-2) + sigmoid(2)) / 2)
        expected = ((sigmoid(-2) + sigmoid(6)) / 2 + sigmoid(2)) / 2
        np.testing.assert_allclose(values['hflip2']['probabilities'][0], expected)
        self.assertNotAlmostEqual(expected, (sigmoid(2) + sigmoid(2)) / 2)
        with self.assertRaisesRegex(ValueError, 'Nx2x10'):
            validation.probability_sets(self.records, self.ids, self.logits[:, 0])

    def test_equal_teacher_average_keeps_continuous_probabilities_and_checks_order(self):
        values = validation.probability_sets(self.records, self.ids, self.logits)['identity']
        members = []
        for probability in (0., 1., .2, .3, .5):
            members.append({**values, 'probabilities': np.full((len(self.ids), 10), probability)})
        result = validation.equal_teacher_average(members)
        np.testing.assert_allclose(result['probabilities'], .4)
        broken = deepcopy(members)
        broken[1]['study_keys'] = list(reversed(broken[1]['study_keys']))
        with self.assertRaisesRegex(ValueError, 'Study coverage'):
            validation.equal_teacher_average(broken)
        with self.assertRaisesRegex(ValueError, 'five'):
            validation.equal_teacher_average(members[:4])

    def test_ten_class_auc_and_undefined_class_rejection(self):
        values = validation.probability_sets(self.records, self.ids, self.logits)['hflip2']
        metrics = validation.auc_metrics(values)
        self.assertEqual(metrics['macro_auc'], 1.)
        self.assertEqual(metrics['per_class_auc'], [1.] * 10)
        with self.assertRaisesRegex(ValueError, 'both outcomes'):
            validation.auc_metrics({**values, 'targets': np.zeros((len(self.ids), 10))})

    def test_prediction_csv_npz_roundtrip_and_no_overwrite(self):
        values = validation.probability_sets(self.records, self.ids, self.logits)['identity']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validation.save_predictions(root, 'test', values)
            with (root / result['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(values['study_keys']) * 10)
            with np.load(root / result['npz'], allow_pickle=False) as archive:
                self.assertEqual(archive['study_keys'].dtype.kind, 'U')
                np.testing.assert_array_equal(archive['targets'], values['targets'])
                np.testing.assert_allclose(archive['probabilities'], values['probabilities'])
            with self.assertRaises(FileExistsError):
                validation.save_predictions(root, 'test', values)


class TeacherTtaGuardTests(unittest.TestCase):
    def fixture(self):
        from test_teacher_collection import TeacherCollectionTests
        fixture = TeacherCollectionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for entry in fixture.entries:
            p = Path(entry['checkpoint']).parent / 'status.json'
            p.write_text(json.dumps({'state': 'completed', 'training_complete': True,
                'inference_ready': True, 'completed_epochs': entry['epochs']}))
        return fixture

    def test_every_teacher_must_have_completed_its_declared_horizon(self):
        fixture = self.fixture()
        ready = validation.completed_teacher_collection(fixture.path, fixture.plan_path)
        self.assertEqual([r['epochs'] for r in ready['teachers']], [20, 12, 12, 12, 12])
        status = Path(fixture.entries[4]['checkpoint']).parent / 'status.json'
        status.write_text(json.dumps({'state': 'paused', 'training_complete': False,
            'inference_ready': False, 'completed_epochs': 11}))
        with patch('torch.cuda.is_available', side_effect=AssertionError('CUDA must stay untouched')) as cuda:
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                validation.completed_teacher_collection(fixture.path, fixture.plan_path)
            cuda.assert_not_called()
        Path(fixture.entries[4]['checkpoint']).unlink()
        with self.assertRaises(FileNotFoundError):
            validation.completed_teacher_collection(fixture.path, fixture.plan_path)

    def test_declared_controller_must_also_have_finished(self):
        fixture = self.fixture()
        status = fixture.root / 'transition_status.json'
        status.write_text(json.dumps({'phase': 'failed', 'all_teachers_complete': False}))
        with self.assertRaisesRegex(ValueError, 'controller has not finished'):
            validation.completed_teacher_collection(fixture.path, fixture.plan_path)

    def test_official_valid_path_must_match_even_if_a_copy_has_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'valid.csv'
            clone = Path(tmp) / 'test.csv'
            original.write_text('same contents')
            clone.write_bytes(original.read_bytes())
            plan = {'official_holdout': {'path': str(original)}}
            binding = {'manifest': str(original), 'plan': plan}
            with patch('validate_teacher_tta.official_binding', return_value=binding):
                self.assertIs(validation.validation_binding(original, 'plan'), binding)
            with patch('validate_teacher_tta.official_binding', return_value={**binding, 'manifest': str(clone)}):
                with self.assertRaisesRegex(ValueError, 'official holdout path'):
                    validation.validation_binding(clone, 'plan')

    def test_gpu_guard_refuses_training_compute_processes_and_unverifiable_state(self):
        with patch('validate_teacher_tta.active_training_processes', return_value=[{'pid': 123}]), \
             patch('validate_teacher_tta.subprocess.check_output') as query:
            with self.assertRaisesRegex(RuntimeError, 'active training'):
                validation.require_idle_gpu()
            query.assert_not_called()
        with patch('validate_teacher_tta.active_training_processes', return_value=[]):
            with patch('validate_teacher_tta.subprocess.check_output', return_value='456\n'):
                with self.assertRaisesRegex(RuntimeError, 'compute processes'):
                    validation.require_idle_gpu()
            with patch('validate_teacher_tta.subprocess.check_output', side_effect=OSError('missing')):
                with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                    validation.require_idle_gpu()
            with patch('validate_teacher_tta.subprocess.check_output', return_value=''):
                validation.require_idle_gpu()

    def test_execute_rejects_raw_data_output_before_cli_logger_can_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw'
            raw.mkdir()
            output = raw / 'must_not_create'
            with patch('validate_teacher_tta.validation_binding'), \
                 patch('validate_teacher_tta.completed_teacher_collection'), \
                 patch('validate_teacher_tta.require_idle_gpu'):
                with self.assertRaisesRegex(ValueError, 'read-only image root'):
                    validation.main(['--output-dir', str(output), '--image-root', str(raw), '--execute'])
            self.assertFalse(output.exists())

    def test_preview_never_checks_cuda_loads_weights_or_creates_outputs(self):
        fixture = self.fixture()
        output = fixture.root / 'validation'
        text = io.StringIO()
        binding = {'manifest': 'valid.csv', 'manifest_sha256': 'e' * 64}
        with patch('validate_teacher_tta.validation_binding', return_value=binding), \
             patch('torch.load', side_effect=AssertionError('no weights in preview')), \
             patch('torch.cuda.is_available', side_effect=AssertionError('no CUDA in preview')), \
             contextlib.redirect_stdout(text):
            self.assertEqual(validation.main(['--teacher-collection', str(fixture.path),
                '--fold-plan', str(fixture.plan_path), '--output-dir', str(output)]), 0)
        report = json.loads(text.getvalue())
        self.assertTrue(report['dry_run'])
        self.assertFalse(report['gpu_used'])
        self.assertEqual(report['checkpoints_loaded'], 0)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
