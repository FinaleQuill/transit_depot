"""Small synthetic CPU archives/metadata; no real model, image, or GPU inference."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_teacher_epochs as diagnostic
from test_kd_softlabels import synthetic_fixture
import test_validate_teacher as teacher_fixtures


def source(name):
    fold = int(name[4])
    epoch = int(name.split('_e')[1])
    complete = epoch != 13
    return {'id': name, 'fold': fold, 'epoch': epoch, 'fixed_horizon': 20 if fold == 0 else 12,
            'training_complete': complete, 'inference_ready': complete, 'weights_source': 'ema',
            'checkpoint_sha256': hashlib.sha256(name.encode()).hexdigest(), 'teacher_pos_weight': [2.] * 10}


class RawLogitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = deepcopy(synthetic_fixture()[0])
        for row in self.records:
            row['split'] = 'valid'
            row['relative_path'] = row['relative_path'].replace('train/', 'valid/', 1)
        self.logits = np.arange(len(self.records) * 20, dtype=np.float32).reshape(-1, 2, 10) / 20 - 3
        self.binding = {'manifest_sha256': 'a'*64, 'fold_plan_sha256': 'b'*64}

    def save(self, name='fold0_e13', logits=None):
        self.root.joinpath('input').mkdir(exist_ok=True)
        return diagnostic.save_raw_logits(self.root/'input', source(name), self.records,
            self.logits if logits is None else logits, binding=self.binding,
            precision='bfloat16', collection_sha256='c'*64)

    def test_raw_archive_retains_views_negative_logits_identity_and_partial_epoch(self):
        entry = self.save()
        rows, logits, metadata = diagnostic.load_raw_logits(self.root/'input', entry)
        self.assertEqual(rows, [{k:r[k] for k in ('image_id', 'subject_id', 'study_id', 'study_key',
            'relative_path', 'file_sha256', 'split', 'targets')} for r in self.records])
        np.testing.assert_array_equal(logits, self.logits)
        self.assertEqual(logits.dtype, np.float32)
        self.assertTrue((logits < 0).any())
        self.assertTrue((logits > 1).any())
        self.assertEqual(metadata['views'], ['identity', 'hflip'])
        self.assertFalse(metadata['source']['training_complete'])
        self.assertFalse(metadata['source']['inference_ready'])
        self.assertFalse(metadata['temperature_applied'])
        self.assertFalse(metadata['pos_weight_correction_applied'])
        with self.assertRaises(FileExistsError):
            self.save()

    def test_raw_arrays_and_labels_cannot_be_replaced_or_silently_coerced(self):
        for logits in (self.logits[:, 0], self.logits.astype(np.float64), self.logits[:1], self.logits * np.nan):
            with self.subTest(shape=logits.shape), self.assertRaisesRegex(ValueError, 'FP32 Nx2x10'):
                self.save(logits=logits)
        for split in ('train', 'test'):
            broken = deepcopy(self.records)
            broken[0]['split'] = split
            with self.assertRaisesRegex(ValueError, 'original valid'):
                diagnostic.checked_records(broken)

    def test_hash_mismatch_and_path_escape_are_rejected(self):
        entry = self.save()
        broken = {**entry, 'npz': '../outside.npz'}
        with self.assertRaisesRegex(ValueError, 'escapes'):
            diagnostic.load_raw_logits(self.root/'input', broken)
        (self.root/'input'/entry['npz']).write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'hash'):
            diagnostic.load_raw_logits(self.root/'input', entry)

    def test_replay_rejects_changed_class_order_or_forward_precision(self):
        entry = self.save()
        metadata_path = self.root/'input'/entry['metadata']
        original = json.loads(metadata_path.read_text())
        for field, replacement in [('class_ids', list(range(10))), ('image_size', 512),
                                   ('precision', 'float32'), ('num_images', True)]:
            with self.subTest(field=field):
                metadata_path.write_text(json.dumps({**original, field: replacement}))
                altered = {**entry, 'metadata_sha256': diagnostic.sha256_file(metadata_path)}
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    diagnostic.load_raw_logits(self.root/'input', altered)

        # A self-consistent archive cannot be reported under a different forward precision.
        metadata_path.write_text(json.dumps(original))
        shutil.rmtree(self.root/'input')
        index = self.sealed_input()
        index['precision'] = 'float16'
        (self.root/'input/raw_index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'input binding'):
            diagnostic.replay_diagnostic(self.root/'input', self.root/'wrong_precision')

    def test_completion_cannot_be_promoted_for_epoch13(self):
        valid = source('fold0_e13')
        diagnostic.validate_raw_source(valid)
        for field in ('training_complete', 'inference_ready'):
            with self.assertRaisesRegex(ValueError, 'completion'):
                diagnostic.validate_raw_source({**valid, field: True})

    def test_prior_correction_happens_per_image_and_view_before_averaging(self):
        records = self.records + [{**self.records[0], 'image_id': 'extra',
            'relative_path': 'valid/p0/s00/extra.jpg', 'file_sha256': 'e'*64}]
        raw = np.zeros((len(records), 2, 10), np.float32)
        raw[0, 0] = -2
        raw[0, 1] = 6
        raw[-1] = 2
        corrected = diagnostic.probabilities_from_raw(records, raw, [4.]*10, correction=True)
        sigmoid = lambda z: np.exp(-np.logaddexp(0., -z))
        expected = ((sigmoid(-2-np.log(4)) + sigmoid(6-np.log(4))) / 2 + sigmoid(2-np.log(4))) / 2
        self.assertAlmostEqual(corrected['hflip2']['probabilities'][0, 0], expected)
        uncorrected = diagnostic.probabilities_from_raw(records, raw, [4.]*10, correction=False)
        mean = uncorrected['hflip2']['probabilities'][0, 0]
        post_study = sigmoid(np.log(mean/(1-mean))-np.log(4))
        self.assertNotAlmostEqual(expected, post_study)
        unit = diagnostic.probabilities_from_raw(records, raw, [1.]*10, correction=True)
        np.testing.assert_allclose(unit['hflip2']['probabilities'], uncorrected['hflip2']['probabilities'])

    def sealed_input(self):
        entries = []
        for name in ['fold0_e20', 'fold1_e12', 'fold2_e12', 'fold3_e12', 'fold4_e12', 'fold0_e13']:
            logits = np.zeros_like(self.logits)
            if name.startswith('fold0'):
                for i, row in enumerate(self.records):
                    logits[i] = (4 if row['targets'][0] else -4) * (1 if name.endswith('e13') else -1)
            entries.append(self.save(name, logits))
        index = {'schema': diagnostic.RAW_SCHEMA, 'complete': True, **self.binding,
                 'teacher_collection_sha256': 'c'*64, 'precision': 'bfloat16', 'sources': entries}
        (self.root/'input/raw_index.json').write_text(json.dumps(index))
        return index

    def test_cpu_replay_reproduces_both_policies_and_single_factor_ensembles(self):
        self.sealed_input()
        with patch('torch.cuda.is_available', side_effect=AssertionError('CPU replay must not query CUDA')):
            result = diagnostic.replay_diagnostic(self.root/'input', self.root/'replay')
        self.assertEqual(len(result['results']), 32)
        self.assertEqual(len(result['A_vs_B']), 4)
        for comparison in result['A_vs_B']:
            self.assertEqual(comparison['A_macro_auc'], 0.)
            self.assertEqual(comparison['B_macro_auc'], 1.)
        self.assertFalse(result['checkpoint_selection_performed'])
        self.assertFalse(result['teacher_collection_modified'])
        self.assertFalse(result['train_soft_labels_generated'])
        self.assertFalse(next(x for x in result['sources'] if x['id']=='fold0_e13')['training_complete'])
        self.assertTrue((self.root/'replay/teacher_epoch_diagnostic_report.json').is_file())
        with self.assertRaises(FileExistsError):
            diagnostic.replay_diagnostic(self.root/'input', self.root/'replay')

    def test_replay_rejects_partial_sources_and_original_data_output(self):
        index = self.sealed_input()
        index['complete'] = False
        (self.root/'input/raw_index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'all six'):
            diagnostic.replay_diagnostic(self.root/'input', self.root/'partial')
        index['complete'] = True
        index['image_root'] = str(self.root/'images')
        (self.root/'input/raw_index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'read-only image root'):
            diagnostic.replay_diagnostic(self.root/'input', self.root/'images'/'new')


class EpochProtocolTests(unittest.TestCase):
    def test_epoch13_uses_original_state_without_changing_completion_or_configuration(self):
        fixture = teacher_fixtures.TeacherDiagnosticTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        early, final = fixture.checkpoint(13), fixture.checkpoint(20)
        for value in (early, final):
            value['loss_config'] = {'pos_weight': [1.]*10}
            value['training_config'].update(seed=43, batch_size=2, accumulation_steps=16)
        before = deepcopy(early)
        result = diagnostic.validate_epoch13(early, final, plan=fixture.plan,
            plan_hash='b'*64, train_records=fixture.records)
        self.assertEqual(early, before)
        self.assertFalse(result['training_complete'])
        changed = deepcopy(final)
        changed['training_config']['seed'] = 47
        with self.assertRaisesRegex(ValueError, 'configuration'):
            diagnostic.validate_epoch13(early, changed, plan=fixture.plan,
                plan_hash='b'*64, train_records=fixture.records)

    def test_preview_does_not_load_models_query_gpu_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root/'e13.pt'
            checkpoint.write_bytes(b'preview marker only')
            collection = {'teachers': [{'fold_index': i, 'epochs': 20 if i==0 else 12,
                'checkpoint': str(root/f'fold{i}.pt')} for i in range(5)]}
            with patch.object(diagnostic, 'validation_binding', return_value={'manifest_sha256':'a'*64}), \
                 patch.object(diagnostic, 'read_teacher_collection', return_value=collection), \
                 patch('torch.load', side_effect=AssertionError('No checkpoint loading')), \
                 patch.object(diagnostic, 'require_idle_gpu', side_effect=AssertionError('No GPU query')), \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(diagnostic.main(['--epoch13-checkpoint', str(checkpoint),
                    '--output-dir', str(root/'output')]), 0)
            report = json.loads(stream.getvalue())
            self.assertEqual(report['A_epochs'], [20,12,12,12,12])
            self.assertEqual(report['B_epochs'], [13,12,12,12,12])
            self.assertEqual(report['checkpoints_loaded'], 0)
            self.assertFalse((root/'output').exists())

    def test_execute_checks_gpu_before_logger_or_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root/'e13.pt';checkpoint.write_bytes(b'preview marker')
            collection = {'teachers': [{'fold_index': i, 'epochs': 20 if i==0 else 12,
                'checkpoint': str(root/f'fold{i}.pt')} for i in range(5)]}
            with patch.object(diagnostic, 'validation_binding'), \
                 patch.object(diagnostic, 'read_teacher_collection', return_value=collection), \
                 patch.object(diagnostic, 'completed_teacher_collection'), \
                 patch.object(diagnostic, 'require_idle_gpu', side_effect=RuntimeError('GPU busy')), \
                 patch('torch.load', side_effect=AssertionError('No weights')):
                with self.assertRaisesRegex(RuntimeError, 'GPU busy'):
                    diagnostic.main(['--epoch13-checkpoint', str(checkpoint), '--output-dir', str(root/'output'), '--execute'])
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
