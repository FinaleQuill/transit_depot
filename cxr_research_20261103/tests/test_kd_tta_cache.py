"""CPU-only raw-view cache tests: patient isolation, integrity and honest TTA schema."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from distill_tta_data import (CACHE_SCHEMA, OUTPUT_TYPE, TARGET_AGGREGATION,
                             canonical_tta, load_soft_labels, normalize_tta,
                             sha256_file, validate_cache_arrays, write_soft_labels)
from test_kd_softlabels import synthetic_fixture


class TtaCacheTests(unittest.TestCase):
    def setUp(self):
        records, plan, digest, teachers, metadata, arrays = synthetic_fixture()
        self.records, self.plan, self.plan_hash, self.teachers = records, plan, digest, teachers
        self.arrays = deepcopy(arrays)
        self.arrays['logits'] = np.stack((arrays['logits'], -arrays['logits'] + 0.25), axis=1)
        self.metadata = dict(metadata, schema=CACHE_SCHEMA, output_type=OUTPUT_TYPE,
            tta='identity_hflip', views=['identity', 'hflip'], view_axis=1,
            target_aggregation=TARGET_AGGREGATION, temperature_applied=False,
            class_weight_alignment_applied=False,
            view_transform='horizontal_flip_after_letterbox_normalization',
            original_train_only=True, official_valid_used=False, test_used=False)
        # The approved mixed fixed horizons must retain their true identities.
        for teacher in self.metadata['teachers'][1:]:
            teacher['epoch'] = 12
            teacher['training_config']['epochs'] = 12

    def validate(self, metadata=None, arrays=None, records=None):
        return validate_cache_arrays(metadata or self.metadata,
            arrays if arrays is not None else self.arrays, plan=self.plan,
            plan_hash=self.plan_hash, records=records or self.records)

    def test_two_distinct_raw_views_survive_and_are_readonly(self):
        cache = self.validate()
        self.assertEqual(cache.logits.shape, (10, 2, 10))
        np.testing.assert_array_equal(cache.logits[:, 1], self.arrays['logits'][:, 1])
        self.assertTrue((cache.logits > 1).any())
        self.assertTrue((cache.logits < 0).any())
        self.assertFalse(cache.logits.flags.writeable)
        self.assertFalse(cache.teacher_pos_weights.flags.writeable)
        self.assertEqual(cache.metadata['teachers'][1]['epoch'], 12)

    def test_identity_is_an_explicit_one_view_cache(self):
        metadata = dict(self.metadata, tta='identity', views=['identity'])
        arrays = dict(self.arrays, logits=self.arrays['logits'][:, :1, :].copy())
        self.assertEqual(self.validate(metadata, arrays).logits.shape, (10, 1, 10))
        self.assertEqual(canonical_tta('hflip2'), 'identity_hflip')
        self.assertEqual(normalize_tta('identity_hflip'), ('identity', 'hflip'))
        with self.assertRaises(ValueError):
            normalize_tta('random_crop')

    def test_metadata_cannot_claim_tempered_probabilities_or_unknown_views(self):
        invalid = [('output_type', 'probabilities'), ('views', ['hflip', 'identity']),
                   ('temperature_applied', True), ('class_weight_alignment_applied', True),
                   ('target_aggregation', 'mean_raw_logits'), ('tta', 'hflip2'),
                   ('official_valid_used', True), ('test_used', True), ('view_axis', 2)]
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(dict(self.metadata, **{key: value}))

    def test_nan_in_second_view_and_wrong_shape_dtype_are_rejected(self):
        arrays = deepcopy(self.arrays)
        arrays['logits'][1, 1, 7] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite FP32 NxVx10'):
            self.validate(arrays=arrays)
        for logits in (self.arrays['logits'][:, 0, :], self.arrays['logits'].astype(np.float16)):
            with self.assertRaisesRegex(ValueError, 'finite FP32 NxVx10'):
                self.validate(arrays=dict(self.arrays, logits=logits))

    def test_original_teacher_patient_and_source_guards_are_not_weakened(self):
        cases = [('training_subject_ids', ['p0', 'p1', 'p2', 'p3', 'p4']),
                 ('selection_protocol', 'best_outer_auc'), ('training_complete', False)]
        for key, value in cases:
            metadata = deepcopy(self.metadata)
            metadata['teachers'][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'strict-OOF'):
                self.validate(metadata)
        records = deepcopy(self.records)
        records[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'original train'):
            self.validate(records=records)
        arrays = deepcopy(self.arrays)
        arrays['source_image_sha256'][0] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.validate(arrays=arrays)

    def test_roundtrip_integrity_no_overwrite_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / 'folds.json'
            plan_path.write_text(json.dumps(self.plan))
            teachers = deepcopy(self.metadata['teachers'])
            for teacher in teachers:
                teacher['fold_plan_sha256'] = sha256_file(plan_path)
            cache_dir = root / 'cache'
            with patch('distill_tta_data._read_validated_plan', return_value=self.plan):
                cache = write_soft_labels(cache_dir, plan_path, self.records,
                                           self.arrays['logits'], teachers)
                again = load_soft_labels(cache_dir, plan_path, self.records)
                np.testing.assert_array_equal(again.logits, cache.logits)
                with np.load(cache_dir / 'soft_labels.npz', allow_pickle=False) as archive:
                    self.assertEqual(archive['image_ids'].dtype.kind, 'U')
                with self.assertRaises(FileExistsError):
                    write_soft_labels(cache_dir, plan_path, self.records,
                                       self.arrays['logits'], teachers)
                with (cache_dir / 'soft_labels.npz').open('ab') as stream:
                    stream.write(b'tamper')
                with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                    load_soft_labels(cache_dir, plan_path, self.records)


    def test_extractor_preserves_flip_order_without_temperature_or_probability(self):
        import contextlib
        import torch
        from extract_soft_labels_tta import _forward_raw_view_logits

        # The callable returns the leftmost pixel as each class's score; flipping
        # must produce the original rightmost pixel, with negative/raw values intact.
        inputs = torch.tensor([[[[-2., 0., 7.]]]])
        seen = []
        def score(tensor):
            seen.append(tensor.clone())
            return tensor[:, 0, 0, 0, None].expand(-1, 10)
        values = _forward_raw_view_logits(score, inputs, tta='identity_hflip',
                                         context_factory=contextlib.nullcontext)
        self.assertEqual(values.shape, (1, 2, 10))
        np.testing.assert_array_equal(values[0, 0], np.full(10, -2.))
        np.testing.assert_array_equal(values[0, 1], np.full(10, 7.))
        self.assertEqual(len(seen), 2)
        np.testing.assert_array_equal(inputs.numpy(), np.asarray([[[[-2., 0., 7.]]]]))
        with self.assertRaises(FloatingPointError):
            _forward_raw_view_logits(lambda tensor: torch.full((1, 10), float('inf')),
                inputs, tta='identity', context_factory=contextlib.nullcontext)

    def test_bad_inputs_are_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / 'folds.json'
            plan_path.write_text(json.dumps(self.plan))
            teachers = deepcopy(self.metadata['teachers'])
            for teacher in teachers:
                teacher['fold_plan_sha256'] = sha256_file(plan_path)
            cache_dir = root / 'cache'
            with patch('distill_tta_data._read_validated_plan', return_value=self.plan):
                with self.assertRaisesRegex(ValueError, 'NxVx10'):
                    write_soft_labels(cache_dir, plan_path, self.records,
                                       self.arrays['logits'][:, 0], teachers)
            self.assertFalse(cache_dir.exists())


if __name__ == '__main__':
    unittest.main()
