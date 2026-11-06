"""CPU invariants for raw/legacy target adaptation; no model or image access."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from one_week.prepare import adapt_cache, read_raw_records


def row(index, patient='subject_0', study='study_0'):
    return {'image_id': f'train:{index:06}', 'split': 'train', 'subject_id': patient,
            'study_id': study, 'study_key': f'{patient}/{study}', 'dicom_id': f'pic_{index}',
            'relative_path': f'train/{patient}/{study}/pic_{index}.jpg',
            'file_sha256': str(index % 10) * 64, 'pixel_sha256': str(index % 10) * 64,
            'targets': (1,) + (0,) * 9, 'status': 'ok', 'training_eligible': 'True',
            'exclusion_reasons': '[]'}


class RawCacheAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.old = [row(0), row(1, 'subject_1', 'study_1')]
        self.raw = [self.old[1].copy(), row(2), self.old[0].copy()]
        self.cache = SimpleNamespace(index={self.old[0]['image_id']: 0,
                                           self.old[1]['image_id']: 1})
        self.plan = {'study_to_fold': {'subject_0/study_0': 0, 'subject_1/study_1': 1}}

    def test_reordered_rows_and_missing_raw_view_have_explicit_immutable_mapping(self):
        result = adapt_cache(self.raw, self.old, self.cache, self.plan)
        np.testing.assert_array_equal(result['cache_indices'], [1, -1, 0])
        np.testing.assert_array_equal(result['kd_mask'], [True, False, True])
        self.assertEqual(result['missing_image_ids'], ['train:000002'])
        self.assertFalse(result['cache_indices'].flags.writeable)
        self.assertFalse(result['kd_mask'].flags.writeable)
        self.assertEqual(self.cache.index, {'train:000000': 0, 'train:000001': 1})

    def test_rejects_source_label_identity_and_hash_changes(self):
        for key, value in [('targets', (0,) * 10), ('file_sha256', 'a' * 64),
                           ('relative_path', 'train/subject_0/study_0/other.jpg')]:
            records = copy.deepcopy(self.raw)
            records[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapt_cache(records, self.old, self.cache, self.plan)

    def test_rejects_cache_rows_outside_raw_and_patient_fold_overlap(self):
        with self.assertRaises(ValueError):
            adapt_cache(self.raw[:-1], self.old, self.cache, self.plan)
        changed = copy.deepcopy(self.raw)
        changed[1] = row(2, 'subject_0', 'study_other')
        plan = {'study_to_fold': dict(self.plan['study_to_fold'], **{'subject_0/study_other': 2})}
        with self.assertRaises(ValueError):
            adapt_cache(changed, self.old, self.cache, plan)

    def test_distinct_pixel_hash_definitions_preserve_identical_source_byte_binding(self):
        records = copy.deepcopy(self.raw)
        records[2]['pixel_sha256'] = 'a' * 64
        result = adapt_cache(records, self.old, self.cache, self.plan)
        self.assertEqual(int(result['cache_indices'][2]), 0)

    def test_raw_reader_preserves_restored_view_without_cache_path(self):
        import csv
        records = [row(0), row(2)]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source_train.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                for record in records:
                    writer.writerow(dict(record, targets=json.dumps(record['targets'])))
            actual = read_raw_records(path, 'train')
            self.assertEqual(len(actual), 2)
            self.assertEqual(actual[1]['image_id'], 'train:000002')
            self.assertNotIn('cache_relative_path', actual[1])
            self.assertEqual(actual[1]['targets'], records[1]['targets'])


if __name__ == '__main__':
    unittest.main()
