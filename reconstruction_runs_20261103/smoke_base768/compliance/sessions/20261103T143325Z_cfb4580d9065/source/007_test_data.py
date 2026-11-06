"""CPU checks for isolation, complete-Study semantics and original-pixel tiles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from reconstruction.data import StudyDataset, letterbox, read_records, study_pos_weights, tile_boxes
from reconstruction.splits import content_groups, prepare_splits, validate_mapping


def record(patient, study, image, target, digest=None):
    digest = digest or hashlib.sha256(image.encode()).hexdigest()
    return {'image_id': image, 'split': 'train', 'subject_id': patient,
            'study_id': study, 'study_key': f'{patient}/{study}', 'dicom_id': image,
            'relative_path': f'train/{patient}/{study}/{image}.png',
            'targets': json.dumps(target), 'view_position': 'AP', 'training_eligible': 'True',
            'status': 'ok', 'file_sha256': digest, 'pixel_sha256': digest}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class SplitTests(unittest.TestCase):
    def test_reuse_isolation_replay_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, mapping = [], {}
            for fold in range(5):
                for y in (0, 1):
                    patient = f'p{fold}_{y}'
                    row = record(patient, 's0', f'i{fold}_{y}', [y]*10)
                    rows.append(row); mapping[row['study_key']] = fold
            source = root/'source.csv'; write_csv(source, rows)
            source_before = source.read_bytes()
            legacy = root/'legacy.json'
            legacy.write_text(json.dumps({'n_splits': 5, 'source_manifest_sha256': hashlib.sha256(source_before).hexdigest(),
                                          'study_to_fold': mapping}))
            plan = prepare_splits(source, legacy, root/'splits')
            self.assertEqual(plan['method'], 'revalidated_existing_fivefold_mapping')
            self.assertEqual([plan['roles'][r]['studies'] for r in ('train','dev','holdout')], [6,2,2])
            self.assertTrue(all(v == 0 for pair in plan['isolation'].values() for v in pair.values()))
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(prepare_splits(source, legacy, root/'splits'), plan)
            train = read_records(root/'splits/train.csv')
            self.assertFalse({r['subject_id'] for r in train} & {'p4_0','p4_1','p0_0','p0_1'})
            with (root/'splits/train.csv').open('a') as stream: stream.write('\n')
            with self.assertRaises(FileExistsError): prepare_splits(source, legacy, root/'splits')

    def test_exact_content_connects_whole_patients(self):
        rows = [record('p0','s0','i0',[1]*10), record('p0','s1','i1',[0]*10),
                record('p1','s2','i2',[1]*10), record('p1','s3','i3',[0]*10)]
        rows[2]['pixel_sha256'] = rows[0]['pixel_sha256']
        groups = content_groups(rows)
        self.assertEqual(groups['p0'], groups['p1'])
        mapping = {r['study_key']: int(r['subject_id'][-1]) for r in rows}
        with self.assertRaisesRegex(ValueError, 'crosses folds'): validate_mapping(rows, mapping, groups)

    def test_fallback_preserves_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [record(f'p{i}', f's{i}', f'i{i}', [i%2]*10) for i in range(30)]
            rows[2]['pixel_sha256'] = rows[0]['pixel_sha256']
            source = root/'source.csv'; write_csv(source, rows)
            a = prepare_splits(source, None, root/'a')
            b = prepare_splits(source, None, root/'b')
            self.assertEqual(a['method'], 'deterministic_patient_content_multilabel_greedy')
            for role in ('train','dev','holdout'):
                self.assertEqual((root/'a'/f'{role}.csv').read_bytes(), (root/'b'/f'{role}.csv').read_bytes())
                self.assertEqual(a['roles'][role]['positive_studies'], b['roles'][role]['positive_studies'])
            owners = {r['subject_id']: role for role in ('train','dev','holdout')
                      for r in read_records(root/'a'/f'{role}.csv')}
            self.assertEqual(owners['p0'], owners['p2'])

    def test_conflicting_study_labels_rejected(self):
        rows = [record('p','s','a',[0]*10), record('p','s','b',[1]*10)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_csv(root/'source.csv', rows)
            with self.assertRaisesRegex(ValueError, 'Inconsistent labels'):
                prepare_splits(root/'source.csv', None, root/'split')


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = []
        self.sources = []
        for i, (patient, study, label) in enumerate([('p0','s0',0),('p0','s0',0),('p1','s1',1)]):
            width, height = 43+i, 61+i
            y, x = np.mgrid[:height,:width]
            pixels = np.stack([(x*7+i*13)%256, (y*5+i*11)%256, (x*3+y*2)%256], axis=-1).astype(np.uint8)
            source = Image.fromarray(pixels)
            row = record(patient, study, f'i{i}', [label]*10)
            path = self.root/row['relative_path']; path.parent.mkdir(parents=True, exist_ok=True)
            source.save(path)
            row['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            row['pixel_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
            row['view_position'] = 'AP' if i != 1 else 'LATERAL'
            self.rows.append(row); self.sources.append(source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_study_and_train_study_weights(self):
        dataset = StudyDataset(self.rows, self.root, image_size=32)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(tuple(sample['images'].shape), (2,3,32,32))
        self.assertEqual(tuple(sample['targets'].shape), (10,))
        self.assertEqual(sample['image_ids'], ['i0','i1'])
        self.assertEqual(sample['views'], ['AP','LATERAL'])
        self.assertEqual(sample['study_key'], 'p0/s0')
        # Two negative images represent just one negative Study.
        self.assertTrue(torch.equal(study_pos_weights(self.rows), torch.ones(10)))

    def test_tiles_read_original_pixels_and_cover_source(self):
        dataset = StudyDataset(self.rows, self.root, image_size=32, local_tiles=True)
        sample = dataset[0]
        self.assertEqual(tuple(sample['images'].shape), (2,5,3,32,32))
        self.assertEqual(tuple(sample['tile_boxes'].shape), (2,5,4))
        source = self.sources[0]
        boxes = tile_boxes(*source.size, True)
        covered = np.zeros((source.height, source.width), dtype=bool)
        for k, (x0,y0,x1,y1) in enumerate(boxes):
            expected = np.asarray(letterbox(source.crop((x0,y0,x1,y1)),32),dtype=np.float32).transpose(2,0,1)/255.
            expected = (expected-np.asarray((.485,.456,.406),dtype=np.float32)[:,None,None])/np.asarray((.229,.224,.225),dtype=np.float32)[:,None,None]
            self.assertTrue(torch.equal(sample['images'][0,k], torch.from_numpy(expected)))
            if k: covered[y0:y1,x0:x1] = True
        self.assertTrue(covered.all())
        self.assertEqual(tuple(sample['targets'].shape), (10,))

    def test_eval_deterministic_and_epoch_augmentation_replay(self):
        evaluation = StudyDataset(self.rows, self.root, image_size=32)
        before = evaluation[0]['images']; evaluation.set_epoch(5)
        self.assertTrue(torch.equal(before,evaluation[0]['images']))
        train = StudyDataset(self.rows, self.root, image_size=32, training=True, seed=43)
        first = train[0]['images']
        self.assertTrue(torch.equal(first, train[0]['images']))
        varied = []
        for epoch in range(1,5):
            train.set_epoch(epoch); varied.append(train[0]['images'])
        self.assertTrue(any(not torch.equal(first,v) for v in varied))
        train.set_epoch(0)
        self.assertTrue(torch.equal(first, train[0]['images']))

    def test_first_read_hash_guard_and_path_guard(self):
        dataset = StudyDataset(self.rows,self.root,image_size=32)
        path = self.root/self.rows[0]['relative_path']
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256 changed'): dataset[0]
        invalid = [dict(r) for r in self.rows]
        invalid[0]['relative_path'] = '../elsewhere.png'
        with self.assertRaisesRegex(ValueError,'escapes'): StudyDataset(invalid,self.root,image_size=32)

    def test_test_records_are_rejected(self):
        invalid = [dict(r,split='test') for r in self.rows]
        with self.assertRaisesRegex(ValueError,'train/valid only'): StudyDataset(invalid,self.root,image_size=32)


if __name__ == '__main__':
    unittest.main()
