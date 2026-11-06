"""CPU checks for patient isolation, original labels and strict image loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from teacher_distill_20261106.data import StudyDataset, validate_records
from teacher_distill_20261106.prepare import balanced_patient_folds, overlap


def row(patient, study, image, target=1):
    labels = [int(i == target) for i in range(1, 11)]
    return dict(image_id=f'train:{image}', split='train', subject_id=patient, study_id=study,
        study_key=f'{patient}/{study}', dicom_id=image,
        relative_path=f'train/{patient}/{study}/{image}.jpg',
        targets=json.dumps(labels), label_text=str(target),
        training_eligible='True', status='ok', exclusion_reasons='[]',
        file_sha256='a'*64, pixel_sha256='b'*64, view_position='AP')


class ManifestTests(unittest.TestCase):
    def test_original_cooccurrence_is_preserved(self):
        r = row('p', 's', 'i')
        r['targets'] = '[1,0,0,0,0,0,0,0,0,1]'
        r['label_text'] = '1;10'
        self.assertEqual(validate_records([r])[0]['targets'], (1,0,0,0,0,0,0,0,0,1))

    def test_conflicting_study_label_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Within-Study'):
            validate_records([row('p','s','i1'), row('p','s','i2',2)])

    def test_test_and_exclusions_rejected(self):
        for mutation in ({'split':'test'}, {'exclusion_reasons':'["old_cleaning"]'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_records([dict(row('p','s','i'), **mutation)])

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'path'):
            validate_records([dict(row('p','s','i'), relative_path='train/p/s/../../i.jpg')])

    def test_patient_folds_are_deterministic_balanced_and_complete(self):
        rows = []
        for i in range(42):
            for j in range(1+i%4):
                r = row(f'p{i}',f's{i}_{j}',f'i{i}_{j}',i%10+1)
                r['file_sha256'] = hashlib.sha256(r['image_id'].encode()).hexdigest()
                r['pixel_sha256'] = r['file_sha256']
                rows.append(r)
        a, stats = balanced_patient_folds(rows)
        b, _ = balanced_patient_folds(list(reversed(rows)))
        self.assertEqual(a,b)
        self.assertEqual(set(a), {r['subject_id'] for r in rows})
        self.assertEqual(stats['fold_patient_capacities'],[14,14,14])
        for fold in range(3):
            train=[r for r in rows if a[r['subject_id']]!=fold]
            oof=[r for r in rows if a[r['subject_id']]==fold]
            self.assertFalse(any(overlap(train,oof).values()))
            self.assertEqual(len(train)+len(oof),len(rows))


class ImageLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows=[]
        for i in range(2):
            r=row('p','s',f'i{i}')
            path=self.root/r['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(47+i*3,29),(40+i*50,80,120)).save(path)
            r['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            self.rows.append(r)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_views_and_paths_survive(self):
        data=StudyDataset(self.rows,self.root,image_size=56,training=False)
        sample=data[0]
        self.assertEqual(tuple(sample['images'].shape),(2,3,56,56))
        self.assertEqual(len(sample['paths']),2)
        self.assertEqual(sample['patient_id'],'p')
        self.assertEqual(sample['targets'].tolist(),[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        data.set_epoch(2)
        self.assertTrue(sample['images'].equal(data[0]['images']))

    def test_missing_image_is_fatal(self):
        (self.root/self.rows[0]['relative_path']).unlink()
        with self.assertRaises(FileNotFoundError):
            StudyDataset(self.rows,self.root,image_size=56)

    def test_progressive_resolution_changes_underlying_loader(self):
        data=StudyDataset(self.rows,self.root,image_size=512,training=False)
        self.assertEqual(tuple(data[0]['images'].shape),(2,3,512,512))
        data.image_size=768
        self.assertEqual(data.image_size,768)
        self.assertEqual(tuple(data[0]['images'].shape),(2,3,768,768))
        with self.assertRaises(ValueError):
            data.image_size=0

    def test_changed_image_is_fatal(self):
        data=StudyDataset(self.rows,self.root,image_size=56)
        path=self.root/self.rows[0]['relative_path']
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256 changed'):
            data[0]


if __name__=='__main__':
    unittest.main()
