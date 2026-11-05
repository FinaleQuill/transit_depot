import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from raw_rebuild.prepare_data import inspect_image, labels, load_official, validate_folds


class OriginalDataTests(unittest.TestCase):
    def test_labels_preserve_original_multilabel_and_no_finding(self):
        ids, target = labels('10;2')
        self.assertEqual(ids, [10, 2])
        self.assertEqual(target, [0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            labels('0;2')

    def test_original_duplicate_images_retained_and_missing_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'train'/'subject_00000'/'study_00000'
            folder.mkdir(parents=True)
            image = Image.new('L', (8, 8), 127)
            image.save(folder/'pic_000000.jpg')
            (folder/'pic_000001.jpg').write_bytes((folder/'pic_000000.jpg').read_bytes())
            fields = ['Dicom_id','Subject_id','Study_id','ViewPosition','PatientOrientation','Rows','Columns','Predict_class']
            with (root/'train.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(3):
                    writer.writerow(dict(Dicom_id=f'pic_{index:06d}',Subject_id='subject_00000',
                                         Study_id='study_00000',ViewPosition='AP',PatientOrientation='',
                                         Rows=8,Columns=8,Predict_class='2;10'))
            rows = load_official(root, 'train')
            self.assertEqual(len(rows), 3)
            checked = [inspect_image((root,row)) for row in rows]
            self.assertEqual([r['training_eligible'] for r in checked], [True,True,True])
            self.assertEqual(checked[0]['file_sha256'], checked[1]['file_sha256'])
            self.assertEqual(checked[0]['pixel_sha256'], checked[1]['pixel_sha256'])
            self.assertEqual(checked[2]['status'], 'error')
            self.assertIn('FileNotFoundError', checked[2]['error'])

    def test_cross_fold_pixel_duplicate_rejected_without_deleting_it(self):
        rows = [dict(study_key='p1/s1',subject_id='p1',file_sha256='a',pixel_sha256='same'),
                dict(study_key='p2/s2',subject_id='p2',file_sha256='b',pixel_sha256='same')]
        with self.assertRaisesRegex(ValueError, 'crosses folds'):
            validate_folds(rows, {'p1/s1':0,'p2/s2':4})
        self.assertEqual(len(rows), 2)
        self.assertTrue(validate_folds(rows, {'p1/s1':0,'p2/s2':0}))


if __name__ == '__main__':
    unittest.main()
