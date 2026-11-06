"""CPU checks of refit authorization, original paths and complete-Study weights."""
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from dataset import CachedSampleError
from one_week.data import RefitDataset, loader


def original_records(root):
    """Two views in one train Study; one view in one official valid Study."""
    records = []
    for index, split in enumerate(('train', 'train', 'valid')):
        subject = 'subject_train' if split == 'train' else 'subject_valid'
        study = 'study_train' if split == 'train' else 'study_valid'
        dicom = f'pic_{index}'
        relative = f'{split}/{subject}/{study}/{dicom}.jpg'
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.arange(25 * 19, dtype=np.uint16).reshape(19, 25)
        pixels = ((pixels + index * 31) % 256).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        records.append(dict(image_id=f'{split}:{index:06}', split=split,
            subject_id=subject, study_id=study, study_key=f'{subject}/{study}',
            dicom_id=dicom, relative_path=relative,
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pixel_sha256='0' * 64, targets=(int(split == 'train'),) * 10,
            status='ok', training_eligible=str(split == 'train')))
    return records


class RefitDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = original_records(self.root)

    def test_valid_augmentation_requires_explicit_refit_and_retains_original_paths(self):
        records, root = self.records, self.root
        original = copy.deepcopy(records)
        with self.assertRaisesRegex(ValueError, 'explicit.*refit authorization'):
            RefitDataset(records, root, image_size=384, training=True, seed=44,
                         valid_is_training=False)
        dataset = RefitDataset(records, root, image_size=384, training=True, seed=44,
                               valid_is_training=True)
        validation = dataset.datasets[1]
        self.assertEqual(validation.split, 'valid')
        self.assertIs(validation.training, True)
        self.assertTrue(validation.records[0]['relative_path'].startswith('valid/'))
        self.assertEqual(validation.paths[0], (root / records[2]['relative_path']).resolve())
        self.assertTrue(any(type(transform).__name__ == 'HorizontalFlip'
                            for transform in validation.transform.transforms))
        sample = dataset[2]
        self.assertEqual(sample['split'], 'valid')
        self.assertEqual(sample['image_id'], records[2]['image_id'])
        torch.testing.assert_close(sample['targets'], torch.zeros(10))
        self.assertEqual(records, original)

    def test_mixed_batch_uses_global_study_weights_and_covers_all_original_images(self):
        records = self.records
        config = dict(image_root=str(self.root), image_size=384, seed=44, include_valid=True,
                      workers=0, batch_size=3)
        data = loader(records, config, training=True, epoch=1)
        batches = list(data)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(set(batch['image_ids']), {record['image_id'] for record in records})
        self.assertEqual(batch['images'].shape, (3, 3, 384, 384))
        self.assertTrue(bool(torch.isfinite(batch['images']).all()))
        # Total N=3, Studies=2: each Study has total weight 1.5. Per-split
        # normalization would instead assign every image weight 1, which is wrong.
        expected_weights = {records[0]['image_id']: .75, records[1]['image_id']: .75,
                            records[2]['image_id']: 1.5}
        actual = dict(zip(batch['image_ids'], batch['loss_weights'].tolist()))
        self.assertEqual(actual, expected_weights)
        self.assertEqual(sum(actual.values()), len(records))
        for identity, target in zip(batch['image_ids'], batch['targets']):
            value = 1. if identity.startswith('train:') else 0.
            torch.testing.assert_close(target, torch.full((10,), value))

    def test_refit_getitem_rejects_changed_original_bytes_before_decoding(self):
        records = self.records
        dataset = RefitDataset(records, self.root, image_size=384, training=True, seed=44,
                               valid_is_training=True)
        path = self.root / records[2]['relative_path']
        # Trailing bytes leave this JPEG decodable, but the accepted original file
        # identity has changed and must be rejected even during a valid refit.
        path.write_bytes(path.read_bytes() + b'changed-after-manifest')
        with Image.open(path) as image:
            image.load()
        with self.assertRaisesRegex(CachedSampleError, 'SHA256 differs'):
            dataset[2]


if __name__ == '__main__':
    unittest.main()
