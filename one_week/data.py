"""Preserve original split identities when explicitly refitting train plus valid."""
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset_overnight import OvernightImageDataset, _seed_worker, _transforms


class RefitDataset(ConcatDataset):
    def __init__(self, records, image_root, *, image_size, training, seed,
                 valid_is_training=False):
        if any(r['split'] not in {'train', 'valid'} for r in records):
            raise ValueError('Only original labeled train/valid may be loaded')
        if training and any(r['split'] == 'valid' for r in records) and not valid_is_training:
            raise ValueError('Official valid requires explicit, frozen refit authorization')
        datasets = []
        for split in ('train', 'valid'):
            subset = [r for r in records if r['split'] == split]
            if not subset:
                continue
            dataset = OvernightImageDataset(subset, image_root, training=training and split == 'train',
                                             image_size=image_size, seed=seed)
            if training and split == 'valid':
                # Keep valid/ paths, targets and identities. Only augmentation is
                # enabled after the validation split has explicitly become training.
                dataset.training = True
                dataset.transform = _transforms(True, seed)
            datasets.append(dataset)
        super().__init__(datasets)
        self.records = [r for dataset in datasets for r in dataset.records]
        counts = Counter(r['study_key'] for r in self.records)
        scale = len(self.records) / len(counts)
        for dataset in datasets:
            dataset.loss_weights = [scale / counts[r['study_key']] for r in dataset.records]

    def reseed(self, seed):
        for index, dataset in enumerate(self.datasets):
            dataset.reseed((seed + index * 104729) % 2**32)


def collate(samples):
    if not samples or any(s['targets'] is None for s in samples):
        raise ValueError('A refit batch must contain original known labels')
    return dict(images=torch.stack([s['image'] for s in samples]),
                targets=torch.stack([s['targets'] for s in samples]),
                image_ids=[s['image_id'] for s in samples],
                loss_weights=torch.tensor([s['loss_weight'] for s in samples], dtype=torch.float32))


def loader(records, config, *, training, epoch=0):
    seed = (config['seed'] + epoch * 1009) % 2**32
    dataset = RefitDataset(records, config['image_root'], image_size=config['image_size'],
                          training=training, seed=seed,
                          valid_is_training=config['include_valid'])
    workers = config['workers']
    options = dict(num_workers=workers, pin_memory=True,
                   generator=torch.Generator().manual_seed(seed), worker_init_fn=_seed_worker)
    if workers:
        options.update(prefetch_factor=2, persistent_workers=False,
                       multiprocessing_context='spawn', timeout=180.)
    return DataLoader(dataset, batch_size=config['batch_size'] if training else 4,
                      shuffle=training, drop_last=False, collate_fn=collate, **options)
