"""Complete original Studies for the new nested teacher/student experiment.

Importing this module does not import torch, read images, or construct models.
No old cleaning exclusions or soft-label caches enter this data path.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Sequence


def validate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError('A complete-Study manifest must not be empty')
    result, seen, studies, owners = [], set(), {}, {}
    for source in records:
        row = dict(source)
        if row.get('split') != 'train':
            raise ValueError('This experiment accepts original official train only')
        for field in ('image_id', 'subject_id', 'study_id', 'study_key', 'dicom_id', 'relative_path'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Missing original identity: {field}')
        if row['image_id'] in seen:
            raise ValueError('Duplicate original image identity')
        seen.add(row['image_id'])
        if row['study_key'] != f"{row['subject_id']}/{row['study_id']}":
            raise ValueError('Study identity does not match patient')
        if owners.setdefault(row['study_id'], row['subject_id']) != row['subject_id']:
            raise ValueError('One Study_id has multiple patients')
        relative = Path(row['relative_path'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 4
                or relative.parts != ('train', row['subject_id'], row['study_id'], row['dicom_id'] + '.jpg')):
            raise ValueError('Original image path is outside its declared Study')
        target = row.get('targets')
        if isinstance(target, str):
            target = json.loads(target)
        if (not isinstance(target, (tuple, list)) or len(target) != 10
                or any(type(x) not in (int, float) or x not in (0, 1) for x in target)):
            raise ValueError('Targets must be the original ten binary Study labels')
        target = tuple(int(x) for x in target)
        if studies.setdefault(row['study_key'], target) != target:
            raise ValueError('Within-Study original labels disagree')
        if row.get('label_text') is not None:
            label_ids = [int(x) for x in row['label_text'].split(';')]
            if (len(label_ids) != len(set(label_ids)) or any(x not in range(1, 11) for x in label_ids)
                    or tuple(int(i in label_ids) for i in range(1, 11)) != target):
                raise ValueError('Targets differ from original label_text')
        if (str(row.get('training_eligible', '')).lower() != 'true'
                or row.get('status') != 'ok' or row.get('exclusion_reasons') != '[]'):
            raise ValueError('Original rows cannot be silently excluded or unresolved')
        for field in ('file_sha256', 'pixel_sha256'):
            if re.fullmatch('[0-9a-f]{64}', str(row.get(field, ''))) is None:
                raise ValueError(f'Missing original integrity fingerprint: {field}')
        row['targets'] = target
        result.append(row)
    return result


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicate manifest columns')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed manifest row')
    return validate_records(rows)


class StudyDataset:
    """One item contains every image and the unchanged label of one Study.

    The tested original loader supplies EXIF orientation, full decoding,
    aspect-preserving LANCZOS letterbox, ImageNet normalization and mild
    contrast/brightness, rotation and horizontal-flip training augmentation.
    No image crops are enabled. First-read file SHA is mandatory per worker;
    missing files, decode errors and changed content raise, never skip.
    Rebuild DataLoader workers after set_epoch; persistent workers are unsupported.
    """
    def __init__(self, manifest, image_root, *, image_size=768, training=False,
                 seed=20261106, augment=True):
        from reconstruction.data import StudyDataset as OriginalStudyDataset

        records = load_manifest(manifest) if isinstance(manifest, (str, Path)) else validate_records(manifest)
        self._dataset = OriginalStudyDataset(records, image_root, image_size=image_size,
            training=training, seed=seed, local_tiles=False, verify_hash=True, augment=augment)

    def __getattr__(self, name):
        # Keep the existing training/evaluation helpers' read-only API available.
        if name == '_dataset':
            raise AttributeError(name)
        return getattr(self._dataset, name)

    @property
    def groups(self):
        return self._dataset.groups

    @groups.setter
    def groups(self, value):
        self._dataset.groups = value

    @property
    def image_size(self):
        return self._dataset.image_size

    @image_size.setter
    def image_size(self, value):
        if type(value) is not int or value < 1:
            raise ValueError('Input size must be a positive integer')
        self._dataset.image_size = value

    def __len__(self):
        return len(self._dataset)

    def set_epoch(self, epoch):
        self._dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self._dataset[index]
        indices = self._dataset.groups[index]
        sample.update(patient_id=sample['subject_id'],
            paths=[str(self._dataset.paths[i]) for i in indices],
            relative_paths=[self._dataset.records[i]['relative_path'] for i in indices])
        return sample


def collate_studies(samples):
    return list(samples)
