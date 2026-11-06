"""Read-only preparation for the raw-data Tiny768 one-week experiment.

The historical cache is validated against its original clean manifest and fold
plan without relaxing that validator. A separate mapping binds its 30,328 rows
to all 30,344 official raw train images. Missing targets receive no KD; they must
receive coefficient-one BCE. No model is created and no test CSV is accessed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path('/home/featurize/data/raw_rebuild_20261105')
LEGACY_ROOT = ROOT.parent / 'cloud_project_v2'


def sha256_file(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_raw_records(path: str | Path, split: str) -> list[dict[str, Any]]:
    """Read raw manifests without the old cache-path/clean-exclusion defaults."""
    if split not in {'train', 'valid'}:
        raise ValueError('Only original train and valid are accepted')
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Malformed raw manifest columns')
        result = list(reader)
    if not result:
        raise ValueError('Empty raw manifest')
    seen, labels = set(), {}
    for row in result:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed raw manifest row')
        if row.get('split') != split or not row.get('image_id') or row['image_id'] in seen:
            raise ValueError('Duplicate/invalid original image identity or split')
        seen.add(row['image_id'])
        subject, study, dicom = (row.get(key) for key in ('subject_id', 'study_id', 'dicom_id'))
        if not all(isinstance(value, str) and value for value in (subject, study, dicom)):
            raise ValueError('Missing original subject/Study/image identity')
        if row.get('study_key') != f'{subject}/{study}':
            raise ValueError('Study identity differs from original subject/Study')
        relative = Path(row.get('relative_path', ''))
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 4
                or relative.parts[:3] != (split, subject, study) or relative.stem != dicom):
            raise ValueError('Original path differs from original image identity')
        target = json.loads(row.get('targets', 'null'))
        if (not isinstance(target, list) or len(target) != 10
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or value not in (0, 1) for value in target)):
            raise ValueError('Original labels must be ten binary targets')
        row['targets'] = tuple(int(value) for value in target)
        if labels.setdefault(row['study_key'], row['targets']) != row['targets']:
            raise ValueError('Conflicting original labels within a Study')
        if row.get('status') != 'ok':
            raise ValueError('Unresolved original image status')
        if split == 'train' and str(row.get('training_eligible', '')).lower() != 'true':
            raise ValueError('Raw train rows must all remain eligible')
        if json.loads(row.get('exclusion_reasons', '[]')):
            raise ValueError('Raw manifest must not apply image exclusions')
        for name in ('file_sha256', 'pixel_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row.get(name, '')):
                raise ValueError('Missing original image SHA256 binding: ' + name)
    return result


def adapt_cache(raw_records: Sequence[dict[str, Any]], legacy_records: Sequence[dict[str, Any]],
                cache: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Map a strictly validated legacy cache without fabricating missing logits."""
    raw = {row['image_id']: row for row in raw_records}
    old = {row['image_id']: row for row in legacy_records}
    if len(raw) != len(raw_records) or len(old) != len(legacy_records):
        raise ValueError('Duplicate image identity in raw/cache source')
    if not set(old) <= set(raw) or set(cache.index) != set(old):
        raise ValueError('Legacy cache/source is not an exact subset of raw train')
    # The two audits use different pixel-hash definitions. Equal accepted file
    # SHA256 binds exactly the same original image bytes; each pixel hash stays
    # independently bound to its own unchanged audit/cache manifest.
    identity_fields = ('split', 'subject_id', 'study_id', 'study_key', 'dicom_id',
                       'relative_path', 'file_sha256')
    patient_folds: dict[str, int] = {}
    for image_id, row in raw.items():
        fold = plan['study_to_fold'].get(row['study_key'])
        if type(fold) is not int or fold not in range(5):
            raise ValueError('Raw Study is not bound to an existing patient fold')
        if patient_folds.setdefault(row['subject_id'], fold) != fold:
            raise ValueError('Patient spans legacy teacher folds')
        if image_id in old:
            previous = old[image_id]
            if any(row.get(name) != previous.get(name) for name in identity_fields):
                raise ValueError('Raw and validated cache image identities/SHA differ: ' + image_id)
            if tuple(row['targets']) != tuple(previous['targets']):
                raise ValueError('Raw and validated cache labels differ: ' + image_id)
    positions = np.asarray([cache.index.get(row['image_id'], -1) for row in raw_records],
                           dtype=np.int64)
    mask = positions >= 0
    positions.setflags(write=False)
    mask.setflags(write=False)
    missing = [row['image_id'] for row, available in zip(raw_records, mask) if not available]
    return {'cache_indices': positions, 'kd_mask': mask, 'missing_image_ids': missing,
            'raw_index': {row['image_id']: index for index, row in enumerate(raw_records)}}


def verify_official_identity(records: Sequence[dict[str, Any]], image_root: Path,
                             split: str) -> None:
    """Reparse only the corresponding official CSV and retain its exact row order."""
    from raw_rebuild.prepare_data import load_official
    official = load_official(image_root, split)
    if len(official) != len(records):
        raise ValueError('Raw manifest does not preserve the complete official CSV')
    names = ('image_id', 'split', 'subject_id', 'study_id', 'study_key', 'dicom_id',
             'relative_path', 'label_text', 'view_position', 'patient_orientation',
             'rows_csv', 'columns_csv')
    for row, source in zip(records, official):
        if any(row.get(name) != source.get(name) for name in names):
            raise ValueError('Raw manifest differs from official row identity/metadata')
        if tuple(row['targets']) != tuple(json.loads(source['targets'])):
            raise ValueError('Raw manifest changed official labels')
        path = image_root / row['relative_path']
        if not path.is_file():
            raise FileNotFoundError('Missing official image: ' + str(path))
        if path.stat().st_size != int(row['file_bytes']):
            raise ValueError('Official image size changed: ' + str(path))


def prepare(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return records, validated legacy cache, explicit raw KD mapping, bindings.

    Keys: train_manifest, valid_manifest, raw_audit, fold_plan, soft_labels,
    image_root. Defaults point at frozen raw/041 source artifacts. Original
    images are checked for existence and accepted length here; the unchanged
    OvernightImageDataset verifies their accepted file SHA256 when reading.
    """
    config = dict(config or {})
    paths = {
        'train_manifest': Path(config.get('train_manifest', RAW_ROOT / 'source_train.csv')),
        'valid_manifest': Path(config.get('valid_manifest', RAW_ROOT / 'valid.csv')),
        'raw_audit': Path(config.get('raw_audit', RAW_ROOT / 'audit.json')),
        'fold_plan': Path(config.get('fold_plan', LEGACY_ROOT / 'data/cv5_overnight/folds.json')),
        'soft_labels': Path(config.get('soft_labels', ROOT.parent /
            'runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2')),
        'image_root': Path(config.get('image_root', '/home/featurize/data/cxr_first')),
    }
    paths = {key: path.resolve(strict=True) for key, path in paths.items()}
    audit = json.loads(paths['raw_audit'].read_text())
    if (audit.get('original_files_modified') is not False
            or audit.get('excluded_rows') != 0 or audit.get('source_rows_preserved') is not True):
        raise ValueError('Raw preparation audit does not certify unfiltered official data')
    bindings: list[tuple[Path, str]] = []
    for key in ('train_manifest', 'valid_manifest'):
        path = paths[key]
        actual = sha256_file(path)
        if audit['manifest_sha256'].get(path.name) != actual:
            raise ValueError('Frozen raw manifest SHA256 changed: ' + key)
        bindings.append((path, actual))
    for split in ('train', 'valid'):
        path = paths['image_root'] / f'{split}.csv'
        actual = sha256_file(path)
        if audit['official_csv_sha256'].get(split) != actual:
            raise ValueError('Official original CSV changed: ' + split)
        bindings.append((path, actual))
    train = read_raw_records(paths['train_manifest'], 'train')
    valid = read_raw_records(paths['valid_manifest'], 'valid')
    expected = ((train, 30344, 18513, 11055), (valid, 1422, 859, 228))
    for records, images, studies, patients in expected:
        counts = (len(records), len({row['study_key'] for row in records}),
                  len({row['subject_id'] for row in records}))
        if counts != (images, studies, patients):
            raise ValueError('Official original split count mismatch: ' + str(counts))
    for key in ('image_id', 'study_key', 'subject_id'):
        if {row[key] for row in train} & {row[key] for row in valid}:
            raise ValueError('Official train/valid identity overlap: ' + key)
    for split, records in (('train', train), ('valid', valid)):
        verify_official_identity(records, paths['image_root'], split)

    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from distill_tta_data import load_soft_labels
    plan = read_fold_plan(paths['fold_plan'])
    legacy_path = Path(plan['source_train_manifest']['path'])
    legacy_records = read_manifest(legacy_path, split='train')
    # Deliberately validate against all original legacy records, never raw or a
    # subset: every existing fold/teacher/patient/hash assertion remains active.
    cache = load_soft_labels(paths['soft_labels'], paths['fold_plan'], legacy_records)
    mapping = adapt_cache(train, legacy_records, cache, plan)
    if len(legacy_records) != 30328 or int(mapping['kd_mask'].sum()) != 30328:
        raise ValueError('Legacy OOF target coverage differs from the accepted 30,328 rows')
    if len(mapping['missing_image_ids']) != 16:
        raise ValueError('Unexpected raw images without legacy OOF targets')
    for path in (paths['raw_audit'], paths['fold_plan'], legacy_path,
                 Path(plan['official_holdout']['path']),
                 paths['soft_labels'] / 'soft_labels.npz',
                 paths['soft_labels'] / 'metadata.json'):
        bindings.append((path, sha256_file(path)))
    missing_rows = [train[mapping['raw_index'][image_id]] for image_id in mapping['missing_image_ids']]
    report = {
        'schema': 'one_week_raw_legacy_oof_adapter_v1',
        'train_images': len(train), 'train_studies': 18513, 'train_patients': 11055,
        'valid_images': len(valid), 'valid_studies': 859, 'valid_patients': 228,
        'kd_images': int(mapping['kd_mask'].sum()), 'bce_only_images': len(missing_rows),
        'bce_only_studies': len({row['study_key'] for row in missing_rows}),
        'bce_only_patients': len({row['subject_id'] for row in missing_rows}),
        'bce_only_image_ids': mapping['missing_image_ids'],
        'missing_kd_policy': 'coefficient_one_hard_bce; zero_kd',
        'legacy_cache_validation_unchanged': True,
        'cross_manifest_image_binding': 'identical_original_file_sha256',
        'pixel_hash_definitions_differ': True,
        'original_train_valid_patients_disjoint': True,
        'official_csv_row_order_labels_paths_preserved': True,
        'all_original_train_images_retained': True, 'excluded_rows': 0,
        'image_file_sha_check': 'unchanged_OvernightImageDataset_on_every_read',
        'test_accessed': False,
        'bindings': [{'path': str(path), 'sha256': digest} for path, digest in bindings],
    }
    return {'records': train, 'train_records': train, 'valid_records': valid,
            'cache': cache, 'plan': plan, 'bindings': bindings, 'audit': report, **mapping}
