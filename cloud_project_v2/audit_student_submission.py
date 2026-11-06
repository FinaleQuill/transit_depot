"""Independently audit a fixed student's complete test CSV, without inference.

This reads identities and probabilities only to check coverage, numeric validity,
and exact Study means. It never computes class priors, fits parameters, chooses
models, or reports test probability distributions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np

OFFICIAL_TEST_MANIFEST_SHA256 = 'fdfd651ff0663b3dd6f232ecbb0cd32949d78092e162ae2595b4563cdbe224f3'
SUBMISSION_HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
IMAGE_HEADER = ['Image_id', 'Subject_id', 'Study_id', 'Predict_class', 'Probability']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image_id', 'split', 'subject_id', 'study_id', 'study_key',
                    'relative_path', 'targets', 'file_sha256'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Original test manifest is missing required columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Original test manifest is empty')
    ids, paths = set(), set()
    for row in rows:
        if (row['split'] != 'test' or any(not row[key] for key in ('image_id', 'subject_id', 'study_id'))
                or row['study_key'] != row['subject_id'] + '/' + row['study_id']):
            raise ValueError('Original test identity or split is invalid')
        if row['targets'].strip() not in ('', 'null', 'None'):
            raise ValueError('Test targets must remain unknown')
        relative = Path(row['relative_path'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 4
                or relative.parts[:3] != ('test', row['subject_id'], row['study_id'])):
            raise ValueError('Original test relative path differs from its Study identity')
        digest = row['file_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Original image SHA256 is missing')
        if row['image_id'] in ids or row['relative_path'] in paths:
            raise ValueError('Original test manifest repeats an image')
        ids.add(row['image_id'])
        paths.add(row['relative_path'])
    return rows


def read_probabilities(path, expected_header, expected_identities, *, image_level):
    """Parse exact identities/classes once, independently of inference helpers."""
    values, count = {}, 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ValueError('Prediction CSV has an unexpected header')
        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Prediction row has an unexpected number of columns')
            try:
                class_id = int(row['Predict_class'])
                probability = float(row['Probability'])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError('Prediction row has an invalid class/probability') from error
            if class_id not in range(1, 11):
                raise ValueError('Prediction row has an unknown class')
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Prediction probability is nonfinite or outside [0,1]')
            study = (row['Subject_id'], row['Study_id'])
            if image_level:
                identity = row['Image_id']
                if identity not in expected_identities or expected_identities[identity] != study:
                    raise ValueError('Image prediction has an unknown image or mismatched Study')
            else:
                identity = study
                if identity not in expected_identities:
                    raise ValueError('Submission contains an unexpected Study')
            existing = values.setdefault(identity, {})
            if class_id in existing:
                raise ValueError('Prediction contains a duplicate identity/class row')
            existing[class_id] = probability
    if set(values) != set(expected_identities):
        raise ValueError('Prediction coverage differs from the original test manifest')
    if any(set(classes) != set(range(1, 11)) for classes in values.values()):
        raise ValueError('Every image/Study must retain all ten classes')
    return {identity: np.asarray([classes[i] for i in range(1, 11)], dtype=np.float64)
            for identity, classes in values.items()}, count


def check_original_images(records, image_root):
    image_root = Path(image_root).resolve(strict=True)
    root = next((candidate.resolve() for candidate in (image_root, image_root / 'cxr_first')
                 if (candidate / 'test').is_dir()), None)
    if root is None:
        raise FileNotFoundError('Original test split is missing')
    for row in records:
        path = (root / row['relative_path']).resolve(strict=True)
        if not path.is_relative_to(root / 'test') or not path.is_file():
            raise ValueError('Original test image path escapes the test root')
        if sha256_file(path) != row['file_sha256']:
            raise ValueError('Original test image SHA256 differs from its frozen manifest')
    return {'checked': len(records), 'sha256_matches': len(records), 'mismatches': 0,
            'resolved_image_root': str(root)}


def audit_submission(*, manifest, image_predictions, submission, image_root=None,
                     check_image_sha=False, expected_images=2218, expected_studies=1414,
                     expected_manifest_sha256=None):
    manifest, image_predictions, submission = map(Path, (manifest, image_predictions, submission))
    original_hashes = {str(path.resolve()): sha256_file(path)
                       for path in (manifest, image_predictions, submission)}
    if (expected_manifest_sha256 is not None
            and original_hashes[str(manifest.resolve())] != expected_manifest_sha256):
        raise ValueError('Test manifest SHA256 differs from the frozen official manifest')
    records = manifest_records(manifest)
    identities = {row['image_id']: (row['subject_id'], row['study_id']) for row in records}
    studies = set(identities.values())
    if len(records) != expected_images or len(studies) != expected_studies:
        raise ValueError('Official image/Study coverage does not match the required counts')
    if check_image_sha and image_root is None:
        raise ValueError('--check-image-sha requires --image-root')
    originals = check_original_images(records, image_root) if check_image_sha else {
        'checked': 0, 'reason': 'Not requested; no raw test images were read by this audit'}
    images, image_rows = read_probabilities(image_predictions, IMAGE_HEADER, identities, image_level=True)
    submitted, submission_rows = read_probabilities(submission, SUBMISSION_HEADER, studies, image_level=False)
    # Follow ORIGINAL MANIFEST order, not CSV row order or a sorted permutation.
    # This matches the deployed FP64 arithmetic while independently rebuilding
    # the sums; no inference accumulator or prediction implementation is called.
    sums, counts = {}, {}
    for row in records:
        key = identities[row['image_id']]
        if key not in sums:
            sums[key] = np.zeros(10, dtype=np.float64)
            counts[key] = 0
        sums[key] += images[row['image_id']]
        counts[key] += 1
    mismatched = 0
    max_error = 0.
    for study in studies:
        expected = sums[study] / counts[study]
        actual = submitted[study]
        max_error = max(max_error, float(np.abs(expected - actual).max()))
        if expected.tobytes() != actual.tobytes():
            mismatched += 1
    if mismatched or max_error != 0:
        raise ValueError(f'Study mean mismatch: {mismatched} Studies, max absolute error {max_error}')
    if image_rows != expected_images * 10 or submission_rows != expected_studies * 10:
        raise ValueError('Prediction row counts do not match full ten-class coverage')
    if any(sha256_file(path) != original_hashes[str(path.resolve())]
           for path in (manifest, image_predictions, submission)):
        raise RuntimeError('An audit input changed during read-only verification')
    return {'complete': True, 'time_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
            'manifest_sha256': original_hashes[str(manifest.resolve())],
            'submission_sha256': original_hashes[str(submission.resolve())],
            'image_predictions_sha256': original_hashes[str(image_predictions.resolve())],
            'images': len(records), 'studies': len(studies),
            'patients': len({row['subject_id'] for row in records}),
            'submission_data_rows': submission_rows, 'image_prediction_data_rows': image_rows,
            'submission_header': SUBMISSION_HEADER, 'image_predictions_header': IMAGE_HEADER,
            'all_classes_present_once': True, 'all_probabilities_finite_in_0_1': True,
            'unknown_test_targets_preserved': True,
            'study_aggregation': {'dtype': 'float64', 'order': 'original_manifest_image_order',
                                  'bitwise_identical': True, 'mismatched_studies': 0,
                                  'max_absolute_error': max_error},
            'original_image_integrity': originals,
            'audit_source_sha256': sha256_file(__file__),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
            'probability_distributions_analyzed': False, 'parameters_fitted': False,
            'test_metric_computed': False, 'formal_T4_certification': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'image-predictions', 'submission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--check-image-sha', action='store_true')
    args = vars(parser.parse_args(argv))
    output = args.pop('output').resolve()
    if output.exists() or output in {args[name].resolve() for name in ('manifest', 'image_predictions', 'submission')}:
        raise FileExistsError('Choose a new, separate audit report path')
    report = audit_submission(**args, expected_manifest_sha256=OFFICIAL_TEST_MANIFEST_SHA256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'images': report['images'], 'studies': report['studies'],
                      'submission_data_rows': report['submission_data_rows'],
                      'study_mean_max_absolute_error': 0., 'audit_report': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
