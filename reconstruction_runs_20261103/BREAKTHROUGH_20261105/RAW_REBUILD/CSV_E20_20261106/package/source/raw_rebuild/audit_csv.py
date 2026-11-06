"""Read-only integrity audit of a four-column complete-Study submission.

No images or models are loaded. Test identities and saved probabilities are
read only to verify coverage and serialization, never to select or fit a model.
NPZ class columns must follow the registered order 1 through 10.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
CLASS_TEXT = tuple(str(i) for i in range(1, 11))
DEFAULT_TOLERANCE = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or '/' in value or '\\' in value):
        raise ValueError(f'Invalid unmodified string identity: {name}')
    return value


def _official_studies(path):
    studies, study_patients, image_ids = {}, {}, set()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Dicom_id', 'Subject_id', 'Study_id'}
        fields = reader.fieldnames
        if (fields is None or len(fields) != len(set(fields))
                or not required <= set(fields)):
            raise ValueError('Official test CSV has missing or duplicate identity columns')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed official test row')
            patient, study, image = (_identity(row[k], k) for k in
                                     ('Subject_id', 'Study_id', 'Dicom_id'))
            if image in image_ids:
                raise ValueError('Official test CSV repeats a Dicom_id')
            image_ids.add(image)
            if study_patients.setdefault(study, patient) != patient:
                raise ValueError('Official Study_id belongs to multiple patients')
            studies[(patient, study)] = studies.get((patient, study), 0) + 1
    if not studies:
        raise ValueError('Official test CSV is empty')
    return studies, len(image_ids)


def _submission(path, expected):
    values, count = {}, 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError('Submission must have the exact four-column header/order')
        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Submission row has an unexpected number of columns')
            identity = (_identity(row['Subject_id'], 'Subject_id'),
                        _identity(row['Study_id'], 'Study_id'))
            if identity not in expected:
                raise ValueError('Submission contains unknown Study or wrong patient identity')
            category = row['Predict_class']
            if category not in CLASS_TEXT:
                raise ValueError('Predict_class must be canonical integer text 1 through 10')
            try:
                probability = float(row['Probability'])
            except (ValueError, TypeError) as exc:
                raise ValueError('Probability must be numeric') from exc
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Probability must be finite and in [0,1]')
            classes = values.setdefault(identity, {})
            if category in classes:
                raise ValueError('Duplicate Study/class row')
            classes[category] = probability
    if set(values) != set(expected):
        raise ValueError('Submission Study coverage differs from the official test CSV')
    if any(set(classes) != set(CLASS_TEXT) for classes in values.values()):
        raise ValueError('Every Study must contain all ten classes exactly once')
    if count != len(expected) * 10:
        raise ValueError('Submission must contain ten rows per official Study')
    return {key: np.asarray([classes[c] for c in CLASS_TEXT], dtype=np.float64)
            for key, classes in values.items()}, count


def _npz_agreement(path, submitted, tolerance):
    with np.load(path, allow_pickle=False) as archive:
        required = {'probabilities', 'study_keys', 'patient_ids'}
        if not required <= set(archive.files):
            raise ValueError('Study NPZ requires probabilities, study_keys, patient_ids')
        probabilities = archive['probabilities']
        keys, patients = archive['study_keys'], archive['patient_ids']
        if (probabilities.shape != (len(submitted), 10)
                or keys.shape != (len(submitted),) or patients.shape != keys.shape):
            raise ValueError('Study NPZ arrays have invalid coverage or shape')
        if probabilities.dtype.kind not in 'fiu':
            raise ValueError('Study NPZ probabilities must be real numeric values')
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Study NPZ probabilities must be finite and in [0,1]')
        if keys.dtype.kind != 'U' or patients.dtype.kind != 'U':
            raise ValueError('Study NPZ identities must be Unicode strings')
        if 'class_ids' in archive.files and not np.array_equal(archive['class_ids'], np.arange(1, 11)):
            raise ValueError('Study NPZ class_ids must be ordered 1 through 10')
        reference = {}
        for index, (key, patient) in enumerate(zip(keys.tolist(), patients.tolist())):
            _identity(patient, 'NPZ patient_ids')
            parts = key.split('/')
            if len(parts) != 2 or parts[0] != patient:
                raise ValueError('Study NPZ key/patient disagreement')
            _identity(parts[1], 'NPZ Study_id')
            identity = (patient, parts[1])
            if identity in reference:
                raise ValueError('Study NPZ repeats a Study')
            reference[identity] = probabilities[index].astype(np.float64, copy=False)
        if set(reference) != set(submitted):
            raise ValueError('Study NPZ coverage differs from submission')
        difference = np.stack([np.abs(submitted[key] - reference[key]) for key in submitted])
        maximum = float(difference.max())
        mismatches = int((difference > tolerance).sum())
        if mismatches:
            raise ValueError(f'CSV/NPZ probability mismatch: {mismatches} cells, max error {maximum}')
    return dict(checked=True, compared_studies=len(submitted), compared_probabilities=difference.size,
                max_absolute_error=maximum, absolute_tolerance=tolerance,
                mismatched_probabilities=mismatches, exact_float64_values=bool((difference == 0).all()),
                class_column_order=list(range(1, 11)))


def audit_csv(csv_path, test_csv, study_predictions=None, *, expected_images=2218,
              expected_studies=1414, absolute_tolerance=DEFAULT_TOLERANCE):
    """Return a passing report or raise; inputs are never modified.

    Count overrides are for synthetic tests or a separately declared official
    dataset. The command-line interface always requires 2218 images/1414 Studies.
    Optional NPZ uses probabilities[N,10], study_keys[N], patient_ids[N].
    Its rows may be reordered; keys must be unmodified ``subject/study`` strings.
    A caller may tighten, but cannot relax, the 1e-12 serialization tolerance.
    """
    if (isinstance(absolute_tolerance, bool) or not isinstance(absolute_tolerance, Real)
            or not math.isfinite(absolute_tolerance)
            or not 0 <= absolute_tolerance <= DEFAULT_TOLERANCE):
        raise ValueError('Probability tolerance must be finite and in [0,1e-12]')
    absolute_tolerance = float(absolute_tolerance)
    paths = {'submission': Path(csv_path).resolve(strict=True),
             'official_test_csv': Path(test_csv).resolve(strict=True)}
    if study_predictions is not None:
        paths['study_predictions'] = Path(study_predictions).resolve(strict=True)
    initial = {name: sha256_file(path) for name, path in paths.items()}
    studies, image_count = _official_studies(paths['official_test_csv'])
    if image_count != expected_images or len(studies) != expected_studies:
        raise ValueError('Official test image/Study counts do not match the required dataset')
    probabilities, row_count = _submission(paths['submission'], studies)
    agreement = (_npz_agreement(paths['study_predictions'], probabilities, absolute_tolerance)
                 if study_predictions is not None else dict(checked=False,
                     reason='No Study NPZ supplied; this audit does not verify model inference'))
    if any(sha256_file(paths[name]) != value for name, value in initial.items()):
        raise RuntimeError('An input changed during the read-only audit')
    return dict(schema_version=1, complete=True, passed=True,
        recorded_at_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        inputs={name: dict(path=str(path), sha256=initial[name], bytes=path.stat().st_size)
                for name, path in paths.items()},
        header=HEADER, images=image_count, studies=len(studies),
        patients=len({patient for patient, _ in studies}), submission_rows=row_count,
        original_string_ids_preserved=True, exact_study_coverage=True,
        all_ten_classes_once=True, duplicate_study_class_rows=0,
        all_probabilities_finite_in_0_1=True, study_prediction_agreement=agreement,
        audit_source_sha256=sha256_file(__file__),
        images_read=False, model_inference_performed=False, training_performed=False,
        parameters_fitted=False, test_metrics_computed=False,
        formal_cost_consistency_certified=False, formal_T4_timing_certified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', dest='csv_path', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--study-predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    inputs = {p.resolve() for p in (args.csv_path, args.test_csv, args.study_predictions) if p is not None}
    if output.exists() or output in inputs:
        raise FileExistsError('Choose a new audit report path separate from every input')
    report = audit_csv(args.csv_path, args.test_csv, args.study_predictions)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
