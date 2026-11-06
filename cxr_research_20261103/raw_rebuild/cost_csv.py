"""Replay complete test Studies with the exact exported CSV predictor.

Model loading and identity/SHA audits precede timing. Each measured unit includes
original-image reads, preprocessing, inference, aggregation and ten CSV records
flushed to the output stream. Sample runs never certify official T4 acceptance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import numpy as np

HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or value in {'.', '..'} or '/' in value or '\\' in value):
        raise ValueError(f'Invalid original string identity: {name}')
    return value


def official_studies(test_csv):
    """Keep the official Study and within-Study image order, without filtering."""
    groups, seen_images, patients = {}, set(), {}
    with Path(test_csv).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if (fields is None or len(fields) != len(set(fields))
                or not {'Subject_id', 'Study_id', 'Dicom_id'} <= set(fields)):
            raise ValueError('Official test CSV has missing/duplicate identity columns')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed official test CSV row')
            patient, study, dicom = (_identity(row[key], key) for key in
                                     ('Subject_id', 'Study_id', 'Dicom_id'))
            if dicom in seen_images:
                raise ValueError(f'Duplicate official Dicom_id: {dicom}')
            seen_images.add(dicom)
            if patients.setdefault(study, patient) != patient:
                raise ValueError('Official Study_id belongs to multiple patients')
            key = f'{patient}/{study}'
            groups.setdefault(key, []).append(dict(subject_id=patient, study_id=study,
                study_key=key, dicom_id=dicom, relative_path=f'{key}/{dicom}.jpg'))
    if not groups:
        raise ValueError('Official test CSV is empty')
    return groups


def complete_studies(images, official, original_root):
    """Reject unknown/duplicate/partial Studies and compare original-file SHA256."""
    images, original_root = Path(images).resolve(strict=True), Path(original_root).resolve(strict=True)
    if not images.is_dir() or not original_root.is_dir():
        raise ValueError('Image roots must be existing directories')
    observed, study_directories, seen_dicoms = {}, set(), set()
    for path in sorted(images.rglob('*')):
        relative = path.relative_to(images)
        if not path.resolve(strict=True).is_relative_to(images):
            raise ValueError(f'Image path escapes its input root: {relative}')
        if path.is_dir():
            if len(relative.parts) == 1:
                _identity(relative.name, 'image directory subject')
            elif len(relative.parts) == 2:
                key = relative.as_posix()
                if key not in official:
                    raise ValueError(f'Unknown Study directory: {key}')
                study_directories.add(key)
            else:
                raise ValueError(f'Unexpected nested image directory: {relative}')
            continue
        if not path.is_file() or len(relative.parts) != 3 or relative.suffix != '.jpg':
            raise ValueError(f'Expected subject/study/dicom.jpg: {relative}')
        patient, study, filename = relative.parts
        key, dicom = f'{patient}/{study}', Path(filename).stem
        if key not in official or dicom not in {row['dicom_id'] for row in official[key]}:
            raise ValueError(f'Image is absent from official test identities: {relative}')
        if dicom in seen_dicoms:
            raise ValueError(f'Duplicate Dicom_id in timing images: {dicom}')
        seen_dicoms.add(dicom)
        observed.setdefault(key, set()).add(dicom)
    if not study_directories:
        raise ValueError('Timing image directory contains no complete Studies')
    groups = []
    for key, source_records in official.items():
        if key not in study_directories:
            continue
        expected = {row['dicom_id'] for row in source_records}
        if observed.get(key, set()) != expected:
            missing = sorted(expected - observed.get(key, set()))
            raise ValueError(f'Incomplete Study {key}; missing original views: {missing}')
        records = []
        for source in source_records:
            row = dict(source)
            relative = Path(row['relative_path'])
            original = original_root / 'test' / relative
            actual = sha256_file(images / relative)
            if actual != sha256_file(original):
                raise ValueError(f'Timing image differs from the official original: {relative}')
            row['file_sha256'] = actual
            records.append(row)
        groups.append(records)
    return groups


def read_submission(path, official):
    """Require the complete, full-precision main submission with all ten classes."""
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError('Main submission must have the exact four-column header/order')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed main submission row')
            patient = _identity(row['Subject_id'], 'submission Subject_id')
            study = _identity(row['Study_id'], 'submission Study_id')
            key = f'{patient}/{study}'
            if key not in official:
                raise ValueError(f'Submission Study identity is not official: {key}')
            if row['Predict_class'] not in {str(index) for index in range(1, 11)}:
                raise ValueError('Submission classes must be canonical integer text 1 through 10')
            category = int(row['Predict_class'])
            value = float(row['Probability'])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Submission probabilities must be finite and in [0,1]')
            classes = result.setdefault(key, {})
            if category in classes:
                raise ValueError(f'Duplicate Study/class in main submission: {key}/{category}')
            classes[category] = value
    if set(result) != set(official):
        raise ValueError('Main submission does not cover the complete official test set')
    if any(set(values) != set(range(1, 11)) for values in result.values()):
        raise ValueError('Every official Study must have exactly ten submission classes')
    return {key: np.asarray([values[index] for index in range(1, 11)], dtype=np.float64)
            for key, values in result.items()}


def compare_probabilities(actual, reference):
    """Check raw tolerance plus four-decimal probabilities, binaries and both ranks."""
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (actual.ndim != 2 or actual.shape[1] != 10 or actual.shape != reference.shape
            or actual.shape[0] == 0 or not np.isfinite(actual).all()
            or not np.isfinite(reference).all()
            or ((actual < 0) | (actual > 1)).any()
            or ((reference < 0) | (reference > 1)).any()):
        raise ValueError('Comparison requires finite probability arrays of matching [Studies,10] shape')
    rounded, expected = np.round(actual, 4), np.round(reference, 4)
    maximum = float(np.max(np.abs(actual - reference)))
    # Pairwise signs preserve ties explicitly, independent of argsort tie behavior.
    within = bool(np.array_equal(np.sign(rounded[:, :, None] - rounded[:, None, :]),
                                 np.sign(expected[:, :, None] - expected[:, None, :])))
    across = bool(np.array_equal(np.sign(rounded[:, None, :] - rounded[None, :, :]),
                                 np.sign(expected[:, None, :] - expected[None, :, :])))
    checks = dict(raw_probability_tolerance=maximum <= 1e-3,
                  four_decimal_probabilities_identical=bool(np.array_equal(rounded, expected)),
                  four_decimal_binary_identical=bool(np.array_equal(rounded >= .5, expected >= .5)),
                  four_decimal_within_study_order_and_ties_identical=within,
                  four_decimal_across_study_order_and_ties_identical=across)
    return dict(checks=checks, passed=all(checks.values()),
                maximum_raw_probability_error=maximum, raw_absolute_tolerance=1e-3,
                rounding='numpy.round(float64, 4), nearest with ties to even',
                binary_threshold='rounded probability >= 0.5',
                compared_studies=int(actual.shape[0]), compared_probabilities=int(actual.size))


def write_round(predictor, groups, image_root, output, synchronize):
    """Do not reuse prepared tensors: image reads are inside every Study timer."""
    elapsed, values = [], []
    with Path(output).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(HEADER)
        stream.flush()
        for records in groups:
            synchronize()
            start = time.perf_counter()
            predicted = predictor.predict_study(records, Path(image_root))
            probability = np.asarray(predicted['blend'], dtype=np.float64)
            if (probability.shape != (10,) or not np.isfinite(probability).all()
                    or ((probability < 0) | (probability > 1)).any()):
                raise ValueError('Shared predictor returned invalid Study probabilities')
            identity = records[0]
            for index, value in enumerate(probability, start=1):
                writer.writerow([identity['subject_id'], identity['study_id'], index, repr(float(value))])
            stream.flush()
            synchronize()
            elapsed.append((time.perf_counter() - start) * 1000.)
            values.append(probability.copy())
    return elapsed, np.stack(values)


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='100 complete Study directories')
    parser.add_argument('--warmup', type=Path, required=True, help='five separate complete Study directories')
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True, help='main fixed blend submission')
    parser.add_argument('--output', type=Path, required=True, help='new report/round CSV directory')
    parser.add_argument('--original-root', type=Path, default=Path('/home/featurize/data/cxr_first'))
    parser.add_argument('--sample-example', action='store_true',
                        help='allow a non-100-Study sample replay; never official T4 certification')
    args = parser.parse_args(argv)
    for name in ('package', 'images', 'warmup', 'test_csv', 'submission', 'original_root'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; prior artifacts are retained')
    if any(args.output.is_relative_to(root) for root in
           (args.package, args.images, args.warmup, args.original_root)):
        raise ValueError('Cost output must remain outside model and image input directories')
    inputs = {name: dict(path=str(path), sha256=sha256_file(path)) for name, path in
              (('official_test_csv', args.test_csv), ('main_submission', args.submission),
               ('package_config', args.package / 'config.json'))}
    official = official_studies(args.test_csv)
    if len(official) != 1414 or sum(map(len, official.values())) != 2218:
        raise ValueError('Official test CSV must preserve all 1414 Studies and 2218 images')
    measured = complete_studies(args.images, official, args.original_root)
    warmup = complete_studies(args.warmup, official, args.original_root)
    if len(warmup) != 5 or (not args.sample_example and len(measured) != 100):
        raise ValueError('Require five warmup and 100 measured Studies; samples need --sample-example')
    keys = [records[0]['study_key'] for records in measured]
    warmup_keys = [records[0]['study_key'] for records in warmup]
    if set(keys) & set(warmup_keys):
        raise ValueError('Measured and warmup Studies must be disjoint')
    reference = read_submission(args.submission, official)
    expected = np.stack([reference[key] for key in keys])
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, complete=False,
        recorded_at_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        protocol='serial complete Study batch=1; five warmups; three rounds; Study-equal average',
        timing_boundary='before original-image reads through CSV rows written and stream flushed; CUDA synchronized',
        sample_example=bool(args.sample_example), measured_studies=len(measured),
        measured_images=sum(map(len, measured)), warmup_studies=len(warmup), warmup_images=sum(map(len, warmup)),
        measured_study_keys=keys, warmup_study_keys=warmup_keys,
        all_original_views_preserved=True, all_input_image_sha256_match_official_originals=True,
        input_files=inputs, package=str(args.package), training_performed=False,
        model_load_count=0, formal_T4_certification=False, organizer_acceptance_required=True,
        cost_source_sha256=sha256_file(__file__))
    try:
        import torch
        from raw_rebuild.export_csv import Predictor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this GPU cost replay')
        predictor = Predictor(args.package)
        report['model_load_count'] = 1
        report['loaded_asset_bytes'] = predictor.loaded_asset_bytes
        report['environment'] = dict(gpu_name=torch.cuda.get_device_name(),
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(), python_version=platform.python_version(),
            platform=platform.platform(), numpy_version=np.__version__)
        for records in warmup:
            predictor.predict_study(records, args.warmup)
        torch.cuda.synchronize()
        report['rounds'] = []
        for index in range(1, 4):
            output = args.output / f'round_{index}.csv'
            times, probabilities = write_round(predictor, measured, args.images, output,
                                                torch.cuda.synchronize)
            agreement = compare_probabilities(probabilities, expected)
            report['rounds'].append(dict(round=index, study_milliseconds=times,
                mean_ms_per_study=float(np.mean(times)), consistency=agreement,
                csv=dict(path=str(output), sha256=sha256_file(output))))
        for source in inputs.values():
            if sha256_file(source['path']) != source['sha256']:
                raise RuntimeError('An input CSV or package configuration changed during cost replay')
        for groups, root in ((measured, args.images), (warmup, args.warmup)):
            for records in groups:
                for record in records:
                    if sha256_file(root / record['relative_path']) != record['file_sha256']:
                        raise RuntimeError('A timing image changed during cost replay')
        mean = float(np.mean([item['study_milliseconds'] for item in report['rounds']]))
        consistent = all(item['consistency']['passed'] for item in report['rounds'])
        report.update(complete=True, consistency_pass=consistent, mean_ms_per_study=mean,
            observed_mean_within_100ms=mean <= 100.,
            official_sample_size_protocol_met=len(measured) == 100,
            maximum_raw_probability_error=max(item['consistency']['maximum_raw_probability_error']
                                                for item in report['rounds']),
            local_checks_pass=bool(consistent and mean <= 100.),
            status='sample_replay_only' if args.sample_example else 'local_replay_only_pending_official_acceptance')
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}', local_checks_pass=False)
        _write_json(args.output / 'cost_report.json', report)
        raise
    _write_json(args.output / 'cost_report.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report['local_checks_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
