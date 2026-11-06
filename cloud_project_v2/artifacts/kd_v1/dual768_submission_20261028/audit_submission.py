"""CPU-only completeness audit of the frozen 033 test submission.

Run only after the root agent confirms inference has ended. This script uses
only the standard library, never imports deployment/training modules, and never
loads test answers, measures disease distributions, or selects parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / 'deliveries/033_dual_tiny768_seed43_25_seed44_75'
HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
IMAGE_HEADER = ['Image_id'] + HEADER
CLASSES = set(range(1, 11))
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def snapshot_entries(document):
    """Accept a direct path mapping or conventional wrapped snapshot mappings."""
    for field in ('files', 'hashes', 'before', 'protected_files', 'protected'):
        if field in document and isinstance(document[field], (dict, list)):
            document = document[field]
            break
    if isinstance(document, list):
        document = {item['path']: item for item in document}
    require(isinstance(document, dict), 'Unsupported before-snapshot schema')
    result = {}
    for name, item in document.items():
        digest = item if isinstance(item, str) else item.get('sha256')
        require(isinstance(digest, str) and len(digest) == 64,
                'Before snapshot entry lacks SHA256')
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        result[path.resolve(strict=True)] = digest
    return result


def probabilities(path, expected, image=False):
    parsed = {}
    row_count = 0
    with path.open(encoding='utf-8', errors='strict', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        require(next(reader, None) == (IMAGE_HEADER if image else HEADER),
                'CSV header is not the exact required UTF8 header')
        for row in reader:
            require(len(row) == (5 if image else 4), 'CSV row has wrong column count')
            if image:
                image_id, subject, study, category, raw_probability = row
                require(image_id in expected and expected[image_id] == (subject, study),
                        'Image identity does not match the frozen manifest')
                key = image_id
            else:
                subject, study, category, raw_probability = row
                key = (subject, study)
                require(key in expected, 'Study identity does not match the frozen manifest')
            c = int(category)
            require(str(c) == category and c in CLASSES, 'Invalid canonical class ID')
            value = float(raw_probability)
            require(math.isfinite(value) and 0 <= value <= 1,
                    'Probability must be finite and in [0,1]')
            require((key, c) not in parsed, 'Duplicate identity/class row')
            parsed[key, c] = value
            row_count += 1
    require(set(parsed) == {(key, c) for key in expected for c in CLASSES},
            'CSV omits images/Studies/classes or has unexpected rows')
    require(row_count == len(expected) * 10, 'CSV total row count mismatch')
    return parsed, row_count


def audit(args, result):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Hide CUDA explicitly for this CPU-only audit')
    manifest = args.manifest.resolve(strict=True)
    model = args.model_dir.resolve(strict=True)
    submission = args.submission.resolve(strict=True)
    image_csv = submission.with_suffix('.image_predictions.csv')
    report_path = submission.with_suffix('.csv.report.json')
    acceptance_path = ROOT / 'docs/DUAL_TINY768_ACCEPTANCE_20261028.json'
    acceptance = read_json(acceptance_path)
    config_path = model / 'config.json'
    config = read_json(config_path)
    report = read_json(report_path)
    before = snapshot_entries(read_json(args.before_snapshot))
    required_protection = {manifest, config_path, acceptance_path}
    required_protection.update((ROOT / name).resolve() for name in
                               ['predict.py', 'deploy_kd.py', 'model_overnight.py', 'runtime_compat.py'])
    members = config['members']
    required_protection.update((model / member['weights_file']).resolve() for member in members)
    require(required_protection <= set(before), 'Before snapshot omits required protected input')
    baseline_now = {path: sha(path) for path in before}
    require(baseline_now == before, 'Protected source changed since before inference')
    output_before = {path: sha(path) for path in [submission, image_csv, report_path]}
    result['before_snapshot_sha256'] = sha(args.before_snapshot)
    result['checks']['source_inputs_unchanged_since_before_inference'] = True

    require(config_path.resolve().parent == model and not config_path.is_symlink(),
            'Deployment config must be a regular package file')
    require(baseline_now[config_path] == acceptance['deployment_config_sha256'],
            'Deployment config differs from frozen acceptance')
    require((ROOT / acceptance['deployment']).resolve() == model,
            'Deployment path differs from frozen acceptance')
    require(acceptance['member_seeds_in_order'] == [43, 44]
            and acceptance['fixed_probability_weights'] == [.25, .75]
            and acceptance['precision'] == 'float16'
            and acceptance['tta'] == 'hflip2' and acceptance['image_size'] == 768,
            'Acceptance does not describe the explicitly frozen deployment')
    require(len(members) == 2 and [member['seed'] for member in members] == [43, 44],
            'Deployment member seeds/order mismatch')
    require(config['model_weights'] == [[.25] * 10, [.75] * 10],
            'Frozen all-class probability weights mismatch')
    require(config['precision'] == 'float16' and config['tta'] == 'hflip2'
            and config['image_size'] == 768 and config['image_sizes'] == [768]
            and config['class_ids'] == list(range(1, 11)) and config['batch_size'] == 1
            and config['ensemble'] is True and config.get('gates') is None
            and config['aggregation'] == 'study_mean_probability'
            and config['member_aggregation'] == 'fixed_weighted_mean_probability'
            and config['tta_aggregation'] == 'mean_sigmoid_probability'
            and all(member['tta'] == 'hflip2' and member['image_size'] == 768 for member in members),
            'Frozen precision/TTA/size/probability aggregation mismatch')
    inventory = []
    names = [member['weights_file'] for member in members] + ['config.json']
    require(config['inference_files'] == names and len(set(names)) == 3,
            'Inference package inventory mismatch')
    for name in names:
        path = model / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink(),
                'Inference file must be a regular file inside model package')
        inventory.append({'path': name, 'bytes': path.stat().st_size,
                          'sha256': baseline_now[path.resolve()]})
    for member in members:
        require(baseline_now[(model / member['weights_file']).resolve()] == member['weights_sha256'],
                'Member weight SHA256 mismatch')
    total_bytes = sum(item['bytes'] for item in inventory)
    require(total_bytes == acceptance['inference_files_bytes'] and total_bytes <= 500_000_000,
            'Actual inference inventory size mismatch')
    result['deployment'] = {'config_sha256': baseline_now[config_path], 'inventory': inventory,
                            'inference_files_bytes': total_bytes, 'seeds': [43, 44],
                            'model_weights': [[.25] * 10, [.75] * 10],
                            'precision': 'float16', 'tta': 'hflip2', 'image_size': 768}
    result['checks']['frozen_deployment_matches'] = True

    # Project the manifest onto identity and pre-existing file digest columns only.
    # No test target/label column is accessed or interpreted.
    identities = {}
    relatives = {}
    by_study = defaultdict(list)
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            image_id = row['image_id']
            subject, study = row['subject_id'], row['study_id']
            relative = Path(row['relative_path'])
            require(row['split'] == 'test' and row['study_key'] == f'{subject}/{study}',
                    'Manifest split/Study identity mismatch')
            require(not relative.is_absolute() and '..' not in relative.parts
                    and len(relative.parts) == 4
                    and relative.parts[:3] == ('test', subject, study),
                    'Manifest relative path escapes its original test Study')
            require(image_id not in identities and relative.as_posix() not in relatives,
                    'Duplicate manifest image identity or original path')
            identities[image_id] = (subject, study)
            relatives[relative.as_posix()] = row['file_sha256']
            by_study[subject, study].append(image_id)
    require(bool(identities), 'Empty original test manifest')
    image_root = args.image_root.resolve(strict=True)
    if not (image_root / 'test').is_dir():
        image_root = (image_root / 'cxr_first').resolve(strict=True)
    original_paths = {path.relative_to(image_root).as_posix(): path
                      for path in (image_root / 'test').rglob('*')
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    require(set(original_paths) == set(relatives),
            'Frozen test manifest does not cover every original test image exactly once')
    image_digest_checks = []
    if args.verify_image_sha:
        for relative, path in sorted(original_paths.items()):
            digest = sha(path)
            require(digest == relatives[relative], 'Original image differs from frozen file SHA256')
            image_digest_checks.append({'relative_path': relative, 'sha256': digest})
    result['original_images'] = {'file_sha256_verified': args.verify_image_sha,
                                 'files': image_digest_checks}
    result['checks']['original_images_studies_patients_exact_coverage'] = True

    study_values, study_rows = probabilities(submission, by_study)
    image_values, image_rows = probabilities(image_csv, identities, image=True)
    max_error = 0.0
    for key, image_ids in by_study.items():
        for c in CLASSES:
            independent_mean = math.fsum(image_values[image_id, c] for image_id in image_ids) / len(image_ids)
            error = abs(independent_mean - study_values[key, c])
            max_error = max(max_error, error)
            require(error <= 1e-14, 'Independent per-image Study mean disagrees with submission')
    result['counts'] = {'images': len(identities), 'studies': len(by_study),
                         'patients': len({key[0] for key in by_study}),
                         'submission_rows': study_rows, 'image_prediction_rows': image_rows,
                         'classes_per_study': 10, 'classes_per_image': 10}
    result['aggregation_verification'] = {'method': 'independent math.fsum/image_count',
                                           'absolute_tolerance': 1e-14,
                                           'maximum_arithmetic_error': max_error}
    for name in ['utf8_exact_headers', 'all_studies_all_ten_classes_exactly_once',
                 'all_images_all_ten_classes_exactly_once', 'probabilities_finite_in_range',
                 'independent_image_to_study_aggregation']:
        result['checks'][name] = True

    require(report['submission_sha256'] == output_before[submission]
            and report['image_predictions_sha256'] == output_before[image_csv]
            and report['deployment_config_sha256'] == baseline_now[config_path]
            and report['manifest_sha256'] == baseline_now[manifest], 'Inference report SHA256 bindings mismatch')
    require(Path(report['submission']).resolve() == submission
            and Path(report['image_predictions']).resolve() == image_csv
            and report['studies'] == len(by_study) and report['images'] == len(identities)
            and report['data_rows'] == study_rows and report['batch_size'] == 1
            and report['precision'] == 'float16' and report['tta'] == 'hflip2'
            and report['ensemble'] is True and report['member_count'] == 2
            and report['image_sizes'] == [768], 'Inference report count/deployment declarations mismatch')
    require(report['deployment_inventory']['files'] == inventory
            and report['deployment_inventory']['total_bytes'] == total_bytes
            and report['deployment_inventory']['size_pass'] is True,
            'Inference report deployment inventory mismatch')
    require(acceptance['T4_latency_certified'] is False
            and report['latency_status'] == 'pending_cost_py_T4_measurement',
            'T4 must remain explicitly uncertified')
    result['checks']['inference_report_bindings_and_counts'] = True
    result['T4_latency_certified'] = False
    result['submission_completeness_pass'] = True
    result['official_compliance_fully_certified'] = False

    after = {path: sha(path) for path in before}
    require(after == before, 'Protected source changed during audit')
    output_after = {path: sha(path) for path in output_before}
    require(output_after == output_before, 'Prediction output changed during audit')
    result['protected_inputs'] = {str(path): {'sha256_before': before[path],
                                  'sha256_after': after[path], 'unchanged': True}
                                  for path in sorted(before)}
    result['outputs'] = {str(path): {'sha256': output_after[path], 'bytes': path.stat().st_size}
                         for path in sorted(output_after)}
    result['checks']['protected_inputs_and_outputs_unchanged_during_audit'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-snapshot', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/clean_v1_reviewed/test_manifest.csv')
    parser.add_argument('--model-dir', type=Path, default=DELIVERY / 'fp16_hflip2')
    parser.add_argument('--submission', type=Path, default=DELIVERY / 'test_predictions/submission.csv')
    parser.add_argument('--image-root', type=Path, default=Path('/home/featurize/data/cxr_first'))
    parser.add_argument('--verify-image-sha', action='store_true')
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'audit.json')
    args = parser.parse_args()
    result = {'audit_schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'CPU-only final test submission completeness; frozen deployment only',
              'audit_script_sha256': sha(__file__), 'checks': {}, 'passed': False,
              'no_gpu_work_by_this_audit': True, 'test_answers_read': False,
              'test_disease_distributions_computed': False, 'model_or_parameter_selection': False,
              'test_used_to_modify_code_or_preprocessing': False, 'platform_upload': False}
    try:
        audit(args, result)
        result['passed'] = True
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'audit': str(args.output),
                      'counts': result.get('counts'), 'error': result.get('error')}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
