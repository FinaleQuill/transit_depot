"""Independent CPU verification of the one registered final candidate only.

Reads original valid labels and completed saved predictions. Does not import
training/inference/scoring code, form alternative candidates, or access test.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np


RUN = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
STUDY_HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
IMAGE_HEADER = ['Image_id', *STUDY_HEADER]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right, description, tolerance=1e-12):
    difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f'{description}: max absolute difference {difference}')
    return difference


def load_probabilities(path, expected, images=False):
    values = {key: np.full(10, np.nan) for key in expected}
    header = IMAGE_HEADER if images else STUDY_HEADER
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != header:
            raise ValueError(f'Unexpected CSV schema: {path}')
        row_count = 0
        for row in reader:
            key = row['Image_id'] if images else (row['Subject_id'], row['Study_id'])
            if key not in expected:
                raise ValueError(f'Unexpected identity: {path}: {key}')
            if images and expected[key] != (row['Subject_id'], row['Study_id']):
                raise ValueError(f'Image patient/Study mismatch: {path}: {key}')
            c, probability = int(row['Predict_class']) - 1, float(row['Probability'])
            if not 0 <= c < 10 or not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'Invalid class/probability: {path}')
            if np.isfinite(values[key][c]):
                raise ValueError(f'Duplicate class/identity: {path}')
            values[key][c] = probability
            row_count += 1
    if row_count != 10 * len(expected) or any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError(f'Incomplete coverage: {path}')
    return values


def pair_auc(y, p, weights):
    result = []
    for c in range(10):
        positive, negative = y[:, c] == 1, y[:, c] == 0
        if not positive.any() or not negative.any():
            result.append(None)
            continue
        difference = p[positive, c, None] - p[negative, c]
        pair_weights = weights[positive, None] * weights[negative]
        numerator = (((difference > 0) + .5 * (difference == 0)) * pair_weights).sum()
        result.append(float(numerator / pair_weights.sum()))
    return {'macro_auc': float(np.mean(result)) if None not in result else None,
            'per_class_auc': result, 'num_studies': len(y)}


def main():
    started = time.monotonic()
    plan_path = RUN / 'registered_plan.json'
    report_path = RUN / 'final_comparison/report.json'
    selection_path = RUN / 'final_comparison/selection.json'
    if not report_path.is_file() or not selection_path.is_file():
        raise RuntimeError('Final report and selection must both exist; no partial output is audited')
    plan, report, selection = [json.loads(p.read_text()) for p in (plan_path, report_path, selection_path)]
    if report.get('complete') is not True:
        raise ValueError('Require completed final comparison')
    frozen_weights = [[1.] + [.8] * 9, [0.] + [.2] * 9]
    if plan['fusion_weights'] != frozen_weights or report['fusion_weights'] != frozen_weights:
        raise ValueError('The fixed candidate changed')
    paths = {'plan': plan_path, 'report': report_path, 'selection': selection_path,
             'source': Path(__file__).resolve(), 'valid_manifest': Path(plan['valid_manifest'])}
    for name in ('cached_sprint_study', 'cached_sprint_image', 'cached_opacity_study', 'cached_opacity_image'):
        paths[name] = Path(plan[name])
    for name in ('candidate', 'sprint', 'opacity'):
        paths[name + '_study'] = RUN / 'final_comparison' / (name + '_predictions.csv')
        paths[name + '_image'] = RUN / 'final_comparison' / (name + '.image_predictions.csv')
    before = {name: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
              for name, path in paths.items()}
    for name in ('valid_manifest', 'cached_sprint_study', 'cached_sprint_image',
                 'cached_opacity_study', 'cached_opacity_image'):
        if before[name]['sha256'] != plan[name + '_sha256']:
            raise ValueError('Frozen input changed: ' + name)
    if (before['candidate_study']['sha256'] != report['candidate_predictions_sha256']
            or before['candidate_image']['sha256'] != report['candidate_image_predictions_sha256']):
        raise ValueError('Candidate CSV is not bound to final report')
    with paths['valid_manifest'].open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels, image_identity, study_images, views = {}, {}, defaultdict(list), defaultdict(list)
    for row in rows:
        if row['split'] != 'valid':
            raise ValueError('Manifest includes non-valid data')
        key = row['subject_id'], row['study_id']
        target = tuple(json.loads(row['targets']))
        if len(target) != 10 or any(v not in (0, 1) for v in target):
            raise ValueError('Invalid original target')
        if labels.setdefault(key, target) != target or row['image_id'] in image_identity:
            raise ValueError('Duplicate image or conflicting original Study targets')
        image_identity[row['image_id']] = key
        study_images[key].append(row['image_id'])
        views[key].append(row['view_position'])
    keys = sorted(labels)
    patients = np.array([k[0] for k in keys])
    single = np.array([len(study_images[k]) == 1 for k in keys])
    if (len(rows), len(keys), len(set(patients)), int(single.sum())) != (1422, 859, 228, 391):
        raise ValueError('Original validation coverage differs')
    study, images = {}, {}
    for name in ('candidate', 'sprint', 'opacity'):
        study[name] = load_probabilities(paths[name + '_study'], labels)
        images[name] = load_probabilities(paths[name + '_image'], image_identity, images=True)
    for name in ('sprint', 'opacity'):
        study['cached_' + name] = load_probabilities(paths['cached_' + name + '_study'], labels)
        images['cached_' + name] = load_probabilities(paths['cached_' + name + '_image'], image_identity, images=True)
    aggregation_delta, cache_delta = {}, {}
    for name in study:
        aggregation_delta[name] = max(close(
            np.array([images[name][image] for image in study_images[key]]).mean(0), study[name][key],
            'Mean image-to-Study aggregation: ' + name) for key in keys)
    for name in ('sprint', 'opacity'):
        cache_delta[name] = {
            'study': max(close(study[name][k], study['cached_' + name][k], 'Parent Study cache: ' + name,
                               tolerance=1e-7) for k in keys),
            'image': max(close(images[name][k], images['cached_' + name][k], 'Parent image cache: ' + name,
                               tolerance=1e-7) for k in image_identity)}
    w0, w1 = np.asarray(frozen_weights)
    image_fusion_delta = max(close(images['candidate'][k], w0 * images['sprint'][k] + w1 * images['opacity'][k],
                                   'Registered image fusion') for k in image_identity)
    study_fusion_delta = max(close(study['candidate'][k], w0 * study['sprint'][k] + w1 * study['opacity'][k],
                                   'Registered Study fusion') for k in keys)
    class1_delta = float(max(abs(study['candidate'][k][0] - study['cached_sprint'][k][0]) for k in keys))
    if class1_delta != 0:
        raise ValueError('Class 1 was not preserved exactly')
    y = np.array([labels[k] for k in keys])
    masks = {'all': np.ones(len(keys), bool), 'single_image_study': single,
             'multiple_image_study': ~single,
             'ap_only': np.array([all(v.upper() == 'AP' for v in views[k]) for k in keys])}
    scores = {}
    for name in ('candidate', 'cached_sprint'):
        probabilities = np.array([study[name][k] for k in keys])
        scores[name] = {}
        for slice_name, mask in masks.items():
            frequencies = Counter(patients[mask])
            weight = np.array([1. / frequencies[p] for p in patients[mask]])
            scores[name][slice_name] = {
                'study_equal': pair_auc(y[mask], probabilities[mask], np.ones(int(mask.sum()))),
                'patient_equal': pair_auc(y[mask], probabilities[mask], weight)}
    metric_differences = []
    sensitivity_names = {'all': 'patient_equal', 'single_image_study': 'patient_equal_single_image',
                         'multiple_image_study': 'patient_equal_multiple_image', 'ap_only': 'patient_equal_ap_only'}
    for name, report_name in [('candidate', 'candidate'), ('cached_sprint', 'baseline')]:
        for slice_name in ('all', 'single_image_study', 'multiple_image_study'):
            saved = report[report_name] if slice_name == 'all' else report[report_name]['slices'][slice_name]
            actual = scores[name][slice_name]['study_equal']
            for metric in ('macro_auc', 'per_class_auc'):
                metric_differences.append(close(actual[metric], saved[metric], f'{name}/{slice_name}/{metric}'))
            if actual['num_studies'] != saved['num_studies']:
                raise ValueError('Reported slice coverage differs')
        for slice_name, saved_name in sensitivity_names.items():
            actual = scores[name][slice_name]['patient_equal']
            saved = report[report_name + '_sensitivity'][saved_name]
            for metric in ('macro_auc', 'per_class_auc'):
                metric_differences.append(close(actual[metric], saved[metric], f'{name}/{saved_name}/{metric}'))
    new, old = scores['candidate'], scores['cached_sprint']
    checks = {
        'macro_strictly_better': new['all']['study_equal']['macro_auc'] > old['all']['study_equal']['macro_auc'],
        'class8_strictly_better': new['all']['study_equal']['per_class_auc'][7] > old['all']['study_equal']['per_class_auc'][7],
        'single_image_macro_not_worse': new['single_image_study']['study_equal']['macro_auc'] >= old['single_image_study']['study_equal']['macro_auc'] - 1e-12,
        'patient_equal_macro_not_worse': new['all']['patient_equal']['macro_auc'] >= old['all']['patient_equal']['macro_auc'] - 1e-12,
        'class1_exactly_preserved': class1_delta == 0.}
    if (checks != report['selection_checks'] or checks != selection['selection_checks']
            or all(checks.values()) != report['candidate_won']
            or all(checks.values()) != selection['candidate_won']):
        raise ValueError('Frozen adoption gates were not applied correctly')
    for name, path in paths.items():
        if digest(path) != before[name]['sha256']:
            raise ValueError('Input changed during verification: ' + name)
    result = {'complete': True, 'passed': True,
              'created_at': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
              'scope': 'CPU-only independent exact positive-negative pair enumeration on the sole completed final candidate; no alternative fusion, GPU, training, or test access.',
              'input_sources': before, 'inputs_unchanged': True,
              'coverage': {'images': len(rows), 'studies': len(keys), 'patients': len(set(patients)),
                           'single_image_studies': int(single.sum()), 'multiple_image_studies': int((~single).sum())},
              'scores': scores, 'max_auc_difference_from_report': max(metric_differences),
              'image_to_study_max_difference': aggregation_delta, 'parent_cache_max_difference': cache_delta,
              'registered_image_fusion_max_difference': image_fusion_delta,
              'registered_study_fusion_max_difference': study_fusion_delta,
              'class1_max_difference': class1_delta, 'selection_checks': checks,
              'candidate_won': all(checks.values()), 'elapsed_seconds': time.monotonic() - started,
              'limitations': ['Same reused official valid; this verifies implementation, not independent generalization.',
                             'Bootstrap intervals are not recomputed here; scoring and gates are independently verified.']}
    destination = OUTPUT / 'independent_final_verification.json'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    text = ['# Independent final validation audit', '',
            f"Recorded {result['created_at']} (Hong Kong).", '',
            'PASS: completed saved predictions cover all 1,422 images, 859 Studies and 228 patients. Original labels were not changed.',
            'All ten AUCs, single/multiple-image slices and per-slice patient-equal metrics reproduce the final report by independent positive-negative pair enumeration.',
            f"Maximum AUC discrepancy: {result['max_auc_difference_from_report']:.3g}. Image/Study fixed fusion and all five frozen selection gates agree. Class 1 is exactly preserved.",
            f"Candidate selected by frozen gates: {result['candidate_won']}. Input hashes stayed unchanged.", '',
            'This is a CPU-only implementation audit of the sole candidate. No alternative fusion, GPU inference, training, test access or independent-generalization claim.', '',
            f'Full evidence: `{destination}`.']
    with (OUTPUT / 'INDEPENDENT_FINAL_VERIFICATION.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps({'passed': True, 'candidate_won': result['candidate_won'],
                      'selection_checks': checks, 'maximum_auc_difference': max(metric_differences),
                      'elapsed_seconds': result['elapsed_seconds'], 'report': str(destination)}))


if __name__ == '__main__':
    main()
