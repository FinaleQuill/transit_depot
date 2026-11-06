"""CPU-only fourth bounded valid screen using frozen per-image probabilities.

No weights are loaded, and no fitting or test inputs are accepted. The output
does not replace the frozen B mean-pooling submission requested by the user.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / 'artifacts/kd_v1/student_auc815_20261031'
PLAN = ARTIFACT / 'fourth_plan.json'
MANIFEST = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
REPORTS = ROOT / 'artifacts/kd_v1/student_upper_bound_20261030/reports'
OUTPUT = ARTIFACT / 'fourth_pooling_report.json'
EXPECTED = ['B_study_rms', 'B_study_max', 'AB_study_rms', 'AB_study_max']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_probabilities(path, *, image=False):
    values = {}
    for row in csv.DictReader(path.open(encoding='utf-8-sig')):
        key = ((row['Subject_id'], row['Study_id'], row['Image_id']) if image
               else (row['Subject_id'], row['Study_id']))
        value = values.setdefault(key, np.full(10, np.nan))
        column = int(row['Predict_class']) - 1
        if not 0 <= column < 10 or np.isfinite(value[column]):
            raise ValueError('Duplicate or invalid probability class')
        value[column] = float(row['Probability'])
    if not values or any(not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                         for p in values.values()):
        raise ValueError('Incomplete or invalid probabilities')
    return values


def prepare_auc(y, probabilities):
    order = np.argsort(probabilities, kind='stable')
    sorted_p = probabilities[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    return order, starts, y[order]


def weighted_auc(prepared, weights):
    """Mann-Whitney AUC with half credit for tied probabilities."""
    order, starts, y = prepared
    pos = np.add.reduceat(weights[order] * y, starts)
    neg = np.add.reduceat(weights[order] * (1 - y), starts)
    denominator = pos.sum() * neg.sum()
    return (float((pos * (np.cumsum(neg) - neg / 2)).sum() / denominator)
            if denominator else None)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior outputs; do not overwrite')
    plan = json.loads(PLAN.read_text())
    if (plan['candidates'] != EXPECTED or plan['bootstrap']['samples'] != 1000
            or plan['bootstrap']['seed'] != 20261031):
        raise ValueError('Predeclared bounded candidate set changed')
    labels, images = {}, {}
    for row in csv.DictReader(MANIFEST.open(encoding='utf-8-sig')):
        if row['split'] != 'valid':
            raise ValueError('Only official valid is accepted')
        key = (row['subject_id'], row['study_id'])
        target = json.loads(row['targets'])
        if key in labels and labels[key] != target:
            raise ValueError('Inconsistent Study targets')
        labels[key] = target
        image_key = (*key, row['image_id'])
        if image_key in images:
            raise ValueError('Duplicate manifest image')
        images[image_key] = key
    keys = sorted(labels)
    if len(images) != 1422 or len(keys) != 859:
        raise ValueError('Original valid coverage required')
    y = np.array([labels[k] for k in keys], dtype=np.int8)
    subjects = np.array([k[0] for k in keys])
    patients, patient_index = np.unique(subjects, return_inverse=True)
    image_keys = sorted(images)
    groups = [np.array([i for i, k in enumerate(image_keys) if k[:2] == study])
              for study in keys]
    counts = np.array([len(g) for g in groups])
    source, provenance, original_study = {}, [], {}
    for name in ['A', 'B']:
        directory = REPORTS / name / 'valid_hflip2'
        report_file = directory / 'validation_report.json'
        report = json.loads(report_file.read_text())
        image_file = directory / 'valid.image_predictions.csv'
        study_file = directory / 'valid_predictions.csv'
        if (not report['complete'] or report['manifest_sha256'] != sha(MANIFEST)
                or report['image_predictions_sha256'] != sha(image_file)
                or report['predictions_sha256'] != sha(study_file)
                or report['image_size'] != 768 or report['tta'] != 'hflip2'
                or report['precision'] != 'float16' or report['member_count'] != 1):
            raise ValueError('Require unchanged SHA-bound original FP16 predictions')
        values = read_probabilities(image_file, image=True)
        original_study[name] = read_probabilities(study_file)
        if set(values) != set(image_keys) or set(original_study[name]) != set(keys):
            raise ValueError('Exact image and Study coverage required')
        source[name] = np.array([values[k] for k in image_keys])
        provenance.append({'name': name, 'report_sha256': sha(report_file),
                           'image_predictions_sha256': sha(image_file),
                           'predictions_sha256': sha(study_file)})
    pools = {'B': source['B'], 'AB': (source['A'] + source['B']) / 2}
    candidates = {}
    for name, values in pools.items():
        for method in ['mean', 'rms', 'max']:
            def aggregate(g):
                if method == 'mean':
                    return values[g].mean(axis=0)
                if method == 'rms':
                    return np.sqrt(np.mean(values[g] ** 2, axis=0))
                return np.max(values[g], axis=0)
            candidates[f'{name}_study_{method}'] = np.array([aggregate(g) for g in groups])
    # Reconstruct original means in original manifest image order, allowing only
    # floating-point summation-order roundoff after sorting the image identifiers.
    original_b = np.array([original_study['B'][k] for k in keys])
    original_ab = np.array([(original_study['A'][k] + original_study['B'][k]) / 2
                            for k in keys])
    for n, original in [('B_study_mean', original_b), ('AB_study_mean', original_ab)]:
        if not np.allclose(candidates[n], original, atol=1e-15, rtol=0):
            raise ValueError('Original mean baseline did not replay')
    prepared = {n: [prepare_auc(y[:, c], p[:, c]) for c in range(10)]
                for n, p in candidates.items()}

    def aucs(name, weights):
        return [weighted_auc(p, weights) for p in prepared[name]]

    def metrics(name, weights):
        values = aucs(name, weights)
        return {'per_class_auc': values, 'macro_auc': float(np.mean(values)),
                'class_1_4_8_mean_auc': float(np.mean(np.array(values)[[0, 3, 7]]))}

    scores = {}
    for name in candidates:
        scores[name] = metrics(name, np.ones(len(keys)))
        scores[name]['slices'] = {}
        for label, mask in [('single_image_study', counts == 1),
                            ('multiple_image_study', counts > 1)]:
            scores[name]['slices'][label] = {'num_studies': int(mask.sum()),
                                            **metrics(name, mask.astype(float))}
    if (abs(scores['B_study_mean']['macro_auc'] - .8105695698808949) > 1e-14
            or abs(scores['AB_study_mean']['macro_auc'] - .8147866171472288) > 1e-14):
        raise ValueError('Mean AUC baselines did not independently replay')
    rng = np.random.default_rng(20261031)
    deltas = {name: [] for name in EXPECTED}
    for _ in range(1000):
        multiplicities = np.bincount(rng.integers(0, len(patients), len(patients)),
                                     minlength=len(patients))
        weights = multiplicities[patient_index]
        bootstrap_auc = {n: aucs(n, weights) for n in candidates}
        for name in EXPECTED:
            reference = name.split('_')[0] + '_study_mean'
            values, baseline = bootstrap_auc[name], bootstrap_auc[reference]
            delta = [None if a is None or b is None else a - b
                     for a, b in zip(values, baseline)]
            macro = None if any(v is None for v in delta) else float(np.mean(delta))
            selected = [delta[c] for c in [0, 3, 7]]
            focused = None if any(v is None for v in selected) else float(np.mean(selected))
            deltas[name].append(delta + [macro, focused])
    intervals = {}
    for name, rows in deltas.items():
        columns = []
        for c in range(12):
            valid = [r[c] for r in rows if r[c] is not None]
            columns.append({'valid_resamples': len(valid), 'ci95':
                            np.quantile(valid, [.025, .975]).tolist() if valid else None})
        reference = name.split('_')[0] + '_study_mean'
        intervals[name] = {'reference': reference, 'per_class_delta': columns[:10],
                           'macro_auc_delta': columns[10],
                           'class_1_4_8_mean_delta': columns[11]}
        scores[name]['macro_auc_delta_from_own_mean'] = (
            scores[name]['macro_auc'] - scores[reference]['macro_auc'])
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(PLAN), 'script_sha256': sha(Path(__file__)),
              'manifest_sha256': sha(MANIFEST), 'source_validations': provenance,
              'num_images': len(images), 'num_studies': len(keys),
              'num_patients': len(patients), 'scores': scores,
              'patient_paired_bootstrap': {'samples': 1000, 'seed': 20261031,
                  'sampling_unit': 'patient_with_all_Studies', 'results': intervals},
              'thresholds_fitted': False, 'parameters_fitted': False,
              'test_used': False, 'new_model_inference': False,
              'formal_T4_certification': False, 'deployment_selected': False,
              'interpretation': plan['interpretation'],
              'auc_method': 'Independent NumPy weighted Mann-Whitney, half-credit ties',
              'stop_rule': plan['stop_rule']}
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({n: s['macro_auc'] for n, s in scores.items()}, indent=2))


if __name__ == '__main__':
    main()
