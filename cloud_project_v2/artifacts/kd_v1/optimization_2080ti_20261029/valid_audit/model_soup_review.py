"""Independent saved-probability review of the one registered weight soup.

CPU only. No production imports, GPU work, fitting, threshold search or training.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

from audit_full_valid import (
    PROJECT, WORK, OUTPUT, MANIFEST, read_manifest, read_predictions, sha256, summary,
)


ROOT = WORK / 'deliveries/035_dual768_2080ti_optimization_20261029'
SOUP = ROOT / 'model_soup_25_75_diagnostic'
SEED = 20261027
SAMPLES = 1000


def read_soup(keys, targets, counts):
    report = json.loads((SOUP / 'diagnostic_report.json').read_text())
    assert report['complete'] and report['diagnostic_only']
    assert report['training_performed'] is False and report['candidate_count'] == 1
    assert report['manifest_sha256'] == sha256(MANIFEST)
    assert report['predictions_sha256'] == sha256(SOUP / 'valid_predictions.csv')
    assert report['image_predictions_sha256'] == sha256(SOUP / 'valid_image_predictions.csv')
    table = {}
    with (SOUP / 'valid_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['Subject_id'], row['Study_id'])
            c = int(row['Predict_class'])
            assert 1 <= c <= 10 and c not in table.setdefault(key, {})
            table[key][c] = float(row['Probability'])
    assert set(table) == set(keys)
    assert all(set(x) == set(range(1, 11)) for x in table.values())
    values = np.asarray([[table[k][c] for c in range(1, 11)] for k in keys])
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    metrics = summary(targets, values)
    assert abs(metrics['macro_auc'] - report['macro_auc']) < 1e-12
    assert np.max(np.abs(np.asarray(metrics['per_class_auc']) - report['per_class_auc'])) < 1e-12
    assert abs(metrics['macro_f1_fixed_0_5'] - report['macro_f1_fixed_0_5']) < 1e-12

    expected_images = {}
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            expected_images[row['image_id']] = (row['subject_id'], row['study_id'])
    images = {}
    with (SOUP / 'valid_image_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            image, key, c = row['Image_id'], (row['Subject_id'], row['Study_id']), int(row['Predict_class'])
            assert image in expected_images and key == expected_images[image]
            assert 1 <= c <= 10 and c not in images.setdefault(image, {})
            images[image][c] = float(row['Probability'])
    assert set(images) == set(expected_images)
    assert all(set(value) == set(range(1, 11)) for value in images.values())
    sums = {key: np.zeros(10, dtype=np.float64) for key in keys}
    for image, image_values in images.items():
        array = np.asarray([image_values[c] for c in range(1, 11)])
        assert np.isfinite(array).all() and (array >= 0).all() and (array <= 1).all()
        sums[expected_images[image]] += array
    means = np.asarray([sums[key] / counts[key] for key in keys])
    assert np.array_equal(means, values)
    return values, {
        'prediction_sha256': sha256(SOUP / 'valid_predictions.csv'),
        'image_prediction_sha256': sha256(SOUP / 'valid_image_predictions.csv'),
        'diagnostic_report_sha256': sha256(SOUP / 'diagnostic_report.json'),
        'coverage': {'images': len(images), 'studies': len(keys), 'patients': len({k[0] for k in keys}),
                     'study_probability_rows': len(keys) * 10, 'image_probability_rows': len(images) * 10},
        'image_to_study_exactly_equal': True,
        'metrics_recomputed': metrics,
        'single_image_study_metrics': summary(targets, values, np.asarray([counts[k] == 1 for k in keys])),
        'multi_image_study_metrics': summary(targets, values, np.asarray([counts[k] > 1 for k in keys])),
        'report_metrics_match': True,
    }


def prepare_auc(y, score):
    order = np.argsort(score, kind='stable')
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    return order, y[order], starts


def weighted_auc(prepared, weight):
    order, sorted_y, starts = prepared
    ordered_weight = weight[order]
    positive = np.add.reduceat(ordered_weight * sorted_y, starts)
    negative = np.add.reduceat(ordered_weight * (1 - sorted_y), starts)
    denominator = positive.sum() * negative.sum()
    if denominator <= 0:
        return np.nan
    negative_before = np.cumsum(negative) - negative
    return float(np.sum(positive * (negative_before + .5 * negative)) / denominator)


def main():
    started = time.perf_counter()
    keys, targets, counts = read_manifest()
    baseline, baseline_metadata = read_predictions(ROOT / 'valid_original', keys, targets, counts)
    soup, soup_metadata = read_soup(keys, targets, counts)
    candidates = [baseline, soup]
    prepared = [[prepare_auc(targets[:, c], values[:, c]) for c in range(10)] for values in candidates]

    subjects = np.asarray([k[0] for k in keys])
    patients, inverse = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(SEED)
    differences = np.empty((SAMPLES, 10))
    maximum_sklearn_error = 0.0
    maximum_explicit_repeat_error = 0.0
    for sample in range(SAMPLES):
        selected = rng.integers(0, len(patients), len(patients))
        patient_counts = np.bincount(selected, minlength=len(patients))
        weight = patient_counts[inverse].astype(np.float64)
        scores = np.asarray([[weighted_auc(prepared[m][c], weight) for c in range(10)] for m in range(2)])
        differences[sample] = scores[1] - scores[0]
        # Validate the production-size weighted algorithm on a fixed first five
        # resamples, including an explicit repeated-Study ROC for the first two.
        if sample < 5:
            for m in range(2):
                for c in range(10):
                    expected = roc_auc_score(targets[:, c], candidates[m][:, c], sample_weight=weight)
                    maximum_sklearn_error = max(maximum_sklearn_error, abs(scores[m, c] - expected))
                    if sample < 2:
                        indices = np.repeat(np.arange(len(keys)), weight.astype(np.int64))
                        repeated = roc_auc_score(targets[indices, c], candidates[m][indices, c])
                        maximum_explicit_repeat_error = max(maximum_explicit_repeat_error, abs(scores[m, c] - repeated))
    # Synthetic ties and zero-frequency observations exercise the tie policy.
    tie_y = np.asarray([0, 1, 0, 1, 0, 1])
    tie_scores = np.asarray([.1, .1, .5, .5, .7, .9])
    tie_weights = np.asarray([3., 0., 2., 4., 1., 2.])
    tied = weighted_auc(prepare_auc(tie_y, tie_scores), tie_weights)
    tie_error = abs(tied - roc_auc_score(tie_y, tie_scores, sample_weight=tie_weights))
    assert maximum_sklearn_error < 1e-12 and maximum_explicit_repeat_error < 1e-12 and tie_error < 1e-12
    macro_delta = np.mean(differences, axis=1)
    valid_macro = np.isfinite(macro_delta)
    def interval(values):
        valid = values[np.isfinite(values)]
        return {'valid_resamples': len(valid), 'ci_95': np.quantile(valid, [.025, .975]).tolist()}
    distribution = OUTPUT / 'model_soup_review_bootstrap_deltas.npz'
    np.savez_compressed(distribution, per_class_delta=differences, macro_auc_delta=macro_delta)
    point_delta = np.asarray(soup_metadata['metrics_recomputed']['per_class_auc']) - baseline_metadata['metrics_recomputed']['per_class_auc']
    output = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
        'candidate_count': 1, 'training_performed': False, 'test_used': False,
        'manifest_sha256': sha256(MANIFEST),
        'reference_033_same_2080ti_environment': baseline_metadata,
        'candidate_fixed_25_75_parameter_average': soup_metadata,
        'point_macro_auc_delta_soup_minus_033': float(np.mean(point_delta)),
        'point_per_class_auc_delta_soup_minus_033': point_delta.tolist(),
        'all_ten_class_auc_point_estimates_decreased': bool((point_delta < 0).all()),
        'patient_paired_bootstrap': {
            'reference': '033_same_environment', 'candidate': 'fixed_25_75_parameter_average',
            'samples': SAMPLES, 'seed': SEED, 'unique_patients': len(patients),
            'sampling_unit': 'patient_with_all_Studies', 'paired': True,
            'sampling_order': 'np.unique subject identifiers sorted; rng.default_rng(seed).integers(0,228,228) each resample',
            'weight_policy': 'Draw228patients_with_replacement; every Study gets its patient draw count; same weights used for both candidates',
            'weighted_auc': 'Sort scores once; sum positive weight times lower-score negative weight plus half equal-score negative weight; divide positive_weight_sum * negative_weight_sum',
            'macro_auc_delta': interval(macro_delta),
            'per_class_delta': [interval(differences[:, c]) for c in range(10)],
            'valid_all_classes_resamples': int(valid_macro.sum()),
            'distribution_path': str(distribution), 'distribution_sha256': sha256(distribution),
            'validation': {'sklearn_weighted_roc_first_5_resamples_max_abs_error': maximum_sklearn_error,
                           'sklearn_explicit_repeat_first_2_resamples_max_abs_error': maximum_explicit_repeat_error,
                           'synthetic_ties_and_zero_weights_abs_error': tie_error},
            'limitations': 'Exploratory percentile paired interval conditional on these fixed trained models and this valid population. It does not correct seed/epoch selection, previous comparisons, valid selection bias, or training randomness. No stable improvement claim and no T4/test certification.',
        },
        'decision': 'Reject this candidate; preserve the negative result and parent artifacts; do not search more weight ratios.',
        'script_sha256': sha256(__file__), 'cpu_wall_seconds': time.perf_counter() - started,
    }
    report = OUTPUT / 'model_soup_review.json'
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(report), 'baseline_auc': baseline_metadata['metrics_recomputed']['macro_auc'],
                      'soup_auc': soup_metadata['metrics_recomputed']['macro_auc'],
                      'point_delta': output['point_macro_auc_delta_soup_minus_033'],
                      'bootstrap': output['patient_paired_bootstrap']['macro_auc_delta'],
                      'soup_single': soup_metadata['single_image_study_metrics']['macro_auc'],
                      'soup_multi': soup_metadata['multi_image_study_metrics']['macro_auc'],
                      'wall_seconds': output['cpu_wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
