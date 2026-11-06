"""Independent CPU-only audit of frozen 033 full-valid predictions.

Uses CSV, manifest labels and NumPy directly; no production inference imports,
model loading, GPU calls, sklearn dependence, test input or bootstrap fitting.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path('/home/featurize/work/cloud_project_v2')
WORK = PROJECT.parent
OUTPUT = Path(__file__).resolve().parent
MANIFEST = PROJECT / 'data/clean_v1_reviewed/valid_manifest.csv'
OLD = WORK / 'deliveries/033_dual_tiny768_seed43_25_seed44_75/valid_fp16_hflip2'
NEW_ROOT = WORK / 'deliveries/035_dual768_2080ti_optimization_20261029'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest():
    labels = {}
    counts = Counter()
    image_ids = set()
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            assert row['split'] == 'valid'
            assert row['image_id'] not in image_ids
            image_ids.add(row['image_id'])
            key = (row['subject_id'], row['study_id'])
            target = tuple(json.loads(row['targets']))
            assert len(target) == 10 and set(target) <= {0, 1}
            assert key not in labels or labels[key] == target
            labels[key] = target
            counts[key] += 1
    keys = sorted(labels)
    assert len(image_ids) == 1422 and len(keys) == 859
    return keys, np.asarray([labels[key] for key in keys], dtype=np.int8), counts


def aucs(targets, values):
    """Mann-Whitney AUC from independent positive/negative pair order."""
    result = []
    for column in range(10):
        positive = values[targets[:, column] == 1, column]
        negative = np.sort(values[targets[:, column] == 0, column])
        if not len(positive) or not len(negative):
            result.append(None)
            continue
        left = np.searchsorted(negative, positive, side='left')
        right = np.searchsorted(negative, positive, side='right')
        result.append(float(np.sum(left + right) / (2 * len(positive) * len(negative))))
    return result


def summary(targets, values, mask=None):
    if mask is not None:
        targets, values = targets[mask], values[mask]
    per_class = aucs(targets, values)
    binary = values >= .5
    tp = np.sum(binary & (targets == 1), axis=0)
    fp = np.sum(binary & (targets == 0), axis=0)
    fn = np.sum(~binary & (targets == 1), axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0)
    return {
        'studies': len(targets), 'macro_auc': float(np.mean(per_class)),
        'per_class_auc': per_class, 'macro_f1_fixed_0_5': float(np.mean(f1)),
        'per_class_positive_studies': np.sum(targets, axis=0).tolist(),
    }


def read_predictions(root, keys, targets, counts):
    path = root / 'valid_predictions.csv'
    report = json.loads((root / 'validation_report.json').read_text())
    assert report['complete'] and report['manifest_sha256'] == sha256(MANIFEST)
    assert report['predictions_sha256'] == sha256(path)
    table = {}
    rows = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['Subject_id'], row['Study_id'])
            column = int(row['Predict_class'])
            assert 1 <= column <= 10
            assert column not in table.setdefault(key, {})
            table[key][column] = float(row['Probability'])
            rows += 1
    assert set(table) == set(keys) and rows == 8590
    assert all(set(value) == set(range(1, 11)) for value in table.values())
    values = np.asarray([[table[key][column] for column in range(1, 11)] for key in keys])
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    metrics = summary(targets, values)
    assert abs(metrics['macro_auc'] - report['macro_auc']) < 1e-12
    assert max(abs(np.asarray(metrics['per_class_auc']) - report['per_class_auc'])) < 1e-12
    assert abs(metrics['macro_f1_fixed_0_5'] - report['macro_f1_fixed_0_5']) < 1e-12
    metadata = {
        'directory': str(root), 'prediction_sha256': sha256(path),
        'report_sha256': sha256(root / 'validation_report.json'),
        'coverage': {'images': 1422, 'studies': len(keys), 'patients': len({key[0] for key in keys}),
                     'probability_rows': rows},
        'metrics_recomputed': metrics,
        'single_image_study_metrics': summary(targets, values, np.asarray([counts[k] == 1 for k in keys])),
        'multi_image_study_metrics': summary(targets, values, np.asarray([counts[k] > 1 for k in keys])),
        'source_report_metrics_match': True,
    }
    return values, metadata


def order_counts(reference, candidate, different_label=None):
    changed = reference != candidate
    result = {
        'pairs_compared': int(changed.size),
        'strict_inversions': int(np.count_nonzero(reference * candidate == -1)),
        'ties_broken': int(np.count_nonzero((reference == 0) & (candidate != 0))),
        'ties_created': int(np.count_nonzero((reference != 0) & (candidate == 0))),
        'any_order_or_tie_changes': int(np.count_nonzero(changed)),
    }
    if different_label is not None:
        result['different_label_order_or_tie_changes'] = int(np.count_nonzero(changed & different_label))
    return result


def compare(reference, candidate, targets):
    difference = np.abs(candidate - reference)
    within_i, within_j = np.triu_indices(10, 1)
    sample_reference = np.sign(reference[:, within_i] - reference[:, within_j])
    sample_candidate = np.sign(candidate[:, within_i] - candidate[:, within_j])
    study_i, study_j = np.triu_indices(len(reference), 1)
    per_class_order = []
    for column in range(10):
        ordered_reference = np.sign(reference[study_i, column] - reference[study_j, column])
        ordered_candidate = np.sign(candidate[study_i, column] - candidate[study_j, column])
        per_class_order.append({'class': column + 1, **order_counts(
            ordered_reference, ordered_candidate,
            targets[study_i, column] != targets[study_j, column])})
    old_aucs, new_aucs = np.asarray(aucs(targets, reference)), np.asarray(aucs(targets, candidate))
    return {
        'probabilities_exactly_equal': bool(np.array_equal(reference, candidate)),
        'max_absolute_probability_difference': float(np.max(difference)),
        'mean_absolute_probability_difference': float(np.mean(difference)),
        'per_class_max_absolute_probability_difference': np.max(difference, axis=0).tolist(),
        'per_class_mean_absolute_probability_difference': np.mean(difference, axis=0).tolist(),
        'probabilities_exceeding_1e_minus_3': int(np.count_nonzero(difference > 1e-3)),
        'changed_probability_values': int(np.count_nonzero(difference)),
        'fixed_p_ge_0_5_binary_changes': int(np.count_nonzero((reference >= .5) != (candidate >= .5))),
        'per_class_fixed_p_ge_0_5_binary_changes': np.count_nonzero(
            (reference >= .5) != (candidate >= .5), axis=0).tolist(),
        'within_study_class_order': order_counts(sample_reference, sample_candidate),
        'per_class_across_study_order': per_class_order,
        'per_class_auc_delta': (new_aucs - old_aucs).tolist(),
        'macro_auc_delta': float(np.mean(new_aucs) - np.mean(old_aucs)),
    }


def main():
    keys, targets, counts = read_manifest()
    old, old_metadata = read_predictions(OLD, keys, targets, counts)
    new, new_metadata = read_predictions(NEW_ROOT / 'valid_original', keys, targets, counts)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cpu_only': True, 'bootstrap_rerun': False,
        'manifest_sha256': sha256(MANIFEST),
        'reference_historical_4090': old_metadata,
        'original_new_2080ti': new_metadata,
        'hardware_environment_comparison': compare(old, new, targets),
        'interpretation': 'Original frozen model weights on different GPU/environment. AUC microdifference is not an optimization or training gain. Valid-only development audit; no official T4 or test consistency claim.',
    }
    optimized = NEW_ROOT / 'valid_optimized'
    if (optimized / 'validation_report.json').is_file():
        optimized_values, optimized_metadata = read_predictions(optimized, keys, targets, counts)
        report['optimized_new_2080ti'] = optimized_metadata
        report['same_environment_code_comparison'] = compare(new, optimized_values, targets)
        report['same_environment_prediction_files_byte_identical'] = (
            new_metadata['prediction_sha256'] == optimized_metadata['prediction_sha256'])
    (OUTPUT / 'full_valid_independent_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'artifact': str(OUTPUT / 'full_valid_independent_audit.json'),
        'macro_auc_4090': old_metadata['metrics_recomputed']['macro_auc'],
        'macro_auc_2080ti_original': new_metadata['metrics_recomputed']['macro_auc'],
        'comparison': report['hardware_environment_comparison'],
        'same_environment_prediction_files_byte_identical': report.get('same_environment_prediction_files_byte_identical'),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
