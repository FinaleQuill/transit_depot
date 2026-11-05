"""One frozen three-member candidate, authenticated against saved image outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, StudyAccumulator,
    prediction_records, read_submission, sha256_file, write_image_prediction, write_prediction)
from sprint.evaluate import _write_json
from student_experiment_evaluation import score_probabilities, validation_labels


def verify_inputs(config):
    if config['schema'] != 'cxr_preserved_sprint_opacity_v1':
        raise ValueError('Require the preregistered preserved Sprint recipe')
    if config['fusion_weights'] != [[1.] + [.8] * 9, [0.] + [.2] * 9]:
        raise ValueError('Only the frozen 80/20 candidate is permitted')
    for name, value in config.items():
        if name.endswith('_sha256') and name[:-7] in config:
            if sha256_file(config[name[:-7]]) != value:
                raise ValueError('Frozen input changed: ' + name[:-7])
    old = json.loads(Path(config['reference_report']).read_text())
    opacity = json.loads(Path(config['opacity_report']).read_text())
    if (old.get('complete') is not True or opacity.get('complete') is not True
            or old['candidate_predictions_sha256'] != config['cached_sprint_study_sha256']
            or old['candidate_image_predictions_sha256'] != config['cached_sprint_image_sha256']
            or old['candidate_weights_sha256'] != config['sprint_config']['candidate_weights_sha256']
            or opacity['member_prediction_sha256']['opacity512'] != config['cached_opacity_study_sha256']):
        raise ValueError('Cached predictions are not the completed parent experiments')
    found = [x for x in opacity['candidate_inference_assets']
             if x['sha256'] == config['opacity_member']['weights_sha256']]
    if len(found) != 1:
        raise ValueError('Opacity export is not the previously evaluated member')


def image_cache(path, records):
    expected = {r['image_id']: (r['subject_id'], r['study_id']) for r in records}
    result = {key: np.full(10, np.nan, dtype=np.float64) for key in expected}
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != IMAGE_CSV_HEADER:
            raise ValueError('Unexpected image prediction schema')
        for row in reader:
            key = row['Image_id']
            if key not in expected or expected[key] != (row['Subject_id'], row['Study_id']):
                raise ValueError('Image identity mismatch')
            c, p = int(row['Predict_class']) - 1, float(row['Probability'])
            if not 0 <= c < 10 or not 0 <= p <= 1 or np.isfinite(result[key][c]):
                raise ValueError('Duplicate or invalid image probability')
            result[key][c] = p
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Missing original image/class prediction')
    return result


def weighted_auc_many(target, prediction, weights):
    """Exact weighted pair AUC for rows of weights, including tied scores."""
    order = np.argsort(prediction, kind='stable')
    y, p = np.asarray(target)[order], np.asarray(prediction)[order]
    w = np.asarray(weights, dtype=np.float64)[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    positive = np.add.reduceat(w * y, starts, axis=1)
    negative = np.add.reduceat(w * (1-y), starts, axis=1)
    denominator = positive.sum(1) * negative.sum(1)
    numerator = (positive * (negative.cumsum(1) - .5 * negative)).sum(1)
    return np.divide(numerator, denominator, out=np.full(len(w), np.nan), where=denominator > 0)


def sensitivity(labels, predictions, counts, records):
    from sklearn.metrics import roc_auc_score
    keys = sorted(labels)
    y = np.array([labels[k] for k in keys])
    p = np.array([predictions[k] for k in keys])
    def score(mask):
        selected = [key for key, include in zip(keys, mask) if include]
        patients = Counter(key[0] for key in selected)
        weight = np.array([1 / patients[key[0]] for key in selected])
        a = [float(roc_auc_score(y[mask, c], p[mask, c], sample_weight=weight))
             if len(np.unique(y[mask, c])) == 2 else None for c in range(10)]
        return {'per_class_auc': a, 'macro_auc': float(np.mean(a)) if None not in a else None,
                'num_studies': int(mask.sum())}
    views = {}
    for row in records:
        views.setdefault((row['subject_id'], row['study_id']), []).append(row.get('view_position', ''))
    ap = np.array([bool(views[k]) and all(v.upper() == 'AP' for v in views[k]) for k in keys])
    single = np.array([counts[k] == 1 for k in keys])
    return {'patient_equal': score(np.ones(len(keys), dtype=bool)),
            'patient_equal_single_image': score(single), 'patient_equal_multiple_image': score(~single),
            'patient_equal_ap_only': score(ap),
            'weight_definition': 'Within each reported slice, each Study is weighted by inverse Study count of its patient in that slice; every represented patient has equal total weight.',
            'independent_validation': False}


def uncertainty(labels, new, old, counts, *, samples, seed):
    keys = sorted(labels)
    y = np.array([labels[k] for k in keys], dtype=np.int8)
    a, b = (np.array([d[k] for k in keys]) for d in (new, old))
    patients = sorted({k[0] for k in keys})
    patient_index = {p: i for i, p in enumerate(patients)}
    index = np.array([patient_index[k[0]] for k in keys])
    single = np.array([counts[k] == 1 for k in keys])
    n = np.bincount(index)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=samples)
    differences, single_differences, balanced_differences = [], [], []
    for start in range(0, samples, 128):
        w = draws[start:start+128, index].astype(float)
        d = np.stack([weighted_auc_many(y[:, c], a[:, c], w) -
                      weighted_auc_many(y[:, c], b[:, c], w) for c in range(10)], axis=1)
        ds = np.stack([weighted_auc_many(y[single, c], a[single, c], w[:, single]) -
                       weighted_auc_many(y[single, c], b[single, c], w[:, single]) for c in range(10)], axis=1)
        wb = w / n[index]
        db = np.stack([weighted_auc_many(y[:, c], a[:, c], wb) -
                       weighted_auc_many(y[:, c], b[:, c], wb) for c in range(10)], axis=1)
        differences.append(d); single_differences.append(ds); balanced_differences.append(db)
    d, ds, db = (np.concatenate(x) for x in (differences, single_differences, balanced_differences))
    valid = np.isfinite(d).all(1)
    def interval(x):
        finite = x[np.isfinite(x)]
        return {'valid_samples': len(finite), 'delta_95_percentile_interval':
                np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}
    return {'samples_requested': samples, 'seed': seed, 'patients': len(patients),
            'samples_valid_all_classes': int(valid.sum()),
            'macro': interval(d.mean(1)), 'class8': interval(d[:, 7]),
            'single_image_macro': interval(ds.mean(1)), 'patient_equal_macro': interval(db.mean(1)),
            'per_class': [interval(d[:, c]) for c in range(10)],
            'interpretation': 'Paired whole-patient bootstrap. Exploratory: reused validation and historical model-selection bias are not corrected.'}


def selection_checks(new, old, new_sensitive, old_sensitive, class1_difference):
    return {'macro_strictly_better': new['macro_auc'] > old['macro_auc'],
            'class8_strictly_better': new['per_class_auc'][7] > old['per_class_auc'][7],
            'single_image_macro_not_worse': new['slices']['single_image_study']['macro_auc'] >=
                old['slices']['single_image_study']['macro_auc'] - 1e-12,
            'patient_equal_macro_not_worse': new_sensitive['patient_equal']['macro_auc'] >=
                old_sensitive['patient_equal']['macro_auc'] - 1e-12,
            'class1_exactly_preserved': class1_difference == 0.}


def evaluate(config):
    verify_inputs(config)
    output = Path(config['output']) / 'final_comparison'
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'evaluation_protocol.json', {'configuration': config,
        'candidate_count': 1, 'new_training_runs': 0, 'alternative_fusions_evaluated': 0,
        'test_images_used': False, 'evaluation_type': 'one complete actual FP16 validation, frozen cached parent predictions authenticated per image'})
    started = time.monotonic()
    try:
        records = prediction_records(config['image_root'], config['valid_manifest'], split='valid')
        labels, counts = validation_labels(records)
        if len(records) != 1422 or len(labels) != 859 or len({k[0] for k in labels}) != 228:
            raise ValueError('Original validation coverage changed')
        caches = [image_cache(config[f'cached_{name}_image'], records) for name in ('sprint', 'opacity')]
        reference = read_submission(config['cached_sprint_study'])
        old_score = score_probabilities(labels, reference, counts)
        if abs(old_score['macro_auc'] - .8179628779375786) > 1e-12:
            raise ValueError('Sprint reference does not reproduce the retained score')
        from robust_ensemble.predictor import RobustPredictor
        predictor = RobustPredictor(config)
        inventory = predictor.inventory
        accumulators = [StudyAccumulator(records) for _ in range(3)]
        predictions = [{} for _ in range(3)]
        names = ['candidate', 'sprint', 'opacity']
        maximum_cache_difference = np.zeros(2)
        with ExitStack() as stack:
            streams, writers = [], []
            for name in names:
                pair = []
                for suffix, header in [('_predictions.csv', CSV_HEADER), ('.image_predictions.csv', IMAGE_CSV_HEADER)]:
                    f = stack.enter_context((output / (name + suffix)).open('x', newline=''))
                    writer = csv.writer(f); writer.writerow(header)
                    streams.append(f); pair.append(writer)
                writers.append(pair)
            for i, row in enumerate(records, 1):
                fused, members = predictor.predict_image_components(row['path'])
                for j in range(2):
                    difference = np.max(np.abs(members[j] - caches[j][row['image_id']]))
                    maximum_cache_difference[j] = max(maximum_cache_difference[j], difference)
                    if difference > 1e-7:
                        raise ValueError(f'Parent image math changed: {names[j+1]} max delta {difference}')
                expected = np.asarray(config['fusion_weights']) * np.stack([cache[row['image_id']] for cache in caches])
                if not np.allclose(fused, expected.sum(0), atol=1e-7, rtol=0):
                    raise ValueError('Actual fixed fusion differs from authenticated cache')
                for j, value in enumerate([fused, *members]):
                    write_image_prediction(writers[j][1], row, value)
                    result = accumulators[j].add(row, value)
                    if result is not None:
                        subject, study, probability = result
                        write_prediction(writers[j][0], subject, study, probability)
                        predictions[j][(subject, study)] = probability.copy()
                if i % 100 == 0 or i == len(records):
                    _write_json(output / 'progress.json', {'state': 'validation', 'images_done': i,
                        'images_total': len(records), 'elapsed_seconds': time.monotonic()-started})
                    print(f'[ROBUST VALID] {i}/{len(records)}', flush=True)
            for stream in streams:
                stream.flush(); os.fsync(stream.fileno())
        if not all(acc.complete() for acc in accumulators):
            raise ValueError('Incomplete Study aggregation')
        predictor.synchronize()
        inference_seconds = time.monotonic() - started
        torch = predictor.torch
        del predictor
        torch.cuda.empty_cache()
        candidate = predictions[0]
        new_score = score_probabilities(labels, candidate, counts)
        new_sensitive = sensitivity(labels, candidate, counts, records)
        old_sensitive = sensitivity(labels, reference, counts, records)
        class1_difference = max(abs(candidate[k][0] - reference[k][0]) for k in labels)
        checks = selection_checks(new_score, old_score, new_sensitive, old_sensitive, class1_difference)
        bootstrap = uncertainty(labels, candidate, reference, counts,
            samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
        verify_inputs(config)
        report = {'complete': True, 'candidate_name': 'Preserved_Sprint80_Opacity512_20',
            'candidate': new_score, 'baseline': old_score,
            'macro_auc_delta': new_score['macro_auc'] - old_score['macro_auc'],
            'per_class_auc_delta': (np.array(new_score['per_class_auc']) - old_score['per_class_auc']).tolist(),
            'class8_auc_delta': new_score['per_class_auc'][7] - old_score['per_class_auc'][7],
            'class1_max_abs_probability_difference': class1_difference,
            'candidate_sensitivity': new_sensitive, 'baseline_sensitivity': old_sensitive,
            'bootstrap': bootstrap, 'selection_checks': checks, 'candidate_won': all(checks.values()),
            'fusion_weights': config['fusion_weights'], 'opacity_weight_per_class': config['opacity_weight_per_class'],
            'candidate_inference_assets': inventory['files'], 'candidate_inference_bytes': inventory['total_bytes'],
            'asset_inventory': inventory, 'size_pass': inventory['total_bytes'] <= 500_000_000,
            'manifest_sha256': config['valid_manifest_sha256'],
            'candidate_predictions_sha256': sha256_file(output / 'candidate_predictions.csv'),
            'candidate_image_predictions_sha256': sha256_file(output / 'candidate.image_predictions.csv'),
            'parent_image_max_abs_cache_difference': maximum_cache_difference.tolist(),
            'parent_scores_diagnostic_only': {name: score_probabilities(labels, p, counts)
                                             for name, p in zip(names[1:], predictions[1:])},
            'validation_elapsed_seconds': inference_seconds, 'total_elapsed_seconds': time.monotonic()-started,
            'validation_images': len(records), 'new_training_runs': 0, 'new_optimizer_updates': 0,
            'candidate_count': 1, 'alternative_fusions_evaluated': 0, 'test_images_used': False,
            'formal_T4_certification': False, 'deployment_accepted': False,
            'selection_note': 'Local robustness gates only; neither independent generalization nor platform improvement has been established.'}
        _write_json(output / 'report.json', report)
        selection = {'selected': 'robust_ensemble' if report['candidate_won'] else 'sprint_candidate',
            'candidate_won': report['candidate_won'], 'selection_checks': checks,
            'fusion_weights': config['fusion_weights'], 'opacity_weight_per_class': config['opacity_weight_per_class'],
            'report': str(output / 'report.json'), 'tta': 'hflip2', 'research_selection_only': True}
        _write_json(output / 'selection.json', selection)
        _write_json(output / 'progress.json', {'state': 'completed', 'candidate_won': report['candidate_won']})
        print(json.dumps({'result': new_score, 'delta': report['macro_auc_delta'], 'checks': checks}), flush=True)
        return report
    except BaseException as error:
        _write_json(output / 'failure.json', {'complete': False, 'error': f'{type(error).__name__}: {error}',
            'elapsed_seconds': time.monotonic()-started, 'selection_changed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
