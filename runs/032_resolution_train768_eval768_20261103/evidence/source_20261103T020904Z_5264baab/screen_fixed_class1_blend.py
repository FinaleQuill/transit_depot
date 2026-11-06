"""Fifth bounded CPU screen: two explicitly valid-selected class1 blends.

This is exploratory manual hyperparameter selection on repeatedly used valid,
not an independent confirmation or a claim of no validation fitting.
"""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np

from screen_study_pooling import (
    ARTIFACT, MANIFEST, REPORTS, prepare_auc, read_probabilities, sha, weighted_auc,
)


def main():
    plan_file = ARTIFACT / 'fifth_plan.json'
    output = ARTIFACT / 'fifth_class1_blend_report.json'
    if output.exists():
        raise FileExistsError('Do not overwrite previous results')
    plan = json.loads(plan_file.read_text())
    expected = {'AB_B75_class1': (.25, .75), 'AB_B100_class1': (0., 1.)}
    if set(plan['candidates']) != set(expected):
        raise ValueError('Do not expand registered candidate set')
    for name, (a, b) in expected.items():
        spec = plan['candidates'][name]
        if (spec['A_class1_weight'] != a or spec['B_class1_weight'] != b
                or spec['other_classes_A_B_weights'] != [.5, .5]):
            raise ValueError('Registered weights changed')
    labels, image_to_study = {}, {}
    for row in csv.DictReader(MANIFEST.open(encoding='utf-8-sig')):
        if row['split'] != 'valid':
            raise ValueError('Only valid inputs accepted')
        key = row['subject_id'], row['study_id']
        target = json.loads(row['targets'])
        if key in labels and labels[key] != target:
            raise ValueError('Inconsistent Study labels')
        labels[key] = target
        image_key = (*key, row['image_id'])
        if image_key in image_to_study:
            raise ValueError('Duplicate image')
        image_to_study[image_key] = key
    keys, image_keys = sorted(labels), sorted(image_to_study)
    if len(keys) != 859 or len(image_keys) != 1422:
        raise ValueError('Full original official valid required')
    groups = [np.array([i for i, k in enumerate(image_keys) if k[:2] == study])
              for study in keys]
    counts = np.array([len(g) for g in groups])
    sources, provenance, study_sources = {}, [], {}
    for name in ['A', 'B']:
        directory = REPORTS / name / 'valid_hflip2'
        report_file = directory / 'validation_report.json'
        report = json.loads(report_file.read_text())
        image_file = directory / 'valid.image_predictions.csv'
        study_file = directory / 'valid_predictions.csv'
        if (report['manifest_sha256'] != sha(MANIFEST) or not report['complete']
                or report['image_predictions_sha256'] != sha(image_file)
                or report['predictions_sha256'] != sha(study_file)
                or report['image_size'] != 768 or report['tta'] != 'hflip2'
                or report['precision'] != 'float16' or report['member_count'] != 1):
            raise ValueError('Original SHA-bound FP16 768/hflip2 inputs required')
        images = read_probabilities(image_file, image=True)
        studies = read_probabilities(study_file)
        if set(images) != set(image_keys) or set(studies) != set(keys):
            raise ValueError('Exact image/Study coverage required')
        sources[name] = np.array([images[k] for k in image_keys])
        study_sources[name] = np.array([studies[k] for k in keys])
        provenance.append({'name': name, 'report_sha256': sha(report_file),
                           'image_predictions_sha256': sha(image_file),
                           'predictions_sha256': sha(study_file)})
    per_image = {'AB_equal': (sources['A'] + sources['B']) / 2}
    weights = {'AB_equal': {'A': [.5] * 10, 'B': [.5] * 10}}
    for name, (a, b) in expected.items():
        wa, wb = np.full(10, .5), np.full(10, .5)
        wa[0], wb[0] = a, b
        per_image[name] = sources['A'] * wa + sources['B'] * wb
        weights[name] = {'A': wa.tolist(), 'B': wb.tolist()}
    candidates = {name: np.array([p[g].mean(axis=0) for g in groups])
                  for name, p in per_image.items()}
    linear_replay = {}
    for name in candidates:
        replay = (study_sources['A'] * weights[name]['A']
                  + study_sources['B'] * weights[name]['B'])
        difference = float(np.max(np.abs(candidates[name] - replay)))
        if difference > 1e-15:
            raise ValueError('Fixed image blend / Study mean did not commute')
        if not np.array_equal(candidates[name][:, 1:], candidates['AB_equal'][:, 1:]):
            raise ValueError('Unregistered class change')
        linear_replay[name] = {'max_absolute_roundoff': difference,
                               'unchanged_other_nine_classes': True}
    y = np.array([labels[k] for k in keys], dtype=np.int8)
    patients, patient_index = np.unique([k[0] for k in keys], return_inverse=True)
    prepared = {name: [prepare_auc(y[:, c], p[:, c]) for c in range(10)]
                for name, p in candidates.items()}

    def aucs(name, sample_weights):
        return [weighted_auc(p, sample_weights) for p in prepared[name]]

    def metrics(name, sample_weights):
        values = aucs(name, sample_weights)
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
        scores[name]['macro_auc_delta_from_AB_equal'] = (
            scores[name]['macro_auc'] - scores['AB_equal']['macro_auc'])
    if abs(scores['AB_equal']['macro_auc'] - .8147866171472288) > 1e-14:
        raise ValueError('Original AB baseline did not replay')
    rng = np.random.default_rng(20261031)
    deltas = {name: [] for name in expected}
    for _ in range(1000):
        multiplicities = np.bincount(rng.integers(0, len(patients), len(patients)),
                                     minlength=len(patients))
        sample_weights = multiplicities[patient_index]
        baseline = aucs('AB_equal', sample_weights)
        for name in expected:
            delta = [None if a is None or b is None else a - b
                     for a, b in zip(aucs(name, sample_weights), baseline)]
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
        intervals[name] = {'per_class_delta': columns[:10], 'macro_auc_delta': columns[10],
                           'class_1_4_8_mean_delta': columns[11]}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(plan_file), 'script_sha256': sha(Path(__file__)),
              'auc_helper_sha256': sha(Path(__file__).with_name('screen_study_pooling.py')),
              'manifest_sha256': sha(MANIFEST), 'source_validations': provenance,
              'num_images': len(image_keys), 'num_studies': len(keys),
              'num_patients': len(patients), 'fixed_class_weights': weights,
              'linear_aggregation_replay': linear_replay, 'scores': scores,
              'patient_paired_bootstrap': {'samples': 1000, 'seed': 20261031,
                  'sampling_unit': 'patient_with_all_Studies', 'reference': 'AB_equal',
                  'results': intervals}, 'validation_selected_hyperparameters': True,
              'validation_fitted_class_weights': True, 'numerical_optimizer_used': False,
              'thresholds_fitted': False, 'test_used': False, 'new_training': False,
              'new_model_inference': False, 'deployment_selected': False,
              'latency_80ms_promised': False, 'formal_T4_certification': False,
              'interpretation': plan['interpretation'], 'stop_rule': plan['stop_rule']}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({n: {'auc': s['macro_auc'], 'delta': s['macro_auc_delta_from_AB_equal']}
                      for n, s in scores.items()}, indent=2))


if __name__ == '__main__':
    main()
