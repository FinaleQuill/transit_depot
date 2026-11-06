"""CPU-only, fixed-path diagnostic of existing official-valid predictions.

Run: python class14_diagnostic.py [--output-directory NEW_DIRECTORY]
Only the six registered evaluation directories and valid_manifest.csv are data inputs.
No image bytes, checkpoints, test files, model frameworks, or inference are accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path('/home/featurize/work/cloud_project_v2')
MANIFEST = PROJECT / 'data/clean_v1_reviewed/valid_manifest.csv'
EVALUATIONS = PROJECT / 'artifacts/resolution_explore_20261103/evaluations'
ARTIFACT = PROJECT / 'artifacts/resolution_e1_analysis_20261103'
PREREGISTRATION = ARTIFACT / 'prereg/class14_diagnostic.txt'
INPUTS = {
    'E1': {'res640': 'res640_best', 'res768': 'res768_best', 'res896': 'res896_best'},
    'E3': {'res640': 'res640_retry1', 'res768': 'res768', 'res896': 'res896'},
}
NAMES = tuple(INPUTS['E1'])
LABELS = ['Enlarged Cardiomediastinum', 'Pneumothorax', 'Consolidation', 'Pneumonia',
          'Edema', 'Cardiomegaly', 'Atelectasis', 'Lung Opacity', 'Pleural Effusion', 'No Finding']
SAMPLES = 2000
SEED = 20261103


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return float(a / b) if b else None


def auc_draws(labels, probabilities, weights):
    """Tie-aware weighted Mann-Whitney AUC, one independent draw per weight row."""
    order = np.argsort(probabilities, kind='stable')
    scores = probabilities[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    sorted_weights = weights[:, order]
    positive = np.add.reduceat(sorted_weights * labels[order], starts, axis=1)
    negative = np.add.reduceat(sorted_weights * (1 - labels[order]), starts, axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def interval(draws):
    finite = draws[np.isfinite(draws)]
    bounds = np.quantile(finite, [.025, .975]).tolist() if len(finite) else None
    return {'ci95': bounds, 'ci_width': bounds[1] - bounds[0] if bounds else None,
            'valid_resamples': len(finite), 'invalid_resamples': len(draws) - len(finite)}


def read_inputs():
    prereg = PREREGISTRATION.read_text()
    if len([line for line in prereg.splitlines() if line.strip()]) != 4:
        raise ValueError('The preregistration must contain exactly four nonblank lines')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = OrderedDict()
    for index, row in enumerate(rows):
        if row['split'] != 'valid' or row['study_key'] != f"{row['subject_id']}/{row['study_id']}":
            raise ValueError('Only original official-valid identities are allowed')
        labels = json.loads(row['targets'])
        if len(labels) != 10 or any(type(value) is not int or value not in (0, 1) for value in labels):
            raise ValueError('Invalid binary targets in official valid')
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:3] != ('valid', row['subject_id'], row['study_id']):
            raise ValueError('A manifest image reference escapes its valid Study')
        group = groups.setdefault(row['study_key'], {'subject': row['subject_id'], 'target': labels, 'images': []})
        if group['target'] != labels:
            raise ValueError('Inconsistent Study labels')
        group['images'].append(index)
    if (len(rows), len(groups), len({row['subject_id'] for row in rows})) != (1422, 859, 228):
        raise ValueError('Official-valid dimensions changed')
    if len({row['image_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    study_keys = np.asarray(list(groups))
    subjects = np.asarray([group['subject'] for group in groups.values()])
    labels = np.asarray([group['target'] for group in groups.values()], dtype=np.int8)
    ap_only = np.asarray([all(rows[i]['view_position'].upper() == 'AP' for i in group['images'])
                          for group in groups.values()])
    expected = {'study_keys': study_keys, 'subject_ids': subjects, 'targets': labels, 'ap_only': ap_only,
                'image_ids': np.asarray([row['image_id'] for row in rows]),
                'image_study_keys': np.asarray([row['study_key'] for row in rows]),
                'image_view_positions': np.asarray([row['view_position'] for row in rows]),
                'image_counts': np.asarray([len(group['images']) for group in groups.values()])}
    binding = {'manifest': {'path': str(MANIFEST), 'sha256': digest(MANIFEST)},
               'preregistration': {'path': str(PREREGISTRATION), 'sha256': digest(PREREGISTRATION)},
               'script': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}, 'evaluations': {}}
    bundles = {'E1': {}, 'E3': {}}
    for epoch, entries in INPUTS.items():
        for name, directory in entries.items():
            source = EVALUATIONS / directory
            paths = {kind: source / kind for kind in ('report.json', 'metadata.json', 'predictions.npz')}
            hashes = {kind: digest(path) for kind, path in paths.items()}
            report = json.loads(paths['report.json'].read_text())
            metadata = json.loads(paths['metadata.json'].read_text())
            if not report.get('complete') or report['metadata_sha256'] != hashes['metadata.json']:
                raise ValueError(f'{epoch}/{name}: incomplete or mismatched metadata')
            if metadata['prediction_files_sha256']['predictions.npz'] != hashes['predictions.npz']:
                raise ValueError(f'{epoch}/{name}: prediction SHA mismatch')
            if metadata['valid_manifest_sha256'] != binding['manifest']['sha256'] or metadata['class_ids'] != list(range(1, 11)):
                raise ValueError(f'{epoch}/{name}: manifest or class order mismatch')
            if metadata['training_compliance']['checkpoint_epoch'] != int(epoch[1:]):
                raise ValueError(f'{epoch}/{name}: checkpoint epoch differs')
            expected_selection = 'valid_selected_best' if epoch == 'E1' else 'fixed_final_E3'
            if metadata['selection_kind'] != expected_selection:
                raise ValueError(f'{epoch}/{name}: checkpoint selection differs')
            with np.load(paths['predictions.npz'], allow_pickle=False) as archive:
                bundle = {key: archive[key] for key in archive.files}
            for field, expected_value in expected.items():
                if not np.array_equal(bundle[field], expected_value):
                    raise ValueError(f'{epoch}/{name}: misaligned {field}')
            for field, shape in (('hflip2', (859, 10)), ('image_probabilities', (1422, 10)),
                                 ('per_view_probabilities', (1422, 2, 10))):
                value = bundle[field]
                if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                    raise ValueError(f'{epoch}/{name}: invalid {field}')
            rebuilt_images = bundle['per_view_probabilities'].astype(np.float32).mean(axis=1).astype(np.float64)
            if not np.array_equal(rebuilt_images, bundle['image_probabilities']):
                raise ValueError(f'{epoch}/{name}: image TTA arithmetic mismatch')
            rebuilt_studies = np.stack([rebuilt_images[group['images']].mean(axis=0) for group in groups.values()])
            if not np.array_equal(rebuilt_studies, bundle['hflip2']):
                raise ValueError(f'{epoch}/{name}: Study aggregation mismatch')
            actual_auc = [float(auc_draws(labels[:, c], bundle['hflip2'][:, c], np.ones((1, 859)))[0])
                          for c in range(10)]
            if not np.allclose(actual_auc, report['scores']['hflip2']['all']['per_class_auc'], atol=1e-12, rtol=0):
                raise ValueError(f'{epoch}/{name}: reported AUC mismatch')
            bundles[epoch][name] = bundle
            binding['evaluations'][epoch + '/' + name] = {'directory': str(source),
                'files': {key: {'path': str(paths[key]), 'sha256': value} for key, value in hashes.items()},
                'selection_kind': metadata['selection_kind'], 'checkpoint_epoch': int(epoch[1:]),
                'source_checkpoint_sha256': metadata['source_checkpoint_sha256'],
                'training_compliance': metadata['training_compliance']}
    return rows, groups, expected, bundles, binding


def diagnostic(rows, groups, expected, bundles):
    y, subjects, ap = expected['targets'], expected['subject_ids'], expected['ap_only']
    study_keys = expected['study_keys']
    unique_subjects, inverse = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(len(unique_subjects), size=len(unique_subjects)),
                       minlength=len(unique_subjects))[inverse] for _ in range(SAMPLES)])
    subject_totals = Counter(subjects)
    group_list = list(groups.values())
    bootstrap = {}
    for name in NAMES:
        print('Computing patient bootstrap:', name, flush=True)
        bootstrap[name] = {}
        for slice_name, mask, classes in (('all', np.ones(len(y), dtype=bool), range(10)),
                                          ('ap_only', ap, (0, 3))):
            result = []
            for c in classes:
                labels = y[mask, c]
                scores = bundles['E1'][name]['hflip2'][mask, c]
                draws = auc_draws(labels, scores, weights[:, mask])
                result.append({'class_id': c + 1, 'label': LABELS[c],
                    'auc': float(auc_draws(labels, scores, np.ones((1, len(labels))))[0]),
                    'positive_studies': int(labels.sum()), 'negative_studies': int(len(labels) - labels.sum()),
                    'positive_patients': len(np.unique(subjects[mask][labels == 1])),
                    'negative_patients': len(np.unique(subjects[mask][labels == 0])), **interval(draws)})
            bootstrap[name][slice_name] = result

    def concentration(indices):
        counts = Counter(subjects[indices])
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        return {'patients': [{'subject_id': patient, 'error_studies': count,
                             'all_valid_studies': subject_totals[patient], 'share_of_errors': ratio(count, len(indices))}
                            for patient, count in ordered],
                'largest_patient_error_studies': ordered[0][1] if ordered else 0,
                'largest_patient_share': ratio(ordered[0][1], len(indices)) if ordered else None,
                'top_three_patients_share': ratio(sum(count for _, count in ordered[:3]), len(indices)),
                'single_patient_more_than_half': bool(ordered and ordered[0][1] > len(indices) / 2)}

    def study_case(index, c):
        group = group_list[index]
        images = []
        for image_index in group['images']:
            row = rows[image_index]
            images.append({'image_id': row['image_id'], 'relative_path': row['relative_path'],
                'view_position': row['view_position'], 'file_sha256_from_manifest': row['file_sha256'],
                'E1_image_probability': {name: float(bundles['E1'][name]['image_probabilities'][image_index, c]) for name in NAMES},
                'E3_image_probability': {name: float(bundles['E3'][name]['image_probabilities'][image_index, c]) for name in NAMES}})
        return {'study_key': str(study_keys[index]), 'subject_id': str(subjects[index]),
                'class_label': int(y[index, c]), 'all_positive_class_ids': (np.flatnonzero(y[index]) + 1).tolist(),
                'ap_only': bool(ap[index]), 'image_count': len(images), 'images': images,
                'E1_probability': {name: float(bundles['E1'][name]['hflip2'][index, c]) for name in NAMES},
                'E3_probability': {name: float(bundles['E3'][name]['hflip2'][index, c]) for name in NAMES}}

    def summarize_errors(mask, c, kind, ordering_probability, control_name):
        indices = np.flatnonzero(mask)
        pool = y[:, c] == (0 if kind == 'FP' else 1)
        controls = pool & ~mask
        pool_n, control_n = int(pool.sum()), int(controls.sum())
        ordering = indices[np.argsort(ordering_probability[indices], kind='stable')]
        if kind == 'FP':
            ordering = ordering[::-1]
        return {'studies': len(indices), 'patients': len(np.unique(subjects[indices])),
                'referenced_images': int(expected['image_counts'][indices].sum()),
                'ap_only_studies': int(ap[indices].sum()), 'ap_only_share': ratio(int(ap[indices].sum()), len(indices)),
                'label_matched_pool': {'class_label': 0 if kind == 'FP' else 1, 'studies': pool_n,
                    'patients': len(np.unique(subjects[pool])), 'ap_only_studies': int((ap & pool).sum()),
                    'ap_only_share': ratio(int((ap & pool).sum()), pool_n)},
                'complement_control': {'name': control_name, 'studies': control_n,
                    'patients': len(np.unique(subjects[controls])), 'ap_only_studies': int((ap & controls).sum()),
                    'ap_only_share': ratio(int((ap & controls).sum()), control_n)},
                'colabels': [{'class_id': d + 1, 'label': LABELS[d],
                    'error_positive_studies': int(y[mask, d].sum()), 'error_denominator': len(indices),
                    'error_rate': ratio(int(y[mask, d].sum()), len(indices)),
                    'pool_positive_studies': int(y[pool, d].sum()), 'pool_denominator': pool_n,
                    'pool_rate': ratio(int(y[pool, d].sum()), pool_n),
                    'control_positive_studies': int(y[controls, d].sum()), 'control_denominator': control_n,
                    'control_rate': ratio(int(y[controls, d].sum()), control_n)} for d in range(10) if d != c],
                'patient_concentration': concentration(indices),
                'cases': [study_case(int(index), c) for index in ordering]}

    classes = {}
    for c in (0, 3):
        models, masks = {}, {}
        for name in NAMES:
            p = bundles['E1'][name]['hflip2'][:, c]
            masks[name] = {'FP': (y[:, c] == 0) & (p >= .5), 'FN': (y[:, c] == 1) & (p < .5)}
            models[name] = {kind: summarize_errors(mask, c, kind, p, 'TN' if kind == 'FP' else 'TP')
                            for kind, mask in masks[name].items()}
        shared = {}
        mean_score = np.mean([bundles['E1'][name]['hflip2'][:, c] for name in NAMES], axis=0)
        for kind in ('FP', 'FN'):
            mask = np.logical_and.reduce([masks[name][kind] for name in NAMES])
            shared[kind] = summarize_errors(mask, c, kind, mean_score, 'label-matched Studies outside the three-model shared set')
        signatures = []
        for kind in ('FP', 'FN'):
            for bits in range(1, 8):
                mask = y[:, c] == (0 if kind == 'FP' else 1)
                names = []
                for j, name in enumerate(NAMES):
                    flag = bool(bits & (1 << j))
                    mask &= masks[name][kind] if flag else ~masks[name][kind]
                    if flag:
                        names.append(name)
                signatures.append({'error_type': kind, 'models_with_error': names, 'studies': int(mask.sum()),
                                   'patients': len(np.unique(subjects[mask])), 'ap_only_studies': int((mask & ap).sum()),
                                   'study_keys': study_keys[mask].tolist()})
        mixed = []
        for subject in unique_subjects:
            indices = np.flatnonzero(subjects == subject)
            if len(indices) > 1 and len(np.unique(y[indices, c])) == 2:
                mixed.append({'subject_id': str(subject), 'studies': len(indices),
                              'positive_studies': int(y[indices, c].sum()),
                              'negative_studies': int(len(indices) - y[indices, c].sum()),
                              'study_keys': study_keys[indices].tolist()})
        positives = y[:, c] == 1
        classes[str(c + 1)] = {'class_id': c + 1, 'zero_based_index': c, 'label': LABELS[c],
            'positive_studies': int(positives.sum()), 'positive_patients': len(np.unique(subjects[positives])),
            'negative_studies': int((~positives).sum()), 'negative_patients': len(np.unique(subjects[~positives])),
            'ap_only_positive_studies': int((positives & ap).sum()),
            'ap_only_positive_patients': len(np.unique(subjects[positives & ap])),
            'multi_study_patient_count': sum(count > 1 for count in subject_totals.values()),
            'multi_study_patients_with_mixed_labels_count': len(mixed), 'multi_study_patients_with_mixed_labels': mixed,
            'models': models, 'shared_across_all_three_E1_models': shared, 'exclusive_error_signatures': signatures}
    e3 = {}
    for name in NAMES:
        e3[name] = {}
        for c in (0, 3):
            p = bundles['E3'][name]['hflip2'][:, c]
            p1 = bundles['E1'][name]['hflip2'][:, c]
            e1_wrong = (p1 >= .5) != y[:, c]
            e3_wrong = (p >= .5) != y[:, c]
            e3[name][str(c + 1)] = {'auc': float(auc_draws(y[:, c], p, np.ones((1, len(y))))[0]),
                'FP': int(((y[:, c] == 0) & (p >= .5)).sum()), 'FN': int(((y[:, c] == 1) & (p < .5)).sum()),
                'E1_correct_E3_wrong': int((~e1_wrong & e3_wrong).sum()),
                'E1_wrong_E3_correct': int((e1_wrong & ~e3_wrong).sum())}
    return bootstrap, classes, e3


def markdown(report):
    lines = ['# Class 1 / class 4 diagnostic', '',
        'All analyses use official valid predictions. E1 means the valid-selected best checkpoint, confirmed epoch 1 for all three training resolutions; E3 is fixed-final context. Inference is common 768 with hflip2.', '',
        'Class 1 = Enlarged Cardiomediastinum (array index 0); class 4 = Pneumonia (array index 3).', '',
        '## Support and patient clustering', '',
        '| Class | Positive Studies / patients | AP-only positive Studies / patients | Mixed-label multi-Study patients |',
        '| --- | ---: | ---: | ---: |']
    for c in report['class14'].values():
        lines.append(f"| {c['class_id']} {c['label']} | {c['positive_studies']} / {c['positive_patients']} | {c['ap_only_positive_studies']} / {c['ap_only_positive_patients']} | {c['multi_study_patients_with_mixed_labels_count']} / {c['multi_study_patient_count']} |")
    lines += ['', 'Patient labels belong to Studies: a patient can contribute both positive and negative Studies. Mixed labels within a patient are not proof of erroneous labels.', '',
        '## E1 absolute AUC and patient bootstrap intervals', '',
        '2,000 draws, seed 20261103; resample all 228 patients with replacement, retain each patient\'s complete set of Studies, and use tie-aware weighted AUC. AP-only uses the same patient draws followed by the AP-only slice. Intervals are 2.5/97.5 percentile intervals; unusable single-label draws are excluded and counted. These are absolute per-model AUC intervals, not paired difference intervals.', '',
        '| Class | res640 AUC [95% CI]; width | res768 AUC [95% CI]; width | res896 AUC [95% CI]; width |',
        '| --- | --- | --- | --- |']
    def cell(row):
        return f"{row['auc']:.6f} [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]; {row['ci_width']:.6f}"
    for c in range(10):
        lines.append('| ' + str(c + 1) + ' ' + LABELS[c] + ' | ' + ' | '.join(cell(report['E1_absolute_auc'][name]['all'][c]) for name in NAMES) + ' |')
    lines += ['', '| AP-only class | res640 AUC [95% CI]; width | res768 AUC [95% CI]; width | res896 AUC [95% CI]; width |',
              '| --- | --- | --- | --- |']
    for index, c in enumerate((1, 4)):
        lines.append('| ' + str(c) + ' | ' + ' | '.join(cell(report['E1_absolute_auc'][name]['ap_only'][index]) for name in NAMES) + ' |')
    lines += ['', '## E1 errors at fixed 0.5, for review only', '',
        'A probability >=0.5 counts as positive. No threshold was fitted or changed. These error counts describe this operating point and do not replace AUC. Shared means the intersection of errors across all three E1 models.', '',
        '| Class | Model / set | Error | Studies / patients | AP-only / errors | Label-matched pool AP-only | Largest patient share |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in report['class14'].values():
        sets = list(c['models'].items()) + [('shared E1 all three', c['shared_across_all_three_E1_models'])]
        for name, values in sets:
            for kind, value in values.items():
                pool = value['label_matched_pool']
                concentration = value['patient_concentration']
                largest = concentration['largest_patient_error_studies']
                lines.append(f"| {c['class_id']} | {name} | {kind} | {value['studies']} / {value['patients']} | {value['ap_only_studies']} / {value['studies']} | {pool['ap_only_studies']} / {pool['studies']} | {largest} / {value['studies']} |")
    lines += ['', 'The JSON provides every error Study, all referenced image IDs and valid relative paths, view positions, all three E1/E3 Study and image probabilities, exclusive error signatures, and full patient counts. Image labels are inherited from the Study: a high per-image probability does not establish an image-level annotation error.', '',
        '## Co-labels of shared E1 errors with explicit controls', '',
        'The pool has the same target-class label as the error set (negative for FP, positive for FN). The complement control excludes the shared error set; it can include errors from a subset of models. Per-model JSON controls are TN for FP and TP for FN. Counts are descriptive, without independence, multiplicity, or confounder adjustment.', '']
    for c in report['class14'].values():
        for kind, values in c['shared_across_all_three_E1_models'].items():
            lines += [f"Class {c['class_id']} shared {kind}:", '',
                '| Co-label | Shared errors positive / total | Label-matched pool positive / total | Complement control positive / total |',
                '| --- | ---: | ---: | ---: |']
            for row in values['colabels']:
                lines.append(f"| {row['class_id']} {row['label']} | {row['error_positive_studies']} / {row['error_denominator']} | {row['pool_positive_studies']} / {row['pool_denominator']} | {row['control_positive_studies']} / {row['control_denominator']} |")
            concentration = values['patient_concentration']
            lines += ['', 'Top error patients: ' + '; '.join(f"{row['subject_id']}: {row['error_studies']} errors / {row['all_valid_studies']} total valid Studies" for row in concentration['patients'][:5]) + '.',
                      f"Single patient contributes more than half of errors: {concentration['single_patient_more_than_half']}.", '']
    lines += ['## Shared E1 case references', '',
        'Three highest-confidence shared FP and three lowest-probability shared FN Studies are listed per class, ordered by the mean of the three E1 probabilities. These are metadata references to all images of each Study; this analysis does not open the images. The full JSON contains every case.', '',
        '| Class / error | Study | E1 probabilities 640 / 768 / 896 | AP-only | Valid image references |',
        '| --- | --- | --- | --- | --- |']
    for c in report['class14'].values():
        for kind, values in c['shared_across_all_three_E1_models'].items():
            for case in values['cases'][:3]:
                scores = ' / '.join(f"{case['E1_probability'][name]:.6f}" for name in NAMES)
                paths = '; '.join(image['relative_path'] for image in case['images'])
                lines.append(f"| {c['class_id']} / {kind} | {case['study_key']} | {scores} | {case['ap_only']} | {paths} |")
    lines += ['## E3 context and existing galleries', '',
        '| Model | Class | E3 AUC | E3 FP / FN | E1 correct → E3 wrong | E1 wrong → E3 correct |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for name, values in report['E3_context'].items():
        for c, row in values.items():
            lines.append(f"| {name} | {c} | {row['auc']:.6f} | {row['FP']} / {row['FN']} | {row['E1_correct_E3_wrong']} | {row['E1_wrong_E3_correct']} |")
    lines += ['', report['existing_gallery_provenance'], '',
        'The two existing galleries are not evidence of shared E1 errors. This analysis neither reads nor overwrites their image files.', '',
        '## Interpretation limits and reproducibility', '',
        'Sparse positive patients, particularly class 1 (16 all-view; 11 AP-only), limit precision. Absolute AUC interval overlap is not a paired significance test. These intervals condition on chosen checkpoints and this valid set; they do not capture training-seed variation or remove repeated-valid checkpoint selection bias.', '',
        'Neither these probabilities nor viewing a radiograph can establish that disease is early, that a label is wrong, or why an error occurred. Clinical adjudication and longitudinal information would be needed. Mixed labels across Studies of one patient can reflect changing clinical state or annotation uncertainty; this analysis cannot distinguish these explanations.', '',
        'No test, image-byte, checkpoint-byte, GPU, training, platform-submission, threshold-fitting, or declaration access was performed.', '',
        f"Manifest SHA-256: `{report['bindings']['manifest']['sha256']}`.",
        f"Preregistration SHA-256: `{report['bindings']['preregistration']['sha256']}`.",
        f"Script SHA-256: `{report['bindings']['script']['sha256']}`.", '',
        'The JSON records SHA-256 for all 18 evaluation inputs, their checkpoint identity and recorded compliance evidence. Reproduce with:', '',
        f"`python {Path(__file__).resolve()} --output-directory /tmp/class14_diagnostic_replay`", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=ARTIFACT / 'analysis')
    args = parser.parse_args()
    rows, groups, expected, bundles, binding = read_inputs()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / 'class14_diagnostic_inputs.json'
    frozen_path.write_text(json.dumps(binding, indent=2, ensure_ascii=False) + '\n')
    bootstrap, classes, e3 = diagnostic(rows, groups, expected, bundles)
    for entry in binding['evaluations'].values():
        for record in entry['files'].values():
            if digest(record['path']) != record['sha256']:
                raise RuntimeError('An evaluation input changed during analysis')
    if digest(MANIFEST) != binding['manifest']['sha256']:
        raise RuntimeError('The official manifest changed during analysis')
    report = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {'split': 'valid', 'inference_resolution': 768, 'tta': 'hflip2', 'review_threshold': .5,
            'bootstrap_samples': SAMPLES, 'bootstrap_seed': SEED, 'bootstrap_unit': 'patient_with_all_Studies',
            'confidence_interval': 'absolute AUC 95% percentile, classwise; no multiplicity adjustment',
            'patient_negative_and_positive_counts_can_overlap': True, 'test_accessed': False,
            'gpu_used': False, 'new_training': False, 'image_bytes_read': False, 'thresholds_fitted': False},
        'bindings': binding, 'manifest_dimensions': {'images': 1422, 'studies': 859, 'patients': 228,
                                                   'ap_only_studies': int(expected['ap_only'].sum())},
        'E1_absolute_auc': bootstrap, 'class14': classes, 'E3_context': e3,
        'existing_gallery_provenance': 'Task context identifies the existing two error galleries as res896 E3 high-confidence misclassifications, not errors shared across models. Their provenance is stated here without reading or changing the galleries.',
        'clinical_limit': 'Images/probabilities cannot establish early disease or label errors; mixed labels across patient Studies are not adjudicated here.'}
    json_path = output / 'class14_diagnostic.json'
    md_path = output / 'class14_diagnostic.md'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    md_path.write_text(markdown(report))
    print(json.dumps({'json': str(json_path), 'json_sha256': digest(json_path), 'markdown': str(md_path),
                      'markdown_sha256': digest(md_path)}, indent=2))


if __name__ == '__main__':
    main()
