"""CPU-only audit of existing validation members; never forms a new fusion."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path('/home/featurize/work')
SOURCE = ROOT / 'reconstruction_runs_20261103/ENSEMBLE_SPRINT_20261105/final_comparison'
OUTPUT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv'
MEMBERS = {
    'preserved_sprint': SOURCE / 'baseline_predictions.csv',
    'previous_three_member': SOURCE / 'candidate_predictions.csv',
    'sprint_base768': SOURCE / 'base768.predictions.csv',
    'context512': SOURCE / 'context512.predictions.csv',
    'opacity512': SOURCE / 'opacity512.predictions.csv',
    'online_reference_041': ROOT / 'cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/reports/A/valid_hflip2/valid_predictions.csv',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_auc(targets, predictions, weights):
    """Weighted AUC by independent exhaustive positive-negative pair counts."""
    values = []
    for c in range(10):
        positive = targets[:, c] == 1
        negative = ~positive
        if not positive.any() or not negative.any():
            raise ValueError('Slice lacks one class polarity')
        differences = predictions[positive, c, None] - predictions[negative, c]
        pair_weights = weights[positive, None] * weights[negative]
        values.append(float((((differences > 0) + 0.5 * (differences == 0)) * pair_weights).sum() / pair_weights.sum()))
    return {'macro_auc': float(np.mean(values)), 'per_class_auc': values}


def read_predictions(path, keys):
    values = defaultdict(dict)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['Subject_id'], row['Study_id']
            cls = int(row['Predict_class'])
            if cls in values[key] or cls not in range(1, 11):
                raise ValueError('Duplicate or invalid class')
            values[key][cls] = float(row['Probability'])
    if set(values) != set(keys) or any(set(v) != set(range(1, 11)) for v in values.values()):
        raise ValueError('Prediction identity or class coverage mismatch')
    matrix = np.asarray([[values[k][c] for c in range(1, 11)] for k in keys], dtype=np.float64)
    if not np.isfinite(matrix).all() or not ((matrix >= 0) & (matrix <= 1)).all():
        raise ValueError('Invalid prediction probabilities')
    return matrix


def main():
    paths = {'manifest': MANIFEST, 'previous_report': SOURCE / 'report.json', **MEMBERS,
             'audit_source': Path(__file__).resolve()}
    sources_before = {name: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
                      for name, path in paths.items()}
    groups, image_counts = {}, Counter()
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['split'] != 'valid':
            raise ValueError('Only official valid is allowed')
        key = row['subject_id'], row['study_id']
        target = tuple(json.loads(row['targets']))
        if groups.setdefault(key, target) != target:
            raise ValueError('Conflicting within-Study labels')
        image_counts[key] += 1
    keys = sorted(groups)
    targets = np.asarray([groups[k] for k in keys], dtype=np.int64)
    patients = np.asarray([k[0] for k in keys])
    frequencies = Counter(patients)
    single = np.asarray([image_counts[k] == 1 for k in keys])
    if (len(rows), len(keys), len(frequencies), int(single.sum())) != (1422, 859, 228, 391):
        raise ValueError('Unexpected coverage')
    predictions = {name: read_predictions(path, keys) for name, path in MEMBERS.items()}
    members = {}
    slices = {'all': np.ones(len(keys), dtype=bool), 'single_image_study': single,
              'multiple_image_study': ~single}
    for name, probabilities in predictions.items():
        member = {}
        for slice_name, mask in slices.items():
            local_counts = Counter(patients[mask])
            weights = np.asarray([1. / local_counts[p] for p in patients[mask]], dtype=np.float64)
            member[slice_name] = {
                'studies': int(mask.sum()), 'patients': len(local_counts),
                'positive_studies_per_class': targets[mask].sum(0).tolist(),
                'study_equal': pair_auc(targets[mask], probabilities[mask], np.ones(mask.sum())),
                'patient_equal': pair_auc(targets[mask], probabilities[mask], weights),
            }
        members[name] = member
    historical = json.loads((SOURCE / 'report.json').read_text())
    expected = {'preserved_sprint': historical['baseline'],
                'previous_three_member': historical['candidate'],
                'online_reference_041': historical['submission_best_041'],
                **{('sprint_base768' if k == 'base768' else k): v
                   for k, v in historical['member_scores_diagnostic_only'].items()}}
    max_difference = 0.
    for name, original in expected.items():
        measured = members[name]['all']['study_equal']
        max_difference = max(max_difference, abs(measured['macro_auc'] - original['macro_auc']),
                             float(np.max(np.abs(np.asarray(measured['per_class_auc']) - original['per_class_auc']))))
    if max_difference > 1e-12:
        raise ValueError('Independent pair-count AUC differs from recorded metrics')
    positive, negative = targets[:, 7] == 1, targets[:, 7] == 0
    base = predictions['preserved_sprint']
    reference_pairs = base[positive, 7, None] - base[negative, 7]
    diversity = {}
    for name in ('context512', 'opacity512', 'online_reference_041'):
        probabilities = predictions[name]
        differences = probabilities[positive, 7, None] - probabilities[negative, 7]
        diversity[name] = {
            'pearson_probability_correlation_vs_sprint': float(np.corrcoef(base[:, 7], probabilities[:, 7])[0, 1]),
            'fraction_all_pairs_sprint_nonwinning_member_winning': float(((reference_pairs <= 0) & (differences > 0)).mean()),
            'fraction_all_pairs_sprint_winning_member_nonwinning': float(((reference_pairs > 0) & (differences <= 0)).mean()),
            'positive_probability_median': float(np.median(probabilities[positive, 7])),
            'negative_probability_median': float(np.median(probabilities[negative, 7])),
            'pair_count': int(positive.sum() * negative.sum()),
            'interpretation': 'Existing-member ordering diagnostic; ties grouped with nonwinning. Not a blended-model gain estimate.',
        }
    after = {name: digest(path) for name, path in paths.items()}
    if any(after[name] != item['sha256'] for name, item in sources_before.items()):
        raise ValueError('Source changed during audit')
    report = {
        'created_at': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        'scope': 'Existing saved official-valid predictions only. No new fusion, training, model forward, test read or parameter fitting.',
        'sources': sources_before, 'sources_unchanged': True,
        'patient_equal_definition': 'Within each slice, each Study weight is inverse number of Studies of its patient in that slice. Labels and predictions are not pooled across Studies.',
        'coverage': {'images': len(rows), 'studies': len(keys), 'patients': len(frequencies),
                     'single_image_studies': int(single.sum()), 'multiple_image_studies': int((~single).sum()),
                     'largest_patient_study_count': max(frequencies.values()),
                     'top10_patients_studies': sum(v for _, v in frequencies.most_common(10)),
                     'top10_patients_study_share': sum(v for _, v in frequencies.most_common(10))/len(keys)},
        'members': members, 'class8_diversity_vs_preserved_sprint': diversity,
        'independent_auc_max_abs_difference_vs_previous_report': max_difference,
        'new_fusions_evaluated': 0,
        'limitations': [
            'Official valid was repeatedly used historically; patient equal weighting is sensitivity analysis, not independent validation.',
            'Different patient weighting defines a diagnostic metric and does not replace official Study Macro-AUC.',
            'Correlation and pair recovery identify possible complementarity, not guaranteed ensemble or online improvement.',
        ],
    }
    (OUTPUT / 'existing_members_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Existing-member audit before the new fixed fusion', '',
             f"Recorded {report['created_at']} (Hong Kong). CPU only; no new fusion evaluated.", '',
             'All six saved prediction sets reproduce the previous report within 1e-12 using independent positive-negative pair counting. Input hashes remained unchanged.', '',
             '| Existing member | Study Macro | Patient-equal Macro | Study class 8 | Patient-equal class 8 | Single-image Macro |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, member in members.items():
        whole = member['all']; single_score = member['single_image_study']['study_equal']['macro_auc']
        lines.append(f"| {name} | {whole['study_equal']['macro_auc']:.6f} | {whole['patient_equal']['macro_auc']:.6f} | {whole['study_equal']['per_class_auc'][7]:.6f} | {whole['patient_equal']['per_class_auc'][7]:.6f} | {single_score:.6f} |")
    lines.extend(['',
        'Opacity512 has lower ordinary Macro than Sprint, but higher patient-equal class 8 (.772457 versus .765057). Within single-image Studies, its patient-equal class 8 is .722222 versus .705341 and class 4 is .794118 versus .744289. This suggests complementary information but does not prove a blended gain.', '',
        'Opacity class-8 probability correlation with Sprint is .923233 (Context .937081; 041 .959446). It wins 5.2735% of all class-8 positive-negative pairs that Sprint does not win, while losing 5.9661% that Sprint wins. These are individual-member diagnostics, not candidate-fusion performance.', '',
        'The former three-member ensemble removed the original Tiny768; its losses in single-image classes 2/4/6 support preserving that member. Context512 has especially weak single-image class 4 (.653764).', '',
        'The ten patients with the most Studies contribute 295/859 Studies; the largest has 89. Patient-equal metrics give each patient equal total weight within each slice, retaining all labels. They are sensitivity analyses on the same repeatedly used validation set, not an unseen-patient confirmation.', '',
        'The recommendation fixed before new-result access is to preserve Sprint and add existing Opacity512 at 20% for classes 2–10, keeping class 1 exactly unchanged. No alternative weight or class-specific mixture was evaluated here.', '',
        'Recommended conservative replacement gates, frozen before candidate evaluation: strictly higher ordinary Macro; strictly higher class-8 AUC; no lower single-image Macro; no lower patient-equal Macro (numerical tolerance 1e-12 only). Report every class and both image-count slices. Any optional class-2/4/6 veto must also be frozen before result access; arbitrary -.0005/-.002 tolerances have no independently established practical meaning.', '',
        'See existing_members_audit.json for full classwise/slice metrics, prediction-source paths and SHA256 values. audit_existing_members.py reproduces the audit without model inference or test data.', ''])
    (OUTPUT / 'EXISTING_MEMBER_AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(OUTPUT / 'existing_members_audit.json'),
                      'markdown': str(OUTPUT / 'EXISTING_MEMBER_AUDIT.md'),
                      'sources_unchanged': True, 'max_auc_difference': max_difference,
                      'new_fusions_evaluated': 0}))


if __name__ == '__main__':
    main()
