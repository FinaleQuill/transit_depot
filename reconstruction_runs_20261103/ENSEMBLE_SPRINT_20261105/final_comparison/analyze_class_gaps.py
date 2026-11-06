"""CPU analysis of saved predictions; no model, image or test data access."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
SAMPLES = 10000
SEED = 20261105
report = json.loads((ROOT / 'report.json').read_text())
manifest = Path(json.loads(Path('/home/featurize/work/cxr_research_20261103/ensemble_sprint/plan.json').read_text())['valid_manifest'])
predictions = ROOT / 'candidate_predictions.csv'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report['manifest_sha256']
assert hashlib.sha256(predictions.read_bytes()).hexdigest() == report['candidate_predictions_sha256']

labels = {}
with manifest.open(encoding='utf-8-sig', newline='') as stream:
    for row in csv.DictReader(stream):
        key = (row['subject_id'], row['study_id'])
        target = json.loads(row['targets'])
        assert row['split'] == 'valid'
        if key in labels:
            assert labels[key] == target
        labels[key] = target
values = {}
with predictions.open(newline='') as stream:
    for row in csv.DictReader(stream):
        key = (row['Subject_id'], row['Study_id'])
        category = int(row['Predict_class']) - 1
        values.setdefault(key, {})
        assert category not in values[key]
        values[key][category] = float(row['Probability'])
assert set(labels) == set(values)
keys = sorted(labels)
y = np.array([labels[key] for key in keys], dtype=np.int8)
p = np.array([[values[key][c] for c in range(10)] for key in keys])
patients, patient_index = np.unique([key[0] for key in keys], return_inverse=True)
assert len(keys) == 859 and len(patients) == 228

prepared = []
for c in range(10):
    order = np.argsort(p[:, c], kind='stable')
    starts = np.flatnonzero(np.r_[True, np.diff(p[order, c]) != 0])
    prepared.append((order, starts, y[order, c]))


def weighted_auc(study_counts, c):
    order, starts, target = prepared[c]
    weights = study_counts[:, order]
    positive = np.add.reduceat(weights * target, starts, axis=1)
    negative = np.add.reduceat(weights * (1-target), starts, axis=1)
    numerator = (positive * (negative.cumsum(axis=1) - .5 * negative)).sum(axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


point = np.array([weighted_auc(np.ones((1, len(keys))), c)[0] for c in range(10)])
assert np.allclose(point, report['candidate']['per_class_auc'], rtol=0, atol=1e-12)
rng = np.random.default_rng(SEED)
bootstrap = []
for start in range(0, SAMPLES, 250):
    n = min(250, SAMPLES-start)
    counts = rng.multinomial(len(patients), np.full(len(patients), 1/len(patients)), size=n)
    study_counts = counts[:, patient_index].astype(float)
    scores = np.column_stack([weighted_auc(study_counts, c) for c in range(10)])
    if start == 0:
        # Check tied weighted ranking against the direct positive/negative formula.
        for c in (0, 7):
            positive, negative = y[:, c] == 1, y[:, c] == 0
            comparison = (p[positive, c, None] > p[None, negative, c]).astype(float)
            comparison += .5 * (p[positive, c, None] == p[None, negative, c])
            a, b = study_counts[0, positive], study_counts[0, negative]
            direct = np.sum(comparison * a[:, None] * b[None, :]) / (a.sum() * b.sum())
            assert abs(direct - scores[0, c]) < 1e-12
    bootstrap.append(scores)
bootstrap = np.concatenate(bootstrap)
valid = np.isfinite(bootstrap).all(axis=1)
bootstrap = bootstrap[valid]
assert len(bootstrap) >= .99 * SAMPLES
gaps = bootstrap - bootstrap.mean(axis=1, keepdims=True)
point_gap = point-point.mean()
ordinary = np.quantile(gaps, [.025, .975], axis=0)
# Ten two-sided comparisons: individual 99.5% percentile intervals provide
# approximate Bonferroni familywise 95% coverage (subject to bootstrap assumptions).
corrected = np.quantile(gaps, [.0025, .9975], axis=0)
absolute = np.quantile(bootstrap, [.025, .975], axis=0)
result = {
    'created_at_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
    'method': 'Patient-cluster nonparametric bootstrap; resample 228 whole patients with replacement, retain every Study; compare each class AUC against the same resample ten-class Macro-AUC.',
    'samples_requested': SAMPLES, 'samples_valid': len(bootstrap), 'seed': SEED,
    'reference': 'Competition ten-class Macro-AUC, including class 10 No Finding',
    'macro_auc': float(point.mean()), 'disease_only_classes_1_to_9_mean': float(point[:9].mean()),
    'macro_auc_95_interval': np.quantile(bootstrap.mean(axis=1), [.025,.975]).tolist(),
    'multiple_comparison_adjustment': 'Ten two-sided comparisons; Bonferroni per-class alpha=.005, percentile interval [.0025,.9975]. Approximate familywise 95% intervals.',
    'limitations': 'Post-training exploratory analysis on a repeatedly used local validation set. Patient resampling addresses within-patient dependence, not validation-selection bias, external distribution shift or label uncertainty. No causal explanation or online improvement is established.',
    'new_model_inference_runs': 0, 'model_or_fusion_changed': False,
    'test_images_read': False, 'submission_generated': False,
    'source_prediction_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
    'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'classes': [],
}
for c in range(10):
    result['classes'].append({
        'class_id': c+1, 'auc': float(point[c]), 'auc_95_interval': absolute[:,c].tolist(),
        'difference_from_macro': float(point_gap[c]),
        'gap_95_interval': ordinary[:,c].tolist(),
        'gap_bonferroni_99_5_interval': corrected[:,c].tolist(),
        'below_macro_at_uncorrected_95': bool(ordinary[1,c] < 0),
        'below_macro_after_10_class_correction': bool(corrected[1,c] < 0),
        'positive_studies': int(y[:,c].sum()),
        'positive_patients': len(set(patient_index[y[:,c] == 1].tolist())),
    })
(ROOT / 'class_gap_analysis_20261105.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
