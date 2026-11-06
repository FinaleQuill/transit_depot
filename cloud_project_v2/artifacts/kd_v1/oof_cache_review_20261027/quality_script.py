"""Read-only CPU diagnostics of a fixed strict-OOF raw-view cache.

Writes only this script's neighboring quality.json. Never builds/loads a model,
selects teachers, fits blending weights or thresholds, or changes source data.
"""
from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone, timedelta

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from dataset import read_manifest
from dataset_overnight import aggregate_study_predictions
from train_overnight import compute_loss_config
from train_distill import distillation_losses


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aucs(y, p):
    values = [float(roc_auc_score(y[:, c], p[:, c])) for c in range(10)]
    return {'macro_auc': float(np.mean(values)), 'per_class_auc': values,
            'studies': len(y), 'positive_studies': y.sum(axis=0).astype(int).tolist()}


def main():
    cache = PROJECT.parent/'runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2'
    manifest = PROJECT/'data/clean_v1_reviewed/train_clean.csv'
    records = read_manifest(manifest, split='train')
    meta = json.loads((cache/'metadata.json').read_text())
    with np.load(cache/'soft_labels.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    assert meta['soft_labels_sha256'] == digest(cache/'soft_labels.npz')
    assert list(arrays['image_ids']) == [r['image_id'] for r in records]
    raw = arrays['logits']
    assert raw.shape == (30328, 2, 10) and raw.dtype == np.float32
    assert np.isfinite(raw).all()
    views = expit(raw.astype(np.float64))
    identity = aggregate_study_predictions(records, arrays['image_ids'], views[:, 0, :])
    both = aggregate_study_predictions(records, arrays['image_ids'], views.mean(axis=1))
    study_fold = {}
    study_views = {}
    for r, fold in zip(records, arrays['fold_indices']):
        assert study_fold.setdefault(r['study_key'], int(fold)) == int(fold)
        study_views[r['study_key']] = study_views.get(r['study_key'], 0) + 1
    sf = np.array([study_fold[k] for k in both['study_keys']])
    weights = np.array([1/len(study_views)/study_views[r['study_key']] for r in records])
    assert np.isclose(weights.sum(), 1)
    labels = np.asarray([r['targets'] for r in records])

    loss = compute_loss_config(records, source_manifest_sha256=digest(manifest), fold=None)
    ws = np.asarray(loss['pos_weight'], dtype=np.float32)
    wt = arrays['teacher_pos_weights']
    aligned = raw - np.log(wt)[:, None, :] + np.log(ws)[None, None, :]
    q = expit(aligned.astype(np.float64)/2).mean(axis=1)
    target_study = aggregate_study_predictions(records, arrays['image_ids'], q)
    # Verify this is the actual production formula, including sigmoid per view
    # before averaging, with a CPU tensor consistency test; not a model forward.
    tensor_q = torch.sigmoid((torch.from_numpy(raw.copy())
        - torch.from_numpy(wt.copy()).log()[:, None, :]
        + torch.from_numpy(ws).log()[None, None, :])/2).mean(dim=1)
    max_q_error = float(np.max(np.abs(tensor_q.numpy()-q)))
    assert max_q_error < 2e-7
    ideal_student = 2*torch.logit(tensor_q)
    _, soft = distillation_losses(ideal_student, torch.as_tensor(labels),
        torch.from_numpy(raw.copy()), torch.from_numpy(wt.copy()),
        torch.from_numpy(ws), temperature=2., align=True)
    assert float(soft.max()) < 2e-6

    quantiles = [0., .001, .01, .5, .99, .999, 1.]
    flat = raw.reshape(-1, 10).astype(np.float64)
    entropy = -(q*np.log2(q)+(1-q)*np.log2(1-q))
    assert np.isfinite(q).all() and ((q > 0) & (q < 1)).all()
    y_study = target_study['targets']
    q_study = target_study['probabilities']
    per_class = []
    for c in range(10):
        per_class.append({'class_id': c+1,
            'positive_studies': int(y_study[:, c].sum()),
            'raw_logits_quantiles': np.quantile(flat[:, c], quantiles).tolist(),
            'raw_logits_std': float(np.std(flat[:, c])),
            'unique_raw_logits': int(len(np.unique(flat[:, c]))),
            'view_logits_mean_absolute_difference': float(np.mean(abs(raw[:, 0, c]-raw[:, 1, c]))),
            'exact_equal_view_logits_fraction': float(np.mean(raw[:, 0, c] == raw[:, 1, c])),
            'raw_sigmoid_view_mean_absolute_difference': float(np.mean(abs(views[:, 0, c]-views[:, 1, c]))),
            'raw_sigmoid_below_1e_6_fraction': float(np.mean(views[:, :, c] < 1e-6)),
            'raw_sigmoid_above_1_minus_1e_6_fraction': float(np.mean(views[:, :, c] > 1-1e-6)),
            'T2_aligned_target_quantiles': np.quantile(q[:, c], quantiles).tolist(),
            'T2_aligned_target_std': float(np.std(q[:, c])),
            'T2_aligned_target_study_equal_mean': float(weights @ q[:, c]),
            'T2_aligned_target_entropy_bits_study_equal_mean': float(weights @ entropy[:, c]),
            'T2_aligned_target_positive_study_mean': float(q_study[y_study[:, c] == 1, c].mean()),
            'T2_aligned_target_negative_study_mean': float(q_study[y_study[:, c] == 0, c].mean()),
            'pooled_raw_hflip2_oof_auc': float(roc_auc_score(y_study[:, c], both['probabilities'][:, c]))})
    fold_results=[]
    for fold in range(5):
        use = sf == fold
        fold_results.append({'fold': fold, 'epoch': next(t['epoch'] for t in meta['teachers'] if t['fold_index']==fold),
            'images': int(np.sum(arrays['fold_indices'] == fold)),
            'patients': len({r['subject_id'] for r, f in zip(records, arrays['fold_indices']) if f == fold}),
            'identity': aucs(identity['targets'][use], identity['probabilities'][use]),
            'hflip2': aucs(both['targets'][use], both['probabilities'][use])})
    report = {'checked_at_hong_kong': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'cache_npz_sha256': digest(cache/'soft_labels.npz'),
        'cache_metadata_sha256': digest(cache/'metadata.json'),
        'train_manifest_sha256': digest(manifest),
        'source_script_sha256': digest(__file__),
        'num_images': len(records), 'num_studies': len(study_views),
        'num_patients': len({r['subject_id'] for r in records}),
        'raw_shape': list(raw.shape), 'stored_dtype': str(raw.dtype),
        'teacher_forward_precision': sorted({t['extraction_precision'] for t in meta['teachers']}),
        'all_finite': bool(np.isfinite(raw).all()),
        'global_raw_logit_min': float(raw.min()), 'global_raw_logit_max': float(raw.max()),
        'quantile_probabilities': quantiles,
        'pooled_identity': aucs(identity['targets'], identity['probabilities']),
        'pooled_hflip2': aucs(both['targets'], both['probabilities']),
        'by_fold': fold_results,
        'student_full_train_pos_weight': ws.tolist(),
        'teacher_to_student_log_weight_offset_max_abs': float(np.max(abs(np.log(ws)[None,:]-np.log(wt)))),
        'effective_soft_target': {'formula': 'mean_views(sigmoid((raw_logits - log(teacher_train_pos_weight) + log(student_full_train_pos_weight))/2))',
            'temperature': 2., 'perview_before_mean': True,
            'native_torch_target_max_absolute_difference': max_q_error,
            'native_KL_when_student_matches_targets_max': float(soft.max()),
            'all_finite': bool(np.isfinite(q).all()), 'min': float(q.min()), 'max': float(q.max()),
            'all_classes_nonconstant': bool((np.std(q,axis=0) > 1e-3).all()),
            'statistics_scope': 'quantiles/std are image-level; target means and mean image entropy use equal Study weighting; positive/negative means are Study-mean targets'},
        'per_class': per_class,
        'recommendation': 'No pathological cache/target distribution found; no repeat extraction or quality-driven hyperparameter change justified. Proceed with the predeclared KD512 versus BCE512 comparison after integrity audit.',
        'limits': ['OOF is not the untouched official-valid split. Different fold teachers, training sample amount and calibration prevent causal comparison with official-valid student/ensemble AUC.',
                   'Per-fold AUC here is post-training descriptive diagnosis, not selection of epochs, teachers or class weights. Do not use it to rewrite the fixed teacher protocol.',
                   'Pos-weight-aligned T2 targets are softened probabilities in the weighted-BCE student logit space, not estimates of class prevalence or calibrated medical probabilities.',
                   'Good OOF ranking or benign soft targets do not guarantee that alpha0.2 distillation improves the student.'],
        'models_loaded': False, 'gpu_used': False, 'training_started': False,
        'teacher_or_target_parameters_fitted': False, 'test_used': False}
    Path(__file__).with_name('quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'pooled_hflip2_auc': report['pooled_hflip2']['macro_auc'],
        'fold_hflip2_auc': [r['hflip2']['macro_auc'] for r in fold_results],
        'effective_target_minmax': [float(q.min()),float(q.max())],
        'raw_logit_minmax': [float(raw.min()),float(raw.max())],
        'per_class_oof_auc': report['pooled_hflip2']['per_class_auc'],
        'per_class_target_entropy_bits': [r['T2_aligned_target_entropy_bits_study_equal_mean'] for r in per_class],
        'native_target_maxerror': max_q_error, 'native_matchingKLmax': float(soft.max())}, indent=2))


if __name__ == '__main__':
    main()
