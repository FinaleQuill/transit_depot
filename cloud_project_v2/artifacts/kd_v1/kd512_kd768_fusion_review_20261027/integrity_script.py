"""Read-only CPU audit of saved 027/029 validation and fixed 030 fusion.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1 in cxr-kd-v1.
Only this directory's integrity.json is written. No model construction/forward.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'integrity.json'
VALID = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
PLAN = ROOT / 'docs/KD512_KD768_EQUAL_PLAN_20261027.json'
DELIVERY = ROOT.parent / 'deliveries'
COMPARISON = DELIVERY / '030_kd512_kd768_equal_diagnostic/comparison_report.json'
NAMES = ['kd512_fp16_hflip2', 'kd768_fp16_hflip2']
DIRS = [DELIVERY / x for x in ['027_tiny512_kd_seed43', '029_tiny768_kd_seed43']]
SOURCE_E8 = '46e06022d963b58d262eaa99ba376de40c3ef182f50a9619f0d09dcf2a0870b7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def hashes(paths):
    return {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}


assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
input_paths = sorted([p for root in DIRS for p in root.rglob('*') if p.is_file()] +
                     [COMPARISON, VALID, PLAN, ROOT / 'data/clean_v1_reviewed/train_clean.csv',
                      ROOT / 'data/cv5_overnight/folds.json', ROOT / 'validate_student.py',
                      ROOT / 'compare_student_candidates.py', ROOT / 'deploy_kd.py'])
before = hashes(input_paths)
manifest_rows = list(csv.DictReader(VALID.open(encoding='utf-8-sig', newline='')))
assert len(manifest_rows) == 1422
for field in ['image_id', 'dicom_id', 'relative_path']:
    assert len({row[field] for row in manifest_rows}) == 1422, field
labels, image_counts = {}, Counter()
for row in manifest_rows:
    assert row['split'] == 'valid'
    key = (row['subject_id'], row['study_id'])
    assert row['study_key'] == '/'.join(key)
    target = tuple(json.loads(row['targets']))
    assert len(target) == 10 and set(target) <= {0, 1}
    assert sorted(json.loads(row['label_ids'])) == [i + 1 for i, flag in enumerate(target) if flag]
    assert key not in labels or labels[key] == target
    labels[key] = target
    image_counts[key] += 1
keys = sorted(labels)
assert len(keys) == 859 and len({k[0] for k in keys}) == 228
targets = np.asarray([labels[k] for k in keys], dtype=np.int8)
assert np.all((targets.sum(axis=0) > 0) & (targets.sum(axis=0) < 859))
comparison = read(COMPARISON)
assert read(PLAN) == {'equal_mean': [0.5, 0.5]}
assert comparison['manifest_sha256'] == sha(VALID)
assert comparison['reference'] == NAMES[0]
assert comparison['num_studies'] == 859 and comparison['num_patients'] == 228
assert comparison['weights_fitted'] is False and comparison['thresholds_fitted'] is False
assert comparison['deployment_selected'] is False and comparison['T4_latency_certified'] is False
assert comparison['fixed_probability_weights'] == {'equal_mean': [[0.5] * 10, [0.5] * 10]}
assert set(comparison['candidates']) == set(NAMES + ['equal_mean'])
boot = comparison['bootstrap']
assert {k: boot[k] for k in ['reference', 'samples', 'seed', 'sampling_unit', 'paired']} == {
    'reference': NAMES[0], 'samples': 1000, 'seed': 20261027,
    'sampling_unit': 'patient_with_all_Studies', 'paired': True}

candidates, validation_results = {}, {}
for i, (name, directory) in enumerate(zip(NAMES, DIRS)):
    deployment = directory / 'fp16_hflip2'
    validation = directory / 'valid_fp16_hflip2'
    cfg = read(deployment / 'config.json')
    export = read(deployment / 'export_report.json')
    report = read(validation / 'validation_report.json')
    assert cfg['precision'] == 'float16' and cfg['tta'] == 'hflip2'
    assert cfg['image_size'] == [512, 768][i] and cfg['ensemble'] is False
    assert len(cfg['members']) == 1 and cfg['weights_source'] == 'ema'
    member = cfg['members'][0]
    assert member['seed'] == 43 and member['tta'] == 'hflip2'
    assert member['image_size'] == [512, 768][i]
    assert member['weights_sha256'] == sha(deployment / member['weights_file'])
    assert cfg['official_valid_manifest_sha256'] == member['official_valid_manifest_sha256'] == sha(VALID)
    assert cfg['source_manifest_sha256'] == member['source_manifest_sha256'] == sha(ROOT / 'data/clean_v1_reviewed/train_clean.csv')
    assert cfg['fold_plan_sha256'] == member['fold_plan_sha256'] == sha(ROOT / 'data/cv5_overnight/folds.json')
    assert report['complete'] is True and report['num_images'] == 1422 and report['num_studies'] == 859
    assert report['precision'] == 'float16' and report['batch_size'] == 1 and report['member_count'] == 1
    assert report['thresholds_fitted'] is False and report['checkpoint_selection_performed'] is False
    assert report['manifest_sha256'] == sha(VALID)
    assert report['deployment_config_sha256'] == sha(deployment / 'config.json')
    assert report['predictions_sha256'] == sha(validation / 'valid_predictions.csv')
    assert report['members'][0]['weights_sha256'] == member['weights_sha256']
    assert report['members'][0]['source_checkpoint_sha256'] == member['source_checkpoint_sha256'] == export['source_checkpoint_sha256']
    if i == 1:
        assert member['source_checkpoint_sha256'] == SOURCE_E8
        prior = read(ROOT / 'artifacts/kd_v1/tiny768_review_20261027/integrity.json')
        assert prior['original_run_hashes'][str(ROOT.parent / 'runs/029_tiny768_kd_seed43/best.pt')] == SOURCE_E8
    inventory = []
    for item in export['files']:
        path = deployment / item['path']
        actual = {'path': item['path'], 'bytes': path.stat().st_size, 'sha256': sha(path)}
        assert actual == item
        inventory.append(actual)
    assert report['deployment_inventory']['files'] == inventory
    assert sum(item['bytes'] for item in inventory) == export['total_bytes'] == report['deployment_inventory']['total_bytes']
    assert export['size_pass'] is True and export['total_bytes'] <= 500_000_000
    records = list(csv.DictReader((validation / 'valid_predictions.csv').open(encoding='utf-8-sig', newline='')))
    assert len(records) == 8590
    assert list(records[0]) == ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
    predictions = {}
    for row in records:
        key = (row['Subject_id'], row['Study_id'])
        category = int(row['Predict_class'])
        assert str(category) == row['Predict_class'] and 1 <= category <= 10
        pair = (key, category)
        assert pair not in predictions
        value = float(row['Probability'])
        assert np.isfinite(value) and 0 <= value <= 1
        predictions[pair] = value
    assert set(predictions) == {(key, c) for key in keys for c in range(1, 11)}
    values = np.asarray([[predictions[key, c] for c in range(1, 11)] for key in keys], dtype=np.float64)
    aucs = [float(roc_auc_score(targets[:, c], values[:, c])) for c in range(10)]
    macro = float(np.mean(aucs))
    f1 = float(f1_score(targets, values >= 0.5, average='macro', zero_division=0))
    close(aucs, report['per_class_auc']); close(macro, report['macro_auc']); close(f1, report['macro_f1_fixed_0_5'])
    close(report['auc_difference_from_source'], macro - cfg['source_validation']['macro_auc'])
    provenance = comparison['source_validations'][i]
    assert provenance['name'] == name and provenance['validation_dir'] == str(validation)
    assert provenance['predictions_sha256'] == sha(validation / 'valid_predictions.csv')
    assert provenance['report_sha256'] == sha(validation / 'validation_report.json')
    assert provenance['probability_report_binding'] == 'sha256'
    candidates[name] = values
    validation_results[name] = {'rows': len(records), 'studies': len(keys), 'patients': 228,
        'duplicate_study_class_pairs': 0, 'missing_or_extra_study_class_pairs': 0,
        'all_probabilities_finite_in_unit_interval': True, 'probability_min': float(values.min()),
        'probability_max': float(values.max()), 'macro_auc': macro, 'per_class_auc': aucs,
        'macro_f1_fixed_0_5': f1, 'source_training_macro_auc': cfg['source_validation']['macro_auc'],
        'auc_difference_from_source': report['auc_difference_from_source'],
        'actual_inference_bytes': export['total_bytes'], 'inventory': inventory,
        'source_checkpoint_sha256': member['source_checkpoint_sha256'],
        'validation_report_sha256': sha(validation / 'validation_report.json'),
        'predictions_sha256': sha(validation / 'valid_predictions.csv')}

state_summaries = {}
for name, directory in zip(NAMES, DIRS):
    state = torch.load(directory / 'fp16_hflip2/student_fp16.pt', map_location='cpu', weights_only=True)
    assert len(state) == 182 and all(torch.is_tensor(t) and t.dtype == torch.float16 and torch.isfinite(t).all().item() for t in state.values())
    state_summaries[name] = {'tensors': len(state), 'numel': sum(t.numel() for t in state.values()),
                           'all_dtype_float16': True, 'all_finite': True, 'load_device': 'cpu',
                           'model_constructed': False, 'forward_calls': 0}
    del state
candidates['equal_mean'] = (candidates[NAMES[0]] + candidates[NAMES[1]]) / 2
recomputed = {}
reference = np.asarray(validation_results[NAMES[0]]['per_class_auc'])
for name, values in candidates.items():
    aucs = np.asarray([roc_auc_score(targets[:, c], values[:, c]) for c in range(10)])
    actual = {'macro_auc': float(aucs.mean()), 'per_class_auc': aucs.tolist(),
              'per_class_delta_from_reference': (aucs - reference).tolist(),
              'macro_auc_delta_from_reference': float(aucs.mean() - reference.mean()), 'slices': {}}
    for label, mask in [('single_image_study', np.asarray([image_counts[k] == 1 for k in keys])),
                        ('multiple_image_study', np.asarray([image_counts[k] > 1 for k in keys]))]:
        sliced = [float(roc_auc_score(targets[mask, c], values[mask, c])) for c in range(10)]
        actual['slices'][label] = {'num_studies': int(mask.sum()), 'per_class_auc': sliced, 'macro_auc': float(np.mean(sliced))}
    saved = comparison['candidates'][name]
    for field in ['macro_auc', 'per_class_auc', 'per_class_delta_from_reference', 'macro_auc_delta_from_reference']:
        close(actual[field], saved[field])
    for label, sliced in actual['slices'].items():
        assert sliced['num_studies'] == saved['slices'][label]['num_studies']
        close(sliced['per_class_auc'], saved['slices'][label]['per_class_auc'])
        close(sliced['macro_auc'], saved['slices'][label]['macro_auc'])
    recomputed[name] = actual
after_paths = sorted([p for root in DIRS for p in root.rglob('*') if p.is_file()] +
                     [p for p in input_paths if not any(p.is_relative_to(root) for root in DIRS)])
after = hashes(after_paths)
assert before == after
result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
    'manifest': {'images': len(manifest_rows), 'studies': len(keys), 'patients': 228,
                 'sha256': sha(VALID), 'positive_studies_per_class': targets.sum(0).tolist(),
                 'single_image_studies': sum(v == 1 for v in image_counts.values()),
                 'multiple_image_studies': sum(v > 1 for v in image_counts.values()),
                 'labels_binary_consistent_within_study_and_label_ids': True,
                 'image_ids_dicom_ids_and_paths_unique': True},
    'validation_results': validation_results, 'fp16_states': state_summaries,
    'comparison_recomputed': recomputed,
    'comparison_report_sha256': sha(COMPARISON), 'equal_plan_sha256': sha(PLAN),
    'fixed_plan': {'equal_mean': [0.5, 0.5]}, 'weights_fitted': False, 'thresholds_fitted': False,
    'bootstrap_metadata_verified': {k: boot[k] for k in ['reference', 'samples', 'seed', 'sampling_unit', 'paired']},
    'bootstrap_not_recomputed_here': True, 'input_file_count': len(before),
    'input_hashes_before': before, 'input_hashes_after': after, 'original_input_files_unchanged': True,
    'no_training': True, 'no_model_construction_or_forward': True, 'no_gpu_work': True, 'test_data_usage': False,
    'limitations': [
        'Saved CSV independently proves 859 Study x 10 class coverage and exact frozen manifest identity/labels; it does not retain per-image outputs or replay image aggregation.',
        '1422 image completion is reported by the validator and protected by its StudyAccumulator.complete guard; no new image inference was run in this audit.',
        '029 source E8 checkpoint SHA was checked against prior completed audit and export/config/report provenance; the 450 MB training checkpoint was not rehashed or loaded again.',
        'CPU fixed probability blend is exploratory; it is not actual composed deployment FP16 validation or official T4 timing certification.',
        'Export/config pending validation hints are immutable export-time metadata; the separately hash-bound completed validation report supplies later validation status.',
        'Training source archives are immutable execution evidence; later equivalent deployment preprocessing is not required to equal the archived training-era deployment code.'
    ]}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': result['status'], 'manifest': result['manifest'],
                  'macro_auc': {n: r['macro_auc'] for n, r in recomputed.items()},
                  'input_files_unchanged': result['input_file_count'], 'output': str(OUT)}, indent=2))
