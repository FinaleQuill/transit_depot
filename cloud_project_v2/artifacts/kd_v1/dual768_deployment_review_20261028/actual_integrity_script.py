"""CPU-only independent audit of completed 033 deployment/valid/4090 artifacts.

Run only after the root agent confirms all three production steps have ended:
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  /home/featurize/.conda/envs/cxr-kd-v1/bin/python /absolute/path/to/this_script.py
Writes actual_integrity.json beside this script. Never constructs a model,
calls CUDA, starts inference/training, waits for artifacts, or reads test data.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('actual_integrity.json')
DELIVERIES = ROOT.parent / 'deliveries'
ACTUAL = DELIVERIES / '033_dual_tiny768_seed43_25_seed44_75'
MODEL = ACTUAL / 'fp16_hflip2'
VALIDATION = ACTUAL / 'valid_fp16_hflip2'
BENCH = ACTUAL / 'latency4090'
SOURCE_NAMES = ['029_tiny768_kd_seed43', '030_tiny768_kd_seed44']
SOURCES = [DELIVERIES / name for name in SOURCE_NAMES]
RUNS = [ROOT.parent / 'runs' / name for name in SOURCE_NAMES]
MANIFEST = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
TRAIN = ROOT / 'data/clean_v1_reviewed/train_clean.csv'
FOLDS = ROOT / 'data/cv5_overnight/folds.json'
COMPARISON = DELIVERIES / '032_dual_tiny768_fixed_weights_diagnostic/comparison_report.json'
SELECTION = DELIVERIES / '031_resource_stress_20261027/inputs/selection.json'
INPUTS = SELECTION.parent
EXPECTED_AUC = 0.8083844422198005
CSV_HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hashes(paths):
    return {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(set(paths))}


def close(actual, expected, atol=1e-14):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def metric(labels, values):
    per_class = [float(roc_auc_score(labels[:, c], values[:, c]))
                 if len(np.unique(labels[:, c])) == 2 else None for c in range(10)]
    return {'macro_auc': float(np.mean([v for v in per_class if v is not None])),
            'per_class_auc': per_class,
            'macro_f1_fixed_0_5': float(f1_score(labels, values >= .5, average='macro', zero_division=0)),
            'per_class_f1_fixed_0_5': f1_score(labels, values >= .5, average=None, zero_division=0).tolist()}


def probabilities(path, expected_keys, image=False):
    data = rows(path)
    assert len(data) == len(expected_keys) * 10, (path, len(data))
    assert list(data[0]) == (['Image_id'] + CSV_HEADER if image else CSV_HEADER)
    parsed = {}
    for row in data:
        key = (row['Subject_id'], row['Study_id'])
        if image:
            image_id = row['Image_id']
            assert Path(image_id).parts[:2] == key
            key = image_id
        category = int(row['Predict_class'])
        assert str(category) == row['Predict_class'] and 1 <= category <= 10
        value = float(row['Probability'])
        assert np.isfinite(value) and 0 <= value <= 1
        assert (key, category) not in parsed
        parsed[key, category] = value
    assert set(parsed) == {(key, c) for key in expected_keys for c in range(1, 11)}
    return np.asarray([[parsed[key, c] for c in range(1, 11)] for key in expected_keys], dtype=np.float64)


def inventory(directory, config):
    names = [member['weights_file'] for member in config['members']] + ['config.json']
    assert config['inference_files'] == names and len(names) == len(set(names))
    actual = []
    for name in names:
        path = directory / name
        assert Path(name).name == name and path.is_file() and not path.is_symlink()
        actual.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    total = sum(item['bytes'] for item in actual)
    assert total <= 500_000_000
    return {'files': actual, 'total_bytes': total, 'limit_bytes': 500_000_000, 'size_pass': True}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly hide CUDA for this CPU-only audit'
    torch.set_num_threads(1)
    # Protect both original source artifacts and the completed production artifacts.
    protected = [p for base in SOURCES + [ACTUAL] for p in base.rglob('*') if p.is_file()]
    protected += [run / filename for run in RUNS for filename in
                  ['best.pt', 'best_metrics.json', 'history.jsonl', 'status.json', 'config.json', 'checkpoint_audit.jsonl']]
    protected += [MANIFEST, TRAIN, FOLDS, COMPARISON, SELECTION,
                  ROOT / 'docs/DUAL_TINY768_SELECTION_20261028.json']
    protected += [ROOT / name for name in ['compose_students.py', 'validate_student.py', 'benchmark_local_resources.py',
                                          'predict.py', 'cost.py', 'deploy_kd.py']]
    # Missing incomplete outputs fail immediately; this script never polls/waits.
    assert (BENCH / 'local_resource_report.json').is_file()
    assert (VALIDATION / 'validation_report.json').is_file()
    selection = read(SELECTION)
    assert selection['origin_split'] == 'valid' and selection['test_used'] is False
    protected += [INPUTS / item['output_relative_path']
                  for group in ['measured', 'warmup'] for item in selection[group]]
    before = hashes(protected)

    manifest_rows = rows(MANIFEST)
    assert len(manifest_rows) == 1422
    for field in ['image_id', 'dicom_id', 'relative_path']:
        assert len({r[field] for r in manifest_rows}) == 1422
    labels, counts, by_relative = {}, Counter(), {}
    for row in manifest_rows:
        key = (row['subject_id'], row['study_id'])
        assert row['split'] == 'valid' and row['study_key'] == '/'.join(key)
        target = tuple(json.loads(row['targets']))
        assert len(target) == 10 and set(target) <= {0, 1}
        assert sorted(json.loads(row['label_ids'])) == [c + 1 for c, flag in enumerate(target) if flag]
        assert key not in labels or labels[key] == target
        labels[key] = target
        counts[key] += 1
        by_relative[row['relative_path']] = row
    keys = sorted(labels)
    assert len(keys) == 859 and len({key[0] for key in keys}) == 228
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    train_rows = rows(TRAIN)
    assert not ({r['subject_id'] for r in train_rows} & {key[0] for key in keys})

    cfg = read(MODEL / 'config.json')
    composed = read(MODEL / 'composition_report.json')
    validation = read(VALIDATION / 'validation_report.json')
    benchmark = read(BENCH / 'local_resource_report.json')
    config_sha = sha(MODEL / 'config.json')
    actual_inventory = inventory(MODEL, cfg)
    assert cfg['schema_version'] == 2 and cfg['precision'] == 'float16'
    assert cfg['tta'] == 'hflip2' and cfg['image_size'] == 768 and cfg['image_sizes'] == [768]
    assert cfg['ensemble'] is True and cfg['batch_size'] == 1 and cfg['class_ids'] == list(range(1, 11))
    assert cfg['gates'] is None and cfg['weights_source'] == 'ema'
    assert cfg['aggregation'] == 'study_mean_probability'
    assert cfg['member_aggregation'] == 'fixed_weighted_mean_probability'
    assert cfg['tta_aggregation'] == 'mean_sigmoid_probability'
    assert cfg['model_weights'] == [[.25] * 10, [.75] * 10]
    assert len(cfg['members']) == 2
    for field, path in [('official_valid_manifest_sha256', MANIFEST), ('source_manifest_sha256', TRAIN), ('fold_plan_sha256', FOLDS)]:
        assert cfg[field] == sha(path)
    for field, value in actual_inventory.items():
        assert composed[field] == value
    assert composed['execute'] is True and composed['member_count'] == 2
    assert composed['export_directory'] == str(MODEL)

    source_summaries, source_probabilities, fp16 = [], [], []
    for index, (source, run, member) in enumerate(zip(SOURCES, RUNS, cfg['members'])):
        model = source / 'fp16_hflip2'
        original_config = read(model / 'config.json')
        original_member = original_config['members'][0]
        expected = dict(original_member, weights_file=f'student_{index:02d}_fp16.pt', tta='hflip2',
                        source_export_config_sha256=sha(model / 'config.json'))
        assert member == expected, 'All member fields must preserve source export provenance'
        assert member['seed'] == [43, 44][index] and member['weights_source'] == 'ema'
        assert member['image_size'] == 768 and member['tta'] == 'hflip2'
        checkpoint_sha = before[str(run / 'best.pt')]['sha256']
        assert member['source_checkpoint_sha256'] == checkpoint_sha
        assert member['weights_sha256'] == sha(MODEL / member['weights_file']) == sha(model / original_member['weights_file'])
        best = read(run / 'best_metrics.json')
        status = read(run / 'status.json')
        native_config = read(run / 'config.json')
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
        e8 = next(item for item in history if item['epoch'] == 8)
        audit = [json.loads(line) for line in (run / 'checkpoint_audit.jsonl').read_text().splitlines()]
        assert best['epoch'] == status['best_epoch'] == 8
        assert status['state'] == 'early_stopped' and status['best_checkpoint_sha256'] == checkpoint_sha
        assert native_config['seed'] == member['seed'] and native_config['image_size'] == 768
        assert native_config['validation_tta'] == 'hflip2'
        native_validation = {k: v for k, v in best.items() if k != 'epoch'}
        assert member['source_validation'] == native_validation == e8['validation']
        assert e8['best_checkpoint_sha256'] == checkpoint_sha
        assert any(item['epoch'] == 8 and item['role'] == 'best' and item['checkpoint_sha256'] == checkpoint_sha for item in audit)
        declared_source = composed['sources'][index]
        assert declared_source == {'model_dir': str(model), 'source_config_sha256': sha(model / 'config.json'),
                                  'source_weights': str(model / original_member['weights_file']),
                                  'target_weights': member['weights_file'], 'weights_sha256': member['weights_sha256'],
                                  'bytes': (MODEL / member['weights_file']).stat().st_size}
        original_report = read(source / 'valid_fp16_hflip2/validation_report.json')
        original_csv = source / 'valid_fp16_hflip2/valid_predictions.csv'
        assert original_report['deployment_config_sha256'] == sha(model / 'config.json')
        assert original_report['predictions_sha256'] == sha(original_csv)
        values = probabilities(original_csv, keys)
        recomputed = metric(targets, values)
        for field in ['macro_auc', 'per_class_auc', 'macro_f1_fixed_0_5']:
            close(recomputed[field], original_report[field])
        source_probabilities.append(values)
        state = torch.load(MODEL / member['weights_file'], map_location='cpu', weights_only=True)
        assert len(state) == 182
        assert all(torch.is_tensor(t) and t.device.type == 'cpu' and t.dtype == torch.float16 and bool(torch.isfinite(t).all()) for t in state.values())
        fp16.append({'seed': member['seed'], 'tensors': len(state), 'numel': sum(t.numel() for t in state.values()),
                     'all_float16_finite_on_cpu': True, 'model_constructed': False, 'forward_calls': 0})
        del state
        source_summaries.append({'seed': member['seed'], 'epoch': 8, 'source_checkpoint_sha256': checkpoint_sha,
                                 'source_export_config_sha256': sha(model / 'config.json'),
                                 'weights_sha256': member['weights_sha256'], 'validation': recomputed})

    for report in [validation, benchmark]:
        assert report['deployment_config_sha256'] == config_sha
    assert validation['deployment_inventory'] == benchmark['inference_files'] == actual_inventory
    assert validation['complete'] is True and validation['num_images'] == 1422 and validation['num_studies'] == 859
    assert validation['precision'] == 'float16' and validation['batch_size'] == 1 and validation['member_count'] == 2
    assert validation['tta'] == 'hflip2' and validation['image_sizes'] == [768] and validation['ensemble'] is True
    assert validation['thresholds_fitted'] is False and validation['checkpoint_selection_performed'] is False
    assert validation['manifest_sha256'] == sha(MANIFEST)
    assert validation['predictions_sha256'] == sha(VALIDATION / 'valid_predictions.csv')
    for actual_member, declared in zip(cfg['members'], validation['members']):
        assert declared == {field: actual_member.get(field) for field in
                            ['source_checkpoint_sha256', 'weights_sha256', 'image_size', 'tta', 'seed']}
    assert cfg['source_validation'] == validation['source_training_validation'] == cfg['members'][0]['source_validation']
    actual = probabilities(VALIDATION / 'valid_predictions.csv', keys)
    actual_metrics = metric(targets, actual)
    for field in ['macro_auc', 'per_class_auc', 'macro_f1_fixed_0_5']:
        close(actual_metrics[field], validation[field])
    close(validation['auc_difference_from_source'], actual_metrics['macro_auc'] - cfg['members'][0]['source_validation']['macro_auc'])
    mathematical = .25 * source_probabilities[0] + .75 * source_probabilities[1]
    mathematical_metrics = metric(targets, mathematical)
    comparison = read(COMPARISON)
    saved = comparison['candidates']['seed43_25_seed44_75']
    assert comparison['fixed_probability_weights']['seed43_25_seed44_75'] == [[.25] * 10, [.75] * 10]
    close(mathematical_metrics['macro_auc'], EXPECTED_AUC)
    close(mathematical_metrics['macro_auc'], saved['macro_auc'])
    close(mathematical_metrics['per_class_auc'], saved['per_class_auc'])
    delta = np.abs(actual - mathematical)
    # Member blending and Study averaging commute mathematically; FP32 addition
    # order may differ by several ulps between actual and saved-CSV arithmetic.
    assert delta.max() <= 2e-7, float(delta.max())
    slices = {}
    for name, mask in [('single_image_study', np.asarray([counts[key] == 1 for key in keys])),
                       ('multiple_image_study', np.asarray([counts[key] > 1 for key in keys]))]:
        slices[name] = {'num_studies': int(mask.sum()), 'actual': metric(targets[mask], actual[mask]),
                        'mathematical': metric(targets[mask], mathematical[mask])}
        assert saved['slices'][name]['num_studies'] == int(mask.sum())
        close(saved['slices'][name]['macro_auc'], slices[name]['mathematical']['macro_auc'])
        close(saved['slices'][name]['per_class_auc'], slices[name]['mathematical']['per_class_auc'])

    # Independently reconstruct fixed timing identity, SHA and complete Studies.
    assert selection['manifest_sha256'] == sha(MANIFEST)
    assert selection['full_study_image_counts'] == {'/'.join(key): value for key, value in counts.items()}
    selection_groups, selected_patients = {}, []
    for name, expected_count, expected_studies in [('measured', 100, 53), ('warmup', 5, 4)]:
        items = selection[name]
        assert len(items) == expected_count
        selected_counts = Counter()
        identities = []
        for item in items:
            original = by_relative[item['relative_path']]
            for field in ['image_id', 'subject_id', 'study_id', 'study_key', 'relative_path', 'file_sha256']:
                assert item[field] == original[field]
            path = INPUTS / item['output_relative_path']
            assert sha(path) == item['sha256'] == original['file_sha256']
            assert not path.is_symlink()
            key = (item['subject_id'], item['study_id'])
            selected_counts[key] += 1
            identities.append('/'.join([*key, path.name]))
        assert len(set(identities)) == expected_count and len(selected_counts) == expected_studies
        assert all(value == counts[key] for key, value in selected_counts.items())
        patients = {key[0] for key in selected_counts}
        selected_patients.append(patients)
        group = {'images': expected_count, 'studies': expected_studies, 'patients': len(patients), 'complete_studies_proven': True}
        assert benchmark['coverage'][name] == group
        for field in ['images', 'studies', 'patients']:
            assert selection['counts'][f'{name}_{field}'] == group[field]
        selection_groups[name] = {'identities': sorted(identities), 'keys': sorted(selected_counts), **group}
    assert not selected_patients[0] & selected_patients[1]
    assert benchmark['coverage']['selection_sha256'] == sha(SELECTION)
    assert benchmark['coverage']['manifest_sha256'] == sha(MANIFEST)
    assert benchmark['coverage']['patient_disjoint'] is True and benchmark['coverage']['hashes_verified'] is True
    assert benchmark['protocol'] == {'measured_images': 100, 'warmup_images': 5, 'rounds': 3,
        'precision': 'float16', 'batch_size': 1, 'execution': 'serial',
        'timing': 'read_file_to_prediction_CSV_write_including_preprocess_postprocess',
        'model_initialization_timed': False, 'aggregation': 'per_image_three_round_mean_then_100_image_mean'}
    assert benchmark['executed'] is True and benchmark['status'] == 'local_stress_completed'
    assert benchmark['T4_latency_certified'] is False and benchmark['official_compliance_pass'] is False
    assert benchmark['T4_elapsed_ms'] is None
    assert benchmark['environment']['official_gpu'] is False and benchmark['environment']['official_environment_compatible'] is False
    assert '4090' in benchmark['environment']['gpu_name']
    assert benchmark['resources']['mode'] == 'native' and benchmark['resources']['requested_thread_percentage'] is None
    assert benchmark['resources']['cuda_visible_sm_count'] == 128
    assert benchmark['requested_resources']['memory_limit_gib'] == 16
    assert benchmark['deployment']['member_count'] == 2 and benchmark['deployment']['tta'] == 'hflip2'
    assert benchmark['deployment']['image_sizes'] == [768] and benchmark['deployment']['ensemble'] is True
    for index, member in enumerate(cfg['members']):
        assert benchmark['deployment']['members'][index] == {field: member.get(field) for field in
            ['weights_file', 'weights_sha256', 'weights_source', 'training_mode', 'source_checkpoint_sha256',
             'source_checkpoint_schema_version', 'seed', 'image_size', 'tta']}
    measured = selection_groups['measured']
    assert benchmark['image_order'] == measured['identities']
    times = np.asarray(benchmark['per_image_round_ms'], dtype=np.float64)
    assert times.shape == (100, 3) and np.isfinite(times).all() and (times >= 0).all()
    means = times.mean(axis=1)
    timing = {'mean_ms': float(means.mean()), 'median_ms': float(np.median(means)),
              'p95_ms': float(np.percentile(means, 95)), 'max_ms': float(means.max()),
              'round_mean_ms': times.mean(axis=0).tolist()}
    close(means, benchmark['per_image_mean_ms'])
    for field, value in timing.items():
        close(value, benchmark[field], atol=1e-12)
    assert benchmark['latency_limit_ms'] == 100
    assert benchmark['latency_pass_on_current_hardware'] == (timing['mean_ms'] <= 100)
    assert benchmark['comparison']['status'] == 'matched'
    assert len(benchmark['round_outputs']) == len(benchmark['comparison']['rounds']) == 3
    benchmark_results, first_study, first_image = [], None, None
    valid_indices = [keys.index(key) for key in measured['keys']]
    for index, declared in enumerate(benchmark['round_outputs']):
        csv_path = BENCH / f'round_{index + 1}.csv'
        image_path = csv_path.with_suffix('.image_predictions.csv')
        assert declared == {'path': csv_path.name, 'sha256': sha(csv_path),
                            'image_predictions': image_path.name, 'image_predictions_sha256': sha(image_path),
                            'studies': 53, 'study_classes': 10, 'image_classes': 10}
        study_values = probabilities(csv_path, measured['keys'])
        image_values = probabilities(image_path, measured['identities'], image=True)
        grouped = defaultdict(list)
        for image_id, values in zip(measured['identities'], image_values):
            grouped[tuple(Path(image_id).parts[:2])].append(values)
        independently_aggregated = np.asarray([np.mean(grouped[key], axis=0) for key in measured['keys']])
        close(study_values, independently_aggregated, atol=1e-7)
        if first_study is None:
            first_study, first_image = study_values.copy(), image_values.copy()
        close(study_values, first_study, atol=1e-7)
        close(image_values, first_image, atol=1e-7)
        assert benchmark['comparison']['rounds'][index] == {'compared_studies': 53, 'mismatched_studies': [],
                                                          'absolute_tolerance': 1e-7, 'match': True}
        difference = np.abs(study_values - actual[valid_indices])
        assert difference.max() <= 1e-7
        benchmark_results.append({'round': index + 1, 'study_csv_rows': 530, 'image_csv_rows': 1000,
                                 'vs_full_valid_probability_maxabs': float(difference.max()),
                                 'vs_full_valid_probability_meanabs': float(difference.mean()),
                                 'vs_first_round_study_maxabs': float(np.abs(study_values - first_study).max()),
                                 'vs_first_round_image_maxabs': float(np.abs(image_values - first_image).max()),
                                 'study_aggregation_maxabs': float(np.abs(study_values - independently_aggregated).max()),
                                 'metrics_on_selected_valid_studies': metric(targets[valid_indices], study_values)})

    after = hashes(protected)
    assert before == after, 'Protected source or production files changed during CPU audit'
    result = {'status': 'passed', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'deployment_config_sha256': config_sha, 'actual_inventory': actual_inventory,
              'member_order_seeds': [43, 44], 'weights_all_classes': cfg['model_weights'],
              'source_members': source_summaries, 'actual_fp16_tensors': fp16,
              'manifest': {'images': 1422, 'studies': 859, 'patients': 228,
                           'single_image_studies': sum(v == 1 for v in counts.values()),
                           'multiple_image_studies': sum(v > 1 for v in counts.values()),
                           'positive_studies_per_class': targets.sum(axis=0).tolist(),
                           'labels_preserved_and_consistent': True, 'train_valid_patients_disjoint': True},
              'actual_validation': {'rows': 8590, **actual_metrics},
              'mathematical_25_75_validation': mathematical_metrics,
              'actual_vs_mathematical': {'maxabs': float(delta.max()), 'meanabs': float(delta.mean()),
                  'per_class_maxabs': delta.max(axis=0).tolist(),
                  'macro_auc_delta': actual_metrics['macro_auc'] - mathematical_metrics['macro_auc'],
                  'per_class_auc_delta': (np.asarray(actual_metrics['per_class_auc']) - np.asarray(mathematical_metrics['per_class_auc'])).tolist(),
                  'reproduces_032_auc_exact_to_1e14': abs(actual_metrics['macro_auc'] - EXPECTED_AUC) <= 1e-14},
              'slices': slices, 'benchmark_timing_recomputed': timing,
              'benchmark_rounds': benchmark_results,
              'benchmark_protocol_and_input_sha_verified': True,
              'benchmark_T4_latency_certified': False, 'benchmark_official_compliance_pass': False,
              'protected_file_count': len(before), 'input_hashes_before': before, 'input_hashes_after': after,
              'source_and_production_files_unchanged': True,
              'no_gpu_work': True, 'no_model_construction_or_forward': True, 'test_data_usage': False,
              'limitations': ['The audit reads saved production outputs and does not replay 1422 image forwards.',
                  'FP32 member-blend/Study averaging order may differ by a few ulps from saved-CSV float64 arithmetic.',
                  'source_training_validation and auc_difference_from_source use member seed43 metadata; they are not dual-model training metrics.',
                  '4090 timing and its 16 GiB allocator cap do not certify or simulate actual T4 performance.',
                  'Patient bootstrap intervals are inherited from the earlier 032 review and are not recomputed here.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'output': str(OUT), 'inference_bytes': actual_inventory['total_bytes'],
                      'actual_macro_auc': actual_metrics['macro_auc'], 'mathematical_macro_auc': mathematical_metrics['macro_auc'],
                      'actual_vs_mathematical': result['actual_vs_mathematical'],
                      'benchmark_timing_recomputed': timing, 'benchmark_rounds': benchmark_results,
                      'protected_files_unchanged': len(before)}, indent=2))


if __name__ == '__main__':
    main()
