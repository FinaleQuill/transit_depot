"""One pre-registered no-training weight average; diagnostic only, never a deployment.

Default invocation prints the registration. --execute evaluates one derived
FP16 model on full official valid, without creating a production config.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

SOURCE_CONFIG_SHA = '9363149e143dec2871ac5fb4fbfdee4885f9f62add6a9f2acf4704635311b590'
SOURCE_WEIGHT_SHA = [
    'edef1f23e591bb52782a03efdc6ee76e10b4d9f115e90654e40f7dbf35f50269',
    '87ed41a966cf96144cc63acf50613a14c66fe8497b51625f120389a97a563489',
]
VALID_MANIFEST_SHA = 'c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112'
WEIGHTS = (.25, .75)
PROJECT = Path(__file__).resolve().parents[4]
REGISTRATION = Path(__file__).with_name('registration.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def merge_states(first, second, torch):
    """Preserve parameter names; ordered CPU float32 arithmetic, then FP16."""
    if not isinstance(first, dict) or not isinstance(second, dict) or not first:
        raise ValueError('Both source states must be nonempty dictionaries')
    if list(first) != list(second):
        raise ValueError('Source state keys/order differ')
    result = {}
    for key, left in first.items():
        right = second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError('Every source value must be a tensor')
        if left.device.type != 'cpu' or right.device.type != 'cpu':
            raise ValueError('Weight averaging must run on CPU')
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f'Source tensor shape/dtype differ: {key}')
        if left.is_floating_point():
            if left.dtype != torch.float16:
                raise ValueError('Frozen source inference tensors must be FP16')
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise ValueError(f'Nonfinite source tensor: {key}')
            value = left.float().mul(WEIGHTS[0]).add(right.float().mul(WEIGHTS[1])).to(torch.float16)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite derived tensor: {key}')
            result[key] = value
        else:
            if left.is_complex() or not torch.equal(left, right):
                raise ValueError(f'Nonfloating source tensors must be identical: {key}')
            result[key] = left.clone()
    return result


def package_versions():
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow')}


def execute(args):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score

    sys.path.insert(0, str(PROJECT))
    from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, Predictor, StudyAccumulator,
                           deployment_inventory, deployment_members, prediction_records,
                           read_submission, validate_deployment_config,
                           write_image_prediction, write_prediction)
    from model_overnight import build_overnight_model

    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; never overwrite a candidate')
    image_root = args.image_root.resolve(strict=True)
    if (destination.is_relative_to(image_root)
            or destination.is_relative_to(args.model_dir.resolve(strict=True))
            or destination.is_relative_to(args.baseline_predictions.resolve(strict=True).parent)):
        raise ValueError('Output cannot be within read-only source images,033 model,or baseline output')
    if any(key.startswith('CUDA_MPS_') for key in os.environ):
        raise RuntimeError('Unexpected MPS environment')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one CUDA GPU; no CPU/FP32 inference substitute')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the same Python3.12 inference environment as the 033 baseline')
    registration = json.loads(REGISTRATION.read_text())
    if (registration['candidate_count'] != 1 or registration['source_weight_sha256'] != SOURCE_WEIGHT_SHA
            or registration['script_sha256'] != sha256(__file__)):
        raise ValueError('Pre-registration does not describe the unique frozen candidate')
    config_path = args.model_dir / 'config.json'
    if sha256(config_path) != SOURCE_CONFIG_SHA or sha256(args.manifest) != VALID_MANIFEST_SHA:
        raise ValueError('Source033 config or official valid manifest differs from registration')
    config = json.loads(config_path.read_text())
    validate_deployment_config(config)
    inventory = deployment_inventory(args.model_dir, config)
    members = deployment_members(config)
    if len(members) != 2 or [member['seed'] for member in members] != [43, 44]:
        raise ValueError('Require original033 seed43/44 members in this order')
    if any(member['image_size'] != 768 or member['tta'] != 'hflip2' for member in members):
        raise ValueError('Require original768 and hflip2')
    if members[0]['model_config'] != members[1]['model_config']:
        raise ValueError('Source model architectures differ')
    if not np.array_equal(np.asarray(config['model_weights']), np.asarray([[.25] * 10, [.75] * 10])):
        raise ValueError('Source033 global probability weights differ')
    source_paths = [args.model_dir / member['weights_file'] for member in members]
    if [sha256(path) for path in source_paths] != SOURCE_WEIGHT_SHA:
        raise ValueError('Frozen033 source weight SHA mismatch')

    baseline_report = json.loads(args.baseline_report.read_text())
    baseline_receipt = json.loads(args.baseline_receipt.read_text())
    environment_probe = json.loads(args.environment_probe.read_text())
    versions = package_versions()
    props = torch.cuda.get_device_properties(0)
    if (baseline_receipt.get('exit_code') != 0 or baseline_receipt.get('seed') != 43
            or baseline_receipt.get('training') is not False
            or baseline_receipt.get('versions') != versions or baseline_receipt.get('gpu') != props.name):
        raise ValueError('Require successful seed43 nontraining033 baseline in exactly the same package/GPU environment')
    if (environment_probe['packages']['numpy'] != versions['numpy']
            or environment_probe['packages']['Pillow'] != versions['Pillow']
            or environment_probe['torch']['version'] != torch.__version__
            or environment_probe['torch']['cuda_build'] != torch.version.cuda
            or environment_probe['torch']['cudnn_version'] != torch.backends.cudnn.version()):
        raise ValueError('Recorded CUDA/cuDNN/inference environment differs')
    if (baseline_report.get('complete') is not True
            or baseline_report.get('deployment_config_sha256') != SOURCE_CONFIG_SHA
            or baseline_report.get('manifest_sha256') != VALID_MANIFEST_SHA
            or baseline_report.get('predictions_sha256') != sha256(args.baseline_predictions)):
        raise ValueError('Baseline predictions/report are not the complete frozen033 validation')

    records = prediction_records(image_root, args.manifest, split='valid')
    labels = {}
    identities = {}
    for row in records:
        target = json.loads(row['targets'])
        if len(target) != 10 or any(type(value) is not int or value not in (0, 1) for value in target):
            raise ValueError('Require all ten original valid labels')
        if row['study_key'] in labels and labels[row['study_key']] != target:
            raise ValueError('Inconsistent labels within a Study')
        labels[row['study_key']] = target
        identities[row['study_key']] = (row['subject_id'], row['study_id'])
        if row.get('file_sha256') and sha256(row['path']) != row['file_sha256']:
            raise ValueError('Original valid image SHA differs from frozen manifest')
    if len(records) != 1422 or len(labels) != 859 or config['official_valid_manifest_sha256'] != VALID_MANIFEST_SHA:
        raise ValueError('Require full official valid:1422 images and859 Studies')
    baseline_predictions = read_submission(args.baseline_predictions)
    if set(baseline_predictions) != set(identities.values()):
        raise ValueError('Baseline Study coverage differs from official valid')
    keys = list(labels)
    targets = np.asarray([labels[key] for key in keys], dtype=np.float64)
    base_values = np.asarray([baseline_predictions[identities[key]] for key in keys], dtype=np.float64)
    base_aucs = [float(roc_auc_score(targets[:, c], base_values[:, c])) for c in range(10)]
    baseline_auc = float(np.mean(base_aucs))
    if abs(baseline_auc - baseline_report['macro_auc']) > 1e-12:
        raise ValueError('Baseline AUC does not reproduce its report')

    destination.mkdir(parents=True, exist_ok=False)
    receipt = {
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'argv': sys.argv,
        'diagnostic_only': True, 'training': False, 'seed': 43,
        'source_config_sha256': SOURCE_CONFIG_SHA, 'source_weight_sha256': SOURCE_WEIGHT_SHA,
        'manifest_sha256': VALID_MANIFEST_SHA, 'versions': versions, 'gpu': props.name,
        'cuda_build': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'inference_source_sha256': {name: sha256(PROJECT / name) for name in
                                    ('deploy_kd.py', 'model_overnight.py', 'runtime_compat.py')},
        'baseline_predictions_sha256': sha256(args.baseline_predictions),
        'baseline_report_sha256': sha256(args.baseline_report),
        'baseline_receipt_sha256': sha256(args.baseline_receipt),
        'environment_probe_sha256': sha256(args.environment_probe),
        'T4_latency_certified': False, 'official_compliance_pass': False,
    }
    started = time.perf_counter()
    exit_code = 1
    try:
        random.seed(43)
        np.random.seed(43)
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / props.total_memory)
        with torch.no_grad():
            states = [torch.load(path, map_location='cpu', weights_only=True) for path in source_paths]
            derived = merge_states(states[0], states[1], torch)
            derived_path = destination / 'derived_weight_average_fp16.pt'
            torch.save(derived, derived_path)
            derived_sha = sha256(derived_path)
            predictor = Predictor.__new__(Predictor)
            predictor.torch = torch
            predictor.device = torch.device('cuda')
            # This minimal runtime handle is diagnostic, never validated as a
            # trained checkpoint or mislabeled as the original two-member config.
            predictor.members = [{'image_size': 768, 'tta': 'hflip2'}]
            predictor.model_weights = [[1.0] * 10]
            model = build_overnight_model(pretrained=False, **members[0]['model_config'])
            model.load_state_dict(derived, strict=True)
            model.eval().requires_grad_(False).to(predictor.device, dtype=torch.float16,
                                                   memory_format=torch.channels_last)
            predictor.models = [model]
            predictor.model = model
            del states, derived
            gc.collect()
        derivation = {
            'schema': 'diagnostic_weight_average_v1', 'diagnostic_only': True,
            'not_a_training_checkpoint': True, 'not_a_production_deployment': True,
            'source_config_sha256': SOURCE_CONFIG_SHA,
            'source_weight_sha256': SOURCE_WEIGHT_SHA,
            'formula': 'FP16(FP32(seed43_FP16)*0.25 + FP32(seed44_FP16)*0.75), ordered CPU multiply then add',
            'integer_tensor_policy': 'Require exact dtype/shape/value equality, then copy unchanged',
            'derived_weight_sha256': derived_sha, 'derived_weight_bytes': derived_path.stat().st_size,
            'source_config_copied_or_rewritten': False, 'training_performed': False,
            'source_inference_inventory': inventory,
        }
        write_json_new(destination / 'derivation.json', derivation)
        accumulator = StudyAccumulator(records)
        predictions = {}
        forward_count = [0]
        hook = model.register_forward_hook(lambda *unused: forward_count.__setitem__(0, forward_count[0] + 1))
        inference_started = time.perf_counter()
        try:
            with (destination / 'valid_predictions.csv').open('x', encoding='utf-8', newline='') as stream, \
                    (destination / 'valid_image_predictions.csv').open('x', encoding='utf-8', newline='') as image_stream:
                writer, image_writer = csv.writer(stream), csv.writer(image_stream)
                writer.writerow(CSV_HEADER)
                image_writer.writerow(IMAGE_CSV_HEADER)
                for row in records:
                    probabilities = predictor.predict_image(row['path'])
                    write_image_prediction(image_writer, row, probabilities)
                    result = accumulator.add(row, probabilities)
                    if result is not None:
                        subject, study, values = result
                        write_prediction(writer, subject, study, values)
                        predictions[(subject, study)] = values
        finally:
            hook.remove()
        predictor.synchronize()
        inference_wall_seconds = time.perf_counter() - inference_started
        if not accumulator.complete() or set(predictions) != set(identities.values()):
            raise RuntimeError('Incomplete valid prediction coverage')
        if forward_count[0] != len(records) * 2:
            raise RuntimeError('Expected precisely one model times two views per image')
        values = np.asarray([predictions[identities[key]] for key in keys], dtype=np.float64)
        aucs = [float(roc_auc_score(targets[:, c], values[:, c])) for c in range(10)]
        macro_auc = float(np.mean(aucs))
        improved = macro_auc > baseline_auc
        report = {
            'complete': True, 'diagnostic_only': True, 'training_performed': False,
            'candidate_count': 1, 'pre_registration_sha256': sha256(REGISTRATION),
            'script_sha256': sha256(__file__),
            'source_config_sha256': SOURCE_CONFIG_SHA, 'source_weight_sha256': SOURCE_WEIGHT_SHA,
            'derived_weight_sha256': derived_sha, 'derived_weight_bytes': derived_path.stat().st_size,
            'manifest_sha256': VALID_MANIFEST_SHA,
            'predictions_sha256': sha256(destination / 'valid_predictions.csv'),
            'image_predictions_sha256': sha256(destination / 'valid_image_predictions.csv'),
            'image_size': 768, 'precision': 'float16', 'tta': 'hflip2', 'batch_size': 1,
            'member_count': 1, 'forwards_per_image': 2, 'actual_forward_count': forward_count[0],
            'num_images': len(records), 'num_studies': len(labels),
            'macro_auc': macro_auc, 'per_class_auc': aucs,
            'macro_f1_fixed_0_5': float(f1_score(targets, values >= .5, average='macro', zero_division=0)),
            'baseline_macro_auc_same_environment': baseline_auc,
            'baseline_per_class_auc_same_environment': base_aucs,
            'auc_difference_from_baseline': macro_auc - baseline_auc,
            'per_class_auc_difference_from_baseline': [a - b for a, b in zip(aucs, base_aucs)],
            'baseline_macro_f1_fixed_0_5': float(f1_score(targets, base_values >= .5, average='macro', zero_division=0)),
            'point_auc_improved': improved,
            'decision': ('diagnostic_point_improvement_requires_uncertainty_resource_and_provenance_review'
                         if improved else 'stop_candidate_not_better_no_additional_weight_search'),
            'thresholds_fitted': False, 'weight_search_performed': False,
            'patient_uncertainty_evaluated': False, 'stable_improvement_claimed': False,
            'inference_wall_seconds_with_csv_and_forward_count_hook': inference_wall_seconds,
            'latency_is_not_a_formal_100_image_benchmark': True,
            'T4_latency_certified': False, 'official_compliance_pass': False,
            'environment': receipt,
            'note': 'Weight averaging changes predictions and is not an equivalent033 optimization. No production config, schema8 training checkpoint, test output, or deployment replacement is created.',
        }
        if [sha256(path) for path in source_paths] != SOURCE_WEIGHT_SHA or sha256(config_path) != SOURCE_CONFIG_SHA:
            raise RuntimeError('Frozen source bytes changed during diagnostic')
        write_json_new(destination / 'diagnostic_report.json', report)
        print(json.dumps({key: report[key] for key in ('macro_auc', 'per_class_auc', 'macro_f1_fixed_0_5',
                                                       'baseline_macro_auc_same_environment',
                                                       'auc_difference_from_baseline', 'decision')}, indent=2))
        exit_code = 0
    except BaseException:
        receipt['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        receipt.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       wall_seconds=time.perf_counter() - started, exit_code=exit_code)
        write_json_new(destination / 'execution_receipt.json', receipt)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--baseline-predictions', type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--baseline-receipt', type=Path)
    parser.add_argument('--environment-probe', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(REGISTRATION.read_text())
        return 0
    for name, value in vars(args).items():
        if name != 'execute' and value is None:
            parser.error('--execute requires --' + name.replace('_', '-'))
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
