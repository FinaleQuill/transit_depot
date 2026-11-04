"""Package and predict the one winning Base768 + two Tiny512 ensemble.

Postrun: python -m ensemble_sprint.deploy --config ensemble_sprint/plan.json --execute
The packaged predict.py and cost.py share complete-Study inference and writing.
Test images are read only after the sole validation comparison has selected a
winner and fixed validation images reproduce the saved candidate predictions.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from deploy_kd import (MAX_MODEL_BYTES, discover_images, match_timing_manifest,
                      prediction_records, read_submission, sha256_file)
from sprint.deploy import _write_studies, check_saved_validation, study_groups
from sprint.evaluate import _write_json

ROOT = Path(__file__).resolve().parents[1]


def selected_assets(config):
    result = [{'id': 'base768', 'path': str(config['base_weights']),
               'sha256': config['base_weights_sha256'], 'image_size': 768}]
    if [member['id'] for member in config['members']] != ['context512', 'opacity512']:
        raise ValueError('Require exactly the two registered Tiny512 members')
    for member in config['members']:
        path = Path(member.get('weights', Path(config['output']) / member['id'] / 'final_fp16.pt'))
        identity = sha256_file(path)
        if member.get('weights_sha256', identity) != identity:
            raise ValueError('Member weight identity differs from its frozen configuration')
        result.append({'id': member['id'], 'path': str(path), 'sha256': identity, 'image_size': 512})
    for asset in result:
        path = Path(asset['path']).resolve(strict=True)
        if sha256_file(path) != asset['sha256']:
            raise ValueError('Selected weight changed: ' + asset['id'])
        asset.update(path=str(path), bytes=path.stat().st_size)
    return result


def require_winner(config):
    comparison = Path(config['output']) / 'final_comparison'
    selection = json.loads((comparison / 'selection.json').read_text())
    report = json.loads((comparison / 'report.json').read_text())
    if (selection.get('candidate_won') is not True or report.get('candidate_won') is not True
            or report.get('complete') is not True):
        raise ValueError('Only the completed winning final comparison permits test prediction')
    if selection.get('report') and Path(selection['report']).resolve() != (comparison / 'report.json').resolve():
        raise ValueError('Selection points to another comparison')
    if any(value is not True for value in selection.get('selection_checks', {}).values()):
        raise ValueError('Selection failed a frozen acceptance check')
    if selection.get('fusion_weights') != config['fusion_weights']:
        raise ValueError('Selection and inference blend disagree')
    for name, field in [('candidate_predictions.csv', 'candidate_predictions_sha256'),
                        ('candidate.image_predictions.csv', 'candidate_image_predictions_sha256')]:
        if sha256_file(comparison / name) != report[field]:
            raise ValueError('Final validation predictions changed')
    if sha256_file(config['valid_manifest']) != report['manifest_sha256']:
        raise ValueError('Original validation manifest changed')
    assets = selected_assets(config)
    validated = {str(Path(item['path']).resolve()): item['sha256']
                 for item in report['candidate_inference_assets']}
    if validated != {asset['path']: asset['sha256'] for asset in assets}:
        raise ValueError('Packaged weights must be precisely the three validated assets')
    if report.get('fusion_weights', report.get('fixed_classwise_probability_weights')) != config['fusion_weights']:
        raise ValueError('Frozen blend differs from the sole evaluation')
    return comparison, selection, report, assets


def _copy_sources(destination):
    pending, seen = ['ensemble_sprint.deploy', 'ensemble_sprint.predictor'], set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if not source.is_file():
            source = ROOT.joinpath(*module.split('.')) / '__init__.py'
            if not source.is_file():
                continue
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        # A namespace directory can lose import precedence to the original
        # regular package later on sys.path. Preserve every existing package
        # initializer and traverse its dependencies, without creating new ones.
        for parent in relative.parents:
            if parent != Path('.') and (ROOT / parent / '__init__.py').is_file():
                pending.append('.'.join((*parent.parts, '__init__')))
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                pending.append(node.module)
    return [{'path': str(path.relative_to(destination)), 'sha256': sha256_file(path)}
            for path in sorted(destination.rglob('*.py'))]


def package_candidate(config, output):
    comparison, selection, report, assets = require_winner(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = output / 'model'
    model.mkdir()
    entries = []
    for asset in assets:
        filename = asset['id'] + '.pt'
        shutil.copyfile(asset['path'], model / filename)
        if sha256_file(model / filename) != asset['sha256']:
            raise RuntimeError('Copied weight differs from validated identity')
        entries.append({'path': filename, 'bytes': (model / filename).stat().st_size,
                        'sha256': asset['sha256'], 'id': asset['id']})
    # Preserve model construction metadata but all actual inference asset paths
    # are local, explicit and immutable; no training-state discovery is needed.
    runtime = {'schema': 'cxr_fixed_three_member_sprint_v1',
               'base_weights': 'base768.pt', 'base_weights_sha256': assets[0]['sha256'],
               'fusion_weights': copy.deepcopy(config['fusion_weights']),
               'members': [], 'seed': config.get('seed', 20261105), 'tta': 'hflip2',
               'output': '.', 'inference_files': entries, 'patient_smoothing': False,
               'selection_sha256': sha256_file(comparison / 'selection.json'),
               'comparison_sha256': sha256_file(comparison / 'report.json')}
    for member, asset in zip(config['members'], assets[1:]):
        item = copy.deepcopy(member)
        for key in list(item):
            if key.startswith('initial_'):
                del item[key]
        item.update(weights=asset['id'] + '.pt', weights_sha256=asset['sha256'])
        runtime['members'].append(item)
    _write_json(model / 'runtime_config.json', runtime)
    sources = _copy_sources(output / 'code')
    if (ROOT / 'licenses').is_dir():
        shutil.copytree(ROOT / 'licenses', output / 'licenses')
    for name, function in [('predict.py', 'predict_main'), ('cost.py', 'cost_main')]:
        (output / name).write_text('from pathlib import Path\nimport sys\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))\n'
            f'from ensemble_sprint.deploy import {function}\n'
            f'if __name__ == "__main__":\n    raise SystemExit({function}())\n')
    (output / 'requirements_runtime.txt').write_text(
        'timm==1.0.30\nhuggingface_hub==0.34.4\nsafetensors==0.8.0\ntqdm==4.67.1\n')
    (output / 'README.md').write_text(
        '# Fixed ensemble candidate\n\nUse the official competition image; install '
        'requirements_runtime.txt if needed, retaining its PyTorch and other core packages.\n\n'
        '`python predict.py --image-root ROOT --manifest TEST_MANIFEST --output NEW/submission.csv`\n\n'
        '`python cost.py submission.csv PICTURES_100 PICTURES_5 --manifest TEST_MANIFEST`\n\n'
        'Timing uses 100 complete measured Studies, five disjoint complete warmup Studies, '
        'three rounds and serial Study batch=1. All images, preprocessing, postprocessing and '
        'CSV writing are included. Validation and submission use identical frozen hflip2 '
        'FP16 member inference and class probability fusion. Official T4 acceptance is pending.\n')
    for filename in ('report.json', 'selection.json'):
        shutil.copyfile(comparison / filename, output / ('validation_' + filename))
    inventory = inspect_package(model)
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if size > MAX_MODEL_BYTES:
        raise ValueError('Model, code and necessary files exceed 500MB')
    receipt = {'state': 'packaged', 'model_inventory': inventory, 'source_files': sources,
               'package_bytes_before_predictions': size, 'size_pass': True,
               'formal_T4_certification': False, 'deployment_accepted': False}
    _write_json(output / 'package_receipt.json', receipt)
    return receipt


def inspect_package(model_dir):
    from ensemble_sprint.predictor import validate_recipe
    root = Path(model_dir).resolve(strict=True)
    config = json.loads((root / 'runtime_config.json').read_text())
    validate_recipe(config)
    if (config.get('schema') != 'cxr_fixed_three_member_sprint_v1'
            or config.get('tta') != 'hflip2' or config.get('patient_smoothing') is not False):
        raise ValueError('Unexpected inference configuration')
    fusion = np.asarray(config['fusion_weights'], dtype=np.float64)
    if (fusion.shape != (3, 10) or not np.isfinite(fusion).all() or (fusion < 0).any()
            or not np.allclose(fusion.sum(0), 1., atol=1e-12, rtol=0)
            or not np.array_equal(fusion[:, 0], [1., 0., 0.])):
        raise ValueError('Invalid fixed ensemble weights')
    files = config['inference_files']
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if (len(files) != 3 or len({item['path'] for item in files}) != 3
            or actual != {item['path'] for item in files} | {'runtime_config.json'}):
        raise ValueError('The package must contain exactly its three registered weight files')
    for item in files:
        path = root / item['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']):
            raise ValueError('Packaged model asset changed: ' + item['path'])
    count = sum((root / item).stat().st_size for item in actual)
    if count > MAX_MODEL_BYTES:
        raise ValueError('Loaded inference assets exceed 500MB')
    return {'files': files, 'total_bytes': count, 'size_pass': True,
            'config_sha256': sha256_file(root / 'runtime_config.json'), 'limit_bytes': MAX_MODEL_BYTES}


def load_predictor(model_dir):
    from ensemble_sprint.predictor import EnsemblePredictor
    root = Path(model_dir).resolve(strict=True)
    inventory = inspect_package(root)
    config = json.loads((root / 'runtime_config.json').read_text())
    config['base_weights'] = str(root / config['base_weights'])
    for member in config['members']:
        member['weights'] = str(root / member['weights'])
    config['output'] = str(root)
    predictor = EnsemblePredictor(config)
    predictor.inventory = inventory
    return predictor


def audit_submission(output, records):
    from cost import read_image_probabilities
    output = Path(output)
    predictions = read_submission(output / 'submission.csv')
    images = read_image_probabilities(output / 'submission.image_predictions.csv')
    identities = {(row['subject_id'], row['study_id'], row['image_id']) for row in records}
    if len(identities) != 2218 or set(images) != identities or len(predictions) != 1414:
        raise ValueError('Incomplete original test image/Study coverage')
    sums, counts = {}, {}
    for row in records:
        key = (row['subject_id'], row['study_id'])
        sums.setdefault(key, np.zeros(10, dtype=np.float64))
        sums[key] += images[(*key, row['image_id'])]
        counts[key] = counts.get(key, 0) + 1
    if set(sums) != set(predictions):
        raise ValueError('Published Studies do not match original test manifest')
    error = max(float(np.abs(predictions[key] - sums[key] / counts[key]).max()) for key in predictions)
    if error != 0.:
        raise ValueError('Study probabilities differ from their original-image FP64 mean')
    return {'complete': True, 'images': len(images), 'studies': len(predictions),
            'data_rows': len(predictions) * 10, 'image_data_rows': len(images) * 10,
            'study_mean_max_absolute_error': error,
            'submission_sha256': sha256_file(output / 'submission.csv'),
            'image_predictions_sha256': sha256_file(output / 'submission.image_predictions.csv'),
            'test_used_for_selection': False, 'test_labels_read': False}


def publish(config):
    comparison, _, _, _ = require_winner(config)
    output = Path(config['output']) / 'deployment_candidate'
    if output.exists():
        raise FileExistsError('Retain existing deployment artifacts; no implicit overwrite or rerun')
    try:
        package = package_candidate(config, output)
        predictor = load_predictor(output / 'model')
        consistency = check_saved_validation(predictor, config, comparison, output)
        manifest = Path(config.get('test_manifest',
            ROOT.parent / 'cloud_project_v2/data/clean_v1_reviewed/test_manifest.csv'))
        records = prediction_records(config['image_root'], manifest)
        if len(records) != 2218 or len(study_groups(records)) != 1414:
            raise ValueError('Original official test coverage changed')
        _write_studies(predictor, records, output / 'submission.csv')
        audit = audit_submission(output, records)
        _write_json(output / 'submission_audit.json', audit)
        files = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                  'sha256': sha256_file(path)} for path in sorted(output.rglob('*')) if path.is_file()]
        package_bytes = sum(item['bytes'] for item in files)
        if package_bytes > MAX_MODEL_BYTES:
            raise ValueError('Complete candidate package exceeds 500MB')
        status = {'state': 'candidate_ready', 'submission': str(output / 'submission.csv'),
                  'submission_sha256': audit['submission_sha256'], 'audit': audit,
                  'manifest_sha256': sha256_file(manifest), 'package': package,
                  'implementation_consistency': consistency, 'package_files': files,
                  'package_bytes_including_predictions': package_bytes,
                  'baseline_submissions_preserved': True, 'test_used_for_selection': False,
                  'formal_T4_certification': False, 'deployment_accepted': False,
                  'pending': 'Official T4 average <=100ms/Study and organizer acceptance'}
        _write_json(output / 'deployment_status.json', status)
        _write_json(Path(config['output']) / 'selected_current.json', {
            'selected': 'three_member_ensemble', 'submission': str(output / 'submission.csv'),
            'submission_sha256': audit['submission_sha256'], 'model_directory': str(output / 'model'),
            'candidate_ready': True, 'formal_T4_certification': False, 'deployment_accepted': False})
        return status
    except BaseException as error:
        if output.is_dir():
            _write_json(output / 'deployment_status.json', {'state': 'failed',
                'error': f'{type(error).__name__}: {error}', 'baseline_submissions_preserved': True,
                'formal_T4_certification': False, 'deployment_accepted': False})
        raise


def predict_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(sys.argv[0]).resolve().parent / 'model')
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.name != 'submission.csv':
        raise ValueError('Require a new submission.csv path')
    if any(args.output.resolve().is_relative_to(Path(path).resolve())
           for path in (args.image_root, args.model_dir)):
        raise ValueError('Output must not modify input images or model assets')
    records = prediction_records(args.image_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_studies(load_predictor(args.model_dir), records, args.output)
    return 0


def cost_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('pictures_100', type=Path)
    parser.add_argument('pictures_5', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path(sys.argv[0]).resolve().parent / 'model')
    parser.add_argument('--output-dir', type=Path, default=Path('cost_result'))
    args = parser.parse_args(argv)
    measured, warmup = discover_images(args.pictures_100), discover_images(args.pictures_5)
    if len(study_groups(measured)) != 100 or len(study_groups(warmup)) != 5:
        raise ValueError('Require 100 measured and five warmup complete Studies')
    if {row['study_key'] for row in measured} & {row['study_key'] for row in warmup}:
        raise ValueError('Warmup and measured Studies must be disjoint')
    for records in (measured, warmup):
        if not match_timing_manifest(records, args.manifest)['complete_studies_proven']:
            raise ValueError('Timing inputs omit original images of a selected Study')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictor = load_predictor(args.model_dir)
    reference = read_submission(args.submission)
    for group in study_groups(warmup):
        predictor.predict_study(group)
    predictor.synchronize()
    rounds, max_error, matched = [], 0., True
    for index in range(3):
        times, results = _write_studies(predictor, measured, args.output_dir / f'round_{index+1}.csv', timing=True)
        rounds.append(times)
        for key, value in results.items():
            if key not in reference:
                raise ValueError('Timed Study is absent from submission')
            max_error = max(max_error, float(np.abs(value - reference[key]).max()))
            matched &= bool(np.array_equal(np.round(value, 4), np.round(reference[key], 4)))
    from cost import timing_environment
    environment = timing_environment(predictor)
    mean = float(np.asarray(rounds, dtype=np.float64).mean())
    result = {'protocol': '100 complete Studies, serial Study batch=1, five warmups, three rounds',
              'rounds_ms': rounds, 'mean_ms_per_study': mean, 'latency_pass': mean <= 100.,
              'four_decimal_consistency': matched, 'maximum_raw_probability_error': max_error,
              'environment': environment, 'model_inventory': predictor.inventory,
              'formal_T4_certification': False, 'organizer_acceptance_required': True,
              'local_checks_pass': matched and max_error <= 1e-3 and mean <= 100.}
    _write_json(args.output_dir / 'cost_report.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['local_checks_pass'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps({'requires': 'sole completed winning comparison', 'test_images_read': False}))
        return
    print(json.dumps(publish(config), indent=2))


if __name__ == '__main__':
    main()
