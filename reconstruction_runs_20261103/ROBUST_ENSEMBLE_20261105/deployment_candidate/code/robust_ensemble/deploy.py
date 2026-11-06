"""Portable, winner-only publication of the frozen Sprint + opacity ensemble.

The complete validated model identities and all acceptance checks are verified
before test images are opened. Prediction and timing share Study inference.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from deploy_kd import (MAX_MODEL_BYTES, discover_images, match_timing_manifest,
                      prediction_records, read_submission, sha256_file)
from ensemble_sprint.deploy import audit_submission
from sprint.deploy import _write_studies, check_saved_validation, study_groups
from sprint.evaluate import _write_json

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCHEMA = 'cxr_robust_ensemble_deployment_v1'


def selected_assets(config):
    """Authenticate the three networks and the Tiny's existing classifier patch."""
    from robust_ensemble.predictor import validate_recipe
    validate_recipe(config)
    sprint = config['sprint_config']
    patch_path = Path(sprint['tiny_patch_config']).resolve(strict=True)
    if patch_path.is_dir():
        patch_path = patch_path / 'config.json'
    patch = json.loads(patch_path.read_text())
    base = Path(patch['base_model_dir'])
    base = base.resolve(strict=True) if base.is_absolute() else (patch_path.parent / base).resolve(strict=True)
    base_config = json.loads((base / 'config.json').read_text())
    member = config['opacity_member']
    entries = [
        ('base768', Path(sprint['candidate_weights']), sprint['candidate_weights_sha256']),
        ('tiny768', base / base_config['weights_file'], patch['base_weights_sha256']),
        ('tiny_config', base / 'config.json', patch['base_config_sha256']),
        ('tiny_patch', patch_path.parent / patch['patch_file'], patch['patch_sha256']),
        ('tiny_patch_config', patch_path, sprint.get('tiny_patch_config_sha256', sha256_file(patch_path))),
        ('opacity512', Path(member['weights']), member['weights_sha256']),
    ]
    result = []
    for identity, path, expected in entries:
        path = path.resolve(strict=True)
        if sha256_file(path) != expected:
            raise ValueError('Validated inference asset changed: ' + identity)
        result.append({'id': identity, 'path': str(path), 'sha256': expected,
                       'bytes': path.stat().st_size})
    if len({item['path'] for item in result}) != len(result):
        raise ValueError('Inference asset roles must have distinct paths')
    return result


def require_winner(config):
    from robust_ensemble.predictor import FUSION_WEIGHTS, validate_recipe
    validate_recipe(config)
    comparison = Path(config['output']).resolve() / 'final_comparison'
    report_path = comparison / 'report.json'
    report = json.loads(report_path.read_text())
    selection = json.loads((comparison / 'selection.json').read_text())
    if (selection.get('candidate_won') is not True or report.get('candidate_won') is not True
            or report.get('complete') is not True):
        raise ValueError('Only the completed winning comparison permits publication')
    checks = selection.get('selection_checks')
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        raise ValueError('Every frozen acceptance check must pass')
    if Path(selection.get('report', '')).resolve() != report_path:
        raise ValueError('Selection points to another report')
    for record in (selection, report):
        if record.get('fusion_weights') != FUSION_WEIGHTS:
            raise ValueError('Validated fusion weights changed')
    for filename, field in [('candidate_predictions.csv', 'candidate_predictions_sha256'),
                            ('candidate.image_predictions.csv', 'candidate_image_predictions_sha256')]:
        if sha256_file(comparison / filename) != report[field]:
            raise ValueError('Saved final validation probabilities changed')
    if sha256_file(config['valid_manifest']) != report['manifest_sha256']:
        raise ValueError('Original validation manifest changed')
    assets = selected_assets(config)
    recorded = report.get('candidate_inference_assets', [])
    actual = {item['path']: item['sha256'] for item in assets}
    validated = {str(Path(item['path']).resolve()): item['sha256'] for item in recorded}
    if len(recorded) != len(validated) or actual != validated:
        raise ValueError('Publication must use exactly the validated inference assets')
    return comparison, selection, report, assets


def _copy_sources(destination):
    """Retain dependency closure and package initializers after relocation."""
    destination = Path(destination)
    pending, seen = ['robust_ensemble.deploy', 'robust_ensemble.predictor'], set()
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
    from robust_ensemble.predictor import FUSION_WEIGHTS
    comparison, _, report, assets = require_winner(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = output / 'model'
    (model / 'tiny').mkdir(parents=True)
    (model / 'tiny_patch').mkdir()
    by_id = {item['id']: item for item in assets}
    patch = json.loads(Path(by_id['tiny_patch_config']['path']).read_text())
    base_config = json.loads(Path(by_id['tiny_config']['path']).read_text())
    targets = {'base768': model / 'base768.pt', 'opacity512': model / 'opacity512.pt',
               'tiny768': model / 'tiny' / base_config['weights_file'],
               'tiny_config': model / 'tiny/config.json',
               'tiny_patch': model / 'tiny_patch' / patch['patch_file']}
    for identity, target in targets.items():
        if target.parent not in (model, model / 'tiny', model / 'tiny_patch'):
            raise ValueError('Model filenames must remain within their package directory')
        shutil.copyfile(by_id[identity]['path'], target)
        if sha256_file(target) != by_id[identity]['sha256']:
            raise RuntimeError('Copied inference asset changed: ' + identity)
    patch['base_model_dir'] = '../tiny'
    _write_json(model / 'tiny_patch/config.json', patch)
    files = [{'path': str(path.relative_to(model)), 'bytes': path.stat().st_size,
              'sha256': sha256_file(path)} for path in sorted(model.rglob('*')) if path.is_file()]
    opacity = {key: copy.deepcopy(config['opacity_member'][key]) for key in
               ('id', 'branch', 'image_size', 'epochs', 'backbone_name')}
    opacity.update(weights='opacity512.pt', weights_sha256=by_id['opacity512']['sha256'])
    runtime = {'schema': DEPLOY_SCHEMA, 'fusion_weights': copy.deepcopy(FUSION_WEIGHTS),
               'opacity_weight_per_class': list(FUSION_WEIGHTS[1]), 'tta': 'hflip2',
               'patient_smoothing': False, 'inference_files': files,
               'sprint_config': {'candidate_weights': 'base768.pt',
                   'candidate_weights_sha256': by_id['base768']['sha256'],
                   'tiny_patch_config': 'tiny_patch/config.json',
                   'tiny_patch_config_sha256': sha256_file(model / 'tiny_patch/config.json')},
               'opacity_member': opacity, 'seed': config.get('seed', 20261105),
               'official_valid_manifest_sha256': report['manifest_sha256'],
               'selection_sha256': sha256_file(comparison / 'selection.json'),
               'comparison_sha256': sha256_file(comparison / 'report.json')}
    _write_json(model / 'runtime_config.json', runtime)
    sources = _copy_sources(output / 'code')
    if (ROOT / 'licenses').is_dir():
        shutil.copytree(ROOT / 'licenses', output / 'licenses')
    for name, function in [('predict.py', 'predict_main'), ('cost.py', 'cost_main')]:
        (output / name).write_text('from pathlib import Path\nimport sys\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))\n'
            f'from robust_ensemble.deploy import {function}\n'
            f'if __name__ == "__main__":\n    raise SystemExit({function}())\n')
    (output / 'requirements_runtime.txt').write_text(
        'timm==1.0.30\nhuggingface_hub==0.34.4\nsafetensors==0.8.0\ntqdm==4.67.1\n')
    (output / 'README.md').write_text(
        '# Fixed Sprint + opacity ensemble\n\n'
        'Use the official competition image with requirements_runtime.txt.\n\n'
        '`python predict.py --image-root ROOT --manifest TEST_MANIFEST --output NEW/submission.csv`\n\n'
        '`python cost.py submission.csv PICTURES_100 PICTURES_5 --manifest TEST_MANIFEST`\n\n'
        'Class 1 retains Sprint exactly. Other classes use 80% Sprint and 20% opacity. '
        'Each member uses fixed FP16 identity+hflip inference; all original Study images '
        'are averaged. Cost uses the identical predictor, 100 complete Studies, five '
        'disjoint warmup Studies and three serial rounds. Official T4 timing remains pending.\n')
    for filename in ('report.json', 'selection.json'):
        shutil.copyfile(comparison / filename, output / ('validation_' + filename))
    inventory = inspect_package(model)
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if size > MAX_MODEL_BYTES:
        raise ValueError('Complete package exceeds 500MB')
    receipt = {'state': 'packaged', 'model_inventory': inventory, 'source_files': sources,
               'package_bytes_before_predictions': size, 'size_pass': True,
               'formal_T4_certification': False, 'deployment_accepted': False}
    _write_json(output / 'package_receipt.json', receipt)
    return receipt


def inspect_package(model_dir):
    from robust_ensemble.predictor import validate_recipe
    root = Path(model_dir).resolve(strict=True)
    config = json.loads((root / 'runtime_config.json').read_text())
    validate_recipe(config)
    if config.get('schema') != DEPLOY_SCHEMA or config.get('patient_smoothing') is not False:
        raise ValueError('Unexpected portable ensemble configuration')
    files = config['inference_files']
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if (len(files) != 6 or len({item['path'] for item in files}) != 6
            or actual != {item['path'] for item in files} | {'runtime_config.json'}):
        raise ValueError('Package must contain exactly the six registered inference assets')
    for item in files:
        path = root / item['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']):
            raise ValueError('Packaged model asset changed: ' + item['path'])
    loaded = _absolute_config(root, config)
    bound = {item['path']: item['sha256'] for item in selected_assets(loaded)}
    registered = {str((root / item['path']).resolve()): item['sha256'] for item in files}
    if bound != registered:
        raise ValueError('Runtime references differ from registered local model assets')
    total = sum((root / filename).stat().st_size for filename in actual)
    if total > MAX_MODEL_BYTES:
        raise ValueError('Loaded inference assets exceed 500MB')
    return {'files': files, 'total_bytes': total, 'size_pass': True,
            'config_sha256': sha256_file(root / 'runtime_config.json'), 'limit_bytes': MAX_MODEL_BYTES}


def _absolute_config(root, config):
    result = copy.deepcopy(config)
    for key in ('candidate_weights', 'tiny_patch_config'):
        relative = Path(result['sprint_config'][key])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Portable model references must be relative and contained')
        result['sprint_config'][key] = str(root / relative)
    relative = Path(result['opacity_member']['weights'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Portable opacity reference must be relative and contained')
    result['opacity_member']['weights'] = str(root / relative)
    return result


def load_predictor(model_dir):
    from robust_ensemble.predictor import RobustPredictor
    root = Path(model_dir).resolve(strict=True)
    inventory = inspect_package(root)
    config = _absolute_config(root, json.loads((root / 'runtime_config.json').read_text()))
    predictor = RobustPredictor(config)
    predictor.inventory = inventory
    return predictor


def publish(config):
    comparison, _, _, _ = require_winner(config)
    output = Path(config['output']) / 'deployment_candidate'
    if output.exists():
        raise FileExistsError('Retain existing deployment artifacts; no implicit overwrite')
    try:
        package = package_candidate(config, output)
        predictor = load_predictor(output / 'model')
        consistency = check_saved_validation(predictor, config, comparison, output)
        manifest = Path(config['test_manifest'])
        if sha256_file(manifest) != config['test_manifest_sha256']:
            raise ValueError('Original official test manifest changed')
        records = prediction_records(config['image_root'], manifest)
        if len(records) != 2218 or len(study_groups(records)) != 1414:
            raise ValueError('Original official test coverage changed')
        _write_studies(predictor, records, output / 'submission.csv')
        audit = audit_submission(output, records)
        _write_json(output / 'submission_audit.json', audit)
        files = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                  'sha256': sha256_file(path)} for path in sorted(output.rglob('*')) if path.is_file()]
        size = sum(item['bytes'] for item in files)
        if size > MAX_MODEL_BYTES:
            raise ValueError('Complete candidate package exceeds 500MB')
        status = {'state': 'candidate_ready', 'submission': str(output / 'submission.csv'),
                  'submission_sha256': audit['submission_sha256'], 'audit': audit,
                  'manifest_sha256': sha256_file(manifest), 'package': package,
                  'implementation_consistency': consistency, 'package_files': files,
                  'package_bytes_including_predictions': size,
                  'baseline_submissions_preserved': True, 'test_used_for_selection': False,
                  'formal_T4_certification': False, 'deployment_accepted': False,
                  'pending': 'Official T4 average <=100ms/Study and organizer acceptance'}
        _write_json(output / 'deployment_status.json', status)
        _write_json(Path(config['output']) / 'deployment_status.json', status)
        _write_json(Path(config['output']) / 'selected_current.json', {
            'selected': 'robust_ensemble', 'submission': str(output / 'submission.csv'),
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
    rounds, maximum, matched = [], 0., True
    for index in range(3):
        times, results = _write_studies(predictor, measured, args.output_dir / f'round_{index+1}.csv', timing=True)
        rounds.append(times)
        for key, value in results.items():
            if key not in reference:
                raise ValueError('Timed Study is absent from submission')
            maximum = max(maximum, float(np.abs(value - reference[key]).max()))
            matched &= bool(np.array_equal(np.round(value, 4), np.round(reference[key], 4)))
    from cost import timing_environment
    mean = float(np.asarray(rounds, dtype=np.float64).mean())
    result = {'protocol': '100 complete Studies, serial Study batch=1, five warmups, three rounds',
              'rounds_ms': rounds, 'mean_ms_per_study': mean, 'latency_pass': mean <= 100.,
              'four_decimal_consistency': matched, 'maximum_raw_probability_error': maximum,
              'environment': timing_environment(predictor), 'model_inventory': predictor.inventory,
              'formal_T4_certification': False, 'organizer_acceptance_required': True,
              'local_checks_pass': matched and maximum <= 1e-3 and mean <= 100.}
    _write_json(args.output_dir / 'cost_report.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['local_checks_pass'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'requires': 'sole completed winning comparison', 'test_images_read': False}))
        return
    print(json.dumps(publish(json.loads(args.config.read_text())), indent=2))


if __name__ == '__main__':
    main()
