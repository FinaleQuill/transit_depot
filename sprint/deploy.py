"""Publish the sole winning sprint candidate; predict and cost share one path.

Postrun: python -m sprint.deploy --config sprint/plan.json --execute
Package: python predict.py --image-root ROOT --manifest TEST --output submission.csv
Timing:  python cost.py submission.csv pictures_100 pictures_5 --manifest TEST

The timing directories must contain 100 and 5 COMPLETE Studies, respectively.
No model, blend, threshold or epoch is selected using test images or timings.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from deploy_kd import (CSV_HEADER, IMAGE_CSV_HEADER, MAX_MODEL_BYTES, StudyAccumulator,
    _letterbox_image, aggregate_member_probabilities, discover_images, match_timing_manifest,
    prediction_records, read_submission, sha256_file, write_image_prediction, write_prediction)
from sprint.evaluate import MODEL_WEIGHTS, _candidate, _tiny, _write_json

ROOT = Path(__file__).resolve().parents[1]


def study_groups(records):
    groups = {}
    for row in records:
        groups.setdefault(row['study_key'], []).append(row)
    return list(groups.values())


def winning_selection(directory):
    directory = Path(directory).resolve(strict=True)
    selection = json.loads((directory / 'selection.json').read_text())
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    if (selection.get('candidate_won') is not True or report.get('candidate_won') is not True
            or report.get('complete') is not True or selection.get('selected') != 'sprint_candidate'):
        raise ValueError('Only the completed sole winning candidate may be published')
    if (Path(selection['report']).resolve() != report_path
            or selection.get('candidate_weights_sha256') != report.get('candidate_weights_sha256')
            or selection.get('tta') != 'hflip2'
            or selection.get('fixed_classwise_probability_weights') != MODEL_WEIGHTS
            or report.get('fixed_classwise_probability_weights') != MODEL_WEIGHTS
            or any(value is not True for value in selection.get('selection_checks', {}).values())):
        raise ValueError('Final selection does not match its frozen comparison')
    if sha256_file(selection['candidate_weights']) != report['candidate_weights_sha256']:
        raise ValueError('Selected weights changed after validation')
    for asset in report.get('candidate_inference_assets', []):
        if sha256_file(asset['path']) != asset['sha256']:
            raise ValueError('Validated inference asset changed: ' + asset['path'])
    for name, field in [('candidate_predictions.csv', 'candidate_predictions_sha256'),
                        ('candidate.image_predictions.csv', 'candidate_image_predictions_sha256')]:
        if sha256_file(directory / name) != report[field]:
            raise ValueError('Saved final validation probabilities changed')
    return selection, report


def _copy_runtime_sources(destination):
    """Copy the local Python dependency closure, preserving package structure."""
    pending, seen = ['sprint.deploy'], set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                pending.append(node.module)
    return [{'path': str(p.relative_to(destination)), 'sha256': sha256_file(p)}
            for p in sorted(destination.rglob('*.py'))]


def prepare_package(comparison, output):
    selection, report = winning_selection(comparison)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Deployment directory already exists; preserve it for review')
    output.mkdir(parents=True)
    model = output / 'model'
    (model / 'tiny').mkdir(parents=True)
    (model / 'tiny_patch').mkdir()
    source_patch = Path(selection['tiny_patch_config']).resolve(strict=True)
    patch_config = json.loads(source_patch.read_text())
    source_base = Path(patch_config['base_model_dir'])
    if not source_base.is_absolute():
        source_base = (source_patch.parent / source_base).resolve()
    base_config = json.loads((source_base / 'config.json').read_text())
    source_weights = source_base / base_config['weights_file']
    source_patch_weights = source_patch.parent / patch_config['patch_file']
    checked = [(source_base / 'config.json', patch_config['base_config_sha256']),
               (source_weights, patch_config['base_weights_sha256']),
               (source_patch_weights, patch_config['patch_sha256'])]
    for source, expected in checked:
        if sha256_file(source) != expected:
            raise ValueError('Frozen Tiny source changed: ' + str(source))
    shutil.copyfile(selection['candidate_weights'], model / 'sprint_fp16.pt')
    shutil.copyfile(source_weights, model / 'tiny' / base_config['weights_file'])
    shutil.copyfile(source_base / 'config.json', model / 'tiny/config.json')
    shutil.copyfile(source_patch_weights, model / 'tiny_patch' / patch_config['patch_file'])
    patch_config['base_model_dir'] = '../tiny'
    _write_json(model / 'tiny_patch/config.json', patch_config)
    files = [{'path': str(p.relative_to(model)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
             for p in sorted(model.rglob('*')) if p.is_file()]
    configuration = {
        'schema': 'cxr_sprint_deployment_v1', 'candidate_weights': 'sprint_fp16.pt',
        'candidate_weights_sha256': report['candidate_weights_sha256'],
        'tiny_patch_config': 'tiny_patch/config.json', 'image_size': 768, 'tta': 'hflip2',
        'fixed_classwise_probability_weights': MODEL_WEIGHTS, 'inference_files': files,
        'patient_smoothing': False, 'selection_sha256': sha256_file(Path(comparison) / 'selection.json'),
        'comparison_sha256': sha256_file(Path(comparison) / 'report.json'),
        'official_valid_manifest_sha256': report['manifest_sha256'],
        'formal_T4_certification': False, 'deployment_accepted': False,
    }
    _write_json(model / 'config.json', configuration)
    sources = _copy_runtime_sources(output / 'code')
    if (ROOT / 'licenses').is_dir():
        shutil.copytree(ROOT / 'licenses', output / 'licenses')
    for name, function in [('predict.py', 'predict_main'), ('cost.py', 'cost_main')]:
        (output / name).write_text('from pathlib import Path\nimport sys\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))\n'
            f'from sprint.deploy import {function}\n'
            f'if __name__ == "__main__":\n    raise SystemExit({function}())\n')
    (output / 'requirements_runtime.txt').write_text(
        'timm==1.0.30\nhuggingface_hub==0.34.4\nsafetensors==0.8.0\ntqdm==4.67.1\n')
    (output / 'README.md').write_text(
        '# Sprint inference candidate\n\n'
        'Use the official competition image and install requirements_runtime.txt if needed. '
        'Its existing PyTorch, torchvision, NumPy, Pillow and scikit-learn are retained.\n\n'
        '`python predict.py --image-root ROOT --manifest TEST_MANIFEST --output NEW/submission.csv`\n\n'
        '`python cost.py submission.csv PICTURES_100 PICTURES_5 --manifest TEST_MANIFEST`\n\n'
        'Timing inputs contain 100 measured and five disjoint warmup complete Studies. '
        'Both commands use the identical full-image FP16 hflip2 and Study aggregation. '
        'The package is selected from local validation; official T4 timing and acceptance remain pending.\n')
    shutil.copyfile(Path(comparison) / 'report.json', output / 'validation_report.json')
    shutil.copyfile(Path(comparison) / 'selection.json', output / 'selection.json')
    inventory = inspect_package(model)
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > MAX_MODEL_BYTES:
        raise ValueError('Weights plus copied code/configuration exceed 500MB')
    receipt = {'state': 'packaged', 'model_inventory': inventory, 'source_files': sources,
               'package_bytes_before_predictions': total, 'size_pass': True,
               'formal_T4_certification': False, 'deployment_accepted': False,
               'selection_sha256': configuration['selection_sha256'],
               'comparison_sha256': configuration['comparison_sha256']}
    _write_json(output / 'package_receipt.json', receipt)
    return receipt


def inspect_package(model_dir):
    root = Path(model_dir).resolve(strict=True)
    config = json.loads((root / 'config.json').read_text())
    if (config.get('schema') != 'cxr_sprint_deployment_v1' or config.get('tta') != 'hflip2'
            or config.get('image_size') != 768 or config.get('patient_smoothing') is not False
            or config.get('fixed_classwise_probability_weights') != MODEL_WEIGHTS):
        raise ValueError('Unsupported or changed sprint deployment recipe')
    registered = config['inference_files']
    if len({item['path'] for item in registered}) != len(registered):
        raise ValueError('Duplicate inference inventory')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != {item['path'] for item in registered} | {'config.json'}:
        raise ValueError('Unexpected or missing model assets')
    for item in registered:
        path = root / item['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']):
            raise ValueError('Inference asset changed: ' + item['path'])
    total = sum((root / name).stat().st_size for name in actual)
    if total > MAX_MODEL_BYTES:
        raise ValueError('Actual inference files exceed 500MB')
    return {'total_bytes': total, 'limit_bytes': MAX_MODEL_BYTES, 'size_pass': True,
            'files': registered, 'config_sha256': sha256_file(root / 'config.json')}


class SprintPredictor:
    """Exact final-validation image math and complete-Study aggregation."""
    def __init__(self, model_dir):
        import torch
        from inference_reproducibility import configure_inference
        self.torch, self.device = torch, 'cuda'
        self.root = Path(model_dir).resolve(strict=True)
        self.inventory = inspect_package(self.root)
        self.config = json.loads((self.root / 'config.json').read_text())
        load_config = dict(self.config)
        for key in ('candidate_weights', 'tiny_patch_config'):
            load_config[key] = str(self.root / load_config[key])
        configure_inference(torch_module=torch)
        # sprint.train establishes this before invoking its sole evaluation in
        # the same process. Preserve the selected kernels in fresh processes.
        torch.backends.cudnn.deterministic = True
        self.tiny = _tiny(load_config)
        self.model, _, _ = _candidate(load_config, torch)

    def synchronize(self):
        self.torch.cuda.synchronize()

    def predict_image(self, path):
        from inference_inputs import load_image_pixels, member_results_to_cpu
        torch = self.torch
        pixels = load_image_pixels(path, [768], _letterbox_image)[768]
        with torch.inference_mode():
            image = self.tiny.input_lookup.to_tensor(pixels)
            views = [image, torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)]
            means, finite = [], []
            for member in (self.tiny.model, self.model):
                probabilities = []
                for view in views:
                    logits = (member(view) if member is self.tiny.model else
                              member(view, return_components=True)['image_logits'])
                    if tuple(logits.shape) != (1, 10):
                        raise ValueError('Require ten single-image logits')
                    finite.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                means.append(torch.stack(probabilities).mean(0))
            values = member_results_to_cpu(torch, means, finite)
        return aggregate_member_probabilities(values, MODEL_WEIGHTS)

    def predict_study(self, records):
        if len(study_groups(records)) != 1:
            raise ValueError('A prediction unit must contain exactly one complete Study')
        accumulator, per_image, result = StudyAccumulator(records), [], None
        for row in records:
            value = self.predict_image(row['path'])
            per_image.append((row, value))
            result = accumulator.add(row, value)
        if not accumulator.complete() or result is None:
            raise ValueError('Incomplete Study')
        return result, per_image


def _write_studies(predictor, records, output, *, timing=False):
    """Shared prediction/writing path; timing includes all views and CSV output."""
    output = Path(output)
    durations, results = [], {}
    with output.open('x', newline='', encoding='utf-8') as stream, \
         output.with_suffix('.image_predictions.csv').open('x', newline='', encoding='utf-8') as sidecar:
        writer, images = csv.writer(stream), csv.writer(sidecar)
        writer.writerow(CSV_HEADER)
        images.writerow(IMAGE_CSV_HEADER)
        stream.flush(); sidecar.flush()
        for group in study_groups(records):
            if timing:
                predictor.synchronize()
            started = time.perf_counter()
            result, per_image = predictor.predict_study(group)
            write_prediction(writer, *result)
            for row, value in per_image:
                write_image_prediction(images, row, value)
            stream.flush(); sidecar.flush()
            if timing:
                predictor.synchronize()
            durations.append((time.perf_counter() - started) * 1000)
            results[(result[0], result[1])] = result[2]
        os.fsync(stream.fileno()); os.fsync(sidecar.fileno())
    if len(results) != len(study_groups(records)):
        raise ValueError('Incomplete output coverage')
    return durations, results


def check_saved_validation(predictor, config, comparison, output):
    """Three fixed Studies check implementation only; never recompute AUC."""
    from cost import read_image_probabilities
    comparison = Path(comparison)
    report = json.loads((comparison / 'report.json').read_text())
    if sha256_file(config['valid_manifest']) != report['manifest_sha256']:
        raise ValueError('Original validation manifest changed')
    reference = read_image_probabilities(comparison / 'candidate.image_predictions.csv')
    records = prediction_records(config['image_root'], config['valid_manifest'], split='valid')
    groups = sorted(study_groups(records), key=lambda g: g[0]['study_key'])[:3]
    maximum, images = 0., 0
    for group in groups:
        _, per_image = predictor.predict_study(group)
        for row, value in per_image:
            expected = reference[(row['subject_id'], row['study_id'], row['image_id'])]
            maximum = max(maximum, float(np.max(np.abs(value - expected))))
            if (not np.allclose(value, expected, atol=1e-6, rtol=0)
                    or not np.array_equal(np.round(value, 4), np.round(expected, 4))):
                raise ValueError('Packaged predictor differs from the sole final validation')
            images += 1
    result = {'match': True, 'studies': len(groups), 'images': images, 'max_absolute_error': maximum,
              'selection': 'first three lexicographic original valid Studies; all images',
              'auc_evaluated': False, 'model_or_recipe_changed': False}
    _write_json(Path(output) / 'implementation_consistency.json', result)
    return result


def publish(config, *, generate_submission=False):
    comparison = Path(config['output']) / 'final_comparison'
    output = Path(config['output']) / 'deployment_candidate'
    selection, report = winning_selection(comparison)
    if not generate_submission:
        result = {
            'state': 'awaiting_user_instruction',
            'selected_candidate': selection['selected'],
            'candidate_macro_auc': report.get('candidate', {}).get('macro_auc'),
            'test_images_read': False,
            'submission_generated': False,
            'next_action': 'Run sprint.deploy with --generate-submission only after explicit user instruction.',
        }
        _write_json(Path(config['output']) / 'deployment_status.json', result)
        return result
    if output.exists():
        raise FileExistsError('Existing deployment candidate is retained without modification')
    try:
        package = prepare_package(comparison, output)
        predictor = SprintPredictor(output / 'model')
        consistency = check_saved_validation(predictor, config, comparison, output)
        manifest = Path(config.get('test_manifest',
            ROOT.parent / 'cloud_project_v2/data/clean_v1_reviewed/test_manifest.csv'))
        records = prediction_records(config['image_root'], manifest)
        if len(records) != 2218 or len(study_groups(records)) != 1414:
            raise ValueError('Unexpected original test coverage; retain candidate for review')
        submission = output / 'submission.csv'
        _, results = _write_studies(predictor, records, submission)
        if set(read_submission(submission)) != set(results):
            raise ValueError('Published submission identities do not match complete test coverage')
        result = {'state': 'candidate_ready', 'submission': str(submission),
            'submission_sha256': sha256_file(submission), 'images': len(records),
            'studies': len(results), 'data_rows': len(results) * 10,
            'manifest_sha256': sha256_file(manifest), 'image_predictions_sha256':
            sha256_file(submission.with_suffix('.image_predictions.csv')),
            'package': package, 'implementation_consistency': consistency,
            'test_used_for_training_or_selection': False, 'test_labels_read': False,
            'baseline_submission_preserved': True, 'formal_T4_certification': False,
            'deployment_accepted': False, 'pending': 'Official T4 mean <=100ms/Study and full acceptance',
            'predict_entry': str(output / 'predict.py'), 'cost_entry': str(output / 'cost.py')}
        result['package_files'] = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                                   'sha256': sha256_file(path)}
                                  for path in sorted(output.rglob('*')) if path.is_file()]
        result['package_bytes_including_predictions'] = sum(item['bytes'] for item in result['package_files'])
        if result['package_bytes_including_predictions'] > MAX_MODEL_BYTES:
            raise ValueError('Complete candidate package exceeds 500MB')
        _write_json(output / 'deployment_status.json', result)
        _write_json(Path(config['output']) / 'selected_current.json', {
            'selected': 'sprint_candidate', 'submission': str(submission),
            'submission_sha256': result['submission_sha256'], 'model_directory': str(output / 'model'),
            'candidate_ready': True, 'deployment_accepted': False, 'formal_T4_certification': False,
            'original_submission_best_preserved': True})
        return result
    except BaseException as error:
        if output.is_dir():
            _write_json(output / 'deployment_status.json', {'state': 'failed',
                'error': f'{type(error).__name__}: {error}', 'deployment_accepted': False,
                'baseline_submission_preserved': True})
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
        raise ValueError('Prediction output must not modify original images or model assets')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = prediction_records(args.image_root, args.manifest)
    predictor = SprintPredictor(args.model_dir)
    _write_studies(predictor, records, args.output)
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
    if {r['study_key'] for r in measured} & {r['study_key'] for r in warmup}:
        raise ValueError('Warmup and measured Studies must be disjoint')
    for records in (measured, warmup):
        if not match_timing_manifest(records, args.manifest)['complete_studies_proven']:
            raise ValueError('Timing requires every original image of each selected Study')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictor = SprintPredictor(args.model_dir)
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
                raise ValueError('Timed Study missing from submission')
            max_error = max(max_error, float(np.max(np.abs(value - reference[key]))))
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
    parser.add_argument('--generate-submission', action='store_true',
                        help='Explicitly authorize reading the official test images and writing a new CSV')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps({'output': str(Path(config['output']) / 'deployment_candidate'),
                          'requires': 'completed winning sole final comparison', 'test_images_read': False}))
        return
    print(json.dumps(publish(config, generate_submission=args.generate_submission), indent=2))


if __name__ == '__main__':
    main()
