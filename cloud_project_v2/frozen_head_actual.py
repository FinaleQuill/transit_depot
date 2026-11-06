"""Actual FP16 validation/serial timing of one preregistered frozen-head pair.

Large temporary merged exports remain in /tmp. This program opens only valid
images, uses fixed hflip2/full decode, and never selects or predicts on test.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

from deploy_kd import CLASS_IDS, MAX_MODEL_BYTES, read_submission, sha256_file
from frozen_head_predictor import FrozenHeadPredictor, export_merged_for_validation
from frozen_head_evaluate import validate_plan
from student_experiment_evaluation import ExperimentalPredictor, evaluate_predictor, write_new_json
from student_ensemble import StudentEnsemble


ROOT = Path(__file__).resolve().parent
VALID_MANIFEST = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
IMAGE_ROOT = Path('/home/featurize/data/cxr_first')
TIMING_INPUTS = ROOT.parent / 'deliveries/034_dual768_proxy_timing_20261029/bundle/inputs'
MAX_SECONDS = 1800


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def registered_pair(plan, pair_name):
    validate_plan(plan)
    _require(pair_name in plan['pair_candidates'], 'Actual validation accepts only a registered pair name')
    declaration = plan['pair_candidates'][pair_name]
    _require(len(declaration['members']) == 2 and len(set(declaration['members'])) == 2,
             'Actual validation requires two distinct registered members')
    return declaration


def prepare_pair(*, plan, plan_sha256, fits_root, pair_name, output, cache_metadata_sha256):
    """CPU-only construction of standard temporary exports and immutable sources."""
    _require(sha256_file(plan) == plan_sha256, 'Registered plan SHA256 changed')
    plan_value = _read(plan)
    declaration = registered_pair(plan_value, pair_name)
    output = Path(output).absolute()
    _require(output.resolve().is_relative_to(Path('/tmp').resolve()), 'Actual models must remain under /tmp')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Actual validation needs a new local output directory')
    fits_root = Path(fits_root).resolve(strict=True)
    output.mkdir(parents=True)
    (output / 'merged').mkdir()
    model_dir = output / 'model'
    model_dir.mkdir()
    members, source_records = [], {}
    for name in declaration['members']:
        encoder = name.split('_')[0]
        base = plan_value['encoders'][encoder]
        original = Path(base['base_model_dir']).resolve(strict=True)
        _require(sha256_file(original / 'config.json') == base['base_config_sha256'], 'Original encoder config changed')
        original_config = _read(original / 'config.json')
        _require(original_config['weights_sha256'] == base['base_weights_sha256']
            and sha256_file(original / original_config['weights_file']) == base['base_weights_sha256'],
            'Original encoder weight identity changed')
        if name in ('A', 'B'):
            source = original
            identity = {'kind': 'original_frozen_encoder', **base}
        else:
            source = fits_root / name
            config_path = source / 'config.json'
            cfg = _read(config_path)
            _require(source.resolve(strict=True).parent == fits_root, 'Fitted member escaped fits-root')
            _require(cfg.get('backbone_name') == base['backbone_name']
                and cfg.get('base_config_sha256') == base['base_config_sha256']
                and cfg.get('base_weights_sha256') == base['base_weights_sha256']
                and cfg.get('feature_dim') == base['feature_dim']
                and cfg.get('replaced_class_ids') == CLASS_IDS,
                'Fitted head changed its registered original encoder or class coverage')
            _require(cfg.get('training_plan_sha256') == plan_sha256
                and cfg.get('feature_cache_metadata_sha256') == cache_metadata_sha256,
                'Fitted head config does not bind this registered plan and feature cache')
            receipt_path = source / 'training_receipt.json'
            training = _read(receipt_path)
            _require(training.get('state') == 'completed' and training.get('encoder') == encoder
                and training.get('recipe') == name.split('_')[1]
                and training.get('training_plan_sha256') == plan_sha256
                and training.get('feature_cache_metadata_sha256') == cache_metadata_sha256
                and training.get('final_epoch') == 12 and training.get('completed_epochs') == 12
                and training.get('head_patch_fp16_sha256') == cfg.get('patch_sha256')
                and training.get('valid_used_for_training') is False
                and training.get('valid_used_for_checkpoint_selection') is False
                and training.get('test_used') is False,
                'Fitted head lacks the fixed-final train-only completion receipt')
            merged_dir = output / 'merged' / name
            merger = export_merged_for_validation(config_path, merged_dir)
            identity = {'kind': 'fitted_frozen_head', 'patch_model_dir': str(source),
                'patch_config_sha256': sha256_file(config_path),
                'patch_sha256': cfg['patch_sha256'],
                'training_receipt_sha256': sha256_file(receipt_path),
                'training_plan_sha256': plan_sha256, 'feature_cache_metadata_sha256': cache_metadata_sha256,
                'base_model_dir': str(original), 'base_config_sha256': base['base_config_sha256'],
                'base_weights_sha256': base['base_weights_sha256'], 'merge_receipt': merger}
            source = merged_dir
        source_config = _read(source / 'config.json')
        destination = model_dir / name
        destination.mkdir()
        for filename in ('config.json', source_config['weights_file']):
            _require(Path(filename).name == filename, 'Member export contains a nonlocal inference filename')
            shutil.copyfile(source / filename, destination / filename)
            _require(sha256_file(source / filename) == sha256_file(destination / filename), 'Temporary model copy changed bytes')
        members.append({'directory': name, 'backbone': base['backbone_name'], 'tta': 'hflip2',
            'config_sha256': sha256_file(destination / 'config.json'),
            'weights_sha256': source_config['weights_sha256']})
        source_records[name] = {**identity, 'actual_member_config_sha256': members[-1]['config_sha256'],
            'actual_member_weights_sha256': members[-1]['weights_sha256'],
            'actual_member_dir': str(destination)}
    config = {'schema_version': 1, 'artifact_type': 'fixed_student_ensemble',
        'class_ids': CLASS_IDS, 'precision': 'float16', 'batch_size': 1, 'image_size': 768,
        'tta': 'hflip2', 'ensemble': True, 'gates': None, 'decoder_candidate': None,
        'members': members, 'model_weights': declaration['weights'], 'rule_name': pair_name,
        'selection_plan_sha256': plan_sha256,
        'official_valid_manifest_sha256': plan_value['cache']['valid_manifest_sha256'],
        'test_used_for_selection': False, 'formal_T4_certification': False,
        'aggregation': 'fixed image class probability fusion after member hflip2; original Study FP64 mean'}
    write_new_json(model_dir / 'config.json', config)
    total = sum(path.stat().st_size for path in model_dir.rglob('*') if path.is_file())
    _require(total <= MAX_MODEL_BYTES, 'Actual ensemble exceeds 500MB of inference assets')
    _require(sha256_file(plan) == plan_sha256, 'Registered plan changed during model preparation')
    write_new_json(output / 'member_sources.json', source_records)
    return model_dir, config, source_records


def probability_identity(actual, expected):
    """Require shape, dtype and byte identity, including all ten class values."""
    left, right = np.asarray(actual), np.asarray(expected)
    return (left.shape == (10,) and right.shape == (10,) and left.dtype == right.dtype
            and left.tobytes() == right.tobytes())


def overlay_merged_probe(source_records, records, image_root):
    """Five valid source images for each new head, with only one loader resident."""
    import torch
    checks = {}
    for name, identity in source_records.items():
        if identity['kind'] != 'fitted_frozen_head':
            continue
        paths = [Path(image_root) / row['relative_path'] for row in records[:5]]
        overlay = FrozenHeadPredictor(identity['patch_model_dir'])
        probabilities = [overlay.predict_image(path) for path in paths]
        del overlay
        gc.collect()
        torch.cuda.empty_cache()
        merged = ExperimentalPredictor(identity['actual_member_dir'], tta='hflip2')
        rows = []
        for row, path, expected in zip(records[:5], paths, probabilities):
            actual = merged.predict_image(path)
            _require(probability_identity(actual, expected), 'Overlay/merged FP16 probabilities differ: ' + name)
            rows.append({'image_id': row['image_id'], 'all_ten_probabilities_bitwise_equal': True,
                'max_absolute_error': float(np.max(np.abs(actual - expected)))})
        checks[name] = {'num_valid_images': 5, 'all_image_probabilities_bitwise_equal': True, 'images': rows}
        del merged
        gc.collect()
        torch.cuda.empty_cache()
    return checks


def copy_timing_inputs(source, output):
    """Copy the immutable official-development 100+5 selection onto local /tmp."""
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    _require(output.resolve().is_relative_to(Path('/tmp').resolve()), 'Timing images must remain on local /tmp')
    if output.exists():
        raise FileExistsError('Timing input copy requires a new directory')
    output.mkdir(parents=True)
    for name in ('pictures_100', 'pictures_5'):
        shutil.copytree(source / name, output / name)
    selection = 'selection.portable.json' if (source / 'selection.portable.json').is_file() else 'selection.json'
    shutil.copyfile(source / selection, output / selection)
    _require(sha256_file(source / selection) == sha256_file(output / selection), 'Timing selection copy changed')
    return output


def audit_complete_valid(directory, records, reported):
    """Rebuild all original Study means independently from saved image rows."""
    from audit_student_submission import IMAGE_HEADER, SUBMISSION_HEADER, read_probabilities
    from cost import compare_predictions
    from student_experiment_evaluation import validation_labels, score_probabilities
    directory = Path(directory)
    identities = {r['image_id']: (r['subject_id'], r['study_id']) for r in records}
    _require(len(identities) == len(records) and all(r['split'] == 'valid' for r in records),
             'Independent audit requires unique original valid identities')
    images, image_rows = read_probabilities(directory / 'valid.image_predictions.csv',
        IMAGE_HEADER, identities, image_level=True)
    studies, study_rows = read_probabilities(directory / 'valid_predictions.csv',
        SUBMISSION_HEADER, set(identities.values()), image_level=False)
    sums, counts = {}, {}
    for row in records:
        key = identities[row['image_id']]
        if key not in sums:
            sums[key], counts[key] = np.zeros(10, dtype=np.float64), 0
        sums[key] += images[row['image_id']]
        counts[key] += 1
    means = {key: value / counts[key] for key, value in sums.items()}
    comparison = compare_predictions(studies, means, atol=0)
    _require(comparison['match'] and comparison['max_absolute_error'] == 0,
             'Complete valid Study outputs differ from independent original-order FP64 image means')
    labels, views = validation_labels(records)
    rescored = score_probabilities(labels, studies, views)
    _require(abs(reported['macro_auc'] - rescored['macro_auc']) <= 1e-12
        and np.array_equal(np.asarray(reported['per_class_auc']), np.asarray(rescored['per_class_auc'])),
        'Complete valid AUC differs from independently reloaded probabilities')
    return {'complete': True, 'num_images': len(images), 'num_studies': len(studies),
        'image_prediction_rows': image_rows, 'study_prediction_rows': study_rows,
        'class_coverage_per_image_and_Study': 10,
        'all_Study_means_bitwise_equal_to_original_order_FP64_replay': True,
        'study_mean_binary_and_rank_comparison': comparison,
        'macro_auc_recomputed': rescored['macro_auc'], 'per_class_auc_recomputed': rescored['per_class_auc'],
        'test_used': False}


def persist_reports(output, report_dir):
    """Copy only explicit small evidence, never temporary weight/config trees."""
    output, report_dir = Path(output), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = [path for directory in ('valid', 'timing') for path in (output / directory).rglob('*')
             if path.is_file() and path.suffix.lower() in ('.json', '.csv', '.log')]
    paths += [output / name for name in ('actual_audit.json', 'overlay_merged_probe.json', 'full_valid_independent_audit.json',
                                       'member_sources.json', 'failure.json') if (output / name).is_file()]
    for source in paths:
        _require(source.stat().st_size <= 15 * 1024**2, 'Unexpectedly large evidence file; refusing persistent copy')
        destination = report_dir / source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _require(sha256_file(destination) == sha256_file(source), 'Existing evidence differs; refusing overwrite')
        else:
            shutil.copyfile(source, destination)
    top = output / 'model/config.json'
    if top.is_file():
        destination = report_dir / 'model_top_config.json'
        if destination.exists():
            _require(sha256_file(destination) == sha256_file(top), 'Existing top model evidence differs')
        else:
            shutil.copyfile(top, destination)


def actual(*, plan, plan_sha256, fits_root, pair_name, output, report_dir, cache):
    from dataset import read_manifest
    from benchmark_local_resources import validate_inputs
    from cost import compare_predictions, compare_image_output, read_image_probabilities
    from inference_reproducibility import configure_inference
    from measure_ensemble_timing import measure
    import torch

    started = time.monotonic()
    output, report_dir = Path(output).absolute(), Path(report_dir).absolute()
    _require(not report_dir.exists(), 'Actual report directory must be new')
    cache = Path(cache).resolve(strict=True)
    _require(not output.resolve().is_relative_to(cache)
        and not output.resolve().is_relative_to(Path(fits_root).resolve(strict=True)),
        'Actual temporary models cannot be written inside immutable cache/fit inputs')
    cache_metadata_path = cache / 'meta.json'
    metadata_sha = sha256_file(cache_metadata_path)
    metadata = _read(cache_metadata_path)
    _require(metadata.get('complete') is True and metadata.get('schema') == 'frozen_fc_cache_v1'
        and metadata.get('plan_sha256') == plan_sha256 and metadata.get('test_used') is False,
        'Require the completed immutable train/valid cache from this registered plan')
    registration = _read(plan)
    _require(sha256_file(VALID_MANIFEST) == registration['cache']['valid_manifest_sha256'],
             'Original valid manifest changed')
    _require(metadata['splits']['valid']['manifest_sha256'] == registration['cache']['valid_manifest_sha256']
        and metadata['splits']['valid']['num_images'] == 1422, 'Cache valid identity or image coverage differs')
    records = read_manifest(VALID_MANIFEST, split='valid')
    _require(len(records) == 1422, 'Actual validation must retain all original valid images')
    _require(not report_dir.is_relative_to(cache) and not report_dir.is_relative_to(Path(fits_root).resolve()),
             'Actual reports cannot overwrite immutable cache/fit inputs')

    def budget():
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError('Actual frozen-head evaluation exceeded its 30-minute budget')

    try:
        model_dir, config, sources = prepare_pair(plan=plan, plan_sha256=plan_sha256,
            fits_root=fits_root, pair_name=pair_name, output=output, cache_metadata_sha256=metadata_sha)
        for row in records:
            _require(sha256_file(IMAGE_ROOT / row['relative_path']) == row['file_sha256'],
                     'Original validation image SHA256 changed: ' + row['image_id'])
        budget()
        configure_inference()
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        probes = overlay_merged_probe(sources, records, IMAGE_ROOT)
        write_new_json(output / 'overlay_merged_probe.json', probes)
        budget()
        predictor = StudentEnsemble(model_dir)
        result = evaluate_predictor(predictor, image_root=IMAGE_ROOT, manifest=VALID_MANIFEST,
                                    output_dir=output / 'valid')
        independent = audit_complete_valid(output / 'valid', records, result)
        write_new_json(output / 'full_valid_independent_audit.json', independent)
        inventory = predictor.inventory
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        budget()
        local_inputs = copy_timing_inputs(TIMING_INPUTS, output / 'timing_inputs')
        timing = measure(model_dir=model_dir, inputs=local_inputs, manifest=VALID_MANIFEST,
                         output_dir=output / 'timing')
        measured, _, coverage = validate_inputs(local_inputs / 'pictures_100', local_inputs / 'pictures_5',
                                               output / 'timing/selection.relocated.json')
        reference = read_submission(output / 'valid/valid_predictions.csv')
        reference_images = read_image_probabilities(output / 'valid/valid.image_predictions.csv')
        canonical_ids = {(r['subject_id'], r['study_id'], Path(r['relative_path']).name): r['image_id'] for r in records}
        comparisons = []
        for index in range(1, 4):
            csv_path = output / f'timing/round_{index}.csv'
            study = compare_predictions(read_submission(csv_path), reference, atol=0,
                allow_reference_superset=True,
                expected_keys={(r['subject_id'], r['study_id']) for r in measured})
            image = compare_image_output(measured, csv_path.with_suffix('.image_predictions.csv'),
                                         reference_images, canonical_ids)
            _require(study['match'] and image['match'] and image['max_absolute_error'] == 0,
                     'Timed image/Study probabilities, binary decisions or order differ from complete valid')
            comparisons.append({'round': index, 'study': study, 'image': image})
        budget()
        _require(sha256_file(plan) == plan_sha256 and sha256_file(cache_metadata_path) == metadata_sha,
                 'Registered plan/cache identity changed during actual validation')
        audit = {'complete': True, 'pair_name': pair_name, 'candidate': pair_name,
            'macro_auc': result['macro_auc'], 'per_class_auc': result['per_class_auc'], 'slices': result['slices'],
            'mean_ms': timing['mean_ms'], 'round_mean_ms': timing['round_mean_ms'],
            'size_bytes': inventory['total_bytes'], 'size_pass': inventory['size_pass'],
            'latency_pass_100ms': timing['mean_ms'] <= 100, 'preferred_latency_pass_80ms': timing['mean_ms'] <= 80,
            'configuration_sha256': sha256_file(model_dir / 'config.json'),
            'selection_plan_sha256': plan_sha256, 'feature_cache_metadata_sha256': metadata_sha,
            'full_valid_manifest_sha256': sha256_file(VALID_MANIFEST),
            'original_image_hashes_match': True, 'num_images': result['num_images'],
            'num_studies': result['num_studies'], 'decoder': None, 'tta': 'hflip2', 'precision': 'float16',
            'overlay_vs_merged': probes, 'timing_vs_complete_valid': comparisons,
            'complete_valid_independent_audit': independent,
            'all_timed_images_and_Studies_bitwise_equal': True, 'timing_coverage': coverage,
            'cpu_proxy_equality_required': False,
            'cpu_proxy_note': 'CPU FP32 classifier proxy is not actual FP16 encoder/head inference.',
            'member_sources': sources, 'model_dir': str(model_dir), 'seconds': time.monotonic() - started,
            'test_used': False, 'new_training_started': False, 'T4_certified': False,
            'development_diagnostic_only': True, 'memory_allocator_limit_gib': 8,
            'source_sha256': sha256_file(__file__)}
        write_new_json(output / 'actual_audit.json', audit)
        persist_reports(output, report_dir)
        return audit
    except BaseException as error:
        if output.is_dir():
            failure = {'complete': False, 'pair_name': pair_name, 'error': repr(error),
                'seconds': time.monotonic() - started, 'test_used': False, 'T4_certified': False}
            path = output / 'failure.json'
            if not path.exists():
                write_new_json(path, failure)
            persist_reports(output, report_dir)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'fits_root', 'output', 'report_dir', 'cache'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--pair-name', required=True)
    result = actual(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ('pair_name', 'macro_auc', 'mean_ms', 'size_bytes')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
