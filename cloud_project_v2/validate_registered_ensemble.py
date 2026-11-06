"""Actual FP16 valid/latency audit of a registered fixed ensemble; no test use."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from audit_student_submission import IMAGE_HEADER, read_probabilities
from benchmark_local_resources import validate_inputs
from cost import compare_image_output, compare_predictions, read_image_probabilities
from dataset import read_manifest
from deploy_kd import read_submission, sha256_file
from evaluate_finetune_candidates import load_validation, normalized_weights, aggregate_images
from inference_reproducibility import configure_inference
from measure_ensemble_timing import measure
from student_experiment_evaluation import evaluate_predictor, validation_labels, write_new_json
from submission_runtime import load_submission_predictor


def validate(*, model_dir, image_root, manifest, inputs, comparison, candidate, output):
    import torch
    model_dir, output = Path(model_dir), Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh actual-validation directory')
    config = json.loads((model_dir / 'config.json').read_text())
    recorded = json.loads(Path(comparison).read_text())
    if (config.get('rule_name') != candidate or config.get('selection_plan_sha256') !=
            recorded['registered_plan_sha256'] or sha256_file(manifest) != recorded['manifest_sha256']):
        raise ValueError('Actual configuration is not the registered candidate/valid manifest')
    records = read_manifest(manifest, split='valid')
    labels, views = validation_labels(records)
    if any(sha256_file(Path(image_root) / row['relative_path']) != row['file_sha256'] for row in records):
        raise ValueError('Original validation image SHA256 changed')
    output.mkdir(parents=True)
    names = [member['directory'] for member in config['members']]
    sources, source_aliases = {}, {}
    for name in names:
        directory = Path(recorded['source_validations'][name]['validation_dir'])
        images, _, _, _ = load_validation(directory, records, recorded['manifest_sha256'], labels, views)
        actual_config_hash = sha256_file(model_dir / name / 'config.json')
        if recorded['source_validations'][name]['deployment_config_sha256'] != actual_config_hash:
            # B was restored from the same persistent EMA in the previous
            # session, adding restoration metadata to its config. Bind that
            # config to its actual complete reproduction, then require every
            # probability to equal the historical reference used in this plan.
            alias = Path(__file__).resolve().parent / 'artifacts/kd_v1/student_B_submission_20261031/full_valid_reproduced'
            alias_report = json.loads((alias / 'validation_report.json').read_text()) if name == 'B' else {}
            if name != 'B' or alias_report.get('deployment_config_sha256') != actual_config_hash:
                raise ValueError('An actual member differs from its cached validation identity')
            restored, _, _, identity = load_validation(alias, records, recorded['manifest_sha256'], labels, views)
            if any(not np.array_equal(images[key], restored[key]) for key in images):
                raise ValueError('Restored B probabilities differ from the historical source')
            source_aliases[name] = {'restored_validation': identity,
                'all_image_probabilities_equal_to_historical_source': True,
                'actual_config_sha256': actual_config_hash}
        sources[name] = images
    configure_inference()
    fraction = 8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(fraction)
    predictor = load_submission_predictor(model_dir)
    result = evaluate_predictor(predictor, image_root=Path(image_root), manifest=Path(manifest),
                                 output_dir=output / 'valid')
    inventory = predictor.inventory
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    exact_cache = config.get('decoder_candidate') is None
    cache_report = {'required': exact_cache, 'decoder_changes_pixels': not exact_cache,
                    'restored_source_aliases': source_aliases}
    if exact_cache:
        weights = normalized_weights(config['model_weights'], len(names))
        expected_images = {row['image_id']: sum((weights[i] * sources[name][row['image_id']]
            for i, name in enumerate(names)), np.zeros(10, dtype=np.float64)) for row in records}
        expected_studies = aggregate_images(records, expected_images)
        actual_studies = read_submission(output / 'valid/valid_predictions.csv')
        identities = {row['image_id']: (row['subject_id'], row['study_id']) for row in records}
        actual_images, _ = read_probabilities(output / 'valid/valid.image_predictions.csv',
            IMAGE_HEADER, identities, image_level=True)
        key = lambda row: (row['subject_id'], row['study_id'] + '/' + row['image_id'])
        image_check = compare_predictions({key(row): actual_images[row['image_id']] for row in records},
            {key(row): expected_images[row['image_id']] for row in records}, atol=0)
        study_check = compare_predictions(actual_studies, expected_studies, atol=0)
        if (not image_check['match'] or not study_check['match'] or
                abs(result['macro_auc'] - recorded['candidates'][candidate]['macro_auc']) > 1e-12):
            raise RuntimeError('Actual combined output differs from cached fixed image fusion')
        cache_report.update(image=image_check, study=study_check, all_outputs_bitwise_equal=True)
    write_new_json(output / 'cached_fusion_audit.json', cache_report)
    timing = measure(model_dir=model_dir, inputs=Path(inputs), manifest=Path(manifest),
                     output_dir=output / 'timing')
    selection = output / 'timing/selection.relocated.json'
    measured, _, _ = validate_inputs(Path(inputs) / 'pictures_100', Path(inputs) / 'pictures_5', selection)
    reference = read_submission(output / 'valid/valid_predictions.csv')
    reference_images = read_image_probabilities(output / 'valid/valid.image_predictions.csv')
    canonical_ids = {(row['subject_id'], row['study_id'], Path(row['relative_path']).name): row['image_id']
                     for row in records}
    comparisons = []
    for index in range(1, 4):
        csv = output / f'timing/round_{index}.csv'
        actual = read_submission(csv)
        study = compare_predictions(actual, reference, atol=0, allow_reference_superset=True,
            expected_keys={(r['subject_id'], r['study_id']) for r in measured})
        image = compare_image_output(measured, csv.with_suffix('.image_predictions.csv'),
                                     reference_images, canonical_ids)
        if not study['match'] or not image['match'] or image['max_absolute_error'] != 0:
            raise RuntimeError('Timed image/Study probabilities or order differ from complete valid')
        comparisons.append({'round': index, 'study': study, 'image': image})
    audit = {'complete': True, 'macro_auc': result['macro_auc'], 'per_class_auc': result['per_class_auc'],
        'slices': result['slices'], 'mean_ms': timing['mean_ms'], 'round_mean_ms': timing['round_mean_ms'],
        'candidate': candidate, 'decoder': config.get('decoder_candidate'),
        'size_bytes': inventory['total_bytes'], 'size_pass': inventory['size_pass'],
        'configuration_sha256': sha256_file(model_dir / 'config.json'),
        'full_valid_manifest_sha256': sha256_file(manifest), 'original_image_hashes_match': True,
        'cached_full_decode_fusion_match': exact_cache, 'timing_vs_complete_valid': comparisons,
        'all_timed_images_and_Studies_bitwise_equal': True, 'test_used': False,
        'new_training_started': False, 'T4_certified': False,
        'source_sha256': sha256_file(__file__), 'memory_allocator_limit_gib': 8}
    write_new_json(output / 'actual_audit.json', audit)
    print(json.dumps({k: audit[k] for k in ('macro_auc', 'mean_ms', 'round_mean_ms', 'size_bytes')}, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model_dir', 'image_root', 'manifest', 'inputs', 'comparison', 'candidate', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=str if name == 'candidate' else Path, required=True)
    validate(**vars(parser.parse_args()))
