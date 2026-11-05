"""Finish a report-write failure from authenticated, complete saved predictions.

No image decoding, model inference, fitting, or alternative candidate evaluation.
The original failed controller records and partial report remain archived.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np

from deploy_kd import prediction_records, read_submission, sha256_file
from robust_ensemble.deploy import selected_assets
from robust_ensemble.evaluate import (image_cache, selection_checks, sensitivity,
                                      uncertainty, verify_inputs)
from sprint.evaluate import _write_json
from student_experiment_evaluation import score_probabilities, validation_labels


def recover(config_path):
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    root = Path(config['output'])
    output = root / 'final_comparison'
    if (output / 'report.json').exists() or (output / 'selection.json').exists():
        raise FileExistsError('Do not repeat or replace a completed comparison')
    failure = json.loads((output / 'failure.json').read_text())
    progress = json.loads((output / 'progress.json').read_text())
    controller = json.loads((root / 'controller_status.json').read_text())
    if (failure['error'] != 'TypeError: Object of type bool_ is not JSON serializable'
            or controller['state'] != 'failed'
            or progress.get('images_done') != 1422 or progress.get('images_total') != 1422):
        raise ValueError('Only the completed-inference JSON report failure is recoverable here')
    if sha256_file(config_path) != controller['plan_sha256']:
        raise ValueError('Frozen plan changed')
    protocol = json.loads((output / 'evaluation_protocol.json').read_text())
    if protocol['configuration'] != config:
        raise ValueError('Original evaluation used a different configuration')
    verify_inputs(config)
    archive = root / 'evidence/report_recovery'
    archive.mkdir(parents=True, exist_ok=False)
    for source in (root / 'controller_status.json', output / 'failure.json',
                   output / 'progress.json', output / 'report.json.tmp'):
        shutil.copyfile(source, archive / source.name)
    source_identity = json.loads((root / 'source_identity.json').read_text())
    source_root = Path(__file__).resolve().parents[1]
    changed = [name for name, digest in source_identity.items()
               if sha256_file(source_root / name) != digest]
    if changed != ['robust_ensemble/evaluate.py']:
        raise ValueError('Unexpected changes to the frozen runtime: ' + repr(changed))

    records = prediction_records(config['image_root'], config['valid_manifest'], split='valid')
    labels, counts = validation_labels(records)
    if (len(records), len(labels), len({k[0] for k in labels})) != (1422, 859, 228):
        raise ValueError('Original valid coverage changed')
    names = ('candidate', 'sprint', 'opacity')
    files = [output / (name + suffix) for name in names
             for suffix in ('_predictions.csv', '.image_predictions.csv')]
    before = {str(path): sha256_file(path) for path in files}
    predictions = {name: read_submission(output / (name + '_predictions.csv')) for name in names}
    images = {name: image_cache(output / (name + '.image_predictions.csv'), records) for name in names}
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['subject_id'], row['study_id'])].append(row['image_id'])
    for name in names:
        if set(predictions[name]) != set(labels):
            raise ValueError('Incomplete Study identity coverage')
        for key, image_ids in grouped.items():
            mean = np.asarray([images[name][image] for image in image_ids]).mean(0)
            if not np.array_equal(mean, predictions[name][key]):
                raise ValueError('Stored image-to-Study mean changed')
    cache_difference = []
    for name in ('sprint', 'opacity'):
        cache = image_cache(config['cached_' + name + '_image'], records)
        difference = float(max(np.abs(images[name][key] - cache[key]).max() for key in cache))
        if difference > 1e-7:
            raise ValueError('Stored parent differs from its frozen prediction')
        cache_difference.append(difference)
    weights = np.asarray(config['fusion_weights'])
    for key in images['candidate']:
        expected = (weights * np.stack([images[name][key] for name in ('sprint', 'opacity')])).sum(0)
        if not np.array_equal(images['candidate'][key], expected):
            raise ValueError('Stored candidate differs from the single frozen fusion')

    candidate = predictions['candidate']
    reference = read_submission(config['cached_sprint_study'])
    new_score, old_score = [score_probabilities(labels, p, counts) for p in (candidate, reference)]
    if abs(old_score['macro_auc'] - .8179628779375786) > 1e-12:
        raise ValueError('Reference score changed')
    new_sensitive, old_sensitive = [sensitivity(labels, p, counts, records) for p in (candidate, reference)]
    class1_difference = float(max(abs(candidate[k][0] - reference[k][0]) for k in labels))
    checks = selection_checks(new_score, old_score, new_sensitive, old_sensitive, class1_difference)
    bootstrap = uncertainty(labels, candidate, reference, counts,
                            samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
    assets = selected_assets(config)
    recipe_bytes = len(json.dumps(config, ensure_ascii=False).encode('utf-8'))
    total_bytes = sum(item['bytes'] for item in assets) + recipe_bytes
    inventory = {'files': assets, 'configuration_bytes': recipe_bytes, 'total_bytes': total_bytes,
                 'limit_bytes': 500_000_000, 'size_pass': total_bytes <= 500_000_000,
                 'scope': 'unique loaded weights, Tiny patch/configurations and serialized recipe'}
    verify_inputs(config)
    if any(sha256_file(path) != digest for path, digest in before.items()):
        raise ValueError('Saved predictions changed during CPU recovery')
    receipt = {'method': 'CPU scoring of the original complete saved validation predictions',
               'recovered_at': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
               'original_failure': failure, 'original_controller_status_preserved': True,
               'original_records_archive': str(archive), 'new_model_inferences': 0,
               'new_candidates': 0, 'prediction_hashes_unchanged': before,
               'source_changes': changed, 'recovery_script_sha256': sha256_file(__file__),
               'scoring_source_sha256': sha256_file(source_root / 'robust_ensemble/evaluate.py'),
               'elapsed_seconds': time.monotonic() - started}
    report = {'complete': True, 'candidate_name': 'Preserved_Sprint80_Opacity512_20',
        'candidate': new_score, 'baseline': old_score,
        'macro_auc_delta': new_score['macro_auc'] - old_score['macro_auc'],
        'per_class_auc_delta': (np.array(new_score['per_class_auc']) - old_score['per_class_auc']).tolist(),
        'class8_auc_delta': new_score['per_class_auc'][7] - old_score['per_class_auc'][7],
        'class1_max_abs_probability_difference': class1_difference,
        'candidate_sensitivity': new_sensitive, 'baseline_sensitivity': old_sensitive,
        'bootstrap': bootstrap, 'selection_checks': checks, 'candidate_won': all(checks.values()),
        'fusion_weights': config['fusion_weights'], 'opacity_weight_per_class': config['opacity_weight_per_class'],
        'candidate_inference_assets': assets, 'candidate_inference_bytes': total_bytes,
        'asset_inventory': inventory, 'size_pass': inventory['size_pass'],
        'manifest_sha256': config['valid_manifest_sha256'],
        'candidate_predictions_sha256': sha256_file(output / 'candidate_predictions.csv'),
        'candidate_image_predictions_sha256': sha256_file(output / 'candidate.image_predictions.csv'),
        'parent_image_max_abs_cache_difference': cache_difference,
        'parent_scores_diagnostic_only': {name: score_probabilities(labels, predictions[name], counts)
                                         for name in ('sprint', 'opacity')},
        'validation_elapsed_seconds': progress['elapsed_seconds'],
        'validation_time_scope': 'original progress timestamp at image 1422; excludes final flush/statistics',
        'original_failed_execution_elapsed_seconds': failure['elapsed_seconds'],
        'report_recovery': receipt, 'validation_images': len(records),
        'new_training_runs': 0, 'new_optimizer_updates': 0, 'candidate_count': 1,
        'alternative_fusions_evaluated': 0, 'test_images_used': False,
        'formal_T4_certification': False, 'deployment_accepted': False,
        'selection_note': 'Local robustness gates only; neither independent generalization nor platform improvement has been established.'}
    selection = {'selected': 'robust_ensemble' if report['candidate_won'] else 'sprint_candidate',
        'candidate_won': report['candidate_won'], 'selection_checks': checks,
        'fusion_weights': config['fusion_weights'], 'opacity_weight_per_class': config['opacity_weight_per_class'],
        'report': str(output / 'report.json'), 'tta': 'hflip2', 'research_selection_only': True,
        'report_recovered_from_complete_predictions': True}
    # Serialize the complete documents before any final result file is created.
    for payload in (report, selection, receipt):
        json.dumps(payload, allow_nan=False)
    _write_json(output / 'report.json', report)
    _write_json(output / 'selection.json', selection)
    _write_json(archive / 'receipt.json', receipt)
    _write_json(root / 'completion_status.json', {'state': 'comparison_completed_after_report_recovery',
        'candidate_won': report['candidate_won'], 'report': str(output / 'report.json'),
        'original_controller_state': 'failed_report_serialization', 'new_model_inferences': 0,
        'new_training_runs': 0, 'next_step': 'winning_export' if report['candidate_won'] else 'retain_sprint'})
    print(json.dumps({'macro_auc': new_score['macro_auc'], 'delta': report['macro_auc_delta'],
                      'checks': checks, 'recovery_seconds': receipt['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    recover(args.config.resolve(strict=True))
