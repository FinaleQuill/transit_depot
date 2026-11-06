"""Re-evaluate registered final FP16 weights, then replay only fixed valid rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bounded_finetune import EVIDENCE, ROOT, PLAN, PLAN_SHA, registered, sha, write


def evaluate_stage(stage):
    import torch
    from dataset import read_manifest
    from inference_reproducibility import configure_inference
    from student_experiment_evaluation import ExperimentalPredictor, evaluate_predictor
    from evaluate_finetune_candidates import evaluate_finetune_candidates

    plan, recipe, _ = registered(stage)
    model = EVIDENCE / 'models' / stage
    config = json.loads((model / 'config.json').read_text())
    source = json.loads((EVIDENCE / f'{stage}.final_ema.json').read_text())
    if (source['registered_plan_sha256'] != PLAN_SHA
            or source['checkpoint_selection'] != 'fixed_final_epoch_EMA'
            or source['source_epoch'] != recipe['epochs']
            or config['source_epoch'] != recipe['epochs']
            or config['source_checkpoint_sha256'] != source['source_checkpoint_sha256']
            or config['student_warm_start']['source_snapshot_sha256'] != recipe['init_sha256']
            or config['weights_sha256'] != source['weights_sha256']
            or sha(model / config['weights_file']) != source['weights_sha256']):
        raise ValueError('Final export is not the registered final EMA')
    manifest = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
    image_root = Path('/home/featurize/data/cxr_first')
    records = read_manifest(manifest, split='valid')
    bad = [r['image_id'] for r in records if sha(image_root / r['relative_path']) != r['file_sha256']]
    if bad:
        raise ValueError('Original valid image SHA256 changed')
    destination = EVIDENCE / 'evaluations' / stage
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'original_image_integrity.json', {
        'images': len(records), 'all_sha256_equal_to_manifest': True,
        'manifest_sha256': sha(manifest), 'test_used': False})
    configure_inference()
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    predictor = ExperimentalPredictor(model, tta='hflip2')
    result = evaluate_predictor(predictor, image_root=image_root, manifest=manifest,
                                 output_dir=destination / 'valid')
    del predictor
    torch.cuda.empty_cache()
    write(destination / 'final_export_binding.json', {
        'stage': stage, 'registered_plan_sha256': PLAN_SHA,
        'final_ema_receipt_sha256': sha(EVIDENCE / f'{stage}.final_ema.json'),
        'deployment_config_sha256': sha(model / 'config.json'),
        'weights_sha256': config['weights_sha256'], 'actual_fp16_macro_auc': result['macro_auc']})
    previous = ROOT / 'artifacts/kd_v1/student_upper_bound_20261030/reports'
    available = {label.lower(): EVIDENCE / 'evaluations' / label / 'valid'
                 for label in ('X1', 'X2')}
    available = {name: directory if (directory / 'validation_report.json').is_file() else None
                 for name, directory in available.items()}
    comparison = evaluate_finetune_candidates(
        a=previous / 'A/valid_hflip2', b=previous / 'B/valid_hflip2', **available,
        manifest=manifest, plan=PLAN, plan_sha256=PLAN_SHA,
        out=EVIDENCE / f'comparison_after_{stage}', bootstrap=1000, seed=46)
    print(json.dumps({name: score['macro_auc'] for name, score in comparison['candidates'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('X1', 'X2'), required=True)
    evaluate_stage(parser.parse_args().stage)
