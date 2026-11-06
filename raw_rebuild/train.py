"""Fresh DINO experiment with original-data coverage and paired diagnostics.

This is a new invocation, not a resume of the stopped cleaned-data experiment.
The numerical training and evaluation kernels remain the previously tested ones.
Only official train supplies optimization and development records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from breakthrough.model import DinoStudyModel
from breakthrough.train import (verify_inputs as verify_split_inputs, optimizer_groups,
                                training_weights, train_epoch, evaluate)
from reconstruction.data import StudyDataset, read_records
from reconstruction.train import digest, now, restore_rng, rng_state, save_checkpoint, write_json
from sprint.train import publish_file, retain_compliance
from training_hooks import compliance_run, start_run, epoch_completed
from raw_rebuild.prepare_data import load_official
from raw_rebuild.comparison import compare_files, stage_decision
from raw_rebuild.extension import validate_extension, compatible_best_configuration

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs(config):
    train, dev, provenance = verify_split_inputs(config)
    original_csv = Path(config['image_root']) / 'train.csv'
    if digest(original_csv) != config['official_train_csv_sha256']:
        raise ValueError('Official original train CSV changed')
    original = load_official(Path(config['image_root']), 'train')
    manifest = read_records(config['source_manifest'])
    if len(original) != 30344 or len(manifest) != 30344:
        raise ValueError('All 30344 original train images are required')
    if len({r['study_key'] for r in original}) != 18513:
        raise ValueError('All 18513 original train Studies are required')
    by_image = {r['image_id']: r for r in manifest}
    if set(by_image) != {r['image_id'] for r in original}:
        raise ValueError('Original image identities changed or were excluded')
    for row in original:
        candidate = by_image[row['image_id']]
        for name in ('split', 'subject_id', 'study_id', 'study_key', 'dicom_id',
                     'relative_path', 'label_text', 'view_position', 'patient_orientation'):
            if row[name] != candidate[name]:
                raise ValueError(f'Original field changed: {name}')
        if tuple(json.loads(row['targets'])) != candidate['targets']:
            raise ValueError('Original labels changed')
        if candidate.get('exclusion_reasons') != '[]':
            raise ValueError('Historical exclusions must not be applied')
    if digest(config['reference_predictions']) != config['reference_sha256']:
        raise ValueError('Frozen patient-excluded CNN reference changed')
    fold = json.loads(Path(config['fold_plan']).read_text())
    if (fold.get('all_original_images_retained') is not True
            or fold.get('fold_assignment_changed') is not False
            or fold.get('exclusions_applied') is not False):
        raise ValueError('Raw-data fold provenance failed')
    legacy_path = Path(fold['legacy_fold_plan'])
    if digest(legacy_path) != fold['legacy_fold_plan_sha256']:
        raise ValueError('Legacy patient assignment identity changed')
    if json.loads(legacy_path.read_text())['study_to_fold'] != fold['study_to_fold']:
        raise ValueError('Current patients no longer match the fixed CNN reference fold')
    if digest(config['reference_metadata']) != config['reference_metadata_sha256']:
        raise ValueError('Frozen CNN training provenance changed')
    reference_metadata = json.loads(Path(config['reference_metadata']).read_text())
    teachers = [t for t in reference_metadata['teachers'] if t['fold_index'] == 4]
    patients = {r['subject_id'] for r in dev}
    if (len(teachers) != 1 or set(teachers[0]['oof_subject_ids']) != patients
            or set(teachers[0]['training_subject_ids']) & patients):
        raise ValueError('CNN reference did not exclude every current dev patient')
    if not 1 <= config['min_epochs'] <= config['epochs'] or config['patience'] < 1:
        raise ValueError('Invalid bounded training budget')
    return train, dev, provenance


def source_files():
    return [ROOT / p for p in (
        'raw_rebuild/__init__.py', 'raw_rebuild/prepare_data.py',
        'raw_rebuild/train.py', 'raw_rebuild/runtime.py', 'raw_rebuild/launch.py',
        'raw_rebuild/comparison.py', 'breakthrough/model.py', 'breakthrough/train.py',
        'raw_rebuild/extension.py',
        'breakthrough/runtime.py', 'breakthrough/launch.py',
        'breakthrough/generalization_monitor.py', 'reconstruction/data.py',
        'reconstruction/model.py', 'reconstruction/train.py', 'reconstruction/splits.py',
        'reconstruction/evaluate.py', 'reconstruction/runtime.py',
        'reconstruction/runtime_local.py', 'reconstruction/provenance.py',
        'sprint/train.py', 'sprint/model.py', 'sprint/loss.py', 'training_hooks.py',
        'training_compliance.py', 'audit_trail.py')]


def save_best(ema, config, evidence, state, output, epoch, score):
    save_checkpoint(state / 'best.pt', dict(ema=ema.state_dict(), epoch=epoch,
        evidence=evidence, macro_auc=score), epoch, 'best_development_ema')
    fp16 = {name: tensor.detach().cpu().half() if tensor.is_floating_point()
            else tensor.detach().cpu() for name, tensor in ema.state_dict().items()}
    save_checkpoint(state / 'best_fp16.pt', dict(model=fp16, configuration=config,
        epoch=epoch, macro_auc=score), epoch, 'portable_development_candidate')
    if shutil.disk_usage(output).free < 200_000_000:
        raise RuntimeError('Local full checkpoint saved; insufficient space to publish best model')
    publish_file(state / 'best_fp16.pt', output / 'best_fp16.pt')


@compliance_run
def run(config, *, resume=False, smoke=False, extension_receipt=None):
    train_rows, dev_rows, provenance = verify_inputs(config)
    output, state = Path(config['output']), Path(config['checkpoint_dir'])
    for directory in (output, state):
        directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(state).free < 5_000_000_000:
        raise RuntimeError('Need 5GB local checkpoint space')
    if shutil.disk_usage(output).free < 500_000_000:
        raise RuntimeError('Need 500MB persistent space for atomic FP16 publication')
    if (output / 'configuration.json').exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume')
    sources = source_files()
    evidence = dict(configuration=config, initialization=provenance,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        official_valid_used=False, test_used=False, teachers_used=False,
        reference_predictions_for_diagnostics_only=True,
        prior_medical_checkpoint_used=False, raw_images_retained=30344,
        original_H_reassigned_to_dev=True, mechanics_smoke=smoke)
    prior = torch.load(state / 'last.pt', map_location='cpu', weights_only=False) if resume else None
    lineage = prior.get('epoch_extension') if prior is not None else None
    if extension_receipt is not None:
        if prior is None or smoke:
            raise ValueError('Epoch extension requires a real full-state resume')
        lineage = validate_extension(prior, evidence, state / 'last.pt', extension_receipt)
    if prior is not None and prior['evidence'] != evidence:
        if extension_receipt is None:
            raise ValueError('Resume configuration/source/data mismatch')
    start_run(state, trainer='raw_rebuild.train', configuration=evidence,
              source_files=sources, resume=state / 'last.pt' if resume else None)
    write_json(output / 'configuration.json', evidence)
    if lineage is not None:
        write_json(output / 'epoch_extension_applied.json', lineage)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DinoStudyModel(None if resume else config['pretrained_path'],
        image_size=config['image_size'], drop_rate=config['drop_rate'], drop_path=config['drop_path'],
        view_chunk_size=config['view_chunk_size'], gradient_checkpointing=True).cuda()
    train = StudyDataset(train_rows, config['image_root'], image_size=config['image_size'],
                         training=True, seed=config['seed'])
    dev = StudyDataset(dev_rows, config['image_root'], image_size=config['image_size'], training=False)
    targets, weights = training_weights(train)
    if not resume:
        prevalence = targets.mean(0)
        bias = np.log(prevalence / (1. - prevalence)) + np.log(weights)
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.tensor(bias, device='cuda', dtype=torch.float32))
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    progress = dict(attempted_updates=0, optimizer_updates=0, skipped_updates=0)
    start, best, meaningful, stale, best_epoch = 1, -1., -1., 0, None
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True)
        ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer']); scaler.load_state_dict(prior['scaler'])
        progress = prior['progress']; start = prior['epoch'] + 1
        best, meaningful, stale, best_epoch = (prior[k] for k in ('best', 'meaningful', 'stale', 'best_epoch'))
        restore_rng(prior['rng'])
        if best_epoch is not None:
            if best_epoch == prior['epoch'] and extension_receipt is None:
                # Recover an interrupted best save/publication from the already
                # committed full checkpoint, without repeating any optimizer step.
                save_best(ema, config, evidence, state, output, best_epoch, best)
            else:
                portable = torch.load(state / 'best_fp16.pt', map_location='cpu', weights_only=False)
                if (not compatible_best_configuration(portable['configuration'], config, lineage)
                        or portable['epoch'] != best_epoch):
                    raise ValueError('Local best model does not match the recovery checkpoint')
                del portable
                publish_file(state / 'best_fp16.pt', output / 'best_fp16.pt')
        del prior
    write_json(output / 'train_statistics.json', dict(studies=len(train), images=len(train_rows),
        patients=len({r['subject_id'] for r in train_rows}), positive_studies=targets.sum(0).tolist(),
        pos_weight=weights.tolist(), patient_weighting='equal original Studies',
        dev_studies=len(dev), dev_images=len(dev_rows), dev_patients=len({r['subject_id'] for r in dev_rows}),
        original_images=30344, excluded_images=0, all_studies_all_views_each_epoch=True))
    write_json(output / 'optimizer_groups.json', [{k: v for k, v in g.items() if k != 'params'}
                                                 for g in optimizer.param_groups])
    if smoke:
        train.groups = train.groups[:64]; dev.groups = dev.groups[:16]
    pos_weight = torch.tensor(weights, dtype=torch.float32, device='cuda')
    outcome = 'completed'
    epochs_to_run = range(start, config['epochs'] + 1)
    if extension_receipt is not None and start - 1 >= config['min_epochs'] and stale >= config['patience']:
        outcome = 'early_stopped'
        epochs_to_run = ()
    for epoch in epochs_to_run:
        metrics = train_epoch(model, ema, train, optimizer, scaler, pos_weight,
                              config, epoch, progress, output)
        report = evaluate(ema, dev, config, epoch, output)
        score = report['overall']['macro_auc']
        if score is None and not smoke:
            raise ValueError('All ten development classes must have evaluable AUC')
        improved = score is not None and score > best
        if improved:
            best, best_epoch = float(score), epoch
        if score is not None and score > meaningful + config['min_delta']:
            meaningful, stale = float(score), 0
        else:
            stale += 1
        metrics.update(macro_auc=score, patient_equal_macro_auc=report['patient_equal_macro_auc'],
            best_macro_auc=best if best >= 0 else None, best_epoch=best_epoch,
            stale_epochs=stale, recorded_at=now())
        save_checkpoint(state / 'last.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), evidence=evidence,
            progress=progress, epoch=epoch, best=best, meaningful=meaningful, stale=stale,
            best_epoch=best_epoch, rng=rng_state(), epoch_extension=lineage), epoch, 'last_full_state')
        # Preserve the complete local recovery point before network publication
        # or CPU diagnostics, either of which may fail independently of training.
        if improved:
            save_best(ema, config, evidence, state, output, epoch, best)
        # Full recovery state stays on the roomy local disk. The persistent
        # workspace receives the portable best model, evidence and predictions.
        write_json(output / 'recovery.json', dict(full_checkpoint=str(state / 'last.pt'),
            sha256=digest(state / 'last.pt'), epoch=epoch,
            full_checkpoint_is_local_disk=True, full_checkpoint_persisted_to_workspace=False))
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n'); stream.flush()
        epoch_completed(epoch, metrics)
        retain_compliance(state, output)
        write_json(output / 'status.json', dict(state='epoch_completed', **metrics))
        print(json.dumps(dict(completed_epoch=metrics)), flush=True)
        if smoke:
            outcome = 'smoke_completed'; break
        comparison = compare_files(output / f'epoch_{epoch:02d}/predictions.npz',
            config['reference_predictions'], repeats=config['comparison_bootstrap_repeats'],
            expected_reference_sha256=config['reference_sha256'],
            output_path=output / f'epoch_{epoch:02d}/comparison.json')
        if (output / 'STOP').exists():
            outcome = 'paused'; break
        if epoch >= config['min_epochs'] and stale >= config['patience']:
            outcome = 'early_stopped'; break
    decision = dict(state='not_evaluated', reason='Smoke or no complete candidate', automatic_submission=False)
    if not smoke and best_epoch is not None:
        selected_comparison = output / f'epoch_{best_epoch:02d}/comparison.json'
        if not selected_comparison.exists():
            compare_files(output / f'epoch_{best_epoch:02d}/predictions.npz',
                config['reference_predictions'], repeats=config['comparison_bootstrap_repeats'],
                expected_reference_sha256=config['reference_sha256'], output_path=selected_comparison)
        decision = stage_decision(json.loads(selected_comparison.read_text()),
                                  training_complete=outcome in {'completed', 'early_stopped'})
        write_json(output / 'stage_decision.json', dict(best_epoch=best_epoch, decision=decision,
            development_selection_bias_remains=True, automatic_submission=False,
            next_stage_requires_fresh_initialization=True))
    result = dict(state=outcome, best_macro_auc=best if best >= 0 else None, best_epoch=best_epoch,
        **progress, updated_at=now(), official_valid_evaluated=False, test_used=False,
        stage_decision=decision, next_action='Review paired evidence; confirm on another patient fold before full-data refit')
    write_json(output / 'status.json', result); write_json(output / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--extension-receipt', type=Path)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps(config, indent=2)); return
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'STOP').exists():
            raise RuntimeError('STOP exists; launch refused')
        try:
            print(json.dumps(run(config, resume=args.resume, smoke=args.smoke,
                                 extension_receipt=args.extension_receipt)), flush=True)
        except BaseException as error:
            write_json(output / 'failure.json', dict(error=f'{type(error).__name__}: {error}', updated_at=now()))
            raise
        finally:
            retain_compliance(config['checkpoint_dir'], output)


if __name__ == '__main__':
    main()
