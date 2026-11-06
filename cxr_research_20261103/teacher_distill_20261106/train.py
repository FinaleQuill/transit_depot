"""Fixed-schedule complete-Study teachers; H is never used for epoch selection.

The numeric minibatch/accumulation/EMA core is the previously exercised DINO
trainer. This new lifecycle binds original data, nested exclusions, generic
initialization and final-epoch-only OOF predictions to immutable identities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from breakthrough.train import train_epoch
from reconstruction.train import (digest, make_loader, now, restore_rng, rng_state,
                                  save_checkpoint, study_packs, write_json)
from training_hooks import compliance_run, epoch_completed, start_run
from teacher_distill_20261106.data import StudyDataset, load_manifest
from teacher_distill_20261106.model import TeacherStudyModel
from teacher_distill_20261106.runtime import EVIDENCE, EXPERIMENT, ROOT, TRAINING_SOURCES


def verify_inputs(config):
    required = {'source_manifest', 'R_manifest', 'H_manifest', 'train_manifest',
                'oof_manifest', 'pretrained_path', 'provenance_path', 'fold_plan'}
    if set(config.get('input_sha256', {})) != required:
        raise ValueError('The eight immutable teacher inputs must be registered')
    for key, expected in config['input_sha256'].items():
        if digest(config[key]) != expected:
            raise ValueError(f'Frozen input changed: {key}')
    if set(config.get('source_sha256', {})) != set(TRAINING_SOURCES):
        raise ValueError('Exactly the registered numeric training sources must be frozen')
    for name, expected in config['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Frozen training source changed: {name}')
    if digest(config['environment_manifest']) != config['environment_manifest_sha256']:
        raise ValueError('Frozen training environment changed')
    provenance = json.loads(Path(config['provenance_path']).read_text())
    expected_repo = {'dinov3_l': 'timm/vit_large_patch16_dinov3.lvd1689m',
                     'convnext_l': 'timm/convnext_large.fb_in22k_ft_in1k_384'}[config['family']]
    if provenance.get('repo') != expected_repo or provenance.get('gated') is not False:
        raise ValueError('Public, non-gated initialization provenance mismatch')
    if provenance.get('sha256') != config['input_sha256']['pretrained_path']:
        raise ValueError('Original public checkpoint SHA256 mismatch')
    if provenance.get('revision') != config['pretrained_revision']:
        raise ValueError('Public revision mismatch')
    if provenance.get('license') not in {'dinov3-license', 'dinov3', 'apache-2.0'}:
        raise ValueError('Unregistered public license')
    rows = {name: load_manifest(config[f'{name}_manifest'])
            for name in ('source', 'R', 'H', 'train', 'oof')}
    plan = json.loads(Path(config['fold_plan']).read_text())
    selected = [r for r in plan['teacher_runs']
                if r['family'] == config['family'] and r['inner_fold'] == config['inner_fold']]
    if len(selected) != 1:
        raise ValueError('Teacher family/inner fold is not in the registered plan')
    for role in ('train', 'oof'):
        if (config[f'{role}_manifest'] != selected[0][f'{role}_manifest']
                or config['input_sha256'][f'{role}_manifest'] != selected[0]['input_sha256'][role]):
            raise ValueError(f'Teacher data does not match its registered inner fold: {role}')
    mapping = plan['patient_to_inner_fold']
    if set(mapping) != {r['subject_id'] for r in rows['R']}:
        raise ValueError('R patient partition does not match the registered inner mapping')
    for role in ('train', 'oof'):
        for row in rows[role]:
            excluded = mapping[row['subject_id']] == config['inner_fold']
            if excluded != (role == 'oof'):
                raise ValueError('Patient assigned to the wrong inner-fold role')
    if any(row['split'] != 'train' for subset in rows.values() for row in subset):
        raise ValueError('Official valid/test may not enter first-stage teacher fitting')
    source = {row['image_id']: row for row in rows['source']}
    if len(source) != 30344 or len({r['study_key'] for r in source.values()}) != 18513:
        raise ValueError('Raw official train coverage was changed')
    for name, subset in rows.items():
        if len(subset) != len({r['image_id'] for r in subset}):
            raise ValueError(f'Duplicate manifest image: {name}')
        for row in subset:
            original = source[row['image_id']]
            for field in ('subject_id', 'study_key', 'relative_path', 'targets',
                          'file_sha256', 'pixel_sha256'):
                if row[field] != original[field]:
                    raise ValueError(f'Original metadata/label changed: {name}/{field}')
    for left, right in (('R', 'H'), ('train', 'oof'), ('train', 'H'), ('oof', 'H')):
        for field in ('subject_id', 'study_key', 'image_id', 'file_sha256', 'pixel_sha256'):
            if {r[field] for r in rows[left]} & {r[field] for r in rows[right]}:
                raise ValueError(f'Nested patient/content leakage: {left}/{right}/{field}')
    for left, right, whole in (('R', 'H', 'source'), ('train', 'oof', 'R')):
        if {r['image_id'] for r in rows[left] + rows[right]} != {r['image_id'] for r in rows[whole]}:
            raise ValueError(f'Coverage not conserved: {left}+{right} != {whole}')
    schedule = config['resolution_schedule']
    if (not schedule or schedule[-1]['through_epoch'] != config['epochs']
            or sorted(x['through_epoch'] for x in schedule) != [x['through_epoch'] for x in schedule]
            or any(x['image_size'] not in (512, 768) for x in schedule)):
        raise ValueError('Invalid fixed resolution schedule')
    if (config.get('selection_policy') != 'fixed_final_ema_no_H_selection'
            or config.get('automatic_submission') is not False):
        raise ValueError('Teacher fitting must not select epochs on H or generate submission')
    if not Path(config['output']).resolve().is_relative_to(EVIDENCE):
        raise ValueError('Wrong evidence directory')
    if not Path(config['checkpoint_dir']).resolve().is_relative_to(EXPERIMENT / 'state'):
        raise ValueError('Wrong checkpoint directory')
    return rows, provenance


def training_statistics(dataset):
    y = np.asarray([dataset.records[g[0]]['targets'] for g in dataset.groups], dtype=np.float64)
    positives = y.sum(0)
    if ((positives <= 0) | (positives >= len(y))).any():
        raise ValueError('Every training class needs positive and negative Studies')
    weights = np.clip(np.sqrt((len(y) - positives) / positives), 1., 5.)
    return y, weights


def image_size_at(config, epoch):
    return next(stage['image_size'] for stage in config['resolution_schedule']
                if epoch <= stage['through_epoch'])


def source_evidence():
    paths = [ROOT/name for name in TRAINING_SOURCES]
    return paths, {str(p.relative_to(ROOT)): digest(p) for p in paths}


def fp16_state(model):
    return {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
            for k, v in model.state_dict().items()}


@torch.inference_mode()
def predict_oof(model, rows, config, output, *, smoke=False):
    """Final EMA only, no outer H access and no fitting on OOF predictions."""
    model.eval()
    sizes = [config['resolution_schedule'][-1]['image_size']] if smoke else config['oof_image_sizes']
    accumulated, metadata = None, None
    per_view_logits = []
    for size in sizes:
        model.set_image_size(size)
        dataset = StudyDataset(rows, config['image_root'], image_size=size, training=False)
        if smoke:
            dataset.groups = dataset.groups[:4]
        for flip in (False, True):
            loader = make_loader(dataset, training=False, workers=config['workers'], seed=config['seed'])
            probabilities, logits_saved, keys, patients, targets = [], [], [], [], []
            for pack, index, _ in study_packs(loader, 1, len(loader)):
                images = [b['images'].to('cuda', non_blocking=True) for b in pack]
                if flip:
                    images = [x.flip(-1) for x in images]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model.forward_studies(images)
                for logit, batch in zip(logits, pack):
                    z = logit.float().cpu().numpy()
                    logits_saved.append(z)
                    probabilities.append(torch.sigmoid(logit.float()).cpu().numpy())
                    keys.append(batch['study_key']); patients.append(batch['subject_id'])
                    targets.append(batch['targets'].numpy())
                if len(keys) % 64 == 0 or len(keys) == len(dataset):
                    write_json(output/'progress.json', dict(state='final_ema_oof_inference',
                        image_size=size, flip=flip, studies_done=len(keys), studies_total=len(dataset),
                        H_images_accessed=False, updated_at=now()))
            if len(keys) != len(dataset) or len(set(keys)) != len(keys):
                raise ValueError('Incomplete or duplicate OOF predictions')
            current = (keys, patients, np.asarray(targets))
            if metadata is None:
                metadata = current
                accumulated = np.zeros((len(keys), 10), dtype=np.float64)
            elif (keys != metadata[0] or patients != metadata[1]
                  or not np.array_equal(current[2], metadata[2])):
                raise ValueError('OOF view identity mismatch')
            accumulated += np.asarray(probabilities)
            per_view_logits.append(np.asarray(logits_saved, dtype=np.float32))
    predictions = accumulated / (2 * len(sizes))
    if not np.isfinite(predictions).all():
        raise FloatingPointError('Nonfinite OOF output')
    keys, patients, targets = metadata
    state = Path(config['checkpoint_dir'])
    np.savez_compressed(state/'oof_predictions.npz', probabilities=predictions,
        weighted_study_logits=np.stack(per_view_logits, axis=1), targets=targets,
        study_keys=np.asarray(keys), patient_ids=np.asarray(patients))
    aucs = [float(roc_auc_score(targets[:, c], predictions[:, c]))
            if len(np.unique(targets[:, c])) == 2 else None for c in range(10)]
    report = dict(studies=len(keys), patients=len(set(patients)), image_sizes=sizes,
        fixed_tta='identity+hflip', per_class_auc=aucs,
        macro_auc=float(np.mean(aucs)) if all(x is not None for x in aucs) else None,
        inference_selection=False, H_images_accessed=False,
        artifact=str(state/'oof_predictions.npz'),
        sha256=digest(state/'oof_predictions.npz'))
    write_json(output/'final_oof_report.json', report)
    return report


@compliance_run
def run(config, *, resume=False, smoke=False):
    rows, provenance = verify_inputs(config)
    output, state = Path(config['output']), Path(config['checkpoint_dir'])
    for p in (output, state):
        p.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(state).free < 20_000_000_000:
        raise RuntimeError('Need 20GB for atomic large-teacher checkpointing')
    if (output/'configuration.json').exists() and not resume:
        raise FileExistsError('Run exists; only explicit exact resume is accepted')
    paths, sources = source_evidence()
    evidence = dict(configuration=config, public_initialization=provenance,
        source_sha256=sources, original_raw_train_images=30344,
        outer_H_excluded_from_entire_training_chain=True,
        official_valid_used=False, test_used=False, mechanics_smoke=smoke,
        prior_task_checkpoint_used=False, selection_policy=config['selection_policy'])
    prior = torch.load(state/'last.pt', map_location='cpu', weights_only=False) if resume else None
    if prior is not None and prior['evidence'] != evidence:
        raise ValueError('Resume configuration, source, provenance or data changed')
    start_run(state, trainer='teacher_distill_20261106.train', configuration=evidence,
              source_files=paths, resume=state/'last.pt' if resume else None)
    write_json(output/'configuration.json', evidence)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    first_size = image_size_at(config, 1)
    model = TeacherStudyModel(config['family'], None if resume else config['pretrained_path'],
        image_size=first_size, drop_rate=config['drop_rate'], drop_path=config['drop_path'],
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        block_checkpointing=config.get('block_checkpointing', True),
        view_chunk_size=config['view_chunk_size'],
        channels_last=config.get('channels_last', False)).cuda()
    train = StudyDataset(rows['train'], config['image_root'], image_size=first_size,
                         training=True, seed=config['seed'])
    targets, weights = training_statistics(train)
    if not resume:
        linear = next(module for module in reversed(list(model.head.modules()))
                      if isinstance(module, torch.nn.Linear))
        bias = np.log(targets.mean(0) / (1. - targets.mean(0))) + np.log(weights)
        with torch.no_grad():
            linear.bias.copy_(torch.as_tensor(bias, device='cuda', dtype=torch.float32))
    ema = deepcopy(model).eval().requires_grad_(False)
    groups = model.optimizer_parameter_groups(backbone_lr=config['backbone_lr'],
        head_lr=config['head_lr'], weight_decay=config['weight_decay'],
        layer_decay=config['layer_decay'])
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), foreach=False)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    progress = dict(attempted_updates=0, optimizer_updates=0, skipped_updates=0)
    start = 1
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True); ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer']); scaler.load_state_dict(prior['scaler'])
        progress = prior['progress']; start = prior['epoch'] + 1
        restore_rng(prior['rng']); del prior
    write_json(output/'train_statistics.json', dict(studies=len(train), images=len(rows['train']),
        patients=len({r['subject_id'] for r in rows['train']}), positive_studies=targets.sum(0).tolist(),
        pos_weight=weights.tolist(), raw_training_manifest=config['train_manifest'],
        oof_studies=len({r['study_key'] for r in rows['oof']}),
        H_excluded=True, training_coverage='one complete pass over all training Studies per epoch'))
    write_json(output/'optimizer_groups.json', [{k:v for k,v in g.items()
        if k not in {'params'}} for g in optimizer.param_groups])
    if smoke:
        # Include the complete largest-view Study to probe the actual peak
        # resolution without truncating any original Study's views.
        largest = max(train.groups, key=len)
        train.groups = train.groups[:7] + ([largest] if largest not in train.groups[:7]
                                         else train.groups[7:8])
    pos_weight = torch.tensor(weights, dtype=torch.float32, device='cuda')
    outcome = 'completed_fixed_schedule'
    for epoch in range(start, config['epochs'] + 1):
        size = image_size_at(config, epoch)
        train.image_size = size; model.set_image_size(size); ema.set_image_size(size)
        metrics = train_epoch(model, ema, train, optimizer, scaler, pos_weight,
                              config, epoch, progress, output)
        metrics.update(image_size=size, recorded_at=now(), H_evaluated=False,
                       oof_evaluated=False, epoch_selection=False)
        full_sha = save_checkpoint(state/'last.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), evidence=evidence,
            progress=progress.copy(), epoch=epoch, rng=rng_state()), epoch, 'last_complete_epoch')
        ema_sha = save_checkpoint(state/'latest_ema_fp16.pt', dict(model=fp16_state(ema),
            configuration=config, epoch=epoch, source_last_sha256=full_sha,
            pos_weight=weights.tolist()), epoch, 'fixed_schedule_ema_snapshot')
        write_json(output/'checkpoint_manifest.json', dict(last=str(state/'last.pt'),
            last_sha256=full_sha, ema=str(state/'latest_ema_fp16.pt'), ema_sha256=ema_sha,
            completed_epoch=epoch, checkpoint_storage='local data disk; evidence on persistent workspace',
            updated_at=now()))
        with (output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n'); stream.flush()
        epoch_completed(epoch, metrics)
        write_json(output/'status.json', dict(state='epoch_completed', **metrics))
        print(json.dumps(dict(completed_epoch=metrics)), flush=True)
        if (output/'STOP').exists() or (EVIDENCE/'STOP').exists():
            outcome = 'stopped_at_complete_epoch'
            break
    report = None
    if outcome == 'completed_fixed_schedule':
        if progress['optimizer_updates'] <= 0:
            raise RuntimeError('No actual optimizer update occurred')
        report = predict_oof(ema, rows['oof'], config, output, smoke=smoke)
        if smoke:
            outcome = 'mechanics_smoke_completed'
    result = dict(state=outcome, completed_epoch=epoch if start <= config['epochs'] else start-1,
        **progress, final_oof=report, updated_at=now(), H_images_accessed=False,
        automatic_submission=False, next_action='Review teacher results after the six fixed teachers; students not auto-started')
    write_json(output/'status.json', result); write_json(output/'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps(config, indent=2)); return
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    with (output/'.training.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'STOP').exists() or (EVIDENCE/'STOP').exists():
            raise RuntimeError('New experiment STOP exists; launch refused')
        try:
            print(json.dumps(run(config, resume=args.resume, smoke=args.smoke)), flush=True)
        except BaseException as error:
            write_json(output/'failure.json', dict(error=f'{type(error).__name__}: {error}', updated_at=now()))
            write_json(output/'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt)
                       else 'failed', error=f'{type(error).__name__}: {error}', updated_at=now()))
            raise


if __name__ == '__main__':
    main()
