"""Matched E0 continuation and original-pixel local Study training.

This is a separate entrypoint; frozen E0/E1 sources remain unchanged. Both
arms start from the verified E0 best EMA, reset optimizer/EMA counters, and
include the unmodified epoch-zero model in selection. No patch gets a label.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from training_hooks import compliance_run, epoch_completed
from training_compliance import read_events
from reconstruction import controller, runtime_local, train as base
from reconstruction.data import StudyDataset
from reconstruction.local_data import LocalStudyDataset
from reconstruction.local_model import LocalStudyModel
from reconstruction.local_compliance import start_run
from reconstruction.model import StudyModel, study_bce_loss
from reconstruction.evaluate import save_evaluation
from reconstruction.diagnostics import align_reference, load_bound_predictions
from reconstruction.provenance import verify_public_initialization

ROOT, RUN_ROOT = base.ROOT, base.RUN_ROOT
SMOKE_ROOT = Path('/home/featurize/data/cxr_reconstruction_smokes_20261104')
EXTRA_KEYS = {
    'parent_controller_plan', 'parent_checkpoint', 'parent_checkpoint_sha256',
    'parent_best_epoch', 'parent_best_macro_auc', 'experiment_arm',
    'augmentation_epoch_offset', 'head_warmup_epochs', 'eval_epoch_zero',
    'fixed_budget', 'smoke_high_view', 'eval_pack_images',
    'reference_epoch_zero', 'epoch_zero_max_abs_tolerance',
}


def source_files():
    names = ['__init__.py', 'train.py', 'train_local.py', 'data.py', 'model.py',
             'local_data.py', 'local_model.py', 'local_compliance.py', 'evaluate.py', 'provenance.py',
             'runtime.py', 'runtime_local.py', 'splits.py', 'controller.py', 'diagnostics.py']
    return [ROOT / 'reconstruction' / name for name in names] + [ROOT / name for name in
            ['model_overnight.py', 'runtime_compat.py', 'training_hooks.py',
             'training_compliance.py', 'audit_trail.py']]


def validate_config(config):
    if not EXTRA_KEYS <= set(config):
        raise ValueError(f'Missing local continuation keys: {EXTRA_KEYS - set(config)}')
    original = {k: v for k, v in config.items() if k not in EXTRA_KEYS}
    records = base.validate_config(original)
    local = config['experiment_arm'] == 'global_local'
    if config['experiment_arm'] not in {'global_control', 'global_local'}:
        raise ValueError('Unknown matched continuation arm')
    if config['aggregation'] != 'mean' or config['use_local'] is not local:
        raise ValueError('Only image-local fusion and Study probability mean are registered')
    if config['head_warmup_epochs'] != 0 or config['global_aux_weight'] != 0.:
        raise ValueError('No unmatched head-only warmup or auxiliary loss is enabled')
    if config['eval_epoch_zero'] is not True or config['fixed_budget'] is not True:
        raise ValueError('Fixed matched training budget and epoch-zero selection are required')
    if type(config['augmentation_epoch_offset']) is not int or config['augmentation_epoch_offset'] < 0:
        raise ValueError('Invalid augmentation epoch offset')
    if type(config['smoke_high_view']) is not bool or (config['smoke_high_view'] and not config['smoke_studies']):
        raise ValueError('Highest-view subset is mechanics-only')
    if type(config['smoke_studies']) is not int or config['smoke_studies'] < 0:
        raise ValueError('Invalid smoke Study count')
    if (type(config['parent_best_epoch']) is not int or config['parent_best_epoch'] < 1
            or not controller._finite_auc(config['parent_best_macro_auc'])
            or not isinstance(config['parent_checkpoint_sha256'], str)
            or len(config['parent_checkpoint_sha256']) != 64):
        raise ValueError('Invalid frozen parent best declaration')
    if config['augmentation_epoch_offset'] != config['parent_best_epoch']:
        raise ValueError('Augmentation offset must start after the selected E0 epoch')
    if config['eval_pack_images'] != 1:
        raise ValueError('Matched evaluation must encode one complete Study per pack')
    if config['epoch_zero_max_abs_tolerance'] != 0.000002:
        raise ValueError('Epoch-zero probability tolerance must remain preregistered at 2e-6')
    reference = config['reference_epoch_zero']
    if not config['smoke_studies'] and local and not isinstance(reference, str):
        raise ValueError('Formal local arm requires the completed control epoch-zero predictions')
    if not local and reference is not None:
        raise ValueError('Global control has no external epoch-zero reference')
    return records


def verify_parent(config):
    plan_path = Path(config['parent_controller_plan']).resolve(strict=True)
    if plan_path != (RUN_ROOT / 'controller/frozen_plan.json').resolve():
        raise ValueError('Only the registered isolated E0 lineage is accepted')
    plan = controller._read_json(plan_path)
    controller.verify_plan(plan)
    parents = [job for job in plan['jobs'] if job['name'] == 'E0']
    if len(parents) != 1:
        raise ValueError('Parent plan must contain exactly one E0')
    job = parents[0]
    verified = controller.verify_success(job)
    verified.pop('verified_at', None)  # Stable resume evidence, not a new timestamp.
    parent = Path(config['parent_checkpoint']).resolve(strict=True)
    if (parent != Path(job['output']).resolve() / 'best.pt'
            or base.digest(parent) != config['parent_checkpoint_sha256']
            or verified['best_epoch'] != config['parent_best_epoch']
            or verified['best_macro_auc'] != config['parent_best_macro_auc']):
        raise ValueError('Parent checkpoint or selected epoch differs from registration')
    for key in ['train_manifest', 'dev_manifest', 'split_plan', 'image_root',
                'pretrained_path', 'provenance_path', 'backbone_name', 'image_size',
                'drop_rate', 'drop_path', 'weight_decay']:
        if config[key] != job['config'][key]:
            raise ValueError(f'Continuation changed a fixed parent input: {key}')
    if job['config']['aggregation'] != 'mean' or job['config']['use_local']:
        raise ValueError('Parent must be the plain global E0 model')
    return {'controller_plan_path': str(plan_path), 'controller_plan_sha256': base.digest(plan_path),
            'checkpoint_path': str(parent), 'checkpoint_sha256': config['parent_checkpoint_sha256'],
            'configuration': job['config'],
            'evidence': controller._read_json(Path(job['output']) / 'configuration.json'),
            'verified_parent': verified}


def parent_state(payload, binding):
    if (not isinstance(payload, dict) or set(payload) != {'ema', 'epoch', 'configuration', 'evidence', 'macro_auc'}
            or payload['epoch'] != binding['verified_parent']['best_epoch']
            or payload['macro_auc'] != binding['verified_parent']['best_macro_auc']
            or payload['configuration'] != binding['configuration']
            or payload['evidence'] != binding['evidence']):
        raise ValueError('Loaded parent payload differs from its verified lifecycle/configuration')
    state = payload['ema']
    if not isinstance(state, dict) or not state or any(not k.startswith('backbone.') for k in state):
        raise ValueError('Only exact E0 backbone tensors may initialize this experiment')
    return state


def select_smoke_groups(dataset, count, highest_view):
    if count < 1 or count > len(dataset):
        raise ValueError('Smoke count must select existing complete Studies')
    groups = list(dataset.groups)
    if highest_view:
        groups.sort(key=lambda group: (-len(group), dataset.records[group[0]]['study_key']))
    dataset.groups = groups[:count]


def make_loader(dataset, *, training, workers, seed):
    options = dict(batch_size=None, shuffle=training, num_workers=workers,
                   pin_memory=True, generator=torch.Generator().manual_seed(seed),
                   worker_init_fn=base.seed_worker)
    if workers:
        options.update(prefetch_factor=1, multiprocessing_context='spawn', timeout=180)
    return DataLoader(dataset, **options)


def forward_pack(model, pack, config, *, return_components=False, device='cuda'):
    images = [batch['images'].to(device, non_blocking=True) for batch in pack]
    kwargs = {'return_components': return_components}
    if config['experiment_arm'] == 'global_local':
        kwargs['region_affines'] = [batch['region_affines'].to(device, non_blocking=True) for batch in pack]
    outputs = model.forward_studies(images, **kwargs)
    if len(outputs) != len(pack):
        raise RuntimeError('Incomplete complete-Study output coverage')
    return outputs


def train_epoch(model, ema, dataset, optimizer, scaler, weights, config, epoch, progress, output):
    dataset.set_epoch(config['augmentation_epoch_offset'] + epoch)
    loader = make_loader(dataset, training=True, workers=config['workers'], seed=config['seed'] + epoch)
    model.train()
    effective = config['effective_studies']
    started = previous_end = time.monotonic()
    loss_sum, count, image_count, region_count, data_wait, compute_seconds = 0., 0, 0, 0, 0., 0.
    seen_keys, seen_images = set(), set()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for pack, index, at_boundary in base.study_packs(loader, config['pack_images'], effective):
        compute_started = time.monotonic()
        data_wait += compute_started - previous_end
        for batch in pack:
            key, ids = batch['study_key'], batch['image_ids']
            if key in seen_keys or seen_images.intersection(ids) or len(ids) != len(set(ids)):
                raise ValueError('Duplicate Study/image in training epoch')
            seen_keys.add(key)
            seen_images.update(ids)
            image_count += len(ids)
            pixels = batch['images']
            region_count += len(pixels) * (pixels.shape[1] if pixels.ndim == 5 else 1)
        window = base.accumulation_size(index, len(loader), effective)
        with torch.autocast('cuda', dtype=torch.float16):
            components = forward_pack(model, pack, config, return_components=True)
            loss = torch.stack([study_bce_loss(c['study_logits'], b['targets'].to('cuda'), weights)
                                for c, b in zip(components, pack)]).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite packed Study loss')
        scaler.scale(loss / window).backward()
        loss_sum += float(loss.detach())
        count += len(pack)
        if at_boundary:
            factor = base.learning_rate_factor(progress['attempted_updates'],
                math.ceil(len(loader) / effective), config['epochs'])
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            progress['attempted_updates'] += 1
            skipped = scaler.get_scale() < old_scale
            progress['skipped_updates'] += int(skipped)
            if not skipped:
                progress['optimizer_updates'] += 1
                base.update_ema(ema, model, progress['optimizer_updates'], config['ema_decay'])
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            status = {'state': 'training', 'epoch': epoch, 'studies_done': count,
                'studies_total': len(loader), 'images_done': image_count, 'regions_done': region_count,
                'mean_loss': loss_sum / count, 'elapsed_seconds': elapsed,
                'data_wait_seconds': data_wait,
                'compute_seconds': compute_seconds + time.monotonic() - compute_started,
                'estimated_epoch_seconds': elapsed / count * len(loader),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'gradient_norm': float(norm) if torch.isfinite(norm) else None,
                'learning_rates': [g['lr'] for g in optimizer.param_groups],
                **progress, 'updated_at': base.now()}
            base.write_json(output / 'progress.json', status)
            if progress['attempted_updates'] % 10 == 0 or count == len(loader):
                print(json.dumps(status), flush=True)
        previous_end = time.monotonic()
        compute_seconds += previous_end - compute_started
    if len(seen_keys) != len(dataset) or len(seen_images) != sum(len(g) for g in dataset.groups):
        raise RuntimeError('Incomplete train Study/image coverage')
    return {'mean_loss': loss_sum / count, 'studies': count, 'images': image_count,
            'regions': region_count, 'seconds': time.monotonic() - started,
            'data_wait_seconds': data_wait, 'compute_seconds': compute_seconds,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), **progress}


@torch.inference_mode()
def evaluate(model, dataset, config, epoch, output):
    model.eval()
    loader = make_loader(dataset, training=False, workers=config['workers'], seed=config['seed'])
    predictions, targets, keys, patients, views = [], [], [], [], []
    started = time.monotonic()
    for pack, i, _ in base.study_packs(loader, config['eval_pack_images'], len(loader)):
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = forward_pack(model, pack, config)
        for logits, batch in zip(outputs, pack):
            predictions.append(logits.float().sigmoid().cpu().numpy())
            targets.append(batch['targets'].numpy())
            keys.append(batch['study_key'])
            patients.append(batch['subject_id'])
            views.append(batch['views'])
        if i % 128 < len(pack):
            base.write_json(output / 'progress.json', {
                'state': 'evaluating_train_mechanics' if config['smoke_studies'] else 'evaluating_inner_dev',
                'epoch': epoch, 'studies_done': i + 1, 'studies_total': len(dataset), 'updated_at': base.now()})
    return save_evaluation(output / f'epoch_{epoch:02d}', np.asarray(predictions), np.asarray(targets),
        keys, patients, views, bootstrap_repeats=0 if config['smoke_studies'] else config['bootstrap_repeats'],
        seed=config['seed'], scope='train_diagnostic' if config['smoke_studies'] else 'inner_validation',
        metadata={'epoch': epoch, 'precision': 'autocast_float16',
            'experiment_arm': config['experiment_arm'], 'mechanics_only': bool(config['smoke_studies']),
            'selection': 'EMA, including unchanged epoch zero; exploratory inner dev only',
            'seconds': time.monotonic() - started})


def selection_score(report, config):
    if config['smoke_studies']:
        return None
    score = report['overall']['macro_auc']
    if not controller._finite_auc(score):
        raise ValueError('Complete internal dev must be evaluable for all ten classes')
    return score


def verify_epoch_zero(config, output):
    if config['smoke_studies']:
        return
    candidate = load_bound_predictions(output / 'epoch_00')
    parent_dir = Path(config['parent_checkpoint']).parent / f"epoch_{config['parent_best_epoch']:02d}"
    parent = load_bound_predictions(parent_dir)
    parent_probabilities = align_reference(candidate, parent)
    base.write_json(output / 'epoch_zero_parent_drift.json', {
        'parent_binding': parent['binding'], 'candidate_binding': candidate['binding'],
        'max_abs_probability_difference': float(np.max(np.abs(candidate['probabilities']-parent_probabilities))),
        'macro_auc_difference': candidate['report']['macro_auc']-parent['report']['macro_auc'],
        'interpretation': 'Different historical evaluation packing/chunks; diagnostic only, no tuning'})
    if config['experiment_arm'] != 'global_local':
        return
    path = Path(config['reference_epoch_zero']).resolve(strict=True)
    if RUN_ROOT.resolve() not in path.parents or path.name != 'epoch_00' or path.parent == output:
        raise ValueError('Local epoch-zero reference must be a separate registered control')
    reference_config = controller._read_json(path.parent / 'configuration.json')
    if reference_config.get('experiment_arm') != 'global_control':
        raise ValueError('Epoch-zero reference is not the global control')
    ignored = {'experiment_arm', 'output', 'use_local', 'reference_epoch_zero'}
    if any(reference_config.get(k) != v for k, v in config.items() if k not in ignored):
        raise ValueError('Epoch-zero reference configuration is not a matched control')
    reference = load_bound_predictions(path)
    if reference['report'].get('metadata', {}).get('epoch') != 0:
        raise ValueError('Reference epoch-zero metadata differs')
    probabilities = align_reference(candidate, reference)
    difference = np.abs(candidate['probabilities']-probabilities)
    maximum = float(difference.max())
    result = {'reference_binding': reference['binding'], 'candidate_binding': candidate['binding'],
              'reference_configuration_sha256': base.digest(path.parent / 'configuration.json'),
              'max_abs_probability_difference': maximum, 'mean_abs_probability_difference': float(difference.mean()),
              'tolerance': config['epoch_zero_max_abs_tolerance'], 'passed': maximum <= config['epoch_zero_max_abs_tolerance']}
    base.write_json(output / 'epoch_zero_equivalence.json', result)
    if not result['passed']:
        raise ValueError('Local epoch-zero probabilities differ from matched global control beyond tolerance')


def checkpoint_payload(model, ema, optimizer, scaler, config, evidence, progress, epoch, best, best_epoch):
    return {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'configuration': config, 'evidence': evidence,
            'progress': deepcopy(progress), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
            'rng': base.rng_state()}


def validate_resume_payload(prior, config, evidence, output):
    if prior['configuration'] != config or prior['evidence'] != evidence:
        raise ValueError('Resume configuration/source/data/parent changed')
    if type(prior['epoch']) is not int or not 0 <= prior['epoch'] <= config['epochs']:
        raise ValueError('Invalid resume epoch')
    history = base_controller_history(output)
    if [row['epoch'] for row in history] != list(range(1, prior['epoch'] + 1)):
        raise ValueError('Resume checkpoint and complete epoch history disagree')
    if history and history[-1]['attempted_updates'] != prior['progress']['attempted_updates']:
        raise ValueError('Resume optimizer progress differs from complete history')
    baseline = controller._read_json(output / 'baseline.json')
    scores = [(0, baseline['macro_auc'])] + [(row['epoch'], row['macro_auc']) for row in history]
    finite = [(epoch, score) for epoch, score in scores if score is not None]
    if finite:
        best_epoch, best = max(finite, key=lambda row: row[1])
        if prior['best_epoch'] != best_epoch or prior['best'] != best:
            raise ValueError('Resume selected best is not supported by baseline/history')
    elif not config['smoke_studies']:
        raise ValueError('Resume lacks a finite selection score')


def verify_resume_lifecycle(output, config):
    events = read_events(output / 'compliance/events.jsonl')
    status = controller._read_json(output / 'status.json')
    allowed = {'paused', 'failed', 'interrupted'} | ({'smoke_completed'} if config['smoke_studies'] else set())
    if status.get('state') not in allowed or not events or events[-1]['event_type'] != 'training_ended':
        raise ValueError('Resume requires a closed interrupted/paused run or explicit mechanics continuation')
    if events[-1]['state'] != status['state']:
        raise ValueError('Resume lifecycle and trainer state disagree')
    sessions = {}
    for event in events:
        if (event.get('trainer') != 'reconstruction.train_local' or event.get('demonstration_only')
                or Path(event.get('run_dir', '')).resolve() != output):
            raise ValueError('Resume lifecycle belongs to a different trainer/run')
        sessions.setdefault(event['session_id'], []).append(event)
    for rows in sessions.values():
        if ([e['event_type'] for e in rows].count('training_started') != 1
                or [e['event_type'] for e in rows].count('training_ended') != 1
                or rows[0]['event_type'] != 'training_started' or rows[-1]['event_type'] != 'training_ended'
                or rows[0].get('previous_unclosed_sessions')):
            raise ValueError('Unclosed prior session requires explicit evidence repair')
    compliance = controller._read_json(output / 'compliance/status.json')
    if (compliance.get('last_event_sha256') != events[-1]['record_sha256']
            or compliance.get('state') != status['state']):
        raise ValueError('Resume compliance status differs from final event')
    for role, filename in [('last_full_state', 'last.pt'), ('best_inner_ema', 'best.pt')]:
        saved = [e for e in events if e['event_type'] == 'checkpoint_saved' and e.get('role') == role]
        if (not saved or Path(saved[-1]['checkpoint_path']).resolve() != output / filename
                or base.digest(output / filename) != saved[-1]['checkpoint_sha256']):
            raise ValueError(f'Resume checkpoint SHA differs from lifecycle: {filename}')
    history = base_controller_history(output)
    if [e['metrics'] for e in events if e['event_type'] == 'epoch_completed'] != history:
        raise ValueError('Resume lifecycle does not cover each complete history row exactly once')


def base_controller_history(output):
    path = output / 'history.jsonl'
    return controller._history(path) if path.exists() else []


@compliance_run
def run(config, *, resume=False):
    train_records, dev_records, split_plan = validate_config(config)
    output = Path(config['output']).resolve()
    allowed = RUN_ROOT.resolve() in output.parents
    allowed = allowed or (bool(config['smoke_studies']) and SMOKE_ROOT.resolve() in output.parents)
    if not allowed:
        raise ValueError('New artifacts must stay inside isolated research output')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Existing output requires explicit resume')
    parent = verify_parent(config)
    public = verify_public_initialization(config['pretrained_path'], config['provenance_path'])
    runtime_local.verify_addons()
    sources = source_files()
    evidence = {**config, 'initialization': public, 'parent': parent,
        'source_sha256': {str(p.relative_to(ROOT)): base.digest(p) for p in sources},
        'train_manifest_sha256': base.digest(config['train_manifest']),
        'dev_manifest_sha256': base.digest(config['dev_manifest']),
        'split_plan_sha256': base.digest(config['split_plan']),
        'runtime_addon_manifest_path': str(runtime_local.ADDON_MANIFEST),
        'runtime_addon_manifest_sha256': runtime_local.ADDON_MANIFEST_SHA256,
        'protocol': 'Verified E0 EMA transfer; reset optimizer/EMA counters; fixed budget; no KD or H access'}
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 3_200_000_000:
        raise RuntimeError('Need 3.2 GB free for atomic full checkpoint plus selected EMA')
    prior = None
    if resume:
        verify_resume_lifecycle(output, config)
        prior = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        validate_resume_payload(prior, config, evidence, output)
    start_run(output, trainer='reconstruction.train_local', configuration=evidence,
              source_files=sources, resume=(output / 'last.pt') if resume else None)
    base.write_json(output / 'configuration.json', evidence)
    random.seed(config['seed']); np.random.seed(config['seed'])
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = dict(drop_rate=config['drop_rate'], drop_path=config['drop_path'], gradient_checkpointing=True,
                  view_chunk_size=config['view_chunk_size'], checkpoint_mode=config['checkpoint_mode'],
                  channels_last=config['channels_last'])
    local = config['experiment_arm'] == 'global_local'
    model = (LocalStudyModel(config['backbone_name'], None, **common) if local else
             StudyModel(config['backbone_name'], None, aggregation='mean', use_local=False, **common))
    if prior is None:
        payload = torch.load(config['parent_checkpoint'], map_location='cpu', weights_only=False)
        state = parent_state(payload, parent)
        model.load_e0_state_dict(state) if local else model.load_state_dict(state, strict=True)
        del state, payload
    model = model.cuda()
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(base.parameter_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda')
    progress = {'attempted_updates': 0, 'optimizer_updates': 0, 'skipped_updates': 0}
    best, best_epoch, start_epoch = -1., 0, 1
    dataset_class = LocalStudyDataset if local else StudyDataset
    train_data = dataset_class(train_records, config['image_root'], image_size=config['image_size'],
        training=True, seed=config['seed'], local_tiles=local)
    all_targets = np.stack([train_records[group[0]]['targets'] for group in train_data.groups])
    positive = all_targets.sum(0)
    if (positive == 0).any() or (positive == len(all_targets)).any():
        raise ValueError('Every training class requires both labels')
    weights = torch.tensor(np.sqrt((len(all_targets)-positive)/positive), dtype=torch.float32, device='cuda')
    base.write_json(output / 'train_statistics.json', {'studies': len(train_data),
        'positive_studies': positive.tolist(), 'pos_weight': weights.cpu().tolist(),
        'source': 'all registered optimization Studies; before mechanics-only subset'})
    evaluation_records = train_records if config['smoke_studies'] else dev_records
    dev_data = dataset_class(evaluation_records, config['image_root'], image_size=config['image_size'],
        training=False, seed=config['seed'], local_tiles=local)
    if config['smoke_studies']:
        for dataset in [train_data, dev_data]:
            select_smoke_groups(dataset, config['smoke_studies'], config['smoke_high_view'])
        base.write_json(output / 'mechanics_scope.json', {
            'scope': 'registered optimization Studies only; no dev images',
            'studies': len(train_data), 'images': sum(len(g) for g in train_data.groups),
            'max_views': max(map(len, train_data.groups)), 'source_dataset': 'inner_training',
            'highest_view_selection': config['smoke_high_view'], 'effect_evidence': False})
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True); ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer']); scaler.load_state_dict(prior['scaler'])
        progress = prior['progress']; best, best_epoch = prior['best'], prior['best_epoch']
        start_epoch = prior['epoch'] + 1
        base.restore_rng(prior['rng'])
        del prior
    else:
        torch.manual_seed(config['seed']+1); torch.cuda.manual_seed_all(config['seed']+1)
        report = evaluate(ema, dev_data, config, 0, output)
        verify_epoch_zero(config, output)
        score = selection_score(report, config)
        best = score if score is not None else -1.
        base.write_json(output / 'baseline.json', {'epoch': 0, 'macro_auc': score,
            'predictions_sha256': report['predictions_sha256'], 'parent_epoch': config['parent_best_epoch'],
            'mechanics_only': bool(config['smoke_studies'])})
        base.save_checkpoint(output / 'best.pt', {'ema': ema.state_dict(), 'epoch': 0,
            'configuration': config, 'evidence': evidence, 'macro_auc': score}, 0, 'best_inner_ema')
        base.save_checkpoint(output / 'last.pt', checkpoint_payload(model, ema, optimizer, scaler, config,
            evidence, progress, 0, best, best_epoch), 0, 'last_full_state')
    state = 'completed'
    if (output / 'STOP').exists():
        state = 'paused'
    else:
        for epoch in range(start_epoch, config['epochs'] + 1):
            metrics = train_epoch(model, ema, train_data, optimizer, scaler, weights, config, epoch, progress, output)
            report = evaluate(ema, dev_data, config, epoch, output)
            score = selection_score(report, config)
            if score is not None and score > best:
                best, best_epoch = score, epoch
                base.save_checkpoint(output / 'best.pt', {'ema': ema.state_dict(), 'epoch': epoch,
                    'configuration': config, 'evidence': evidence, 'macro_auc': score}, epoch, 'best_inner_ema')
            metrics.update(epoch=epoch, augmentation_epoch=config['augmentation_epoch_offset']+epoch,
                           macro_auc=score, best_macro_auc=best if best >= 0 else None,
                           best_epoch=best_epoch, recorded_at=base.now())
            base.save_checkpoint(output / 'last.pt', checkpoint_payload(model, ema, optimizer, scaler, config,
                evidence, progress, epoch, best, best_epoch), epoch, 'last_full_state')
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            epoch_completed(epoch, metrics)
            print(json.dumps({'completed_epoch': metrics}), flush=True)
            if config['smoke_studies']:
                state = 'smoke_completed'
                break
            if (output / 'STOP').exists():
                state = 'paused'
                break
            # No score-dependent early stop: both formal arms consume the fixed epoch budget.
    completed = len(base_controller_history(output))
    result = {'state': state, 'best_macro_auc': best if best >= 0 else None,
        'best_epoch': best_epoch, 'completed_epochs': completed, 'fixed_budget': True,
        'epoch_zero_evaluated': True, 'experiment_arm': config['experiment_arm'],
        **progress, 'updated_at': base.now(), 'confirmation_evaluated': False,
        'official_valid_evaluated': False, 'test_accessed': False}
    base.write_json(output / 'status.json', result)
    base.write_json(output / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = controller._read_json(args.config)
    if not args.execute:
        validate_config(config); verify_parent(config)
        print('Configuration, train/dev metadata and E0 lineage verified; no model execution.')
        return
    try:
        run(config, resume=args.resume)
    except BaseException as error:
        output = Path(config['output']).resolve()
        ledger = output / 'compliance/status.json'
        if ledger.is_file():
            status = controller._read_json(ledger)
            if status.get('state') in {'failed', 'interrupted'}:
                result = {'state': status['state'], 'error': f'{type(error).__name__}: {error}',
                          'updated_at': base.now(), 'compliance_status': str(ledger)}
                base.write_json(output / 'status.json', result)
                base.write_json(output / 'progress.json', result)
        raise


if __name__ == '__main__':
    main()
