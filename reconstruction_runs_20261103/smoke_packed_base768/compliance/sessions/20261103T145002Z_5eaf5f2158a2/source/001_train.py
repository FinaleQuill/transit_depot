"""Clean, complete-Study supervised training on a registered inner split.

No old medical checkpoint or teacher cache can initialize this trainer. Each
microbatch contains every image from one Study; accumulation weights Studies
equally, including the final partial window. Confirmation H is never opened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader

from training_hooks import compliance_run, start_run, epoch_completed, checkpoint_saved
from reconstruction.data import StudyDataset, read_records
from reconstruction.model import StudyModel, study_bce_loss
from reconstruction.evaluate import save_evaluation
from reconstruction.provenance import verify_public_initialization

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT.parent / 'reconstruction_runs_20261103'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, *, training, workers, seed):
    options = dict(batch_size=None, shuffle=training, num_workers=workers,
                   pin_memory=True, generator=torch.Generator().manual_seed(seed),
                   worker_init_fn=seed_worker)
    if workers:
        options.update(prefetch_factor=2, multiprocessing_context='spawn', timeout=180)
    return DataLoader(dataset, **options)


def accumulation_size(index, total, effective_batch):
    if not 0 <= index < total or effective_batch < 1:
        raise ValueError('Invalid accumulation window')
    return min(effective_batch, total - (index // effective_batch) * effective_batch)


def learning_rate_factor(update, updates_per_epoch, epochs):
    # One epoch linear warmup, then cosine to 10% of peak. Same update budget
    # and schedule for matched arms; no schedule restart when resuming.
    warmup = updates_per_epoch
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1., (update - warmup) / max(1, (epochs - 1) * updates_per_epoch))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(ema, model, updates, decay=.999):
    # Warmup avoids averaging the random new classifier for thousands of steps.
    actual = min(decay, (1 + updates) / (10 + updates))
    destinations = ema.state_dict()
    for name, value in model.state_dict().items():
        destination = destinations[name]
        if destination.is_floating_point():
            destination.lerp_(value.detach(), 1 - actual)
        else:
            destination.copy_(value)
    return actual


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, value, epoch, role):
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sha = digest(path)
    checkpoint_saved(path, sha, epoch=epoch, role=role)
    return sha


def parameter_groups(model, config):
    groups = {}
    for kind, named_parameters in model.named_parameter_groups().items():
        for name, parameter in named_parameters:
            decay = 0. if parameter.ndim <= 1 or name.endswith('.bias') else config['weight_decay']
            groups.setdefault((kind, decay), []).append(parameter)
    return [dict(params=params, lr=config[f'{kind}_lr'], base_lr=config[f'{kind}_lr'],
                 weight_decay=decay, group_name=kind)
            for (kind, decay), params in groups.items()]


def study_packs(loader, max_images, effective_studies):
    """Pack whole Studies, without crossing an optimizer accumulation window."""
    pack, images = [], 0
    for index, batch in enumerate(loader):
        pixels = batch['images']
        size = len(pixels) * (pixels.shape[1] if pixels.ndim == 5 else 1)
        if pack and (max_images <= 0 or images + size > max_images):
            yield pack, index - 1, False
            pack, images = [], 0
        pack.append(batch)
        images += size
        boundary = (index + 1) % effective_studies == 0 or index + 1 == len(loader)
        if boundary:
            yield pack, index, True
            pack, images = [], 0
    if pack:
        raise RuntimeError('Packing failed to flush complete optimizer window')


def train_epoch(model, ema, dataset, optimizer, scaler, weights, config, epoch, progress, output):
    dataset.set_epoch(epoch)
    loader = make_loader(dataset, training=True, workers=config['workers'], seed=config['seed'] + epoch)
    model.train()
    effective = config['effective_studies']
    started = time.monotonic()
    loss_sum, count, image_count = 0., 0, 0
    seen_keys, seen_images = set(), set()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    pack_images = config.get('pack_images', 0)
    for pack, index, at_boundary in study_packs(loader, pack_images, effective):
        for batch in pack:
            key, ids = batch['study_key'], batch['image_ids']
            if key in seen_keys or seen_images.intersection(ids) or len(ids) != len(set(ids)):
                raise ValueError('Duplicate Study/image in training epoch')
            seen_keys.add(key)
            seen_images.update(ids)
            image_count += len(ids)
        window = accumulation_size(index, len(loader), effective)
        images = [batch['images'].to('cuda', non_blocking=True) for batch in pack]
        with torch.autocast('cuda', dtype=torch.float16):
            if pack_images > 0:
                components = model.forward_studies(images, return_components=True)
            else:
                components = [model(value, return_components=True) for value in images]
            losses = [study_bce_loss(component['study_logits'], batch['targets'].to('cuda'), weights,
                       global_study_logits=component['global_study_logits'],
                       global_aux_weight=config['global_aux_weight'])
                      for component, batch in zip(components, pack)]
            loss = torch.stack(losses).sum()
        if not torch.isfinite(loss) or len(components) != len(pack):
            raise FloatingPointError('Invalid packed Study loss or output coverage')
        scaler.scale(loss / window).backward()
        loss_sum += float(loss.detach())
        count += len(pack)
        if at_boundary:
            factor = learning_rate_factor(progress['attempted_updates'],
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
                update_ema(ema, model, progress['optimizer_updates'], config['ema_decay'])
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            status = {'state': 'training', 'epoch': epoch, 'studies_done': count,
                'studies_total': len(loader), 'images_done': image_count,
                'mean_loss': loss_sum / count, 'elapsed_seconds': elapsed,
                'estimated_epoch_seconds': elapsed / count * len(loader),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'gradient_norm': float(norm) if torch.isfinite(norm) else None,
                'learning_rates': [g['lr'] for g in optimizer.param_groups],
                **progress, 'updated_at': now()}
            write_json(output / 'progress.json', status)
            if progress['attempted_updates'] % 10 == 0 or count == len(loader):
                print(json.dumps(status), flush=True)
    if len(seen_keys) != len(dataset) or len(seen_images) != sum(len(g) for g in dataset.groups):
        raise RuntimeError('Incomplete train Study/image coverage')
    return {'mean_loss': loss_sum / count, 'studies': count, 'images': image_count,
            'seconds': time.monotonic() - started,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), **progress}


@torch.inference_mode()
def evaluate(model, dataset, config, epoch, output):
    model.eval()
    loader = make_loader(dataset, training=False, workers=config['workers'], seed=config['seed'])
    predictions, targets, keys, patients, views = [], [], [], [], []
    started = time.monotonic()
    for pack, i, _ in study_packs(loader, config.get('pack_images', 0), len(loader)):
        with torch.autocast('cuda', dtype=torch.float16):
            images = [batch['images'].to('cuda', non_blocking=True) for batch in pack]
            logits_list = (model.forward_studies(images) if config.get('pack_images', 0) > 0
                           else [model(value) for value in images])
        if len(logits_list) != len(pack):
            raise RuntimeError('Incomplete evaluation Study predictions')
        for logits, batch in zip(logits_list, pack):
            predictions.append(logits.float().sigmoid().cpu().numpy())
            targets.append(batch['targets'].numpy())
            keys.append(batch['study_key'])
            patients.append(batch['subject_id'])
            views.append(batch['views'])
        if i % 128 < len(pack):
            write_json(output / 'progress.json', {'state': 'evaluating_inner_dev', 'epoch': epoch,
                'studies_done': i + 1, 'studies_total': len(dataset), 'updated_at': now()})
    report = save_evaluation(output / f'epoch_{epoch:02d}',
        np.asarray(predictions), np.asarray(targets), keys, patients, views,
        bootstrap_repeats=config['bootstrap_repeats'], seed=config['seed'],
        scope='inner_validation', metadata={'epoch': epoch, 'precision': 'autocast_float16',
            'selection': 'EMA only; inner dev; exploratory after repeated selection',
            'seconds': time.monotonic() - started})
    return report


def validate_config(config):
    required = {'train_manifest', 'dev_manifest', 'split_plan', 'pretrained_path',
        'provenance_path', 'output', 'image_root', 'backbone_name', 'aggregation',
        'image_size', 'use_local', 'seed', 'epochs', 'workers', 'effective_studies',
        'backbone_lr', 'head_lr', 'weight_decay', 'drop_rate', 'drop_path',
        'ema_decay', 'clip_grad', 'global_aux_weight', 'patience', 'min_delta',
        'min_epochs', 'bootstrap_repeats', 'view_chunk_size', 'smoke_studies', 'pack_images',
        'checkpoint_mode', 'channels_last'}
    if set(config) != required:
        raise ValueError(f'Configuration keys differ: {set(config) ^ required}')
    if config['epochs'] < 1 or config['effective_studies'] < 1 or not 0 <= config['workers'] <= 12:
        raise ValueError('Invalid epochs/batch/workers')
    if config['patience'] < 1 or config['min_epochs'] < 1 or config['min_delta'] < 0:
        raise ValueError('Invalid early stopping configuration')
    for name in ('backbone_lr', 'head_lr', 'clip_grad'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if config['backbone_name'] != 'convnext_base.fb_in22k_ft_in1k_384':
        raise ValueError('This experiment is registered for the original public ConvNeXt Base')
    if config['image_size'] % 32 or config['image_size'] < 128:
        raise ValueError('Expected resolution divisible by 32')
    if config['aggregation'] not in {'mean', 'attention'}:
        raise ValueError('Unknown aggregation')
    if type(config['pack_images']) is not int or config['pack_images'] < 0:
        raise ValueError('Invalid whole-Study packing image cap')
    if config['checkpoint_mode'] not in {'nested', 'chunks'} or type(config['channels_last']) is not bool:
        raise ValueError('Invalid execution mode')
    plan = json.loads(Path(config['split_plan']).read_text())
    if (plan.get('schema') != 'reconstruction_patient_content_holdout_v1'
            or plan.get('holdout_fold') != 4 or plan.get('dev_fold') != 0
            or plan.get('train_folds') != [1, 2, 3]
            or plan.get('source_manifest_sha256') !=
            'f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef'
            or any(value for pair in plan.get('isolation', {}).values() for value in pair.values())):
        raise ValueError('Missing or invalid registered isolated split')
    for role, name in [('train', 'train_manifest'), ('dev', 'dev_manifest')]:
        expected = plan['roles'][role]
        if (Path(config[name]).resolve() != Path(expected['path']).resolve()
                or digest(config[name]) != expected['sha256']):
            raise ValueError(f'{role} path/content differs from registered role')
    train = read_records(config['train_manifest'])
    dev = read_records(config['dev_manifest'])
    if {r['subject_id'] for r in train} & {r['subject_id'] for r in dev}:
        raise ValueError('Train/dev patient leakage')
    for name in ('train_manifest', 'dev_manifest'):
        if Path(config[name]).resolve().parent != Path(config['split_plan']).resolve().parent:
            raise ValueError('Manifests must belong to registered split directory')
    return train, dev, plan


@compliance_run
def run(config, *, resume=False):
    train_records, dev_records, split_plan = validate_config(config)
    initialization = verify_public_initialization(config['pretrained_path'], config['provenance_path'])
    output = Path(config['output']).resolve()
    if RUN_ROOT.resolve() not in output.parents:
        raise ValueError('New artifacts must stay in isolated research output')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Output already contains evidence; use a new run or explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 4_000_000_000:
        raise RuntimeError('Need 4 GB free for atomic last checkpoint and best EMA')
    sources = [ROOT / 'reconstruction' / name for name in
        ('__init__.py', 'train.py', 'data.py', 'model.py', 'evaluate.py', 'provenance.py', 'runtime.py', 'splits.py')]
    sources += [ROOT / name for name in ('model_overnight.py', 'runtime_compat.py',
        'training_hooks.py', 'training_compliance.py', 'audit_trail.py')]
    evidence = {**config, 'initialization': initialization,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'train_manifest_sha256': digest(config['train_manifest']),
        'dev_manifest_sha256': digest(config['dev_manifest']),
        'split_plan_sha256': digest(config['split_plan']),
        'protocol': 'new generic init; no teachers; complete Study; confirmation unopened'}
    prior = None
    if resume:
        prior = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        if prior['configuration'] != config or prior['evidence'] != evidence:
            raise ValueError('Resume configuration/source/data changed')
    start_run(output, trainer='reconstruction.train', configuration=evidence,
              source_files=sources, resume=(output / 'last.pt') if resume else None)
    write_json(output / 'configuration.json', evidence)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = StudyModel(config['backbone_name'], None if resume else config['pretrained_path'],
        aggregation=config['aggregation'], drop_rate=config['drop_rate'],
        drop_path=config['drop_path'], gradient_checkpointing=True,
        view_chunk_size=config['view_chunk_size'], use_local=config['use_local'],
        checkpoint_mode=config['checkpoint_mode'], channels_last=config['channels_last']).cuda()
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(parameter_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda')
    progress = {'attempted_updates': 0, 'optimizer_updates': 0, 'skipped_updates': 0}
    best, meaningful_best, stale, start_epoch, best_epoch = -1., -1., 0, 1, None
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True)
        ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer'])
        scaler.load_state_dict(prior['scaler'])
        progress = prior['progress']
        best, meaningful_best, stale = prior['best'], prior['meaningful_best'], prior['stale']
        best_epoch = prior['best_epoch']
        start_epoch = prior['epoch'] + 1
        restore_rng(prior['rng'])
        del prior
    else:
        # Reset after new-head construction so E0/E1 share the stochastic
        # backbone stream despite different numbers of initialization draws.
        torch.manual_seed(config['seed'] + 1)
        torch.cuda.manual_seed_all(config['seed'] + 1)
    train_data = StudyDataset(train_records, config['image_root'], image_size=config['image_size'],
        training=True, seed=config['seed'], local_tiles=config['use_local'])
    dev_data = StudyDataset(dev_records, config['image_root'], image_size=config['image_size'],
        training=False, seed=config['seed'], local_tiles=config['use_local'])
    all_targets = np.stack([train_records[group[0]]['targets'] for group in train_data.groups])
    positive = all_targets.sum(0)
    if (positive == 0).any() or (positive == len(all_targets)).any():
        raise ValueError('Every training class requires positive and negative Studies')
    weights = torch.tensor(np.sqrt((len(all_targets) - positive) / positive),
                           dtype=torch.float32, device='cuda')
    write_json(output / 'train_statistics.json', {'studies': len(train_data),
        'positive_studies': positive.tolist(), 'pos_weight': weights.cpu().tolist(),
        'source': 'inner training Studies only'})
    if config['smoke_studies']:
        # Explicit mechanics-only run; never used for checkpoint selection.
        train_data.groups = train_data.groups[:config['smoke_studies']]
        dev_data.groups = dev_data.groups[:config['smoke_studies']]
    state = 'completed'
    for epoch in range(start_epoch, config['epochs'] + 1):
        metrics = train_epoch(model, ema, train_data, optimizer, scaler, weights,
                              config, epoch, progress, output)
        report = evaluate(ema, dev_data, config, epoch, output)
        score = report['overall']['macro_auc']
        if score is None and not config['smoke_studies']:
            raise ValueError('Full inner dev is not evaluable for every class')
        if score is not None and score > best:
            best, best_epoch = score, epoch
            save_checkpoint(output / 'best.pt', {'ema': ema.state_dict(), 'epoch': epoch,
                'configuration': config, 'evidence': evidence, 'macro_auc': score}, epoch, 'best_inner_ema')
        if score is not None and score > meaningful_best + config['min_delta']:
            meaningful_best, stale = score, 0
        else:
            stale += 1
        metrics.update(epoch=epoch, macro_auc=score, best_macro_auc=best,
                       best_epoch=best_epoch, stale_epochs=stale, recorded_at=now())
        save_checkpoint(output / 'last.pt', {'model': model.state_dict(), 'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'configuration': config, 'evidence': evidence, 'progress': progress,
            'epoch': epoch, 'best': best, 'meaningful_best': meaningful_best, 'stale': stale,
            'best_epoch': best_epoch, 'rng': rng_state()}, epoch, 'last_full_state')
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        epoch_completed(epoch, metrics)
        print(json.dumps({'completed_epoch': metrics}), flush=True)
        if config['smoke_studies']:
            state = 'smoke_completed'
            break
        if (output / 'STOP').exists():
            state = 'paused'
            break
        if epoch >= config['min_epochs'] and stale >= config['patience']:
            state = 'early_stopped'
            break
    result = {'state': state, 'best_macro_auc': best if best >= 0 else None,
        'best_epoch': best_epoch, **progress, 'updated_at': now(),
        'confirmation_evaluated': False, 'official_valid_evaluated': False}
    write_json(output / 'status.json', result)
    write_json(output / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.execute:
        validate_config(config)
        print('Configuration and train/dev records verified; --execute starts GPU training.')
        return
    try:
        run(config, resume=args.resume)
    except BaseException as error:
        output = Path(config['output']).resolve()
        # Never replace another run's state after a refused existing-output
        # launch. Compliance is authoritative for errors after start_run.
        ledger = output / 'compliance' / 'status.json'
        if ledger.is_file():
            status = json.loads(ledger.read_text())
            if status.get('state') in {'failed', 'interrupted'}:
                result = {'state': status['state'], 'error': f'{type(error).__name__}: {error}',
                          'updated_at': now(), 'compliance_status': str(ledger)}
                write_json(output / 'status.json', result)
                write_json(output / 'progress.json', result)
        raise


if __name__ == '__main__':
    main()
