"""Single fixed sprint: four train-only epochs, final EMA, one validation comparison."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from reconstruction.data import StudyDataset, read_records
from reconstruction.train import (accumulation_size, digest, learning_rate_factor, make_loader,
    now, restore_rng, rng_state, save_checkpoint, study_packs, update_ema, write_json)
from training_hooks import compliance_run, start_run, epoch_completed
from sprint.model import SprintStudyModel
from sprint.loss import SprintLoss

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs(config):
    expected_hashes = {name[:-7]: expected for name, expected in config.items()
                       if name.endswith('_sha256') and name[:-7] in config}
    expected_hashes.update(config.get('input_sha256', {}))
    if not {'train_manifest', 'valid_manifest', 'base_snapshot', 'base_patch'} <= expected_hashes.keys():
        raise ValueError('Training inputs require explicit immutable identities')
    for name, expected in expected_hashes.items():
        if digest(config[name]) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    train = read_records(config['train_manifest'])
    valid = read_records(config['valid_manifest'])
    if any(r['split'] != 'train' for r in train) or any(r['split'] != 'valid' for r in valid):
        raise ValueError('Official split identity mismatch')
    for field in ('subject_id', 'study_key', 'file_sha256', 'pixel_sha256'):
        if {r[field] for r in train} & {r[field] for r in valid}:
            raise ValueError(f'Train/validation overlap: {field}')
    if len(train) != 30328 or len({r['study_key'] for r in train}) != 18513:
        raise ValueError('Expected complete reviewed official train')
    if len(valid) != 1422 or len({r['study_key'] for r in valid}) != 859:
        raise ValueError('Expected complete official validation')
    if config['epochs'] != 4 or config['effective_studies'] != 32 or config['image_size'] != 768:
        raise ValueError('This trainer implements the fixed four-epoch Base768 Study sprint')
    receipt = json.loads(Path(config['base_patch_receipt']).read_text())
    if (receipt.get('state') != 'completed' or receipt.get('valid_used_for_training') is not False
            or receipt.get('test_used') is not False
            or receipt.get('train_manifest_sha256') != config['train_manifest_sha256']
            or receipt.get('head_patch_fp32_sha256') != config['base_patch_sha256']):
        raise ValueError('Classifier patch must have authenticated train-only provenance')
    return train


def publish_file(source, destination):
    """Publish a completed local asset without exposing a partial copy."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with Path(source).open('rb') as incoming, temporary.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=4 * 1024**2)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    temporary.replace(destination)


def retain_compliance(state_dir, output):
    source = Path(state_dir) / 'compliance'
    if source.is_dir():
        for path in source.rglob('*'):
            if path.is_file() and path.name != '.writer.lock':
                publish_file(path, Path(output) / 'compliance' / path.relative_to(source))


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith('backbone.'):
            role, lr = 'spatial', config['spatial_lr']
        elif name.startswith('backbone.head.fc.'):
            role, lr = 'classifier', config['head_lr']
        else:
            role, lr = 'backbone', config['backbone_lr']
        decay = 0. if parameter.ndim <= 1 or name.endswith('.bias') else config['weight_decay']
        groups.setdefault((role, lr, decay), []).append(parameter)
    return [dict(params=p, lr=lr, base_lr=lr, weight_decay=decay, group_name=role)
            for (role, lr, decay), p in groups.items()]


def gamma_augment(images, config):
    # Transform already mild geometric/exposure-augmented training images only.
    mean = images[0].new_tensor([.485, .456, .406])[None, :, None, None]
    std = images[0].new_tensor([.229, .224, .225])[None, :, None, None]
    output = []
    for pixels in images:
        if random.random() < .5:
            gamma = random.uniform(*config['augmentation']['gamma_range'])
            pixels = ((pixels * std + mean).clamp(0., 1.).pow(gamma) - mean) / std
        output.append(pixels)
    return output


def epoch_train(model, ema, dataset, criterion, optimizer, scaler, config, epoch, progress, output):
    dataset.set_epoch(epoch)
    loader = make_loader(dataset, training=True, workers=config['workers'], seed=config['seed'] + epoch)
    model.train()
    criterion.train()
    # Fresh queues per epoch prevent mixing stale logits across many optimizer updates.
    criterion.reset_queue()
    effective = config['effective_studies']
    started = time.monotonic()
    loss_total, bce_total, rank_total, count, images_count = 0., 0., 0., 0, 0
    seen_keys, seen_images = set(), set()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for pack, index, boundary in study_packs(loader, config['pack_images'], effective):
        for batch in pack:
            if batch['study_key'] in seen_keys or seen_images.intersection(batch['image_ids']):
                raise ValueError('Duplicate training Study or image')
            seen_keys.add(batch['study_key'])
            seen_images.update(batch['image_ids'])
            images_count += len(batch['image_ids'])
        images = gamma_augment([b['images'].to('cuda', non_blocking=True) for b in pack], config)
        targets = torch.stack([b['targets'] for b in pack]).to('cuda', non_blocking=True)
        patients = [b['subject_id'] for b in pack]
        with torch.autocast('cuda', dtype=torch.float16):
            logits = torch.stack(model.forward_studies(images))
        components = criterion(logits, targets, patient_ids=patients, return_components=True)
        loss = components['loss']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite sprint loss')
        scaler.scale(loss * len(pack) / accumulation_size(index, len(loader), effective)).backward()
        criterion.enqueue(logits.detach(), targets, patient_ids=patients)
        count += len(pack)
        loss_total += float(loss.detach()) * len(pack)
        bce_total += float(components['bce'].detach()) * len(pack)
        rank_total += float(components['ranking'].detach()) * len(pack)
        if boundary:
            factor = learning_rate_factor(progress['attempted_updates'],
                math.ceil(len(loader) / effective), config['epochs'])
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            progress['attempted_updates'] += 1
            progress['skipped_updates'] += int(skipped)
            if not skipped:
                progress['optimizer_updates'] += 1
                update_ema(ema, model, progress['optimizer_updates'], config['ema_decay'])
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            status = dict(state='training', epoch=epoch, epochs=config['epochs'],
                studies_done=count, studies_total=len(loader), images_done=images_count,
                loss=loss_total/count, bce=bce_total/count, ranking=rank_total/count,
                elapsed_seconds=elapsed, estimated_epoch_seconds=elapsed/count*len(loader),
                estimated_remaining_training_seconds=elapsed/count*(len(loader)*(config['epochs']-epoch+1)-count),
                gradient_norm=float(norm) if torch.isfinite(norm) else None,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(), **progress, updated_at=now())
            write_json(output / 'progress.json', status)
            if progress['attempted_updates'] % 10 == 0 or count == len(loader):
                print(json.dumps(status), flush=True)
    if len(seen_keys) != len(dataset) or len(seen_images) != sum(map(len, dataset.groups)):
        raise ValueError('Incomplete official training coverage')
    return dict(epoch=epoch, studies=count, images=images_count, loss=loss_total/count,
        bce=bce_total/count, ranking=rank_total/count, seconds=time.monotonic()-started,
        **progress, recorded_at=now(), validation_evaluated=False)


@compliance_run
def run(config, *, resume=False):
    output, state_dir = Path(config['output']), Path(config['checkpoint_dir'])
    output.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    if (output / 'configuration.json').exists() and not resume:
        raise FileExistsError('Run exists; explicit resume required')
    if shutil.disk_usage(state_dir).free < 5_000_000_000:
        raise RuntimeError('Not enough local checkpoint disk')
    train_records = verify_inputs(config)
    sources = [ROOT / 'sprint' / name for name in
        ('train.py', 'model.py', 'loss.py', 'evaluate.py', 'runtime.py', 'launch.py')] + [ROOT / p for p in (
        'reconstruction/model.py', 'reconstruction/data.py', 'reconstruction/train.py',
        'model_overnight.py', 'training_hooks.py', 'training_compliance.py', 'audit_trail.py')]
    evidence = dict(configuration=config, source_sha256={str(p): digest(p) for p in sources},
        protocol='Official train only; existing compliant task weights; final epoch EMA; unique official valid comparison',
        former_research_holdout_independent=False)
    prior = None
    if resume:
        prior = torch.load(state_dir / 'last.pt', map_location='cpu', weights_only=False)
        if prior['evidence'] != evidence:
            raise ValueError('Resume inputs or source changed')
    # The full state and both recorded checkpoints belong to this local run;
    # completed compact weights and small lifecycle evidence are copied to work.
    start_run(state_dir, trainer='sprint.train', configuration=evidence, source_files=sources,
              resume=state_dir / 'last.pt' if resume else None)
    retain_compliance(state_dir, output)
    write_json(output / 'configuration.json', evidence)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SprintStudyModel(backbone_name=config['backbone_name'], pretrained_path=None,
        spatial_dim=config['spatial_dim'], drop_rate=config['drop_rate'], drop_path=config['drop_path'],
        gradient_checkpointing=True, view_chunk_size=config['view_chunk_size'],
        checkpoint_mode=config['checkpoint_mode'], channels_last=config['channels_last'])
    if prior is None:
        payload = torch.load(config['base_snapshot'], map_location='cpu', weights_only=True)
        if (payload.get('artifact_type') != 'student_ema_snapshot'
                or payload.get('snapshot_precision') != 'fp32'
                or payload.get('weights_source') != 'ema'
                or payload.get('epoch') != 10
                or payload.get('class_ids') != list(range(1, 11))):
            raise ValueError('Require the registered E10 FP32 Base EMA snapshot')
        state = dict(payload[config['base_state_key']])
        patch = torch.load(config['base_patch'], map_location='cpu', weights_only=True)
        if set(patch) != {'head.fc.weight', 'head.fc.bias'}:
            raise ValueError('Unexpected frozen-head patch')
        state.update(patch)
        model.load_legacy_backbone(state)
        del payload, state, patch
    model.cuda().to(memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    dataset = StudyDataset(train_records, config['image_root'], image_size=config['image_size'],
                           training=True, seed=config['seed'], local_tiles=False)
    all_targets = np.stack([train_records[g[0]]['targets'] for g in dataset.groups])
    positives = all_targets.sum(0)
    weights = torch.tensor(np.sqrt((len(dataset)-positives)/positives), device='cuda', dtype=torch.float32)
    criterion = SprintLoss(weights, ranking_weight=config['ranking_weight'],
        class1_weight=config['class1_weight'], queue_capacity=config['queue_capacity']).cuda()
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda')
    progress = dict(attempted_updates=0, optimizer_updates=0, skipped_updates=0)
    start_epoch = 1
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True)
        ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer'])
        scaler.load_state_dict(prior['scaler'])
        criterion.load_state_dict(prior['criterion'])
        progress = prior['progress']
        start_epoch = prior['epoch'] + 1
        restore_rng(prior['rng'])
        del prior
    write_json(output / 'train_statistics.json', dict(studies=len(dataset), images=len(train_records),
        positive_studies=positives.tolist(), pos_weight=weights.tolist(), source='official train only'))
    write_json(output / 'status.json', dict(state='training', updated_at=now()))
    for epoch in range(start_epoch, config['epochs']+1):
        metrics = epoch_train(model, ema, dataset, criterion, optimizer, scaler, config, epoch, progress, output)
        save_checkpoint(state_dir / 'last.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), criterion=criterion.state_dict(),
            configuration=config, evidence=evidence, progress=progress, epoch=epoch, rng=rng_state()), epoch, 'last_full_state')
        # A compact persistent EMA survives a machine change, unlike the local full state.
        compact = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                   for k, v in ema.state_dict().items()}
        sha = save_checkpoint(state_dir / 'sprint_fp16.pt', dict(state_dict=compact,
            configuration=config, epoch=epoch, evidence=evidence), epoch, 'latest_ema_fp16')
        publish_file(state_dir / 'sprint_fp16.pt', output / 'sprint_fp16.pt')
        if digest(output / 'sprint_fp16.pt') != sha:
            raise RuntimeError('Persistent EMA copy differs from the completed local export')
        del compact
        metrics['checkpoint_sha256'] = sha
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n'); stream.flush(); os.fsync(stream.fileno())
        epoch_completed(epoch, metrics)
        retain_compliance(state_dir, output)
        write_json(output / 'status.json', dict(state='training', completed_epochs=epoch,
                   total_epochs=config['epochs'], updated_at=now()))
        print(json.dumps({'completed_epoch':metrics}), flush=True)
        if (output / 'STOP').exists():
            result = dict(state='paused', completed_epochs=epoch, updated_at=now())
            write_json(output / 'status.json', result)
            return result
    del model, ema, optimizer, criterion, scaler, dataset
    gc.collect()
    torch.cuda.empty_cache()
    verify_inputs(config)
    write_json(output / 'status.json', dict(state='final_unique_validation', updated_at=now()))
    from sprint.evaluate import evaluate
    evaluation_config = dict(config, output=str(output / 'final_comparison'),
                             candidate_weights=str(output / 'sprint_fp16.pt'),
                             candidate_weights_sha256=digest(output / 'sprint_fp16.pt'))
    result = evaluate(evaluation_config)
    result = dict(state='completed', comparison=result, updated_at=now())
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
        verify_inputs(config)
        print('Frozen inputs and official train/valid isolation verified.')
        return
    state_dir = Path(config['checkpoint_dir'])
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / '.training.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another sprint trainer owns this run') from None
        try:
            run(config, resume=args.resume)
        except BaseException as error:
            write_json(Path(config['output']) / 'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                error=f'{type(error).__name__}: {error}', updated_at=now()))
            raise
        finally:
            retain_compliance(state_dir, Path(config['output']))

if __name__ == '__main__':
    main()
