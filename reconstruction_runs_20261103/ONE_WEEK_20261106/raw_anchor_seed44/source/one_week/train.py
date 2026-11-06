"""Fixed-epoch Tiny/P2 student; never constructs or trains a teacher model.

The historical 25-epoch LR horizon is separate from the fixed 9-epoch cutoff.
Official valid is evaluated once for the train-only anchor, and never evaluated
by a train+valid refit. Missing old OOF targets receive coefficient-one BCE.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from kd_common import sha256_file
from train_distill import DistillConfig, alpha_at, distillation_losses, supervised_bce_loss
from train_overnight import (build_optimizer, capture_rng, cpu_tree, restore_rng,
                             save_checkpoint, set_learning_rates)


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def recipe(config):
    result = DistillConfig(fold_plan=Path(config['fold_plan']), soft_labels=Path(config['soft_labels']),
        output=Path(config['output']), image_root=Path(config['image_root']),
        seed=config['seed'], image_size=768, pos_weight_power=.5, amp_dtype='float16',
        batch_size=config['batch_size'], accumulation_steps=config['accumulation_steps'],
        epochs=config['lr_horizon_epochs'], patience=0,
        num_workers=config['workers'], pretrained_provenance_path=Path(config['provenance_path']),
        pretrained_weights_path=Path(config['pretrained_path']))
    result.validate()
    if (config['image_size'] != 768 or config['epochs'] != 9
            or config['lr_horizon_epochs'] != 25 or config['batch_size'] != 8
            or config['accumulation_steps'] != 4):
        raise ValueError('This experiment fixes the champion P2 recipe and 9/25 cutoff/horizon')
    return result


def class_weights(records):
    studies = {}
    for row in records:
        target = tuple(row['targets'])
        if row['study_key'] in studies and studies[row['study_key']] != target:
            raise ValueError('Conflicting original Study targets')
        studies[row['study_key']] = target
    targets = np.asarray(list(studies.values()), dtype=np.float64)
    positive = targets.sum(0)
    negative = len(targets) - positive
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError('Class weights require both labels for all classes')
    return ((negative / positive) ** .5).astype(np.float32), dict(
        formula='sqrt(negative_studies/positive_studies)', positive_studies=positive.tolist(),
        negative_studies=negative.tolist(), studies=len(studies), images=len(records),
        counted_splits=sorted({r['split'] for r in records}))


def batch_objective(logits, targets, student_weights, image_ids, cache, alpha):
    """Retain original P2 for known OOF rows, full BCE for missing OOF rows."""
    hard = supervised_bce_loss(logits, targets, student_weights)
    positions = [i for i, identity in enumerate(image_ids) if identity in cache.index]
    mask = torch.zeros(len(image_ids), dtype=torch.float32, device=logits.device)
    soft = torch.zeros_like(hard)
    if positions:
        indices = [cache.index[image_ids[i]] for i in positions]
        select = torch.tensor(positions, device=logits.device, dtype=torch.long)
        teacher = torch.as_tensor(cache.logits[indices], device=logits.device, dtype=torch.float32)
        teacher_weights = torch.as_tensor(cache.teacher_pos_weights[indices], device=logits.device,
                                          dtype=torch.float32)
        _, selected_soft = distillation_losses(logits[select], targets[select], teacher,
                                              teacher_weights, student_weights, temperature=2., align=True)
        soft = soft.index_copy(0, select, selected_soft)
        mask[select] = 1.
    return (1. - alpha * mask) * hard + alpha * soft, hard, soft, mask


def bind_sources(output):
    root = Path(__file__).resolve().parents[1]
    selected = list((root / 'one_week').glob('*.py')) + [root / name for name in (
        'train_distill.py', 'train_overnight.py', 'dataset_overnight.py', 'dataset.py',
        'model_overnight.py', 'distill_data.py', 'distill_tta_data.py', 'cv_splits.py',
        'student_experiments.py', 'kd_common.py', 'runtime_compat.py')]
    bindings = {str(path): sha256_file(path) for path in selected}
    snapshot = output / 'source'
    for source in selected:
        destination = snapshot / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return bindings


def check_bindings(bindings):
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise RuntimeError('A frozen training input changed: ' + path)


def persist_inference(output, payload, checkpoint_digest, config):
    """Publish the matching EMA atomically, including after checkpoint recovery."""
    inference = {key: payload[key] for key in ('schema', 'epoch', 'model_config', 'class_ids',
        'image_size', 'weights_source', 'aggregation', 'deployment_tta',
        'official_valid_used_for_training', 'config_digest')}
    inference.update(full_checkpoint_sha256=checkpoint_digest,
        training_complete=payload['epoch'] == config['epochs'],
        model_state_dict={key: value.half() if value.is_floating_point() else value
                          for key, value in payload['model_state_dict'].items()})
    fp16_digest = save_checkpoint(output / 'latest_ema_fp16.pt', inference)
    write(output / 'recovery.json', dict(path=str(Path(config['checkpoint_dir']) / 'last.pt'),
        sha256=checkpoint_digest, epoch=payload['epoch'], fp16_path=str(output / 'latest_ema_fp16.pt'),
        fp16_sha256=fp16_digest, complete_optimizer_state_is_local_only=True))
    return fp16_digest


def train_epoch(model, ema, optimizer, scaler, data, cache, weights, cfg, config,
                progress, epoch, output, stop):
    del ema  # EMA is updated only by the optimizer post hook.
    model.train()
    expected = {r['image_id'] for r in data.dataset.records}
    seen = set()
    total = hard_total = soft_total = weight_total = 0.
    kd_images = 0
    started = time.monotonic()
    windows = math.ceil(len(data) / cfg.accumulation_steps)
    updates_before = progress['optimizer_updates']
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(data):
        if stop['immediate']:
            raise InterruptedError('User signal: discarding incomplete epoch; last complete state retained')
        identities = batch['image_ids']
        if (len(set(identities)) != len(identities) or seen.intersection(identities)
                or not set(identities) <= expected):
            raise ValueError('Repeated or foreign image in training epoch')
        seen.update(identities)
        images = batch['images'].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = batch['targets'].cuda(non_blocking=True)
        loss_weights = batch['loss_weights'].cuda(non_blocking=True)
        if not bool(torch.isfinite(loss_weights).all()) or bool((loss_weights <= 0).any()):
            raise ValueError('Invalid per-image Study weights')
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images)
        alpha = alpha_at(cfg, epoch, index, len(data))
        per_image, hard, soft, mask = batch_objective(logits, targets, weights, identities, cache, alpha)
        weighted_sum = (per_image * loss_weights).sum()
        if not bool(torch.isfinite(weighted_sum)):
            raise FloatingPointError('Nonfinite student objective')
        first = index // cfg.accumulation_steps * cfg.accumulation_steps
        window_images = min(cfg.batch_size * cfg.accumulation_steps,
                            len(data.dataset) - first * cfg.batch_size)
        scaler.scale(weighted_sum / window_images).backward()
        progress['backward_batches'] += 1
        total += float(weighted_sum.detach())
        hard_total += float((hard.detach() * loss_weights).sum())
        soft_total += float((soft.detach() * loss_weights).sum())
        weight_total += float(loss_weights.sum())
        kd_images += int(mask.sum())
        if (index + 1) % cfg.accumulation_steps == 0 or index + 1 == len(data):
            rates = set_learning_rates(optimizer, cfg, progress['optimizer_updates'], windows)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            before = progress['optimizer_updates']
            scaler.step(optimizer)
            scaler.update()
            if progress['optimizer_updates'] == before:
                progress['amp_skipped_steps'] += 1
            optimizer.zero_grad(set_to_none=True)
        if index == 0 or (index + 1) % 25 == 0:
            receipt = dict(state='training', updated_at=now(), epoch=epoch, batch=index + 1,
                batches=len(data), images_seen=len(seen), expected_images=len(expected),
                running_loss=total / len(seen), kd_images=kd_images,
                elapsed_seconds=time.monotonic() - started, **progress)
            write(output / 'progress.json', receipt)
            print(json.dumps(receipt), flush=True)
    if seen != expected or not math.isclose(weight_total, len(expected), rel_tol=2e-6, abs_tol=.001):
        raise RuntimeError('Incomplete original-image coverage or incorrect Study weights')
    if (progress['ema_updates'] != progress['optimizer_updates']
            or progress['optimizer_updates'] <= updates_before):
        raise RuntimeError('EMA and optimizer diverged or this epoch made no optimizer updates')
    return dict(loss=total / len(expected), hard_bce=hard_total / len(expected),
        soft_kl_T2=soft_total / len(expected), images=len(seen), full_manifest_coverage=True,
        kd_images=kd_images, bce_only_images=len(seen) - kd_images, alpha_end=alpha,
        seconds=time.monotonic() - started, learning_rates=rates,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)


def run(config, *, resume=False):
    from one_week.data import loader
    from one_week.prepare import prepare
    from model_overnight import TINY_BACKBONE_NAME, build_overnight_model
    from student_experiments import load_local_pretrained_provenance
    from timm.utils import ModelEmaV3

    cfg = recipe(config)
    output, state = Path(config['output']), Path(config['checkpoint_dir'])
    output.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    if (output / 'STOP').exists():
        raise RuntimeError('STOP exists; no implicit restart')
    if (output / 'configuration.json').exists() and not resume:
        raise FileExistsError('Run already exists; use explicit resume only')
    prepared = prepare(config)
    train_records, valid_records, cache = prepared['records'], prepared['valid_records'], prepared['cache']
    records = list(train_records) + (list(valid_records) if config['include_valid'] else [])
    expected_images, expected_studies = (31766, 19372) if config['include_valid'] else (30344, 18513)
    if (len(records), len({r['study_key'] for r in records})) != (expected_images, expected_studies):
        raise ValueError('Actual optimization records do not cover the registered complete data')
    weights_np, loss_stats = class_weights(records)
    provenance = load_local_pretrained_provenance(Path(config['provenance_path']),
                                                  Path(config['pretrained_path']), TINY_BACKBONE_NAME)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if resume:
        saved_config = json.loads((output / 'configuration.json').read_text())
        if saved_config['config_digest'] != config_digest:
            raise ValueError('Exact continuation cannot change the frozen configuration')
        bindings = saved_config['bindings']
        check_bindings(bindings)
    else:
        bindings = {str(path): digest for path, digest in prepared['bindings']}
        bindings.update(bind_sources(output))
        bindings.update({config['pretrained_path']: sha256_file(config['pretrained_path']),
                         config['provenance_path']: sha256_file(config['provenance_path'])})
        write(output / 'configuration.json', dict(config=config, config_digest=config_digest,
            bindings=bindings, model_config=cfg.model_config(), pretrained_provenance=provenance,
            teacher_provenance=cache.metadata['teachers'], raw_coverage=prepared['audit'],
            pos_weight=weights_np.tolist(), loss_statistics=loss_stats,
            model_selection='fixed_epoch9_final_ema_no_early_stopping',
            official_valid_used_for_training=config['include_valid'], test_used=False))
    write(output / 'data_audit.json', {**prepared['audit'],
        'optimization_images': len(records), 'optimization_studies': expected_studies,
        'optimization_splits': sorted({r['split'] for r in records}),
        'optimization_kd_images': len(cache.index),
        'optimization_bce_only_images': len(records) - len(cache.index),
        'validation_disabled_after_merging': config['include_valid']})
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    model = build_overnight_model(pretrained=not resume,
        pretrained_weights_path=str(config['pretrained_path']) if not resume else None,
        **cfg.model_config()).cuda().to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=cfg.ema_decay, device=torch.device('cuda'), use_warmup=True,
                    warmup_power=.75, exclude_buffers=True, foreach=True)
    ema.module.requires_grad_(False)
    optimizer = build_optimizer(model, cfg, fused=False)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    progress = dict(completed_epochs=0, backward_batches=0, optimizer_updates=0,
                    ema_updates=0, amp_skipped_steps=0)
    if resume:
        checkpoint = torch.load(state / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config_digest'] != config_digest or checkpoint['bindings'] != bindings:
            raise ValueError('Checkpoint identity differs from this experiment')
        model.load_state_dict(checkpoint['raw_model_state_dict'], strict=True)
        ema.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        progress.update(checkpoint['progress'])
        restore_rng(checkpoint['rng_state'])
        persist_inference(output, checkpoint, sha256_file(state / 'last.pt'), config)
        del checkpoint
    weights = torch.tensor(weights_np, device='cuda')
    stop = dict(immediate=False)
    handlers = {}

    def request_stop(_signal, _frame):
        stop['immediate'] = True

    def after_step(_optimizer, _args, _kwargs):
        progress['optimizer_updates'] += 1
        ema.update(model, step=progress['optimizer_updates'])
        progress['ema_updates'] += 1

    hook = optimizer.register_step_post_hook(after_step)
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, request_stop)
    try:
        for epoch in range(progress['completed_epochs'] + 1, config['epochs'] + 1):
            if (output / 'STOP').exists() or (output.parent / 'STOP').exists():
                write(output / 'status.json', dict(state='stopped', updated_at=now(), **progress))
                return
            check_bindings(bindings)
            data = loader(records, config, training=True, epoch=epoch)
            training = train_epoch(model, ema, optimizer, scaler, data, cache, weights, cfg, config,
                                   progress, epoch, output, stop)
            del data
            progress['completed_epochs'] = epoch
            # Verify the finished epoch before persisting an inference candidate.
            check_bindings(bindings)
            payload = dict(schema='one_week_tiny_p2_v1', artifact_type='fixed_epoch_student',
                epoch=epoch, config_digest=config_digest, bindings=bindings, configuration=config,
                model_config=cfg.model_config(), class_ids=list(range(1, 11)), image_size=768,
                weights_source='ema', aggregation='study_mean_probability', deployment_tta='hflip2',
                official_valid_used_for_training=config['include_valid'],
                pretrained_provenance=provenance, teacher_provenance=cache.metadata['teachers'],
                model_state_dict=cpu_tree(ema.module.state_dict()),
                raw_model_state_dict=cpu_tree(model.state_dict()),
                optimizer_state_dict=cpu_tree(optimizer.state_dict()),
                scaler_state_dict=cpu_tree(scaler.state_dict()), rng_state=capture_rng(),
                progress=dict(progress))
            digest = save_checkpoint(state / 'last.pt', payload)
            fp16_digest = persist_inference(output, payload, digest, config)
            del payload
            receipt = dict(epoch=epoch, training=training, progress=dict(progress),
                           checkpoint_sha256=digest, fp16_sha256=fp16_digest, saved_at=now())
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + '\n')
            write(output / 'status.json', dict(state='training', updated_at=now(), **progress))
            print(json.dumps(receipt), flush=True)
        if not config['include_valid']:
            from train_distill import evaluate_student
            write(output / 'status.json', dict(state='final_validation', updated_at=now(), **progress))
            # The sole quality check uses the same rounded FP16 EMA parameters
            # that were actually published, rather than the training FP32 EMA.
            ema.module.half()
            validation_data = loader(valid_records, config, training=False)
            evaluation = evaluate_student(ema.module, validation_data, valid_records,
                                            torch.device('cuda'), torch.float16, cfg)
            with (output / 'valid_predictions.npz').open('wb') as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in evaluation.items()
                                              if k != 'metrics'})
            write(output / 'final_validation.json', dict(complete=True, metrics=evaluation['metrics'],
                studies=len(evaluation['targets']), images=len(valid_records), epoch=config['epochs'],
                evaluated_fp16_sha256=sha256_file(output / 'latest_ema_fp16.pt'),
                validation_precision='float16_weights_float16_autocast',
                selection='single_end_of_run_quality_check_not_epoch_selection',
                independent_confirmation=False, official_valid_previously_reused=True))
            del validation_data
        check_bindings(bindings)
        write(output / 'status.json', dict(state='completed', updated_at=now(), **progress))
        write(output / 'progress.json', dict(state='completed', updated_at=now(), **progress))
    except BaseException as error:
        state_name = 'stopped_by_signal' if isinstance(error, InterruptedError) else 'failed'
        write(output / 'status.json', dict(state=state_name, error=f'{type(error).__name__}: {error}',
              incomplete_epoch_discarded=True, updated_at=now(), **progress))
        write(output / 'progress.json', dict(state=state_name, updated_at=now(), **progress))
        raise
    finally:
        hook.remove()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    recipe(config)
    if not args.execute:
        print(json.dumps(config, indent=2)); return
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.trainer.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config, resume=args.resume)


if __name__ == '__main__':
    main()
