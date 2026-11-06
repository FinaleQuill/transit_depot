"""DINOv2 training from public initialization with patient-isolated development.

Every original Study is covered once per epoch. No historical medical weights,
teacher cache, official validation or test image enters this training invocation.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from breakthrough.model import DinoStudyModel
from reconstruction.data import StudyDataset, read_records
from reconstruction.splits import content_groups, validate_mapping
from reconstruction.train import (accumulation_size, digest, learning_rate_factor,
    make_loader, now, restore_rng, rng_state, save_checkpoint, study_packs,
    update_ema, write_json)
from reconstruction.evaluate import evaluate_studies
from sprint.train import publish_file, retain_compliance
from training_hooks import compliance_run, start_run, epoch_completed

ROOT = Path(__file__).resolve().parents[1]

def verify_inputs(config):
    required = {'source_manifest', 'fold_plan', 'train_manifest', 'dev_manifest',
                'pretrained_path', 'provenance_path'}
    if set(config['input_sha256']) != required:
        raise ValueError('Exactly six registered input identities are required')
    for name, sha in config['input_sha256'].items():
        if digest(config[name]) != sha:
            raise ValueError(f'Input identity changed: {name}')
    provenance = json.loads(Path(config['provenance_path']).read_text())
    if (provenance['sha256'] != digest(config['pretrained_path'])
            or provenance['repo'] != 'timm/vit_base_patch14_reg4_dinov2.lvd142m'
            or provenance['license'] != 'apache-2.0' or provenance['gated'] is not False
            or provenance['pretraining_dataset'] != 'LVD-142M'):
        raise ValueError('Public initialization provenance failed')
    original = read_records(config['source_manifest'])
    train, dev = read_records(config['train_manifest']), read_records(config['dev_manifest'])
    if any(r['split'] != 'train' for r in original + train + dev):
        raise ValueError('Training and development must both come from official train')
    plan = json.loads(Path(config['fold_plan']).read_text())
    if plan['source_manifest_sha256'] != digest(config['source_manifest']):
        raise ValueError('Fold plan does not describe the original manifest')
    split_rows = [dict(r, targets=json.dumps(r['targets'])) for r in original]
    validate_mapping(split_rows, plan['study_to_fold'], content_groups(original))
    for field in ('subject_id', 'study_key', 'file_sha256', 'pixel_sha256'):
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError(f'Development overlap: {field}')
    expected = {r['image_id']: r for r in original}
    if len(train) + len(dev) != len(original) or {r['image_id'] for r in train + dev} != set(expected):
        raise ValueError('Split does not conserve all original rows')
    for rows, folds in ((train, {0, 1, 2, 3}), (dev, {4})):
        for row in rows:
            if row != expected[row['image_id']] or plan['study_to_fold'][row['study_key']] not in folds:
                raise ValueError('Changed label/metadata or wrong fold role')
    return train, dev, provenance

def optimizer_groups(model, config):
    groups = {}
    depth = len(model.backbone.blocks)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith('backbone.'):
            layer, lr = depth + 1, config['head_lr']
        else:
            if name.startswith('backbone.blocks.'):
                layer = int(name.split('.')[2]) + 1
            elif name.startswith(('backbone.norm.', 'backbone.fc_norm.')):
                layer = depth
            else:
                layer = 0
            lr = config['backbone_lr'] * config['layer_decay'] ** (depth - layer)
        no_decay = parameter.ndim <= 1 or name.endswith('.bias') or any(
            word in name for word in ('pos_embed', 'cls_token', 'reg_token'))
        decay = 0. if no_decay else config['weight_decay']
        group = groups.setdefault((layer, lr, decay), {'params': [], 'parameter_names': [],
            'lr': lr, 'base_lr': lr, 'weight_decay': decay, 'layer': layer})
        group['params'].append(parameter)
        group['parameter_names'].append(name)
    return list(groups.values())

def training_weights(dataset):
    targets = np.asarray([dataset.records[g[0]]['targets'] for g in dataset.groups], dtype=np.float64)
    positive = targets.sum(0)
    if (positive <= 0).any() or (positive >= len(targets)).any():
        raise ValueError('Each training class requires positive and negative Studies')
    weights = np.clip(np.sqrt((len(targets) - positive) / positive), 1., 4.)
    return targets, weights

def train_epoch(model, ema, dataset, optimizer, scaler, pos_weight, config, epoch, progress, output):
    dataset.set_epoch(epoch)
    loader = make_loader(dataset, training=True, workers=config['workers'], seed=config['seed'] + epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen, seen_images, total_loss, count = set(), set(), 0., 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for pack, index, boundary in study_packs(loader, config['pack_images'], config['effective_studies']):
        for batch in pack:
            if batch['study_key'] in seen or seen_images.intersection(batch['image_ids']):
                raise ValueError('Repeated Study/image inside training epoch')
            seen.add(batch['study_key'])
            seen_images.update(batch['image_ids'])
        images = [b['images'].to('cuda', non_blocking=True) for b in pack]
        target = torch.stack([b['targets'] for b in pack]).to('cuda', non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = torch.stack(model.forward_studies(images))
        # Study and class means, with accumulation normalized by actual tail size.
        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(),
            pos_weight=pos_weight, reduction='none').mean(1).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss / accumulation_size(index, len(loader), config['effective_studies'])).backward()
        count += len(pack)
        total_loss += float(loss.detach())
        if boundary:
            factor = learning_rate_factor(progress['attempted_updates'],
                math.ceil(len(loader) / config['effective_studies']), config['epochs'])
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale
            progress['attempted_updates'] += 1
            progress['skipped_updates'] += int(skipped)
            if not skipped:
                progress['optimizer_updates'] += 1
                update_ema(ema, model, progress['optimizer_updates'], config['ema_decay'])
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            status = dict(state='training', epoch=epoch, epochs=config['epochs'],
                studies_done=count, studies_total=len(loader), images_done=len(seen_images),
                loss=total_loss/count, elapsed_seconds=elapsed,
                estimated_epoch_seconds=elapsed/count*len(loader),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                gradient_norm=float(norm) if torch.isfinite(norm) else None,
                head_lr=max(g['lr'] for g in optimizer.param_groups),
                **progress, updated_at=now())
            write_json(output / 'progress.json', status)
            if progress['attempted_updates'] % 10 == 0 or count == len(loader):
                print(json.dumps(status), flush=True)
    if len(seen) != len(dataset) or len(seen_images) != sum(map(len, dataset.groups)):
        raise ValueError('Incomplete epoch coverage')
    return dict(epoch=epoch, loss=total_loss/count, studies=count, images=len(seen_images),
                seconds=time.monotonic()-started, **progress)

@torch.inference_mode()
def evaluate(model, dataset, config, epoch, output):
    model.eval()
    loader = make_loader(dataset, training=False, workers=config['workers'], seed=config['seed'])
    probabilities, targets, keys, patients, views = [], [], [], [], []
    for pack, index, _ in study_packs(loader, config['pack_images'], len(loader)):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model.forward_studies([b['images'].to('cuda', non_blocking=True) for b in pack])
        for prediction, batch in zip(logits, pack):
            probabilities.append(prediction.float().sigmoid().cpu().numpy())
            targets.append(batch['targets'].numpy())
            keys.append(batch['study_key'])
            patients.append(batch['subject_id'])
            views.append(batch['views'])
        if index % 128 < len(pack):
            write_json(output / 'progress.json', dict(state='evaluating_development', epoch=epoch,
                studies_done=index+1, studies_total=len(dataset), updated_at=now()))
    p, y = np.asarray(probabilities), np.asarray(targets)
    if len(keys) != len(dataset) or len(set(keys)) != len(keys):
        raise ValueError('Development coverage mismatch')
    destination = output / f'epoch_{epoch:02d}'
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / 'predictions.npz', probabilities=p, targets=y,
        study_keys=np.asarray(keys), patient_ids=np.asarray(patients),
        views_json=np.asarray([json.dumps(v) for v in views]))
    report = evaluate_studies(p, y, keys, patients, views, bootstrap_repeats=config['bootstrap_repeats'])
    counts = Counter(patients)
    sample_weight = np.asarray([1. / counts[p] for p in patients])
    patient_aucs = [roc_auc_score(y[:, c], p[:, c], sample_weight=sample_weight)
                   if len(np.unique(y[:, c])) == 2 else None for c in range(10)]
    patient_macro = float(np.mean(patient_aucs)) if all(v is not None for v in patient_aucs) else None
    report.update(patient_equal_macro_auc=patient_macro, patient_equal_per_class_auc=patient_aucs,
        holdout_H_accessed=True, old_H_role='Explicitly reassigned fold4 to DEVELOPMENT for this fresh model',
        selection_metric='Study macro AUC only; patient equal AUC is diagnostic', epoch=epoch)
    write_json(destination / 'report.json', report)
    return report

@compliance_run
def run(config, *, resume=False, smoke=False):
    train_rows, dev_rows, provenance = verify_inputs(config)
    output, state = Path(config['output']), Path(config['checkpoint_dir'])
    output.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(state).free < 5_000_000_000:
        raise RuntimeError('Need 5GB local checkpoint space')
    if (output / 'configuration.json').exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume')
    sources = [ROOT / 'breakthrough' / s for s in ('__init__.py', 'model.py', 'train.py', 'runtime.py')] + [ROOT / s for s in (
        'reconstruction/data.py', 'reconstruction/model.py', 'reconstruction/train.py',
        'reconstruction/splits.py', 'reconstruction/evaluate.py', 'sprint/train.py',
        'training_hooks.py', 'training_compliance.py', 'audit_trail.py')]
    evidence = dict(configuration=config, initialization=provenance,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        official_valid_used=False, test_used=False, teachers_used=False,
        prior_medical_checkpoint_used=False, original_H_reassigned_to_dev=True,
        mechanics_smoke=smoke)
    prior = torch.load(state / 'last.pt', map_location='cpu', weights_only=False) if resume else None
    if prior is not None and prior['evidence'] != evidence:
        raise ValueError('Resume configuration/source/data mismatch')
    start_run(state, trainer='breakthrough.train', configuration=evidence,
              source_files=sources, resume=state/'last.pt' if resume else None)
    write_json(output/'configuration.json', evidence)
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
        # Calibrated intercept for the weighted BCE, calculated on train only.
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
        model.load_state_dict(prior['model'], strict=True); ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer']); scaler.load_state_dict(prior['scaler'])
        progress = prior['progress']; start = prior['epoch'] + 1
        best, meaningful, stale, best_epoch = prior['best'], prior['meaningful'], prior['stale'], prior['best_epoch']
        restore_rng(prior['rng']); del prior
    write_json(output/'train_statistics.json', dict(studies=len(train), images=len(train_rows),
        patients=len({r['subject_id'] for r in train_rows}), positive_studies=targets.sum(0).tolist(),
        pos_weight=weights.tolist(), patient_weighting='equal original Studies',
        dev_studies=len(dev), dev_patients=len({r['subject_id'] for r in dev_rows})))
    write_json(output/'optimizer_groups.json', [{k:v for k,v in g.items() if k != 'params'}
                                               for g in optimizer.param_groups])
    if smoke:
        train.groups = train.groups[:64]
        dev.groups = dev.groups[:16]
    pos_weight = torch.tensor(weights, dtype=torch.float32, device='cuda')
    outcome = 'completed'
    for epoch in range(start, config['epochs']+1):
        metrics = train_epoch(model, ema, train, optimizer, scaler, pos_weight, config, epoch, progress, output)
        report = evaluate(ema, dev, config, epoch, output)
        score = report['overall']['macro_auc']
        if score is None and not smoke:
            raise ValueError('Every development class must have evaluable AUC')
        if score is not None and score > best:
            best, best_epoch = float(score), epoch
            value = dict(ema=ema.state_dict(), epoch=epoch, evidence=evidence, macro_auc=best)
            save_checkpoint(state/'best.pt', value, epoch, 'best_development_ema')
            fp16 = {name: tensor.detach().cpu().half() if tensor.is_floating_point()
                    else tensor.detach().cpu() for name,tensor in ema.state_dict().items()}
            save_checkpoint(state/'best_fp16.pt', dict(model=fp16, configuration=config,
                epoch=epoch, macro_auc=best), epoch, 'portable_development_candidate')
            publish_file(state/'best_fp16.pt', output/'best_fp16.pt')
        if score is not None and score > meaningful + config['min_delta']:
            meaningful, stale = float(score), 0
        else:
            stale += 1
        metrics.update(macro_auc=score, patient_equal_macro_auc=report['patient_equal_macro_auc'],
            best_macro_auc=best if best>=0 else None, best_epoch=best_epoch,
            stale_epochs=stale, recorded_at=now())
        save_checkpoint(state/'last.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), evidence=evidence,
            progress=progress, epoch=epoch, best=best, meaningful=meaningful, stale=stale,
            best_epoch=best_epoch, rng=rng_state()), epoch, 'last_full_state')
        if not smoke:
            # Keep one complete persistent recovery point, replacing atomically.
            publish_file(state/'last.pt', output/'last.pt')
        retain_compliance(state, output)
        with (output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n'); stream.flush()
        epoch_completed(epoch, metrics)
        print(json.dumps(dict(completed_epoch=metrics)), flush=True)
        write_json(output/'status.json', dict(state='epoch_completed', **metrics))
        if smoke:
            outcome='smoke_completed'; break
        if (output/'STOP').exists():
            outcome='paused'; break
        if epoch >= config['min_epochs'] and stale >= config['patience']:
            outcome='early_stopped'; break
    result = dict(state=outcome, best_macro_auc=best if best>=0 else None, best_epoch=best_epoch,
        **progress, updated_at=now(), official_valid_evaluated=False, test_used=False,
        next_action='Review development evidence before full-train refit; do not use fold4 as independent confirmation')
    write_json(output/'status.json', result); write_json(output/'progress.json', result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps(config, indent=2)); return
    output=Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    with (output/'.training.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'STOP').exists():
            raise RuntimeError('STOP exists; launch refused')
        try:
            print(json.dumps(run(config, resume=args.resume, smoke=args.smoke)), flush=True)
            retain_compliance(config['checkpoint_dir'], output)
        except BaseException as error:
            write_json(output/'failure.json', dict(error=f'{type(error).__name__}: {error}', updated_at=now()))
            retain_compliance(config['checkpoint_dir'], output)
            raise

if __name__=='__main__':
    main()
