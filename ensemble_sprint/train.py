"""Train two complementary ensemble members without intermediate validation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from reconstruction.data import StudyDataset, read_records
from reconstruction.train import (digest, now, rng_state, restore_rng, save_checkpoint,
                                   write_json)
from sprint.train import epoch_train, optimizer_groups, publish_file, retain_compliance
from training_hooks import compliance_run, start_run, epoch_completed
from ensemble_sprint.model import EnsembleStudyModel
from ensemble_sprint.loss import EnsembleLoss

ROOT = Path(__file__).resolve().parents[1]


def verify_plan(config):
    if config['schema'] != 'cxr_fixed_three_member_sprint_v1':
        raise ValueError('Require the fixed ensemble plan')
    for name, value in config.items():
        if name.endswith('_sha256') and name[:-7] in config:
            if digest(config[name[:-7]]) != value:
                raise ValueError('Frozen input changed: ' + name[:-7])
    train, valid = read_records(config['train_manifest']), read_records(config['valid_manifest'])
    if len(train) != 30328 or len(valid) != 1422:
        raise ValueError('Original train/valid coverage changed')
    if len({x['study_key'] for x in train}) != 18513 or len({x['study_key'] for x in valid}) != 859:
        raise ValueError('Original Study coverage changed')
    if any(x['split'] != 'train' for x in train) or any(x['split'] != 'valid' for x in valid):
        raise ValueError('Official split identity changed')
    for field in ('subject_id', 'study_key', 'file_sha256', 'pixel_sha256'):
        if {x[field] for x in train} & {x[field] for x in valid}:
            raise ValueError('Train/valid overlap: ' + field)
    if [m['id'] for m in config['members']] != ['context512', 'opacity512']:
        raise ValueError('Only the two registered complementary members may train')
    for member in config['members']:
        if member['image_size'] != 512 or member['backbone_name'] != 'convnext_tiny.fb_in22k_ft_in1k_384':
            raise ValueError('Frozen Tiny512 architecture changed')
        if digest(member['initial_weights']) != member['initial_weights_sha256']:
            raise ValueError('Warm-start source changed')
        if member.get('initial_patch'):
            if digest(member['initial_patch']) != member['initial_patch_sha256']:
                raise ValueError('Warm-start classifier changed')
            receipt = json.loads(Path(member['initial_patch_receipt']).read_text())
            if (receipt.get('state') != 'completed' or receipt.get('valid_used_for_training') is not False
                    or receipt.get('test_used') is not False
                    or receipt.get('train_manifest_sha256') != config['train_manifest_sha256']
                    or receipt.get('head_patch_fp32_sha256') != member['initial_patch_sha256']):
                raise ValueError('Classifier patch training boundary not authenticated')
    weights = np.asarray(config['fusion_weights'], dtype=float)
    if weights.shape != (3, 10) or (weights < 0).any() or not np.allclose(weights.sum(0), 1):
        raise ValueError('Invalid fixed probability fusion')
    if not np.array_equal(weights[:, 0], [1., 0., 0.]):
        raise ValueError('Class 1 remains the frozen Sprint Base score')
    return train


def initial_state(member, config):
    # Full checkpoints include optimizer/RNG metadata; authenticity was checked
    # before this trusted local load. Only the explicit FP32 EMA is inherited.
    payload = torch.load(member['initial_weights'], map_location='cpu', weights_only=False)
    if (payload.get('epoch') != member['initial_epoch'] or payload.get('weights_source') != 'ema'
            or payload.get('class_ids') != list(range(1, 11))
            or payload.get('source_manifest_sha256') != config['train_manifest_sha256']
            or payload.get('official_valid_manifest_sha256') != config['valid_manifest_sha256']):
        raise ValueError('Wrong warm-start parent or training split')
    state = dict(payload['model_state_dict'])
    if any(x.dtype != torch.float32 or not bool(torch.isfinite(x).all()) for x in state.values()):
        raise ValueError('Require the finite exact FP32 parent EMA')
    provenance = {key: payload.get(key) for key in
                  ('pretrained_provenance', 'teacher_provenance', 'distillation', 'model_config')}
    if member.get('initial_patch'):
        patch = torch.load(member['initial_patch'], map_location='cpu', weights_only=True)
        if set(patch) != {'head.fc.weight', 'head.fc.bias'}:
            raise ValueError('Unexpected classifier patch')
        state.update(patch)
    return state, provenance


@compliance_run
def run(config, member, *, resume=False):
    records = verify_plan(config)
    output = Path(config['output']) / member['id']
    local = Path(config['checkpoint_dir']) / member['id']
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    if (output / 'configuration.json').exists() and not resume:
        raise FileExistsError('Existing member; never implicitly restart training')
    if shutil.disk_usage(local).free < 3_000_000_000:
        raise RuntimeError('Insufficient checkpoint space')
    if (Path(config['output']) / 'STOP').exists():
        raise RuntimeError('Ensemble STOP is present')
    source_names = ['ensemble_sprint/train.py', 'ensemble_sprint/model.py', 'ensemble_sprint/loss.py',
                    'sprint/train.py', 'sprint/loss.py', 'reconstruction/data.py',
                    'reconstruction/model.py', 'reconstruction/train.py', 'model_overnight.py',
                    'training_hooks.py', 'training_compliance.py', 'audit_trail.py']
    sources = [ROOT / name for name in source_names]
    evidence = {'plan': config, 'member': member,
                'source_sha256': {str(p): digest(p) for p in sources},
                'protocol': 'official train only; fixed final EMA; no member/epoch validation search'}
    prior = torch.load(local / 'last.pt', map_location='cpu', weights_only=False) if resume else None
    if prior is not None and prior['evidence'] != evidence:
        raise ValueError('Resume source/configuration changed')
    start_run(local, trainer='ensemble_sprint.train', configuration=evidence, source_files=sources,
              resume=local / 'last.pt' if resume else None)
    write_json(output / 'configuration.json', evidence)
    retain_compliance(local, output)
    random.seed(member['seed']); np.random.seed(member['seed'])
    torch.manual_seed(member['seed']); torch.cuda.manual_seed_all(member['seed'])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EnsembleStudyModel(backbone_name=member['backbone_name'], branch=member['branch'],
        pretrained_path=None, drop_rate=member['drop_rate'], drop_path=member['drop_path'],
        gradient_checkpointing=True, view_chunk_size=member['view_chunk_size'],
        checkpoint_mode='chunks', channels_last=True)
    state, provenance = initial_state(member, config)
    model.load_legacy_backbone(state)
    del state
    write_json(output / 'parent_provenance.json', provenance)
    model.cuda().to(memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    dataset = StudyDataset(records, config['image_root'], image_size=512,
                           training=True, seed=member['seed'], local_tiles=False)
    targets = np.stack([records[g[0]]['targets'] for g in dataset.groups])
    positives = targets.sum(0)
    weights = torch.tensor(np.sqrt((len(dataset)-positives)/positives), dtype=torch.float32, device='cuda')
    criterion = EnsembleLoss(weights, class_weights10=member['class_weights'],
        ranking_weight=member['ranking_weight'], queue_capacity=512).cuda()
    optimizer = torch.optim.AdamW(optimizer_groups(model, member), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda')
    progress = dict(attempted_updates=0, optimizer_updates=0, skipped_updates=0)
    start = 1
    if prior is not None:
        model.load_state_dict(prior['model'], strict=True); ema.load_state_dict(prior['ema'], strict=True)
        optimizer.load_state_dict(prior['optimizer']); scaler.load_state_dict(prior['scaler'])
        criterion.load_state_dict(prior['criterion']); progress = prior['progress']
        start = prior['epoch'] + 1; restore_rng(prior['rng']); del prior
    write_json(output / 'train_statistics.json', {'studies':len(dataset), 'images':len(records),
        'positive_studies':positives.tolist(), 'pos_weight':weights.tolist(),
        'class_weights':member['class_weights'], 'source':'official train only'})
    for epoch in range(start, member['epochs']+1):
        metrics = epoch_train(model, ema, dataset, criterion, optimizer, scaler,
                              member, epoch, progress, output)
        save_checkpoint(local / 'last.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), criterion=criterion.state_dict(),
            progress=progress, epoch=epoch, rng=rng_state(), evidence=evidence), epoch, 'last_full_state')
        compact = {k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                   for k,v in ema.state_dict().items()}
        identity = save_checkpoint(local / 'final_fp16.pt', dict(state_dict=compact,
            configuration=member, epoch=epoch, evidence=evidence), epoch, 'latest_fixed_budget_ema')
        publish_file(local / 'final_fp16.pt', output / 'final_fp16.pt')
        if digest(output / 'final_fp16.pt') != identity:
            raise RuntimeError('Persistent export checksum mismatch')
        del compact
        metrics['checkpoint_sha256'] = identity
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n'); stream.flush(); os.fsync(stream.fileno())
        epoch_completed(epoch, metrics); retain_compliance(local, output)
        write_json(output / 'status.json', {'state':'training', 'completed_epochs':epoch,
                   'epochs':member['epochs'], 'updated_at':now()})
        print(json.dumps({'completed_epoch':metrics}), flush=True)
        if (Path(config['output']) / 'STOP').exists() or (output / 'STOP').exists():
            result = {'state':'paused','completed_epochs':epoch,'updated_at':now()}
            write_json(output / 'status.json',result); return result
    result = {'state':'completed', 'completed_epochs':member['epochs'],
              'weights':str(output / 'final_fp16.pt'), 'weights_sha256':identity,
              'validation_evaluated':False, 'updated_at':now()}
    write_json(output / 'status.json',result)
    write_json(output / 'progress.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--member',required=True,choices=['context512','opacity512'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    member=next(m for m in config['members'] if m['id']==args.member)
    if not args.execute:
        verify_plan(config); state,_=initial_state(member,config)
        print(json.dumps({'verified':True,'member':args.member,'parent_tensors':len(state),
                          'epochs':member['epochs'],'image_size':member['image_size']}));return
    local=Path(config['checkpoint_dir'])/args.member;local.mkdir(parents=True,exist_ok=True)
    output=Path(config['output'])/args.member;output.mkdir(parents=True,exist_ok=True)
    with (local/'.training.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(config,member,resume=args.resume)
        except BaseException as error:
            write_json(output/'status.json',{'state':'failed','error':f'{type(error).__name__}: {error}',
                                            'updated_at':now()});raise
        finally:
            retain_compliance(local,output)


if __name__=='__main__':
    main()
