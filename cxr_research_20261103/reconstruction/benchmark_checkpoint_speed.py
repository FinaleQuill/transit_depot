"""Bounded, train-only activation-retention probe from a closed E2 checkpoint.

The parent starts two sequential worker processes, each with its own CUDA
context and the same full model/EMA/Adam/scaler/RNG state. Probe updates are
discarded. No scientific checkpoint, history or lifecycle is written. This is
an isolated compute estimate, not a completed training epoch or inference cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


SCHEMA = 'cxr_e2_checkpoint_speed_benchmark_v1'
MODES = ('original', 'retain_small')
VIEWS = (1, 2, 7)
MIN_FREE_BYTES = 1536 * 1024**2
MAX_TOTAL_SECONDS = 550
SCOPE = ('Train-only complete-Study forward/backward and isolated optimizer/EMA probes; '
         'loss divided by the formal accumulation size, but each probe updates after one Study. '
         'Not a real 32-Study accumulated update, full epoch or deployment measurement.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def value_sha(value):
    """Canonical exact tensor/state hashing, including dtype, shape and keys."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b'tensor\0' + str(tensor.dtype).encode() + b'\0')
            add(list(tensor.shape))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b'array\0' + str(item.dtype).encode() + b'\0')
            add(list(item.shape)); digest.update(item.tobytes(order='C'))
        elif isinstance(item, dict):
            digest.update(b'dict\0')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                add(key); add(item[key])
            digest.update(b'enddict\0')
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b'\0')
            for entry in item:
                add(entry)
            digest.update(b'endsequence\0')
        elif item is None or type(item) in (str, bool, int, float):
            digest.update(type(item).__name__.encode() + b'\0')
            encoded = item.hex() if type(item) is float else json.dumps(item, ensure_ascii=False)
            digest.update(encoded.encode() + b'\0')
        else:
            raise TypeError(f'Unsupported exact state type: {type(item)}')

    add(value)
    return digest.hexdigest()


def workload(records):
    from reconstruction.data import study_groups
    groups = study_groups(records)
    if any(row['split'] != 'train' for row in records):
        raise ValueError('Benchmark may only use the registered optimization training manifest')
    distribution = Counter(map(len, groups))
    if max(distribution) != 7 or not all(n in distribution for n in VIEWS):
        raise ValueError('Expected registered one/two/seven-image training Studies')
    selected = {}
    for n in VIEWS:
        group = min((g for g in groups if len(g) == n), key=lambda g: records[g[0]]['study_key'])
        selected[str(n)] = {'study_key': records[group[0]]['study_key'],
            'image_ids': [records[i]['image_id'] for i in group],
            'source_sha256': [records[i]['file_sha256'] for i in group]}
    return {'study_image_count_distribution': {str(k): v for k, v in sorted(distribution.items())},
            'study_count': len(groups), 'image_count': len(records), 'selected': selected}


def verify_files(plan):
    for path, expected in plan['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Bound benchmark input changed: {path}')


def _worker(plan_path, mode):
    import numpy as np
    import torch
    from reconstruction import train_local as frozen
    from reconstruction import train_local_checkpoint_speed as adapter
    from reconstruction.data import read_records
    from reconstruction.local_data import LocalStudyDataset
    from reconstruction.local_model import LocalStudyModel
    from reconstruction.model import study_bce_loss

    plan = read_json(plan_path)
    output = Path(plan['output'])
    verify_files(plan)
    config = read_json(plan['config_path'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Probe requires one exclusive CUDA device')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(plan['checkpoint_path'], map_location='cpu', weights_only=False)
    if payload.get('configuration') != config:
        raise ValueError('Checkpoint configuration differs from the registered scientific configuration')
    evidence = read_json(Path(config['output']) / 'configuration.json')
    if payload.get('evidence') != evidence:
        raise ValueError('Full checkpoint evidence differs from actual run evidence')
    frozen.validate_resume_payload(payload, config, evidence, Path(config['output']))
    if not 1 <= payload['epoch'] < config['epochs']:
        raise ValueError('Probe requires remaining training budget and a closed positive epoch')
    next_epoch = payload['epoch'] + 1
    records = read_records(config['train_manifest'])
    if workload(records) != plan['workload']:
        raise ValueError('Training selection or image-count distribution changed')
    dataset = LocalStudyDataset(records, config['image_root'], image_size=config['image_size'],
        training=True, seed=config['seed'], local_tiles=True)
    dataset.set_epoch(config['augmentation_epoch_offset'] + next_epoch)
    positive = np.stack([records[g[0]]['targets'] for g in dataset.groups]).sum(0)
    if (positive == 0).any() or (positive == len(dataset)).any():
        raise ValueError('Incomplete original training label support')
    weights_cpu = torch.tensor(np.sqrt((len(dataset) - positive) / positive), dtype=torch.float32)
    weights = weights_cpu.cuda()
    common = dict(drop_rate=config['drop_rate'], drop_path=config['drop_path'],
        gradient_checkpointing=True, view_chunk_size=config['view_chunk_size'],
        checkpoint_mode=config['checkpoint_mode'], channels_last=config['channels_last'])
    model = LocalStudyModel(config['backbone_name'], None, **common).cuda()
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(frozen.base.parameter_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler('cuda')
    binding = adapter.load_profile(plan['profile_binding']['path'], plan['profile_binding']['sha256'])
    indices = {records[g[0]]['study_key']: i for i, g in enumerate(dataset.groups)}
    properties = torch.cuda.get_device_properties(0)
    result = {'schema': SCHEMA, 'mode': mode, 'complete': False, 'scope': SCOPE,
        'device': {'name': properties.name, 'total_memory_bytes': properties.total_memory},
        'checkpoint_epoch': payload['epoch'], 'augmentation_epoch': dataset.epoch,
        'full_optimizer_and_ema_resident': True, 'weights_sha256': value_sha(weights_cpu),
        'accumulation_gradient_buffers': 'All trainable FP32 parameter gradients allocated before forward; zero values, not an actual 32-Study accumulated gradient',
        'preallocated_gradient_bytes': sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
        'shapes': {}, 'checkpoint_sha256': plan['checkpoint_sha256']}
    started = time.monotonic()

    def restore():
        optimizer.zero_grad(set_to_none=True)
        model.load_state_dict(payload['model'], strict=True)
        ema.load_state_dict(payload['ema'], strict=True)
        # Adam's CPU step tensors may otherwise alias the checkpoint payload
        # and be incremented in-place, corrupting later probe restorations.
        optimizer.load_state_dict(deepcopy(payload['optimizer']))
        scaler.load_state_dict(payload['scaler'])
        frozen.base.restore_rng(payload['rng'])
        model.train()
        return deepcopy(payload['progress'])

    def step(pack, progress, *, fingerprints):
        optimizer.zero_grad(set_to_none=True)
        # Formal accumulation usually enters a forward pass while previous
        # Studies' FP32 gradient buffers remain resident. Keep that memory in
        # this isolated probe; zero contents allow an exact one-Study update.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            components = frozen.forward_pack(model, pack, config, return_components=True)
            raw_loss = torch.stack([study_bce_loss(c['study_logits'], b['targets'], weights)
                                    for c, b in zip(components, pack)]).sum()
        if not torch.isfinite(raw_loss):
            raise FloatingPointError('Nonfinite probe training loss')
        scaler.scale(raw_loss / config['effective_studies']).backward()
        scaler.unscale_(optimizer)
        torch.cuda.synchronize()
        forward_backward_seconds = time.perf_counter() - begin
        exact = {}
        if fingerprints:
            exact = {'logits_sha256': value_sha([c['study_logits'] for c in components]),
                'components_sha256': value_sha(components), 'loss_sha256': value_sha(raw_loss),
                'gradients_sha256': value_sha({name: p.grad for name, p in model.named_parameters()}),
                'rng_after_backward_sha256': value_sha(frozen.base.rng_state())}
        torch.cuda.synchronize()
        update_begin = time.perf_counter()
        factor = frozen.base.learning_rate_factor(progress['attempted_updates'],
            math.ceil(len(dataset) / config['effective_studies']), config['epochs'])
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite probe gradient norm')
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        skipped = scaler.get_scale() < old_scale
        progress['attempted_updates'] += 1
        progress['skipped_updates'] += int(skipped)
        if not skipped:
            progress['optimizer_updates'] += 1
            frozen.base.update_ema(ema, model, progress['optimizer_updates'], config['ema_decay'])
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_begin
        if fingerprints:
            exact.update(model_after_sha256=value_sha(model.state_dict()),
                ema_after_sha256=value_sha(ema.state_dict()),
                optimizer_after_sha256=value_sha(optimizer.state_dict()),
                scaler_after_sha256=value_sha(scaler.state_dict()),
                rng_after_update_sha256=value_sha(frozen.base.rng_state()),
                progress_after=deepcopy(progress), skipped_update=bool(skipped))
        free, total = torch.cuda.mem_get_info()
        del components, raw_loss
        return {'forward_backward_seconds': forward_backward_seconds, 'optimizer_ema_seconds': update_seconds,
            'free_device_bytes_after_step': free, 'device_total_bytes': total, 'exact': exact}

    context = adapter.checkpoint_adapter(config, binding) if mode == 'retain_small' else nullcontext()
    with context:
        for views in VIEWS:
            selected = plan['workload']['selected'][str(views)]
            batch = dataset[indices[selected['study_key']]]
            if batch['image_ids'] != selected['image_ids'] or len(batch['images']) != views:
                raise ValueError('Probe changed the complete Study image order')
            input_digest = value_sha(batch)
            pack = [{**batch, **{key: batch[key].cuda() for key in ['images', 'region_affines', 'targets']}}]
            del batch
            gc.collect(); torch.cuda.empty_cache()
            progress = restore()
            torch.cuda.reset_peak_memory_stats()
            initial_rng = value_sha(frozen.base.rng_state())
            initial_state = value_sha({'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'progress': progress})
            equivalence = step(pack, progress, fingerprints=True)
            progress = restore()
            warmup = step(pack, progress, fingerprints=False)
            timings = [step(pack, progress, fingerprints=False) for _ in range(3)]
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            # CUDA context/non-PyTorch allocations remain outside reserved.
            external_bytes = max(0, total - free - torch.cuda.memory_reserved())
            margin = total - peak_reserved - external_bytes
            row = {'views': views, 'regions': views * 5, 'study_key': selected['study_key'],
                'input_sha256': input_digest, 'initial_rng_sha256': initial_rng,
                'initial_state_sha256': initial_state,
                'equivalence': equivalence['exact'], 'warmup_count': 1, 'timing_count': 3,
                'timings': timings, 'warmup_seconds': warmup['forward_backward_seconds'],
                'median_forward_backward_seconds': statistics.median(t['forward_backward_seconds'] for t in timings),
                'median_optimizer_ema_seconds': statistics.median(t['optimizer_ema_seconds'] for t in timings),
                'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
                'estimated_min_device_headroom_bytes': margin,
                'timing_final_rng_sha256': value_sha(frozen.base.rng_state()),
                'checkpointing_expected': mode == 'original' or views * 5 > binding['profile']['max_retained_regions']}
            result['shapes'][str(views)] = row
            write_json(output / f'{mode}.partial.json', result)
            del pack
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
    verify_files(plan)
    result.update(complete=True, outcome='valid', worker_seconds=time.monotonic() - started)
    write_json(output / f'{mode}.json', result)
    return result


def compare(plan, baseline, candidate, elapsed):
    if not baseline.get('complete') or baseline.get('outcome') != 'valid':
        raise ValueError('Baseline execution did not complete validly')
    result = {'schema': SCHEMA, 'complete': True, 'adopt': False, 'state': 'rejected',
        'reason': None, 'scope': SCOPE, 'checkpoint_sha256': plan['checkpoint_sha256'],
        'profile_binding': plan['profile_binding'], 'baseline_valid': True,
        'candidate_oom': candidate.get('outcome') == 'out_of_memory',
        'total_benchmark_seconds': elapsed, 'input_sha256': plan['input_sha256'],
        'workload': plan['workload'], 'baseline_training_epoch_seconds': plan['baseline_training_epoch_seconds'],
        'minimum_device_headroom_bytes': MIN_FREE_BYTES,
        'weighted_speedup': None, 'predicted_seconds_saved_per_epoch': None,
        'net_first_epoch_saving': None}
    if result['candidate_oom']:
        result['reason'] = 'candidate_out_of_memory_original_execution_retained'
        return result
    if not candidate.get('complete') or candidate.get('outcome') != 'valid':
        raise ValueError('Candidate failed for a reason other than CUDA out of memory')
    exact, mismatches = True, []
    for views in VIEWS:
        a, b = baseline['shapes'][str(views)], candidate['shapes'][str(views)]
        for field in ['input_sha256', 'initial_rng_sha256', 'initial_state_sha256',
                      'equivalence', 'timing_final_rng_sha256']:
            if a[field] != b[field]:
                exact = False; mismatches.append(f'{views}_views:{field}')
        if a['equivalence']['skipped_update'] or b['equivalence']['skipped_update']:
            exact = False; mismatches.append(f'{views}_views:optimizer_update_was_skipped')
    exact = exact and baseline['weights_sha256'] == candidate['weights_sha256']
    result.update(exact_equivalence=exact, mismatches=mismatches,
        per_shape_speedup={str(n): baseline['shapes'][str(n)]['median_forward_backward_seconds'] /
                          candidate['shapes'][str(n)]['median_forward_backward_seconds'] for n in VIEWS})
    margin = min(s['estimated_min_device_headroom_bytes'] for s in candidate['shapes'].values())
    result['candidate_min_device_headroom_bytes'] = margin
    distribution = {int(n): count for n, count in plan['workload']['study_image_count_distribution'].items()}
    before, after = 0., 0.
    for views, count in distribution.items():
        if views <= 2:
            before += count * baseline['shapes'][str(views)]['median_forward_backward_seconds']
            after += count * candidate['shapes'][str(views)]['median_forward_backward_seconds']
        else:
            # Unmodified >=3-view execution is shared in numerator/denominator.
            # Intermediate counts use an explicit per-image estimate from the
            # measured seven-view fallback, not invented measured timings.
            unchanged = count * views / 7 * baseline['shapes']['7']['median_forward_backward_seconds']
            before += unchanged; after += unchanged
    updates = math.ceil(plan['workload']['study_count'] / plan['effective_studies'])
    update_cost = statistics.median(s['median_optimizer_ema_seconds'] for s in baseline['shapes'].values()) * updates
    before += update_cost; after += update_cost
    speedup = before / after
    saved = plan['baseline_training_epoch_seconds'] * (1 - 1 / speedup)
    result.update(weighted_speedup=speedup, predicted_seconds_saved_per_epoch=saved,
        net_first_epoch_saving=saved - elapsed, estimated_baseline_compute_seconds=before,
        estimated_candidate_compute_seconds=after,
        estimate_method='Training Study-count weights; 1/2 views measured; >=3 views unchanged, cost scaled from measured 7-view fallback; Adam/EMA cost at one per 32 Studies',
        estimate_caveat='Compute proxy only; excludes real loader, accumulation interactions, validation and saves; no completed-epoch speed claim')
    reasons = []
    if not exact: reasons.append('exact_equivalence_failed')
    if margin < MIN_FREE_BYTES: reasons.append('less_than_1_5_GiB_device_headroom')
    if speedup < 1.08: reasons.append('weighted_speedup_below_1_08')
    if saved <= elapsed + 300: reasons.append('one_epoch_saving_does_not_cover_probe_plus_300_seconds')
    result['adopt'] = not reasons
    result['state'] = 'adopt' if result['adopt'] else 'rejected'
    result['reason'] = 'all_exact_memory_speed_and_net_saving_gates_passed' if not reasons else ';'.join(reasons)
    return result


def parent(args):
    started = time.monotonic()
    from reconstruction import train_local as frozen
    from reconstruction import train_local_checkpoint_speed as adapter
    from reconstruction.data import read_records
    output = args.output.resolve()
    config_path, checkpoint = args.config.resolve(strict=True), args.checkpoint.resolve(strict=True)
    config = read_json(config_path)
    binding = adapter.load_profile(args.profile, args.profile_sha256)
    if (config['experiment_arm'] != 'global_local' or config['pack_images'] != 8
            or config['effective_studies'] != 32 or config['image_size'] != 768):
        raise ValueError('Benchmark only covers the current fixed E2 local configuration')
    if checkpoint != Path(config['output']).resolve() / 'last.pt':
        raise ValueError('Probe must initialize from the actual full local checkpoint')
    if output == Path(config['output']).resolve() or Path(config['output']).resolve() in output.parents:
        raise ValueError('Probe artifacts must stay outside the scientific training run')
    if args.execute:
        if read_json(Path(config['output']) / 'status.json').get('state') != 'paused':
            raise ValueError('Probe requires the original training run to be safely paused')
        frozen.verify_resume_lifecycle(Path(config['output']), config)
    history = frozen.base_controller_history(Path(config['output']))
    if not history or not 0 < history[-1]['seconds'] < 86400:
        raise ValueError('Need an actual complete training epoch time for net-saving calculation')
    inputs = [config_path, checkpoint, Path(config['train_manifest']), Path(config['split_plan']),
              Path(config['output']) / 'configuration.json', Path(config['output']) / 'history.jsonl',
              Path(binding['path']), Path(__file__).resolve(), *frozen.source_files(), *adapter.source_files()]
    plan = {'schema': SCHEMA, 'output': str(output), 'config_path': str(config_path),
        'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'profile_binding': binding, 'effective_studies': config['effective_studies'],
        'workload': workload(read_records(config['train_manifest'])),
        'baseline_training_epoch_seconds': history[-1]['seconds'],
        'input_sha256': {str(p): sha(p) for p in inputs}, 'scope': SCOPE}
    if not args.execute:
        print(json.dumps({'preview_only': True, 'plan': plan}, ensure_ascii=False)); return
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / 'PLAN.json'; write_json(plan_path, plan)
    reports = {}
    try:
        for mode in MODES:
            remaining = MAX_TOTAL_SECONDS - (time.monotonic() - started)
            if remaining <= 1: raise TimeoutError('Bounded checkpoint probe budget exhausted')
            command = [sys.executable, '-u', '-m', 'reconstruction.benchmark_checkpoint_speed',
                       '--worker', '--mode', mode, '--plan', str(plan_path)]
            with (output / f'{mode}.log').open('x') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=remaining, check=False)
            path = output / f'{mode}.json'
            if not path.is_file():
                raise RuntimeError(f'{mode} worker exited {completed.returncode} without a result')
            report = read_json(path); reports[mode] = report
            if completed.returncode and not (mode == 'retain_small' and completed.returncode == 3
                                             and report.get('outcome') == 'out_of_memory'):
                raise RuntimeError(f'{mode} worker failed: {report.get("error", completed.returncode)}')
        verify_files(plan)
        result = compare(plan, reports['original'], reports['retain_small'], time.monotonic() - started)
        result['worker_result_sha256'] = {mode: sha(output / f'{mode}.json') for mode in MODES}
        write_json(output / 'RESULT.json', result); write_json(output / 'result.json', result)
    except BaseException as error:
        result = {'schema': SCHEMA, 'complete': False, 'adopt': False, 'state': 'failed',
            'reason': f'{type(error).__name__}: {error}', 'checkpoint_sha256': plan['checkpoint_sha256'],
            'profile_binding': binding, 'baseline_valid': reports.get('original', {}).get('complete', False),
            'candidate_oom': False, 'total_benchmark_seconds': time.monotonic() - started}
        write_json(output / 'RESULT.json', result); write_json(output / 'result.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--profile-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument('--plan', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.plan or not args.mode or args.execute:
            parser.error('Worker requires one frozen plan and mode')
        try:
            _worker(args.plan, args.mode)
        except BaseException as error:
            import torch
            is_oom = isinstance(error, torch.OutOfMemoryError)
            result = {'schema': SCHEMA, 'mode': args.mode, 'complete': False,
                'outcome': 'out_of_memory' if is_oom else 'failed',
                'error': f'{type(error).__name__}: {error}', 'scope': SCOPE}
            write_json(Path(read_json(args.plan)['output']) / f'{args.mode}.json', result)
            traceback.print_exc()
            raise SystemExit(3 if is_oom else 1)
        return
    if any(getattr(args, k) is None for k in ['config', 'checkpoint', 'output', 'profile', 'profile_sha256']):
        parser.error('Parent requires config, checkpoint, output, profile and profile-sha256')
    parent(args)


if __name__ == '__main__':
    main()
