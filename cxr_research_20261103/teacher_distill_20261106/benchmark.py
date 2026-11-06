"""Isolated complete-Study GPU throughput probes; no training run is resumed.

The default prints the proposed cases without constructing a model or reading
images.  --execute caches ten fixed TRAIN Studies (including the maximum-view
Study) and runs each configuration in its own child process.  Consequently an
OOM cannot leave optimizer/EMA tensors in the next case's CUDA context.

This measures model/update throughput on fixed decoded inputs, not end-to-end
DataLoader throughput or competition inference time.  No H/OOF images are read.
The first update has dropout disabled for output/loss/gradient diagnostics;
actual configured dropout is restored for one warmup and three steady updates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback


CASES = {
    'nested_chunk1_pack1': dict(chunk_checkpointing=True,
        block_checkpointing=True, view_chunk_size=1, pack_images=1),
    'blocks_chunk1_pack1': dict(chunk_checkpointing=False,
        block_checkpointing=True, view_chunk_size=1, pack_images=1),
    'blocks_chunk2_pack4': dict(chunk_checkpointing=False,
        block_checkpointing=True, view_chunk_size=2, pack_images=4),
    'blocks_chunk2_pack4_channels_last': dict(chunk_checkpointing=False,
        block_checkpointing=True, view_chunk_size=2, pack_images=4,
        channels_last=True),
}
DEFAULT_CASES = ('blocks_chunk1_pack1', 'blocks_chunk2_pack4')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def packs(studies, image_limit):
    """Preserve every view and permit one full Study to exceed the pack limit."""
    current, count = [], 0
    for study in studies:
        views = len(study['images'])
        if current and count + views > image_limit:
            yield current
            current, count = [], 0
        current.append(study)
        count += views
    if current:
        yield current


def prepare_input(config, size, destination):
    import numpy as np
    import torch
    from teacher_distill_20261106.data import StudyDataset, load_manifest

    rows = load_manifest(config['train_manifest'])
    dataset = StudyDataset(rows, config['image_root'], image_size=size,
                           training=False, seed=config['seed'])
    if len(dataset) < 10:
        raise ValueError('The fixed benchmark requires ten complete TRAIN Studies')
    largest = max(range(len(dataset)), key=lambda i: len(dataset.groups[i]))
    remaining = [i for i in range(len(dataset)) if i != largest]
    random.Random(config['seed']).shuffle(remaining)
    indices = [largest, *remaining[:9]]
    studies = []
    for index in indices:
        item = dataset[index]
        studies.append(dict(images=item['images'], targets=item['targets'],
            study_key=item['study_key'], image_ids=item['image_ids']))
    y = np.asarray([dataset.records[group[0]]['targets'] for group in dataset.groups], dtype=np.float64)
    prevalence = y.mean(0)
    if ((prevalence <= 0) | (prevalence >= 1)).any():
        raise ValueError('Each training class needs positive and negative support')
    weights = np.clip(np.sqrt((1 - prevalence) / prevalence), 1., 5.)
    payload = dict(studies=studies, weights=torch.from_numpy(weights).float(),
        bias=torch.from_numpy(np.log(prevalence / (1 - prevalence)) + np.log(weights)).float(),
        metadata=dict(image_size=size, studies=10,
            images=sum(len(s['images']) for s in studies),
            view_counts=[len(s['images']) for s in studies],
            max_training_views=len(dataset.groups[largest]),
            selected_study_keys=[s['study_key'] for s in studies],
            train_manifest_sha256=digest(config['train_manifest']),
            H_images_accessed=False, OOF_images_accessed=False,
            preprocessing='fixed decoded full-image input; no augmentation; compute probe only'))
    torch.save(payload, destination)
    return payload['metadata']


def worker(args):
    import numpy as np
    import torch
    from copy import deepcopy
    from reconstruction.train import update_ema
    from teacher_distill_20261106.model import TeacherStudyModel, study_bce_loss

    config = json.loads(Path(args.config).read_text())
    case = CASES[args.case]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(case=args.case, options=case, state='initializing',
                  H_images_accessed=False, OOF_images_accessed=False)
    model = ema = optimizer = payload = None
    try:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        payload = torch.load(args.input, map_location='cpu', weights_only=True)
        model = TeacherStudyModel(config['family'], config['pretrained_path'],
            image_size=args.image_size, drop_rate=config['drop_rate'],
            drop_path=config['drop_path'], gradient_checkpointing=case['chunk_checkpointing'],
            block_checkpointing=case['block_checkpointing'],
            view_chunk_size=case['view_chunk_size'],
            channels_last=case.get('channels_last', False)).cuda()
        with torch.no_grad():
            model.head[-1].bias.copy_(payload['bias'].cuda())
        ema = deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(
            backbone_lr=config['backbone_lr'], head_lr=config['head_lr'],
            weight_decay=config['weight_decay'], layer_decay=config['layer_decay']))
        scaler = torch.cuda.amp.GradScaler()
        weights = payload['weights'].cuda()
        report.update(metadata=payload['metadata'], device=torch.cuda.get_device_name(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            gradient_comparison='16 deterministic coordinates per trainable parameter + complete gradient norm; not an exhaustive tensor comparison')
        updates = 0

        def step(*, numerical=False):
            nonlocal updates
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            loss_total, predictions = 0., []
            for pack in packs(payload['studies'], case['pack_images']):
                images = [study['images'].cuda(non_blocking=False) for study in pack]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model.forward_studies(images)
                    losses = [study_bce_loss(logit, study['targets'].cuda(), weights)
                              for logit, study in zip(logits, pack)]
                    loss = torch.stack(losses).sum() / len(payload['studies'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite benchmark Study loss')
                loss_total += float(loss.detach())
                if numerical:
                    predictions.extend(logit.detach().sigmoid().cpu().numpy() for logit in logits)
                scaler.scale(loss).backward()
                del images, logits, losses, loss
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError('Nonfinite benchmark gradient')
            samples, names = [], []
            if numerical:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        flat = parameter.grad.detach().flatten()
                        indices = torch.linspace(0, len(flat) - 1,
                            min(16, len(flat)), device=flat.device).long()
                        samples.append(flat.index_select(0, indices).float())
                        names.append(name)
                np.savez_compressed(output/'numerical.npz', probabilities=np.asarray(predictions),
                    gradient_samples=torch.cat(samples).cpu().numpy(),
                    gradient_names=np.asarray(names), loss=np.asarray(loss_total),
                    unclipped_gradient_norm=grad_norm.detach().float().cpu().numpy())
            before_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < before_scale
            if skipped:
                raise FloatingPointError('AMP skipped a benchmark optimizer update')
            updates += 1
            update_ema(ema, model, updates, config['ema_decay'])
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            value = dict(seconds=seconds, studies_per_second=10 / seconds,
                images_per_second=payload['metadata']['images'] / seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                loss=loss_total, gradient_norm=float(grad_norm), optimizer_updates=updates,
                skipped_updates=0)
            print(json.dumps(dict(case=args.case, numerical=numerical, **value)), flush=True)
            return value

        # Only this first comparison disables random dropout/stochastic depth.
        changed = []
        for module in model.modules():
            for field in ('p', 'drop_prob'):
                if hasattr(module, field) and isinstance(getattr(module, field), (int, float)):
                    changed.append((module, field, getattr(module, field)))
                    setattr(module, field, 0.)
        report['numerical_update'] = step(numerical=True)
        for module, field, value in changed:
            setattr(module, field, value)
        del changed
        report['warmup_update'] = step()
        report['steady_updates'] = [step() for _ in range(3)]
        report['steady_mean_seconds'] = float(np.mean([s['seconds'] for s in report['steady_updates']]))
        report['steady_studies_per_second'] = 10 / report['steady_mean_seconds']
        report['peak_allocated_bytes'] = max(s['peak_allocated_bytes'] for s in report['steady_updates'])
        report['peak_reserved_bytes'] = max(s['peak_reserved_bytes'] for s in report['steady_updates'])
        report['state'] = 'completed'
    except torch.cuda.OutOfMemoryError as error:
        report.update(state='cuda_oom', error=str(error))
    except Exception as error:
        report.update(state='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        write_json(output/'report.json', report)
        # Process exit, not empty_cache alone, releases all case-owned context.
        model = ema = optimizer = payload = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if report['state'] == 'completed' else 2


def compare(reference, candidate):
    import numpy as np

    with np.load(reference/'numerical.npz') as base, np.load(candidate/'numerical.npz') as test:
        if not np.array_equal(base['gradient_names'], test['gradient_names']):
            return dict(passed=False, reason='gradient parameter identities differ')
        p_error = float(np.max(np.abs(base['probabilities'] - test['probabilities'])))
        g_error = float(np.max(np.abs(base['gradient_samples'] - test['gradient_samples'])))
        loss_error = float(abs(base['loss'] - test['loss']))
        norm_relative = float(abs(base['unclipped_gradient_norm'] - test['unclipped_gradient_norm'])
                              / max(float(base['unclipped_gradient_norm']), 1e-12))
        passed = (np.allclose(base['probabilities'], test['probabilities'], rtol=.002, atol=.002)
            and np.allclose(base['gradient_samples'], test['gradient_samples'], rtol=.05, atol=.0002)
            and loss_error <= .0002 and norm_relative <= .01)
        return dict(passed=bool(passed), max_probability_error=p_error,
            max_gradient_sample_error=g_error, absolute_loss_error=loss_error,
            relative_unclipped_gradient_norm_error=norm_relative,
            note='Diagnostic FP16 batch-kernel tolerance; no automatic scientific/model selection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-size', type=int, choices=(512, 768), default=768)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(DEFAULT_CASES))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--case', choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument('--input', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.case or not args.input:
            parser.error('Internal worker requires case and fixed input')
        return worker(args)
    if not args.execute:
        print(json.dumps(dict(executed=False, cases={name: CASES[name] for name in args.cases}, image_size=args.image_size,
            studies=10, forced_max_view_Study=True, numerical_updates=1,
            warmup_updates=1, steady_updates=3,
            input_coverage='train only; no H/OOF images; no sample/view deletion'), indent=2))
        return 0
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(config_sha256=digest(args.config), benchmark_source_sha256=digest(__file__),
        started_unix=time.time(), results={}, no_formal_run_checkpoint_reused=True)
    with tempfile.TemporaryDirectory(prefix='fixed_train_inputs_', dir=output) as temporary:
        fixed = Path(temporary)/'inputs.pt'
        metadata['input'] = prepare_input(config, args.image_size, fixed)
        reference_case = args.cases[0]
        for case in args.cases:
            case_dir = output/case
            case_dir.mkdir()
            argv = [sys.executable, '-u', '-m', 'teacher_distill_20261106.benchmark',
                '--worker', '--case', case, '--input', str(fixed),
                '--config', str(Path(args.config).resolve()), '--output', str(case_dir),
                '--image-size', str(args.image_size)]
            print(json.dumps(dict(case=case, state='starting_isolated_process')), flush=True)
            with (case_dir/'worker.log').open('w') as log:
                try:
                    process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.timeout_seconds, check=False)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    returncode = -9
                    write_json(case_dir/'report.json', dict(state='timeout', case=case))
            report_path = case_dir/'report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else dict(state='worker_exit_without_report')
            report['returncode'] = returncode
            if (case != reference_case and report['state'] == 'completed'
                    and metadata['results'].get(reference_case, {}).get('state') == 'completed'):
                report['numerical_alignment'] = compare(output/reference_case, case_dir)
                report['numerical_reference_case'] = reference_case
            metadata['results'][case] = report
            write_json(output/'summary.json', metadata)
            print(json.dumps(dict(case=case, state=report['state'],
                steady_mean_seconds=report.get('steady_mean_seconds'),
                peak_allocated_bytes=report.get('peak_allocated_bytes'),
                numerical_alignment=report.get('numerical_alignment'))), flush=True)
    metadata['completed_unix'] = time.time()
    write_json(output/'summary.json', metadata)
    return 0 if all(x['state'] == 'completed' for x in metadata['results'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
