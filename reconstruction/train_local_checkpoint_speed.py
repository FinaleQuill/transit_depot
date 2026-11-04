"""Bounded activation retention for small E2 local training packs.

Only activation storage versus backward recomputation changes. Pixel tensors,
Study packing, encoder chunk boundaries, model parameters, optimizer, loss and
stochastic forward order stay under the frozen E2 trainer. Adoption additionally
requires an external measured memory/throughput/gradient probe certificate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import torch

from reconstruction import train_local as trainer
from reconstruction import train_local_performance as performance
from reconstruction import train_local_earlystop as stopping
from reconstruction import early_stop_policy
from training_hooks import _invocation


PROFILE = {
    'schema': 'cxr_e2_checkpoint_execution_v1',
    'profile_id': 'e2_retain_small_packs_20261104',
    'max_retained_regions': 10,
    'applies_to_arms': ['global_local'],
    'scope': 'training_activation_retention_only',
}
_BINDINGS_ACTIVE = False


def source_files():
    return [Path(__file__).resolve()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != set(PROFILE):
        raise ValueError('Checkpoint profile has missing or unregistered fields')
    if any(type(profile[k]) is not type(v) or profile[k] != v for k, v in PROFILE.items()):
        raise ValueError('Only the registered small-pack activation-retention profile is allowed')
    return profile


def load_profile(path, expected_sha256=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Checkpoint profile must be a regular immutable file')
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError('Checkpoint profile must be a file')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
            or digest != expected_sha256):
        raise ValueError('Checkpoint profile differs from the bound SHA256')
    return {'path': str(path), 'sha256': digest,
            'profile': deepcopy(validate_profile(json.loads(content)))}


profile_binding = load_profile


def verify_binding(binding):
    if load_profile(binding['path'], binding['sha256']) != binding:
        raise ValueError('Checkpoint profile changed after binding')


def pack_regions(pack):
    if not isinstance(pack, (list, tuple)) or not pack:
        raise ValueError('Activation policy requires a nonempty complete Study pack')
    result = 0
    for batch in pack:
        pixels = batch['images']
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim not in (4, 5)
                or min(pixels.shape) < 1 or pixels.shape[-3] != 3):
            raise ValueError('Invalid complete Study pixels for activation policy')
        result += len(pixels) * (pixels.shape[1] if pixels.ndim == 5 else 1)
    return result


def _recorder():
    slot = _invocation.get()
    if slot is None or slot['recorder'] is None:
        raise RuntimeError('Checkpoint execution evidence requires an active lifecycle session')
    return slot['recorder']


@contextmanager
def checkpoint_adapter(config, binding):
    """Wrap forward storage only; original function calls and shapes are intact."""
    global _BINDINGS_ACTIVE
    if _BINDINGS_ACTIVE:
        raise RuntimeError('Nested checkpoint execution adapters are not supported')
    verify_binding(binding)
    if (config.get('experiment_arm') not in binding['profile']['applies_to_arms']
            or config.get('use_local') is not True
            or config.get('checkpoint_mode') != 'chunks'
            or config.get('view_chunk_size') != 8):
        raise ValueError('Activation profile does not match the frozen E2 local execution')
    output = Path(config['output']).resolve()
    source = source_files()[0]
    source_sha256 = sha(source)
    original_start = trainer.start_run
    original_forward = trainer.forward_pack
    original_epoch = trainer.train_epoch
    counters = None

    def verify():
        verify_binding(binding)
        if sha(source) != source_sha256:
            raise ValueError('Checkpoint execution adapter source changed after binding')

    def recorded_start(run_dir, *, trainer: str, configuration, source_files=(), resume=None):
        verify()
        if Path(run_dir).resolve() != output:
            raise ValueError('Checkpoint adapter was called for another run')
        result = original_start(run_dir, trainer=trainer, configuration=configuration,
            source_files=[*source_files, source, Path(binding['path'])], resume=resume)
        _recorder()._record('checkpoint_execution_profile_activated', {
            'checkpoint_profile': deepcopy(binding),
            'adapter_source': str(source), 'adapter_source_sha256': source_sha256,
            'scientific_configuration_changed': False,
            'forward_packing_chunking_and_rng_order_unchanged': True,
            'activation_retention_only': True,
        })
        return result

    def recorded_forward(model, pack, run_config, *, return_components=False, device='cuda'):
        if run_config != config:
            raise ValueError('Checkpoint adapter received a changed trainer configuration')
        active = model.training and torch.is_grad_enabled()
        if not active:
            return original_forward(model, pack, run_config,
                                    return_components=return_components, device=device)
        regions = pack_regions(pack)
        if (type(model.gradient_checkpointing) is not bool or model.gradient_checkpointing is not True
                or model.checkpoint_mode != 'chunks' or model.view_chunk_size != 8):
            raise ValueError('Model checkpoint policy differs from the bound frozen execution')
        retained = regions <= binding['profile']['max_retained_regions']
        previous = model.gradient_checkpointing
        if retained:
            model.gradient_checkpointing = False
        try:
            result = original_forward(model, pack, run_config,
                                      return_components=return_components, device=device)
        finally:
            model.gradient_checkpointing = previous
        if counters is not None:
            branch = 'retained' if retained else 'checkpointed'
            counters[f'{branch}_packs'] += 1
            counters[f'{branch}_regions'] += regions
            counters['total_packs'] += 1
            counters['total_regions'] += regions
            counters['max_regions'] = max(counters['max_regions'], regions)
            key = str(regions)
            counters['pack_region_counts'][key] = counters['pack_region_counts'].get(key, 0) + 1
        return result

    def recorded_epoch(model, ema, dataset, optimizer, scaler, weights, run_config, epoch, progress, run_output):
        nonlocal counters
        verify()
        if run_config != config or Path(run_output).resolve() != output:
            raise ValueError('Checkpoint epoch belongs to a different configuration or output')
        if counters is not None:
            raise RuntimeError('Nested training epochs are not supported')
        counters = {'retained_packs': 0, 'retained_regions': 0,
                    'checkpointed_packs': 0, 'checkpointed_regions': 0,
                    'total_packs': 0, 'total_regions': 0, 'max_regions': 0,
                    'pack_region_counts': {}}
        try:
            metrics = original_epoch(model, ema, dataset, optimizer, scaler, weights,
                                     run_config, epoch, progress, run_output)
            verify()
            if metrics['regions'] != counters['total_regions']:
                raise ValueError('Activation execution counts do not cover the complete training epoch')
            _recorder()._record('checkpoint_execution_epoch', {
                'epoch': epoch, 'checkpoint_profile_sha256': binding['sha256'],
                'checkpoint_profile_id': binding['profile']['profile_id'],
                'counts': deepcopy(counters), 'complete_training_epoch': True,
                'evaluation_and_checkpoint_save_pending': True,
            })
            return metrics
        finally:
            counters = None

    _BINDINGS_ACTIVE = True
    trainer.start_run, trainer.forward_pack, trainer.train_epoch = recorded_start, recorded_forward, recorded_epoch
    try:
        yield {'checkpoint_profile': deepcopy(binding),
               'adapter_source': str(source), 'adapter_source_sha256': source_sha256}
    finally:
        trainer.start_run, trainer.forward_pack, trainer.train_epoch = original_start, original_forward, original_epoch
        _BINDINGS_ACTIVE = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execution-profile', type=Path, required=True)
    parser.add_argument('--execution-profile-sha256', required=True)
    parser.add_argument('--stopping-policy', type=Path, required=True)
    parser.add_argument('--stopping-policy-sha256', required=True)
    parser.add_argument('--checkpoint-profile', type=Path, required=True)
    parser.add_argument('--checkpoint-profile-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = trainer.controller._read_json(args.config)
    stopping_binding = early_stop_policy.load_policy(args.stopping_policy, args.stopping_policy_sha256)
    checkpoint_binding = load_profile(args.checkpoint_profile, args.checkpoint_profile_sha256)
    with performance.profile_adapter(config, args.execution_profile, args.execution_profile_sha256) as execution:
        with stopping.stopping_adapter(config, stopping_binding):
            with checkpoint_adapter(config, checkpoint_binding) as checkpoint_execution:
                if not args.execute:
                    trainer.validate_config(config); trainer.verify_parent(config)
                    print(json.dumps({'state': 'preflight_passed', 'execution_profile': execution,
                        'stopping_policy': stopping_binding, 'checkpoint_execution': checkpoint_execution,
                        'model_execution': False}))
                    return
                try:
                    trainer.run(config, resume=args.resume)
                except BaseException as error:
                    output = Path(config['output']).resolve()
                    ledger = output / 'compliance/status.json'
                    if ledger.is_file():
                        status = trainer.controller._read_json(ledger)
                        if status.get('state') in {'failed', 'interrupted'}:
                            result = {'state': status['state'], 'error': f'{type(error).__name__}: {error}',
                                      'updated_at': trainer.base.now(), 'compliance_status': str(ledger)}
                            trainer.base.write_json(output / 'status.json', result)
                            trainer.base.write_json(output / 'progress.json', result)
                    raise


if __name__ == '__main__':
    main()
