"""Audited CPU input-pipeline profile around the frozen E2 trainer.

Only loader concurrency and prefetch depth can change. The frozen research
configuration, checkpoint evidence, optimizer and RNG restoration stay under
``train_local.run``. Actual execution settings and this adapter are recorded
in each new lifecycle session, without rewriting historical evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from reconstruction import train_local as trainer
from training_hooks import _invocation

PROFILE_SCHEMA = 'cxr_e2_loader_execution_profile_v1'
PROFILE_KEYS = {'schema', 'profile_id', 'loader', 'applies_to_arms', 'scope'}
LOADER_FIXED = {'multiprocessing_context': 'spawn', 'in_order': True,
                'pin_memory': True, 'persistent_workers': False, 'timeout': 180}
LOADER_KEYS = set(LOADER_FIXED) | {'num_workers', 'prefetch_factor'}
IN_ORDER_ARGUMENT_SUPPORTED = 'in_order' in inspect.signature(DataLoader).parameters
_BINDINGS_ACTIVE = False


def source_files():
    return [Path(__file__).resolve()]


def validate_profile(profile):
    """Reject any option that could alter samples, pixels or training math."""
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
        raise ValueError('Execution profile has missing or unregistered fields')
    if profile['schema'] != PROFILE_SCHEMA or profile['scope'] != 'loader_execution_only':
        raise ValueError('Only the registered loader execution profile is supported')
    if (not isinstance(profile['profile_id'], str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', profile['profile_id']) is None):
        raise ValueError('Invalid execution profile identity')
    if profile['applies_to_arms'] != ['global_control', 'global_local']:
        raise ValueError('One execution profile must cover both matched arms')
    loader = profile['loader']
    if not isinstance(loader, dict) or set(loader) != LOADER_KEYS:
        raise ValueError('Loader profile has missing or unregistered fields')
    for key, expected in LOADER_FIXED.items():
        if type(loader[key]) is not type(expected) or loader[key] != expected:
            raise ValueError(f'Forbidden loader change: {key}')
    for key, upper in [('num_workers', 32), ('prefetch_factor', 8)]:
        if type(loader[key]) is not int or not 1 <= loader[key] <= upper:
            raise ValueError(f'Invalid bounded loader option: {key}')
    return profile


def load_profile(path, expected_sha256=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Execution profile must be a regular immutable file')
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError('Execution profile must be a file')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
            or digest != expected_sha256):
        raise ValueError('Execution profile differs from the bound SHA256')
    profile = validate_profile(json.loads(content))
    adapter = source_files()[0]
    return {'path': str(path), 'sha256': digest, 'profile': profile,
            'adapter_source': str(adapter),
            'adapter_source_sha256': trainer.base.digest(adapter)}


def verify_binding(binding):
    if (trainer.base.digest(binding['path']) != binding['sha256']
            or trainer.base.digest(binding['adapter_source']) != binding['adapter_source_sha256']):
        raise ValueError('Execution profile or adapter changed after binding')
    validate_profile(binding['profile'])


def build_loader(dataset, *, training, workers, seed, profile):
    """Preserve ordered sampling and per-epoch worker lifecycle exactly."""
    validate_profile(profile)
    if type(training) is not bool or type(workers) is not int or type(seed) is not int:
        raise ValueError('Invalid frozen loader call')
    options = deepcopy(profile['loader'])
    # Before PyTorch introduced this argument, its loader always returned
    # sampler order. The official E2 runtime supports an explicit True.
    if not IN_ORDER_ARGUMENT_SUPPORTED:
        options.pop('in_order')
    options.update(batch_size=None, shuffle=training,
                   generator=torch.Generator().manual_seed(seed),
                   worker_init_fn=trainer.base.seed_worker)
    loader = DataLoader(dataset, **options)
    actual = {'batch_size': loader.batch_size, 'shuffle': training,
              'num_workers': loader.num_workers, 'prefetch_factor': loader.prefetch_factor,
              'multiprocessing_context': loader.multiprocessing_context.get_start_method(),
              'in_order': getattr(loader, 'in_order', True), 'pin_memory': loader.pin_memory,
              'persistent_workers': loader.persistent_workers, 'timeout': loader.timeout,
              'worker_init_fn': 'reconstruction.train.seed_worker',
              'sampler': type(loader.sampler).__name__}
    if (actual['batch_size'] is not None
            or any(actual[k] != v for k, v in profile['loader'].items())
            or actual['sampler'] != ('RandomSampler' if training else 'SequentialSampler')):
        raise ValueError('Constructed loader differs from the registered execution profile')
    return loader, actual


@contextmanager
def profile_adapter(config, path, expected_sha256=None):
    """Temporarily bind only loader creation and session evidence capture."""
    global _BINDINGS_ACTIVE
    if _BINDINGS_ACTIVE:
        raise RuntimeError('Nested loader execution adapters are not supported')
    binding = load_profile(path, expected_sha256)
    if (config.get('experiment_arm') not in binding['profile']['applies_to_arms']
            or type(config.get('workers')) is not int or config['workers'] < 0):
        raise ValueError('Execution profile does not match this frozen trainer configuration')
    execution = {**binding, 'registered_workers': config['workers'],
                 'registered_prefetch_factor': 1,
                 'overridden_fields': ['loader.num_workers', 'loader.prefetch_factor'],
                 'in_order_argument_supported': IN_ORDER_ARGUMENT_SUPPORTED}
    original_loader, original_start = trainer.make_loader, trainer.start_run

    def recorded_start(run_dir, *, trainer: str, configuration, source_files=(), resume=None):
        verify_binding(binding)
        if 'execution_profile' in configuration:
            raise ValueError('Frozen scientific evidence must not contain execution overrides')
        actual_configuration = {**deepcopy(configuration), 'execution_profile': deepcopy(execution)}
        actual_sources = list(source_files) + [Path(binding['adapter_source']), Path(binding['path'])]
        return original_start(run_dir, trainer=trainer, configuration=actual_configuration,
                              source_files=actual_sources, resume=resume)

    def recorded_loader(dataset, *, training, workers, seed):
        verify_binding(binding)
        if workers != config['workers']:
            raise ValueError('Frozen trainer requested an unexpected worker setting')
        slot = _invocation.get()
        if slot is None or slot['recorder'] is None:
            raise RuntimeError('Actual loaders must be covered by an active lifecycle session')
        loader, actual = build_loader(dataset, training=training, workers=workers,
                                      seed=seed, profile=binding['profile'])
        slot['recorder']._record('data_loader_created', {
            'execution_profile_sha256': binding['sha256'],
            'execution_profile_id': binding['profile']['profile_id'],
            'registered_workers': workers, 'training': training,
            'dataset_epoch': getattr(dataset, 'epoch', None),
            'study_count': len(dataset), 'seed': seed, 'loader': actual})
        return loader

    _BINDINGS_ACTIVE = True
    trainer.make_loader, trainer.start_run = recorded_loader, recorded_start
    try:
        yield deepcopy(execution)
    finally:
        trainer.make_loader, trainer.start_run = original_loader, original_start
        _BINDINGS_ACTIVE = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execution-profile', type=Path, required=True)
    parser.add_argument('--execution-profile-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = trainer.controller._read_json(args.config)
    with profile_adapter(config, args.execution_profile, args.execution_profile_sha256) as binding:
        if not args.execute:
            trainer.validate_config(config)
            trainer.verify_parent(config)
            print(json.dumps({'state': 'preflight_passed', 'execution_profile': binding,
                              'model_execution': False, 'resume_requested': args.resume}))
            return
        try:
            trainer.run(config, resume=args.resume)
        except BaseException as error:
            # Preserve the frozen entrypoint's failed/interrupted status path.
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
