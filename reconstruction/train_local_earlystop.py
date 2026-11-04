"""Apply the user's patience-three budget amendment to the frozen E2 trainer.

The adapter checks only complete, durably recorded epochs. It writes a STOP
marker synchronously before the frozen trainer can enter the next epoch.
The trainer truthfully closes as paused; the amended controller separately
certifies policy completion. Historical lifecycle/configuration is untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

from reconstruction import early_stop_policy as policy_module
from reconstruction import train_local as trainer
from reconstruction import train_local_performance as performance
from training_hooks import _invocation

_BINDINGS_ACTIVE = False


def source_files():
    return [Path(__file__).resolve(), Path(policy_module.__file__).resolve()]


def _recorder():
    slot = _invocation.get()
    if slot is None or slot['recorder'] is None:
        raise RuntimeError('Stopping decisions require an active lifecycle session')
    return slot['recorder']


def history_decision(output, policy):
    baseline = trainer.controller._read_json(Path(output) / 'baseline.json')
    if baseline.get('epoch') != 0:
        raise ValueError('Stopping baseline must be the real untrained epoch zero')
    history = trainer.base_controller_history(Path(output))
    return policy_module.decision(baseline['macro_auc'], history, policy), history


def create_stop(output, record):
    """Publish a complete owned marker atomically; never overwrite user STOP."""
    output = Path(output)
    descriptor, temporary = tempfile.mkstemp(prefix='.early_stop_', dir=output)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, output / 'STOP')
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def stopping_adapter(config, binding):
    global _BINDINGS_ACTIVE
    if _BINDINGS_ACTIVE:
        raise RuntimeError('Nested stopping-policy adapters are not supported')
    policy_module.verify_binding(binding)
    policy = binding['policy']
    if (config.get('experiment_arm') not in policy['applies_to_arms']
            or config.get('epochs') != policy['max_epochs']
            or config.get('smoke_studies') != 0
            or config.get('min_delta') != policy['min_delta']):
        raise ValueError('Stopping amendment does not match the registered E2 arm')
    output = Path(config['output']).resolve()
    sources = {str(p): policy_module.sha(p) for p in source_files()}
    original_start, original_completed = trainer.start_run, trainer.epoch_completed

    def verify():
        policy_module.verify_binding(binding)
        if any(policy_module.sha(path) != expected for path, expected in sources.items()):
            raise ValueError('Stopping adapter source changed after binding')

    def recorded_start(run_dir, *, trainer: str, configuration, source_files=(), resume=None):
        verify()
        if Path(run_dir).resolve() != output:
            raise ValueError('Stopping adapter was called for another run')
        if resume is not None:
            prior, _ = history_decision(output, policy)
            if prior['stop']:
                raise ValueError('Amended budget already completed; refusing another training epoch')
        result = original_start(run_dir, trainer=trainer, configuration=configuration,
            source_files=[*source_files, *map(Path, sources), Path(binding['path'])], resume=resume)
        _recorder()._record('stopping_policy_activated', {
            'policy_binding': deepcopy(binding), 'source_sha256': deepcopy(sources),
            'adapter_source': str(Path(__file__).resolve()),
            'adapter_source_sha256': sources[str(Path(__file__).resolve())],
            'scientific_budget_amendment': True,
            'configuration_and_learning_rate_schedule_unchanged': True,
            'completed_epochs_on_entry': len(trainer_module_history(output)),
        })
        return result

    def recorded_completed(epoch, metrics):
        verify()
        original_completed(epoch, metrics)
        current, history = history_decision(output, policy)
        if current['completed_epochs'] != epoch or not history or history[-1] != dict(metrics):
            raise ValueError('Stopping check does not match the durable complete epoch')
        record = {
            'schema': 'cxr_e2_stopping_check_v1', 'stopping_policy': deepcopy(binding),
            'policy_sha256': binding['sha256'],
            'source_sha256': deepcopy(sources), 'decision': current,
            'history_sha256': policy_module.sha(output / 'history.jsonl'),
            'baseline_sha256': policy_module.sha(output / 'baseline.json'),
            'updated_at': trainer.base.now(),
            'stop_created': False, 'stop_sha256': None,
        }
        if current['stop'] and epoch < policy['max_epochs']:
            record['stop_created'] = create_stop(output, {
                'schema': 'cxr_e2_early_stopping_marker_v1',
                'stopping_policy_sha256': binding['sha256'], 'epoch': epoch,
                'reason': current['stop_reason'], 'history_sha256': record['history_sha256'],
            })
            if (output / 'STOP').exists():
                record['stop_sha256'] = policy_module.sha(output / 'STOP')
        trainer.base.write_json(output / 'early_stopping.json', record)
        _recorder()._record('early_stopping_check', record)

    _BINDINGS_ACTIVE = True
    trainer.start_run, trainer.epoch_completed = recorded_start, recorded_completed
    try:
        yield {'stopping_policy': deepcopy(binding), 'source_sha256': deepcopy(sources)}
    finally:
        trainer.start_run, trainer.epoch_completed = original_start, original_completed
        _BINDINGS_ACTIVE = False


def trainer_module_history(output):
    # Separate name avoids the recorded_start trainer-identity argument.
    return trainer.base_controller_history(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execution-profile', type=Path, required=True)
    parser.add_argument('--execution-profile-sha256', required=True)
    parser.add_argument('--stopping-policy', type=Path, required=True)
    parser.add_argument('--stopping-policy-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = trainer.controller._read_json(args.config)
    binding = policy_module.load_policy(args.stopping_policy, args.stopping_policy_sha256)
    with performance.profile_adapter(config, args.execution_profile, args.execution_profile_sha256) as execution:
        with stopping_adapter(config, binding):
            if not args.execute:
                trainer.validate_config(config); trainer.verify_parent(config)
                print(json.dumps({'state': 'preflight_passed', 'execution_profile': execution,
                    'stopping_policy': binding, 'model_execution': False}))
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
