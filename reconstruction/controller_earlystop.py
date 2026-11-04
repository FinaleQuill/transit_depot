"""User-authorized E2 patience-three amendment; preserve the original evidence."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from reconstruction import controller as base, controller_local as local
from reconstruction import controller_performance as perf
from reconstruction.early_stop_policy import load_policy, decision

SCHEMA = 'cxr_e2_early_stopping_amendment_v1'


def verified_baseline(output):
    baseline = base._read_json(Path(output) / 'baseline.json')
    report = base._read_json(Path(output) / 'epoch_00/report.json')
    if (baseline.get('epoch') != 0 or baseline.get('macro_auc') != report['macro_auc'] or
            baseline.get('predictions_sha256') != report['predictions_sha256']):
        raise ValueError('Stopping baseline differs from the bound epoch-zero evaluation')
    return baseline['macro_auc']


def sources():
    return [Path(__file__).with_name(n).resolve() for n in
            ['controller_earlystop.py', 'train_local_earlystop.py', 'early_stop_policy.py']]


def verify_early_plan(plan, amendment, performance):
    perf.verify_performance_plan(plan, performance)
    if amendment.get('schema') != SCHEMA:
        raise ValueError('Unknown early-stopping amendment')
    binding = amendment['policy_binding']
    if load_policy(binding['path'], binding['sha256']) != binding:
        raise ValueError('Stopping policy changed')
    for path, digest in amendment['source_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError('Frozen early-stopping source changed')
    if set(amendment['source_sha256']) != {str(p) for p in sources()}:
        raise ValueError('Early-stopping source inventory changed')
    for path, digest in amendment['snapshot_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError('Amendment evidence snapshot changed')
    for record in amendment['transition_arms'].values():
        for path, prefix in record.get('prefix_files', {}).items():
            if perf._prefix(path, prefix['bytes']) != prefix['sha256']:
                raise ValueError('Pre-amendment history changed')


def verify_policy_sessions(job, amendment, events):
    """Keep historical sessions unchanged; bind every newly executed session."""
    prior = set(amendment['transition_arms'][job['name']]['session_ids'])
    binding = amendment['policy_binding']
    output = Path(job['output'])
    groups = {}
    for event in events:
        groups.setdefault(event['session_id'], []).append(event)
    if not prior <= set(groups):
        raise ValueError('Pre-amendment lifecycle session disappeared')
    for sid, rows in groups.items():
        if sid in prior:
            continue
        activated = [r for r in rows if r['event_type'] == 'stopping_policy_activated']
        if len(activated) != 1 or activated[0].get('policy_binding') != binding:
            raise ValueError('New training session lacks the bound stopping policy')
        adapter = str(Path(__file__).with_name('train_local_earlystop.py').resolve())
        if (activated[0].get('adapter_source') != adapter or
                activated[0].get('adapter_source_sha256') != amendment['source_sha256'][adapter]):
            raise ValueError('Stopping adapter differs from amendment')
        folder = output / 'compliance' / rows[0]['session_evidence']
        manifest = base._read_json(folder / 'source_manifest.json')
        required = {binding['path']: binding['sha256'],
                    **{p: s for p, s in amendment['source_sha256'].items()
                       if Path(p).name != 'controller_earlystop.py'}}
        for path, digest in required.items():
            if sum(r['original_path'] == path and r['sha256'] == digest for r in manifest) != 1:
                raise ValueError('Stopping source or policy not archived in actual session')
        epochs = [r['epoch'] for r in rows if r['event_type'] == 'epoch_completed']
        checks = [r for r in rows if r['event_type'] == 'early_stopping_check']
        if [r['decision']['completed_epochs'] for r in checks] != epochs:
            raise ValueError('Completed epochs lack synchronous stopping decisions')
        history = base._history(output / 'history.jsonl', strict=True)
        baseline = verified_baseline(output)
        for check in checks:
            n = check['decision']['completed_epochs']
            if (check.get('policy_sha256') != binding['sha256'] or
                    check['decision'] != decision(baseline, history[:n], binding['policy'])):
                raise ValueError('Recorded stopping decision does not replay')


def verify_stopped_run(job, amendment, performance, *, require_complete=True):
    output = Path(job['output'])
    status = base._read_json(output / 'status.json')
    if status.get('state') not in {'paused', 'completed'}:
        raise ValueError('Failed or interrupted training is not a normal early stop')
    result = local.verify_run(job, require_complete=status['state'] == 'completed')
    result['execution_sessions'] = perf.verify_execution_sessions(job, performance)
    history = base._history(output / 'history.jsonl', strict=True)
    baseline = verified_baseline(output)
    assessment = decision(baseline, history, amendment['policy_binding']['policy'])
    if (status.get('completed_epochs') != len(history) or
            status.get('best_epoch') != assessment['best_epoch'] or
            status.get('best_macro_auc') != assessment['best_macro_auc'] or
            any(status.get(k) is not False for k in
                ['confirmation_evaluated', 'official_valid_evaluated', 'test_accessed'])):
        raise ValueError('Final status does not match full history and isolation evidence')
    events = local.closed_sessions(output)
    verify_policy_sessions(job, amendment, events)
    if assessment['first_stop_epoch'] is not None and assessment['first_stop_epoch'] < len(history):
        raise ValueError('Training continued past the registered stopping rule')
    stop_owned = False
    if assessment['stop'] and status['state'] == 'paused':
        transition = amendment['transition_arms'][job['name']]
        # This one authorized migration was requested while epoch 3 was running.
        # Do not retrospectively relabel its real paused trainer lifecycle.
        if (transition.get('existing') and transition.get('adopt_completed_boundary') is True and
                transition['epochs'] == len(history) and
                transition['ledger_sha256'] == base.sha(output / 'compliance/events.jsonl') and
                (output / 'STOP').is_file() and base.sha(output / 'STOP') == transition['stop_sha256']):
            stop_owned = True
        else:
            side_path = output / 'early_stopping.json'
            side = base._read_json(side_path) if side_path.exists() else {}
            checks = [e for e in events if e['event_type'] == 'early_stopping_check']
            fields = ['policy_sha256', 'decision', 'stop_created', 'stop_sha256',
                      'history_sha256', 'baseline_sha256']
            stop_owned = bool(side.get('stop_created') is True and
                side.get('policy_sha256') == amendment['policy_binding']['sha256'] and
                side.get('decision') == assessment and (output / 'STOP').is_file() and
                side.get('stop_sha256') == base.sha(output / 'STOP') and checks and
                all(checks[-1].get(k) == side.get(k) for k in fields) and
                side.get('history_sha256') == base.sha(output / 'history.jsonl') and
                side.get('baseline_sha256') == base.sha(output / 'baseline.json'))
            if stop_owned:
                marker = base._read_json(output / 'STOP')
                stop_owned = marker == {
                    'schema': 'cxr_e2_early_stopping_marker_v1',
                    'stopping_policy_sha256': amendment['policy_binding']['sha256'],
                    'epoch': len(history), 'reason': assessment['stop_reason'],
                    'history_sha256': base.sha(output / 'history.jsonl')}
        if not stop_owned:
            raise ValueError('External or unbound STOP cannot be claimed as policy completion')
    finished = assessment['stop'] and (status['state'] == 'completed' or stop_owned)
    if require_complete and not finished:
        raise ValueError('Stopping policy has not completed; explicit continuation required')
    result.update(policy_complete=bool(finished), trainer_state=status['state'],
                  policy_state='early_stopped' if stop_owned else status['state'],
                  stopping_policy_sha256=amendment['policy_binding']['sha256'],
                  stopping_decision=assessment, equal_update_budget=False)
    return result


def prepare_amendment(directory, plan, performance, binding, request):
    """Only under the old controller lock after its complete epoch boundary."""
    status = base._read_json(directory / 'STATUS.json')
    if status.get('state') != 'paused' or status.get('active_container') is not None:
        raise ValueError('Amendment requires the old controller to close as paused')
    if status.get('controller_pid') != request['previous_controller_pid']:
        raise ValueError('A different controller closed than the requested migration')
    if base.sha(directory / 'STOP') != request['requested_stop_sha256']:
        raise ValueError('Controller STOP differs from the authorized migration request')
    destination = directory / 'early_stopping_transition'
    if destination.exists():
        raise FileExistsError('Preserve existing amendment evidence; do not retry migration')
    perf.verify_performance_plan(plan, performance)
    snapshots = {f'source_{p.name}': p.read_bytes() for p in sources()}
    for name in ['STATUS.json', 'PROGRESS.md', 'STOP', 'frozen_plan.json', 'performance_plan.json']:
        snapshots[f'controller_{name}'] = (directory / name).read_bytes()
    snapshots['policy.json'] = Path(binding['path']).read_bytes()
    arms = {}
    for job in plan['jobs']:
        output = Path(job['output'])
        if not output.exists() or not any(output.iterdir()):
            arms[job['name']] = {'existing': False, 'session_ids': [], 'prefix_files': {}}
            continue
        if job['name'] != 'global_control':
            raise ValueError('This migration was authorized before the local arm started')
        verified = local.verify_run(job, require_complete=False)
        perf.verify_execution_sessions(job, performance)
        history = base._history(output / 'history.jsonl', strict=True)
        if len(history) != request['boundary_epoch']:
            raise ValueError('Migration closed at a different epoch than requested')
        events = local.closed_sessions(output)
        prefixes = {}
        for name in ['history.jsonl', 'compliance/events.jsonl', 'status.json', 'configuration.json', 'STOP']:
            data = (output / name).read_bytes()
            snapshots[f"{job['name']}_{name.replace('/', '_')}"] = data
            if name in {'history.jsonl', 'compliance/events.jsonl'}:
                prefixes[str(output / name)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        arms[job['name']] = {'existing': True, 'session_ids': sorted({r['session_id'] for r in events}),
            'epochs': len(history), 'ledger_sha256': base.sha(output / 'compliance/events.jsonl'),
            'stop_sha256': base.sha(output / 'STOP'), 'adopt_completed_boundary': True,
            'prefix_files': prefixes, 'verified_before_amendment': verified}
    amendment = {'schema': SCHEMA, 'created_at': base.now(), 'policy_binding': binding,
        'source_sha256': {str(p): base.sha(p) for p in sources()}, 'transition_arms': arms,
        'scientific_budget_changed': True, 'original_evidence_rewritten': False,
        'request': request, 'learning_rate_schedule': 'original six-epoch schedule, not compressed',
        'selection_caveat': 'User time-budget amendment during global epoch 3 after seeing epochs 1/2; exploratory',
        'snapshot_sha256': {str(destination / n): hashlib.sha256(b).hexdigest() for n, b in snapshots.items()}}
    destination.mkdir()
    for name, data in snapshots.items():
        with (destination / name).open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    base.write_json(directory / 'early_stopping_plan.json', amendment)
    verify_early_plan(plan, amendment, performance)
    return amendment


class EarlyStopController(perf.PerformanceController):
    def __init__(self, directory, plan, performance, amendment, **kwargs):
        super().__init__(directory, plan, performance, **kwargs)
        self.amendment = amendment
        self.amendment_sha256 = base.sha(self.directory / 'early_stopping_plan.json')
        self.state['stopping_policy'] = amendment['policy_binding']
        self.state['early_stopping_plan'] = str(self.directory / 'early_stopping_plan.json')
        self.state['equal_update_budget'] = False

    def verify_frozen(self):
        if base.sha(self.directory / 'early_stopping_plan.json') != self.amendment_sha256:
            raise ValueError('Early-stopping amendment changed')
        super().verify_frozen()
        verify_early_plan(self.plan, self.amendment, self.performance)

    def launch(self, arguments, **kwargs):
        arguments = list(arguments)
        if kwargs['gpu']:
            if arguments[:2] != ['-m', 'reconstruction.train_local']:
                raise ValueError('Only the registered E2 trainer may launch')
            execution = self.performance['execution_profile']
            binding = self.amendment['policy_binding']
            arguments[1] = 'reconstruction.train_local_earlystop'
            arguments += ['--execution-profile', execution['path'], '--execution-profile-sha256', execution['sha256'],
                          '--stopping-policy', binding['path'], '--stopping-policy-sha256', binding['sha256']]
        self.verify_frozen()
        return local.LocalController.launch(self, arguments, **kwargs)

    def run(self):
        verified = []
        for job in self.plan['jobs']:
            self.verify_frozen()
            if self.stop_requested():
                self.publish('paused', stop_reason=self.stop_requested()); return self.state
            self.active_job = job
            self.publish('checking', current_job=job['name'], trainer_progress={}, actual_epoch_auc=[])
            output = Path(job['output'])
            existing = output.exists() and any(output.iterdir())
            result = verify_stopped_run(job, self.amendment, self.performance, require_complete=False) if existing else None
            if not result or not result['policy_complete']:
                if existing:
                    if job['config_path'] not in self.resume_runs:
                        raise ValueError('Unfinished output needs explicit --resume-run')
                    if (output / 'STOP').exists():
                        archive = self.directory / 'resume_stop_markers'
                        archive.mkdir(exist_ok=True)
                        os.replace(output / 'STOP', archive / f"{job['name']}_patience3_{time.time_ns()}.STOP")
                stamp = f'{int(time.time())}_{os.getpid()}'
                args = ['-m', 'reconstruction.train_local', '--config', job['config_path'], '--execute']
                if existing:
                    args.append('--resume')
                code = self.launch(args, log_name=f"{job['name']}_patience3_{stamp}.log", gpu=True,
                    container_name=f"cxr-e2-{job['name'].replace('_','-')}-p3-{stamp}")
                if self.stop_requested():
                    self.publish('paused', stop_reason=self.stop_requested(), trainer_exit_code=code); return self.state
                if code:
                    raise RuntimeError(f'Trainer exited {code}; evidence preserved; no automatic retry')
                result = verify_stopped_run(job, self.amendment, self.performance)
            verified.append(result)
            base.write_json(self.directory / f"{job['name']}_verified.json", result)
            self.publish('verified', completed_jobs=list(verified))
            self.active_job = None
        if self.stop_requested():
            self.publish('paused', stop_reason=self.stop_requested()); return self.state
        self.publish('summarizing', current_job='paired_comparison', trainer_progress={})
        stamp = f'{int(time.time())}_{os.getpid()}'
        code = self.launch(['-m', 'reconstruction.controller_earlystop', '--controller-dir', str(self.directory), '--summarize'],
            log_name=f'comparison_patience3_{stamp}.log', gpu=False, container_name=f'cxr-e2-compare-p3-{stamp}')
        if self.stop_requested():
            self.publish('paused', stop_reason=self.stop_requested()); return self.state
        if code or not base._read_json(self.directory / 'comparison/report.json').get('complete'):
            raise RuntimeError('Paired comparison failed; no automatic retry')
        self.verify_frozen()
        self.publish('completed', current_job=None, comparison_report=str(self.directory / 'comparison/REPORT.md'))
        return self.state


def summarize(directory):
    import numpy as np
    from reconstruction.diagnostics import align_reference, load_bound_predictions
    from reconstruction.evaluate import save_evaluation
    directory = Path(directory)
    plan = base._read_json(directory / 'frozen_plan.json')
    performance = base._read_json(directory / 'performance_plan.json')
    amendment = base._read_json(directory / 'early_stopping_plan.json')
    verify_early_plan(plan, amendment, performance)
    certificates = [verify_stopped_run(j, amendment, performance) for j in plan['jobs']]
    comparison = directory / 'comparison'
    if comparison.exists() and any(comparison.iterdir()):
        raise FileExistsError('Existing comparison evidence must be inspected; no automatic overwrite/retry')
    comparison.mkdir(exist_ok=True)
    base.write_json(comparison / 'report.json', {'complete': False, 'state': 'summarizing',
        'stopping_policy': amendment['policy_binding'], 'started_at': base.now()})
    selected = [load_bound_predictions(Path(c['prediction_path']).parent) for c in certificates]
    initial = [load_bound_predictions(Path(j['output']) / 'epoch_00') for j in plan['jobs']]
    delta = np.abs(align_reference(initial[0], initial[1]) - initial[0]['probabilities'])
    if float(delta.max()) > plan['jobs'][0]['config']['epoch_zero_max_abs_tolerance']:
        raise ValueError('Initial arm predictions were not equivalent')
    summaries = {}
    for name, a, b in [('selected_local_vs_control', selected[0], selected[1]),
                       ('local_vs_initial', initial[1], selected[1]), ('control_vs_initial', initial[0], selected[0])]:
        summaries[name] = save_evaluation(directory / 'comparison' / name, align_reference(a, b), a['targets'],
            a['study_keys'], a['patient_ids'], a['views'], reference_probabilities=a['probabilities'],
            bootstrap_repeats=1000, seed=plan['jobs'][0]['config']['seed'], scope='inner_validation',
            metadata={'comparison': name, 'reference_binding': a['binding'], 'candidate_binding': b['binding'],
                      'stopping_policy': amendment['policy_binding'], 'equal_update_budget': False})
    report = {**summaries['selected_local_vs_control'], 'complete': True, 'certificates': certificates,
        'comparisons': summaries, 'epoch_zero_max_abs_delta': float(delta.max()),
        'stopping_policy': amendment['policy_binding'], 'equal_update_budget': False,
        'selection_caveat': 'Same-dev best selection and user budget amendment during global epoch 3; no selection-bias or seed correction',
        'interpretation': 'Same stopping rule, possibly unequal epochs/updates; local-input plus extra-head package',
        'followup_training_launched': False, 'deployment_selected': False}
    shutil.copyfile(directory / 'comparison/selected_local_vs_control/predictions.npz', directory / 'comparison/predictions.npz')
    if base.sha(directory / 'comparison/predictions.npz') != report['predictions_sha256']:
        raise ValueError('Primary report/prediction identity differs')
    lines = ['# E2 global/local paired review', '',
        'Both arms use patience=3, min_delta=0.0001, at most 6 epochs; epoch zero remains eligible.',
        'The time-budget amendment was requested during global epoch 3; actual update budgets can differ.', '',
        '| Arm | Actual epochs | Best epoch | Stop reason |', '| --- | ---: | ---: | --- |']
    for c in certificates:
        lines.append(f"| {c['name']} | {c['epochs']} | {c['best_epoch']} | {c['stopping_decision']['stop_reason']} |")
    lines += ['', '| Comparison | Reference AUC | Candidate AUC | Delta | Patient 95% CI |',
              '| --- | ---: | ---: | ---: | --- |']
    for name, item in summaries.items():
        s = item['overall']
        lines.append(f"| {name} | {s['reference']['macro_auc']:.8f} | {s['macro_auc']:.8f} | {s['macro_auc_delta']:+.8f} | {s['patient_bootstrap']['paired_delta_macro_auc']['ci95']} |")
    lines += ['', 'Exploratory same-development selection; patient intervals do not remove selection bias or training seed uncertainty.',
              'No H/official valid/test, E3/E4, submission, patient smoothing or deployment was executed.', '']
    (directory / 'comparison/REPORT.md').write_text('\n'.join(lines))
    base.write_json(directory / 'comparison/report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-dir', type=Path, required=True)
    parser.add_argument('--transition-request', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--resume-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    directory = base._inside(args.controller_dir, local.RUN_ROOT / 'E2_20261104')
    if args.summarize:
        if args.execute or args.transition_request or args.resume_run:
            parser.error('Summary cannot launch training')
        summarize(directory); return
    if not args.execute:
        parser.error('Explicit --execute is required')
    request = base._read_json(args.transition_request) if args.transition_request else None
    wait_path = directory / 'EARLY_STOP_TRANSITION.json'
    with (directory / '.controller.lock').open('a+b') as lock:
        if request:
            deadline = time.monotonic() + 7200
            while True:
                if (directory / 'CANCEL_EARLY_STOP_TRANSITION').exists():
                    raise RuntimeError('User canceled the pending transition; old STOP retained')
                for path, digest in request['source_sha256'].items():
                    if base.sha(path) != digest:
                        raise ValueError('Requested controller source changed while waiting')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Previous controller did not close within 2 hours; no retry')
                    if base.sha(directory / 'STOP') != request['requested_stop_sha256']:
                        raise ValueError('Migration STOP changed while waiting')
                    base.write_json(wait_path, {'state': 'waiting_for_complete_epoch', 'pid': os.getpid(),
                        'boundary_epoch': request['boundary_epoch'], 'updated_at': base.now()})
                    time.sleep(10)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if request and (directory / 'CANCEL_EARLY_STOP_TRANSITION').exists():
            raise RuntimeError('User canceled the pending transition; old STOP retained')
        active = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True, check=True, timeout=20)
        if any(n.startswith('cxr-e2-') for n in active.stdout.splitlines()):
            raise RuntimeError('An E2 container is active; refusing duplicate launch')
        plan = base._read_json(directory / 'frozen_plan.json')
        performance = base._read_json(directory / 'performance_plan.json')
        if request:
            binding = load_policy(request['policy_path'], request['policy_sha256'])
            amendment = prepare_amendment(directory, plan, performance, binding, request)
            # Archived exact marker authorizes this single migration, not future STOP requests.
            os.replace(directory / 'STOP', directory / 'early_stopping_transition' / 'consumed_controller.STOP')
        else:
            amendment = base._read_json(directory / 'early_stopping_plan.json')
            if (directory / 'STOP').exists():
                raise ValueError('Controller STOP remains; explicit inspection required before resume')
        verify_early_plan(plan, amendment, performance)
        resumes = args.resume_run or ([Path(plan['jobs'][0]['config_path'])] if request else [])
        controller = EarlyStopController(directory, plan, performance, amendment, resume_runs=resumes)
        handlers = {s: signal.signal(s, controller.signal_stop) for s in [signal.SIGTERM, signal.SIGINT]}
        try:
            base.write_json(wait_path, {'state': 'activated', 'pid': os.getpid(), 'updated_at': base.now(),
                'amendment_sha256': base.sha(directory / 'early_stopping_plan.json')})
            result = controller.run()
        except BaseException as error:
            controller.publish('failed', error=f'{type(error).__name__}: {error}'); raise
        finally:
            for s, h in handlers.items(): signal.signal(s, h)
        if result['state'] != 'completed':
            raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Startup failures leave the original controller state intact; launch log
        # and nonzero process exit preserve the failure without automatic retries.
        print(f'EARLY_STOP_CONTROLLER_EXIT: {type(error).__name__}: {error}', flush=True)
        if '--transition-request' in sys.argv and '--controller-dir' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--controller-dir') + 1])
            base.write_json(directory / 'EARLY_STOP_TRANSITION.json', {
                'state': 'paused' if isinstance(error, SystemExit) and error.code == 2 else 'failed',
                'pid': os.getpid(), 'updated_at': base.now(), 'error': f'{type(error).__name__}: {error}'})
        raise
