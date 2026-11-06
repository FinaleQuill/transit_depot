"""One bounded, train-only activation-retention probe and audited E2 continuation.

The coordinator owns the existing queue lock, preserves every original plan,
and never retries the probe or training. Only its uniquely named temporary
probe container may be killed on timeout; training always uses epoch-boundary
STOP. An unsuccessful but valid candidate falls back to the original trainer.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from reconstruction import controller as base, controller_local as local
from reconstruction import controller_performance as perf, controller_earlystop as early
from reconstruction import runtime_local

SCHEMA = 'cxr_e2_checkpoint_execution_plan_v1'
REQUEST_SCHEMA = 'cxr_e2_checkpoint_speed_transition_request_v1'
BENCHMARK_SCHEMA = 'cxr_e2_checkpoint_speed_benchmark_v1'
PROFILE = {'schema': 'cxr_e2_checkpoint_execution_v1',
    'profile_id': 'e2_retain_small_packs_20261104', 'max_retained_regions': 10,
    'applies_to_arms': ['global_local'], 'scope': 'training_activation_retention_only'}
PROBE_TIMEOUT = 600
WAIT_TIMEOUT = 10800
CANCEL_FILE = 'CANCEL_CHECKPOINT_SPEED_TRANSITION'


def sources():
    return [Path(__file__).with_name(n).resolve() for n in
            ['controller_checkpoint_speed.py', 'train_local_checkpoint_speed.py', 'benchmark_checkpoint_speed.py']]


def profile_binding(path, expected_sha):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Checkpoint profile must be a regular immutable file')
    path = path.resolve(strict=True)
    profile = base._read_json(path)
    if (base.sha(path) != expected_sha or set(profile) != set(PROFILE) or
            any(type(profile[k]) is not type(v) or profile[k] != v for k, v in PROFILE.items())):
        raise ValueError('Checkpoint profile differs from the registered profile')
    return {'path': str(path), 'sha256': expected_sha, 'profile': profile}


def transition_status(directory, state, **details):
    base.write_json(Path(directory) / 'CHECKPOINT_SPEED_TRANSITION.json',
        {'state': state, 'pid': os.getpid(), 'updated_at': base.now(), **details})


def verify_request(directory, request):
    if (request.get('schema') != REQUEST_SCHEMA or request.get('boundary_epoch') != 2 or
            type(request.get('previous_controller_pid')) is not int):
        raise ValueError('Only the authorized local epoch-two transition is supported')
    if (Path(directory) / CANCEL_FILE).exists():
        raise RuntimeError('Checkpoint-speed transition canceled; training STOP retained')
    marker = Path(directory) / 'STOP'
    if not marker.is_file() or base.sha(marker) != request['requested_stop_sha256']:
        raise ValueError('Migration STOP changed; no automatic continuation')
    if set(request['source_sha256']) != {str(p) for p in sources()}:
        raise ValueError('Transition source inventory differs')
    for path, digest in request['source_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError('Transition source changed while waiting')
    profile_binding(request['profile_path'], request['profile_sha256'])


def wait_for_lock(lock, directory, request, *, timeout=WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while True:
        verify_request(directory, request)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Prior queue did not close within the bounded wait; no retry')
            transition_status(directory, 'waiting_for_complete_epoch', boundary_epoch=2,
                              previous_controller_pid=request['previous_controller_pid'])
            time.sleep(10)


def assert_no_e2_container():
    result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True,
                            text=True, check=True, timeout=20)
    if any(n.startswith('cxr-e2-') for n in result.stdout.splitlines()):
        raise RuntimeError('An E2 container is active; refusing duplicate execution')


def assert_gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    if result.stdout.strip():
        raise RuntimeError('GPU has another compute process; bounded probe requires exclusive access')


def load_plans(directory):
    directory = Path(directory)
    return tuple(base._read_json(directory / p) for p in
                 ['frozen_plan.json', 'performance_plan.json', 'early_stopping_plan.json'])


def verify_closed_boundary(directory, request, plan, performance, amendment):
    verify_request(directory, request)
    early.verify_early_plan(plan, amendment, performance)
    status = base._read_json(Path(directory) / 'STATUS.json')
    if (status.get('state') != 'paused' or status.get('active_container') is not None or
            status.get('controller_pid') != request['previous_controller_pid']):
        raise ValueError('The requested old controller has not closed as paused')
    if [j['name'] for j in plan['jobs']] != ['global_control', 'global_local']:
        raise ValueError('The original E2 queue changed')
    global_job, local_job = plan['jobs']
    certificates = [early.verify_stopped_run(global_job, amendment, performance),
                    early.verify_stopped_run(local_job, amendment, performance, require_complete=False)]
    if (certificates[0]['epochs'] != 3 or not certificates[0]['policy_complete'] or
            certificates[1]['epochs'] != 2 or certificates[1]['policy_complete'] or
            certificates[1]['trainer_state'] != 'paused'):
        raise ValueError('Transition must preserve completed global3 and paused local2')
    if not (Path(local_job['output']) / 'STOP').is_file():
        raise ValueError('Local epoch-boundary STOP evidence is absent')
    return certificates


def snapshot_transition(directory, request, plan, performance, amendment, certificates):
    destination = Path(directory) / 'checkpoint_speed_transition'
    if destination.exists():
        raise FileExistsError('Transition evidence already exists; inspect before any manual recovery')
    snapshots = {'request.json': (json.dumps(request, sort_keys=True, indent=2) + '\n').encode()}
    for name in ['STATUS.json', 'PROGRESS.md', 'STOP', 'frozen_plan.json',
                 'performance_plan.json', 'early_stopping_plan.json']:
        snapshots[f'controller_{name}'] = (Path(directory) / name).read_bytes()
    snapshots['profile.json'] = Path(request['profile_path']).read_bytes()
    for source in sources():
        snapshots[f'source_{source.name}'] = source.read_bytes()
    arms = {}
    for job, certificate in zip(plan['jobs'], certificates):
        output = Path(job['output'])
        events = local.closed_sessions(output)
        prefixes = {}
        for name in ['history.jsonl', 'compliance/events.jsonl', 'status.json', 'configuration.json', 'STOP']:
            data = (output / name).read_bytes()
            snapshots[f"{job['name']}_{name.replace('/', '_')}"] = data
            if name in {'history.jsonl', 'compliance/events.jsonl'}:
                prefixes[str(output / name)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        arms[job['name']] = {'session_ids': sorted({e['session_id'] for e in events}),
            'epochs': certificate['epochs'], 'certificate': certificate, 'prefix_files': prefixes}
    destination.mkdir()
    for name, data in snapshots.items():
        with (destination / name).open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    record = {'created_at': base.now(), 'request': request, 'transition_arms': arms,
        'source_sha256': request['source_sha256'],
        'snapshot_sha256': {str(destination / n): hashlib.sha256(b).hexdigest() for n, b in snapshots.items()},
        'prior_plan_sha256': {str(Path(directory) / n): base.sha(Path(directory) / n) for n in
                             ['frozen_plan.json', 'performance_plan.json', 'early_stopping_plan.json']}}
    base.write_json(destination / 'transition.json', record)
    return record


def kill_probe_only(name, child):
    if not name.startswith('cxr-e2-checkpoint-probe-'):
        raise ValueError('Only this bounded temporary probe may be killed')
    if base._running_container(name):
        subprocess.run(['docker', 'kill', name], capture_output=True, text=True, check=True, timeout=20)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        child.terminate()
        child.wait(timeout=20)
    if base._running_container(name):
        raise RuntimeError('Probe container remains active; training will not restart')


def benchmark_result(path, binding, checkpoint_sha):
    result = base._read_json(path)
    if (result.get('schema') != BENCHMARK_SCHEMA or result.get('complete') is not True or
            result.get('baseline_valid') is not True or type(result.get('adopt')) is not bool or
            result.get('profile_binding') != binding or result.get('checkpoint_sha256') != checkpoint_sha):
        raise ValueError('Benchmark is incomplete, failed, or bound to different inputs')
    if result['adopt']:
        if (result.get('candidate_oom') is not False or result.get('exact_equivalence') is not True or
                result.get('candidate_min_device_headroom_bytes', 0) < 1610612736 or
                result.get('weighted_speedup', 0) < 1.08 or
                result.get('predicted_seconds_saved_per_epoch', 0) <= result.get('total_benchmark_seconds', float('inf')) + 300):
            raise ValueError('Adopted benchmark does not meet the exactness, memory, speed and net-time gates')
    return result


def verify_benchmark_artifacts(benchmark):
    from reconstruction import benchmark_checkpoint_speed as probe
    path = Path(benchmark['path'])
    for artifact, digest in benchmark['artifact_sha256'].items():
        if Path(artifact).parent != path.parent or base.sha(artifact) != digest:
            raise ValueError('Frozen probe artifact changed')
    required = {str(path.parent / name) for name in
                ['PLAN.json', 'original.json', 'retain_small.json', 'RESULT.json', 'result.json', 'EXTERNAL_ACCEPTANCE.json']}
    if not required <= set(benchmark['artifact_sha256']):
        raise ValueError('Probe artifact inventory lacks complete worker evidence')
    result = base._read_json(path)
    plan = base._read_json(path.parent / 'PLAN.json')
    workers = {mode: base._read_json(path.parent / f'{mode}.json') for mode in probe.MODES}
    expected = probe.compare(plan, workers['original'], workers['retain_small'], result['total_benchmark_seconds'])
    expected['worker_result_sha256'] = {mode: base.sha(path.parent / f'{mode}.json') for mode in probe.MODES}
    if (expected != result or (path.parent / 'result.json').read_bytes() != path.read_bytes() or
            plan['checkpoint_sha256'] != result['checkpoint_sha256'] or
            plan['profile_binding'] != result['profile_binding'] or
            plan['input_sha256'].get(plan['checkpoint_path']) != result['checkpoint_sha256']):
        raise ValueError('Probe result does not replay from its exact bound worker evidence')
    acceptance = base._read_json(path.parent / 'EXTERNAL_ACCEPTANCE.json')
    if (acceptance != benchmark['acceptance'] or acceptance['benchmark_sha256'] != base.sha(path) or
            acceptance['controller_observed_seconds'] != benchmark['controller_observed_seconds'] or
            acceptance['adopt'] is not benchmark['adopt']):
        raise ValueError('External adoption evidence differs')
    expected_adopt = bool(result['adopt'] and result['predicted_seconds_saved_per_epoch'] >
                          benchmark['controller_observed_seconds'] + 300)
    if benchmark['adopt'] is not expected_adopt:
        raise ValueError('External adoption does not cover actual probe wall time')


def run_probe(directory, request, local_job, binding, checkpoint_sha):
    started = time.monotonic()
    output = Path(request['benchmark_output']).resolve()
    allowed_root = Path(directory).resolve().parent / 'checkpoint_speed_20261104'
    if output.parent != allowed_root or output.name != 'benchmark':
        raise ValueError('Probe outputs must use the dedicated checkpoint-speed benchmark directory')
    if output.exists():
        raise FileExistsError('Existing probe evidence cannot be overwritten or automatically retried')
    log_path = output.parent / 'benchmark.launch.log'
    if log_path.exists():
        raise FileExistsError('Prior benchmark launch exists; no automatic retry')
    name = f'cxr-e2-checkpoint-probe-{os.getpid()}-{time.time_ns()}'
    arguments = ['-m', 'reconstruction.benchmark_checkpoint_speed', '--config', local_job['config_path'],
        '--checkpoint', str(Path(local_job['output']) / 'last.pt'), '--output', str(output),
        '--profile', binding['path'], '--profile-sha256', binding['sha256'], '--execute']
    command = runtime_local.command(arguments, gpu=True, name=name)
    verify_request(directory, request)
    assert_no_e2_container()
    assert_gpu_idle()
    deadline = time.monotonic() + PROBE_TIMEOUT
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        transition_status(directory, 'benchmarking', probe_container=name, probe_pid=child.pid,
                          probe_log=str(log_path), timeout_seconds=PROBE_TIMEOUT)
        try:
            while child.poll() is None:
                verify_request(directory, request)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded GPU probe exceeded 600 seconds; no automatic retry')
                try:
                    child.wait(timeout=min(5, max(.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError(f'Probe failed with exit {child.returncode}; no training restarted')
        except BaseException:
            kill_probe_only(name, child)
            raise
        finally:
            log.flush(); os.fsync(log.fileno())
    if base._running_container(name):
        kill_probe_only(name, child)
        raise RuntimeError('Docker client exited before probe container; no training restarted')
    verify_request(directory, request)
    result_path = output / 'RESULT.json'
    result = benchmark_result(result_path, binding, checkpoint_sha)
    elapsed = time.monotonic() - started
    adopt = bool(result['adopt'] and result['predicted_seconds_saved_per_epoch'] > elapsed + 300)
    acceptance = {'adopt': adopt, 'benchmark_sha256': base.sha(result_path),
        'controller_observed_seconds': elapsed, 'minimum_net_saving_seconds': 300,
        'reason': 'all_probe_and_external_time_gates_passed' if adopt else
                  ('external_wall_time_gate_not_met' if result['adopt'] else 'probe_rejected_candidate')}
    base.write_json(output / 'EXTERNAL_ACCEPTANCE.json', acceptance)
    benchmark = {'path': str(result_path), 'sha256': base.sha(result_path), 'result': result,
        'controller_observed_seconds': elapsed, 'container': name, 'adopt': adopt,
        'acceptance': acceptance,
        'artifact_sha256': {str(p): base.sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    verify_benchmark_artifacts(benchmark)
    return benchmark


def create_execution_plan(directory, transition, binding, benchmark):
    destination = Path(directory) / 'checkpoint_execution_plan.json'
    if destination.exists() or benchmark['adopt'] is not True:
        raise ValueError('Only a fresh successful probe may create an execution plan')
    plan = {'schema': SCHEMA, 'created_at': base.now(), **transition,
            'checkpoint_profile': binding, 'benchmark': benchmark,
            'scientific_configuration_changed': False, 'historical_sessions_relabelled': False,
            'scope': 'local training activation retention after completed epoch 2'}
    base.write_json(destination, plan)
    return plan


def verify_execution_plan(plan, performance, amendment, execution):
    early.verify_early_plan(plan, amendment, performance)
    if execution.get('schema') != SCHEMA or execution.get('scientific_configuration_changed') is not False:
        raise ValueError('Unexpected checkpoint execution plan')
    binding = execution['checkpoint_profile']
    if profile_binding(binding['path'], binding['sha256']) != binding:
        raise ValueError('Checkpoint execution profile changed')
    if set(execution['source_sha256']) != {str(p) for p in sources()}:
        raise ValueError('Checkpoint execution source inventory changed')
    for group in ['source_sha256', 'snapshot_sha256', 'prior_plan_sha256']:
        for path, digest in execution[group].items():
            if base.sha(path) != digest:
                raise ValueError(f'Frozen checkpoint execution evidence changed: {path}')
    benchmark = execution['benchmark']
    if base.sha(benchmark['path']) != benchmark['sha256']:
        raise ValueError('Adoption benchmark changed')
    verify_benchmark_artifacts(benchmark)
    expected_checkpoint = execution['transition_arms']['global_local']['certificate']['checkpoint_sha256']['last.pt']
    if benchmark_result(benchmark['path'], binding, expected_checkpoint) != benchmark['result'] or not benchmark['adopt']:
        raise ValueError('Execution plan lacks successful bound adoption evidence')
    for arm in execution['transition_arms'].values():
        for path, prefix in arm['prefix_files'].items():
            if perf._prefix(path, prefix['bytes']) != prefix['sha256']:
                raise ValueError('Pre-transition training evidence changed')


def verify_checkpoint_sessions(job, execution):
    events = local.closed_sessions(job['output'])
    prior = set(execution['transition_arms'][job['name']]['session_ids'])
    groups = {}
    for event in events:
        groups.setdefault(event['session_id'], []).append(event)
    if not prior <= set(groups):
        raise ValueError('Pre-transition sessions disappeared')
    binding = execution['checkpoint_profile']
    adapter = str(Path(__file__).with_name('train_local_checkpoint_speed.py').resolve())
    expected_sha = execution['source_sha256'][adapter]
    verified = []
    history = base._history(Path(job['output']) / 'history.jsonl', strict=True)
    for sid, rows in groups.items():
        activated = [e for e in rows if e['event_type'] == 'checkpoint_execution_profile_activated']
        if sid in prior:
            if activated:
                raise ValueError('Historical session was relabelled with a checkpoint profile')
            continue
        if job['name'] != 'global_local' or len(activated) != 1:
            raise ValueError('Only new local sessions may activate exactly one checkpoint profile')
        event = activated[0]
        if (event.get('checkpoint_profile') != binding or event.get('adapter_source') != adapter or
                event.get('adapter_source_sha256') != expected_sha or
                event.get('scientific_configuration_changed') is not False or
                event.get('activation_retention_only') is not True or
                event.get('forward_packing_chunking_and_rng_order_unchanged') is not True):
            raise ValueError('Activated checkpoint profile differs from the frozen execution')
        output = Path(job['output'])
        manifest = base._read_json(output / 'compliance' / rows[0]['session_evidence'] / 'source_manifest.json')
        for path, digest in [(adapter, expected_sha), (binding['path'], binding['sha256'])]:
            if sum(r['original_path'] == path and r['sha256'] == digest for r in manifest) != 1:
                raise ValueError('Checkpoint adapter/profile lacks archived session evidence')
        completed = [e for e in rows if e['event_type'] == 'epoch_completed']
        counts = [e for e in rows if e['event_type'] == 'checkpoint_execution_epoch']
        if [e['epoch'] for e in counts] != [e['epoch'] for e in completed]:
            raise ValueError('Every closed complete epoch requires one activation execution count')
        for record, complete in zip(counts, completed):
            n = record['epoch']
            if not 3 <= n <= 6 or rows.index(record) >= rows.index(complete):
                raise ValueError('Checkpoint execution was not recorded before a post-transition epoch closed')
            c = record['counts']
            distribution = c['pack_region_counts']
            if (any(not str(k).isdigit() or int(k) < 1 or type(v) is not int or v < 1 for k, v in distribution.items()) or
                    record.get('checkpoint_profile_sha256') != binding['sha256'] or
                    record.get('checkpoint_profile_id') != PROFILE['profile_id'] or
                    record.get('complete_training_epoch') is not True or
                    record.get('evaluation_and_checkpoint_save_pending') is not True):
                raise ValueError('Malformed activation execution epoch')
            expected = {'total_packs': sum(distribution.values()),
                'total_regions': sum(int(k) * v for k, v in distribution.items()),
                'max_regions': max(map(int, distribution), default=0),
                'retained_packs': sum(v for k, v in distribution.items() if int(k) <= 10),
                'retained_regions': sum(int(k) * v for k, v in distribution.items() if int(k) <= 10),
                'checkpointed_packs': sum(v for k, v in distribution.items() if int(k) > 10),
                'checkpointed_regions': sum(int(k) * v for k, v in distribution.items() if int(k) > 10)}
            if (any(type(c.get(k)) is not int or c[k] != v for k, v in expected.items()) or
                    c['total_regions'] != complete['metrics']['regions'] or
                    c['total_regions'] != history[n - 1]['regions'] or c['retained_packs'] < 1):
                raise ValueError('Activation counts do not cover the actual complete epoch')
        verified.append({'session_id': sid, 'epochs': [e['epoch'] for e in completed],
                         'checkpoint_profile_sha256': binding['sha256']})
    return {'historical_sessions': sorted(prior), 'checkpoint_execution_sessions': verified}


class CheckpointSpeedController(early.EarlyStopController):
    def __init__(self, directory, plan, performance, amendment, execution, **kwargs):
        super().__init__(directory, plan, performance, amendment, **kwargs)
        self.execution = execution
        self.execution_sha256 = base.sha(self.directory / 'checkpoint_execution_plan.json')
        self.state['checkpoint_execution_profile'] = execution['checkpoint_profile']
        self.state['checkpoint_execution_plan'] = str(self.directory / 'checkpoint_execution_plan.json')

    def verify_frozen(self):
        if base.sha(self.directory / 'checkpoint_execution_plan.json') != self.execution_sha256:
            raise ValueError('Checkpoint execution plan changed')
        super().verify_frozen()
        verify_execution_plan(self.plan, self.performance, self.amendment, self.execution)

    def verify_job(self, job, *, require_complete=True):
        self.verify_frozen()
        result = early.verify_stopped_run(job, self.amendment, self.performance, require_complete=require_complete)
        result['checkpoint_execution_sessions'] = verify_checkpoint_sessions(job, self.execution)
        return result

    def launch(self, arguments, **kwargs):
        arguments = list(arguments)
        if kwargs['gpu']:
            local_job = self.plan['jobs'][1]
            if (arguments[:2] != ['-m', 'reconstruction.train_local'] or
                    arguments[arguments.index('--config') + 1] != local_job['config_path']):
                raise ValueError('Checkpoint continuation may only train the registered local arm')
            arguments[1] = 'reconstruction.train_local_checkpoint_speed'
            for option, binding in [('execution-profile', self.performance['execution_profile']),
                                    ('stopping-policy', self.amendment['policy_binding']),
                                    ('checkpoint-profile', self.execution['checkpoint_profile'])]:
                arguments += [f'--{option}', binding['path'], f'--{option}-sha256', binding['sha256']]
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
            result = self.verify_job(job, require_complete=False)
            if not result['policy_complete']:
                if job['name'] != 'global_local' or job['config_path'] not in self.resume_runs:
                    raise ValueError('Only explicit continuation of the existing local checkpoint is allowed')
                if (output / 'STOP').exists():
                    archive = self.directory / 'resume_stop_markers'
                    archive.mkdir(exist_ok=True)
                    os.replace(output / 'STOP', archive / f"local_checkpoint_speed_{time.time_ns()}.STOP")
                stamp = f'{int(time.time())}_{os.getpid()}'
                code = self.launch(['-m', 'reconstruction.train_local', '--config', job['config_path'], '--execute', '--resume'],
                    log_name=f'local_checkpoint_speed_{stamp}.log', gpu=True,
                    container_name=f'cxr-e2-global-local-checkpoint-{stamp}')
                if self.stop_requested():
                    self.publish('paused', stop_reason=self.stop_requested(), trainer_exit_code=code); return self.state
                if code:
                    raise RuntimeError(f'Trainer exited {code}; evidence retained; no automatic retry')
                result = self.verify_job(job)
            verified.append(result)
            base.write_json(self.directory / f"{job['name']}_verified.json", result)
            self.publish('verified', completed_jobs=list(verified))
            self.active_job = None
        if self.stop_requested():
            self.publish('paused', stop_reason=self.stop_requested()); return self.state
        self.publish('summarizing', current_job='paired_comparison', trainer_progress={})
        stamp = f'{int(time.time())}_{os.getpid()}'
        code = self.launch(['-m', 'reconstruction.controller_checkpoint_speed', '--controller-dir', str(self.directory), '--summarize'],
            log_name=f'comparison_checkpoint_speed_{stamp}.log', gpu=False,
            container_name=f'cxr-e2-compare-checkpoint-{stamp}')
        if self.stop_requested():
            self.publish('paused', stop_reason=self.stop_requested()); return self.state
        if code or not base._read_json(self.directory / 'comparison/report.json').get('complete'):
            raise RuntimeError('Paired comparison failed; no automatic retry')
        self.verify_frozen()
        self.publish('completed', current_job=None, comparison_report=str(self.directory / 'comparison/REPORT.md'))
        return self.state


def summarize(directory):
    directory = Path(directory)
    plan, performance, amendment = load_plans(directory)
    execution = base._read_json(directory / 'checkpoint_execution_plan.json')
    verify_execution_plan(plan, performance, amendment, execution)
    session_evidence = {j['name']: verify_checkpoint_sessions(j, execution) for j in plan['jobs']}
    # The frozen summary writes complete=True. Hold that flag false until the
    # additional execution certificates have also been published and rechecked.
    original_write = base.write_json
    report_path = directory / 'comparison/report.json'
    def hold_completion(path, value):
        if Path(path).resolve() == report_path.resolve() and value.get('complete') is True:
            value = {**value, 'complete': False, 'state': 'verifying_checkpoint_execution'}
        return original_write(path, value)
    base.write_json = hold_completion
    try:
        report = early.summarize(directory)
    finally:
        base.write_json = original_write
    verify_execution_plan(plan, performance, amendment, execution)
    after = {j['name']: verify_checkpoint_sessions(j, execution) for j in plan['jobs']}
    if after != session_evidence:
        raise ValueError('Checkpoint execution evidence changed during comparison')
    report.update(checkpoint_execution_profile=execution['checkpoint_profile'],
                  checkpoint_execution_sessions=session_evidence,
                  checkpoint_execution_plan_sha256=base.sha(directory / 'checkpoint_execution_plan.json'),
                  scientific_configuration_changed=False)
    with (directory / 'comparison/REPORT.md').open('a') as stream:
        stream.write('\nLocal epochs after completed epoch 2 use audited small-pack activation retention; '
                     'the training data, model, learning-rate schedule and stopping rule are unchanged.\n')
    original_write(report_path, report)
    return report


def run_selected_controller(directory, plan, performance, amendment, execution=None, *, resumes=()):
    if execution is None:
        controller = early.EarlyStopController(directory, plan, performance, amendment, resume_runs=resumes)
    else:
        controller = CheckpointSpeedController(directory, plan, performance, amendment, execution, resume_runs=resumes)
    handlers = {s: signal.signal(s, controller.signal_stop) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        result = controller.run()
    except BaseException as error:
        controller.publish('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if result['state'] != 'completed':
        raise SystemExit(2)
    return result


def transition(directory, request):
    plan, performance, amendment = load_plans(directory)
    certificates = verify_closed_boundary(directory, request, plan, performance, amendment)
    assert_no_e2_container()
    binding = profile_binding(request['profile_path'], request['profile_sha256'])
    transition_record = snapshot_transition(directory, request, plan, performance, amendment, certificates)
    benchmark = run_probe(directory, request, plan['jobs'][1], binding,
                          certificates[1]['checkpoint_sha256']['last.pt'])
    # Revalidate immutable plans and the exact closed checkpoint after the probe.
    verify_closed_boundary(directory, request, plan, performance, amendment)
    assert_no_e2_container()
    execution = None
    if benchmark['adopt']:
        execution = create_execution_plan(directory, transition_record, binding, benchmark)
        verify_execution_plan(plan, performance, amendment, execution)
    base.write_json(Path(directory) / 'checkpoint_speed_transition' / 'decision.json', {
        'adopt': execution is not None, 'benchmark': benchmark, 'decided_at': base.now(),
        'next_epoch': 3, 'fallback': 'original earlystop trainer' if execution is None else None})
    verify_request(directory, request)
    os.replace(Path(directory) / 'STOP', Path(directory) / 'checkpoint_speed_transition' / 'consumed_controller.STOP')
    transition_status(directory, 'activated' if execution else 'fallback_activated',
        benchmark_path=benchmark['path'], checkpoint_profile_adopted=execution is not None,
        next_epoch=3, no_training_epochs_repeated=True)
    return run_selected_controller(directory, plan, performance, amendment, execution,
                                   resumes=[plan['jobs'][1]['config_path']])


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
    if not args.execute or (args.transition_request and args.resume_run):
        parser.error('Explicit --execute is required; transition has its fixed local continuation')
    with (directory / '.controller.lock').open('a+b') as lock:
        if args.transition_request:
            request = base._read_json(args.transition_request)
            wait_for_lock(lock, directory, request)
            transition(directory, request)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / 'STOP').exists():
                raise ValueError('Controller STOP remains; manual inspection required')
            assert_no_e2_container()
            plan, performance, amendment = load_plans(directory)
            execution = base._read_json(directory / 'checkpoint_execution_plan.json')
            verify_execution_plan(plan, performance, amendment, execution)
            run_selected_controller(directory, plan, performance, amendment, execution, resumes=args.resume_run)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(f'CHECKPOINT_SPEED_CONTROLLER_EXIT: {type(error).__name__}: {error}', flush=True)
        if '--controller-dir' in sys.argv and '--summarize' not in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--controller-dir') + 1])
            transition_status(directory,
                'paused' if isinstance(error, SystemExit) and error.code == 2 else 'failed',
                error=f'{type(error).__name__}: {error}')
        raise
