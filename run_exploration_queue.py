"""Durable serial, preregistered train-only exploration queue; never submits.

A new job starts only after the previous job's metrics and lifecycle ledger
verify. Missing preregistration or incomplete compliance is a hard stop.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parent
IMAGE = 'crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)

def now():
    return datetime.now(timezone.utc).isoformat()

def docker_command(script, arguments, *, gpu=True):
    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--ipc=host']
    if gpu:
        command += ['--gpus', 'all']
    command += ['--user', f'{os.getuid()}:{os.getgid()}',
        '-v', '/home/featurize:/home/featurize',
        '-v', '/home/featurize/data:/home/featurize/data:ro',
        '-v', str(ROOT / 'data') + ':' + str(ROOT / 'data') + ':ro',
        '-v', '/tmp/cxr_train_addons_20261103:/opt/cxr_train_addons:ro',
        '-w', str(ROOT), '-e', 'PYTHONPATH=/opt/cxr_train_addons:' + str(ROOT),
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'NO_ALBUMENTATIONS_UPDATE=1',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
        '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'TZ=Asia/Hong_Kong',
        '--entrypoint', 'python3', IMAGE, '-u', str(ROOT / script), *map(str, arguments)]
    return command

def preregistered(job):
    p = Path(job['preregistration'])
    if not p.is_file():
        raise RuntimeError('STOP: missing or changed preregistration: ' + job['name'])
    content = p.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != job['preregistration_sha256']:
        raise RuntimeError('STOP: missing or changed preregistration: ' + job['name'])
    lines = [v for v in content.decode('utf-8').splitlines() if v.strip()]
    if len(lines) != 4:
        raise RuntimeError('STOP: preregistration must contain four nonempty lines')
    return {'path': str(p), 'sha256': digest, 'observed_mtime_ns': p.stat().st_mtime_ns}

def verify_run(run_dir, *, allow_paused=False):
    from training_compliance import read_events
    run_dir = Path(run_dir)
    status = read(run_dir / 'status.json')
    terminal = {'completed', 'early_stopped', 'already_terminal'}
    if allow_paused:
        terminal.add('paused')
    if status.get('state') not in terminal:
        raise RuntimeError('STOP: training has not reached an accepted terminal state')
    events = read_events(run_dir / 'compliance/events.jsonl')
    starts = {r['session_id'] for r in events if r['event_type'] == 'training_started'}
    ends = {r['session_id'] for r in events if r['event_type'] == 'training_ended'}
    if not starts or starts != ends or events[-1]['event_type'] != 'training_ended':
        raise RuntimeError('STOP: incomplete compliance lifecycle')
    if events[-1]['state'] != status['state']:
        raise RuntimeError('STOP: lifecycle/status mismatch')
    history = [json.loads(v) for v in (run_dir / 'history.jsonl').read_text().splitlines()]
    epochs = {r['epoch'] for r in events if r['event_type'] == 'epoch_completed'}
    if not history or {r['epoch'] for r in history} != set(range(1, status['completed_epochs']+1)) or epochs != {r['epoch'] for r in history}:
        raise RuntimeError('STOP: epoch metrics/compliance coverage incomplete')
    checkpoints = {}
    for name in ('best', 'last'):
        path = run_dir / (name + '.pt')
        digest = sha(path)
        if digest != status[name + '_checkpoint_sha256']:
            raise RuntimeError('STOP: checkpoint/status SHA mismatch')
        if not any(r['event_type'] == 'checkpoint_saved' and r['checkpoint_sha256'] == digest for r in events):
            raise RuntimeError('STOP: checkpoint lacks compliance save event')
        checkpoints[name] = {'path': str(path), 'sha256': digest}
    return {'verified_at': now(), 'state': status['state'], 'completed_epochs': status['completed_epochs'],
            'history_sha256': sha(run_dir / 'history.jsonl'), 'ledger_sha256': sha(run_dir / 'compliance/events.jsonl'),
            'ledger_end_record_sha256': events[-1]['record_sha256'], 'checkpoints': checkpoints}


def verify_job_binding(job, receipt):
    """Bind a terminal run to preregistered critical config and all captured sources."""
    run_dir = Path(job['run_dir'])
    expected = job.get('expected_configuration')
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError('STOP: training job lacks a registered expected configuration')
    observed = read(run_dir/'config.json')
    for key, value in expected.items():
        if observed.get(key) != value:
            raise RuntimeError(f'STOP: run config differs from registration for {key}')
    from training_compliance import read_events
    events = read_events(run_dir/'compliance/events.jsonl')
    starts = [event for event in events if event['event_type'] == 'training_started']
    if not starts:
        raise RuntimeError('STOP: training lifecycle has no start record')
    expected_source_sha = job.get('expected_source_sha256')
    if not isinstance(expected_source_sha, dict) or not expected_source_sha:
        raise RuntimeError('STOP: training job lacks a registered expected source identity')
    expected_sources = {str(Path(path).resolve()): digest for path, digest in expected_source_sha.items()}
    for event in starts:
        session_dir = (run_dir/'compliance'/event['session_evidence']).resolve(strict=True)
        if not session_dir.is_relative_to((run_dir/'compliance'/'sessions').resolve(strict=True)):
            raise RuntimeError('STOP: compliance session evidence path escapes its run')
        config_path = session_dir/'configuration.json'
        source_manifest = session_dir/'source_manifest.json'
        if (sha(config_path) != event['configuration_sha256']
                or sha(source_manifest) != event['source_manifest_sha256']):
            raise RuntimeError('STOP: run configuration/source snapshot does not match compliance record')
        captured = read(config_path)
        for key, value in expected.items():
            if captured.get(key) != value:
                raise RuntimeError(f'STOP: compliance configuration differs from registration for {key}')
        source_rows = read(source_manifest)
        captured_sources = {str(Path(row['original_path']).resolve()): row['sha256'] for row in source_rows}
        if captured_sources != expected_sources:
            raise RuntimeError('STOP: compliance source files differ from registered source identity')
    expected_warm_start = job.get('expected_warm_start_sha256')
    if expected_warm_start:
        warm_start = observed.get('student_warm_start') or {}
        if warm_start.get('source_snapshot_sha256') != expected_warm_start:
            raise RuntimeError('STOP: student initialization snapshot differs from registration')
    latest = starts[-1]
    return {'configuration_sha256': latest['configuration_sha256'],
            'source_manifest_sha256': latest['source_manifest_sha256'],
            'session_id': latest['session_id']}

def active_run(run_dir, config_path=None, *, proc_root='/proc'):
    """Inspect live commands without depending on a stale PID file."""
    python_pids = []
    other_pids = []
    for p in Path(proc_root).iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        identity = str(run_dir).encode() in args or config_path and str(config_path).encode() in args
        if identity and b'--execute' in args and any(a.endswith(b'train_distill.py') or a.endswith(b'exploration_train.py') for a in args):
            (python_pids if args[0].endswith(b'python3') else other_pids).append(int(p.name))
    return (python_pids or other_pids or [None])[0]


def require_timing_evidence(report, mode, label):
    timing = ((report.get('timing') or {}).get('modes') or {}).get(mode)
    if not timing:
        raise RuntimeError(f'STOP: {label} timing record is missing for {mode}')
    p50, p95 = timing.get('p50_ms'), timing.get('p95_ms')
    if (not isinstance(p50, (int, float)) or not math.isfinite(p50) or p50 < 0
            or not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 < p50
            or type(timing.get('observations')) is not int or timing['observations'] != 90):
        raise RuntimeError(f'STOP: {label} timing metrics are invalid for {mode}')
    return timing


def safe_read(path):
    try:
        return read(path)
    except FileNotFoundError:
        return None


def progress(job):
    if job.get('kind') != 'train':
        return None
    run_dir = Path(job['run_dir'])
    status = safe_read(run_dir / 'status.json') or {}
    history_path = run_dir / 'history.jsonl'
    history = [json.loads(v) for v in history_path.read_text().splitlines() if v.strip()] if history_path.exists() else []
    logs = list(run_dir.glob('console_*.log'))
    if (run_dir/'evidence').exists():
        logs.extend((run_dir/'evidence').glob('console_*.log'))
    logs = sorted(logs, key=lambda p:p.stat().st_mtime)
    batch = None
    if logs:
        with logs[-1].open('rb') as stream:
            stream.seek(max(0, logs[-1].stat().st_size-8192))
            matches = re.findall(rb'student epoch=(\d+) batch=(\d+)/(\d+)', stream.read())
        if matches:
            epoch, done, total = map(int, matches[-1])
            batch = {'epoch': epoch, 'completed_batches': done, 'total_batches': total}
    return {'state': status.get('state', 'not_started'), 'completed_epochs': status.get('completed_epochs', 0),
            'target_epochs': job.get('epochs', 3), 'best_auc': status.get('best_auc'), 'best_epoch': status.get('best_epoch'),
            'current_batch': batch, 'current_log':str(logs[-1]) if logs else None,
            'history': [{'epoch':r['epoch'], 'macro_auc':r.get('validation',{}).get('macro_auc'),
               'per_class_auc':r.get('validation',{}).get('per_class_auc'), 'training_seconds':r.get('training',{}).get('seconds')} for r in history]}


class Queue:
    def __init__(self, plan_path):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read(plan_path)
        self.root = Path(self.plan['output'])
        self.status_path = Path(self.plan.get('status_file', self.root/'STATUS.json'))
        previous = safe_read(self.status_path)
        if previous and previous.get('state') in {'stopped', 'failed', 'FAILED'}:
            write(self.root/'attempts'/'previous_controller_attempt.json', previous)
        self.status = {'state': 'STARTING', 'controller_pid': os.getpid(), 'started_utc': now(),
                       'plan': str(self.plan_path), 'plan_sha256': sha(plan_path), 'current_job': None,
                       'completed_jobs': [], 'completed_metrics': {}, 'compliance_records': {}, 'failure_reason': None,
                       'no_test': True, 'no_submission': True, 'no_declaration_changes': True}
        self.index = 0
        self.phase_started = time.time()
        for name in ('logs', 'launches', 'receipts', 'failures'):
            (self.root/name).mkdir(parents=True, exist_ok=True)

    def heartbeat(self, phase=None):
        job = self.plan['jobs'][self.index] if self.index < len(self.plan['jobs']) else None
        if phase:
            self.status['phase'] = phase
        try:
            p = progress(job) if job else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            p = {'state': 'progress_unavailable', 'error': f'{type(error).__name__}: {error}'}
        remaining = 0
        for idx, pending in enumerate(self.plan['jobs'][self.index:], self.index):
            estimate = pending.get('estimated_seconds', 600)
            if idx == self.index:
                if p and 'completed_epochs' in p:
                    completed = p['completed_epochs']
                    fraction = 0
                    batch = p['current_batch']
                    if batch and batch['epoch'] > completed:
                        fraction = .9*batch['completed_batches']/batch['total_batches']
                    seconds = [v['training_seconds'] for v in p['history'] if v['training_seconds']]
                    each = sum(seconds)/len(seconds)+180 if seconds else estimate/p['target_epochs']
                    estimate = max(0, p['target_epochs']-completed-fraction)*each
                else:
                    estimate = max(30, estimate-(time.time()-self.phase_started))
            remaining += estimate
        active = self.status['state'] not in {'FAILED', 'COMPLETED'}
        self.status.update(updated_utc=now(), current_progress=p,
            estimated_finish_utc=(datetime.now(timezone.utc)+timedelta(seconds=remaining)).isoformat() if active else None,
            estimated_remaining_hours=round(remaining/3600,2) if active else None,
            eta_basis='Observed training epoch speed plus validation; later steps use conservative plan estimates; not a deadline',
            current_arm=job['name'] if job else None, current_job=job['name'] if job else None,
            current_log=(p.get('current_log') or str(self.root/'logs'/(job['name']+'.log'))) if job else None)
        write(self.status_path, self.status)
        write(self.root/'status.json', self.status)
        lines = [f"State: {self.status['state']}", f"Updated UTC: {self.status['updated_utc']}",
                 f"Current: {self.status['current_job']} / {self.status.get('phase')}",
                 f"ETA UTC: {self.status['estimated_finish_utc']} (estimate only)",
                 f"Failure: {self.status['failure_reason']}", f"Log: {self.status['current_log']}",
                 '', 'Completed metrics:', json.dumps(self.status['completed_metrics'], ensure_ascii=False, indent=2)]
        self.status_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')

    def check_stop(self):
        if (self.root/'STOP').exists() or self.status_path.with_name('STOP').exists():
            raise RuntimeError('User STOP marker: queue will not advance')
        if sha(self.plan_path) != self.status['plan_sha256']:
            raise RuntimeError('Queue plan changed after launch')
        for filename, digest in self.plan.get('source_sha256', {}).items():
            if sha(ROOT/filename) != digest:
                raise RuntimeError('Registered source changed during queue: '+filename)
        pre = self.status.get('current_preregistration')
        if pre and sha(pre['path']) != pre['sha256']:
            raise RuntimeError('Preregistration changed after arm registration')

    def validate_plan(self):
        if 'status_file' not in self.plan:
            raise RuntimeError('Queue plan requires status_file for durable controller state')
        jobs = self.plan.get('jobs')
        if not isinstance(jobs, list):
            raise RuntimeError('Queue plan jobs must be a list')
        names = set()
        allowed = {'train', 'evaluate', 'compare', 'report'}
        for job in jobs:
            if not isinstance(job, dict) or job.get('kind') not in allowed:
                raise RuntimeError('Queue job must declare a supported kind')
            name = job.get('name')
            if not isinstance(name, str) or not name or name in names:
                raise RuntimeError('Queue job names must be nonempty and unique')
            names.add(name)

    def command(self, job, command, preregistration):
        self.check_stop()
        record = self.root/'launches'/(job['name']+'_'+str(time.time_ns())+'.json')
        write(record, {'launched_at':now(), 'command':command, 'preregistration':preregistration})
        self.status['launch_record'] = str(record)
        self.status['state'] = 'RUNNING'
        with (self.root/'logs'/(job['name']+'.log')).open('a') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            self.status['child_pid'] = child.pid
            try:
                while child.poll() is None:
                    self.check_stop()
                    self.heartbeat()
                    time.sleep(20)
                self.check_stop()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            self.status['child_pid'] = None
            if child.returncode:
                raise RuntimeError(f'Command exit {child.returncode}: {job["name"]}; see {self.root/"logs"/(job["name"]+".log")}')

    def training(self, job, pre):
        run_dir = Path(job['run_dir'])
        existing_pid = active_run(run_dir, job.get('config_path'))
        if existing_pid:
            self.status['state'] = 'RUNNING'
            self.status['adopted_existing_training'] = str(run_dir)
            while existing_pid:
                try:
                    self.check_stop()
                except BaseException:
                    try:
                        os.kill(existing_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 1800
                    while active_run(run_dir, job.get('config_path')) and time.monotonic() < deadline:
                        time.sleep(20)
                    raise
                self.heartbeat('watching_existing_training')
                time.sleep(20)
                existing_pid = active_run(run_dir, job.get('config_path'))
        status = safe_read(run_dir/'status.json')
        if status and status['state'] in {'completed','early_stopped','already_terminal'}:
            receipt = verify_run(run_dir)
        else:
            args = list(job['arguments'])
            if status:
                if status['state'] != 'paused' or not job.get('allow_existing_pause_resume'):
                    raise RuntimeError('Existing nonterminal run needs explicit recovery: '+job['name'])
                verify_run(run_dir, allow_paused=True)
                for flag in ('--init-student-ema','--init-student-ema-sha256'):
                    if flag in args:
                        idx=args.index(flag); del args[idx:idx+2]
                args += ['--resume',str(run_dir/'last.pt')]
            self.heartbeat('training')
            self.command(job, docker_command(job['script'],args+['--execute']),pre)
            receipt=verify_run(run_dir)
        if receipt['completed_epochs'] != job.get('epochs',3):
            raise RuntimeError('Fixed-budget run did not complete every preregistered epoch')
        # Registration must predate the first real lifecycle start, including adopted runs.
        from training_compliance import read_events
        events=read_events(run_dir/'compliance/events.jsonl')
        first=next(e for e in events if e['event_type']=='training_started')
        if pre['observed_mtime_ns']/1e9 >= datetime.fromisoformat(first['recorded_at_utc']).timestamp():
            raise RuntimeError('Preregistration was not written before training start')
        receipt['job_binding'] = verify_job_binding(job, receipt)
        self.status['completed_metrics'][job['name']] = progress(job)
        return receipt

    def evaluation(self, job, pre):
        run_receipt = verify_run(job['run_dir']) if job.get('run_dir') else None
        checkpoint = Path(job['checkpoint'])
        digest = run_receipt['checkpoints'][job['checkpoint_role']]['sha256'] if run_receipt else job['checkpoint_sha256']
        output=Path(job['output'])
        args=['evaluate','--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--output',str(output),
              '--train-gap-studies',str(job.get('train_gap_studies',0))]
        if job.get('timing',True): args+=['--timing']
        if job.get('roi'): args+=['--roi']
        if job.get('mil'): args+=['--mil']
        self.heartbeat('evaluating_valid_and_timing')
        self.command(job,docker_command('exploration_evaluate.py',args),pre)
        report=read(output/'report.json'); metadata=read(output/'metadata.json')
        if not report.get('complete') or report.get('test_accessed') is not False or metadata['source_checkpoint_sha256']!=digest:
            raise RuntimeError('Evaluation compliance incomplete')
        if report['metadata_sha256'] != sha(output/'metadata.json'):
            raise RuntimeError('Evaluation metadata SHA mismatch')
        for filename, expected in metadata['prediction_files_sha256'].items():
            if Path(filename).name!=filename or sha(output/filename)!=expected:
                raise RuntimeError('Prediction artifact SHA mismatch')
            if Path(filename).suffix.lower() == '.csv':
                raise RuntimeError('Unexpected CSV artifact in this exploration round')
        gap = report.get('sampled_train_gap')
        if job.get('train_gap_studies', 0):
            sample_path = output/'sampled_train_predictions.npz'
            if (not gap or gap.get('num_studies') != job['train_gap_studies']
                    or sha(sample_path) != gap.get('prediction_sha256')):
                raise RuntimeError('Sampled-train gap evidence is incomplete or changed')
        if job.get('roi'):
            diagnostics_path = output/'roi_diagnostics.json'
            diagnostics = read(diagnostics_path) if diagnostics_path.is_file() else []
            if (metadata.get('experiment', {}).get('roi_version') != job.get('roi_version')
                    or len(diagnostics) != 1422
                    or any(item.get('effective_magnification', 1.) < 1.
                           for item in diagnostics)):
                raise RuntimeError('ROI evaluation geometry evidence is incomplete or shrinks input')
        if job.get('mil'):
            maps = list((output/'attention').glob('class_*.npy'))
            images = list((output/'attention').glob('class_*.png'))
            if len(maps) != 10 or len(images) != 10:
                raise RuntimeError('MIL needs ten finite attention maps and heatmap images')
        if run_receipt and not metadata.get('training_compliance',{}).get('ledger_sha256'):
            raise RuntimeError('Evaluation missing training compliance binding')
        if job.get('timing', True):
            require_timing_evidence(report, 'hflip2', 'Evaluation')
        score=report['scores']['hflip2']
        self.status['completed_metrics'][job['name']] = {'macro_auc':score['all']['macro_auc'],
             'ap_only_macro_auc':score['ap_only']['macro_auc'], 'per_class_auc':score['all']['per_class_auc'],
             'timing':(report.get('timing') or {}).get('modes'), 'report':str(output/'report.json')}
        return {'verified_at':now(),'checkpoint_sha256':digest,'report_sha256':sha(output/'report.json'),
                'metadata_sha256':sha(output/'metadata.json'),'training_compliance':run_receipt}

    def comparison(self,job,pre):
        args=['compare','--inputs',*job['inputs'],'--output',job['output'],'--bootstrap',str(job.get('bootstrap',2000))]
        if job.get('timing_fusions'): args+=['--timing-fusions']
        self.heartbeat('patient_bootstrap_fusion_and_sensitivity')
        self.command(job,docker_command('exploration_evaluate.py',args,gpu=job.get('timing_fusions',False)),pre)
        path=Path(job['output'])/'comparison.json'; report=read(path)
        if not report.get('complete') or report.get('test_accessed') is not False:
            raise RuntimeError('Comparison metrics/compliance incomplete')
        probabilities=Path(job['output'])/'candidate_study_probabilities.npz'
        if sha(probabilities) != report.get('candidate_probabilities_sha256'):
            raise RuntimeError('Comparison probability artifact SHA mismatch')
        for name, item in report['inputs'].items():
            metadata_path=Path(item['directory'])/'metadata.json'
            if sha(metadata_path) != item['metadata_sha256']:
                raise RuntimeError('Comparison input metadata changed: '+name)
        if job.get('timing_fusions'):
            names={item.split('=',1)[0] for item in job['inputs']}
            pairs=[('res768','res896'),('res768','roi'),('tiny_a','tiny_b')]
            for left,right in pairs:
                if left in names and right in names:
                    key=f'{left}_{right}_equal_hflip2'
                    timing=report['candidates'].get(key,{}).get('timing') or {}
                    if not timing:
                        raise RuntimeError('Fusion timing is preregistered but missing: '+key)
                    require_timing_evidence({'timing':{'modes':{'hflip2':timing}}}, 'hflip2', key)
        self.status['completed_metrics'][job['name']]={'report':str(path),
            'candidate_macro_auc':{name:v['scores']['all']['macro_auc'] for name,v in report['candidates'].items()},
            'resolution_sensitivity':report['resolution_sensitivity']}
        return {'verified_at':now(),'comparison_sha256':sha(path),'test_accessed':False}

    def run(self):
        with (self.root/'.queue.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if (self.root/'FAILED').exists():
                    raise RuntimeError('FAILED marker exists: explicit reviewed recovery required')
                self.validate_plan()
                for self.index,job in enumerate(self.plan['jobs']):
                    self.phase_started=time.time(); self.status['current_job']=job['name']
                    self.check_stop()
                    pre=preregistered(job)
                    self.status['current_preregistration'] = pre
                    self.check_stop()
                    self.heartbeat('checking_preregistration')
                    if job['kind']=='train': receipt=self.training(job,pre)
                    elif job['kind']=='evaluate': receipt=self.evaluation(job,pre)
                    elif job['kind']=='compare': receipt=self.comparison(job,pre)
                    elif job['kind']=='report':
                        self.command(job,[os.sys.executable,str(ROOT/'summarize_exploration.py'),'--plan',str(self.plan_path)],pre)
                        path=Path(self.plan['report_file'])
                        if not path.is_file(): raise RuntimeError('Morning report is missing')
                        receipt={'verified_at':now(),'report_sha256':sha(path)}
                    else: raise ValueError('Unknown job kind')
                    self.check_stop()
                    receipt_path=self.root/'receipts'/(job['name']+'.json')
                    write(receipt_path,{**receipt,'preregistration':pre})
                    self.status['completed_jobs'].append(job['name'])
                    self.status['compliance_records'][job['name']] = {
                        'receipt_path':str(receipt_path), 'preregistration_sha256':pre['sha256'],
                        'run_ledger_sha256':(receipt.get('ledger_sha256') or
                            (receipt.get('training_compliance') or {}).get('ledger_sha256')),
                        'receipt_sha256':sha(receipt_path)}
                    self.heartbeat('metrics_and_compliance_verified')
                self.index=len(self.plan['jobs']); self.status['state']='COMPLETED'
                self.heartbeat('all_registered_work_completed')
            except BaseException as error:
                self.status['state']='FAILED'; self.status['failure_reason']=f'{type(error).__name__}: {error}'
                self.status['traceback']=traceback.format_exc()
                try:
                    self.heartbeat('stopped_without_advancing')
                except BaseException as status_error:
                    self.status['status_write_error']=f'{type(status_error).__name__}: {status_error}'
                    self.status['updated_utc']=now()
                    write(self.status_path, self.status)
                    write(self.root/'status.json', self.status)
                try:
                    subprocess.run([os.sys.executable, str(ROOT/'summarize_exploration.py'),
                                    '--plan', str(self.plan_path)], cwd=ROOT, timeout=60,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except BaseException as report_error:
                    self.status['partial_report_error'] = f'{type(report_error).__name__}: {report_error}'
                write(self.root/'FAILED',self.status)
                write(self.status_path.with_name('FAILED'),self.status)
                raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    Queue(parser.parse_args().plan).run()
