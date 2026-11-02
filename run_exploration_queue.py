"""Durable serial, preregistered train-only exploration queue; never submits.

A new job starts only after the previous job's metrics and lifecycle ledger
verify. Missing preregistration or incomplete compliance is a hard stop.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

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
    if not p.is_file() or sha(p) != job['preregistration_sha256']:
        raise RuntimeError('STOP: missing or changed preregistration: ' + job['name'])
    lines = [v for v in p.read_text().splitlines() if v.strip()]
    if len(lines) != 4:
        raise RuntimeError('STOP: preregistration must contain four nonempty lines')
    return {'path': str(p), 'sha256': sha(p), 'observed_mtime_ns': p.stat().st_mtime_ns}

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

def active_run(run_dir):
    """Read process commands; avoid treating this controller's argument as a trainer."""
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'--execute' in args and str(run_dir).encode() in args and any(a.endswith(b'train_distill.py') or a.endswith(b'exploration_train.py') for a in args):
            return True
    return False

def train_job(job, root):
    pre = preregistered(job)
    run_dir = Path(job['run_dir'])
    while active_run(run_dir):
        write(root / 'status.json', {'state': 'waiting_existing_training', 'job': job['name'], 'updated_at': now()})
        time.sleep(20)
    status = read(run_dir / 'status.json') if (run_dir / 'status.json').exists() else None
    if status and status['state'] in {'completed', 'early_stopped', 'already_terminal'}:
        receipt = verify_run(run_dir)
    else:
        args = list(job['arguments'])
        if status:
            if status['state'] != 'paused' or not job.get('allow_existing_pause_resume'):
                raise RuntimeError('STOP: existing nonterminal run requires explicit recovery: ' + job['name'])
            verify_run(run_dir, allow_paused=True)
            for flag in ('--init-student-ema', '--init-student-ema-sha256'):
                if flag in args:
                    i = args.index(flag)
                    del args[i:i+2]
            args += ['--resume', str(run_dir / 'last.pt')]
        command = docker_command(job['script'], args + ['--execute'])
        launch_path = root / 'launches' / (job['name'] + '_' + str(time.time_ns()) + '.json')
        write(launch_path, {'preregistration': pre, 'command': command, 'launched_at': now(), 'resume': bool(status)})
        write(root / 'status.json', {'state': 'training', 'job': job['name'], 'updated_at': now(), 'launch': str(launch_path)})
        with (root / 'logs' / (job['name'] + '.log')).open('a') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f'STOP: training failed ({result.returncode}): {job["name"]}; retain evidence')
        receipt = verify_run(run_dir)
    write(root / 'receipts' / (job['name'] + '.json'), {**receipt, 'preregistration': pre})
    return receipt

def run(plan_path):
    plan = read(plan_path)
    root = Path(plan['output'])
    for directory in ('logs', 'launches', 'receipts'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    with (root / '.queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for job in plan['jobs']:
                if (root / 'STOP').exists():
                    raise RuntimeError('STOP: user stop marker')
                train_job(job, root)
            write(root / 'status.json', {'state': 'training_completed', 'updated_at': now(), 'plan_sha256': sha(plan_path)})
        except BaseException as error:
            write(root / 'status.json', {'state': 'stopped', 'error': f'{type(error).__name__}: {error}', 'updated_at': now()})
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
