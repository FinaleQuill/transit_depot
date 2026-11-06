"""Detached six-teacher controller; no retries or automatic student/full refit."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from teacher_distill_20261106.runtime import EVIDENCE, ROOT, command


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def control(queue_path):
    queue = json.loads(queue_path.read_text())
    if queue['teacher_count'] != 6 or len(queue['runs']) != 6 or queue['automatic_submission']:
        raise ValueError('Unexpected first-stage scope')
    with (EVIDENCE/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(state='starting', controller_pid=os.getpid(), ppid=os.getppid(),
            session_id=os.getsid(0), stdin=os.readlink('/proc/self/fd/0'), started_at=now(),
            queue=str(queue_path), queue_sha256=sha(queue_path), completed=[],
            automatic_retries=0, automatic_students=False, automatic_submission=False)
        write(EVIDENCE/'controller_status.json', receipt)
        for index, run in enumerate(queue['runs']):
            if (EVIDENCE/'STOP').exists():
                receipt.update(state='stopped_before_next_teacher', stopped_at=now())
                write(EVIDENCE/'controller_status.json', receipt); return 0
            path = Path(run['config'])
            if sha(path) != run['config_sha256']:
                raise ValueError('Queued configuration changed')
            config = json.loads(path.read_text()); output = Path(config['output'])
            output.mkdir(parents=True, exist_ok=True)
            if (output/'configuration.json').exists():
                raise FileExistsError('No automatic resume or repeat of a started teacher')
            name = f"cxr-kd-20261106-{run['run_id'].replace('_', '-')}"
            cmd = command(['-m', 'teacher_distill_20261106.train', '--config', str(path), '--execute'],
                          config, name=name)
            receipt.update(state='running_teacher', current_index=index, current_run=run['run_id'],
                           container=name, current_started_at=now())
            write(EVIDENCE/'controller_status.json', receipt)
            with (output/'training.log').open('a', buffering=1) as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                receipt.update(docker_client_pid=process.pid)
                write(EVIDENCE/'controller_status.json', receipt)
                code = process.wait()
            trainer = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
            if code != 0 or trainer.get('state') != 'completed_fixed_schedule':
                receipt.update(state='failed' if code else 'stopped', exit_code=code,
                               trainer_status=trainer, finished_at=now())
                write(EVIDENCE/'controller_status.json', receipt); return code or 1
            receipt['completed'].append(dict(run_id=run['run_id'], status=trainer))
            write(EVIDENCE/'controller_status.json', receipt)
        receipt.update(state='awaiting_teacher_review', finished_at=now(), current_run=None,
                       next_action='Report teacher quality and cost before starting students; no automatic next major stage')
        write(EVIDENCE/'controller_status.json', receipt)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=EVIDENCE/'teacher_queue.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--controller', action='store_true')
    args = parser.parse_args(); queue_path = args.queue.resolve(strict=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.controller:
        raise SystemExit(control(queue_path))
    if not args.execute:
        print(queue_path.read_text()); return
    with (EVIDENCE/'.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (EVIDENCE/'STOP').exists() or (EVIDENCE/'launch.json').exists():
            raise RuntimeError('Experiment already launched or STOP exists')
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if busy.strip():
            raise RuntimeError('GPU is already in use; no concurrent training is permitted')
        cmd = [sys.executable, '-u', '-m', 'teacher_distill_20261106.launch',
               '--queue', str(queue_path), '--controller']
        with (EVIDENCE/'controller.log').open('a', buffering=1) as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        receipt = dict(controller_pid=process.pid, launched_at=now(), detached=True,
                       command=cmd, scope='six first-stage teachers only')
        write(EVIDENCE/'launch.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    main()
