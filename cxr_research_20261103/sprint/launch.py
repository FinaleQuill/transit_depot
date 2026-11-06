"""Detach one fixed training-and-comparison job; never retry or extend its budget."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from sprint.runtime import ROOT, command


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_controller(config_path, config, *, resume=False):
    output = Path(config['output'])
    with (output / '.controller.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another sprint controller owns this run') from None
        if (output / 'configuration.json').exists() and not resume:
            raise FileExistsError('Training already exists; no automatic restart')
        if (output / 'STOP').exists():
            raise RuntimeError('Sprint STOP is present; explicit review is required before launch')
        container = 'codex-cxr-sprint-20261104'
        arguments = ['-m', 'sprint.train', '--config', str(config_path), '--execute']
        if resume:
            arguments.append('--resume')
        cmd = command(arguments, config, name=container)
        receipt = {'state': 'starting', 'started_at': now(), 'controller_pid': os.getpid(),
            'controller_ppid': os.getppid(), 'session_id': os.getsid(0),
            'process_group': os.getpgrp(), 'has_tty': os.isatty(0),
            'stdin': os.readlink('/proc/self/fd/0'), 'container': container,
            'config_path': str(config_path),
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'command': cmd, 'automatic_retries': 0,
            'completed_training_then_single_comparison': True,
            'checkpoint_dir': config['checkpoint_dir'],
            'comparison_directory': str(output / 'final_comparison')}
        status = output / 'controller_status.json'
        write_json(status, receipt)
        with (output / 'training.log').open('a', encoding='utf-8', buffering=1) as log:
            try:
                process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                receipt.update(state='running', docker_client_pid=process.pid, updated_at=now())
                write_json(status, receipt)
                code = process.wait()
                trainer = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
                comparison = output / 'final_comparison' / 'report.json'
                if code == 0 and trainer.get('state') == 'completed' and comparison.exists():
                    report = json.loads(comparison.read_text())
                    if report.get('complete') and report.get('candidate_won'):
                        receipt.update(state='preparing_winning_submission', updated_at=now())
                        write_json(status, receipt)
                        deployment_command = command(['-m', 'sprint.deploy', '--config',
                            str(config_path), '--execute'], config, name=container + '-export')
                        receipt['deployment_command'] = deployment_command
                        write_json(status, receipt)
                        with (output / 'deployment.log').open('a', encoding='utf-8', buffering=1) as deployment_log:
                            code = subprocess.call(deployment_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=deployment_log, stderr=subprocess.STDOUT)
                        receipt['deployment_exit_code'] = code
                        if code:
                            receipt.update(state='deployment_failed', trainer_state=trainer,
                                exit_code=code, finished_at=now())
                            write_json(status, receipt)
                            return code
                receipt.update(state=trainer.get('state', 'completed' if code == 0 else 'failed'),
                    trainer_state=trainer, exit_code=code, finished_at=now())
                write_json(status, receipt)
                return code
            except BaseException as error:
                receipt.update(state='controller_failed', error=f'{type(error).__name__}: {error}', finished_at=now())
                write_json(status, receipt)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'sprint/plan.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = Path(config['output'])
    if not args.execute and not args.controller:
        print(json.dumps({'config': str(config_path), 'output': str(output),
            'checkpoint_dir': config['checkpoint_dir'], 'epochs': config['epochs'],
            'launch': 'Add --execute to detach the authorized fixed sprint.'}, indent=2))
        return
    for key in ('output', 'checkpoint_dir'):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    if args.controller:
        raise SystemExit(run_controller(config_path, config, resume=args.resume))
    # This launch lock prevents two simultaneous shell invocations from creating
    # two controllers; the persistent controller lock protects the whole run.
    with (output / '.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / '.controller.lock').open('a+b') as controller_lock:
            try:
                fcntl.flock(controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('The sprint controller is already running') from None
        if (output / 'configuration.json').exists() and not args.resume:
            raise FileExistsError('Training state already exists; no implicit restart')
        command(['-m', 'sprint.train', '--config', str(config_path), '--execute'], config)
        argv = [sys.executable, '-u', '-m', 'sprint.launch', '--controller', '--config', str(config_path)]
        if args.resume:
            argv.append('--resume')
        with (output / 'controller.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'state': 'launched', 'launched_at': now(), 'controller_pid': process.pid,
            'config': str(config_path), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'command': argv, 'detached': True, 'log': str(output / 'training.log'),
            'status': str(output / 'controller_status.json')}
        write_json(output / 'launch.json', receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
