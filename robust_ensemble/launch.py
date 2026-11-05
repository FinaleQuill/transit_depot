"""Detached one-candidate validation followed by export only on frozen gates."""
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
from sprint.launch import write_json


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def controller(path, config):
    output = Path(config['output'])
    with (output / '.controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'controller_status.json').exists():
            raise FileExistsError('No implicit repeat of the frozen candidate')
        # Freeze the full local import closure of validation, publication and
        # the controller, including package initializers used after relocation.
        sources = [
            'benchmark_local_resources.py', 'compare_student_candidates.py',
            'cost.py', 'dataset.py', 'deploy_kd.py', 'ensemble_sprint/__init__.py',
            'ensemble_sprint/deploy.py', 'ensemble_sprint/model.py',
            'ensemble_sprint/predictor.py', 'fast_lanczos.py', 'frozen_head_predictor.py',
            'inference_inputs.py', 'inference_reproducibility.py', 'inference_runtime.py',
            'kd_common.py', 'model_overnight.py', 'native_lanczos/__init__.py', 'predict.py',
            'reconstruction/__init__.py', 'reconstruction/model.py',
            'reconstruction/runtime.py', 'reconstruction/runtime_local.py',
            'robust_ensemble/__init__.py', 'robust_ensemble/deploy.py',
            'robust_ensemble/evaluate.py', 'robust_ensemble/launch.py',
            'robust_ensemble/plan.json', 'robust_ensemble/predictor.py',
            'runtime_compat.py', 'sprint/deploy.py', 'sprint/evaluate.py',
            'sprint/launch.py', 'sprint/model.py', 'sprint/runtime.py',
            'student_draft_pair.py', 'student_ensemble.py', 'student_experiment_evaluation.py',
            'student_fixed_pair.py', 'submission_runtime.py', 'submit_student_candidate.py',
        ]
        evidence = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
        source_dir = output / 'source_snapshot'
        source_dir.mkdir(exist_ok=False)
        for p in sources:
            dest = source_dir / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / p).read_bytes())
        write_json(output / 'source_identity.json', evidence)
        status = {'state': 'starting', 'started_at': now(), 'controller_pid': os.getpid(),
            'controller_ppid': os.getppid(), 'session_id': os.getsid(0),
            'stdin': os.readlink('/proc/self/fd/0'), 'has_tty': os.isatty(0),
            'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'completed_steps': [], 'new_training_runs': 0, 'automatic_retries': 0}
        write_json(output / 'controller_status.json', status)
        def execute(name, arguments):
            if (output / 'STOP').exists():
                status.update(state='stopped', finished_at=now())
                write_json(output / 'controller_status.json', status)
                return False
            for p, identity in evidence.items():
                if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != identity:
                    raise RuntimeError('Frozen source changed: ' + p)
            cmd = command(arguments, config, name='cxr-robust-20261105-' + name)
            status.update(state='running', current_step=name, updated_at=now(), command=cmd)
            write_json(output / 'controller_status.json', status)
            with (output / (name + '.log')).open('a', buffering=1) as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                write_json(output / 'controller_status.json', status)
                code = child.wait()
            if code:
                status.update(state='failed', exit_code=code, finished_at=now())
                write_json(output / 'controller_status.json', status)
                return False
            status['completed_steps'].append(name)
            write_json(output / 'controller_status.json', status)
            return True
        try:
            if not execute('final_compare', ['-m', 'robust_ensemble.evaluate', '--config', str(path)]):
                return
            report = json.loads((output / 'final_comparison/report.json').read_text())
            if report['candidate_won'] and config['generate_winning_test_csv']:
                if not execute('winning_export', ['-m', 'robust_ensemble.deploy', '--config', str(path), '--execute']):
                    return
            status.update(state='completed', finished_at=now(), candidate_won=report['candidate_won'],
                final_validation_runs=1, generated_new_csv=bool(report['candidate_won'] and config['generate_winning_test_csv']),
                final_report=str(output / 'final_comparison/report.json'))
            write_json(output / 'controller_status.json', status)
        except BaseException as error:
            status.update(state='failed', error=f'{type(error).__name__}: {error}', finished_at=now())
            write_json(output / 'controller_status.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'robust_ensemble/plan.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    path = args.config.resolve(strict=True)
    config = json.loads(path.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    if args.controller:
        controller(path, config)
        return
    if not args.execute:
        print(json.dumps({'candidate': config['actual_member_order'], 'new_training_runs': 0,
            'selection_rule': config['selection_rule'], 'output': str(output)}, ensure_ascii=False))
        return
    with (output / '.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / p).exists() for p in ('launch.json', 'controller_status.json', 'STOP')):
            raise FileExistsError('Run already launched or stopped; no automatic repeat')
        running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
        if running:
            raise RuntimeError('Another GPU job is running')
        argv = [sys.executable, '-u', '-m', 'robust_ensemble.launch', '--controller', '--config', str(path)]
        with (output / 'controller.log').open('a', buffering=1) as stream:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'state': 'launched', 'launched_at': now(), 'controller_pid': child.pid,
            'detached': True, 'new_training_runs': 0, 'config': str(path),
            'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        write_json(output / 'launch.json', receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
