"""Run the registered anchor and two fixed student refits, one GPU job at a time.

No automatic retries, teacher training, submission generation or external writes.
The anchor's sole validation is a quality veto before either final refit starts.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from one_week.runtime import OUTPUT_ROOT, ROOT, command


def now():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_quality_gate(stage_output):
    report = json.loads((stage_output / 'final_validation.json').read_text())
    metrics = report['metrics']
    macro = metrics['macro_auc']
    # These deliberately coarse vetoes are frozen before training; this is not
    # a search for the highest of many models on the repeatedly reused valid.
    reasons = []
    if report.get('complete') is not True or report.get('studies') != 859 or report.get('images') != 1422:
        reasons.append('The complete official validation was not reproduced')
    if not isinstance(macro, (int, float)) or not .800 <= macro <= 1.:
        reasons.append('Anchor macro AUC fell below the predeclared .800 quality floor')
    return dict(passed=not reasons, reasons=reasons, macro_auc=macro, minimum_macro_auc=.800,
                used_for_epoch_selection=False, independently_validated=False)


def control(plan_path):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / '.controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT_ROOT / 'STOP').exists():
            raise RuntimeError('Pipeline STOP exists')
        plan = json.loads(plan_path.read_text())
        receipt = dict(state='starting', started_at=now(), controller_pid=os.getpid(),
            parent_pid=os.getppid(), session_id=os.getsid(0), stdin=os.readlink('/proc/self/fd/0'),
            plan=str(plan_path), plan_sha256=sha(plan_path), automatic_retries=0,
            teacher_training_allowed=False, automatic_submission=False, completed_stages=[])
        status_path = OUTPUT_ROOT / 'pipeline_status.json'
        write(status_path, receipt)
        try:
            for stage in plan['stages']:
                if (OUTPUT_ROOT / 'STOP').exists():
                    receipt.update(state='stopped', finished_at=now()); write(status_path, receipt); return 0
                config_path = Path(stage['config'])
                if sha(config_path) != stage['sha256'] or sha(plan_path) != receipt['plan_sha256']:
                    raise RuntimeError('Frozen pipeline or stage configuration changed')
                config = json.loads(config_path.read_text())
                output = Path(config['output'])
                if (output / 'configuration.json').exists() or (output / 'STOP').exists():
                    raise RuntimeError('Stage is already started or stopped; no implicit restart')
                cmd = command(['-m', 'one_week.train', '--config', str(config_path), '--execute'],
                               config, name='cxr-one-week-' + stage['name'])
                receipt.update(state='training', active_stage=stage['name'], updated_at=now(), command=cmd,
                               output=str(output), checkpoint_dir=config['checkpoint_dir'])
                write(status_path, receipt)
                with (output / 'training.log').open('a', buffering=1) as log:
                    process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT)
                    receipt['docker_client_pid'] = process.pid
                    write(status_path, receipt)
                    code = process.wait()
                trainer_status = json.loads((output / 'status.json').read_text())
                if code != 0 or trainer_status.get('state') != 'completed':
                    receipt.update(state='failed_or_stopped', exit_code=code, trainer_status=trainer_status,
                                   finished_at=now()); write(status_path, receipt); return code or 1
                receipt['completed_stages'].append(stage['name'])
                if stage['name'] == 'raw_anchor_seed44':
                    gate = anchor_quality_gate(output)
                    write(OUTPUT_ROOT / 'anchor_quality_gate.json', gate)
                    receipt['anchor_quality_gate'] = gate
                    if not gate['passed']:
                        receipt.update(state='anchor_vetoed_refits', finished_at=now())
                        write(status_path, receipt); return 0
                receipt.update(updated_at=now()); write(status_path, receipt)
            receipt.update(state='completed', finished_at=now(),
                           final_policy='retain seed44 primary and predeclared seed44/43 equal-mean alternative; no valid selection')
            write(status_path, receipt)
            return 0
        except BaseException as error:
            receipt.update(state='failed', error=f'{type(error).__name__}: {error}', finished_at=now())
            write(status_path, receipt)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'one_week/plan.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--controller', action='store_true')
    args = parser.parse_args()
    plan = args.plan.resolve(strict=True)
    if args.controller:
        raise SystemExit(control(plan))
    if not args.execute:
        print(plan.read_text()); return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / '.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT_ROOT / 'launch.json').exists() or (OUTPUT_ROOT / 'STOP').exists():
            raise RuntimeError('Already launched or stopped; no automatic restart')
        # Verify all stage paths, dependencies and manifest bindings before detaching.
        for stage in json.loads(plan.read_text())['stages']:
            config_path = Path(stage['config'])
            if sha(config_path) != stage['sha256']:
                raise ValueError('Stage config fingerprint changed')
            config = json.loads(config_path.read_text())
            command(['-m', 'one_week.train', '--config', str(config_path), '--execute'], config)
        cmd = [sys.executable, '-u', '-m', 'one_week.pipeline', '--plan', str(plan), '--controller']
        with (OUTPUT_ROOT / 'controller.log').open('a', buffering=1) as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        receipt = dict(launched_at=now(), controller_pid=process.pid, command=cmd,
                       plan_sha256=sha(plan), detached=True)
        write(OUTPUT_ROOT / 'launch.json', receipt)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
