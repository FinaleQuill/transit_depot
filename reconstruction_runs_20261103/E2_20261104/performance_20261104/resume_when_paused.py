"""One-shot background handoff after the existing E2 epoch closes; no retries."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
from reconstruction import controller as base
from reconstruction.controller_local import verify_run

ROOT = base.RUN_ROOT / 'E2_20261104'
PERF = ROOT / 'performance_20261104'
REPO = base.ROOT

def publish(state, **fields):
    base.write_json(PERF / 'handoff_status.json', {'state': state, 'updated_at': base.now(), **fields})

def main():
    with (PERF / '.handoff.lock').open('a+b') as handoff_lock:
        fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'performance_launch.json').exists():
            raise FileExistsError('A performance launch already exists; do not start it twice')
        ready_path = PERF / 'READY_TO_SWITCH.json'
        ready = base._read_json(ready_path)
        ready_sha = base.sha(ready_path)
        if not ready['passed']:
            raise ValueError('Execution checks have not passed')
        plan = base._read_json(ROOT / 'controller/frozen_plan.json')
        started = time.monotonic()
        while True:
            if time.monotonic() - started > 5400:
                raise TimeoutError('Original epoch did not close within 90 minutes; no retry')
            if base.sha(ready_path) != ready_sha:
                raise ValueError('Readiness evidence changed')
            for path, expected in ready['bound_sha256'].items():
                if base.sha(path) != expected:
                    raise ValueError('Bound execution evidence changed: ' + path)
            base.verify_plan(plan)
            state = base._read_json(ROOT / 'controller/STATUS.json')
            if state['controller_pid'] != ready['original_controller_pid']:
                raise RuntimeError('Another controller took over; refusing duplicate handoff')
            if state['state'] == 'paused' and state.get('active_container') is None:
                proc = Path('/proc') / str(ready['original_controller_pid']) / 'stat'
                if not proc.exists():
                    break
                try:
                    old_state = proc.read_text().rsplit(')', 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if old_state == 'Z':
                    break
                publish('waiting_for_original_controller_exit', original_controller_pid=ready['original_controller_pid'])
                time.sleep(1)
                continue
            if state['state'] not in {'stopping', 'training', 'checking'}:
                raise RuntimeError('Original controller did not reach a safe pause: ' + state['state'])
            publish('waiting_for_complete_epoch', original_controller_pid=state['controller_pid'],
                    trainer_progress=state.get('trainer_progress', {}), ready_sha256=ready_sha)
            time.sleep(10)
        lock_path = ROOT / 'controller/.controller.lock'
        with lock_path.open('a+b') as original_lock:
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True)
            if any(n.startswith('cxr-e2-') for n in names.splitlines()):
                raise RuntimeError('An E2 container remains active')
            if (ROOT / 'controller/performance_plan.json').exists():
                raise FileExistsError('An execution transition already exists')
            checked = verify_run(plan['jobs'][0], require_complete=False)
            if checked['epochs'] != 1:
                raise ValueError('Expected the preserved complete first epoch')
            stop = ROOT / 'controller/STOP'
            if base.sha(stop) != ready['controller_stop_sha256']:
                raise ValueError('STOP request changed since readiness')
            archived_stop = PERF / 'original_controller_STOP'
            if archived_stop.exists():
                raise FileExistsError(archived_stop)
            base.write_json(PERF / 'closed_epoch_verified.json', checked)
            os.replace(stop, archived_stop)
        profile = PERF / 'loader_workers6_prefetch2.json'
        args = [sys.executable, '-u', '-m', 'reconstruction.controller_performance',
                '--controller-dir', str(ROOT / 'controller'), '--configs',
                *[j['config_path'] for j in plan['jobs']], '--execution-profile', str(profile),
                '--execution-profile-sha256', base.sha(profile), '--execute', '--resume-controller',
                '--resume-run', plan['jobs'][0]['config_path']]
        log_path = ROOT / 'controller.performance.launch.log'
        with log_path.open('xb', buffering=0) as log:
            child = subprocess.Popen(args, cwd=REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        launch = {'started_at': base.now(), 'pid':child.pid, 'argv':args, 'cwd':str(REPO),
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  'log':str(log_path), 'stdin':'/dev/null','start_new_session':True,
                  'old_controller_pid':ready['original_controller_pid'], 'ready_sha256':ready_sha,
                  'helper_sha256':base.sha(__file__), 'profile_sha256':base.sha(profile),
                  'remaining_budget':'global epoch2..6 then local epoch0 evaluation and epoch1..6 then fixed comparisons',
                  'AI_goal_created':False}
        base.write_json(ROOT / 'performance_launch.json', launch)
        publish('launched', launch=launch)
        print(json.dumps(launch, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        publish('failed', error=f'{type(error).__name__}: {error}', automatic_retry=False)
        raise
