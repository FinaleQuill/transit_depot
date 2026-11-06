"""Host-only A timing/submission interlude, preserving the live B process."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LOCAL = Path('/tmp/cloud_student_runs_20261030/041_A_submission_20261030')
EVIDENCE = ROOT / 'artifacts/kd_v1/student_A_submission_20261030'
DELIVERY = ROOT.parent / 'deliveries/041_tiny768_p2_A_submission_20261030'
CONTAINER = 'a8daf4091276'
B_PID = 171397
B_LOG = ROOT / 'artifacts/kd_v1/student_upper_bound_20261030/B.training.log'
SPY = Path('/tmp/cloud_student_pause_inspector/bin/py-spy')


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(name, data):
    destination = EVIDENCE / name
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination)


def safe_compute_stack(rows, pid):
    """Positive compute evidence, not merely absence of a queue frame."""
    main = [row for row in rows if row.get('thread_name') == 'MainThread' and row.get('pid') == pid]
    if len(main) != 1:
        return False, []
    frames = main[0].get('frames', [])
    is_train = any(f.get('name') == 'train_study_epoch' and str(f.get('filename', '')).endswith('/study_student.py') for f in frames)
    waiting = any('/torch/utils/data/dataloader.py' in str(f.get('filename', ''))
                  or f.get('name') in ('_next_data', '_get_data', '_try_get_data')
                  or (str(f.get('filename', '')).endswith('/queues.py') and f.get('name') == 'get')
                  for f in frames)
    compute = any((f.get('name') in ('forward', 'forward_features', 'backward', '_engine_run_backward', 'step')
                   and any(part in str(f.get('filename', '')) for part in ('/torch/', '/timm/')))
                  for f in frames)
    return is_train and compute and not waiting, [
        {k: frame.get(k) for k in ('name', 'filename', 'line')} for frame in frames]


def command(args, label):
    path = EVIDENCE / (label + '.log')
    with path.open('xb', buffering=0) as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} exited {result.returncode}; see {path}')


def unpause():
    state = subprocess.check_output(['docker', 'inspect', CONTAINER, '--format', '{{.State.Paused}}'], text=True).strip()
    if state == 'true':
        subprocess.run(['docker', 'unpause', CONTAINER], check=True, stdout=subprocess.DEVNULL)


def execute():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if (EVIDENCE / 'execution.json').exists() or DELIVERY.exists():
        raise FileExistsError('Choose a new interlude; never overwrite a prior delivery')
    pipeline = json.loads((ROOT / 'artifacts/kd_v1/student_upper_bound_20261030/pipeline_execution.json').read_text())
    protected = {name: digest(ROOT / name) for name in pipeline['source_sha256']}
    assert protected == pipeline['source_sha256'], 'Running pipeline sources changed'
    initial_model = {name: digest(LOCAL / 'model' / name) for name in ('config.json', 'student_fp16.pt')}
    proof = json.loads((ROOT / 'artifacts/kd_v1/student_upper_bound_20261030/A_export_tensor_parity_20261030.json').read_text())
    assert proof['complete'] and proof['all_tensor_bytes_equal']
    inputs = ROOT.parent / 'deliveries/034_dual768_proxy_timing_20261029/bundle/inputs'
    selection = json.loads((inputs / 'selection.portable.json').read_text())
    selection['manifest_path'] = str(ROOT / 'data/clean_v1_reviewed/valid_manifest.csv')
    (LOCAL / 'selection.relocated.json').write_text(json.dumps(selection, indent=2) + '\n')
    record = {'started_hong_kong': now(), 'B_container': CONTAINER, 'B_host_pid': B_PID,
              'B_progress_before': B_LOG.read_text().splitlines()[-1],
              'A_model_sha256': initial_model, 'frozen_execution_source_sha256': protected,
              'pause_attempts': [], 'T4_certified': False, 'B_context_remains_resident': True,
              'B_model_or_optimizer_checkpoint_created_by_pause': False}
    save('execution.json', record)
    # Independent host watchdog guarantees an eventual unpause even if this
    # orchestrator is killed. It sleeps and consumes no CPU during timing.
    watchdog_code = "import subprocess,time; time.sleep(900); s=subprocess.run(['docker','inspect'," + repr(CONTAINER) + ",'--format','{{.State.Paused}}'],capture_output=True,text=True); subprocess.run(['docker','unpause'," + repr(CONTAINER) + "]) if s.stdout.strip()=='true' else None"
    watchdog = subprocess.Popen([sys.executable, '-c', watchdog_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'Host signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    accepted = False
    try:
        for attempt in range(20):
            subprocess.run(['docker', 'pause', CONTAINER], check=True, stdout=subprocess.DEVNULL)
            try:
                result = subprocess.run(['sudo', '-n', str(SPY), 'dump', '--pid', str(B_PID), '--json', '--nonblocking'], capture_output=True, text=True, timeout=10, check=True)
                accepted, frames = safe_compute_stack(json.loads(result.stdout), B_PID)
                record['pause_attempts'].append({'at': now(), 'accepted_compute_boundary': accepted, 'main_thread_frames': frames})
                if accepted:
                    record['paused_hong_kong'] = now()
                    break
            except (subprocess.SubprocessError, ValueError) as error:
                record['pause_attempts'].append({'at': now(), 'accepted_compute_boundary': False, 'error': str(error)})
            unpause()
            time.sleep(.2)
        if not accepted:
            raise RuntimeError('Could not positively verify a safe B compute boundary')
        # Do not time until queued B kernels have completed. The frozen B CUDA
        # context remains allocated and is explicitly reported, not hidden.
        samples = []
        for attempt in range(20):
            value = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
            utilization = int(value.split(',')[0])
            samples.append({'at': now(), 'gpu_utilization_and_memory_mib': value})
            if len(samples) >= 2 and utilization == 0 and int(samples[-2]['gpu_utilization_and_memory_mib'].split(',')[0]) == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('GPU did not become idle after B freeze')
        record['idle_samples_before_timing'] = samples
        record['B_log_bytes_while_frozen'] = B_LOG.stat().st_size
        save('execution.json', record)
        wrapper = str(ROOT / 'run_student_environment.sh')
        command([wrapper, '-u', 'benchmark_local_resources.py', '--model-dir', str(LOCAL / 'model'),
                 '--pictures-100', str(inputs / 'pictures_100'), '--pictures-5', str(inputs / 'pictures_5'),
                 '--selection-json', str(LOCAL / 'selection.relocated.json'),
                 '--output-dir', str(LOCAL / 'latency_2080ti_hflip2'), '--profile', 'A_E9_hflip2_B_frozen',
                 '--baseline-sm-count', '68', '--memory-limit-gib', '8', '--execute'], 'A.latency')
        record['latency_completed_hong_kong'] = now()
        save('execution.json', record)
        command([wrapper, '-u', 'predict.py', '--model-dir', str(LOCAL / 'model'),
                 '--image-root', '/home/featurize/data/cxr_first', '--manifest', str(ROOT / 'data/clean_v1_reviewed/test_manifest.csv'),
                 '--output', str(LOCAL / 'test_predictions/submission.csv')], 'A.prediction')
        command([wrapper, '-u', 'audit_student_submission.py', '--manifest', str(ROOT / 'data/clean_v1_reviewed/test_manifest.csv'),
                 '--image-predictions', str(LOCAL / 'test_predictions/submission.image_predictions.csv'),
                 '--submission', str(LOCAL / 'test_predictions/submission.csv'),
                 '--output', str(LOCAL / 'test_predictions/submission.audit.json')], 'A.audit')
        assert B_LOG.stat().st_size == record['B_log_bytes_while_frozen'], 'B progressed during A interlude'
        assert all(digest(LOCAL / 'model' / name) == sha for name, sha in initial_model.items())
        assert all(digest(ROOT / name) == sha for name, sha in protected.items())
        DELIVERY.mkdir(parents=True, exist_ok=False)
        inventory = {}
        for original in list((LOCAL / 'test_predictions').glob('*')) + [LOCAL / 'latency_2080ti_hflip2/local_resource_report.json']:
            if not original.is_file():
                continue
            if shutil.disk_usage(DELIVERY).free < original.stat().st_size + 80 * 1024**2:
                raise OSError('Preserve the work disk reserve')
            destination = DELIVERY / original.name
            shutil.copyfile(original, destination)
            assert digest(destination) == digest(original)
            inventory[destination.name] = {'bytes': destination.stat().st_size, 'sha256': digest(destination)}
        record['delivery'] = str(DELIVERY)
        record['delivery_inventory'] = inventory
        record['A_task_complete'] = True
        save('execution.json', record)
    except BaseException as error:
        record['A_task_complete'] = False
        record['error'] = f'{type(error).__name__}: {error}'
        save('execution.json', record)
        raise
    finally:
        unpause()
        record['resumed_hong_kong'] = now()
        save('execution.json', record)
        watchdog.terminate()
        watchdog.wait(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    # Wait for one new 100-Study progress line to prove the SAME B process is
    # alive after unpause. No inference or heavy hashing runs concurrently now.
    for attempt in range(90):
        lines = B_LOG.read_text().splitlines()
        if lines and lines[-1] != record['B_progress_before'] and B_LOG.stat().st_size > record['B_log_bytes_while_frozen']:
            record['B_progress_after'] = lines[-1]
            record['B_process_still_exists'] = Path(f'/proc/{B_PID}').exists()
            record['B_resume_verified'] = record['B_process_still_exists'] and lines[-1].startswith('study_student ')
            save('execution.json', record)
            if record['B_resume_verified']:
                print(json.dumps({k: record[k] for k in ('delivery', 'B_progress_after', 'B_host_pid', 'B_resume_verified')}, ensure_ascii=False, indent=2))
                return
            raise RuntimeError('B emitted an unexpected line after resume')
        time.sleep(1)
    raise RuntimeError('No new B progress line after 90 seconds; inspect the running process')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print('Preview only: freeze B at a verified compute boundary, time A, predict/audit new CSV, restore B.')
