"""Run the approved A -> validation table -> B -> validation/timing sequence.

Full training states and their rolling duplicates stay on local disk. Work gets
explicit inference-only model backups and bounded small evidence. No test run,
teacher inference, external upload, or modification of existing assets occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
LOCAL = Path('/tmp/cloud_student_runs_20261030')
EVIDENCE = ROOT / 'artifacts/kd_v1/student_upper_bound_20261030'
RUNS = {'A': LOCAL / '039_tiny768_p2_seed44', 'B': LOCAL / '040_base768_p2_p3_seed44'}
REFERENCE = ROOT.parent / 'deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2'
VALID = ROOT / 'data/clean_v1_reviewed/valid_manifest.csv'
RESERVE = 80 * 1024 ** 2
BACKUP_COMMITTED = {}


def stamp():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stage(name, **details):
    value = {'stage': name, 'updated_hong_kong': stamp(), 'controller_container_pid': os.getpid(),
             'runs': {k: str(v) for k, v in RUNS.items()}, **details}
    write_json(EVIDENCE / 'pipeline_status.json', value)
    print(f'[{stamp()}] stage={name} {json.dumps(details, ensure_ascii=False)}', flush=True)


def commands():
    a_record = json.loads((EVIDENCE / 'A.command.json').read_text())
    # This controller already runs inside the verified organizer image.
    a_command = [sys.executable, *a_record['command'][1:]]
    b_command = [sys.executable, '-u', str(ROOT / 'study_student.py'),
        '--fold-plan', str(ROOT / 'data/cv5_overnight/folds.json'),
        '--soft-labels', str(ROOT.parent / 'runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2'),
        '--output', str(RUNS['B']), '--image-root', '/home/featurize/data/cxr_first',
        '--pretrained-provenance-path', str(EVIDENCE / 'base_original_pretrained_provenance.json'),
        '--pretrained-weights-path', str(ROOT.parent / 'deliveries/038_dual768_optimized_project_20261030/pretrain_model/timm__convnext_base.fb_in22k_ft_in1k_384_bc0cfafc7d77/model.safetensors'),
        '--seed', '44', '--execute']
    evaluations = {}
    for label in ('A', 'B'):
        evaluations[label] = [sys.executable, '-u', str(ROOT / 'student_experiment_evaluation.py'),
            '--checkpoint', str(RUNS[label] / 'best.pt'),
            '--image-root', '/home/featurize/data/cxr_first', '--manifest', str(VALID),
            '--reference-model-dir', str(REFERENCE),
            '--reference-validation-dir', str(LOCAL / 'reference_033_current_2080ti'),
            '--output-dir', str(LOCAL / f'{label}_evaluation'), '--label', label, '--bootstrap', '1000']
    evaluations['B'] += ['--latency-inputs', str(ROOT.parent / 'deliveries/034_dual768_proxy_timing_20261029/bundle/inputs')]
    return a_command, b_command, evaluations


def freeze_sources():
    names = ['run_student_upper_bound.py', 'student_checkpoint_backup.py',
             'study_student.py', 'student_experiment_evaluation.py', 'student_experiments.py',
             'train_distill.py', 'train_overnight.py', 'dataset.py', 'dataset_overnight.py',
             'distill_data.py', 'distill_tta_data.py', 'model_overnight.py', 'kd_common.py',
             'deploy_kd.py', 'cost.py', 'inference_inputs.py', 'fast_lanczos.py']
    return {name: digest(ROOT / name) for name in names}


def require_sources(expected):
    changed = [name for name, sha in expected.items() if digest(ROOT / name) != sha]
    if changed:
        raise RuntimeError('Execution source changed after controller launch: ' + ', '.join(changed))


def backup(label):
    from student_checkpoint_backup import backup_run
    status_path = RUNS[label] / 'status.json'
    signature = None
    if status_path.is_file():
        current = json.loads(status_path.read_text())
        signature = (current.get('best_checkpoint_sha256'), current.get('last_checkpoint_sha256'),
                     current.get('state'))
        if signature == BACKUP_COMMITTED.get(label, (None,))[0]:
            return {**BACKUP_COMMITTED[label][1], 'controller_cached_no_op': True}
    result = backup_run(RUNS[label], LOCAL / 'full_checkpoint_backups' / label,
                        EVIDENCE / 'models' / label)
    write_json(EVIDENCE / f'{label}.backup_status.json', result)
    if result.get('state') == 'complete':
        BACKUP_COMMITTED[label] = (signature, result)
    return result


def sync_report(label):
    source = LOCAL / f'{label}_evaluation'
    target = EVIDENCE / 'reports' / label
    target.mkdir(parents=True, exist_ok=True)
    # Exported weights/config stay local; the separate model snapshots are the
    # persistent model backup. Do not duplicate large teacher ancestry configs.
    names = ['REPORT.md', 'REPORT.before_latency.md', 'metrics.json', 'metrics.before_latency.json',
             'valid_hflip2/validation_report.json', 'valid_hflip2/valid_predictions.csv',
             'valid_hflip2/valid.image_predictions.csv', 'latency_2080ti/latency_report.json']
    for name in names:
        original = source / name
        if not original.is_file():
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.stat().st_size > 25 * 1024 ** 2:
            raise RuntimeError('Unexpectedly large report; refusing work duplication')
        if shutil.disk_usage(target).free < original.stat().st_size + RESERVE:
            raise OSError('Not enough work space for report while preserving reserve')
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        shutil.copyfile(original, temporary)
        temporary.replace(destination)


def require_final_backup(label):
    from student_checkpoint_backup import refresh_run_metadata
    for attempt in range(3):
        result = backup(label)
        if result.get('state') == 'complete':
            result = refresh_run_metadata(RUNS[label], EVIDENCE / 'models' / label)
            write_json(EVIDENCE / f'{label}.backup_status.json', result)
            if result.get('state') == 'complete' and result.get('metadata_refreshed'):
                return result
        print(f'[{stamp()}] {label} final backup retry {attempt+1}: {result.get("reason")}', flush=True)
        time.sleep(1.)
    raise RuntimeError(f'{label} ended, but its complete backups are pending; see backup_status.json')


def wait_existing_a(interval):
    """Attach to A launched in its independent tmux session; never launch twice."""
    stage('A_training', attached_to_existing_tmux='student_a_20261030')
    last_reported = None
    while True:
        status_path = RUNS['A'] / 'status.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            state = status.get('state')
            receipts = sorted((RUNS['A'] / 'evidence').glob('command_*.json'))
            receipt = json.loads(receipts[-1].read_text()) if receipts else None
            if receipt is not None and receipt.get('exit_code') is not None:
                if receipt['exit_code'] != 0:
                    raise RuntimeError(f"A logger reported nonzero exit {receipt['exit_code']}")
                if state not in ('completed', 'early_stopped'):
                    raise RuntimeError('A process exited without a validated terminal state')
            if state in ('failed', 'paused'):
                raise RuntimeError('A did not complete: ' + json.dumps(status))
            backup('A')
            epoch = status.get('completed_epochs', 0)
            if epoch != last_reported:
                stage('A_training', training_status=status)
                last_reported = epoch
            if state in ('completed', 'early_stopped'):
                if epoch < 1 or not status.get('last_checkpoint_sha256'):
                    raise RuntimeError('A terminal state lacks a validated checkpoint')
                # The original logger must finish its real command receipt.
                if receipt is None or receipt.get('exit_code') is None:
                    time.sleep(interval)
                    continue
                require_final_backup('A')
                return status
        time.sleep(interval)


def run_child(command, *, label, name, interval):
    log_path = EVIDENCE / f'{label}.{name}.log'
    if log_path.exists():
        raise FileExistsError('Refuse to overwrite prior child log: ' + str(log_path))
    with log_path.open('xb', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        stage(f'{label}_{name}', child_container_pid=process.pid, log=str(log_path))
        while process.poll() is None:
            if name == 'training':
                backup(label)
            else:
                sync_report(label)
            time.sleep(interval)
        code = process.wait()
    if code:
        if name == 'training':
            backup(label)
        else:
            sync_report(label)
        raise RuntimeError(f'{label} {name} failed with exit {code}; see {log_path}')
    if name == 'training':
        require_final_backup(label)
    else:
        sync_report(label)
    if name == 'training':
        status = json.loads((RUNS[label] / 'status.json').read_text())
        if status.get('state') not in ('completed', 'early_stopped'):
            raise RuntimeError('Training exited without successful validated terminal state')


def execute(*, attach_a, interval):
    import fcntl
    from training_evidence_monitor import snapshot_source
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / 'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (EVIDENCE / 'pipeline_execution.json').exists():
            raise FileExistsError('This controller already has an execution record; inspect before rerunning')
        a_command, b_command, evaluations = commands()
        sources = freeze_sources()
        snapshot_source(ROOT, EVIDENCE / 'pipeline_source')
        record = {'started_hong_kong': stamp(), 'controller_container_pid': os.getpid(),
                  'attached_a': attach_a, 'commands': {'A': a_command, 'B': b_command, 'evaluations': evaluations},
                  'source_sha256': sources, 'teachers_frozen': True, 'test_predictions_enabled': False,
                  'backup_limits': 'Full optimizer/RNG state stays in two local copies on the same ephemeral instance; work stores exact best FP32 EMA and last FP16 inference-only backups. Transfer full local run directories before replacing the instance.'}
        write_json(EVIDENCE / 'pipeline_execution.json', record)
        if shutil.disk_usage(LOCAL).free < 12 * 1024 ** 3:
            raise OSError('Reserve >=12GiB local space for complete checkpoints and backups')
        try:
            if attach_a:
                wait_existing_a(interval)
            else:
                run_child(a_command, label='A', name='training', interval=interval)
            require_sources(sources)
            run_child(evaluations['A'], label='A', name='evaluation', interval=interval)
            print((EVIDENCE / 'reports/A/REPORT.md').read_text(), flush=True)
            # The full A-vs-033 table is durable before B begins; no user gate.
            require_sources(sources)
            stage('A_report_saved_B_starting', report=str(EVIDENCE / 'reports/A/REPORT.md'))
            run_child(b_command, label='B', name='training', interval=interval)
            require_sources(sources)
            run_child(evaluations['B'], label='B', name='evaluation', interval=interval)
            stage('completed', A_report=str(EVIDENCE / 'reports/A/REPORT.md'),
                  B_report=str(EVIDENCE / 'reports/B/REPORT.md'),
                  T4_cost_report='still_pending', engineering_optimization_performed=False)
        except BaseException as error:
            stage('failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attach-a', action='store_true')
    parser.add_argument('--interval', type=float, default=30.)
    args = parser.parse_args(argv)
    if not 1 <= args.interval <= 60:
        parser.error('interval must be between 1 and 60 seconds')
    if not args.execute:
        a, b, evaluation = commands()
        print(json.dumps({'preview': True, 'training_started': False, 'A': a, 'B': b,
                          'evaluation': evaluation}, indent=2))
        return 0
    execute(attach_a=args.attach_a, interval=args.interval)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
