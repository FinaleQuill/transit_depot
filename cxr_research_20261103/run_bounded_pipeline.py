"""Detached, time-bounded training/evaluation handoff; never predicts test.

Attach to the existing X1 run, verify its final EMA, evaluate it, then execute
the already registered X2 stage if the remaining budget permits. A filesystem
lock prevents a second pipeline. The user returns for final selection/timing.
"""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_bounded_finetune import (ROOT, EVIDENCE, PLAN, PLAN_SHA, PACKING_SHA,
    WRAPPER, registered, run_directory, stamp, write)


STATE = {'state': 'starting', 'stage': 'attach_X1', 'test_predicted': False,
         'final_submission_selected': False, 'registered_plan_sha256': PLAN_SHA,
         'packing_addendum_sha256': PACKING_SHA, 'pipeline_pid': os.getpid()}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def update(**values):
    STATE.update(values, updated_hong_kong=stamp())
    write(EVIDENCE / 'pipeline_status.json', STATE)
    lines = ['# 本轮后台训练状态', '', f"状态：{STATE['state']}；阶段：{STATE['stage']}",
        f"更新时间：{STATE['updated_hong_kong']}", '截止：2026-11-02 02:37（香港）', '',
        '只运行登记的train微调和valid评估。不会自动预测test、覆盖044或上传。', '']
    for stage in ('X1', 'X2'):
        status = read_json(run_directory(stage) / 'status.json', {})
        log = EVIDENCE / f'{stage}_packed.training.log'
        tail = log.read_text(errors='replace').splitlines()[-1:] if log.is_file() else []
        final = read_json(EVIDENCE / f'{stage}.final_ema.json')
        lines += [f"{stage}：已完成 {status.get('completed_epochs', 0)}/4 轮；"
                  f"训练状态 {status.get('state', '尚未开始')}；最终FP16已保存：{final is not None}",
                  '最新日志：' + (tail[0] if tail else '等待训练日志'), '']
    if STATE.get('valid_scores'):
        lines += ['已完成的valid结果（不是平台test分数）：', '']
        lines += [f'- {name}: {score:.9f}' for name, score in STATE['valid_scores'].items()]
        lines += ['']
    if STATE.get('reason'):
        lines += ['说明：' + STATE['reason'], '']
    lines += ['完成标志：pipeline_status.json 的 state 为 completed。',
              '若为 failed，请查看日志并通知代理；若为 budget_exhausted，不再启动后续训练。',
              '训练完成后通知代理进行选模、实际组合复现与端到端计时；当前044继续保留。', '']
    (EVIDENCE / 'STATUS.md').write_text('\n'.join(lines))


def launcher_alive(stage):
    record = read_json(EVIDENCE / f'{stage}_packed.execution.json', {})
    pid = record.get('host_launcher_pid')
    if type(pid) is not int:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:
        return False
    return 'run_bounded_finetune.py' in args and stage in args


def kill_container(stage, *, force=False):
    name = f'codex-bounded-study-{stage.lower()}-packed-20261101'
    args = ['docker', 'kill', *([] if force else ['--signal=TERM']), name]
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)


def run_inference(stage, *, export=False):
    command = ['bash', str(WRAPPER), '-u',
        str(ROOT / ('run_bounded_finetune.py' if export else 'evaluate_bounded_finetune.py')),
        '--stage', stage]
    if export:
        command.extend(['--action', 'export', '--execute'])
    filename = f'{stage}.fallback_export.log' if export else f'{stage}.evaluation.log'
    environment = {**os.environ, 'STUDENT_RUNTIME_MODE': 'inference'}
    with (EVIDENCE / filename).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if time.time() >= DEADLINE:
                process.terminate()
                raise TimeoutError('Wall-clock budget reached during evaluation')
            update()
            time.sleep(15)
        if process.wait() != 0:
            raise RuntimeError(f'{stage} inference verification failed; see {filename}')


def wait_training(stage, owned_process=None):
    update(stage='train_' + stage, state='running')
    stop_requested = None
    while True:
        final = read_json(EVIDENCE / f'{stage}.final_ema.json')
        record = read_json(EVIDENCE / f'{stage}_packed.execution.json', {})
        status = read_json(run_directory(stage) / 'status.json', {})
        if final is not None:
            recipe = registered(stage)[1]
            if (status.get('state') != 'completed' or status.get('completed_epochs') != recipe['epochs']
                    or final.get('registered_plan_sha256') != PLAN_SHA
                    or final.get('packing_addendum_sha256') != PACKING_SHA
                    or final.get('source_epoch') != recipe['epochs']):
                raise ValueError('Training completion/final export does not match registration')
            if owned_process is not None and owned_process.wait(timeout=90) != 0:
                raise RuntimeError(f'{stage} launcher exited unsuccessfully')
            update()
            return
        if record.get('state') == 'incomplete_or_failed' or status.get('state') in ('failed', 'paused'):
            raise RuntimeError(f'{stage} did not finish its fixed four epochs; no replacement model selected')
        if not launcher_alive(stage):
            if status.get('state') == 'completed' and status.get('completed_epochs') == 4:
                # The interactive tool session may have disappeared after model
                # training. Recover only the final export, never retrain X1.
                run_inference(stage, export=True)
                continue
            name = f'codex-bounded-study-{stage.lower()}-packed-20261101'
            alive = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                capture_output=True, text=True, timeout=20)
            if alive.returncode != 0 or alive.stdout.strip() != 'true':
                raise RuntimeError(f'{stage} launcher and training container are absent')
        if record.get('started_hong_kong'):
            started = datetime.fromisoformat(record['started_hong_kong']).timestamp()
            cap = min(started + record['wall_seconds_cap'], DEADLINE - 3600)
            if time.time() >= cap and stop_requested is None:
                kill_container(stage)
                stop_requested = time.time()
                update(reason=f'{stage} 已请求按预算停止；不将部分训练称作第4轮候选。')
            if stop_requested is not None and time.time() - stop_requested >= 600:
                kill_container(stage, force=True)
                raise TimeoutError(f'{stage} exceeded its training time cap')
        update()
        time.sleep(15)


def evaluate_and_review(stage):
    update(stage='validate_' + stage)
    comparison_path = EVIDENCE / f'comparison_after_{stage}/comparison_report.json'
    if not comparison_path.is_file():
        run_inference(stage)
    comparison = read_json(comparison_path)
    if not comparison.get('complete') or comparison.get('registered_plan_sha256') != PLAN_SHA:
        raise ValueError('Missing or mismatched complete fixed-rule evaluation')
    scores = {name: row['macro_auc'] for name, row in comparison['candidates'].items()}
    history = [json.loads(line) for line in (run_directory(stage) / 'history.jsonl').read_text().splitlines()]
    review = {'stage': stage, 'reviewed_hong_kong': stamp(), 'valid_scores': scores,
        'history_epoch_seconds': [row['training']['seconds'] for row in history],
        'comparison_report': str(comparison_path), 'test_used': False,
        'decision_basis': 'Fixed recipes and final epochs; no extension or new weight search based on intermediate valid.',
        'limitations': 'Reused official valid; paired intervals do not correct historical selection. Final timing and real T4 retest remain pending.'}
    write(EVIDENCE / f'{stage}.phase_review.json', review)
    update(valid_scores=scores)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = registered('X1')[0]
        global DEADLINE
        DEADLINE = datetime.fromisoformat(plan['budget']['deadline_hong_kong']).timestamp()
        try:
            update(state='running', stage='attach_existing_X1')
            wait_training('X1')
            evaluate_and_review('X1')
            if time.time() + 7 * 3600 > DEADLINE:
                update(state='budget_exhausted', stage='X1_complete_X2_not_started',
                       reason='剩余时间不足以预留X2最多6小时及最终验证1小时；保留X1结果等待复盘。')
                return
            if run_directory('X2').exists() or (EVIDENCE / 'X2_packed.execution.json').exists():
                raise FileExistsError('X2 already exists; refusing a duplicate training launch')
            update(stage='launch_X2', reason='X1来源、完整valid及固定规则已核验；剩余预算足够，执行已登记弱KD对照。')
            with (EVIDENCE / 'X2.controller.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_bounded_finetune.py'),
                    '--stage', 'X2', '--execute'], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                # Wait until the launch receipt exists before attaching.
                for _ in range(30):
                    if (EVIDENCE / 'X2_packed.execution.json').exists():
                        break
                    if process.poll() is not None:
                        raise RuntimeError('X2 launcher failed before writing its execution receipt')
                    time.sleep(1)
                wait_training('X2', process)
            evaluate_and_review('X2')
            update(state='completed', stage='ready_for_review',
                reason='两项固定4轮训练、最终FP16持久化及完整valid固定规则评估已完成；等待代理选模和端到端计时。没有自动生成test或更改044。')
        except BaseException as exc:
            update(state='failed', reason=f'{type(exc).__name__}: {exc}')
            (EVIDENCE / 'pipeline_error.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
