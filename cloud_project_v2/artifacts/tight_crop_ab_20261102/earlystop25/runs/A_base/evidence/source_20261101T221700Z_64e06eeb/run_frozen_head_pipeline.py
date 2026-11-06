"""Detached frozen-FC experiments, fixed-rule evaluation, and actual valid audit.

The selected 045 test CSV is never read. No new test prediction is scheduled.
All large caches/merged weights stay local; only small patches/evidence persist.
"""
from __future__ import annotations

from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'docs/FROZEN_HEAD_PLAN_20261101.json'
PLAN_SHA = 'a6b906e232c9304983d16644329de8edffbb8a49079e5f0e4cf6a063ce551565'
EVIDENCE = ROOT / 'artifacts/kd_v1/frozen_head_20261101'
LOCAL = Path('/tmp/cloud_student_runs_20261030/frozen_head_20261101')
CACHE = LOCAL / 'cache'
FITS = EVIDENCE / 'fits'
WRAPPER = ROOT / 'run_student_environment.sh'
HEADS = ['A_F1', 'A_F2', 'B_F1', 'B_F2']
STATE = {'state': 'starting', 'stage': 'attach_feature_cache', 'pipeline_pid': os.getpid(),
         'plan_sha256': PLAN_SHA, 'test_used_for_training': False, 'new_test_prediction_scheduled': False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def update(**values):
    STATE.update(values, updated_hong_kong=stamp())
    write(EVIDENCE/'pipeline_status.json', STATE)
    cache_log = EVIDENCE/'cache.log'
    tail = cache_log.read_text(errors='replace').splitlines()[-1:] if cache_log.exists() else []
    lines = ['# 冻结骨干、重训分类头：后台状态', '',
        f"状态：{STATE['state']}；阶段：{STATE['stage']}", f"更新：{STATE['updated_hong_kong']}",
        '原预算截止：2026-11-02 02:37（香港）', '',
        '045最高分CSV已经生成并审计；本后台不读取它作训练/选择，不自动生成新的test CSV。',
        '缓存：'+(tail[0] if tail else '等待日志'), '']
    for name in HEADS:
        receipt = read(FITS/name/'training_receipt.json', {})
        lines.append(f"{name}: {receipt.get('state','尚未开始')}，"
                     f"{receipt.get('completed_epochs',0)}/12轮")
    lines += ['', 'CPU缓存分数只是代理；最终以GPU FP16完整valid和端到端计时为准。']
    if STATE.get('actual_results'):
        lines += ['', '实际GPU完整valid：']
        for name, result in STATE['actual_results'].items():
            lines.append(f"- {name}: AUC={result['macro_auc']:.9f}，{result['mean_ms']:.3f}ms")
    if STATE.get('reason'):
        lines += ['', '说明：'+STATE['reason']]
    lines += ['', '全局state=completed时通知代理复盘；failed/budget_exhausted时也可直接通知。',
              '不要重复启动本pipeline或缓存。大文件在/tmp；小patch与日志已保存在此目录。', '']
    (EVIDENCE/'STATUS.md').write_text('\n'.join(lines))


def container_exists(name):
    result = subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],
                            capture_output=True,text=True,timeout=20)
    return result.returncode == 0 and result.stdout.strip() == 'true'


def kill_container(name):
    subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)


def require_sources():
    if sha(PLAN) != PLAN_SHA or any(sha(ROOT/name) != value for name,value in SOURCES.items()):
        raise RuntimeError('A registered plan or frozen execution source changed')


def run_stage(stage, arguments, maximum_seconds):
    require_sources()
    update(stage=stage)
    container = 'codex-frozen-'+stage.lower().replace('_','-')+'-20261101'
    if container_exists(container):
        raise RuntimeError('A stage container is already running; refusing duplication')
    command = ['bash',str(WRAPPER),'-u',*map(str,arguments)]
    record = {'stage':stage,'started_hong_kong':stamp(),'command':command,
              'container':container,'plan_sha256':PLAN_SHA}
    write(EVIDENCE/(stage+'.command.json'),record)
    environment = {**os.environ,'STUDENT_RUNTIME_MODE':'inference','STUDENT_CONTAINER_NAME':container}
    started = time.monotonic()
    with (EVIDENCE/(stage+'.log')).open('x') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                    stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            if time.monotonic()-started > maximum_seconds or time.time() > DEADLINE-300:
                kill_container(container)
                process.wait(timeout=60)
                raise TimeoutError('Budget reached during '+stage)
            update()
            time.sleep(10)
        record.update(exit_code=process.wait(),finished_hong_kong=stamp(),
                      seconds=time.monotonic()-started)
    write(EVIDENCE/(stage+'.command.json'),record)
    if record['exit_code'] != 0:
        raise RuntimeError(stage+' failed; inspect its log')


def await_cache():
    update(state='running',stage='feature_cache')
    launch = read(EVIDENCE/'cache_launch.json')
    if launch is None:
        raise RuntimeError('No existing feature cache launch; never silently launch a duplicate')
    started = datetime.fromisoformat(launch['started_hong_kong']).timestamp()
    limit = min(started+REGISTRATION['budget']['cache_seconds_max'],DEADLINE-3*3600)
    while True:
        metadata = read(CACHE/'meta.json',{})
        if metadata.get('complete') is True:
            if metadata.get('plan_sha256') != PLAN_SHA or not all(
                    metadata['splits']['valid']['baseline_exact_reproduction'][name]['all_image_probabilities_bitwise_equal']
                    for name in ('A','B')):
                raise ValueError('Completed cache is not the registered exact frozen baseline')
            shutil.copyfile(CACHE/'meta.json',EVIDENCE/'cache_metadata.json')
            for name in ('A','B'):
                shutil.copyfile(CACHE/metadata['encoders'][name]['head_file'],EVIDENCE/(name+'_original_head_fp32.pt'))
            return
        if time.time() >= limit:
            kill_container(launch['container'])
            raise TimeoutError('Feature extraction exceeded its fixed time budget')
        if not container_exists(launch['container']):
            raise RuntimeError('Feature cache exited without complete baseline reproduction')
        update()
        time.sleep(10)


def write_patch_config(name):
    encoder,recipe = name.split('_')
    receipt = read(FITS/name/'training_receipt.json')
    declaration = REGISTRATION['encoders'][encoder]
    patch = FITS/name/'patch_fp16.pt'
    if (receipt.get('state') != 'completed' or receipt.get('final_epoch') != 12
            or receipt.get('training_plan_sha256') != PLAN_SHA
            or receipt.get('head_patch_fp16_sha256') != sha(patch)):
        raise ValueError('A classifier did not finish its registered final epoch')
    base = read(Path(declaration['base_model_dir'])/'config.json')
    config = {'schema_version':1,'artifact_type':'frozen_head_student_fp16',
        **{key:declaration[key] for key in ('base_model_dir','base_config_sha256','base_weights_sha256',
                                          'backbone_name','feature_dim')},
        'patch_file':'patch_fp16.pt','patch_sha256':sha(patch),'replaced_class_ids':list(range(1,11)),
        'class_ids':list(range(1,11)),'tta':'hflip2','image_size':768,'precision':'float16','epochs':12,
        'encoder':encoder,'recipe':recipe,'training_plan_sha256':PLAN_SHA,
        'feature_cache_metadata_sha256':receipt['feature_cache_metadata_sha256'],
        'training_receipt_sha256':sha(FITS/name/'training_receipt.json'),
        'training_config':receipt['training_config'],
        **{key:base[key] for key in ('source_manifest_sha256','fold_plan_sha256','official_valid_manifest_sha256')},
        'test_used_for_training_or_selection':False}
    write(FITS/name/'config.json',config)


def final_actual_comparison():
    # Only complete, actually reproduced valid probabilities enter this review.
    code = '''import json,numpy as np
from pathlib import Path
from dataset import read_manifest
from deploy_kd import read_submission,sha256_file
from compare_student_candidates import paired_intervals
from student_experiment_evaluation import score_probabilities,validation_labels
root=Path("artifacts/kd_v1/frozen_head_20261101")
records=read_manifest("data/clean_v1_reviewed/valid_manifest.csv",split="valid")
labels,counts=validation_labels(records);keys=sorted(labels)
paths={"current_best_817149":Path("artifacts/kd_v1/bounded_finetune_20261101/actual_combination_review/full_decode/valid")}
paths.update({p.parent.name:p.parent/"valid" for p in (root/"actual").glob("*/actual_audit.json")})
arrays={};scores={};bindings={}
for name,path in paths.items():
 report=json.loads((path/"validation_report.json").read_text())
 assert report["complete"] and report["predictions_sha256"]==sha256_file(path/"valid_predictions.csv")
 values=read_submission(path/"valid_predictions.csv");scores[name]=score_probabilities(labels,values,counts)
 arrays[name]=np.array([values[k] for k in keys]);bindings[name]={"report_sha256":sha256_file(path/"validation_report.json"),"predictions_sha256":sha256_file(path/"valid_predictions.csv")}
intervals=paired_intervals(np.array([labels[k] for k in keys]),arrays,np.array([k[0] for k in keys]),samples=1000,seed=50,reference="current_best_817149")
(root/"actual_comparison.json").write_text(json.dumps({"scores":scores,"paired_patient_bootstrap":intervals,"bindings":bindings,"test_used":False,"selection_adjusted":False},indent=2)+chr(10))
'''
    run_stage('actual_comparison',['-c',code],900)


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    with (EVIDENCE/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        global REGISTRATION,DEADLINE,SOURCES
        if sha(PLAN) != PLAN_SHA:
            raise ValueError('Registered head plan changed')
        REGISTRATION = read(PLAN)
        DEADLINE = datetime.fromisoformat(REGISTRATION['deadline_hong_kong']).timestamp()
        names = ['frozen_head_cache.py','frozen_head_fit.py','frozen_head_evaluate.py',
                 'frozen_head_predictor.py','frozen_head_actual.py','student_ensemble.py',
                 'inference_inputs.py','deploy_kd.py','cost.py','model_overnight.py',
                 'run_frozen_head_pipeline.py']
        SOURCES = {name:sha(ROOT/name) for name in names}
        write(EVIDENCE/'source_receipt.json',{'plan_sha256':PLAN_SHA,'sources_sha256':SOURCES,'created_hong_kong':stamp()})
        try:
            await_cache()
            require_sources()
            fit_started = time.monotonic()
            for name in HEADS:
                encoder,recipe = name.split('_')
                output = FITS/name
                if output.exists():
                    raise FileExistsError('Refusing duplicate or overwritten head fitting: '+name)
                remaining = REGISTRATION['budget']['all_head_fitting_seconds_max']-(time.monotonic()-fit_started)
                if remaining < 60 or time.time()+1800 > DEADLINE:
                    raise TimeoutError('Not enough fixed fitting/evaluation budget')
                run_stage('fit_'+name,[ROOT/'frozen_head_fit.py','--cache',CACHE,'--plan',PLAN,
                    '--plan-sha256',PLAN_SHA,'--encoder',encoder,'--recipe',recipe,'--output',output,'--execute'],remaining)
                write_patch_config(name)
            run_stage('proxy_evaluation',[ROOT/'frozen_head_evaluate.py','--cache',CACHE,'--plan',PLAN,
                '--plan-sha256',PLAN_SHA,'--fits-root',FITS,'--output',EVIDENCE/'proxy_evaluation'],1200)
            report = read(EVIDENCE/'proxy_evaluation/comparison_report.json')
            if not report.get('complete') or report.get('plan_sha256') != PLAN_SHA:
                raise ValueError('Missing registered complete proxy evaluation')
            shortlist = report['actual_gpu_full_valid_shortlist']
            actual_started = time.monotonic()
            results = {}
            for item in shortlist:
                name = item['pair_name']
                remaining = REGISTRATION['budget']['actual_evaluation_seconds_max']-(time.monotonic()-actual_started)
                if remaining < 600 or time.time()+900 > DEADLINE:
                    raise TimeoutError('Insufficient budget for another whole actual validation')
                run_stage('actual_'+name,[ROOT/'frozen_head_actual.py','--plan',PLAN,
                    '--plan-sha256',PLAN_SHA,'--fits-root',FITS,'--pair-name',name,'--cache',CACHE,
                    '--output',LOCAL/'actual'/name,'--report-dir',EVIDENCE/'actual'/name],remaining)
                audit = read(EVIDENCE/'actual'/name/'actual_audit.json')
                if not audit or not audit.get('complete'):
                    raise ValueError('Actual candidate did not complete its integrity audit')
                results[name] = {'macro_auc':audit['macro_auc'],'mean_ms':audit['mean_ms']}
                update(actual_results=results)
            final_actual_comparison()
            best_name = max(results,key=lambda name:results[name]['macro_auc'])
            best = results[best_name]
            update(state='completed',stage='ready_for_review',actual_results=results,
                target_met_in_actual_valid=best['macro_auc']>=.82,
                improved_over_existing_best=best['macro_auc']>.8171491901128627,
                reason='四个固定12轮小头、预登记组合代理评估和最高两项实际FP16/完整valid/计时均完成；等待复盘。045CSV保持既定版本，没有自动新test预测。')
            write(EVIDENCE/'completion.json',{**STATE,'best_new_pair':best_name,'best_new_actual':best,
                'T4_certified':False,'selection_bias_corrected':False})
        except BaseException as error:
            state = 'budget_exhausted' if isinstance(error,TimeoutError) else 'failed'
            update(state=state,reason=f'{type(error).__name__}: {error}')
            (EVIDENCE/'pipeline_error.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
