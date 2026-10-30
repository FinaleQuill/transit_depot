"""Build a complete, editable A handoff on local disk without touching live B."""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
RUN=Path('/tmp/cloud_student_runs_20261030/039_tiny768_p2_seed44')
DEFAULT=Path('/tmp/A_training_handoff_20261030')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('Unexpected original training source: '+old[:70])
    return text.replace(old,new,1)


def patch_training(original):
    text=replace_once(original,
        '    pretrained_weights_path: Path | None = None\n',
        '    pretrained_weights_path: Path | None = None\n    student_init_checkpoint: Path | None = None\n')
    text=replace_once(text,'    prior.setdefault("pos_weight_power", 1.0)',
        '    prior.setdefault("student_init_checkpoint", None)\n    prior.setdefault("pos_weight_power", 1.0)')
    helper='''def load_student_initialization(path: Path, model_config: dict):
    """Load a student EMA, not teacher weights, with explicit new-run ancestry."""
    from deploy_kd import validate_student_checkpoint
    digest = sha256_file(path)
    parent = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise RuntimeError("Parent student checkpoint changed while reading")
    validate_student_checkpoint(parent)
    if parent["model_config"]["backbone_name"] != model_config["backbone_name"]:
        raise ValueError("Warm-start student backbone does not match the new model")
    provenance = json.loads(json.dumps(parent["pretrained_provenance"]))
    provenance["student_warm_start"] = {
        "parent_checkpoint_sha256": digest, "source_epoch": parent["epoch"],
        "weights_source": "ema", "optimizer_state_restored": False,
        "rng_state_restored": False, "new_experiment": True,
        "teacher_weights_loaded": False,
    }
    return parent, provenance


'''
    text=replace_once(text,'def run_training(config: DistillConfig) -> dict[str, Any]:',
        helper+'def run_training(config: DistillConfig) -> dict[str, Any]:')
    text=replace_once(text,'    checkpoint = None\n    serialized = _configuration(config)',
        '    checkpoint = None\n    warm_start = None\n    serialized = _configuration(config)')
    text=replace_once(text,'    else:\n        if config.pretrained_provenance_path is not None:',
        '    else:\n        if config.student_init_checkpoint is not None:\n'
        '            warm_start, provenance = load_student_initialization(\n'
        '                config.student_init_checkpoint, config.model_config())\n'
        '        elif config.pretrained_provenance_path is not None:')
    text=replace_once(text,'    model = build_overnight_model(pretrained=checkpoint is None,',
        '    model = build_overnight_model(pretrained=checkpoint is None and warm_start is None,')
    text=replace_once(text,'pretrained_weights_path=provenance["weights_path"] if checkpoint is None else None,',
        'pretrained_weights_path=provenance["weights_path"] if checkpoint is None and warm_start is None else None,')
    text=replace_once(text,'    ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,',
        '    if warm_start is not None:\n'
        '        model.load_state_dict(warm_start["model_state_dict"], strict=True)\n'
        '        warm_start = None\n'
        '    ema = ModelEmaV3(model, decay=config.ema_decay, device=device, use_warmup=True,')
    text=replace_once(text,'    parser.add_argument("--resume", type=Path)',
        '    parser.add_argument("--student-init-checkpoint", type=Path)\n    parser.add_argument("--resume", type=Path)')
    text=replace_once(text,'"audit_trail.py", "student_experiments.py"))',
        '"audit_trail.py", "student_experiments.py", "portable_a.py"))')
    compile(text,'packaged/train_distill.py','exec')
    return text


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)


def build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a new bundle directory')
    output.mkdir(parents=True)
    (output/'project').mkdir();(output/'tools').mkdir()
    status=json.loads((RUN/'status.json').read_text())
    for role in ('best','last'):
        if sha(RUN/(role+'.pt'))!=status[role+'_checkpoint_sha256']:
            raise ValueError('Original full checkpoint identity differs: '+role)
    archive=next((RUN/'evidence').glob('source_*'))
    code_names=['train_distill.py','train_overnight.py','student_experiments.py','model_overnight.py',
        'dataset.py','dataset_overnight.py','cv_splits.py','distill_data.py','distill_tta_data.py',
        'runtime_compat.py','kd_common.py','audit_trail.py','training_evidence_monitor.py',
        'planned_handoff.py','export_student.py','deploy_kd.py','student_experiment_evaluation.py',
        'compare_student_candidates.py','inference_inputs.py','inference_runtime.py',
        'inference_reproducibility.py','fast_lanczos.py','prepare_inference.py','predict.py',
        'cost.py','validate_student.py','benchmark_local_resources.py','requirements_kd.txt',
        'requirements_inference.txt']
    for name in code_names:
        # Original A training dependencies come from its actual execution snapshot.
        source=archive/name if (archive/name).is_file() else ROOT/name
        copy(source,output/'project'/name)
    original=(output/'project/train_distill.py').read_text()
    modified=patch_training(original)
    (output/'project/train_distill.py').write_text(modified)
    (output/'PORTABILITY.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True),modified.splitlines(keepends=True),
        fromfile='original_A_execution/train_distill.py',tofile='project/train_distill.py')))
    for name in ('run_in_image.py','prepare_exact_resume.py','verify_files.py'):
        copy(ROOT/'handoff_a'/name,output/'tools'/name)
    copy(ROOT/'handoff_a/portable_a.py',output/'project/portable_a.py')
    copy(ROOT/'handoff_a/README_zh.md',output/'README.md')
    (output/'project/AGENTS.md').write_text(
        '本目录是A学生的可编辑交接副本。先读../README.md和../rules两PDF。\n'
        '只训练学生；直接复用冻结OOF，不训练教师。原始胸片/parent_run/oof只读，新增run写/experiments。\n'
        'train/valid保持分离，test不得参与训练、拟合或模型选择；禁止外部数据/违规初始化。\n'
        '先预览再显式--execute，运行中不改源码；保留真实日志/指标/来源与完整断点。\n')
    for name in ('train_clean.csv','valid_manifest.csv','test_manifest.csv'):
        copy(ROOT/'data/clean_v1_reviewed'/name,output/'project/data/clean_v1_reviewed'/name)
    copy(ROOT/'data/cv5_overnight/folds.json',output/'project/data/cv5_overnight/folds.json')
    shutil.copytree(RUN,output/'parent_run',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    oof=ROOT.parent/'runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2'
    for name in ('soft_labels.npz','metadata.json'):copy(oof/name,output/'oof'/name)
    copy(ROOT.parent/'runs/030_tiny768_kd_seed44/pretrained_provenance.json',output/'provenance/original_tiny.json')
    copy(ROOT.parent/'deliveries/038_dual768_optimized_project_20261030/pretrain_model/timm__convnext_tiny.fb_in22k_ft_in1k_384_a5cbe10cfa2e/model.safetensors',output/'pretrain/model.safetensors')
    shutil.copytree(ROOT/'native_lanczos',output/'project/native_lanczos',ignore=shutil.ignore_patterns('__pycache__','build','dist'))
    for name in ('test_kd_student.py','test_kd_student_tta.py','test_kd_resume_commit.py','test_student_experiments.py'):
        copy(ROOT/'tests'/name,output/'project/tests'/name)
    for source,name in [(Path('/tmp/dual768_competition_addons'),'inference_addons'),
                        (Path('/tmp/cloud_student_training_addons_cp312'),'training_addons')]:
        shutil.copytree(source,output/'environment'/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','bin'))
    copy(archive/'environment_lock.txt',output/'environment/original_execution_lock.txt')
    copy(ROOT/'artifacts/kd_v1/student_takeover_20261030/student_training_addons_requirements.txt',
         output/'environment/training_addons_requirements.txt')
    for name in ('rules.pdf','rules_1.pdf'):copy(ROOT.parent/name,output/'rules'/name)
    for source,name in [
        (ROOT/'artifacts/kd_v1/student_upper_bound_20261030/reports/A','A_valid'),
        (ROOT.parent/'deliveries/041_tiny768_p2_A_submission_20261030/local_resource_report.json','latency_2080ti.json'),
    ]:
        if source.is_dir():shutil.copytree(source,output/'reports'/name)
        else:copy(source,output/'reports'/name)
    info={'created_hong_kong':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
          'purpose':'editable A student handoff; no teacher training required',
          'best_epoch':9,'last_epoch':14,'best_checkpoint_sha256':status['best_checkpoint_sha256'],
          'last_checkpoint_sha256':status['last_checkpoint_sha256'],
          'original_A_state':'early_stopped','actual_fp16_valid_macro_auc':0.809115940852,
          'original_training_sources_preserved':'parent_run/evidence/source_*',
          'working_training_code_portability_diff':'PORTABILITY.patch',
          'teacher_weights_included':False,'teacher_retraining_required':False,
          'raw_images_included':False,'test_predictions_included':False,
          'original_run_modified':False,'live_B_modified_or_paused':False,
          'git_commit_at_packaging':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (output/'BUNDLE_INFO.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    manifest=[]
    for p in sorted(output.rglob('*')):
        if p.is_file():manifest.append({'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p)})
    (output/'MANIFEST.json').write_text(json.dumps({'files':manifest,'file_count':len(manifest),
        'total_file_bytes':sum(row['bytes'] for row in manifest)},indent=2)+'\n')
    print(json.dumps({'bundle':str(output),'files':len(manifest),'uncompressed_bytes':sum(x['bytes'] for x in manifest)},indent=2))


def archive(output):
    output=Path(output).resolve();target=output.with_suffix('.zip')
    if target.exists():raise FileExistsError(target)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True,strict_timestamps=False) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and not any(x in p.relative_to(output).parts for x in ('experiments','__pycache__')):
                z.write(p,output.name+'/'+str(p.relative_to(output)))
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP CRC failure: '+bad)
    receipt={'zip':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'CRC_verified':True}
    target.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+target.name+'\n')
    target.with_suffix('.zip.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def refresh_manifest(output):
    output=Path(output).resolve()
    rows=[]
    for path in sorted(output.rglob('*')):
        relative=path.relative_to(output)
        if (not path.is_file() or str(relative)=='MANIFEST.json'
                or any(value in relative.parts for value in ('experiments','__pycache__'))):
            continue
        rows.append({'path':str(relative),'bytes':path.stat().st_size,'sha256':sha(path)})
    (output/'MANIFEST.json').write_text(json.dumps({'files':rows,'file_count':len(rows),
        'total_file_bytes':sum(row['bytes'] for row in rows)},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--archive-only',action='store_true')
    parser.add_argument('--refresh-manifest',action='store_true')
    args=parser.parse_args()
    if args.refresh_manifest:refresh_manifest(args.output)
    elif args.archive_only:archive(args.output)
    else:build(args.output)
