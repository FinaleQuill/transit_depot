"""Freeze the authorized train/valid exploration plan without launching it."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/resolution_explore_20261103'
QUEUE = BASE / 'queue'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    plan_path = BASE/'queue_plan.json'
    recovery_path = QUEUE/'attempts'/'active_recovery.json'
    recovery = read(recovery_path) if recovery_path.is_file() else None
    archive = QUEUE/'attempts'/'original_resolution_plan.json'
    if not archive.exists():
        write(archive, read(plan_path))
    old = read(archive)
    old_status = QUEUE/'status.json'
    archived_status = QUEUE/'attempts'/'original_permission_failure_status.json'
    if old_status.exists() and not archived_status.exists():
        write(archived_status, read(old_status))
    run640 = Path(old['jobs'][0]['run_dir'])
    cfg640 = read(run640/'config.json')
    events = [json.loads(x) for x in (run640/'compliance/events.jsonl').read_text().splitlines()]
    first = next(x for x in events if x['event_type']=='training_started')
    manifest = read(run640/'compliance'/first['session_evidence']/'source_manifest.json')
    prior_sources = {x['original_path']:x['sha256'] for x in manifest}
    current_sources = {p:sha(p) for p in prior_sources}
    source_equivalence = None
    if current_sources != prior_sources:
        # Preserve the real historical SHA. The only accepted difference is a
        # documented legacy-resume default, which is inactive for all these runs.
        legacy_patch = ("    # Older checkpoints predate fixed-resolution validation; None means the\n"
                        "    # historical behavior of validating at the training image size.\n"
                        "    prior.setdefault('validation_size', None)\n")
        trainer_path = str(ROOT/'train_distill.py')
        changed = [p for p in prior_sources if current_sources[p] != prior_sources[p]]
        current_text = Path(trainer_path).read_text()
        equivalent_digest = hashlib.sha256(current_text.replace(legacy_patch, '', 1).encode()).hexdigest()
        if (changed != [trainer_path] or current_text.count(legacy_patch) != 1
                or equivalent_digest != prior_sources[trainer_path]
                or cfg640.get('validation_size') != 768):
            raise RuntimeError('Base trainer has an unreviewed change since the 640 control')
        starts = [e for e in events if e['event_type']=='training_started']
        session_evidence = []
        for event in starts:
            captured = read(run640/'compliance'/event['session_evidence']/'configuration.json')
            if captured.get('validation_size') != 768:
                raise RuntimeError('Historical resume lacks the explicit validation size needed for equivalence')
            captured_sources = read(run640/'compliance'/event['session_evidence']/'source_manifest.json')
            if {r['original_path']:r['sha256'] for r in captured_sources} != prior_sources:
                raise RuntimeError('Historical sessions have different source identities')
            trainer_row = next(r for r in captured_sources if r['original_path']==trainer_path)
            archived_trainer = run640/'compliance'/trainer_row['archived_path']
            if (sha(archived_trainer) != prior_sources[trainer_path]
                    or archived_trainer.read_bytes() != current_text.replace(legacy_patch, '', 1).encode()):
                raise RuntimeError('Historical trainer snapshot differs from source-equivalence proof')
            session_evidence.append({'session_id':event['session_id'],
                'configuration_sha256':event['configuration_sha256'],
                'source_manifest_sha256':event['source_manifest_sha256'],
                'validation_size':captured['validation_size']})
        source_equivalence = QUEUE/'attempts'/'reviewed_legacy_resume_source_equivalence.json'
        write(source_equivalence, {
            'reviewed_utc':datetime.now(timezone.utc).isoformat(),
            'prior_source_sha256':prior_sources[trainer_path],
            'current_source_sha256':current_sources[trainer_path],
            'exact_inserted_text':legacy_patch, 'allowed_fragment_occurrences':1,
            'historical_sessions':session_evidence,
            'source_after_removing_exact_patch_sha256':equivalent_digest,
            'applies_to':'train_distill.resume_training_configuration only',
            'reason':'031 sessions already declare validation_size=768 so setdefault is a no-op; new arms use verified EMA warm-start, not resume',
            'training_forward_loss_optimizer_augmentation_changes':False,
            'historical_run_evidence_modified':False})
    extension_sources = {**current_sources, **{str(ROOT/n):sha(ROOT/n) for n in
        ('exploration_train.py','roi_scale.py','tight_crop.py','spatial_mil.py')}}
    roi_gallery = ROOT/'artifacts/roi_magnify_explore_20261103_v2_2/gallery'
    gallery = read(roi_gallery/'metrics.json')
    if gallery['count']!=48 or gallery['method_version']!='body_envelope_magnify_only_v2_2':
        raise RuntimeError('Final reviewed ROI gallery missing')
    for filename,digest in gallery['code_sha256'].items():
        if sha(ROOT/filename)!=digest:
            raise RuntimeError('ROI gallery differs from current source: '+filename)
    if any(x['diagnostics']['effective_magnification']<1 for x in gallery['items']):
        raise RuntimeError('ROI gallery contains shrinkage')
    review = roi_gallery/'SELF_REVIEW.md'
    if not review.is_file():
        raise RuntimeError('ROI self-review must be written before queue registration')
    excluded = {'model_config','preprocessing','distillation','student_warm_start','resume',
                'init_student_ema','init_student_ema_sha256'}
    common = {k:v for k,v in cfg640.items() if k not in excluded}
    jobs=[]
    evaluations={}
    comparisons={}
    def register(name,kind,prereg,**kwargs):
        path=BASE/'prereg'/prereg
        if len([x for x in path.read_text().splitlines() if x.strip()])!=4:
            raise RuntimeError('Four-line preregistration required: '+str(path))
        job={'name':name,'kind':kind,'preregistration':str(path),
             'preregistration_sha256':sha(path),**kwargs}
        jobs.append(job)
        return job
    runs={}
    for original in old['jobs']:
        name=original['name']; runs[name]=original['run_dir']
    runs.update(roi=str(ROOT.parent/'runs/034_roi_magnify_train768_seed44_20261103'),
                mil=str(ROOT.parent/'runs/035_mil_tiny768_seed44_20261103'))
    def train_resolution(index):
        original=copy.deepcopy(old['jobs'][index]); name=original.pop('name')
        prereg=Path(original.pop('preregistration')).name
        original.pop('preregistration_sha256')
        size=int(original['arguments'][original['arguments'].index('--image-size')+1])
        if name != 'res640' and '--resume' in original['arguments']:
            raise RuntimeError('New resolution arm must use its registered EMA warm-start')
        register(name,'train',prereg,**original,epochs=3,estimated_seconds={640:4200,768:6000,896:7800}[size],
            expected_configuration={**common,'output':runs[name],'image_size':size},
            expected_source_sha256=prior_sources if name=='res640' else current_sources,
            expected_warm_start_sha256='aa3c632e7764b57a522920369e16d9cb4768e43e008c0588439b4438aa88fd6f',
            console_log=str(QUEUE/'logs'/(name+'.log')))
    def train_extension(name):
        path=BASE/'configs'/(name+'_seed44.json')
        cfg=read(path)
        expected={**common,**cfg}
        for k in ('resume','init_student_ema','init_student_ema_sha256'):
            expected.pop(k,None)
        register(name,'train','roi_magnify_seed44.txt' if name=='roi' else 'mil_seed44.txt',
            run_dir=runs[name],script='exploration_train.py',config_path=str(path),
            arguments=['--config',str(path)],epochs=3,estimated_seconds=7200 if name=='roi' else 6300,
            expected_configuration=expected,expected_source_sha256=extension_sources,
            expected_warm_start_sha256=cfg['init_student_ema_sha256'],
            allow_existing_pause_resume=False,console_log=str(QUEUE/'logs'/(name+'.log')))
    def evaluate(name,role='last',retry=None):
        suffix='' if role=='last' else '_best'
        folder=BASE/'evaluations'/(name+suffix)
        job_name='eval_'+name+suffix+(f'_retry{retry}' if retry else '')
        prereg='eval_'+name+suffix+(f'_retry{retry}' if retry else '')+'.txt'
        register(job_name,'evaluate',prereg,
            run_dir=runs[name],checkpoint=str(Path(runs[name])/(role+'.pt')),checkpoint_role=role,
            output=str(folder),timing=True,train_gap_studies=1000,estimated_seconds=720,
            roi=name=='roi',mil=name=='mil',roi_version=gallery['method_version'] if name=='roi' else None)
        evaluations[name+suffix]=str(folder/'report.json')
        return folder
    def compare(name,members,prereg,timing=False,best=False):
        folder=BASE/'comparisons'/name
        inputs=[f'{n}={BASE/"evaluations"/(n+("_best" if best and n != "champion041" else ""))}' for n in members]
        register('compare_'+name,'compare',prereg,inputs=inputs,output=str(folder),
            bootstrap=2000,timing_fusions=timing,estimated_seconds=1200)
        comparisons[name]=str(folder/'comparison.json')
    def report(name):
        register(name,'report','morning_report.txt',estimated_seconds=20)
    anchor=ROOT/'artifacts/kd_v1/student_upper_bound_20261030/models/A/best_ema_fp32.pt'
    anchor_receipt=anchor.parent/'backup_receipt.json'
    def evaluate_anchor():
        folder=BASE/'evaluations/champion041'
        register('eval_champion041','evaluate','eval_champion041.txt',
            checkpoint=str(anchor),checkpoint_sha256=sha(anchor),output=str(folder),
            timing=True,train_gap_studies=1000,estimated_seconds=720,
            historical_checkpoint_evidence=str(anchor_receipt),
            historical_checkpoint_evidence_sha256=sha(anchor_receipt))
        evaluations['champion041']=str(folder/'report.json')
    for index,name in enumerate(('res640','res768','res896')):
        train_resolution(index)
        if name=='res640' and recovery and recovery.get('retry_job')=='eval_res640_retry1':
            evaluate(name,retry=1)
        else:
            evaluate(name)
    compare('resolution',['res640','res768','res896'],'compare_resolution.txt',timing=True)
    report('report_resolution')
    evaluate_anchor()
    train_extension('roi'); evaluate('roi')
    compare('roi',['res768','roi'],'compare_roi.txt',timing=True)
    report('report_roi')
    train_extension('mil'); evaluate('mil')
    compare('final',['res640','res768','res896','roi','mil','champion041'],'compare_final.txt')
    report('report_primary_final')
    for name in ('res640','res768','res896','roi','mil'):
        evaluate(name,'best')
    compare('resolution_best',['res640','res768','res896','roi','mil','champion041'],
            'compare_resolution_best.txt',timing=True,best=True)
    report('report_final')
    source_names=set(Path(p).name for p in extension_sources)
    source_names.update(('exploration_evaluate.py','run_exploration_queue.py','summarize_exploration.py',
        'inference_inputs.py','inference_reproducibility.py','prepare_exploration_queue.py'))
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'output':str(QUEUE),
        'status_file':str(BASE/'STATUS.json'),'report_file':str(BASE/'MORNING_REPORT.md'),
        'roi_gallery':str(roi_gallery),'roi_self_review':str(review),'roi_self_review_sha256':sha(review),
        'jobs':jobs,'evaluation_reports':evaluations,'comparison_reports':comparisons,
        'source_sha256':{n:sha(ROOT/n) for n in sorted(source_names)},
        'registered_input_sha256':{str(path):sha(path) for path in (
            BASE/'configs/roi_seed44.json',BASE/'configs/mil_seed44.json',
            roi_gallery/'metrics.json',review,anchor,anchor_receipt,
            *([recovery_path] if recovery else []))},
        'comparison':'Primary fixed E3 matched budget; secondary valid-selected best; all infer768+hflip2',
        'no_test':True,'no_submission':True,'no_declaration_changes':True,'no_csv':True,
        'latency_not_admission_gate':True,'recommendations_require_human_final_members':True,
        'prior_failed_attempt':str(archived_status),
        'prior_640_recovery':str(QUEUE/'permission_recovery.json'),
        'reviewed_source_equivalence':str(source_equivalence) if source_equivalence else None,
        'recovery':recovery}
    write(plan_path,plan)
    print(json.dumps({'plan':str(plan_path),'jobs':len(jobs),'names':[j['name'] for j in jobs],
                      'plan_sha256':sha(plan_path)},indent=2))

if __name__=='__main__':
    main()
