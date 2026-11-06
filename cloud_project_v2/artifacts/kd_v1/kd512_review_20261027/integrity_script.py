"""Read-only completed KD027 audit; actual model weights are never forwarded."""
from pathlib import Path
import hashlib,json,re,math
from datetime import datetime,timezone,timedelta
import sys
sys.path.insert(0, '/home/featurize/work/cloud_project_v2')
import torch
from deploy_kd import validate_student_checkpoint
from kd_common import validate_pretrained_provenance,TINY_BACKBONE
ROOT=Path('/home/featurize/work/cloud_project_v2')
RUN=ROOT.parent/'runs/027_tiny512_kd_seed43'
OUT=ROOT/'artifacts/kd_v1/kd512_review_20261027'
errors=[]
def require(cond,msg):
 if not cond:errors.append(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
status=read(RUN/'status.json');config=read(RUN/'config.json');bestmetrics=read(RUN/'best_metrics.json')
history=[json.loads(l) for l in (RUN/'history.jsonl').read_text().splitlines()]
ledger=[json.loads(l) for l in (RUN/'checkpoint_audit.jsonl').read_text().splitlines()]
receipt=read(next((RUN/'evidence').glob('command*.json')))
archive=Path(receipt['source_archive']);sources=read(archive/'source_manifest.json')
require(receipt['exit_code']==0 and receipt['coverage_from_start']=='complete' and not receipt['preexisting_run_artifacts'],'native receipt completion')
require(status['state']=='early_stopped' and status['completed_epochs']==status['validated_epochs']==12,'completed and validated12')
require([r['epoch'] for r in history]==list(range(1,13)),'history epoch sequence')
previous=None
for r in ledger:
 payload={k:v for k,v in r.items() if k!='record_sha256'}
 require(r['previous_record_sha256']==previous and hashlib.sha256(canon(payload)).hexdigest()==r['record_sha256'],'ledger chain')
 previous=r['record_sha256']
 for p,digest in r['source_version']['files'].items():require(sha(archive/Path(p).name)==digest,'ledger archived source '+Path(p).name)
 require(hashlib.sha256(canon(r['source_version']['files'])).hexdigest()==r['source_version']['manifest_sha256'],'runtime source version manifest')
for item in sources['files']:
 require(sha(archive/item['path'])==item['sha256'],'source archive '+item['path'])
require((archive/'environment_lock.txt').stat().st_size>0,'environment lock')
environment=(archive/'environment_lock.txt').read_text()
require(sources['python_version'].startswith('3.12.'),'archived Python3.12')
require(all(v in environment for v in ['torch==2.10.0+cu128','torchvision==0.25.0+cu128','numpy==1.26.4','timm==1.0.30','albumentations==2.0.8']),'archived critical dependency versions')
plan_for_binding=read(ROOT/config['fold_plan'])
checkpoint_data_bindings={'fold_plan_sha256':sha(ROOT/config['fold_plan']),'source_manifest_sha256':sha(plan_for_binding['source_train_manifest']['path']),'official_valid_manifest_sha256':sha(plan_for_binding['official_holdout']['path'])}
last_events=[r for r in ledger if r['role']=='last'];best_events=[r for r in ledger if r['role']=='best']
require([r['epoch'] for r in last_events]==list(range(1,13)),'all12 last ledger records')
require([r['epoch'] for r in best_events]==list(range(1,8)),'all7 best ledger records')
require(all(r['historical_recovery'] is False and r['evidence_source']=='trainer' for r in ledger),'allnative checkpoint events')
log=Path(receipt['console_log']).read_text()
expectedmarks={1,*range(100,3792,100)};coverage={}
for ep in range(1,13):
 found=[int(v) for v in re.findall(rf'^student epoch={ep} batch=(\d+)/3791 ',log,re.M)]
 coverage[str(ep)]={'expected_progress_marks':len(expectedmarks),'found_progress_marks':len(found),'complete':set(found)==expectedmarks and len(found)==len(expectedmarks)}
 require(coverage[str(ep)]['complete'],'batch log coverage '+str(ep))
 require(len(re.findall(rf'^student epoch={ep} valid_AUC=',log,re.M))==1,'validation log summary '+str(ep))
row_by_ep={r['epoch']:r for r in history}
for r in history:
 ep=r['epoch'];train=r['training'];val=r['validation']
 require(train['images']==30328 and train['full_manifest_coverage'] is True,'full training image coverage')
 require(abs(sum(val['per_class_auc'])/10-val['macro_auc'])<1e-14,'macro/perclass AUC '+str(ep))
 require(val['num_studies']==859 and val['validation_tta']=='hflip2' and val['image_size']==512,'validation coverage/protocol')
 require(math.isfinite(train['hard_bce']) and math.isfinite(train['soft_kl_T2']) and train['soft_kl_T2']>=0,'finite components')
 for role in ['best','last']:
  key=role+'_checkpoint_sha256'
  hashes=[x['checkpoint_sha256'] for x in ledger if x['role']==role and x['epoch']<=ep]
  require(bool(hashes) and hashes[-1]==r[key],role+' history ledger SHA '+str(ep))
checkpoints={};pretrains={}
cache=Path(config['soft_labels']);cache=(ROOT/cache).resolve() if not cache.is_absolute() else cache
metadata=read(cache/'metadata.json');cache_hashes={'npz':sha(cache/'soft_labels.npz'),'metadata':sha(cache/'metadata.json')}
require(metadata['soft_labels_sha256']==cache_hashes['npz'],'cache byte SHA')
require(config['training_mode']=='distillation' and config['alpha']==.2 and config['temperature']==2 and config['soft_labels'] is not None,'KD configuration')
for role,expected_ep in [('best',7),('last',12)]:
 file=RUN/(role+'.pt');digest=sha(file)
 require(digest==status[role+'_checkpoint_sha256'],'actual '+role+' SHA')
 cp=torch.load(file,map_location='cpu',weights_only=True,mmap=True)
 validate_student_checkpoint(cp)
 require(all(cp[k]==v for k,v in checkpoint_data_bindings.items()),'actual checkpoint source train/valid/fold bindings '+role)
 require(cp['schema_version']==8 and cp['epoch']==expected_ep and cp['best_epoch']==7,'checkpoint identity '+role)
 require(cp['training_config']['training_mode']=='distillation' and cp['training_config']['seed']==43,'checkpoint seed/mode '+role)
 require(cp['teacher_provenance']==metadata['teachers'],'exact cache teacher ancestry '+role)
 require(cp['distillation']['soft_labels_sha256']==cache_hashes['npz'] and cp['distillation']['soft_metadata_sha256']==cache_hashes['metadata'],'checkpoint exact cache bindings '+role)
 require(cp['distillation']['alpha']==.2 and cp['distillation']['temperature']==2 and cp['distillation']['teacher_views']==['identity','hflip'],'exact distillation protocol '+role)
 require(cp['training_precision']=='bfloat16' and cp['weights_source']=='ema' and cp['validation_tta']=='hflip2','training precision/EMA/validation '+role)
 require(cp['validation']==row_by_ep[expected_ep]['validation'] or cp['validation']=={k:v for k,v in row_by_ep[expected_ep]['validation'].items()},'stored checkpoint metrics '+role)
 require(cp['best_auc']==status['best_auc'] and cp['progress']['amp_skipped_steps']==0,'checkpoint best/AMP '+role)
 require(cp['progress']['completed_epochs']==cp['progress']['validated_epochs']==expected_ep,'checkpoint epoch counters '+role)
 require(cp['progress']['backward_batches']==3791*expected_ep,'backward count '+role)
 require(cp['progress']['optimizer_updates']==cp['progress']['ema_updates']==948*expected_ep,'optimizer/EMA count '+role)
 require(len(cp['optimizer_state_dict']['state'])>0 and cp['rng_state'] and cp['raw_model_state_dict'],'full optimizer/raw/RNG continuation '+role)
 if role=='last':require(cp['early_stopping']['bad_epochs']==5 and cp['early_stopping']['reference_auc']==status['best_auc'],'terminal earlystop state')
 p=cp['pretrained_provenance'];validate_pretrained_provenance(p,TINY_BACKBONE)
 require(sha(p['weights_path'])==p['weights_sha256'],'original student pretrain SHA')
 material_checks={k:hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'] for k,v in p['materials'].items()}
 require(all(material_checks.values()),'student pretrained material hashes')
 pretrains[p['weights_sha256']]={'repo_id':p['repo_id'],'revision':p['revision'],'license':p['license'],'pretrain_dataset':p['pretrain_dataset'],'finetune_dataset':p['finetune_dataset'],'weight_sha_verified':True,'material_hash_checks':material_checks}
 checkpoints[role]={'epoch':cp['epoch'],'bytes':file.stat().st_size,'sha256':digest,'macro_auc':cp['validation']['macro_auc'],'progress':cp['progress'],'early_stopping':cp['early_stopping'],'native_protocol_validation_passed':True,'model_tensor_count':len(cp['model_state_dict']),'model_numel':sum(t.numel() for t in cp['model_state_dict'].values()),'floating_FP16_parameter_bytes_estimate':sum(t.numel()*2 for t in cp['model_state_dict'].values() if t.is_floating_point())}
 if role=='best':require(bestmetrics['epoch']==7 and bestmetrics['macro_auc']==cp['validation']['macro_auc'],'best metrics file')
 del cp
plan_path=ROOT/config['fold_plan'];plan=read(plan_path)
manifest_hashes={'fold_plan':sha(plan_path),'train':sha(plan['source_train_manifest']['path']),'valid':sha(plan['official_holdout']['path'])}
require(manifest_hashes['train']==metadata['source_manifest_sha256'] and manifest_hashes['fold_plan']==metadata['fold_plan_sha256'],'manifest/cache source binding')
teachers=[]
for t in metadata['teachers']:
 require(sha(t['checkpoint_path'])==t['checkpoint_sha256'],'teacher checkpoint SHA '+str(t['fold_index']))
 require(set(t['training_subject_ids']).isdisjoint(t['oof_subject_ids']),'teacher patient exclusion '+str(t['fold_index']))
 require(t['training_complete'] and t['weights_source']=='ema' and t['epoch'] in (12,20),'teacher final EMA '+str(t['fold_index']))
 require(t['official_valid_used_for_training'] is False and t['test_used_for_training'] is False and t['outer_fold_used_for_selection'] is False,'teacher split/selection truth '+str(t['fold_index']))
 p=t['pretrained_provenance']
 if p['weights_sha256'] not in pretrains:
  require(sha(p['weights_path'])==p['weights_sha256'],'original teacher pretrain SHA')
  mats={k:hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'] for k,v in p['materials'].items()};require(all(mats.values()),'teacher pretrained material SHA')
  pretrains[p['weights_sha256']]={'repo_id':p['repo_id'],'revision':p['revision'],'license':p['license'],'pretrain_dataset':p['pretrain_dataset'],'finetune_dataset':p['finetune_dataset'],'weight_sha_verified':True,'material_hash_checks':mats}
 teachers.append({'fold':t['fold_index'],'epoch':t['epoch'],'checkpoint_sha256':t['checkpoint_sha256'],'patient_exclusion_verified':True})
require([t['epoch'] for t in teachers]==[20,12,12,12,12],'frozen teacher budgets')
duration=(datetime.fromisoformat(receipt['ended_at'])-datetime.fromisoformat(receipt['started_at'])).total_seconds()
report={'checked_at_hong_kong':datetime.now(timezone(timedelta(hours=8))).isoformat(),'complete':not errors,'mismatches':errors,'receipt_exit_code':receipt['exit_code'],'duration_seconds':duration,'native_training_evidence_complete_for_this_student':not errors,'checkpoint_audit_records':len(ledger),'all12_epoch_console_coverage':coverage,'source_archive_files_verified':len(sources['files']),'runtime_source_authority':'Original execution archive and ledger; current trainer may be changed by the post-run status display fix.','runtime_source_pre_fix_check':'source_before_status_fix.json','checkpoints':checkpoints,'teacher_ancestry':teachers,'pretrains':pretrains,'cache_hashes':cache_hashes,'manifest_hashes':manifest_hashes,'checkpoint_train_valid_fold_bindings_independently_checked':checkpoint_data_bindings,'archived_expected_python_and_critical_dependency_versions_verified':not errors,'checkpoint_selection':'official valid earlystop by macroAUC, no officialvalid training/test use','fp16_export_exists':(ROOT.parent/'deliveries/027_tiny512_kd_seed43').exists(),'student_probability_comparison_available':False,'model_constructions':0,'model_forwards':0,'GPU_work':False,'runtime_sources_modified_by_audit':False,'historical_checkpoint_binary_retention':'Only current E7 best and E12 last verified; per-save ledger is metadata and hashes, not proof all old binary files survive.','final_submission_certified':False}
(OUT/'integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'complete':report['complete'],'mismatches':errors,'duration_seconds':duration,'best':checkpoints['best'],'last':checkpoints['last'],'ledger_records':len(ledger),'source_files':len(sources['files'])},ensure_ascii=False,indent=2))
assert not errors,errors
