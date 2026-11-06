from pathlib import Path
import torch,json,hashlib,ast,numpy as np,shutil,datetime
R=Path('/home/featurize/work/cloud_project_v2');P=R/'artifacts/compliance_closeout_20261102/package';base=R/'artifacts/kd_v1/student_upper_bound_20261030/models/A';src=base/'metadata/evidence/source_202611030T052954Z_dc99f91e'
src=base/'metadata/evidence/source_20261030T052954Z_dc99f91e';export=src/'export_student.py';deploy=src/'deploy_kd.py'
expected='348901151c0ec1aa42b09015bf69c6be3b642173f58c9beb8b0161b271fbeb39'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=torch.load(base/'best_ema_fp32.pt',map_location='cpu',weights_only=True)
# These metadata fields were copied from original full checkpoint by the archived backup script.
# No model is built, no tensor is converted, no weights are exported.
model_config={**d['model_config'],'gradient_checkpointing':False}
common={'model_config':model_config,'image_size':d['image_size'],'training_mode':d.get('training_mode','distillation'),'source_checkpoint_schema_version':d['schema_version'],'tta':'hflip2','preprocessing':d['preprocessing'],'pretrained_provenance':d['pretrained_provenance'],'teacher_provenance':d['teacher_provenance'],'distillation':d.get('distillation',{}),'source_checkpoint_sha256':d['source_checkpoint_sha256'],'source_manifest_sha256':d['source_manifest_sha256'],'fold_plan_sha256':d['fold_plan_sha256'],'official_valid_manifest_sha256':d['official_valid_manifest_sha256'],'source_validation':d.get('validation',{}),'weights_source':'ema','seed':d.get('training_config',{}).get('seed')}
weights=R.parent/'deliveries/044_tiny_base_class1B_draft768_20261031/model/A/student_fp16.pt';common['weights_sha256']=sha(weights)
env={'np':np,'Any':object,'TINY_BACKBONE':'convnext_tiny.fb_in22k_ft_in1k_384','CLASS_IDS':list(range(1,11)),'validate_deployment_config':lambda config:None}
for path,name in [(deploy,'normalized_model_weights'),(export,'make_deployment_config')]:
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
config=env['make_deployment_config']([{**common,'weights_file':'student_fp16.pt'}],source_manifest_sha256=d['source_manifest_sha256'],fold_plan_sha256=d['fold_plan_sha256'],official_valid_manifest_sha256=d['official_valid_manifest_sha256'])
b=(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();actual=hashlib.sha256(b).hexdigest()
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_config_sha256':expected,'reconstructed_config_sha256':actual,'bytes':len(b),'exact_match':actual==expected,'snapshot_sha256':sha(base/'best_ema_fp32.pt'),'original_checkpoint_sha256':d['source_checkpoint_sha256'],'original_full_checkpoint_exists':(Path('/tmp/cloud_student_runs_20261030/full_checkpoint_backups/A/best.pt')).exists(),'source_export_script_sha256':sha(export),'source_deploy_script_sha256':sha(deploy),'existing_weights_sha256':common['weights_sha256'],'method':'CPU-only torch.load(weights_only=True) of retained EMA snapshot; archived export config-builder and normalized_model_weights functions extracted via AST; model validation call suppressed because no model/forward is needed for byte reconstruction; existing weights SHA used without exporting weights; original deterministic JSON serialization; write only if original receipt SHA fully matches.','model_forward_executed':False,'weights_exported':False}
if actual==expected:
 out=P/'evidence/weights/041_original_config.json';out.write_bytes(b)
 result['recovered_package_path']=out.relative_to(P).as_posix()
 for name,path in [('041_config_export_source.py',export),('041_config_deploy_source.py',deploy),('A_backup_receipt.json',base/'backup_receipt.json')]:shutil.copy2(path,P/'evidence/weights'/name)
 (P/'evidence/weights/041_config_recovery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
