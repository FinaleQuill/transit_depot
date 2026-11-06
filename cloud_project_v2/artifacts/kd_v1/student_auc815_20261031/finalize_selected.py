from pathlib import Path
import sys
sys.path.insert(0,'/home/featurize/work/cloud_project_v2')
import json,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
from deploy_kd import sha256_file
from student_experiment_evaluation import write_new_json
from student_draft_pair import DraftStudentPair
from predict import predict_submission
from audit_student_submission import audit_submission,OFFICIAL_TEST_MANIFEST_SHA256
repo=Path('/home/featurize/work/cloud_project_v2');root=Path('/tmp/student_candidate_20261031')
art=repo/'artifacts/kd_v1/student_auc815_20261031'
summary=json.loads((art/'actual_pair_summary.json').read_text())
selected=summary['draft_target768']
assert abs(selected['macro_auc']-.8166770734158482)<1e-14 and abs(selected['mean_ms']-84.82014125668533)<1e-9
source=root/'pair_model_draft';delivery=repo.parent/'deliveries/044_tiny_base_class1B_draft768_20261031'
assert not delivery.exists()
model_bytes=sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
assert shutil.disk_usage(delivery.parent).free>model_bytes+80*1024**2
assert model_bytes<=500000000
delivery.mkdir()
selection={'selected_hong_kong':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),'candidate':'AB_B100_class1_JPEG_draft_target768','model_config_sha256':sha256_file(source/'config.json'),'selection_inputs':'Full original valid scores and original100+5 valid timing only','validation_selected_hyperparameters':True,'validation_fitted_class_weights':True,'test_used_for_selection':False,'new_training':False,'formal_T4_certification':False,'user_selected_current_candidate':True,'future_80ms_goal_retained':True,'further_optimization_stopped_by_user':True,**selected}
write_new_json(delivery/'SELECTION_BEFORE_TEST.json',selection)
shutil.copytree(source,delivery/'model')
for p in source.rglob('*'):
 if p.is_file():assert sha256_file(p)==sha256_file(delivery/'model'/p.relative_to(source))
start=time.perf_counter()
report=predict_submission(model_dir=delivery/'model',image_root=root/'data/cxr_first',manifest=repo/'data/clean_v1_reviewed/test_manifest.csv',output=delivery/'submission.csv',predictor_factory=DraftStudentPair)
assert report['member_count']==2 and report['data_rows']==14140
elapsed=time.perf_counter()-start
audit=audit_submission(manifest=repo/'data/clean_v1_reviewed/test_manifest.csv',image_predictions=delivery/'submission.image_predictions.csv',submission=delivery/'submission.csv',image_root=root/'data/cxr_first',check_image_sha=True,expected_manifest_sha256=OFFICIAL_TEST_MANIFEST_SHA256)
write_new_json(delivery/'independent_audit.json',audit)
write_new_json(delivery/'latency_2080ti.json',json.loads((art/'actual_pair_draft_target768/actual_report.json').read_text()))
write_new_json(art/'optimized_submission_complete.json',{'selected':selection,'submission':report,'prediction_wall_seconds':elapsed,'audit_pass':True,'persistent_model_directory':str(delivery/'model'),'completed_hong_kong':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()})
print(json.dumps({'delivery':str(delivery),'macro_auc':selected['macro_auc'],'mean_ms':selected['mean_ms'],'sha256':report['submission_sha256'],'prediction_wall_seconds':elapsed,'audit_pass':True},indent=2),flush=True)
