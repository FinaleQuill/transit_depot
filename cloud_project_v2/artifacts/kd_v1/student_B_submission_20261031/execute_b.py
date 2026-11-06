from pathlib import Path
import sys
sys.path.insert(0,'/home/featurize/work/cloud_project_v2')
import csv, gc, hashlib, json, shutil, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from restore_student_export import restore_best_export
from submit_student_candidate import SubmissionStudentPredictor
from student_experiment_evaluation import evaluate_predictor, write_new_json
from predict import predict_submission
from deploy_kd import read_submission, sha256_file, read_manifest
from inference_reproducibility import configure_inference
from benchmark_local_resources import prepare_cuda, validate_inputs, validate_round_output
from cost import time_round, summarize_timings, timing_environment, compare_predictions, read_image_probabilities, compare_image_output
from audit_student_submission import audit_submission, OFFICIAL_TEST_MANIFEST_SHA256
root=Path('/tmp/student_candidate_20261031'); repo=Path('/home/featurize/work/cloud_project_v2')
evidence=repo/'artifacts/kd_v1/student_B_submission_20261031'
delivery=repo.parent/'deliveries/043_base768_B_submission_20261031'
manifest=repo/'data/clean_v1_reviewed/valid_manifest.csv';test_manifest=manifest.with_name('test_manifest.csv')
source=repo/'artifacts/kd_v1/student_upper_bound_20261030/models/B'
historical=repo/'artifacts/kd_v1/student_upper_bound_20261030/reports/B/valid_hflip2'
def save(name,value):write_new_json(evidence/name,value)
def now():return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
record={'started_hong_kong':now(),'fixed_model':'B E10 EMA, Base768 FP16 hflip2, original Study mean','test_used_for_selection':False,'new_training':False,'platform_upload':False}
print('STAGE restore_export',now(),flush=True)
restored=restore_best_export(snapshot=source/'best_ema_fp32.pt',receipt=source/'backup_receipt.json',output=root/'B_export',historical_weights_sha256='903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411')
save('B_export_restored.json',restored)
print('STAGE full_valid',now(),flush=True)
configure_inference();quota=prepare_cuda(torch,requested_thread_percentage=None,baseline_sm_count=68,memory_limit_gib=8.)
model=root/'B_export'; predictor=SubmissionStudentPredictor(model)
start=time.perf_counter();valid=evaluate_predictor(predictor,image_root=root/'data/cxr_first',manifest=manifest,output_dir=root/'B_valid')
record['valid_wall_seconds']=time.perf_counter()-start
old=read_submission(historical/'valid_predictions.csv');actual=read_submission(root/'B_valid/valid_predictions.csv')
comparison=compare_predictions(actual,old,atol=0,expected_keys=set(old))
old_images=read_image_probabilities(historical/'valid.image_predictions.csv');new_images=read_image_probabilities(root/'B_valid/valid.image_predictions.csv')
assert set(old_images)==set(new_images)
image_error=max(float(np.max(np.abs(new_images[k]-old_images[k]))) for k in old_images)
assert comparison['match'] and image_error==0, (comparison,image_error)
assert abs(valid['macro_auc']-0.8105695698808949)<1e-14
save('B_full_valid_reproduction.json',{'macro_auc':valid['macro_auc'],'historical_comparison':comparison,'all_1422_image_probabilities_exact':True,'max_image_probability_error':image_error,'valid_wall_seconds':record['valid_wall_seconds'],'quota':quota})
print('STAGE timing',now(),flush=True)
inputs=root/'timing_inputs'
shutil.copytree(repo.parent/'deliveries/034_dual768_proxy_timing_20261029/bundle/inputs',inputs)
selection=json.loads((inputs/'selection.portable.json').read_text());selection['manifest_path']=str(manifest)
write_new_json(inputs/'selection.relocated.json',selection)
measured,warmup,coverage=validate_inputs(inputs/'pictures_100',inputs/'pictures_5',inputs/'selection.relocated.json')
originals={(r['subject_id'],r['study_id'],Path(r['relative_path']).name):r['image_id'] for r in read_manifest(manifest,split='valid')}
latency=root/'B_timing';latency.mkdir()
for row in warmup:predictor.predict_image(row['path'])
predictor.synchronize()
durations=[];comparisons=[];image_checks=[];round_outputs=[]
for index in range(3):
 output=latency/f'round_{index+1}.csv'
 times,values=time_round(predictor,measured,output)
 comparisons.append(compare_predictions(values,actual,atol=0,expected_keys={(r['subject_id'],r['study_id']) for r in measured},allow_reference_superset=True))
 image_checks.append(compare_image_output(measured,output.with_suffix('.image_predictions.csv'),new_images,originals))
 round_outputs.append(validate_round_output(measured,output,values));durations.append(times)
 assert comparisons[-1]['match'] and image_checks[-1]['match'] and image_checks[-1]['max_absolute_error']==0
 print('TIMING_ROUND',index+1,float(np.mean(times)),flush=True)
rounds,per_image,mean=summarize_timings(durations)
report={'mean_ms':mean,'round_mean_ms':rounds.mean(1).tolist(),'per_image_round_ms':rounds.T.tolist(),'p95_ms':float(np.percentile(per_image,95)),'max_per_image_round_mean_ms':float(np.max(per_image)),'coverage':coverage,'comparisons_with_full_valid':comparisons,'image_comparisons_with_full_valid':image_checks,'round_outputs':round_outputs,'environment':timing_environment(predictor),'quota':quota,'T4_certified':False,'under_80ms':mean<=80,'input_storage':'local root disk; original image bytes','protocol':'100 original valid, 5 separate warmup, batch1 serial3 rounds; read_file_to_actual_CSV_write; initialization excluded; no concurrent extraction/install/GPU work'}
save('B_latency_2080ti.json',report)
# The copied report above retains all 100x3 timings; round CSVs are copied below.
print('STAGE test_submission',now(),flush=True)
del predictor;gc.collect();torch.cuda.empty_cache()
start=time.perf_counter()
submission=predict_submission(model_dir=model,image_root=root/'data/cxr_first',manifest=test_manifest,output=delivery/'submission.csv',predictor_factory=SubmissionStudentPredictor)
record['test_wall_seconds']=time.perf_counter()-start
print('STAGE independent_audit',now(),flush=True)
audit=audit_submission(manifest=test_manifest,image_predictions=delivery/'submission.image_predictions.csv',submission=delivery/'submission.csv',image_root=root/'data/cxr_first',check_image_sha=True,expected_manifest_sha256=OFFICIAL_TEST_MANIFEST_SHA256)
write_new_json(delivery/'independent_audit.json',audit)
write_new_json(delivery/'latency_2080ti.json',report)
write_new_json(delivery/'B_export_restoration.json',restored)
shutil.copytree(root/'B_valid',evidence/'full_valid_reproduced')
shutil.copytree(latency,evidence/'timing_round_outputs')
record.update(completed_hong_kong=now(),submission=submission,latency_mean_ms=mean,macro_auc=valid['macro_auc'],audit_pass=True,model_local_path=str(model),persistent_best_snapshot=str(source/'best_ema_fp32.pt'))
save('execution_complete.json',record)
print(json.dumps({'submission':str(delivery/'submission.csv'),'macro_auc':valid['macro_auc'],'mean_ms':mean,'test_wall_seconds':record['test_wall_seconds'],'audit_pass':True},indent=2),flush=True)
