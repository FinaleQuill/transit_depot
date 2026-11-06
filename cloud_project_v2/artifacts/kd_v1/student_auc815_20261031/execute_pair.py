from pathlib import Path
import sys
sys.path.insert(0,'/home/featurize/work/cloud_project_v2')
import csv,gc,json,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from restore_student_export import restore_best_export
from student_fixed_pair import FixedStudentPair,make_pair
from student_draft_pair import DraftStudentPair
from student_experiment_evaluation import evaluate_predictor,write_new_json
from inference_reproducibility import configure_inference
from benchmark_local_resources import prepare_cuda,validate_inputs,validate_round_output
from deploy_kd import read_manifest,read_submission,sha256_file
from cost import time_round,summarize_timings,timing_environment,compare_predictions,read_image_probabilities,compare_image_output
root=Path('/tmp/student_candidate_20261031');repo=Path('/home/featurize/work/cloud_project_v2');art=repo/'artifacts/kd_v1/student_auc815_20261031'
manifest=repo/'data/clean_v1_reviewed/valid_manifest.csv';plan=art/'fifth_plan.json'
source=repo/'artifacts/kd_v1/student_upper_bound_20261030/models/A'
restore=restore_best_export(snapshot=source/'best_ema_fp32.pt',receipt=source/'backup_receipt.json',output=root/'A_export',historical_weights_sha256='fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f')
write_new_json(art/'A_restoration_for_pair.json',restore)
make_pair([root/'A_export',root/'B_export'],root/'pair_model',plan=plan,candidate='AB_B100_class1',source_receipt=art/'execution_inputs.json')
quota=prepare_cuda(torch,requested_thread_percentage=None,baseline_sm_count=68,memory_limit_gib=8.)
inputs=root/'timing_inputs'
measured,warmup,coverage=validate_inputs(inputs/'pictures_100',inputs/'pictures_5',inputs/'selection.relocated.json')
originals={(r['subject_id'],r['study_id'],Path(r['relative_path']).name):r['image_id'] for r in read_manifest(manifest,split='valid')}
summary={}
for name,loader in [('full_decode',FixedStudentPair),('draft_target768',DraftStudentPair)]:
 configure_inference()
 model=root/'pair_model'
 if name=='draft_target768':
  model=root/'pair_model_draft'
  model.mkdir()
  for label in ('A','B'):
   shutil.copytree(root/'pair_model'/label,model/label,copy_function=lambda s,d:shutil.copyfile(s,d))
  config=json.loads((root/'pair_model/config.json').read_text())
  config['decoder_candidate']='JPEG_draft_target768';config['decoder_plan_sha256']=sha256_file(art/'seventh_decoder_plan.json')
  config['preprocessing_changes_pixels']=True
  write_new_json(model/'config.json',config)
 output=root/f'pair_{name}';output.mkdir()
 print('STAGE',name,'full_valid',datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),flush=True)
 predictor=loader(model)
 start=time.perf_counter()
 valid=evaluate_predictor(predictor,image_root=root/'data/cxr_first',manifest=manifest,output_dir=output/'valid')
 elapsed=time.perf_counter()-start
 if name=='full_decode':
  expected=json.loads((art/'fifth_class1_blend_report.json').read_text())
  # Binding to the pre-registered result, not selecting from this live run.
  assert abs(valid['macro_auc']-.8168300303208815)<1e-14,valid['macro_auc']
  assert np.allclose(valid['per_class_auc'],expected['scores']['AB_B100_class1']['per_class_auc'],rtol=0,atol=1e-15)
  base=repo/'artifacts/kd_v1/student_upper_bound_20261030/reports'
  reference_a=read_image_probabilities(base/'A/valid_hflip2/valid.image_predictions.csv')
  reference_b=read_image_probabilities(base/'B/valid_hflip2/valid.image_predictions.csv')
  actual_images=read_image_probabilities(output/'valid/valid.image_predictions.csv')
  wa=np.array([0.]+[.5]*9);wb=1-wa
  image_error=max(float(np.max(np.abs(actual_images[k]-(reference_a[k]*wa+reference_b[k]*wb)))) for k in reference_a)
  assert image_error<2e-15,image_error
  write_new_json(output/'cached_prediction_reproduction.json',{'max_image_probability_error':image_error,'all_class_auc_match_registered_result':True,'new_model_weights':False,'test_used':False})
 print('ACTUAL_VALID',name,valid['macro_auc'],flush=True)
 timing=output/'timing';timing.mkdir()
 values=read_submission(output/'valid/valid_predictions.csv');images=read_image_probabilities(output/'valid/valid.image_predictions.csv')
 for row in warmup:predictor.predict_image(row['path'])
 predictor.synchronize()
 durations=[];comparisons=[];imagechecks=[]
 for index in range(3):
  target=timing/f'round_{index+1}.csv'
  times,predictions=time_round(predictor,measured,target)
  comparison=compare_predictions(predictions,values,atol=0,expected_keys={(r['subject_id'],r['study_id']) for r in measured},allow_reference_superset=True)
  imagecheck=compare_image_output(measured,target.with_suffix('.image_predictions.csv'),images,originals)
  assert comparison['match'] and imagecheck['match'] and imagecheck['max_absolute_error']==0
  validate_round_output(measured,target,predictions)
  durations.append(times);comparisons.append(comparison);imagechecks.append(imagecheck)
  print('ROUND_MS',name,index+1,float(np.mean(times)),flush=True)
 rounds,per_image,mean=summarize_timings(durations)
 assets=sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
 report={'candidate':name,'macro_auc':valid['macro_auc'],'mean_ms':mean,'round_mean_ms':rounds.mean(1).tolist(),'per_image_round_ms':rounds.T.tolist(),'p95_ms':float(np.percentile(per_image,95)),'valid_wall_seconds':elapsed,'comparisons_with_full_valid':comparisons,'image_comparisons_with_full_valid':imagechecks,'coverage':coverage,'environment':timing_environment(predictor),'asset_bytes':assets,'size_pass':assets<=500000000,'test_used':False,'new_training':False,'validation_selected_hyperparameters':True,'validation_fitted_class_weights':True,'meets_requested_auc_and_latency':valid['macro_auc']>=.815 and mean<=80,'T4_certified':False,'protocol':'original valid100+5, batch1 serial3, read_file_to_actual_CSV_write; local disk; no concurrent heavy tasks'}
 write_new_json(output/'actual_report.json',report)
 persistent=art/f'actual_pair_{name}';shutil.copytree(output,persistent)
 summary[name]={k:report[k] for k in ['macro_auc','mean_ms','round_mean_ms','meets_requested_auc_and_latency','asset_bytes']}
 del predictor;gc.collect();torch.cuda.empty_cache()
write_new_json(art/'actual_pair_summary.json',summary)
print(json.dumps(summary,indent=2),flush=True)
