from pathlib import Path
import sys
sys.path.insert(0,'/home/featurize/work/cloud_project_v2')
import gc,json,shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from student_draft_pair import DraftStudentPair
from student_experiment_evaluation import evaluate_predictor,write_new_json
from inference_reproducibility import configure_inference
from benchmark_local_resources import prepare_cuda,validate_inputs,validate_round_output
from deploy_kd import read_manifest,read_submission,sha256_file
from cost import time_round,summarize_timings,timing_environment,compare_predictions,read_image_probabilities,compare_image_output
root=Path('/tmp/student_candidate_20261031');repo=Path('/home/featurize/work/cloud_project_v2');art=repo/'artifacts/kd_v1/student_auc815_20261031'
manifest=repo/'data/clean_v1_reviewed/valid_manifest.csv'
model=root/'pair_model_mixed_views';model.mkdir()
for label in ('A','B'):shutil.copytree(root/'pair_model_draft'/label,model/label)
config=json.loads((root/'pair_model_draft/config.json').read_text());config['view_candidate']='A_identity_B_hflip2';config['view_plan_sha256']=sha256_file(art/'eighth_view_plan.json')
write_new_json(model/'config.json',config)
quota=prepare_cuda(torch,requested_thread_percentage=None,baseline_sm_count=68,memory_limit_gib=8.)
inputs=root/'timing_inputs'
measured,warmup,coverage=validate_inputs(inputs/'pictures_100',inputs/'pictures_5',inputs/'selection.relocated.json')
originals={(r['subject_id'],r['study_id'],Path(r['relative_path']).name):r['image_id'] for r in read_manifest(manifest,split='valid')}
summary={}
for name,loader in [('mixed_views',DraftStudentPair)]:
 configure_inference()
 output=root/f'pair_{name}';output.mkdir()
 print('STAGE',name,'full_valid',datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),flush=True)
 predictor=loader(model)
 start=time.perf_counter()
 valid=evaluate_predictor(predictor,image_root=root/'data/cxr_first',manifest=manifest,output_dir=output/'valid')
 elapsed=time.perf_counter()-start
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
write_new_json(art/'mixed_view_summary.json',summary)
print(json.dumps(summary,indent=2),flush=True)
