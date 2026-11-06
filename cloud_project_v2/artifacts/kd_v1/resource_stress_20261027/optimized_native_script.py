"""Retiming after bit-exact preprocessing; E8/E13 pair is a timing proxy only."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path('/home/featurize/work/cloud_project_v2')
sys.path.insert(0,str(ROOT))
from deploy_kd import read_submission,sha256_file
import numpy as np
BASE=ROOT.parent/'deliveries/031_resource_stress_20261027'
OUT=BASE/'optimized_native_controls'
assert not OUT.exists();OUT.mkdir()
PYTHON='/home/featurize/.conda/envs/cxr-kd-v1/bin/python'
models={
 'kd512':ROOT.parent/'deliveries/027_tiny512_kd_seed43/fp16_hflip2',
 'kd768':ROOT.parent/'deliveries/029_tiny768_kd_seed43/fp16_hflip2',
 'equal512_768':BASE/'models/kd512_kd768_equal_hflip2',
 'two768_timing_only':BASE/'models/two768_E8_E13_timing_only'}
env={k:v for k,v in os.environ.items() if not k.startswith('CUDA_MPS_')}
env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',
           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
source_sha=sha256_file(ROOT/'deploy_kd.py')
rows=[]
for name,model in models.items():
 command=[PYTHON,'-u','benchmark_local_resources.py','--model-dir',str(model),
   '--pictures-100',str(BASE/'inputs/pictures_100'),'--pictures-5',str(BASE/'inputs/pictures_5'),
   '--selection-json',str(BASE/'inputs/selection.json'),'--output-dir',str(OUT/name),
   '--profile','optimized_native_'+name,'--memory-limit-gib','16','--execute']
 print('OPTIMIZED_START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as stream:
  done=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
 assert done.returncode==0,(name,done.returncode)
 report=json.loads((OUT/name/'local_resource_report.json').read_text())
 assert report['resources']['mode']=='native' and report['resources']['cuda_visible_sm_count']==128
 assert report['comparison']['status']=='matched'
 row=dict(candidate=name,mean_ms=report['mean_ms'],p95_ms=report['p95_ms'],max_ms=report['max_ms'],
   round_mean_ms=report['round_mean_ms'],report=str(OUT/name/'local_resource_report.json'),
   report_sha256=sha256_file(OUT/name/'local_resource_report.json'),command=command)
 prior=BASE/'native_controls'/name/'round_1.csv'
 if prior.exists():
  old,new=read_submission(prior),read_submission(OUT/name/'round_1.csv')
  assert old.keys()==new.keys()
  row['max_probability_difference_from_original']=max(float(np.abs(old[k]-new[k]).max()) for k in old)
  assert row['max_probability_difference_from_original']==0,row
 else:
  row['accuracy_claim']='none; E8/E13 checkpoints from one run only proxy two768 forward cost'
 rows.append(row)
 print('OPTIMIZED_DONE',name,report['mean_ms'],flush=True)
 assert sha256_file(ROOT/'deploy_kd.py')==source_sha
 (OUT/'optimized_native_summary.json').write_text(json.dumps(dict(complete=len(rows)==4,cases=rows,
   deploy_source_sha256=source_sha,CLAHE_enabled=False,DALI_used=False,true_T4=False,
   per_image_shared_preprocessing=True,per_image_shared_TTA_inputs=True,training_started=False),indent=2)+'\n')
