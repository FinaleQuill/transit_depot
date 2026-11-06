"""Same timing wrapper without MPS, for a native4090 control."""
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT=Path('/home/featurize/work/cloud_project_v2')
BASE=ROOT.parent/'deliveries/031_resource_stress_20261027'
OUT=BASE/'native_controls'
PYTHON='/home/featurize/.conda/envs/cxr-kd-v1/bin/python'
MODELS={
 'kd512':ROOT.parent/'deliveries/027_tiny512_kd_seed43/fp16_hflip2',
 'kd768':ROOT.parent/'deliveries/029_tiny768_kd_seed43/fp16_hflip2',
 'equal512_768':BASE/'models/kd512_kd768_equal_hflip2'}
summary=json.loads((BASE/'results_v1/resource_stress_summary.json').read_text())
assert summary['complete'] and summary['controller_stop']['returncode']==0
assert not OUT.exists()
OUT.mkdir()
env={k:v for k,v in os.environ.items() if not k.startswith('CUDA_MPS_')}
env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',
           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
rows=[]
for name,model in MODELS.items():
 command=[PYTHON,'-u','benchmark_local_resources.py','--model-dir',str(model),
   '--pictures-100',str(BASE/'inputs/pictures_100'),'--pictures-5',str(BASE/'inputs/pictures_5'),
   '--selection-json',str(BASE/'inputs/selection.json'),'--output-dir',str(OUT/name),
   '--profile','native_'+name,'--memory-limit-gib','16','--execute']
 started=datetime.now(timezone.utc).isoformat()
 print('NATIVE_START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as stream:
  r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
 assert r.returncode==0,(name,r.returncode)
 report=json.loads((OUT/name/'local_resource_report.json').read_text())
 assert report['resources']['cuda_visible_sm_count']==128
 assert report['resources']['mode']=='native'
 assert report['comparison']['status']=='matched'
 rows.append(dict(candidate=name,mean_ms=report['mean_ms'],p95_ms=report['p95_ms'],
   max_ms=report['max_ms'],round_mean_ms=report['round_mean_ms'],report=str(OUT/name/'local_resource_report.json'),
   command=command,started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat()))
 print('NATIVE_DONE',name,report['mean_ms'],flush=True)
 (OUT/'native_summary.json').write_text(json.dumps(dict(complete=len(rows)==3,cases=rows,
   true_T4=False,CUDA_MPS_environment_removed=True),indent=2)+'\n')
