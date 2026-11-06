"""Native4090 diagnostic decomposition; hooks are local, no source edits."""
from pathlib import Path
import csv
import json
import os
import statistics
import sys
import time
ROOT=Path('/home/featurize/work/cloud_project_v2')
sys.path.insert(0,str(ROOT))
import torch
import deploy_kd
from PIL import Image
from deploy_kd import Predictor,StudyAccumulator,discover_images,write_prediction,write_image_prediction

BASE=ROOT.parent/'deliveries/031_resource_stress_20261027'
OUT=BASE/'native_breakdown'
assert not OUT.exists()
assert not any(k.startswith('CUDA_MPS_') for k in os.environ)
OUT.mkdir()
models={
 'kd512':ROOT.parent/'deliveries/027_tiny512_kd_seed43/fp16_hflip2',
 'kd768':ROOT.parent/'deliveries/029_tiny768_kd_seed43/fp16_hflip2',
 'equal512_768':BASE/'models/kd512_kd768_equal_hflip2'}
records=discover_images(BASE/'inputs/pictures_100')
warmup=discover_images(BASE/'inputs/pictures_5')
metadata=[]
for r in records:
 with Image.open(r['path']) as image:
  metadata.append(dict(size=list(image.size),mode=image.mode,format=image.format,bytes=r['path'].stat().st_size))
torch.cuda.init()
p=torch.cuda.get_device_properties(0)
assert p.multi_processor_count==128
torch.cuda.set_per_process_memory_fraction(16*1024**3/p.total_memory)
all_results={}
for name,modeldir in models.items():
 predictor=Predictor(modeldir)
 for r in warmup:predictor.predict_image(r['path'])
 predictor.synchronize()
 original=deploy_kd.load_image_arrays
 preprocessing=[]
 events=[]
 def timed_arrays(*a,**kw):
  t=time.perf_counter();out=original(*a,**kw)
  preprocessing.append((time.perf_counter()-t)*1000)
  return out
 def begin(module,args):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  events.append((start,end));start.record()
 def end(module,args,result):
  events[-1][1].record()
 hooks=[h for m in predictor.models for h in [m.register_forward_pre_hook(begin),m.register_forward_hook(end)]]
 deploy_kd.load_image_arrays=timed_arrays
 accumulator=StudyAccumulator(records)
 rows=[]
 try:
  with (OUT/(name+'.csv')).open('x',newline='') as stream, (OUT/(name+'.images.csv')).open('x',newline='') as istream:
   writer=csv.writer(stream);iwriter=csv.writer(istream)
   writer.writerow(deploy_kd.CSV_HEADER);iwriter.writerow(deploy_kd.IMAGE_CSV_HEADER)
   stream.flush();istream.flush()
   for record in records:
    events.clear();preprocessing.clear();predictor.synchronize()
    t0=time.perf_counter();prob=predictor.predict_image(record['path']);t1=time.perf_counter()
    write_image_prediction(iwriter,record,prob);istream.flush()
    done=accumulator.add(record,prob)
    if done:write_prediction(writer,*done);stream.flush()
    predictor.synchronize();t2=time.perf_counter()
    rows.append(dict(image_id=record['image_id'],preprocessing_ms=sum(preprocessing),
     predict_image_ms=(t1-t0)*1000,aggregation_and_csv_ms=(t2-t1)*1000,total_ms=(t2-t0)*1000,
     model_forward_event_span_ms=sum(a.elapsed_time(b) for a,b in events)))
  assert accumulator.complete()
 finally:
  deploy_kd.load_image_arrays=original
  for h in hooks:h.remove()
 means={k:statistics.mean(row[k] for row in rows) for k in rows[0] if k!='image_id'}
 result=dict(physical_gpu=p.name,sm_count=p.multi_processor_count,model=name,num_images=len(rows),
  means=means,rows=rows,image_metadata=metadata,official_timing=False,T4_measurement=False,
  note='One diagnostic round with local hooks; CUDA event forward span includes host submission gaps and is not isolated kernel compute. Main three-round measurements remain authoritative for this hardware.')
 (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 all_results[name]=means
 print('BREAKDOWN',name,json.dumps(means),flush=True)
 del predictor
 torch.cuda.empty_cache()
(OUT/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
