"""CPU-only paired actual FP16 BCE/KD comparison; one predeclared equal blend."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,time
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
started=time.perf_counter()
root=Path('/home/featurize/work/cloud_project_v2')
out=root/'artifacts/kd_v1/kd512_fp16_review_20261027'
manifest=root/'data/clean_v1_reviewed/valid_manifest.csv'
folders={'BCE_fp16_hflip2':root.parent/'deliveries/026_tiny512_bce_seed43/valid_fp16_hflip2',
         'KD_fp16_hflip2':root.parent/'deliveries/027_tiny512_kd_seed43/valid_fp16_hflip2'}
repeats,seed=1000,20261027
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
labels={};counts=Counter()
with manifest.open(encoding='utf-8-sig',newline='') as stream:
 for row in csv.DictReader(stream):
  assert row['split']=='valid'
  key=f"{row['subject_id']}/{row['study_id']}";target=json.loads(row['targets'])
  assert len(target)==10 and all(v in (0,1) for v in target)
  assert key not in labels or labels[key]==target
  labels[key]=target;counts[key]+=1
keys=np.asarray(sorted(labels),dtype=str)
y=np.asarray([labels[k] for k in keys],dtype=np.int8)
subjects=np.asarray([k.split('/')[0] for k in keys],dtype=str)
assert len(keys)==859 and len(np.unique(subjects))==228 and sum(counts.values())==1422
models={};reports={};sources={'manifest_sha256':digest(manifest)}
for name,folder in folders.items():
 report=json.loads((folder/'validation_report.json').read_text())
 assert report['complete'] and report['precision']=='float16' and report['tta']=='hflip2'
 assert report['image_size']==512 and report['member_count']==1 and report['batch_size']==1
 assert report['manifest_sha256']==sources['manifest_sha256']
 prediction=folder/'valid_predictions.csv';assert digest(prediction)==report['predictions_sha256']
 values={};nrows=0
 with prediction.open(encoding='utf-8-sig',newline='') as stream:
  for row in csv.DictReader(stream):
   key=f"{row['Subject_id']}/{row['Study_id']}";c=int(row['Predict_class'])-1
   assert key in labels and 0<=c<10
   values.setdefault(key,{});assert c not in values[key]
   probability=float(row['Probability']);assert np.isfinite(probability) and 0<=probability<=1
   values[key][c]=probability;nrows+=1
 assert set(values)==set(labels) and nrows==8590 and all(set(v)==set(range(10)) for v in values.values())
 models[name]=np.asarray([[values[k][c] for c in range(10)] for k in keys],dtype=np.float64)
 reports[name]=report
 sources[name]={'predictions_sha256':digest(prediction),'report_sha256':digest(folder/'validation_report.json'),
                'deployment_config_sha256':report['deployment_config_sha256'],'member_source_sha256':report['members'][0]['source_checkpoint_sha256']}
# Exactly one predeclared fixed mixture. No weight search or fitting.
models['equal_BCE_KD_blend']=(models['BCE_fp16_hflip2']+models['KD_fp16_hflip2'])/2
patients,inverse=np.unique(subjects,return_inverse=True)
rng=np.random.default_rng(seed)
draws=rng.integers(len(patients),size=(repeats,len(patients)))
cw=np.asarray([np.bincount(d,minlength=len(patients)) for d in draws],dtype=np.float64)
w=cw[:,inverse]
def weighted_auc_all(probabilities,weights):
 output=np.empty((len(weights),10),dtype=np.float64)
 for c in range(10):
  order=np.argsort(probabilities[:,c],kind='stable');ordered=probabilities[order,c]
  starts=np.r_[0,np.flatnonzero(np.diff(ordered)!=0)+1]
  positive=np.add.reduceat(weights[:,order]*y[order,c],starts,axis=1)
  negative=np.add.reduceat(weights[:,order]*(1-y[order,c]),starts,axis=1)
  below=np.cumsum(negative,axis=1)-negative
  numerator=np.sum(positive*(below+0.5*negative),axis=1)
  denominator=positive.sum(1)*negative.sum(1)
  output[:,c]=np.divide(numerator,denominator,out=np.full(len(weights),np.nan),where=denominator>0)
 return output
point={};boot={}
repeated=np.concatenate([np.flatnonzero(inverse==j) for j in draws[0]])
for name,p in models.items():
 auc=weighted_auc_all(p,np.ones((1,len(y))))[0]
 assert np.allclose(auc,[roc_auc_score(y[:,c],p[:,c]) for c in range(10)],rtol=0,atol=1e-12)
 repeated_auc=weighted_auc_all(p,w[:1])[0]
 for c in range(10):
  if np.unique(y[repeated,c]).size==2:assert abs(repeated_auc[c]-roc_auc_score(y[repeated,c],p[repeated,c]))<1e-12
 point[name]={'macro_auc':float(auc.mean()),'per_class_auc':auc.tolist(),
              'macro_f1_fixed_0_5':float(f1_score(y,p>=.5,average='macro',zero_division=0))}
 if name in reports:
  assert abs(point[name]['macro_auc']-reports[name]['macro_auc'])<1e-12
  assert abs(point[name]['macro_f1_fixed_0_5']-reports[name]['macro_f1_fixed_0_5'])<1e-12
 boot[name]=weighted_auc_all(p,w)
pairs={'KD_minus_BCE':('BCE_fp16_hflip2','KD_fp16_hflip2'),
       'equal_blend_minus_KD':('KD_fp16_hflip2','equal_BCE_KD_blend'),
       'equal_blend_minus_BCE':('BCE_fp16_hflip2','equal_BCE_KD_blend')}
comparisons={}
for name,(a,b) in pairs.items():
 delta=boot[b]-boot[a];valid=np.isfinite(delta).all(axis=1);macro=delta[valid].mean(1)
 perclass=np.asarray(point[b]['per_class_auc'])-point[a]['per_class_auc']
 comparisons[name]={'reference':a,'candidate':b,'macro_delta':point[b]['macro_auc']-point[a]['macro_auc'],
                    'paired_patient_percentile_ci95':np.quantile(macro,[.025,.975]).tolist(),
                    'positive_fraction_of_bootstrap_deltas':float((macro>0).mean()),'valid_resamples':int(valid.sum()),
                    'per_class_delta':perclass.tolist(),'per_class_percentile_ci95':np.quantile(delta[valid],[.025,.975],axis=0).T.tolist(),
                    'improved_classes':int((perclass>0).sum()),'worsened_classes':int((perclass<0).sum())}
views=np.asarray([counts[k] for k in keys]);slices={}
for name,mask in [('single_image_Study',views==1),('multi_image_Study',views>1)]:
 item={'studies':int(mask.sum()),'patients':len(np.unique(subjects[mask])),'positive_studies':y[mask].sum(0).tolist(),'points':{}}
 for model,p in models.items():
  auc=[float(roc_auc_score(y[mask,c],p[mask,c])) if np.unique(y[mask,c]).size==2 else None for c in range(10)]
  item['points'][model]={'macro_auc':float(np.mean([v for v in auc if v is not None])),'per_class_auc':auc}
 slices[name]=item
result={'checked_at_hong_kong':datetime.now(timezone(timedelta(hours=8))).isoformat(),'sources':sources,
        'seed':seed,'repeats':repeats,'sampling_unit':'patient clusters with replacement; all Studies and repeated multiplicities preserved',
        'interval':'exploratory paired percentile intervals, not adjusted for checkpoint selection or multiple comparisons',
        'studies':len(keys),'patients':len(patients),'num_images':sum(counts.values()),'positive_studies':y.sum(0).tolist(),
        'positive_patients':[len(np.unique(subjects[y[:,c]==1])) for c in range(10)],
        'points':point,'comparisons':comparisons,'Study_view_slices':slices,
        'slice_caveat':'Patients overlap between single/multiple-image slices and case mix differs; this does not establish resolution or blur causation.',
        'comparison_caveat':'BCE and KD both FP16/batch1/hflip2 with same fulltrain/seed43 and matched configuration; bestepochs6and7 were selected on this samevalid. Bootstrap intervals do not remove selection bias.',
        'blend_caveat':'Exactly one predeclared50/50 saved-Study-probability diagnostic, not an actual composed deployment validation. TwoTiny architecture is supported but actual bytes and T4 latency remain to be checked on that frozen composite.',
        'test_used':False,'models_loaded':False,'gpu_used':False,'weights_fitted':False,'seconds':time.perf_counter()-started}
(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'seconds':result['seconds'],'points':{k:v['macro_auc'] for k,v in point.items()},
 'comparisons':{k:{q:v[q] for q in ['macro_delta','paired_patient_percentile_ci95','improved_classes','worsened_classes']} for k,v in comparisons.items()},
 'slices':{k:{'studies':v['studies'],'patients':v['patients'],'macro_auc':{a:b['macro_auc'] for a,b in v['points'].items()}} for k,v in slices.items()}},ensure_ascii=False,indent=2))
