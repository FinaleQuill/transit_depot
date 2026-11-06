"""Read-only exact positive/negative pair decomposition of saved valid outputs.
No test access, GPU, model execution, or parameter fitting.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json
import pandas as pd
import numpy as np
ROOT=Path('/home/featurize/work')
OUT=ROOT/'reconstruction_runs_20261103/BREAKTHROUGH_20261105/reassessment'
manifest=ROOT/'cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv'
rows=pd.read_csv(manifest,dtype=str,keep_default_na=False)
studies=rows.drop_duplicates('study_key').sort_values('study_key')
keys=studies.study_key.to_numpy();patients=studies.subject_id.to_numpy()
y=np.array([json.loads(x) for x in studies.targets],dtype=np.uint8)
patient_counts=Counter(patients)
image_counts=rows.groupby('study_key').size().reindex(keys).to_numpy()
ap_only=rows.groupby('study_key').view_position.apply(lambda x: all(v=='AP' for v in x)).reindex(keys).to_numpy(bool)
paths={
'041':ROOT/'cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/reports/A/valid_hflip2/valid_predictions.csv',
'sprint':ROOT/'reconstruction_runs_20261103/SPRINT_20261104/final_comparison/candidate_predictions.csv',
'robust':ROOT/'reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/final_comparison/candidate_predictions.csv'}
P={}
for name,path in paths.items():
 d=pd.read_csv(path,dtype={'Subject_id':str,'Study_id':str})
 d['key']=d.Subject_id+'/'+d.Study_id
 assert len(d)==8590 and not d.duplicated(['key','Predict_class']).any()
 assert set(d.key)==set(keys)
 a=d.pivot(index='key',columns='Predict_class',values='Probability').reindex(keys)
 assert list(a.columns)==list(range(1,11)) and a.shape==(859,10) and not a.isna().any().any()
 P[name]=a.to_numpy(float)

def native(x):
 if isinstance(x,np.ndarray):return native(x.tolist())
 if isinstance(x,np.generic):return native(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [native(v) for v in x]
 return x

def metrics(mask):
 labels=y[mask]; pat=patients[mask]; n=len(pat)
 counts=Counter(pat)
 res={'studies':n,'patients':len(counts),'images':int(image_counts[mask].sum()),'positive_studies':labels.sum(0),
      'positive_patients':[len(set(pat[labels[:,c]==1])) for c in range(10)],'models':{},'pairs':[]}
 for c in range(10):
  pids=pat[labels[:,c]==1];nids=pat[labels[:,c]==0];same=pids[:,None]==nids[None,:]
  res['pairs'].append({'total':same.size,'within':int(same.sum()),'between':int((~same).sum()),'within_fraction':same.mean() if same.size else None,
   'patients_with_both_labels':len(set(pids)&set(nids))})
 for name,probs in P.items():
  probs=probs[mask]; values=[]
  for c in range(10):
   pos=labels[:,c]==1;neg=~pos;p=probs[pos,c,None];q=probs[neg,c][None,:]
   wins=(p>q).astype(float)+.5*(p==q)
   same=pat[pos,None]==pat[neg][None,:]
   w=np.array([1/counts[v] for v in pat]);pw=w[pos,None]*w[neg][None,:]
   within_pat=[]
   for patient in sorted(set(pat[pos])&set(pat[neg])):
    pm=pat[pos]==patient;nm=pat[neg]==patient
    within_pat.append(float(wins[np.ix_(pm,nm)].mean()))
   value={'overall':wins.mean() if wins.size else None,
          'within':wins[same].mean() if same.any() else None,
          'between':wins[~same].mean() if (~same).any() else None,
          'within_patient_equal':np.mean(within_pat) if within_pat else None,
          'overall_patient_equal':(wins*pw).sum()/pw.sum() if pw.sum() else None}
   if wins.size and same.any() and (~same).any():
    assert abs(value['overall']-(value['within']*same.mean()+value['between']*(1-same.mean())))<1e-12
   values.append(value)
  res['models'][name]={'per_class':values,'macro':{k:float(np.mean([v[k] for v in values])) if all(v[k] is not None for v in values) else None for k in values[0]}}
 for baseline in ['041','sprint']:
  res['robust_minus_'+baseline]={'per_class':[{k:float(b[k]-a[k]) if a[k] is not None and b[k] is not None else None for k in a} for a,b in zip(res['models'][baseline]['per_class'],res['models']['robust']['per_class'])],
    'macro':{k:float(res['models']['robust']['macro'][k]-v) if v is not None and res['models']['robust']['macro'][k] is not None else None for k,v in res['models'][baseline]['macro'].items()}}
 return native(res)

masks={'all':np.ones(len(keys),bool),
 'single_study_patients':np.array([patient_counts[p]==1 for p in patients]),
 'multi_study_patients':np.array([patient_counts[p]>1 for p in patients]),
 'patients_at_least_5_studies':np.array([patient_counts[p]>=5 for p in patients]),
 'single_image_studies':image_counts==1,'multi_image_studies':image_counts>1,
 'ap_only':ap_only,'not_ap_only':~ap_only}
res={'created_at':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest,*paths.values()]},
 'test_accessed':False,'model_executed':False,'parameters_fitted':False,'class_ids':list(range(1,11)),
 'definition':'AUC_c = mean over all positive-negative Study pairs of 1[p_pos>p_neg] + .5*1[equal]. Split pairs by identical official patient ID. Overall_c = f_within_c*AUC_within_c+(1-f_within_c)*AUC_between_c. Macro is unweighted average of all 10 class AUCs, only when each class is defined. Within_patient_equal averages per-patient AUC only among patients with both labels for that class; different support across classes. Between is between-patient ranking, not proof of static shortcut use.',
 'slices':{name:metrics(mask) for name,mask in masks.items()}}
# Within-patient per-class numerator/denominator contribution, exact LOO range.
# No probability fitting: leave each validation patient out of existing predictions.
loo=[]
for patient in sorted(set(patients)):
 sub=metrics(patients!=patient)
 loo.append({'removed_patient_hash':hashlib.sha256(patient.encode()).hexdigest()[:16],
             'removed_studies':int((patients==patient).sum()),
             'robust_minus_041':sub['robust_minus_041']['macro'],
             'robust_minus_sprint':sub['robust_minus_sprint']['macro']})
res['leave_one_patient_out']={'n':len(loo),'diagnostic_only_no_selection_correction':True,'ranges':{},'records':loo}
for baseline in ['041','sprint']:
 res['leave_one_patient_out']['ranges'][baseline]={}
 for metric in ['overall','within','between','within_patient_equal','overall_patient_equal']:
  vals=[r['robust_minus_'+baseline][metric] for r in loo if r['robust_minus_'+baseline][metric] is not None]
  res['leave_one_patient_out']['ranges'][baseline][metric]={'minimum':min(vals),'maximum':max(vals),'num_positive':sum(v>0 for v in vals),'n_evaluable':len(vals)}
(OUT/'patient_ranking_audit.json').write_text(json.dumps(native(res),indent=2))
print(json.dumps({'slice_summary':{k:{'n':v['studies'],'patients':v['patients'],'macros':{name:z['macro'] for name,z in v['models'].items()},'robust_minus_sprint':v['robust_minus_sprint']['macro']} for k,v in res['slices'].items()},'pair_counts':res['slices']['all']['pairs'],'loo_ranges':res['leave_one_patient_out']['ranges']},indent=2))
