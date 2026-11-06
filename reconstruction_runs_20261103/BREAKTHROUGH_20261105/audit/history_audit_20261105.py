import pandas as pd, numpy as np, json, hashlib
from pathlib import Path
base=Path('/home/featurize/work')
manifest=base/'cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv'
df=pd.read_csv(manifest,dtype=str).drop_duplicates('study_key').sort_values('study_key')
Y=np.array([json.loads(x) for x in df.targets],dtype=float)
ids=df.subject_id.to_numpy(); keys=df.study_key.to_numpy(); pats,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
paths={
'041':base/'cloud_project_v2/artifacts/kd_v1/student_upper_bound_20261030/reports/A/valid_hflip2/valid_predictions.csv',
'local_best_AF2_BF2':base/'cloud_project_v2/artifacts/kd_v1/frozen_head_20261101/actual/A_F2_B_F2/valid/valid_predictions.csv',
'sprint':base/'reconstruction_runs_20261103/SPRINT_20261104/final_comparison/candidate_predictions.csv',
'robust':base/'reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/final_comparison/candidate_predictions.csv'}
def read(p):
 d=pd.read_csv(p,dtype={'Subject_id':str,'Study_id':str});d['key']=d.Subject_id+'/'+d.Study_id
 pvt=d.pivot(index='key',columns='Predict_class',values='Probability').reindex(keys)
 assert len(pvt)==859 and pvt.shape[1]==10 and not pvt.isna().any().any()
 return pvt.to_numpy(float)
Ps={n:read(p) for n,p in paths.items()}
# Pairwise exact ties reusable positive/negative comparison matrices.
pairs={n:[(P[Y[:,c]==1,c,None]>P[Y[:,c]==0,c][None,:]).astype(float)+.5*(P[Y[:,c]==1,c,None]==P[Y[:,c]==0,c][None,:]) for c in range(10)] for n,P in Ps.items()}
def auc(n,w):
 out=[]
 for c,M in enumerate(pairs[n]):
  a=w[Y[:,c]==1];b=w[Y[:,c]==0];den=a.sum()*b.sum()
  out.append((a@M@b)/den if den else np.nan)
 return np.array(out)
ones=np.ones(len(Y));eq=1/counts[inv]
res={'source_sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()},'n_study':len(Y),'n_patient':len(pats),'positives_study':Y.sum(0).tolist(),'positives_patient':[len(set(ids[Y[:,c]==1])) for c in range(10)],'top10_patient_studies':int(np.sort(counts)[-10:].sum()),'metric':{n:{'auc':auc(n,ones).tolist(),'macro':float(auc(n,ones).mean()),'patient_equal_auc':auc(n,eq).tolist(),'patient_equal_macro':float(auc(n,eq).mean())} for n in Ps}}
a=auc('robust',ones)-auc('041',ones)
res['robust_minus_041']=a.tolist();res['class1_fraction_of_net_macro_gain']=float(a[0]/a.sum())
print(json.dumps(res,indent=2))
Path('/tmp/history_audit_20261105.json').write_text(json.dumps(res,indent=2))
orders={n:[np.argsort(P[:,c],kind='stable') for c in range(10)] for n,P in Ps.items()}
tiestarts={n:[np.r_[0,np.flatnonzero(np.diff(P[o,c]))+1] for c,o in enumerate(orders[n])] for n,P in Ps.items()}
def auc_sorted(n,w):
 out=[]
 for c,o in enumerate(orders[n]):
  yy=Y[o,c];ww=w[o];st=tiestarts[n][c]
  pos=np.add.reduceat(ww*yy,st);neg=np.add.reduceat(ww*(1-yy),st)
  out.append(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
 return np.array(out)
assert np.max(abs(auc_sorted('041',ones)-auc('041',ones))) < 1e-12
rng=np.random.default_rng(20261105);deltas=[]
for _ in range(2000):
 w=np.bincount(rng.integers(0,len(pats),size=len(pats)),minlength=len(pats))[inv]
 deltas.append(auc_sorted('robust',w)-auc_sorted('041',w))
deltas=np.array(deltas)
res['paired_robust_minus_041_bootstrap']={'n':2000,'seed':20261105,'class_ci':np.quantile(deltas,[.025,.975],axis=0).tolist(),'macro_ci':np.quantile(deltas.mean(1),[.025,.975]).tolist(),'note':'Exploratory; repeated model selection on the same official valid is uncorrected.'}
Path('/tmp/history_audit_20261105.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res['paired_robust_minus_041_bootstrap'],indent=2))
