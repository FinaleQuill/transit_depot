"""CPU-only paired patient-cluster bootstrap of predeclared teacher candidates."""
from pathlib import Path
import json, hashlib, time
from datetime import datetime, timezone, timedelta
import numpy as np
from sklearn.metrics import roc_auc_score

started=time.perf_counter()
root=Path('/home/featurize/work/cloud_project_v2')
source=root.parent/'runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_epoch_logits_v2'
report=json.loads((source/'teacher_epoch_diagnostic_report.json').read_text())
out=root/'artifacts/kd_v1/teacher_epoch_results_review_20261027/statistics.json'
seed=20261027
repeats=1000
models={}
for item in report['results']:
    if item['member'].startswith(('A_','B_')) or item['member'] in ['fold0_e13','fold0_e20']:
        if item['member'].startswith('fold0') and (item['policy']!='raw_probability' or item['tta']!='hflip2'):continue
        key=(item['member'][0] if item['member'].startswith(('A_','B_')) else item['member'])+'_'+('raw' if item['policy']=='raw_probability' else 'corrected')+'_'+item['tta']
        path=source/item['predictions']['npz']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['predictions']['npz_sha256']
        with np.load(path,allow_pickle=False) as data:
            row={k:data[k].copy() for k in data.files}
        models[key]=row
ref=next(iter(models.values()))
y=ref['targets'].astype(np.int8)
subjects=ref['subject_ids']
keys=ref['study_keys']
for row in models.values():
    for name in ['targets','subject_ids','study_keys','study_ids']:assert np.array_equal(row[name],ref[name]),name
patients,inverse=np.unique(subjects,return_inverse=True)
rng=np.random.default_rng(seed)
draws=rng.integers(len(patients),size=(repeats,len(patients)))
cluster_weights=np.zeros((repeats,len(patients)),dtype=np.float64)
for i,d in enumerate(draws):cluster_weights[i]=np.bincount(d,minlength=len(patients))
w=cluster_weights[:,inverse]

def weighted_auc_all(probabilities, weights):
    """AUC with repeated patient clusters represented by exact integer weights."""
    output=np.empty((len(weights),10),dtype=np.float64)
    for c in range(10):
        order=np.argsort(probabilities[:,c],kind='stable')
        ordered=probabilities[order,c]
        starts=np.r_[0,np.flatnonzero(np.diff(ordered)!=0)+1]
        positives=np.add.reduceat(weights[:,order]*y[order,c],starts,axis=1)
        negatives=np.add.reduceat(weights[:,order]*(1-y[order,c]),starts,axis=1)
        below=np.cumsum(negatives,axis=1)-negatives
        numerator=np.sum(positives*(below+0.5*negatives),axis=1)
        denominator=np.sum(positives,axis=1)*np.sum(negatives,axis=1)
        output[:,c]=np.divide(numerator,denominator,out=np.full(len(weights),np.nan),where=denominator>0)
    return output

point={}
boot={}
for name,row in models.items():
    p=row['probabilities']
    auc=weighted_auc_all(p,np.ones((1,len(y))))[0]
    assert np.allclose(auc,[roc_auc_score(y[:,c],p[:,c]) for c in range(10)],rtol=0,atol=1e-12)
    # Direct repeated-index comparison confirms multiplicity and tie handling.
    repeated=np.concatenate([np.flatnonzero(inverse==j) for j in draws[0]])
    check=weighted_auc_all(p,w[:1])[0]
    for c in range(10):
        if np.unique(y[repeated,c]).size==2:assert abs(check[c]-roc_auc_score(y[repeated,c],p[repeated,c]))<1e-12
    point[name]={'macro_auc':float(auc.mean()),'per_class_auc':auc.tolist()}
    boot[name]=weighted_auc_all(p,w)

comparisons={
    'E13_replacement_raw_hflip2':('A_raw_hflip2','B_raw_hflip2'),
    'E13_replacement_raw_identity':('A_raw_identity','B_raw_identity'),
    'E13_replacement_corrected_hflip2':('A_corrected_hflip2','B_corrected_hflip2'),
    'E13_replacement_corrected_identity':('A_corrected_identity','B_corrected_identity'),
    'prior_correction_A_hflip2':('A_raw_hflip2','A_corrected_hflip2'),
    'prior_correction_B_hflip2':('B_raw_hflip2','B_corrected_hflip2'),
    'prior_correction_A_identity':('A_raw_identity','A_corrected_identity'),
    'TTA_A_raw':('A_raw_identity','A_raw_hflip2'),
    'TTA_B_raw':('B_raw_identity','B_raw_hflip2'),
    'TTA_A_corrected':('A_corrected_identity','A_corrected_hflip2'),
    'single_fold0_E13_vs_E20_hflip2':('fold0_e20_raw_hflip2','fold0_e13_raw_hflip2'),
}
results={}
for name,(a,b) in comparisons.items():
    delta=boot[b]-boot[a]
    valid=np.isfinite(delta).all(axis=1)
    mean=delta[valid].mean(axis=1)
    class_delta=np.asarray(point[b]['per_class_auc'])-point[a]['per_class_auc']
    results[name]={'reference':a,'candidate':b,'macro_delta':point[b]['macro_auc']-point[a]['macro_auc'],
        'paired_patient_percentile_ci95':np.quantile(mean,[.025,.975]).tolist(),
        'positive_fraction_of_bootstrap_deltas':float((mean>0).mean()),'valid_resamples':int(valid.sum()),
        'per_class_delta':class_delta.tolist(),'per_class_percentile_ci95':np.quantile(delta[valid],[.025,.975],axis=0).T.tolist(),
        'improved_classes':int((class_delta>0).sum()),'worsened_classes':int((class_delta<0).sum())}

with np.load(source/'raw_logits/fold0_e20.npz',allow_pickle=False) as raw:
    rawkeys=raw['study_keys']
counts={key:int((rawkeys==key).sum()) for key in keys}
views=np.asarray([counts[key] for key in keys])
slices={}
for label,mask in [('single_image_Study',views==1),('multi_image_Study',views>1)]:
    entry={'studies':int(mask.sum()),'patients':len(np.unique(subjects[mask])),'per_class_positives':y[mask].sum(0).tolist(),'candidates':{}}
    for name in ['A_raw_hflip2','B_raw_hflip2','A_corrected_hflip2','B_corrected_hflip2']:
        p=models[name]['probabilities']
        auc=[float(roc_auc_score(y[mask,c],p[mask,c])) if np.unique(y[mask,c]).size==2 else None for c in range(10)]
        entry['candidates'][name]={'macro_auc':float(np.mean([a for a in auc if a is not None])),'per_class_auc':auc}
    slices[label]=entry
result={'checked_at_hong_kong':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'source_report_sha256':hashlib.sha256((source/'teacher_epoch_diagnostic_report.json').read_bytes()).hexdigest(),
    'seed':seed,'repeats':repeats,'sampling_unit':'patient cluster with replacement, all Studies and repeated multiplicities preserved',
    'interval':'paired percentile bootstrap, exploratory and not adjusted for selection/multiple comparisons',
    'studies':len(y),'patients':len(patients),'positive_studies':y.sum(0).tolist(),
    'positive_patients':[len(np.unique(subjects[y[:,c]==1])) for c in range(10)],
    'points':point,'comparisons':results,'Study_view_slices':slices,
    'test_used':False,'models_loaded':False,'gpu_used':False,'seconds':time.perf_counter()-started}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'seconds':result['seconds'],'comparisons':{k:{j:v[j] for j in ['macro_delta','paired_patient_percentile_ci95','improved_classes','worsened_classes']} for k,v in results.items()},'slices':slices},ensure_ascii=False,indent=2))
