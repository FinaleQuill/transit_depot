"""Saved-probability CPU-only patient paired analysis; no model import or fitting."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,time
import numpy as np
from sklearn.metrics import roc_auc_score

started=time.perf_counter()
root=Path('/home/featurize/work/cloud_project_v2')
student=root.parent/'deliveries/026_tiny512_bce_seed43/valid_fp16_hflip2'
teacher=root.parent/'runs/024_convnext_v1_base768_mixed20_12_cv5/teacher_epoch_logits_v2'
out=root/'artifacts/kd_v1/bce_fp16_review_20261027/comparison.json'
repeats,seed=1000,20261027
sreport=json.loads((student/'validation_report.json').read_text())
treport=json.loads((teacher/'teacher_epoch_diagnostic_report.json').read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(student/'valid_predictions.csv')==sreport['predictions_sha256']
manifest=root/'data/clean_v1_reviewed/valid_manifest.csv'
assert digest(manifest)==sreport['manifest_sha256']==treport['manifest_sha256']
labels={};counts=Counter()
for row in csv.DictReader(manifest.open()):
    key=f"{row['subject_id']}/{row['study_id']}"
    y=json.loads(row['targets'])
    assert len(y)==10 and all(v in (0,1) for v in y)
    assert key not in labels or labels[key]==y
    labels[key]=y;counts[key]+=1
keys=np.asarray(sorted(labels),dtype=str)
y=np.asarray([labels[k] for k in keys],dtype=np.int8)
subjects=np.asarray([k.split('/')[0] for k in keys],dtype=str)
values={};nrows=0
for row in csv.DictReader((student/'valid_predictions.csv').open()):
    key=f"{row['Subject_id']}/{row['Study_id']}";c=int(row['Predict_class'])-1
    assert key in labels and 0<=c<10
    values.setdefault(key,{})
    assert c not in values[key]
    value=float(row['Probability']);assert np.isfinite(value) and 0<=value<=1
    values[key][c]=value;nrows+=1
assert set(values)==set(labels) and nrows==len(keys)*10
models={'student_fp16_hflip2':np.asarray([[values[k][c] for c in range(10)] for k in keys])}
for policy,key in [('raw_probability','teacher_A_raw_hflip2'),('train_pos_weight_corrected_probability','teacher_A_corrected_hflip2')]:
    entry=next(r for r in treport['results'] if r['member']=='A_20_12_12_12_12' and r['tta']=='hflip2' and r['policy']==policy)
    path=teacher/entry['predictions']['npz'];assert digest(path)==entry['predictions']['npz_sha256']
    with np.load(path,allow_pickle=False) as data:
        loc={k:i for i,k in enumerate(data['study_keys'])}
        assert set(loc)==set(keys)
        order=[loc[k] for k in keys]
        assert np.array_equal(data['targets'][order],y)
        assert np.array_equal(data['subject_ids'][order],subjects)
        models[key]=data['probabilities'][order].copy()
# Exactly one declared equal-probability blend; no weight search.
models['diagnostic_equal_student_raw_teacher_blend']=(models['student_fp16_hflip2']+models['teacher_A_raw_hflip2'])/2
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
    assert np.allclose(auc,[roc_auc_score(y[:,c],p[:,c]) for c in range(10)],atol=1e-12,rtol=0)
    check=weighted_auc_all(p,w[:1])[0]
    for c in range(10):
        if np.unique(y[repeated,c]).size==2:assert abs(check[c]-roc_auc_score(y[repeated,c],p[repeated,c]))<1e-12
    point[name]={'macro_auc':float(auc.mean()),'per_class_auc':auc.tolist()}
    boot[name]=weighted_auc_all(p,w)
assert abs(point['student_fp16_hflip2']['macro_auc']-sreport['macro_auc'])<1e-12
pairs={
    'student_minus_raw_teacher':('teacher_A_raw_hflip2','student_fp16_hflip2'),
    'student_minus_corrected_teacher':('teacher_A_corrected_hflip2','student_fp16_hflip2'),
    'diagnostic_blend_minus_student':('student_fp16_hflip2','diagnostic_equal_student_raw_teacher_blend'),
}
comparisons={}
for name,(a,b) in pairs.items():
    d=boot[b]-boot[a];valid=np.isfinite(d).all(axis=1);macro=d[valid].mean(1)
    delta=np.asarray(point[b]['per_class_auc'])-point[a]['per_class_auc']
    comparisons[name]={'reference':a,'candidate':b,'macro_delta':point[b]['macro_auc']-point[a]['macro_auc'],
        'paired_patient_percentile_ci95':np.quantile(macro,[.025,.975]).tolist(),
        'positive_fraction_of_bootstrap_deltas':float((macro>0).mean()),'valid_resamples':int(valid.sum()),
        'per_class_delta':delta.tolist(),'per_class_percentile_ci95':np.quantile(d[valid],[.025,.975],axis=0).T.tolist(),
        'improved_classes':int((delta>0).sum()),'worsened_classes':int((delta<0).sum())}
views=np.asarray([counts[k] for k in keys])
slices={}
for name,mask in [('single_image_Study',views==1),('multi_image_Study',views>1)]:
    item={'studies':int(mask.sum()),'patients':len(np.unique(subjects[mask])),'positive_studies':y[mask].sum(0).tolist(),'points':{}}
    for model,p in models.items():
        auc=[float(roc_auc_score(y[mask,c],p[mask,c])) if np.unique(y[mask,c]).size==2 else None for c in range(10)]
        item['points'][model]={'macro_auc':float(np.mean([a for a in auc if a is not None])),'per_class_auc':auc}
    slices[name]=item
result={'checked_at_hong_kong':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'sources':{'student_prediction_sha256':digest(student/'valid_predictions.csv'),'student_report_sha256':digest(student/'validation_report.json'),'teacher_report_sha256':digest(teacher/'teacher_epoch_diagnostic_report.json'),'manifest_sha256':digest(manifest)},
    'seed':seed,'repeats':repeats,'sampling_unit':'patient cluster with replacement; all Studies and repeated patient multiplicities retained',
    'interval':'exploratory paired percentile; not corrected for checkpoint selection or multiple comparisons',
    'num_images':sum(counts.values()),'studies':len(keys),'patients':len(patients),'positive_studies':y.sum(0).tolist(),
    'positive_patients':[len(np.unique(subjects[y[:,c]==1])) for c in range(10)],
    'points':point,'comparisons':comparisons,'Study_view_slices':slices,
    'slice_caveat':'Patients can occur in both slices; case mix differs, so slice gaps do not establish resolution/blur causes.',
    'blend_caveat':'One predeclared 50/50 probability blend is solely an offline complementarity diagnostic. Five Base teachers plus Tiny exceed the500MB model-weight budget even inFP16; T4latencywasnotmeasured. No blend was exported or selected. Blend gain does not establish knowledge distillation gain.',
    'comparison_caveat':'Student FP16/batch1 uses fulltrain and valid-selectedE6; teacher BF16 diagnostics use five80percenttrain fixed-finalEMAs. Pointdifferences includeprecision/data/selectionconfounds and do not establish causal KD or resolution effects.',
    'test_used':False,'models_loaded':False,'gpu_used':False,'weights_fitted':False,'seconds':time.perf_counter()-started}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'seconds':result['seconds'],'points':{k:v['macro_auc'] for k,v in point.items()},'comparisons':{k:{j:v[j] for j in ['macro_delta','paired_patient_percentile_ci95','improved_classes','worsened_classes']} for k,v in comparisons.items()},'slices':{k:{'studies':v['studies'],'patients':v['patients'],'macro_auc':{a:b['macro_auc'] for a,b in v['points'].items()}} for k,v in slices.items()}},ensure_ascii=False,indent=2))
