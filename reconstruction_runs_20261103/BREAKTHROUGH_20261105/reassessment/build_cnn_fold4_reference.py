"""Reconstruct one immutable same-fold CNN benchmark from validated OOF cache."""
from pathlib import Path
import sys
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np

ROOT=Path('/home/featurize/work')
REPO=ROOT/'cxr_research_20261103'
sys.path.insert(0,str(REPO))
from dataset import read_manifest
from distill_tta_data import load_soft_labels
from study_shift_audit import read, SOURCES

OUT=ROOT/'reconstruction_runs_20261103/BREAKTHROUGH_20261105/reassessment'
CACHE=ROOT/'runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2'
FOLDS=ROOT/'cloud_project_v2/data/cv5_overnight/folds.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def auc(y,p,w=None):
    if len(np.unique(y))!=2:return None
    w=np.ones(len(y)) if w is None else w
    order=np.argsort(p,kind='stable');y=y[order];p=p[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(p))+1]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    return float(np.dot(pos,np.cumsum(neg)-neg+.5*neg)/(pos.sum()*neg.sum()))

def metrics(y,p,patients):
    counts=Counter(patients);w=np.array([1/counts[x] for x in patients])
    a=[auc(y[:,c],p[:,c]) for c in range(10)]
    b=[auc(y[:,c],p[:,c],w) for c in range(10)]
    return dict(studies=len(y),patients=len(counts),positive_studies=y.sum(0).astype(int).tolist(),
        positive_patients=[len(set(patients[y[:,c]==1])) for c in range(10)],
        per_class_auc=a,macro_auc=float(np.mean(a)) if all(x is not None for x in a) else None,
        patient_equal_per_class_auc=b,patient_equal_macro_auc=float(np.mean(b)) if all(x is not None for x in b) else None)

def main():
    # Tie-aware NumPy implementation is checked against independent pair enumeration.
    yt=np.array([0,1,1,0,1]);pt=np.array([.2,.2,.9,.9,.7])
    pairs=[(float(x>z)+.5*float(x==z)) for x in pt[yt==1] for z in pt[yt==0]]
    assert abs(auc(yt,pt)-np.mean(pairs))<1e-15
    records=read_manifest(SOURCES['official_train'],split='train')
    cache=load_soft_labels(CACHE,FOLDS,records)
    dev=read_manifest(SOURCES['dino_development'],split='train')
    plan=json.loads(FOLDS.read_text())
    teacher=[t for t in cache.metadata['teachers'] if t['fold_index']==4][0]
    assert set(r['subject_id'] for r in dev)==set(teacher['oof_subject_ids'])
    assert not set(r['subject_id'] for r in dev)&set(teacher['training_subject_ids'])
    assert set(r['image_id'] for r in dev)==set(teacher['oof_image_ids'])
    assert all(plan['study_to_fold'][r['study_key']]==4 for r in dev)
    assert cache.metadata['views'][0]=='identity'
    groups=defaultdict(list)
    for row in dev:groups[row['study_key']].append(row)
    keys,patients,targets,predictions,views=[],[],[],[],[]
    for key,rows in groups.items():
        idx=[cache.index[r['image_id']] for r in rows]
        raw=cache.logits[idx,0,:].astype(np.float64)
        image_probs=1./(1.+np.exp(-raw))
        keys.append(key);patients.append(rows[0]['subject_id'])
        targets.append(rows[0]['targets']);predictions.append(image_probs.mean(0))
        views.append([r['view_position'] for r in rows])
    y,p,patients=np.asarray(targets),np.asarray(predictions),np.asarray(patients)
    assert p.shape==(3702,10) and y.shape==p.shape
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    n=Counter(patients);count=np.array([n[x] for x in patients]);ap=np.array([set(v)=={'AP'} for v in views])
    masks=dict(all=np.ones(len(p),bool),single_image=np.array([len(v)==1 for v in views]),
        multi_image=np.array([len(v)>1 for v in views]),AP_only=ap,
        patient_1_study=count==1,patient_2to4_studies=(count>=2)&(count<=4),
        patient_5plus_studies=count>=5,patient_10plus_studies=count>=10,
        AP_only_and_5plus_studies=ap&(count>=5))
    reports={name:metrics(y[m],p[m],patients[m]) for name,m in masks.items()}
    destination=OUT/'cnn_fold4_reference.npz'
    np.savez_compressed(destination,probabilities=p,targets=y,study_keys=np.asarray(keys),
        patient_ids=patients,views_json=np.asarray([json.dumps(v) for v in views]))
    report=dict(schema='validated_oof_cnn_fold4_reference_v1',scope='existing train-only OOF baseline for future paired DINO evaluation',
        source_model='ConvNeXt V1 Base768 fold4 fixed epoch12 final EMA',
        evaluation='identity only; raw logits sigmoid then unweighted probability mean over all Study images',
        cache_integrity='distill_tta_data.load_soft_labels full validation passed',
        teacher_excluded_all_dev_patients=True,dev_subject_count=2211,dev_image_count=6064,
        no_gpu_used=True,test_data_opened=False,calibration_or_fusion_search=False,
        checkpoint_sha256_declared=teacher['checkpoint_sha256'],
        checkpoint_weights_rehashed_this_audit=False,
        source_sha256={str(q):sha(q) for q in (CACHE/'metadata.json',CACHE/'soft_labels.npz',FOLDS,SOURCES['official_train'],SOURCES['dino_development'])},
        reference_npz=str(destination),reference_npz_sha256=sha(destination),slices=reports,
        note='Reference extraction was BF16, current DINO FP16; resolution/loss/supervision differ. Comparison establishes a useful fixed baseline, not an architecture-only matched ablation.')
    (OUT/'cnn_fold4_reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('studies','patients','macro_auc','patient_equal_macro_auc')} for k,v in reports.items()},indent=2))

if __name__=='__main__':main()
