"""CPU-only development comparison against a patient-excluded CNN reference.

This monitor never reads test, loads model weights, updates training parameters,
selects an alternative checkpoint, generates submission predictions or stops a
trainer. It consumes only fully completed epoch predictions and adds diagnostic
evidence. The historical fold4 confirmation role has been explicitly retired.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from reconstruction.evaluate import _RankedClass, evaluate_studies, json_native

def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_native(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def read_prediction(path):
    with np.load(path,allow_pickle=False) as data:
        required={'probabilities','targets','study_keys','patient_ids','views_json'}
        if not required <= set(data.files):
            raise ValueError('Missing prediction identity fields')
        value={k:data[k] for k in required}
    n=len(value['study_keys'])
    if (value['probabilities'].shape != (n,10) or value['targets'].shape != (n,10)
            or len(set(value['study_keys'])) != n
            or not np.isfinite(value['probabilities']).all()
            or not np.isin(value['targets'],[0,1]).all()
            or ((value['probabilities']<0)|(value['probabilities']>1)).any()):
        raise ValueError('Invalid prediction coverage, labels or probabilities')
    order=np.argsort(value['study_keys'])
    return {k:v[order] for k,v in value.items()}

def aligned(candidate, reference):
    for key in ('study_keys','patient_ids','targets','views_json'):
        if not np.array_equal(candidate[key],reference[key]):
            raise ValueError('Reference and candidate differ in '+key)

def summarize(y,p,patients):
    result=[];ranks=[]
    for c in range(10):
        rank=_RankedClass(y[:,c],p[:,c]);ranks.append(rank)
        result.append(rank.score(np.ones(len(y)))[0])
    macro=float(np.mean(result)) if all(v is not None for v in result) else None
    return dict(studies=len(y),patients=len(set(patients)),per_class_auc=result,macro_auc=macro,
        positive_studies=y.sum(0).astype(int).tolist(),positive_patients=[
        len(set(patients[y[:,c]==1])) for c in range(10)]),ranks

def compare(candidate,reference,repeats=300):
    aligned(candidate,reference)
    p,y,ids=(candidate[k] for k in ('probabilities','targets','patient_ids'))
    r=reference['probabilities'];views=[json.loads(v) for v in candidate['views_json']]
    report=evaluate_studies(p,y,candidate['study_keys'].tolist(),ids.tolist(),views,
        bootstrap_repeats=repeats,seed=20261105,reference_probabilities=r)
    report.update(holdout_H_accessed=True,fold4_role='development, not independent confirmation',
        test_accessed=False,platform_score_predicted=False,
        checkpoint_selection_changed=False,automatic_submission=False)
    counts=Counter(ids)
    freq=np.asarray([counts[v] for v in ids])
    masks={'one_study_patient':freq==1,'multiple_study_patient':freq>=2,
        'at_least_5_studies_patient':freq>=5,'at_least_10_studies_patient':freq>=10}
    unique,inv=np.unique(ids,return_inverse=True)
    rng=np.random.default_rng(20261105)
    draws=[np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
           for _ in range(repeats)]
    for name,mask in masks.items():
        score,ranks=summarize(y[mask],p[mask],ids[mask])
        old,old_ranks=summarize(y[mask],r[mask],ids[mask])
        score['reference']=old
        score['macro_auc_delta']=score['macro_auc']-old['macro_auc'] if score['macro_auc'] is not None and old['macro_auc'] is not None else None
        differences=[]
        for weights in draws:
            a=[rank.score(weights[mask])[0] for rank in ranks]
            b=[rank.score(weights[mask])[0] for rank in old_ranks]
            if all(v is not None for v in a+b):
                differences.append(float(np.mean(a)-np.mean(b)))
        score['paired_patient_bootstrap']={'valid_replicates':len(differences),
            'attempted_replicates':repeats,'ci95':np.quantile(differences,[.025,.975]).tolist() if differences else None}
        score['exploratory_small_support']=min(score['positive_patients'])<20
        report['slices'][name]=score
    report['diagnostic_only']=True
    report['limitations']=['same development cohort used for repeated checkpoint selection',
        'CNN768 fixed12 image loss versus DINO518 Study loss; not a single-factor architecture ablation',
        'reference raw logits extracted with BF16, current model FP16',
        'no guarantee of hidden-test improvement; full-train refit and independent-fold confirmation still required']
    return report

def process(run,reference,output,repeats):
    history=run/'history.jsonl'
    epochs=[]
    if history.exists():
        for line in history.read_text().splitlines():
            try:
                epochs.append(int(json.loads(line)['epoch']))
            except (json.JSONDecodeError,KeyError):
                continue
    baseline=read_prediction(reference)
    completed=[]
    for epoch in sorted(set(epochs)):
        destination=output/f'epoch_{epoch:02d}.json'
        if destination.exists():
            completed.append(epoch);continue
        path=run/f'epoch_{epoch:02d}/predictions.npz'
        report=compare(read_prediction(path),baseline,repeats)
        report.update(epoch=epoch,candidate_sha256=sha(path),reference_sha256=sha(reference),updated_at=now())
        write(destination,report);completed.append(epoch)
        print(json.dumps({'epoch':epoch,'macro_auc':report['macro_auc'],
            'reference_macro_auc':report['overall']['reference']['macro_auc'],
            'delta':report['overall']['macro_auc_delta']}),flush=True)
    return completed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--max-hours',type=float,default=36.)
    parser.add_argument('--repeats',type=int,default=300)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if sha(args.reference)!=args.reference_sha256:
        raise ValueError('Frozen reference checksum differs')
    with (args.output/'.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        while True:
            epochs=process(args.run,args.reference,args.output,args.repeats)
            controller=args.run/'controller_status.json'
            state=json.loads(controller.read_text()).get('state') if controller.exists() else None
            done=state in {'completed','early_stopped','paused','failed','controller_failed'}
            write(args.output/'monitor_status.json',dict(state='finished' if done else 'monitoring',
                controller_state=state,completed_epochs=epochs,pid=os.getpid(),updated_at=now(),
                cpu_only=True,test_accessed=False,modifies_training=False))
            if not args.watch or done:
                break
            if time.monotonic()-started>args.max_hours*3600:
                write(args.output/'monitor_status.json',dict(state='monitor_deadline_reached',
                    completed_epochs=epochs,updated_at=now(),training_stopped=False))
                break
            time.sleep(30)

if __name__=='__main__':
    main()
