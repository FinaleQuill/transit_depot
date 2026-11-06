"""Fixed, image-local binary probability fusion; no fitted validation weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def fuse_probabilities(values, method, *, pos_weight=None):
    """Fuse two complete ten-class image predictions before Study averaging."""
    values=np.asarray(values,dtype=np.float64)
    if (values.ndim not in (2,3) or values.shape[0]!=2 or values.shape[-1]!=10
            or not np.isfinite(values).all() or (values<0).any() or (values>1).any()):
        raise ValueError('Require two finite probability arrays with ten classes')
    if method=='probability_mean':
        return values.mean(0)
    if method=='logit_mean':
        clipped=np.clip(values,1e-7,1-1e-7)
        logit=np.mean(np.log(clipped)-np.log1p(-clipped),axis=0)
        return np.exp(-np.logaddexp(0.,-logit))
    if method=='train_debiased_probability_mean':
        weights=np.asarray(pos_weight,dtype=np.float64)
        if weights.shape!=(10,) or not np.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('Require ten fixed positive training-derived BCE weights')
        corrected=values/(values+weights*(1-values))
        return corrected.mean(0)
    raise ValueError('Unknown fixed fusion method')


def class_aucs(targets, values):
    """Independent midrank AUC for the CPU screening; deployment uses sklearn."""
    result=[]
    for c in range(10):
        y,s=targets[:,c],values[:,c]
        positive=int(y.sum()); negative=len(y)-positive
        if not positive or not negative:
            raise ValueError('Need both classes in each original-valid target')
        order=np.argsort(s,kind='stable'); sorted_values=s[order]
        starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
        ends=np.r_[starts[1:],len(s)]
        ranks=np.repeat((starts+1+ends)/2,ends-starts)
        result.append(float((np.dot(ranks,y[order])-positive*(positive+1)/2)/(positive*negative)))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_values(directory, manifest_rows):
    directory=Path(directory)
    report=json.loads((directory/'validation_report.json').read_text())
    file=directory/'valid.image_predictions.csv'
    if (report.get('complete') is not True or report['tta']!='hflip2'
            or report['image_size']!=768 or report['member_count']!=1
            or report['image_predictions_sha256']!=sha(file)):
        raise ValueError('Require original complete 768 hflip2 image references')
    result={}
    with file.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=(row['Subject_id'],row['Study_id'],row['Image_id'])
            column=int(row['Predict_class'])
            if not 1<=column<=10 or column in result.setdefault(key,{}):
                raise ValueError('Duplicate or malformed image/class identity')
            result[key][column]=float(row['Probability'])
    keys=[(r['subject_id'],r['study_id'],r['image_id']) for r in manifest_rows]
    if set(result)!=set(keys) or any(set(r)!=set(range(1,11)) for r in result.values()):
        raise ValueError('Need exact original image coverage, with all classes')
    return np.asarray([[result[k][c] for c in range(1,11)] for k in keys])


def screen(manifest, validations, plan_path, output):
    plan=json.loads(Path(plan_path).read_text())
    with Path(manifest).open(encoding='utf-8-sig') as f:
        records=list(csv.DictReader(f))
    if len(records)!=1422 or any(row['split']!='valid' for row in records):
        raise ValueError('Screening only accepts complete original official valid')
    labels={}
    groups={}
    for index,row in enumerate(records):
        key=(row['subject_id'],row['study_id']); target=json.loads(row['targets'])
        if key in labels and labels[key]!=target:
            raise ValueError('Inconsistent original Study labels')
        labels[key]=target
        groups.setdefault(key,[]).append(index)
    keys=sorted(labels); targets=np.asarray([labels[k] for k in keys])
    sources=np.stack([image_values(p,records) for p in validations])
    methods={'AB_logit_mean':'logit_mean',
             'AB_train_debiased_probability_mean':'train_debiased_probability_mean'}
    if list(methods)!=plan['candidate_methods']:
        raise ValueError('Registered bounded candidate set changed')
    result={}
    for name,method in {'AB_probability_mean':'probability_mean',**methods}.items():
        images=fuse_probabilities(sources,method,pos_weight=plan['train_only_pos_weight'])
        studies=np.stack([images[groups[k]].mean(0) for k in keys])
        aucs=class_aucs(targets,studies)
        result[name]={'macro_auc':float(np.mean(aucs)),'per_class_auc':aucs}
    output=Path(output)
    with output.open('x') as f:
        json.dump({'scores':result,'plan_sha256':sha(plan_path),
            'manifest_sha256':sha(manifest),'test_used':False,'trained':False,
            'validation_fitted_parameters':False},f,indent=2)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--validations',type=Path,nargs=2,required=True)
    parser.add_argument('--plan-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    screen(**vars(parser.parse_args()))
