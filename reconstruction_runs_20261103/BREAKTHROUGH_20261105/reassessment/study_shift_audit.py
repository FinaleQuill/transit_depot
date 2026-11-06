"""CPU metadata audit of official train/valid and current train-only folds.

Reads no test data or predictions. Previously stored test inventory is quoted
only for identity/coverage context, not to derive thresholds or model choices.
"""
from pathlib import Path
import csv
import hashlib
import json
from collections import defaultdict, Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path('/home/featurize/work')
OUT = ROOT / 'reconstruction_runs_20261103/BREAKTHROUGH_20261105/reassessment'
SOURCES = {
    'official_train': ROOT/'cloud_project_v2/data/clean_v1_reviewed/train_clean.csv',
    'official_valid': ROOT/'cloud_project_v2/data/clean_v1_reviewed/valid_manifest.csv',
    'dino_optimization': ROOT/'reconstruction_runs_20261103/BREAKTHROUGH_20261105/splits/train.csv',
    'dino_development': ROOT/'reconstruction_runs_20261103/BREAKTHROUGH_20261105/splits/dev.csv',
}

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if any(r['split'] not in ('train','valid') for r in rows):
        raise ValueError('Test data are not accepted')
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['study_key']].append(row)
    patients=Counter(v[0]['subject_id'] for v in grouped.values())
    studies=[]
    for key, values in grouped.items():
        targets={tuple(json.loads(v['targets'])) for v in values}
        assert len(targets)==1
        views={v['view_position'].strip().upper() or 'MISSING' for v in values}
        orientations={v['patient_orientation'].strip().upper() or 'MISSING' for v in values}
        subject=values[0]['subject_id']
        studies.append(dict(key=key, patient=subject, target=next(iter(targets)), images=len(values),
            views=views, orientations=orientations, patient_studies=patients[subject],
            width_mean=np.mean([int(v['width']) for v in values]),
            height_mean=np.mean([int(v['height']) for v in values])))
    return rows,studies

def slice_stat(studies):
    if not studies:
        return dict(studies=0,patients=0)
    y=np.asarray([s['target'] for s in studies],dtype=np.int64)
    grouped=defaultdict(list)
    for s in studies:
        grouped[s['patient']].append(s)
    n=len(studies)
    per_patient=[np.asarray([s['target'] for s in values],dtype=np.float64).mean(0)
                 for values in grouped.values()]
    return dict(studies=n, patients=len(grouped), images=sum(s['images'] for s in studies),
        positive_studies=y.sum(0).tolist(),
        positive_patients=[len({s['patient'] for s in studies if s['target'][c]}) for c in range(10)],
        prevalence=y.mean(0).tolist(), patient_equal_prevalence=np.mean(per_patient,axis=0).tolist(),
        disease_cardinality_mean=float(y[:,:9].sum(1).mean()),
        no_finding_prevalence=float(y[:,9].mean()),
        ap_only_fraction=float(np.mean([s['views']=={'AP'} for s in studies])),
        single_image_fraction=float(np.mean([s['images']==1 for s in studies])),
        any_lateral_fraction=float(np.mean([bool(s['views'] & {'LATERAL','LL'}) for s in studies])),
        any_recumbent_fraction=float(np.mean(['RECUMBENT' in s['orientations'] for s in studies])))

def audit(studies):
    n=len(studies)
    patients=defaultdict(list)
    for s in studies:
        patients[s['patient']].append(s)
    counts=np.array([len(v) for v in patients.values()],dtype=np.int64)
    criteria={
        'all':lambda s:True,
        'single_image':lambda s:s['images']==1,
        'multi_image':lambda s:s['images']>1,
        'AP_only':lambda s:s['views']=={'AP'},
        'PA_present':lambda s:'PA' in s['views'],
        'lateral_present':lambda s:bool(s['views'] & {'LATERAL','LL'}),
        'recumbent_present':lambda s:'RECUMBENT' in s['orientations'],
        'missing_view_present':lambda s:'MISSING' in s['views'],
        'patient_1_study':lambda s:s['patient_studies']==1,
        'patient_2to4_studies':lambda s:2<=s['patient_studies']<=4,
        'patient_5to9_studies':lambda s:5<=s['patient_studies']<=9,
        'patient_10plus_studies':lambda s:s['patient_studies']>=10,
        'patient_5plus_studies':lambda s:s['patient_studies']>=5,
        'patient_20plus_studies':lambda s:s['patient_studies']>=20,
        'AP_only_and_5plus_studies':lambda s:s['views']=={'AP'} and s['patient_studies']>=5,
        'AP_single_image_and_5plus_studies':lambda s:s['views']=={'AP'} and s['images']==1 and s['patient_studies']>=5,
    }
    result={name:slice_stat([s for s in studies if f(s)]) for name,f in criteria.items()}
    for value in result.values():value['study_fraction']=value['studies']/n
    mixed=[]
    multi=[v for v in patients.values() if len(v)>1]
    for c in range(10):
        pos,neg,both=0,0,0
        within_pairs=0
        for group in patients.values():
            positives=sum(s['target'][c] for s in group)
            negatives=len(group)-positives
            pos+=int(positives>0);neg+=int(negatives>0)
            both+=int(positives>0 and negatives>0)
            within_pairs+=positives*negatives
        total_pos=sum(s['target'][c] for s in studies)
        mixed.append(dict(class_id=c+1,positive_patients=pos,negative_patients=neg,
            mixed_label_patients=both,mixed_fraction_among_multi=both/len(multi),
            same_patient_positive_negative_pairs=within_pairs,
            all_positive_negative_pairs=total_pos*(n-total_pos),
            same_patient_pair_fraction=within_pairs/(total_pos*(n-total_pos))))
    return dict(summary=result['all'], slices=result, longitudinal_label_change=mixed,
        patient_counts=dict(patients=len(patients),studies=n,mean=float(counts.mean()),
            median=float(np.median(counts)),p90=float(np.quantile(counts,.9)),
            max=int(counts.max()),single_study_patient_fraction=float(np.mean(counts==1)),
            multi_study_study_fraction=float(counts[counts>1].sum()/n),
            top10_study_fraction=float(np.sort(counts)[-10:].sum()/n),
            kish_patient_weight_n=float(n*n/(counts.astype(float)**2).sum())),
        varying_full_label_vector_patients=sum(len({tuple(s['target']) for s in group})>1 for group in multi),
        multi_study_patients=len(multi))

def main():
    report=dict(recorded_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        schema='train_valid_study_structure_audit_v1',test_data_opened=False,
        test_predictions_opened=False,model_or_training_changed=False,
        bin_policy='Exploratory round-count bins fixed in this script; not fitted to test',
        source_sha256={},datasets={})
    datasets={}
    for name,path in SOURCES.items():
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        rows,studies=read(path)
        report['datasets'][name]=audit(studies)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        report['source_sha256'][str(path)]=before
        datasets[name]=(rows,studies)
    a,b=datasets['dino_optimization'][0],datasets['dino_development'][0]
    report['new_split_overlap']={field:len({r[field] for r in a}&{r[field] for r in b})
        for field in ('subject_id','study_key','file_sha256','pixel_sha256')}
    inventory=ROOT/'reconstruction_runs_20261103/BREAKTHROUGH_20261105/audit/cxr_rules_data_audit_20261105.json'
    old=json.loads(inventory.read_text())
    report['quoted_existing_test_inventory']={k:old['test'][k] for k in ('images','studies','patients','study_image_count_hist','max_studies_per_patient')}
    report['quoted_inventory_sha256']=hashlib.sha256(inventory.read_bytes()).hexdigest()
    report['quoted_inventory_note']='Only previously published integrity counts. No test image/manifest/prediction newly opened; not used to select model, weights, threshold or sampling.'
    (OUT/'study_shift_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name, data in report['datasets'].items():
        print(name,json.dumps(data['patient_counts']))
        for k in ['all','patient_1_study','patient_5plus_studies','patient_10plus_studies','AP_only_and_5plus_studies']:
            s=data['slices'][k]
            print(k,{q:s.get(q) for q in ('studies','patients','study_fraction','positive_patients','no_finding_prevalence','disease_cardinality_mean','ap_only_fraction','single_image_fraction','any_recumbent_fraction')})

if __name__=='__main__':main()
