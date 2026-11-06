"""Immutable patient/content-isolated research splits from official TRAIN only.

Precommitted H=legacy fold4, development validation=fold0, optimization=fold1-3.
The trainer receives train.csv/dev.csv only. H may be counted during split
preparation but neither its labels nor images enter training/model selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random

DEFAULT_MAIN = Path('/home/featurize/work/cloud_project_v2')
DEFAULT_OUTPUT = Path('/home/featurize/work/reconstruction_runs_20261103/splits')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_rows(rows):
    # Keep split preparation CPU-only and dependency-free.
    seen, studies = set(), {}
    for row in rows:
        if row.get('split') != 'train' or str(row.get('training_eligible')).lower() != 'true':
            raise ValueError('Split preparation accepts eligible official train only')
        if not row.get('image_id') or row['image_id'] in seen:
            raise ValueError('Duplicate or missing image identity')
        seen.add(row['image_id'])
        if not row.get('subject_id') or row.get('study_key') != f"{row['subject_id']}/{row.get('study_id')}":
            raise ValueError('Invalid original Study/patient identity')
        target = json.loads(row['targets'])
        if (not isinstance(target, list) or len(target) != 10
                or any(type(x) not in (int, float) or x not in (0, 1) for x in target)):
            raise ValueError('Expected unchanged ten binary Study labels')
        if studies.setdefault(row['study_key'], tuple(target)) != tuple(target):
            raise ValueError('Inconsistent labels inside Study')
        for field in ('file_sha256', 'pixel_sha256'):
            v = row.get(field, '')
            if len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
                raise ValueError('Missing original content hash')
    if not studies:
        raise ValueError('Empty source manifest')
    return studies


def content_groups(rows):
    """Union whole patients connected by either file or pixel exact identity."""
    parent = {r['subject_id']: r['subject_id'] for r in rows}
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for field in ('file_sha256', 'pixel_sha256'):
        seen = {}
        for row in rows:
            patient = row['subject_id']
            previous = seen.setdefault(row[field], patient)
            a, b = find(previous), find(patient)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {p: find(p) for p in parent}


def validate_mapping(rows, mapping, groups):
    studies = _validate_rows(rows)
    if set(mapping) != set(studies) or any(type(x) is not int or not 0 <= x < 5 for x in mapping.values()):
        raise ValueError('Fold map must cover each original Study once using folds0..4')
    assignments = {}
    for row in rows:
        group, fold = groups[row['subject_id']], mapping[row['study_key']]
        if assignments.setdefault(group, fold) != fold:
            raise ValueError('Patient or exact-content group crosses folds')
    for fold in range(5):
        labels = [y for key, y in studies.items() if mapping[key] == fold]
        positive = [sum(y[c] for y in labels) for c in range(10)]
        if not labels or any(x in (0, len(labels)) for x in positive):
            raise ValueError(f'Fold{fold} lacks class-positive or class-negative support')


def grouped_folds(rows, groups, seed):
    """Fallback: deterministic rare-label-first balancing at patient/content level."""
    studies = _validate_rows(rows)
    study_patients = {r['study_key']: r['subject_id'] for r in rows}
    grouped = defaultdict(list)
    for key in studies:
        grouped[groups[study_patients[key]]].append(key)
    total = [sum(y[c] for y in studies.values()) for c in range(10)]
    if len(grouped) < 5 or any(x < 5 or len(studies)-x < 5 for x in total):
        raise ValueError('Insufficient independent groups/class support for five folds')
    positives = {g: [sum(studies[k][c] for k in ks) for c in range(10)] for g, ks in grouped.items()}
    rng = random.Random(seed)
    order = sorted(grouped)
    rng.shuffle(order)
    order.sort(key=lambda g: (sum(positives[g][c]/total[c] for c in range(10)), len(grouped[g])), reverse=True)
    target = [x/5 for x in total] + [(len(studies)-x)/5 for x in total] + [len(studies)/5]
    loads = [[0.] * 21 for _ in range(5)]
    mapping = {}
    for group in order:
        n = len(grouped[group])
        vector = positives[group] + [n-x for x in positives[group]] + [n]
        candidates = list(range(5)); rng.shuffle(candidates)
        fold = min(candidates, key=lambda f: sum(((loads[f][i]+v)**2-loads[f][i]**2)/target[i]**2 for i,v in enumerate(vector)))
        loads[fold] = [a+b for a,b in zip(loads[fold], vector)]
        mapping.update({k: fold for k in grouped[group]})
    validate_mapping(rows, mapping, groups)
    return mapping


def _stats(rows):
    studies = {r['study_key']: json.loads(r['targets']) for r in rows}
    return {'images': len(rows), 'studies': len(studies), 'patients': len({r['subject_id'] for r in rows}),
            'positive_studies': [sum(y[c] for y in studies.values()) for c in range(10)]}


def _csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def prepare_splits(train_manifest=DEFAULT_MAIN/'data/clean_v1_reviewed/train_clean.csv',
                   legacy_plan=DEFAULT_MAIN/'data/cv5_overnight/folds.json', output=DEFAULT_OUTPUT,
                   holdout_fold=4, dev_fold=0, seed=20261103):
    if holdout_fold != 4 or dev_fold != 0:
        raise ValueError('This preregistered experiment fixes H=fold4 and dev=fold0')
    source, output = Path(train_manifest).resolve(strict=True), Path(output).resolve()
    source_sha = sha(source)
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    if not fields or len(set(fields)) != len(fields) or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Malformed source CSV')
    _validate_rows(rows)
    groups = content_groups(rows)
    mapping, legacy_sha, fallback_reason = None, None, None
    if legacy_plan is not None and Path(legacy_plan).is_file():
        legacy_sha = sha(legacy_plan)
        legacy = json.loads(Path(legacy_plan).read_text())
        try:
            if legacy.get('source_manifest_sha256') != source_sha or legacy.get('n_splits') != 5:
                raise ValueError('Legacy split is not bound to this source manifest')
            candidate = legacy['study_to_fold']
            validate_mapping(rows, candidate, groups)
            mapping = candidate
        except (KeyError, ValueError) as exc:
            fallback_reason = str(exc)
    else:
        fallback_reason = 'No existing five-fold plan supplied'
    if mapping is None:
        mapping = grouped_folds(rows, groups, seed)
        method = 'deterministic_patient_content_multilabel_greedy'
    else:
        method = 'revalidated_existing_fivefold_mapping'
    subsets = {'train': [], 'dev': [], 'holdout': []}
    for row in rows:
        fold = mapping[row['study_key']]
        subsets['holdout' if fold == holdout_fold else 'dev' if fold == dev_fold else 'train'].append(row)
    overlaps = {}
    for a, b in [('train','dev'), ('train','holdout'), ('dev','holdout')]:
        overlaps[a+'__'+b] = {key: len({r[key] for r in subsets[a]} & {r[key] for r in subsets[b]})
                             for key in ('subject_id','study_key','file_sha256','pixel_sha256')}
        if any(overlaps[a+'__'+b].values()):
            raise ValueError('Patient/Study/content isolation failed')
    payloads = {name+'.csv': _csv_bytes(subset, fields) for name, subset in subsets.items()}
    plan = {'schema': 'reconstruction_patient_content_holdout_v1', 'source_manifest': str(source),
            'source_manifest_sha256': source_sha, 'legacy_plan_sha256': legacy_sha,
            'method': method, 'fallback_reason': fallback_reason, 'seed': seed,
            'holdout_fold': holdout_fold, 'dev_fold': dev_fold, 'train_folds': [1,2,3],
            'roles': {name: {'path': str(output/(name+'.csv')), 'sha256': hashlib.sha256(payloads[name+'.csv']).hexdigest(),
                             **_stats(subset)} for name, subset in subsets.items()},
            'isolation': overlaps, 'source_rows_preserved': True, 'labels_rewritten': False,
            'holdout_usage': 'split-preparation metadata only; excluded from all new training and upstream weights',
            'official_valid_read': False, 'test_read': False}
    payloads['split_plan.json'] = (json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True)+'\n').encode()
    if sha(source) != source_sha:
        raise RuntimeError('Source manifest changed during split preparation')
    if output.exists():
        for name, content in payloads.items():
            path = output/name
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(f'Existing split differs; refusing replacement: {path}')
        return plan
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        with (output/name).open('xb') as stream:
            stream.write(content)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', type=Path, default=DEFAULT_MAIN/'data/clean_v1_reviewed/train_clean.csv')
    parser.add_argument('--legacy-plan', type=Path, default=DEFAULT_MAIN/'data/cv5_overnight/folds.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare_splits(**vars(args))
    print(json.dumps({'method': result['method'], 'roles': result['roles'], 'isolation': result['isolation']}, indent=2))


if __name__ == '__main__':
    main()
