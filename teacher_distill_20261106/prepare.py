"""Freeze original-data nested patient folds, without model inference/training.

The existing full decode audit is reused only after its hashes and original CSV
rows are verified. Every current original image path and file size is checked.
The runtime loader additionally verifies SHA on first read and decodes every use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from raw_rebuild.prepare_data import load_official
from teacher_distill_20261106.data import load_manifest, validate_records


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def publish(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Refusing to replace a different frozen data artifact: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def summarize(rows):
    studies = {}
    for row in rows:
        studies.setdefault(row['study_key'], []).append(row)
    patients = Counter(group[0]['subject_id'] for group in studies.values())
    target = np.asarray([json.loads(g[0]['targets']) if isinstance(g[0]['targets'], str)
                         else g[0]['targets'] for g in studies.values()], dtype=int)
    return dict(images=len(rows), studies=len(studies), patients=len(patients),
        positive_studies=target.sum(0).tolist(),
        ap_only_studies=sum(all(r.get('view_position') == 'AP' for r in group) for group in studies.values()),
        single_image_studies=sum(len(group) == 1 for group in studies.values()),
        studies_per_patient_histogram=dict(sorted(Counter(patients.values()).items())),
        excluded_images=0)


def balanced_patient_folds(records, *, seed=20261106, n_splits=3):
    """Deterministic greedy balancing of patient/Study disease and geometry counts.

    Features are ten disease-positive patient counts, ten positive Study counts,
    total Studies/images, AP-only/single-image Studies, four patient frequency
    strata and patient counts. The seed only breaks equal priority/score ties.
    Labels or statistics from H, valid or test never enter this assignment.
    """
    checked = validate_records(records)
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError('At least two inner folds are required')
    studies, by_patient = {}, defaultdict(list)
    for row in checked:
        studies.setdefault(row['study_key'], []).append(row)
    for group in studies.values():
        by_patient[group[0]['subject_id']].append(group)
    patients = sorted(by_patient)
    if len(patients) < n_splits:
        raise ValueError('Fewer patients than folds')
    vectors = []
    for patient in patients:
        groups = by_patient[patient]
        y = np.asarray([g[0]['targets'] for g in groups], dtype=np.float64)
        n = len(groups)
        bucket = 0 if n == 1 else 1 if n <= 4 else 2 if n <= 9 else 3
        vectors.append(np.r_[y.max(0), y.sum(0), n, sum(map(len, groups)),
            sum(all(r.get('view_position') == 'AP' for r in g) for g in groups),
            sum(len(g) == 1 for g in groups), [int(i == bucket) for i in range(4)], 1])
    vectors = np.asarray(vectors)
    totals = vectors.sum(0)
    target = np.maximum(totals / n_splits, 1.)
    # Rare positive features and large patients are allocated first.
    priority = (vectors / np.maximum(totals, 1.)).sum(1)
    rng = np.random.default_rng(seed)
    priority_ties = rng.random(len(patients))
    order = sorted(range(len(patients)), key=lambda i: (-priority[i], priority_ties[i], patients[i]))
    capacities = np.full(n_splits, len(patients) // n_splits, dtype=int)
    capacities[:len(patients) % n_splits] += 1
    counts = np.zeros(n_splits, dtype=int)
    loads = np.zeros((n_splits, vectors.shape[1]), dtype=np.float64)
    mapping = {}
    for index in order:
        vector = vectors[index]
        # Difference in squared standardized fold imbalance after assignment.
        scores = (((loads + vector) / target) ** 2 - (loads / target) ** 2).sum(1)
        scores[counts >= capacities] = np.inf
        best = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0., atol=1e-12))
        fold = int(best[int(rng.integers(len(best)))])
        mapping[patients[index]] = fold
        loads[fold] += vector
        counts[fold] += 1
    feature_names = ([f'positive_patients_class_{i}' for i in range(1, 11)]
        + [f'positive_studies_class_{i}' for i in range(1, 11)]
        + ['studies', 'images', 'ap_only_studies', 'single_image_studies',
           'patients_1_study', 'patients_2_4_studies', 'patients_5_9_studies',
           'patients_10_plus_studies', 'patients'])
    return mapping, dict(method='greedy standardized feature imbalance; rare/large patients first',
        seed=seed, n_splits=n_splits, feature_names=feature_names,
        feature_totals=totals.tolist(), fold_features=loads.tolist(),
        fold_patient_capacities=capacities.tolist(),
        fitting_source='R only; no H, official valid or test statistics')


def overlap(a, b):
    return {field: len({r[field] for r in a} & {r[field] for r in b})
            for field in ('subject_id', 'study_key', 'image_id', 'file_sha256', 'pixel_sha256')}


def _check_source_path(task):
    root, row = task
    try:
        path = (root / row['relative_path']).resolve(strict=True)
        if not path.is_relative_to(root / 'train'):
            raise ValueError('Original source path escaped the original train root')
        if not path.is_file() or path.stat().st_size != int(row['file_bytes']):
            raise ValueError('Original source is not a regular file or its size changed')
        return None
    except Exception as exc:
        return dict(image_id=row['image_id'], relative_path=row['relative_path'],
                    error=f'{type(exc).__name__}: {exc}')


def prepare(*, root, source_dir, output, families=('dinov3_l', 'convnext_l'),
            seed=20261106, workers=16):
    root, source_dir, output = Path(root).resolve(strict=True), Path(source_dir).resolve(strict=True), Path(output).resolve()
    if len(families) != 2 or len(set(families)) != 2 or any('/' in x or not x for x in families):
        raise ValueError('Exactly two distinct plain teacher family names are required')
    if seed != 20261106:
        raise ValueError('The first-stage fold seed is frozen at 20261106')
    output.mkdir(parents=True, exist_ok=True)
    source_csv, source_plan, audit_path = (source_dir / name for name in ('source_train.csv', 'folds.json', 'audit.json'))
    source_hashes = {str(path): sha256(path) for path in (root/'train.csv', source_csv, source_plan, audit_path)}
    audit = json.loads(audit_path.read_text())
    if (audit['official_csv_sha256']['train'] != source_hashes[str(root/'train.csv')]
            or audit['manifest_sha256']['source_train.csv'] != source_hashes[str(source_csv)]
            or audit['manifest_sha256']['folds.json'] != source_hashes[str(source_plan)]
            or audit.get('excluded_rows') != 0 or audit.get('original_files_modified') is not False
            or audit['datasets']['train']['decoded_images'] != 30344
            or audit['datasets']['train']['failed_images'] != 0):
        raise ValueError('Original full-decode audit lineage is missing or changed')
    with source_csv.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    checked, official = load_manifest(source_csv), load_official(root, 'train')
    if len(records) != 30344 or len(official) != 30344 or summarize(records)['studies'] != 18513 or summarize(records)['patients'] != 11055:
        raise ValueError('Original train must retain 30344 images / 18513 Studies / 11055 patients')
    compare_fields = ('image_id', 'split', 'subject_id', 'study_id', 'study_key', 'dicom_id',
        'relative_path', 'label_text', 'label_ids', 'view_position', 'patient_orientation', 'rows_csv', 'columns_csv')
    for raw, saved, validated in zip(official, records, checked, strict=True):
        if any(raw[field] != saved[field] for field in compare_fields) or tuple(json.loads(raw['targets'])) != validated['targets']:
            raise ValueError('Official original CSV identity, labels, order or fields changed')
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, failure in enumerate(pool.map(_check_source_path, ((root, row) for row in records)), 1):
            if failure is not None:
                errors.append(failure)
            if count % 5000 == 0 or count == len(records):
                print(json.dumps(dict(original_paths_checked=count, total=len(records), failures=len(errors))), flush=True)
    if errors:
        publish(output/'FAILED_SOURCE_PATHS.json', json_bytes(errors))
        raise RuntimeError(f'{len(errors)} original images missing/changed; no split artifacts published')
    outer = json.loads(source_plan.read_text())
    if (outer.get('all_original_images_retained') is not True
            or outer.get('exclusions_applied') is not False
            or outer.get('dev_fold') != 4):
        raise ValueError('Existing raw patient fold provenance is not the expected original fold4')
    mapping = outer['study_to_fold']
    if set(mapping) != {row['study_key'] for row in records}:
        raise ValueError('Outer mapping does not exactly cover original train')
    patients_outer = {}
    for row in records:
        fold = mapping[row['study_key']]
        if type(fold) is not int or fold not in range(5) or patients_outer.setdefault(row['subject_id'], fold) != fold:
            raise ValueError('Invalid outer mapping or patient crosses outer folds')
    R = [row for row in records if mapping[row['study_key']] != 4]
    H = [row for row in records if mapping[row['study_key']] == 4]
    r_summary, h_summary = summarize(R), summarize(H)
    if (r_summary['studies'], r_summary['patients'], r_summary['images']) != (14811, 8844, 24277):
        raise ValueError('R no longer matches the registered raw outer training population')
    if (h_summary['studies'], h_summary['patients'], h_summary['images']) != (3702, 2211, 6067):
        raise ValueError('H no longer matches the registered raw development population')
    if any(overlap(R, H).values()) or len(R) + len(H) != len(records):
        raise ValueError('R/H overlap or original coverage loss')
    assignment, stratification = balanced_patient_folds(R, seed=seed)
    payloads = {'source_train.csv': csv_bytes(records, fields), 'R.csv': csv_bytes(R, fields),
        'H.csv': csv_bytes(H, fields), 'R_inference.csv': csv_bytes(R, fields)}
    inner_reports, teacher_runs = [], []
    for fold in range(3):
        train = [row for row in R if assignment[row['subject_id']] != fold]
        oof = [row for row in R if assignment[row['subject_id']] == fold]
        if any(overlap(train, oof).values()) or any(overlap(train, H).values()) or any(overlap(oof, H).values()):
            raise ValueError('Nested teacher training/OOF/H integrity overlap')
        if len(train) + len(oof) != len(R) or {r['image_id'] for r in train+oof} != {r['image_id'] for r in R}:
            raise ValueError('Nested fold lost original R images')
        names = dict(train=f'inner_{fold}/train.csv', oof=f'inner_{fold}/oof.csv', development='H.csv')
        payloads[names['train']], payloads[names['oof']] = csv_bytes(train, fields), csv_bytes(oof, fields)
        train_report, oof_report = summarize(train), summarize(oof)
        if any(n == 0 or n == train_report['studies'] for n in train_report['positive_studies']):
            raise ValueError('Teacher train requires positive and negative Studies in all classes')
        inner_reports.append(dict(fold=fold, train=train_report, oof=oof_report,
            train_oof_overlap=overlap(train, oof), train_H_overlap=overlap(train, H)))
        for family in families:
            teacher_runs.append(dict(run_id=f'{family}_inner{fold}', family=family, inner_fold=fold,
                train_manifest=str(output/names['train']), oof_manifest=str(output/names['oof']),
                development_manifest=str(output/'H.csv'),
                input_sha256={role:hashlib.sha256(payloads[path]).hexdigest() for role,path in names.items()},
                training_patients=train_report['patients'], oof_patients=oof_report['patients'],
                H_patients_excluded=True, complete_study_all_original_views=True,
                old_teacher_or_soft_cache_reused=False))
    plan = dict(schema='original_nested_teacher_distillation_data_v1', seed=seed,
        image_root=str(root), output=str(output), n_inner_folds=3, teacher_families=list(families),
        source_hashes=source_hashes, original_train=summarize(records), R=r_summary, H=h_summary,
        outer_source_fold=4, H_role='development confirmation, historically used; not an independent holdout',
        H_is_independent=False, official_valid_used=False, test_used=False,
        old_soft_labels_used=False, exclusions_applied=False, labels_changed=False,
        no_finding_or_other_cooccurrences_preserved=True, all_original_images_retained=True,
        source_manifest=str(output/'source_train.csv'), R_manifest=str(output/'R.csv'),
        R_inference_manifest=str(output/'R_inference.csv'), H_manifest=str(output/'H.csv'),
        patient_to_inner_fold=assignment, stratification=stratification,
        inner_folds=inner_reports, teacher_runs=teacher_runs,
        manifest_sha256={name:hashlib.sha256(content).hexdigest() for name,content in payloads.items()},
        integrity_checks=dict(current_original_paths_checked=len(records), current_file_sizes_checked=len(records),
            full_decode_audit_reused=str(audit_path), full_decode_audit_sha256=source_hashes[str(audit_path)],
            decoded_again_during_preparation=False, runtime_first_read_SHA256_required=True,
            runtime_all_images_fully_decoded=True, missing_or_changed_paths=0),
        inference_roles=dict(R='fresh train-only OOF extraction; only its excluded-fold teachers permitted',
            H='teacher validation/confirmation only; no teacher/student fitting',
            R_inference='all R original Studies for extraction/coverage checks, not an old cache'))
    payloads['plan.json'] = json_bytes(plan)
    for path, before in source_hashes.items():
        if sha256(path) != before:
            raise ValueError(f'Protected source changed during preparation: {path}')
    for name, content in payloads.items():
        publish(output/name, content)
    receipt = dict(schema='nested_data_preparation_complete_v1', complete=True,
        plan_sha256=hashlib.sha256(payloads['plan.json']).hexdigest(), original_images=30344,
        original_studies=18513, original_patients=11055, teacher_runs=len(teacher_runs),
        R_H_complete_coverage=True, fitting_or_inference_started=False)
    publish(output/'_SUCCESS.json', json_bytes(receipt))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default='/home/featurize/data/cxr_first')
    parser.add_argument('--source-dir', type=Path, default='/home/featurize/data/raw_rebuild_20261105')
    parser.add_argument('--output', type=Path, default='/home/featurize/data/kd_rebuild_20261106')
    parser.add_argument('--families', nargs=2, default=['dinov3_l', 'convnext_l'])
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    plan = prepare(**vars(args))
    print(json.dumps({k:plan[k] for k in ('output','original_train','R','H','inner_folds')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
