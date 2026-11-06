"""Build original, unfiltered manifests and audit structure without test fitting.

The official CSV is the only row/label source. Historical folds supply patient
assignments only, never exclusions. Every referenced original image is decoded
and hashed. Missing/unreadable images fail preparation with an explicit report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

from PIL import Image, ImageOps

FIELDS = ['image_id', 'split', 'subject_id', 'study_id', 'study_key', 'dicom_id',
          'relative_path', 'label_text', 'label_ids', 'targets', 'view_position',
          'patient_orientation', 'rows_csv', 'columns_csv', 'status',
          'training_eligible', 'exclusion_reasons', 'file_sha256', 'pixel_sha256',
          'width', 'height', 'mode', 'file_bytes']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(text):
    values = [int(x) for x in text.split(';')]
    if not values or len(set(values)) != len(values) or any(x not in range(1, 11) for x in values):
        raise ValueError(f'Invalid original labels: {text!r}')
    return values, [int(i in values) for i in range(1, 11)]


def load_official(root, split):
    with (Path(root) / f'{split}.csv').open(newline='', encoding='utf-8-sig') as stream:
        source = list(csv.DictReader(stream))
    records, seen, study_labels = [], set(), {}
    for index, raw in enumerate(source):
        dicom, patient, study = (raw[k] for k in ('Dicom_id', 'Subject_id', 'Study_id'))
        if dicom in seen:
            raise ValueError(f'Duplicate Dicom_id in original CSV: {dicom}')
        seen.add(dicom)
        key = f'{patient}/{study}'
        row = dict(image_id=f'{split}:{index:06d}', split=split, subject_id=patient,
                   study_id=study, study_key=key, dicom_id=dicom,
                   relative_path=f'{split}/{key}/{dicom}.jpg',
                   view_position=raw['ViewPosition'], patient_orientation=raw['PatientOrientation'],
                   rows_csv=raw['Rows'], columns_csv=raw['Columns'], status='ok',
                   training_eligible=split == 'train', exclusion_reasons='[]')
        if split != 'test':
            ids, target = labels(raw['Predict_class'])
            if study_labels.setdefault(key, target) != target:
                raise ValueError(f'Original within-Study label disagreement: {key}')
            row.update(label_text=raw['Predict_class'], label_ids=json.dumps(ids), targets=json.dumps(target))
        records.append(row)
    return records


def inspect_image(task):
    root, row = task
    path = Path(root) / row['relative_path']
    try:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            if source.mode in {'I', 'F'} or source.mode.startswith('I;'):
                raise ValueError(f'Non-8-bit input: {source.mode}')
            mode = source.mode
            image = ImageOps.exif_transpose(source).convert('RGB')
            pixel_sha = hashlib.sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest()
            width, height = image.size
        return dict(row, file_sha256=hashlib.sha256(data).hexdigest(), pixel_sha256=pixel_sha,
                    width=width, height=height, mode=mode, file_bytes=len(data))
    except Exception as exc:
        return dict(row, status='error', error=f'{type(exc).__name__}: {exc}')


def summarize(rows):
    studies = defaultdict(list)
    for row in rows:
        studies[row['study_key']].append(row)
    patients = Counter(key.split('/')[0] for key in studies)
    result = dict(images=len(rows), studies=len(studies), patients=len(patients),
                  image_count_per_study=dict(sorted(Counter(map(len, studies.values())).items())),
                  studies_per_patient_hist=dict(sorted(Counter(patients.values()).items())),
                  max_studies_per_patient=max(patients.values()),
                  mean_studies_per_patient=len(studies)/len(patients),
                  fraction_studies_from_patients_with_at_least_5_studies=sum(n for n in patients.values() if n >= 5)/len(studies),
                  single_image_study_fraction=sum(len(x) == 1 for x in studies.values())/len(studies),
                  view_position_image_counts=dict(Counter(r['view_position'] for r in rows)),
                  ap_only_study_fraction=sum(all(r['view_position'] == 'AP' for r in group) for group in studies.values())/len(studies),
                  image_modes=dict(Counter(r['mode'] for r in rows)),
                  min_width=min(r['width'] for r in rows), max_width=max(r['width'] for r in rows),
                  min_height=min(r['height'] for r in rows), max_height=max(r['height'] for r in rows),
                  dimensions_differ_from_csv=sum((r['width'],r['height']) != (int(r['columns_csv']),int(r['rows_csv'])) for r in rows),
                  decoded_images=len(rows), failed_images=0)
    if rows[0]['split'] != 'test':
        targets = [json.loads(g[0]['targets']) for g in studies.values()]
        result['positive_studies'] = [sum(y[c] for y in targets) for c in range(10)]
        result['positive_patients'] = [len({g[0]['subject_id'] for g in studies.values() if json.loads(g[0]['targets'])[c]}) for c in range(10)]
        repeated = [p for p,n in patients.items() if n > 1]
        result['multi_study_patients'] = len(repeated)
        result['patients_with_changing_labels'] = sum(len({g[0]['targets'] for g in studies.values() if g[0]['subject_id'] == p}) > 1 for p in repeated)
    return result


def validate_folds(rows, mapping):
    study_keys = {r['study_key'] for r in rows}
    if set(mapping) != study_keys or any(type(v) is not int or v not in range(5) for v in mapping.values()):
        raise ValueError('Legacy fold mapping must exactly cover every original Study')
    assignments = {}
    for row in rows:
        fold = mapping[row['study_key']]
        for field in ('subject_id', 'file_sha256', 'pixel_sha256'):
            key = (field, row[field])
            if assignments.setdefault(key, fold) != fold:
                raise ValueError(f'Original patient/content crosses folds: {field}')
    return True


def csv_bytes(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def publish(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite differing audit artifact: {path}')
        return
    with path.open('xb') as stream:
        stream.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode()


def audit_tar(path, test_rows, root):
    test_by_name = {r['dicom_id'] + '.jpg': r for r in test_rows}
    groups = defaultdict(list)
    with tarfile.open(path) as archive:
        images = [m for m in archive.getmembers() if m.isfile() and Path(m.name).suffix.lower() in {'.jpg','.png','.jpeg'}]
        for member in images:
            rel = Path(member.name)
            row = test_by_name.get(rel.name)
            if row is None:
                raise ValueError(f'Cost example missing from test CSV: {rel.name}')
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != row['file_sha256']:
                raise ValueError(f'Cost/example original mismatch: {rel.name}')
            groups[rel.parts[1]].append(row)
    result = {name: summarize(rows) for name, rows in groups.items()}
    result['all_sample_image_bytes_match_original_test'] = True
    result['sha256'] = digest(path)
    result['is_complete_official_timing_set'] = False
    return result


def prepare(root, legacy_plan, output, cost_tar, workers=6):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = {split: digest(root/f'{split}.csv') for split in ('train','valid','test')}
    data, errors = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split in ('train','valid','test'):
            raw = load_official(root, split)
            checked = []
            for i,row in enumerate(pool.map(inspect_image, ((str(root),r) for r in raw), chunksize=16), 1):
                checked.append(row)
                if row['status'] != 'ok':
                    errors.append(row)
                if i % 2000 == 0 or i == len(raw):
                    print(json.dumps(dict(split=split, decoded=i, total=len(raw), errors=len(errors))), flush=True)
            data[split] = checked
    if errors:
        publish(output/'FAILED_IMAGES.json', json_bytes(errors))
        raise RuntimeError(f'{len(errors)} missing/unreadable files; no manifests created')
    raw_train, valid, test = data['train'], data['valid'], data['test']
    if len(raw_train) != 30344 or len({r['study_key'] for r in raw_train}) != 18513:
        raise ValueError('Original train count must be 30344 images/18513 Studies')
    legacy = json.loads(Path(legacy_plan).read_text())
    mapping = legacy['study_to_fold']
    validate_folds(raw_train, mapping)
    train = [r for r in raw_train if mapping[r['study_key']] != 4]
    dev = [r for r in raw_train if mapping[r['study_key']] == 4]
    if {r['image_id'] for r in train + dev} != {r['image_id'] for r in raw_train} or len(train)+len(dev) != len(raw_train):
        raise ValueError('Train/dev coverage mismatch')
    for split, old_sha in before.items():
        if digest(root/f'{split}.csv') != old_sha:
            raise ValueError(f'Original CSV changed while reading: {split}')
    payloads = {name+'.csv': csv_bytes(rows) for name,rows in
                [('source_train',raw_train), ('train',train), ('dev',dev), ('valid',valid)]}
    plan = dict(schema='original_unfiltered_fivefold_v1', source_manifest=str(output/'source_train.csv'),
                source_manifest_sha256=hashlib.sha256(payloads['source_train.csv']).hexdigest(),
                official_train_csv=str(root/'train.csv'), official_train_csv_sha256=before['train'],
                legacy_fold_plan=str(Path(legacy_plan).resolve()), legacy_fold_plan_sha256=digest(legacy_plan),
                n_splits=5, study_to_fold=mapping, train_folds=[0,1,2,3], dev_fold=4,
                all_original_images_retained=True, labels_changed=False, exclusions_applied=False,
                fold_assignment_changed=False, dev_role='development, not independent holdout',
                test_used_for_training_or_selection=False)
    payloads['folds.json'] = json_bytes(plan)
    overlaps = {f'{a}_{b}': {field:len({r[field] for r in data[a]} & {r[field] for r in data[b]})
                            for field in ('subject_id','study_key','dicom_id')}
                for a,b in [('train','valid'),('train','test'),('valid','test')]}
    report = dict(schema='original_data_integrity_and_structure_v1', original_root=str(root),
                  official_csv_sha256=before, datasets={s:summarize(r) for s,r in data.items()},
                  split_roles={'optimization':summarize(train), 'development':summarize(dev)},
                  original_split_identity_overlap=overlaps,
                  source_rows_preserved=True, excluded_rows=0, original_files_modified=False,
                  pixel_hash_definition='SHA256(big-endian uint32 oriented width,height + EXIF-oriented RGB bytes)',
                  purpose='Test is read only for integrity/structure audit; no test labels, training, model selection or fitted statistics.',
                  cost_example=audit_tar(cost_tar, test, root),
                  manifest_sha256={name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()})
    payloads['audit.json'] = json_bytes(report)
    for name,content in payloads.items():
        publish(output/name, content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default='/home/featurize/data/cxr_first')
    parser.add_argument('--legacy-plan', type=Path, default='/home/featurize/work/cloud_project_v2/data/cv5_overnight/folds.json')
    parser.add_argument('--output', type=Path, default='/home/featurize/data/raw_rebuild_20261105')
    parser.add_argument('--cost-tar', type=Path, default='/home/featurize/work/cost_data_test.tar')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({'output':str(args.output), 'datasets':result['datasets'], 'split_roles':result['split_roles']}, indent=2))


if __name__ == '__main__':
    main()
