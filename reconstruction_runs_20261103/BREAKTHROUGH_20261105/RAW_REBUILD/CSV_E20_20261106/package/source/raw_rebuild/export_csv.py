"""Frozen epoch-20 DINO / fold-4 CNN export and complete-Study inference.

No training, test fitting, TTA, cross-Study pooling, or threshold filtering.
The same Predictor is used for development, submission and cost replay.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageOps
import torch

from raw_rebuild.prepare_data import load_official

MEAN = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
HEADER = ['Subject_id', 'Study_id', 'Predict_class', 'Probability']
MAX_BYTES = 500_000_000


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def export(package, dino_checkpoint, cnn_checkpoint):
    package.mkdir(parents=True, exist_ok=False)
    members = {}
    for name, source, field, epoch in [('dino', dino_checkpoint, 'ema', 20),
                                      ('cnn', cnn_checkpoint, 'model_state_dict', 12)]:
        original_sha = sha(source)
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        if checkpoint['epoch'] != epoch:
            raise ValueError(f'Wrong fixed epoch: {name}')
        state = checkpoint[field]
        # Autocast uses half-precision matrix/conv operands. Keep norms, biases,
        # layer scales and positional/CLS/register embeddings in source FP32.
        compressed = {key: value.half() if value.is_floating_point() and
                      value.ndim >= 2 and key.endswith('.weight') else value
                      for key, value in state.items()}
        destination = package / f'{name}.pt'
        torch.save(compressed, destination)
        members[name] = dict(file=destination.name, bytes=destination.stat().st_size,
                             sha256=sha(destination), source=str(source),
                             source_sha256=original_sha, epoch=epoch,
                             fp16_tensors=sum(v.dtype == torch.float16 for v in compressed.values()),
                             fp32_tensors=sum(v.dtype == torch.float32 for v in compressed.values()))
        if name == 'cnn':
            members[name]['model_config'] = dict(checkpoint['model_config'], gradient_checkpointing=False)
        del checkpoint, state, compressed
    config = dict(schema='raw_dino20_cnn50_v1', created=timestamp(), members=members,
                  class_ids=list(range(1, 11)), weights={'dino': .5, 'cnn': .5},
                  image_sizes={'dino': 518, 'cnn': 768}, image_chunk=2,
                  storage='FP16 matrix/conv weights; FP32 other tensors',
                  execution='FP32 master parameters; CUDA FP16 autocast',
                  views='all original images; identity only; complete Study probability mean',
                  cnn_history='Frozen historical teacher, not newly trained on raw data',
                  training_started=False, test_fitting=False, cross_study_pooling=False)
    write_json(package / 'config.json', config)
    total = sum(p.stat().st_size for p in package.iterdir() if p.is_file())
    if total > MAX_BYTES:
        raise ValueError(f'Model assets exceed 500MB: {total}')
    print(json.dumps(dict(exported=True, loaded_asset_bytes=total, package=str(package))), flush=True)


def group_records(records):
    groups, seen, owners = {}, set(), {}
    for row in records:
        key = row['subject_id'] + '/' + row['study_id']
        if row['study_key'] != key or row['dicom_id'] in seen:
            raise ValueError('Invalid identity or duplicate image')
        if owners.setdefault(row['study_id'], row['subject_id']) != row['subject_id']:
            raise ValueError('Study belongs to multiple patients')
        seen.add(row['dicom_id'])
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError('No complete Studies')
    return list(groups.values())


class Predictor:
    def __init__(self, package):
        from breakthrough.model import DinoStudyModel
        from model_overnight import build_overnight_model

        if not torch.cuda.is_available():
            raise RuntimeError('Frozen inference requires CUDA FP16')
        torch.set_num_threads(1)
        torch.manual_seed(20261105)
        torch.cuda.manual_seed_all(20261105)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        package = Path(package)
        self.config = json.loads((package / 'config.json').read_text())
        if self.config['schema'] != 'raw_dino20_cnn50_v1' or self.config['weights'] != {'dino': .5, 'cnn': .5}:
            raise ValueError('Unknown or altered frozen recipe')
        files = [package / self.config['members'][name]['file'] for name in ['dino', 'cnn']]
        self.loaded_asset_bytes = (package / 'config.json').stat().st_size + sum(p.stat().st_size for p in files)
        if self.loaded_asset_bytes > MAX_BYTES:
            raise ValueError('Loaded assets exceed 500MB')
        for name, path in zip(['dino', 'cnn'], files):
            info = self.config['members'][name]
            if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
                raise ValueError(f'Altered inference weights: {name}')
        self.dino = DinoStudyModel(None, image_size=518, gradient_checkpointing=False,
                                   view_chunk_size=2)
        self.cnn = build_overnight_model(pretrained=False, **self.config['members']['cnn']['model_config'])
        for model, path in zip([self.dino, self.cnn], files):
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.eval().requires_grad_(False).cuda()
        self.cnn.to(memory_format=torch.channels_last)
        self.local = threading.local()

    def prepare(self, records, image_root):
        from reconstruction.data import letterbox
        from dataset import letterbox_rgb
        from dataset_overnight import _transforms

        if not records or len({r['study_key'] for r in records}) != 1:
            raise ValueError('Predict one nonempty complete Study at a time')
        if not hasattr(self.local, 'cnn_transform'):
            self.local.cnn_transform = _transforms(False, 43)
        dino, cnn, hashes = [], [], []
        root = Path(image_root).resolve(strict=True)
        for row in records:
            path = (root / row['relative_path']).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('Image escaped declared image root')
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if row.get('file_sha256') and digest != row['file_sha256']:
                raise ValueError(f'Image SHA mismatch: {row["dicom_id"]}')
            with Image.open(io.BytesIO(data)) as original:
                original.load()
                if original.mode in {'I', 'F'} or original.mode.startswith('I;'):
                    raise ValueError(f'Unsupported original pixel mode: {path}')
                image = ImageOps.exif_transpose(original).convert('RGB')
                array = np.asarray(letterbox(image, 518), dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
                dino.append(torch.from_numpy(np.ascontiguousarray((array - MEAN) / STD)))
                cnn.append(self.local.cnn_transform(image=np.asarray(letterbox_rgb(image, 768)))['image'])
            hashes.append(dict(dicom_id=row['dicom_id'], relative_path=row['relative_path'],
                               file_sha256=digest, bytes=len(data)))
        return dict(dino=torch.stack(dino), cnn=torch.stack(cnn), images=hashes)

    @torch.inference_mode()
    def predict_prepared(self, prepared):
        with torch.autocast('cuda', dtype=torch.float16):
            dino_result = self.dino.forward_studies([prepared['dino'].cuda()], return_components=True)[0]
            dino_logits = dino_result['study_logits']
            if not torch.isfinite(dino_result['image_logits']).all() or not torch.isfinite(dino_logits).all():
                raise ValueError('Nonfinite DINO logits')
            dino = dino_logits.sigmoid().float().cpu().numpy().astype(np.float64)
            logits = torch.cat([self.cnn(chunk.cuda().contiguous(memory_format=torch.channels_last)).float()
                                for chunk in prepared['cnn'].split(2)], dim=0).cpu().numpy().astype(np.float64)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite CNN image logits')
        cnn = (1 / (1 + np.exp(-logits))).mean(axis=0)
        result = dict(dino=dino, cnn=cnn, blend=.5 * dino + .5 * cnn)
        if any(v.shape != (10,) or not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in result.values()):
            raise ValueError('Invalid model probability')
        return result

    def predict_study(self, records, image_root):
        return self.predict_prepared(self.prepare(records, image_root))


def infer(package, records, image_root, output, *, development=False, workers=6):
    output.mkdir(parents=True, exist_ok=False)
    groups = group_records(records)
    predictor = Predictor(package)
    values = {name: [] for name in ['dino', 'cnn', 'blend']}
    images, keys, patients = [], [], []
    start = time.monotonic()
    streams = {name: (output / f'{name}.csv.partial').open('x', newline='') for name in ['dino', 'blend']}
    writers = {name: csv.writer(stream, lineterminator='\n') for name, stream in streams.items()}
    for writer in writers.values():
        writer.writerow(HEADER)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending, iterator = deque(), iter(groups)
            for _ in range(min(12, len(groups))):
                group = next(iterator)
                pending.append((group, pool.submit(predictor.prepare, group, image_root)))
            while pending:
                group, future = pending.popleft()
                prepared = future.result()
                prediction = predictor.predict_prepared(prepared)
                keys.append(group[0]['study_key']); patients.append(group[0]['subject_id'])
                images.extend(prepared['images'])
                for name, probability in prediction.items():
                    values[name].append(probability)
                    if name in writers:
                        for category, value in enumerate(probability, 1):
                            writers[name].writerow([group[0]['subject_id'], group[0]['study_id'], category, repr(float(value))])
                following = next(iterator, None)
                if following is not None:
                    pending.append((following, pool.submit(predictor.prepare, following, image_root)))
                if len(keys) % 25 == 0 or len(keys) == len(groups):
                    progress = dict(time=timestamp(), phase='development' if development else 'test',
                                    studies_completed=len(keys), total_studies=len(groups),
                                    images_completed=len(images), total_images=len(records),
                                    elapsed_seconds=time.monotonic()-start, training=False)
                    write_json(output / 'progress.json', progress)
                    if len(keys) % 100 == 0 or len(keys) == len(groups):
                        print(json.dumps(progress), flush=True)
        if len(images) != len(records):
            raise ValueError('Incomplete image coverage')
    finally:
        for stream in streams.values():
            stream.close()
    for name in streams:
        (output / f'{name}.csv.partial').rename(output / f'{name}.csv')
    reports = {}
    for name in values:
        array = np.stack(values[name])
        payload = dict(probabilities=array, study_keys=np.asarray(keys), patient_ids=np.asarray(patients),
                       class_ids=np.arange(1, 11), image_counts=np.asarray([len(g) for g in groups]))
        if development:
            from reconstruction.evaluate import evaluate_studies
            targets = np.asarray([json.loads(g[0]['targets']) for g in groups])
            views = [[r['view_position'] for r in g] for g in groups]
            payload.update(targets=targets, views_json=np.asarray([json.dumps(v) for v in views]))
            reports[name] = evaluate_studies(array, targets, keys, patients, views, bootstrap_repeats=0)
        np.savez_compressed(output / f'{name}.npz', **payload)
    write_json(output / 'images.json', images)
    report = dict(completed=True, time=timestamp(), studies=len(groups), images=len(images),
                  patients=len(set(patients)), rows=len(groups)*10, elapsed_seconds=time.monotonic()-start,
                  loaded_asset_bytes=predictor.loaded_asset_bytes, package_config_sha256=sha(package/'config.json'),
                  cuda_device=torch.cuda.get_device_name(), metrics=reports, training_started=False,
                  csv_sha256={name: sha(output/f'{name}.csv') for name in streams})
    write_json(output / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}), flush=True)
    if reports:
        print(json.dumps({name: r['macro_auc'] for name, r in reports.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    item = sub.add_parser('export')
    item.add_argument('--package', required=True, type=Path)
    item.add_argument('--dino-checkpoint', required=True, type=Path)
    item.add_argument('--cnn-checkpoint', required=True, type=Path)
    item = sub.add_parser('infer')
    item.add_argument('--package', required=True, type=Path)
    item.add_argument('--image-root', required=True, type=Path)
    source = item.add_mutually_exclusive_group(required=True)
    source.add_argument('--dev-manifest', type=Path)
    source.add_argument('--official-test', action='store_true')
    item.add_argument('--output', required=True, type=Path)
    item.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.action == 'export':
        export(args.package, args.dino_checkpoint, args.cnn_checkpoint)
    else:
        if args.workers < 1:
            parser.error('workers must be positive')
        if args.dev_manifest:
            with args.dev_manifest.open(newline='') as stream:
                records = list(csv.DictReader(stream))
            if any(r['split'] != 'train' for r in records):
                raise ValueError('Development must use original train split')
        else:
            records = load_official(args.image_root, 'test')
        infer(args.package, records, args.image_root, args.output,
              development=bool(args.dev_manifest), workers=args.workers)


if __name__ == '__main__':
    main()
