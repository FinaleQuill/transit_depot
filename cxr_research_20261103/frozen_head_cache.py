"""Cache frozen train/valid classifier inputs with the exact inference pixels.

Only the unchanged official train/valid manifests are accepted. No test split,
normalization fitting, parameter update, or validation-driven choice occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import read_manifest
from deploy_kd import MEAN, STD, _letterbox_image, sha256_file
from inference_inputs import normalization_table, load_image_pixels
from inference_reproducibility import configure_inference
from student_experiment_evaluation import ExperimentalPredictor


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class ExactFeatureImages(Dataset):
    def __init__(self, records, root):
        self.records, self.root = records, Path(root).resolve(strict=True)
        splits = {r['split'] for r in records}
        if len(splits) != 1 or not splits <= {'train', 'valid'}:
            raise ValueError('Frozen feature cache accepts original train or valid only')
        self.split = next(iter(splits))
        self.table = normalization_table(MEAN, STD).astype(np.float16)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != self.split:
            raise ValueError('Feature image escaped its original split')
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root / self.split):
            raise ValueError('Feature image escaped the read-only image root')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['file_sha256']:
            raise ValueError('Original feature image SHA256 mismatch: ' + row['image_id'])
        with Image.open(BytesIO(data)) as source:
            source.load()
            ImageOps.exif_transpose(source, in_place=True)
            if source.mode in {'I', 'F'} or source.mode.startswith('I;'):
                raise ValueError('Non-8-bit image has no registered intensity policy')
            image = source if source.mode == 'L' else source.convert('RGB')
            pixels = np.asarray(_letterbox_image(image, 768, 'L' if image.mode == 'L' else 'RGB'))
            if pixels.ndim == 2:
                normalized = self.table[:, pixels]
            else:
                normalized = np.stack([self.table[c, pixels[:, :, c]] for c in range(3)])
        return index, torch.from_numpy(np.ascontiguousarray(normalized))


def cache_worker_init(_worker):
    import fast_lanczos
    torch.set_num_threads(1)
    # Parallelize across images; retain the identical integer resampling output.
    fast_lanczos.default_threads = lambda: 1


def extract(*, plan_path, plan_sha256, output, image_root):
    if sha256_file(plan_path) != plan_sha256:
        raise ValueError('Frozen-head plan identity changed')
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output).resolve()
    if not output.is_relative_to(Path('/tmp')) or output.exists():
        raise ValueError('Feature cache requires a fresh local /tmp directory')
    output.mkdir(parents=True)
    configure_inference()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    root = Path(__file__).resolve().parent
    metadata = {'schema': 'frozen_fc_cache_v1', 'plan_sha256': plan_sha256,
        'started_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        'feature_contract': plan['cache']['feature_contract'], 'test_used': False,
        'complete': False, 'encoders': {}, 'splits': {}, 'source_sha256': sha256_file(__file__)}
    predictors, captured, hooks = {}, {}, []
    for name, declaration in plan['encoders'].items():
        base = Path(declaration['base_model_dir'])
        if sha256_file(base / 'config.json') != declaration['base_config_sha256']:
            raise ValueError('Frozen encoder config changed')
        predictor = ExperimentalPredictor(base, tta='hflip2')
        if predictor.config['weights_sha256'] != declaration['base_weights_sha256']:
            raise ValueError('Frozen encoder weight identity changed')
        predictor.model.eval().requires_grad_(False)
        predictors[name] = predictor
        def capture(_module, args, *, label=name):
            captured[label] = args[0]
        hooks.append(predictor.model.head.fc.register_forward_pre_hook(capture))
        head = {'head.fc.weight': predictor.model.head.fc.weight.detach().float().cpu(),
                'head.fc.bias': predictor.model.head.fc.bias.detach().float().cpu()}
        file = output / f'{name}_head_fp32.pt'
        torch.save(head, file)
        metadata['encoders'][name] = {**declaration, 'head_file': file.name,
            'head_sha256': sha256_file(file)}
    write_json(output / 'meta.json', metadata)
    started = time.monotonic()
    deadline = datetime.fromisoformat(plan['deadline_hong_kong']).timestamp() - 3 * 3600
    maximum = plan['budget']['cache_seconds_max']
    try:
        for split, manifest_name in [('train', 'train_clean.csv'), ('valid', 'valid_manifest.csv')]:
            manifest = root / 'data/clean_v1_reviewed' / manifest_name
            if sha256_file(manifest) != plan['cache'][split + '_manifest_sha256']:
                raise ValueError('Original feature manifest changed')
            records = read_manifest(manifest, split=split)
            directory = output / split
            directory.mkdir()
            record_file = directory / 'records.jsonl'
            with record_file.open('x') as stream:
                for row in records:
                    stream.write(json.dumps({key: row[key] for key in
                        ('image_id', 'study_key', 'subject_id', 'targets', 'split')}) + '\n')
            features, probabilities = {}, {}
            for name, declaration in plan['encoders'].items():
                features[name] = np.lib.format.open_memmap(directory / f'{name}_features.npy',
                    mode='w+', dtype=np.float16, shape=(len(records), 2, declaration['feature_dim']))
                probabilities[name] = np.lib.format.open_memmap(directory / f'{name}_base_probabilities.npy',
                    mode='w+', dtype=np.float32, shape=(len(records), 10))
            dataset = ExactFeatureImages(records, image_root)
            # Verify CPU half-table normalization against the real GPU lookup.
            _, sample = dataset[0]
            pixels = load_image_pixels(Path(image_root) / records[0]['relative_path'], [768], _letterbox_image)[768]
            lookup = next(iter(predictors.values())).input_lookup.to_tensor(pixels)
            if not torch.equal(sample.unsqueeze(0).to('cuda', memory_format=torch.channels_last), lookup):
                raise ValueError('Cached input normalization differs from production FP16 lookup')
            workers = plan['cache'][split + '_workers']
            options = dict(batch_size=plan['cache'][split + '_batch'], shuffle=False,
                num_workers=workers, pin_memory=True, worker_init_fn=cache_worker_init)
            if workers:
                options.update(prefetch_factor=2, multiprocessing_context='spawn', timeout=180)
            loader = DataLoader(dataset, **options)
            offset = 0
            with torch.inference_mode():
                for indices, images in loader:
                    if time.monotonic() - started > maximum or time.time() >= deadline:
                        raise TimeoutError('Frozen feature cache reached its registered time budget')
                    if indices.tolist() != list(range(offset, offset + len(indices))):
                        raise ValueError('Feature loader changed the original manifest order')
                    images = images.to('cuda', memory_format=torch.channels_last, non_blocking=True)
                    end = offset + len(indices)
                    per_model = {name: [] for name in predictors}
                    for view in range(2):
                        transformed = images if view == 0 else torch.flip(images, (-1,)).contiguous(memory_format=torch.channels_last)
                        for name, predictor in predictors.items():
                            logits = predictor.model(transformed)
                            value = captured.pop(name)
                            if (value.shape != (len(indices), plan['encoders'][name]['feature_dim'])
                                    or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(logits).all())):
                                raise FloatingPointError('Frozen feature/logit shape or finiteness differs')
                            features[name][offset:end, view] = value.cpu().numpy()
                            per_model[name].append(torch.sigmoid(logits.float()))
                    for name in predictors:
                        probabilities[name][offset:end] = torch.stack(per_model[name]).mean(0).cpu().numpy()
                    offset = end
                    if offset == len(records) or offset % (512 if split == 'train' else 100) < len(indices):
                        print(f'[FEATURES] {split} {offset}/{len(records)} seconds={time.monotonic()-started:.1f}', flush=True)
            if offset != len(records):
                raise RuntimeError('Incomplete original image coverage')
            split_metadata = {'records_file': str(record_file.relative_to(output)),
                'records_sha256': sha256_file(record_file), 'manifest_sha256': sha256_file(manifest),
                'num_images': len(records), 'num_studies': len({r['study_key'] for r in records}),
                'features': {}, 'base_probabilities': {}}
            for name in predictors:
                features[name].flush()
                probabilities[name].flush()
                file = directory / f'{name}_features.npy'
                split_metadata['features'][name] = {'file': str(file.relative_to(output)),
                    'sha256': sha256_file(file), 'shape': list(features[name].shape), 'dtype': 'float16'}
                probability_file = directory / f'{name}_base_probabilities.npy'
                split_metadata['base_probabilities'][name] = {'file': str(probability_file.relative_to(output)),
                    'sha256': sha256_file(probability_file), 'shape': list(probabilities[name].shape), 'dtype': 'float32'}
                if split == 'valid':
                    from evaluate_finetune_candidates import load_validation
                    from student_experiment_evaluation import validation_labels
                    labels, counts = validation_labels(records)
                    reference = root / f'artifacts/kd_v1/student_upper_bound_20261030/reports/{name}/valid_hflip2'
                    original, _, _, identity = load_validation(reference, records,
                        plan['cache']['valid_manifest_sha256'], labels, counts)
                    maximum_error = max(float(np.max(np.abs(probabilities[name][i].astype(np.float64)
                        - original[row['image_id']]))) for i, row in enumerate(records))
                    if maximum_error != 0:
                        raise RuntimeError(f'Frozen {name} cache forward differs from production baseline: {maximum_error}')
                    split_metadata.setdefault('baseline_exact_reproduction', {})[name] = {
                        'all_image_probabilities_bitwise_equal': True, 'max_absolute_error': maximum_error,
                        'source_validation': identity}
            metadata['splits'][split] = split_metadata
            write_json(output / 'meta.json', metadata)
            del loader, features, probabilities
        metadata.update(complete=True, seconds=time.monotonic()-started,
            finished_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat())
        write_json(output / 'meta.json', metadata)
        print('[CACHE_COMPLETE]', metadata['seconds'], flush=True)
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan-path', 'plan-sha256', 'output', 'image-root'):
        parser.add_argument('--' + name, type=str if name == 'plan-sha256' else Path, required=True)
    extract(**vars(parser.parse_args()))
