"""Isolated ROI-only or spatial-MIL-only extensions of the audited Tiny trainer.

The complete official train, frozen OOF cache, hflip2 validation, optimizer,
EMA, and lifecycle hooks remain the existing implementation. No test input.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, fields
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

import dataset_overnight as data
import model_overnight as models
import train_distill as trainer
import train_overnight as common


@dataclass(frozen=True)
class ExplorationConfig(trainer.DistillConfig):
    roi: bool = False
    mil: bool = False

    def validate(self):
        super().validate()
        if self.roi and self.mil:
            raise ValueError('Each new run changes one package: ROI or MIL, not both')
        if self.image_size != 768 or self.validation_size != 768 or self.validation_tta != 'hflip2':
            raise ValueError('ROI/MIL controls require fixed 768 training/validation and hflip2')


class ROIDataset(data.OvernightImageDataset):
    def __getitem__(self, index):
        from roi_scale import normalize_roi
        from dataset import CachedSampleError
        record = self.records[index]
        try:
            path = self.paths[index].resolve()
            if not path.is_relative_to(self.raw_root / self.split):
                raise ValueError('Image path escaped original split')
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != record['file_sha256']:
                raise ValueError('Original image SHA256 mismatch')
            with Image.open(BytesIO(content)) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                if upright.mode in {'I', 'F'} or upright.mode.startswith('I;'):
                    raise ValueError('Unsupported source intensity policy')
                canvas, _ = normalize_roi(upright.convert('RGB'), self.image_size)
            pixels = np.array(canvas, dtype=np.uint8, copy=True)
            tensor = self.transform(image=pixels)['image'].to(dtype=torch.float32)
            if tensor.device.type != 'cpu' or tuple(tensor.shape) != (3,self.image_size,self.image_size):
                raise ValueError('Invalid ROI image tensor')
        except Exception as error:
            raise CachedSampleError(f'image_id={record["image_id"]}: {type(error).__name__}: {error}') from error
        return {'image': tensor, 'targets': torch.tensor(record['targets'], dtype=torch.float32),
            'image_id': record['image_id'], 'study_key': record['study_key'],
            'subject_id': record['subject_id'], 'study_id': record['study_id'],
            'split': self.split, 'loss_weight': self.loss_weights[index]}


def install_extensions(config):
    from roi_scale import VERSION as ROI_VERSION
    extension = {'roi': config.roi, 'mil': config.mil,
        'roi_version': ROI_VERSION if config.roi else None,
        'spatial_pooling': 'class_attention_v1' if config.mil else 'gap',
        'inference_size': 768, 'tta': 'hflip2'}
    original_capture = trainer.capture_source_version
    def capture(paths):
        extra = ['exploration_train.py', 'roi_scale.py', 'tight_crop.py', 'spatial_mil.py']
        return original_capture([*paths, *(Path(__file__).with_name(n) for n in extra)])
    trainer.capture_source_version = capture
    original_preprocessing = trainer.student_preprocessing_config
    def preprocessing(cfg):
        return {**original_preprocessing(cfg), 'experiment': extension}
    trainer.student_preprocessing_config = preprocessing
    original_save = common.save_checkpoint
    def save(path, payload):
        return original_save(path, {**payload, 'experiment': extension,
            'architecture_extension': extension['spatial_pooling']})
    common.save_checkpoint = save
    if config.roi:
        data.OvernightImageDataset = ROIDataset
    if config.mil:
        from spatial_mil import install_spatial_mil
        original_build = models.build_overnight_model
        def build(**kwargs):
            return install_spatial_mil(original_build(**kwargs))
        models.build_overnight_model = build
        original_initialize = trainer.initialize_model_from_student_ema
        def initialize(model, state):
            extra = set(model.state_dict()) - set(state)
            if extra != {'head.attention.weight'} or set(state)-set(model.state_dict()):
                raise ValueError('MIL warm start may add only zero-initialized attention weights')
            merged = {**state, 'head.attention.weight': torch.zeros_like(model.state_dict()['head.attention.weight'])}
            original_initialize(model, merged)
        trainer.initialize_model_from_student_ema = initialize
    return extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    value = json.loads(args.config.read_text())
    path_fields = {'fold_plan','soft_labels','output','image_root','resume','init_student_ema',
                   'pretrained_provenance_path','pretrained_weights_path'}
    cfg = {k: Path(v) if k in path_fields and v is not None else v for k,v in value.items()}
    if args.resume:
        cfg.update(resume=args.resume, init_student_ema=None, init_student_ema_sha256=None)
    config = ExplorationConfig(**cfg)
    config.validate()
    extension = install_extensions(config)
    if not args.execute:
        print(json.dumps({'dry_run': True, 'configuration': trainer._configuration(config), 'experiment': extension}, indent=2))
        return
    trainer.validate_output_paths(config)
    print(json.dumps(trainer.run_training(config), indent=2))


if __name__ == '__main__':
    main()
