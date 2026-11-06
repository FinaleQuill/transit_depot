"""Explicit fixed Tiny/Base pair selected on valid; original per-image math.

Class 1 may use B at 75% or 100%; all other classes retain equal probability
weights. The two candidates are registered before this loader runs. Neither
training nor any test-derived choice occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from deploy_kd import (BASE_BACKBONE, TINY_BACKBONE, CLASS_IDS, MAX_MODEL_BYTES,
                       MEAN, STD, Predictor, normalized_model_weights, sha256_file)
from student_experiment_evaluation import ExperimentalPredictor, write_new_json


def registered_weights(plan, candidate):
    plan = json.loads(Path(plan).read_text())
    choice = plan['candidates'][candidate]
    a, b = choice['A_class1_weight'], choice['B_class1_weight']
    if ((a, b) not in ((.25, .75), (0., 1.))
            or choice['other_classes_A_B_weights'] != [.5, .5]
            or plan['validation_selected_hyperparameters'] is not True
            or plan['validation_fitted_class_weights'] is not True
            or plan['test_used'] is not False):
        raise ValueError('Require an explicitly validation-selected registered fixed pair')
    return [[a] + [.5] * 9, [b] + [.5] * 9]


def make_pair(models, output, *, plan, candidate, source_receipt):
    models, output = [Path(p).resolve(strict=True) for p in models], Path(output)
    if len(models) != 2 or output.exists():
        raise ValueError('Require two restored exports and a new output directory')
    weights = registered_weights(plan, candidate)
    identities = json.loads(Path(source_receipt).read_text())['models']
    checked = []
    for name, root, backbone, identity in zip(('A', 'B'), models,
            (TINY_BACKBONE, BASE_BACKBONE), identities):
        config = json.loads((root / 'config.json').read_text())
        if (identity['label'] != name or config['backbone_name'] != backbone
                or config['artifact_type'] != 'experimental_student_fp16'
                or config['tta'] != 'hflip2' or config['image_size'] != 768
                or config['source_checkpoint_sha256'] != identity['source_checkpoint_sha256']
                or config['source_epoch'] != identity['source_epoch']
                or config['weights_sha256'] != sha256_file(root / config['weights_file'])):
            raise ValueError('Pair member differs from the original audited A/B identity')
        checked.append((name, root, config))
    if len({c['official_valid_manifest_sha256'] for _, _, c in checked}) != 1:
        raise ValueError('Pair members must share the untouched original valid')
    output.mkdir(parents=True)
    members = []
    for name, root, config in checked:
        destination = output / name
        destination.mkdir()
        for filename in ('config.json', config['weights_file']):
            shutil.copyfile(root / filename, destination / filename)
            if sha256_file(root / filename) != sha256_file(destination / filename):
                raise RuntimeError('Copied member bytes differ')
        members.append({'directory': name, 'backbone': config['backbone_name'],
                        'config_sha256': sha256_file(destination / 'config.json'),
                        'weights_sha256': config['weights_sha256']})
    config = {'schema_version': 1, 'artifact_type': 'fixed_classwise_student_pair',
        'class_ids': CLASS_IDS, 'precision': 'float16', 'batch_size': 1,
        'image_size': 768, 'tta': 'hflip2', 'ensemble': True, 'gates': None,
        'source_members': members, 'model_weights': weights, 'candidate': candidate,
        'official_valid_manifest_sha256': checked[0][2]['official_valid_manifest_sha256'],
        'selection_plan_sha256': sha256_file(plan),
        'selection_plan': json.loads(Path(plan).read_text()),
        'validation_selected_hyperparameters': True, 'validation_fitted_class_weights': True,
        'test_used_for_selection': False, 'formal_T4_certification': False,
        'aggregation': 'fixed class-wise image probability blend, then original Study probability mean'}
    write_new_json(output / 'config.json', config)
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > MAX_MODEL_BYTES:
        raise ValueError('Actual pair inference assets exceed the size limit')
    return {'total_inference_bytes': total, 'size_pass': True, 'model_weights': weights}


class FixedStudentPair(Predictor):
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir).resolve(strict=True)
        config = json.loads((self.model_dir / 'config.json').read_text())
        if (config.get('artifact_type') != 'fixed_classwise_student_pair'
                or config['class_ids'] != CLASS_IDS or config['precision'] != 'float16'
                or config['batch_size'] != 1 or config['tta'] != 'hflip2'
                or config['image_size'] != 768 or len(config['source_members']) != 2):
            raise ValueError('Malformed fixed pair')
        selection = config['selection_plan']
        choice = selection['candidates'][config['candidate']]
        expected = [[choice['A_class1_weight']] + [.5] * 9,
                    [choice['B_class1_weight']] + [.5] * 9]
        if (choice['B_class1_weight'] not in (.75, 1.)
                or choice['A_class1_weight'] + choice['B_class1_weight'] != 1.
                or choice['other_classes_A_B_weights'] != [.5, .5]
                or selection['test_used'] is not False
                or config['model_weights'] != expected):
            raise ValueError('Frozen class weights differ from their declared plan')
        self.models, self.members = [], []
        for entry, backbone in zip(config['source_members'], (TINY_BACKBONE, BASE_BACKBONE)):
            source = (self.model_dir / entry['directory']).resolve(strict=True)
            if (source.parent != self.model_dir
                    or sha256_file(source / 'config.json') != entry['config_sha256']):
                raise ValueError('Fixed pair member config changed')
            loaded = ExperimentalPredictor(source, tta='hflip2')
            if (loaded.config['backbone_name'] != backbone
                    or loaded.config['weights_sha256'] != entry['weights_sha256']
                    or loaded.config['official_valid_manifest_sha256'] != config['official_valid_manifest_sha256']):
                raise ValueError('Fixed pair member identity changed')
            self.torch, self.device = loaded.torch, loaded.device
            self.models.append(loaded.model)
            self.members.append(loaded.members[0])
        from inference_inputs import ExactU8Input
        self.model = self.models[0]
        self.model_weights = normalized_model_weights(config['model_weights'], 2).tolist()
        self.config = {**config, 'members': self.members}
        self.input_lookup = ExactU8Input(self.torch, self.device, MEAN, STD, image_sizes=[768])
        files = [{'path': str(p.relative_to(self.model_dir)), 'bytes': p.stat().st_size,
                  'sha256': sha256_file(p)} for p in sorted(self.model_dir.rglob('*')) if p.is_file()]
        total = sum(item['bytes'] for item in files)
        self.inventory = {'files': files, 'total_bytes': total,
                          'limit_bytes': MAX_MODEL_BYTES, 'size_pass': total <= MAX_MODEL_BYTES}
        if not self.inventory['size_pass']:
            raise ValueError('Loaded pair inference assets exceed the size limit')
