"""Strict, explicit authorization for the 18-to-23 epoch continuation.

The parent checkpoint is never rewritten. All optimizer, scaler, EMA, RNG,
best-selection and early-stop state continues from the completed epoch 18.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def only_epoch_extension(old, new):
    expected = dict(old, epochs=23, lr_schedule_epochs=18)
    return old.get('epochs') == 18 and 'lr_schedule_epochs' not in old and new == expected


def validate_extension(prior, evidence, checkpoint_path, receipt_path):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    parent = prior['evidence']
    if receipt.get('schema') != 'authorized_raw_dino_18_to_23_v1':
        raise ValueError('Unrecognized epoch extension receipt')
    if receipt.get('authorization') != 'User explicitly requested five additional epochs, all other settings unchanged':
        raise ValueError('Missing explicit epoch extension authorization')
    if prior['epoch'] != 18 or receipt.get('completed_parent_epoch') != 18:
        raise ValueError('Extension requires the fully completed epoch 18 checkpoint')
    if not only_epoch_extension(parent['configuration'], evidence['configuration']):
        raise ValueError('Only epochs 18->23 with the original LR schedule may change')
    old_other = {k: v for k, v in parent.items() if k not in {'configuration', 'source_sha256'}}
    new_other = {k: v for k, v in evidence.items() if k not in {'configuration', 'source_sha256'}}
    if old_other != new_other:
        raise ValueError('Initialization, data usage or other evidence changed')
    old_sources, new_sources = parent['source_sha256'], evidence['source_sha256']
    if set(new_sources) != set(old_sources) | {'raw_rebuild/extension.py'}:
        raise ValueError('Unexpected source inventory change')
    changed = {k for k in old_sources if old_sources[k] != new_sources[k]}
    if changed != {'raw_rebuild/train.py', 'raw_rebuild/launch.py', 'breakthrough/train.py'}:
        raise ValueError('Unexpected training source changes')
    if identity(parent) != receipt.get('parent_evidence_sha256'):
        raise ValueError('Parent evidence differs from the authorized checkpoint')
    if identity(evidence) != receipt.get('new_evidence_sha256'):
        raise ValueError('New evidence differs from the reviewed extension')
    if hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() != receipt.get('parent_checkpoint_sha256'):
        raise ValueError('Parent checkpoint identity changed')
    return dict(receipt=receipt, receipt_path=str(receipt_path),
                receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                parent_configuration=parent['configuration'])


def compatible_best_configuration(candidate, current, lineage):
    if candidate == current:
        return True
    return (isinstance(lineage, dict)
            and candidate == lineage.get('parent_configuration')
            and only_epoch_extension(candidate, current))
