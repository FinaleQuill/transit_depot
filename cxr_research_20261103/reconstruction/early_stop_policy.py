"""Explicit E2 budget amendment; independent of training and model frameworks.

The original six-epoch learning-rate schedule and highest-score checkpoint
selection remain unchanged. This policy only decides when a complete arm has
used its amended training budget. Epoch zero is a baseline, not a bad epoch.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = 'cxr_e2_early_stopping_v1'
POLICY = {
    'schema': SCHEMA, 'policy_id': 'e2_patience3_20261104',
    'patience': 3, 'min_delta': 0.0001, 'max_epochs': 6,
    'metric': 'macro_auc', 'mode': 'max', 'baseline_epoch': 0,
    'applies_to_arms': ['global_control', 'global_local'],
}


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        raise ValueError('Stopping policy has missing or unregistered fields')
    for key, expected in POLICY.items():
        if type(policy[key]) is not type(expected) or policy[key] != expected:
            raise ValueError(f'Unregistered stopping-policy setting: {key}')
    return policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path, expected_sha=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Stopping policy must be a regular immutable file')
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError('Stopping policy must be a file')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and (
            not isinstance(expected_sha, str)
            or re.fullmatch(r'[0-9a-f]{64}', expected_sha) is None
            or digest != expected_sha):
        raise ValueError('Stopping policy differs from the bound SHA256')
    policy = validate_policy(json.loads(content))
    return {'path': str(path), 'sha256': digest, 'policy': deepcopy(policy)}


def verify_binding(binding):
    if load_policy(binding['path'], binding['sha256']) != binding:
        raise ValueError('Stopping policy content changed after binding')


def _auc(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Stopping requires finite full-development AUC in [0, 1]')
    return float(value)


def decision(baseline_auc, history, policy):
    """Replay a complete history, rejecting any training beyond the first stop.

    Patience resets only above the last significant best plus min_delta. The
    actual highest score is tracked separately, exactly as the frozen trainer
    selects best.pt. Several individually tiny improvements can therefore
    accumulate into a significant improvement before patience is exhausted.
    """
    validate_policy(policy)
    if not isinstance(history, list) or len(history) > policy['max_epochs']:
        raise ValueError('Invalid or over-budget complete history')
    baseline = _auc(baseline_auc)
    best, best_epoch = baseline, 0
    significant_best, significant_best_epoch = baseline, 0
    bad_epochs, first_stop_epoch, stop_reason = 0, None, None
    for epoch, row in enumerate(history, 1):
        if first_stop_epoch is not None:
            raise ValueError(f'History exceeds first policy stop at epoch {first_stop_epoch}')
        if not isinstance(row, dict) or type(row.get('epoch')) is not int or row['epoch'] != epoch:
            raise ValueError('History must contain consecutive complete epochs from one')
        score = _auc(row.get(policy['metric']))
        if score > best:
            best, best_epoch = score, epoch
        if score > significant_best + policy['min_delta']:
            significant_best, significant_best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
        if epoch == policy['max_epochs']:
            first_stop_epoch, stop_reason = epoch, 'max_epochs_reached'
        elif bad_epochs >= policy['patience']:
            first_stop_epoch, stop_reason = epoch, 'patience_exhausted'
    return {
        'completed_epochs': len(history), 'bad_epochs': bad_epochs,
        'best_macro_auc': best, 'best_epoch': best_epoch,
        'significant_best_macro_auc': significant_best,
        'significant_best_epoch': significant_best_epoch,
        'stop': first_stop_epoch is not None, 'stop_reason': stop_reason,
        'first_stop_epoch': first_stop_epoch,
    }
