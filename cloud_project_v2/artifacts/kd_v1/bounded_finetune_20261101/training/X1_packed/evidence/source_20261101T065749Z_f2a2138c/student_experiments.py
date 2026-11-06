"""Student-only objective and offline initialization; never modifies teachers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from kd_common import sha256_file, validate_pretrained_provenance


def load_local_pretrained_provenance(provenance_path: Path, weights_path: Path,
                                    backbone_name: str) -> dict[str, Any]:
    original = json.loads(Path(provenance_path).read_text(encoding='utf-8'))
    validate_pretrained_provenance(original, backbone_name)
    weights = Path(weights_path).resolve(strict=True)
    if sha256_file(weights) != original['weights_sha256']:
        raise ValueError('Offline original initialization SHA256 mismatch')
    value = copy.deepcopy(original)
    value['original_recorded_weights_path'] = original.get('weights_path')
    value['weights_path'] = str(weights)
    value['restored_from_provenance_sha256'] = sha256_file(provenance_path)
    return validate_pretrained_provenance(value, backbone_name)


def compute_student_loss_config(records, *, source_manifest_sha256: str,
                                fold=None, fold_plan_sha256=None,
                                pos_weight_power: float = .5) -> dict[str, Any]:
    if type(pos_weight_power) not in (int, float) or pos_weight_power not in (1., .5):
        raise ValueError('Only original or square-root Study positive weights are registered')
    from train_overnight import compute_loss_config
    original = compute_loss_config(records, source_manifest_sha256=source_manifest_sha256,
                                   fold=fold, fold_plan_sha256=fold_plan_sha256)
    if pos_weight_power == 1.:
        return original
    return {**original, 'original_pos_weight': list(original['pos_weight']),
            'pos_weight': [value ** .5 for value in original['pos_weight']],
            'pos_weight_power': .5,
            'pos_weight_formula': 'sqrt(negative_studies/positive_studies)'}
