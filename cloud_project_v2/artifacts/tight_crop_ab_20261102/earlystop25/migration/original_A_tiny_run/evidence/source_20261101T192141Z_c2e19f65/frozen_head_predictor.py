"""A fixed FP16 classifier patch over an immutable experimental Tiny/Base export.

Only the declared classifier rows are replaced at model initialization. Image
decoding, preprocessing, forward order and hflip probability averaging are the
inherited production implementation; no extra encoder forward is introduced.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import tempfile

from deploy_kd import CLASS_IDS, MAX_MODEL_BYTES, sha256_file
from student_experiment_evaluation import ExperimentalPredictor


BASE_BACKBONE = 'convnext_base.fb_in22k_ft_in1k_384'
TINY_BACKBONE = 'convnext_tiny.fb_in22k_ft_in1k_384'
FEATURE_DIMS = {BASE_BACKBONE: 1024, TINY_BACKBONE: 768}
PATCH_KEYS = {'head.fc.weight', 'head.fc.bias'}


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def validate_patch_config(config):
    """Metadata validation is CPU-only and does not open model weights."""
    classes = config.get('replaced_class_ids')
    if (config.get('artifact_type') != 'frozen_head_student_fp16'
            or config.get('schema_version') != 1
            or config.get('backbone_name') not in FEATURE_DIMS
            or config.get('class_ids') != CLASS_IDS
            or config.get('precision') != 'float16'
            or config.get('image_size') != 768
            or type(config.get('feature_dim')) is not int
            or config.get('feature_dim') != FEATURE_DIMS.get(config.get('backbone_name'))
            or config.get('tta') != 'hflip2'
            or not isinstance(classes, list) or not classes
            or any(type(item) is not int or item not in CLASS_IDS for item in classes)
            or classes != sorted(set(classes))):
        raise ValueError('Malformed fixed Tiny/Base768 FP16 classifier patch configuration')
    name = config.get('patch_file')
    if (not isinstance(name, str) or Path(name).name != name
            or name in {'', '.', '..', 'config.json'}):
        raise ValueError('Patch file must be a plain filename inside model-dir')
    base = config.get('base_model_dir')
    if not isinstance(base, str) or not base:
        raise ValueError('The immutable base model directory must be declared')
    for field in ('base_config_sha256', 'base_weights_sha256', 'patch_sha256'):
        if not _sha(config.get(field)):
            raise ValueError('Missing or malformed patch source identity: ' + field)


def apply_classifier_patch(model, patch, *, class_ids, feature_dim=1024):
    """Validate all tensors before any mutation; support exact original-head patches."""
    import torch
    if (not isinstance(class_ids, list) or not class_ids
            or any(type(item) is not int or item not in CLASS_IDS for item in class_ids)
            or class_ids != sorted(set(class_ids))):
        raise ValueError('Classifier rows must be unique sorted original class IDs')
    fc = model.head.fc
    if (not isinstance(fc, torch.nn.Linear) or fc.in_features != feature_dim
            or fc.out_features != 10 or fc.bias is None
            or fc.weight.dtype != torch.float16 or fc.bias.dtype != torch.float16):
        raise ValueError('Frozen model classifier is not the declared ten-class FP16 Linear')
    if not isinstance(patch, dict) or set(patch) != PATCH_KEYS:
        raise ValueError('A classifier patch must contain exactly head.fc.weight/head.fc.bias')
    shapes = {'head.fc.weight': (len(class_ids), feature_dim), 'head.fc.bias': (len(class_ids),)}
    for name, shape in shapes.items():
        value = patch[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float16 or tuple(value.shape) != shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Classifier patch tensor shape, dtype or finiteness differs: ' + name)
    indices = torch.tensor([item - 1 for item in class_ids], dtype=torch.long, device=fc.weight.device)
    with torch.no_grad():
        fc.weight.index_copy_(0, indices, patch['head.fc.weight'].to(device=fc.weight.device))
        fc.bias.index_copy_(0, indices, patch['head.fc.bias'].to(device=fc.bias.device))
    return model


class FrozenHeadPredictor(ExperimentalPredictor):
    """Reuse one frozen encoder and the unmodified per-image prediction math."""

    def __init__(self, model_dir, *, device='cuda'):
        root = Path(model_dir).resolve(strict=True)
        config_path = root / 'config.json'
        config_sha = sha256_file(config_path)
        config = json.loads(config_path.read_text())
        validate_patch_config(config)
        base = Path(config['base_model_dir'])
        base = (root / base).resolve(strict=True) if not base.is_absolute() else base.resolve(strict=True)
        if base == root:
            raise ValueError('The immutable base export and classifier patch must be separate directories')
        base_config_path = base / 'config.json'
        if sha256_file(base_config_path) != config['base_config_sha256']:
            raise ValueError('Immutable base config SHA256 differs')
        base_config = json.loads(base_config_path.read_text())
        if (base_config.get('backbone_name') != config['backbone_name']
                or base_config.get('image_size') != 768
                or base_config.get('weights_sha256') != config['base_weights_sha256']):
            raise ValueError('Frozen classifier source must be the declared Tiny/Base768 export')
        weights = base / base_config['weights_file']
        if (not isinstance(base_config.get('weights_file'), str)
                or Path(base_config['weights_file']).name != base_config['weights_file']
                or weights.is_symlink() or not weights.is_file()
                or weights.resolve().parent != base):
            raise ValueError('Base weight file must be a regular file in its export directory')
        weights = weights.resolve(strict=True)
        patch_path = root / config['patch_file']
        if (patch_path.is_symlink() or not patch_path.is_file()
                or patch_path.resolve().parent != root
                or sha256_file(patch_path) != config['patch_sha256']):
            raise ValueError('Classifier patch SHA256 or file location differs')
        for field in ('source_manifest_sha256', 'fold_plan_sha256', 'official_valid_manifest_sha256'):
            if field in config and config[field] != base_config.get(field):
                raise ValueError('Classifier patch changed its original train/valid identity: ' + field)
        paths = {path.resolve() for path in (config_path, patch_path, base_config_path, weights)}
        files = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                 for path in sorted(paths)]
        total = sum(item['bytes'] for item in files)
        if total > MAX_MODEL_BYTES:
            raise ValueError('Frozen encoder, classifier patch and all required configs exceed 500MB')
        super().__init__(base, tta='hflip2', device=device)
        patch = self.torch.load(patch_path, map_location='cpu', weights_only=True)
        apply_classifier_patch(self.model, patch, class_ids=config['replaced_class_ids'],
                               feature_dim=config['feature_dim'])
        # Recheck every loaded immutable input after model initialization.
        if (sha256_file(config_path) != config_sha
                or sha256_file(base_config_path) != config['base_config_sha256']
                or sha256_file(weights) != config['base_weights_sha256']
                or sha256_file(patch_path) != config['patch_sha256']):
            raise RuntimeError('Frozen classifier inputs changed during loading')
        self.model_dir = root
        self.members = [{**self.members[0], 'classifier_patch_sha256': config['patch_sha256'],
                         'replaced_class_ids': list(config['replaced_class_ids'])}]
        self.config = {**base_config, **config, 'members': self.members, 'ensemble': False,
                       'source_base_config': base_config,
                       'source_checkpoint_sha256': config['patch_sha256'],
                       'source_patch_checkpoint_sha256': config['patch_sha256'],
                       'source_parent_checkpoint_sha256': base_config.get('source_checkpoint_sha256'),
                       'source_parent_weights_sha256': config['base_weights_sha256'],
                       'source_parent_config_sha256': config['base_config_sha256'],
                       'effective_weights_identity': {'encoder': config['base_weights_sha256'],
                                                      'classifier': config['patch_sha256']}}
        self.inventory = {'files': files, 'total_bytes': total, 'limit_bytes': MAX_MODEL_BYTES,
                          'size_pass': True, 'unique_encoder_files': 1}


def export_merged_for_validation(config_path, output):
    """Export an audited base+patch into a new /tmp-only standard FP16 export.

    Existing evaluation/ensemble loaders require a complete weight state. This
    temporary representation is tensor-identical to FrozenHeadPredictor; the
    persistent deployment still needs only the original encoder and tiny patch.
    No image, model forward, GPU, original file write, or candidate selection is
    performed here.
    """
    import torch
    from model_overnight import validate_model_config

    config_path = Path(config_path).resolve(strict=True)
    root = config_path.parent
    output = Path(output).absolute()
    if not output.resolve().is_relative_to(Path('/tmp').resolve()):
        raise ValueError('Merged validation weights must remain in /tmp, outside persistent work storage')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Merged validation export requires a new output directory')
    config_sha = sha256_file(config_path)
    config = json.loads(config_path.read_text())
    validate_patch_config(config)
    base = Path(config['base_model_dir'])
    base = (root / base).resolve(strict=True) if not base.is_absolute() else base.resolve(strict=True)
    if base == root:
        raise ValueError('Classifier patch and immutable base must be separate directories')
    base_config_path = base / 'config.json'
    if sha256_file(base_config_path) != config['base_config_sha256']:
        raise ValueError('Immutable base config SHA256 differs')
    base_config = json.loads(base_config_path.read_text())
    model_config = validate_model_config(base_config['model_config'])
    if (base_config.get('artifact_type') != 'experimental_student_fp16'
            or base_config.get('backbone_name') != config['backbone_name']
            or model_config['backbone_name'] != config['backbone_name']
            or base_config.get('class_ids') != CLASS_IDS
            or base_config.get('image_size') != 768 or base_config.get('precision') != 'float16'
            or base_config.get('weights_sha256') != config['base_weights_sha256']):
        raise ValueError('Merged source must be the declared standard Tiny/Base FP16 export')
    weight_name = base_config.get('weights_file')
    if not isinstance(weight_name, str) or Path(weight_name).name != weight_name:
        raise ValueError('Base weights must use a plain filename')
    weights_path, patch_path = base / weight_name, root / config['patch_file']
    for path, expected_sha in ((weights_path, config['base_weights_sha256']),
                               (patch_path, config['patch_sha256'])):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError('Merged export source file location or SHA256 differs: ' + str(path))
    for field in ('source_manifest_sha256', 'fold_plan_sha256', 'official_valid_manifest_sha256'):
        if field in config and config[field] != base_config.get(field):
            raise ValueError('Classifier patch changed original data identity: ' + field)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    patch = torch.load(patch_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not PATCH_KEYS.issubset(state):
        raise ValueError('Base export must contain the complete head.fc tensor state')
    for name, value in state.items():
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.device.type != 'cpu' or value.is_complex()
                or value.is_floating_point() and (
                    value.dtype != torch.float16 or not bool(torch.isfinite(value).all()))):
            raise ValueError('Base export must contain finite CPU FP16 state tensors')
    # Reuse the same complete validation/row replacement as the deployment loader.
    # This tiny CPU layer avoids constructing or executing the full encoder.
    feature_dim = config['feature_dim']
    if tuple(state['head.fc.weight'].shape) != (10, feature_dim) or tuple(state['head.fc.bias'].shape) != (10,):
        raise ValueError('Base classifier dimensions differ from the declared encoder')
    from types import SimpleNamespace
    classifier = torch.nn.Linear(feature_dim, 10).half()
    classifier.load_state_dict({'weight': state['head.fc.weight'], 'bias': state['head.fc.bias']}, strict=True)
    apply_classifier_patch(SimpleNamespace(head=SimpleNamespace(fc=classifier)), patch,
                           class_ids=config['replaced_class_ids'], feature_dim=feature_dim)
    # Replacing existing dictionary values preserves the original state key order.
    merged = state.copy()
    merged['head.fc.weight'] = classifier.weight.detach().clone()
    merged['head.fc.bias'] = classifier.bias.detach().clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.merge-frozen-head-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'export'
        staging.mkdir()
        merged_path = staging / 'student_fp16.pt'
        torch.save(merged, merged_path)
        reloaded = torch.load(merged_path, map_location='cpu', weights_only=True)
        if list(reloaded) != list(state):
            raise RuntimeError('Merged export changed original state key order')
        for name, expected in merged.items():
            actual = reloaded[name]
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or actual.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    != expected.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()):
                raise RuntimeError('Merged tensor bytes differ: ' + name)
        merged_config = copy.deepcopy(base_config)
        merged_config.update(schema_version=1, artifact_type='experimental_student_fp16',
            weights_file='student_fp16.pt', weights_sha256=sha256_file(merged_path),
            weights_source='frozen_encoder_plus_trained_classifier',
            source_checkpoint_sha256=config['patch_sha256'],
            source_patch_checkpoint_sha256=config['patch_sha256'],
            source_parent_checkpoint_sha256=base_config.get('source_checkpoint_sha256'),
            source_parent_weights_sha256=config['base_weights_sha256'],
            source_parent_config_sha256=config['base_config_sha256'],
            source_patch_config_sha256=config_sha,
            source_epoch=config.get('epochs', config.get('source_epoch')),
            source_training_config=copy.deepcopy(config.get('source_training_config',
                config.get('training_config', {}))),
            frozen_head_patch=copy.deepcopy(config),
            source_validation={}, formal_submission_ready=False, tta='hflip2', gates=None)
        (staging / 'config.json').write_text(json.dumps(merged_config, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n')
        size = merged_path.stat().st_size + (staging / 'config.json').stat().st_size
        if size > MAX_MODEL_BYTES:
            raise ValueError('Merged inference weights and required configuration exceed 500MB')
        receipt = {'artifact_type': 'frozen_head_merged_validation_export',
            'base_weights_sha256': config['base_weights_sha256'],
            'base_config_sha256': config['base_config_sha256'],
            'patch_sha256': config['patch_sha256'], 'patch_config_sha256': config_sha,
            'merged_weights_sha256': sha256_file(merged_path),
            'merged_config_sha256': sha256_file(staging / 'config.json'),
            'original_state_key_order_preserved': True,
            'frozen_encoder_tensor_bytes_preserved': True,
            'untouched_tensor_count': len(state) - 2,
            'patched_class_ids': list(config['replaced_class_ids']),
            'actual_inference_bytes': size, 'size_pass': True,
            'merged_weights_equivalent_to_base_plus_patch': True,
            'model_forward_executed': False, 'gpu_used': False,
            'temporary_validation_export_only': True}
        (staging / 'export_report.json').write_text(json.dumps(receipt, indent=2) + '\n')
        for path, digest in ((config_path, config_sha), (base_config_path, config['base_config_sha256']),
                             (weights_path, config['base_weights_sha256']), (patch_path, config['patch_sha256'])):
            if sha256_file(path) != digest:
                raise RuntimeError('Merged classifier inputs changed while exporting')
        if output.exists() or output.is_symlink():
            raise FileExistsError('Merged output appeared during export')
        staging.rename(output)
    return receipt
