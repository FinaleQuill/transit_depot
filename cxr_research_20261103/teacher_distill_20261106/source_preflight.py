"""CPU-only public model compatibility and environment evidence, no forward."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    import torch
    import timm
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    root = Path('/home/featurize/data/kd_rebuild_20261106')
    result = dict(scope='CPU construction and checkpoint compatibility only; no model forward',
                  python=sys.version, executable=sys.executable,
                  image='crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b',
                  packages={x:importlib.metadata.version(x) for x in (
                      'torch','torchvision','timm','numpy','Pillow','scikit-learn',
                      'safetensors','huggingface-hub','pytest','pandas')}, models=[])
    for key, model_name in (
        ('dinov3_l','vit_large_patch16_dinov3.lvd1689m'),
        ('convnext_large','convnext_large.fb_in22k_ft_in1k_384'),
    ):
        kwargs = dict(img_size=512, dynamic_img_size=True) if key == 'dinov3_l' else {}
        model = timm.create_model(model_name, pretrained=False, num_classes=0, **kwargs)
        state = load_file(str(root / 'assets' / key / 'model.safetensors'), device='cpu')
        # The classification head of the ImageNet CNN is intentionally removed
        # for a new official-competition head. No backbone mismatch is allowed.
        ignored = sorted(k for k in state if k in ('head.fc.weight','head.fc.bias'))
        for k in ignored:
            del state[k]
        incompatible = model.load_state_dict(state, strict=False)
        entry = dict(key=key, name=model_name, model_class=type(model).__name__,
                     num_features=model.num_features, parameters=sum(p.numel() for p in model.parameters()),
                     missing_keys=incompatible.missing_keys, unexpected_keys=incompatible.unexpected_keys,
                     ignored_pretraining_classifier_keys=ignored, construction_kwargs=kwargs,
                     grad_checkpointing_available=hasattr(model,'set_grad_checkpointing'),
                     first_state_keys=list(state)[:8], last_state_keys=list(state)[-8:])
        result['models'].append(entry)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f'Backbone checkpoint compatibility failed: {entry}')
        print(json.dumps(entry),flush=True)
        del state, model
        gc.collect()
    destination = root / 'environment' / 'cpu_preflight.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(str(destination),flush=True)


if __name__ == '__main__':
    main()
