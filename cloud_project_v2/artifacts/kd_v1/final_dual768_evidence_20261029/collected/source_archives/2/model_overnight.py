"""Approved ConvNeXt V1 Base/Tiny with GAP/LayerNorm/Dropout/Linear heads.

Importing this file never constructs a model or requests pretrained weights.
Only an explicit build_overnight_model(pretrained=True) may download weights.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

from runtime_compat import prepare_runtime

prepare_runtime()

from torch import nn

BASE_BACKBONE_NAME = "convnext_base.fb_in22k_ft_in1k_384"
TINY_BACKBONE_NAME = "convnext_tiny.fb_in22k_ft_in1k_384"
BACKBONE_NAME = BASE_BACKBONE_NAME
MODEL_NAME = "convnext_v1_base_gap"
TINY_MODEL_NAME = "convnext_v1_tiny_gap"
APPROVED_BACKBONES = (BASE_BACKBONE_NAME, TINY_BACKBONE_NAME)
NUM_CLASSES = 10
MODEL_CONFIG_KEYS = ("backbone_name", "drop_rate", "drop_path_rate", "gradient_checkpointing")
_DEFAULTS = {
    "backbone_name": BACKBONE_NAME,
    "drop_rate": 0.2,
    "drop_path_rate": 0.15,
    "gradient_checkpointing": True,
}


def validate_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate architecture metadata without initializing or probing a model."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    unknown = set(model_config) - set(MODEL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported model configuration: {sorted(unknown)}")
    config = {**_DEFAULTS, **model_config}
    if config["backbone_name"] not in APPROVED_BACKBONES:
        raise ValueError(f"Only approved ConvNeXt V1 weights are supported: {APPROVED_BACKBONES}")
    for key in ("drop_rate", "drop_path_rate"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key} must be finite and in [0, 1)")
        config[key] = float(value)
    if type(config["gradient_checkpointing"]) is not bool:
        raise TypeError("gradient_checkpointing must be bool")
    return config


def build_overnight_model(*, pretrained: bool, pretrained_weights_path: str | None = None,
                          **model_config: Any) -> nn.Module:
    """Create an approved V1 model; checkpoint restore requires pretrained=False."""
    if type(pretrained) is not bool:
        raise TypeError("pretrained must be explicitly True or False")
    if pretrained_weights_path is not None:
        if not pretrained or not isinstance(pretrained_weights_path, str) or not pretrained_weights_path:
            raise ValueError("pretrained_weights_path is only valid with pretrained=True and a nonempty path")
        if not Path(pretrained_weights_path).is_file():
            raise FileNotFoundError("Declared pretrained weights file does not exist")
    config = validate_model_config(model_config)
    import timm

    model = timm.create_model(
        config["backbone_name"], pretrained=pretrained,
        num_classes=NUM_CLASSES, global_pool="avg",
        drop_rate=config["drop_rate"], drop_path_rate=config["drop_path_rate"],
        **({"pretrained_cfg_overlay": {"file": pretrained_weights_path}}
           if pretrained_weights_path is not None else {}),
    )
    model.set_grad_checkpointing(config["gradient_checkpointing"])
    return model
