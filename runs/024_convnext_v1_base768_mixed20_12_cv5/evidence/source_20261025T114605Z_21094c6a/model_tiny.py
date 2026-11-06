"""ConvNeXt-Tiny 十类多标签基线：GAP、LayerNorm、Dropout 与 Linear。

仅显式调用 build_tiny_model(pretrained=True) 才会向 timm 请求预训练权重。
不包含类别注意力、排序损失或 Sigmoid；Study 概率聚合由训练/推理入口负责。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from runtime_compat import prepare_runtime

prepare_runtime()

from torch import nn

from dataset import NUM_CLASSES

TINY_MODEL_NAME = "convnext_tiny.fb_in1k"
MODEL_CONFIG_KEYS = (
    "backbone_name", "drop_rate", "drop_path_rate", "gradient_checkpointing",
)
_MODEL_DEFAULTS = {
    "backbone_name": TINY_MODEL_NAME,
    "drop_rate": 0.1,
    "drop_path_rate": 0.05,
    "gradient_checkpointing": False,
}

__all__ = [
    "TINY_MODEL_NAME", "MODEL_CONFIG_KEYS", "validate_model_config", "build_tiny_model",
]


def validate_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """只校验结构配置，不实例化模型；拒绝未知字段和模型替换。"""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config 必须为映射。")
    unknown = set(model_config) - set(MODEL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"不支持的模型配置键：{', '.join(sorted(map(str, unknown)))}")
    config = {**_MODEL_DEFAULTS, **model_config}
    if config["backbone_name"] != TINY_MODEL_NAME:
        raise ValueError(f"backbone_name 必须是 {TINY_MODEL_NAME}。")
    for key in ("drop_rate", "drop_path_rate"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key} 必须为 [0,1) 内的有限数值。")
        config[key] = float(value)
    if type(config["gradient_checkpointing"]) is not bool:
        raise TypeError("gradient_checkpointing 必须为布尔值。")
    return config


def build_tiny_model(*, pretrained: bool, **model_config: Any) -> nn.Module:
    """构建 timm 标准 ConvNeXt-Tiny；恢复已有权重时必须 pretrained=False。

    标准头保留预训练 LayerNorm，随后使用 Dropout 和十维 Linear。
    返回对象原生支持 get_classifier()、named_parameters() 与 channels_last。
    4090 默认不启用重算；需要缩减激活显存时可显式开启检查点。
    """
    if type(pretrained) is not bool:
        raise TypeError("pretrained 必须显式传入布尔值。")
    config = validate_model_config(model_config)
    import timm

    model = timm.create_model(
        config["backbone_name"], pretrained=pretrained,
        num_classes=NUM_CLASSES, global_pool="avg",
        drop_rate=config["drop_rate"], drop_path_rate=config["drop_path_rate"],
    )
    model.set_grad_checkpointing(config["gradient_checkpointing"])
    return model
