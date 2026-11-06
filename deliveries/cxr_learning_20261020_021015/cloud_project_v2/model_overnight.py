"""ConvNeXt V2-Base with its standard GAP/LayerNorm/Dropout/Linear head.

Importing this file never constructs a model or requests pretrained weights.
Only an explicit build_overnight_model(pretrained=True) may download weights.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from runtime_compat import prepare_runtime

# 【学习包注释】兼容层只负责准备运行时；真正的网络在 build_overnight_model 中显式创建。
prepare_runtime()

from torch import nn

# 主干结构来自 timm：ConvNeXt V2-Base；后缀标识 FCMAE 预训练、ImageNet-1k 微调的权重配方。
# 这里没有把整张 768×768 图片展平成 ANN 输入。timm 内部保留空间网格，通过卷积逐层提取特征。
# Base 的四阶段通道为 128/256/512/1024，块数为 3/3/27/3；详解见 LEARNING_GUIDE.md。
BACKBONE_NAME = "convnextv2_base.fcmae_ft_in1k"
MODEL_NAME = "convnextv2_base_gap"
# 输出的十个位置与数据集标签顺序一一对应；每个数先是 logit，模型本身不做 sigmoid。
# 多个标签可以同时为阳性，因此最终需要十个独立的二分类分数，不用 softmax 强制总和为 1。
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
    # 只校验配置字典，不构建网络、不读图片、不发起训练，便于检查恢复用的架构元数据。
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    unknown = set(model_config) - set(MODEL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported model configuration: {sorted(unknown)}")
    # 传入的配置覆盖默认值；拼写错误的额外键在上面直接报错，避免悄悄使用意外结构。
    config = {**_DEFAULTS, **model_config}
    if config["backbone_name"] != BACKBONE_NAME:
        raise ValueError(f"backbone_name must be {BACKBONE_NAME}")
    # dropout 在分类头随机屏蔽部分特征；drop path 在训练时随机跳过残差分支。
    # 二者用途和所在位置不同；eval() 验证时不做这些随机丢弃。
    for key in ("drop_rate", "drop_path_rate"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key} must be finite and in [0, 1)")
        config[key] = float(value)
    if type(config["gradient_checkpointing"]) is not bool:
        raise TypeError("gradient_checkpointing must be bool")
    return config


def build_overnight_model(*, pretrained: bool, **model_config: Any) -> nn.Module:
    """Create the Base model; restoring a checkpoint must use pretrained=False."""
    # 首次迁移学习可显式用 pretrained=True。恢复本项目 checkpoint 时用 False，
    # 随后由训练/推理脚本加载自己的参数；这样不会先多余地读取或请求预训练权重。
    if type(pretrained) is not bool:
        raise TypeError("pretrained must be explicitly True or False")
    config = validate_model_config(model_config)
    # 真正的卷积块实现在已安装的 timm/models/convnext.py 中，本文件是配置与创建入口。
    import timm

    # 768 输入时主干尺寸：N×3×768×768 → N×128×192×192 →
    # N×256×96×96 → N×512×48×48 → N×1024×24×24。
    # global_pool="avg" 把每张 24×24 特征图求平均；经 LayerNorm/Dropout 后，
    # 最后的 Linear 把 1024 个特征映射成 10 个 logits，返回形状 N×10。
    model = timm.create_model(
        config["backbone_name"], pretrained=pretrained,
        num_classes=NUM_CLASSES, global_pool="avg",
        drop_rate=config["drop_rate"], drop_path_rate=config["drop_path_rate"],
    )
    # 激活检查点用“反向传播时重算部分前向”的方式省显存；它不同于保存到磁盘的权重断点。
    # 它不修改网络要学习的目标，也不意味着冻结主干，代价是多一些计算。
    model.set_grad_checkpointing(config["gradient_checkpointing"])
    return model
