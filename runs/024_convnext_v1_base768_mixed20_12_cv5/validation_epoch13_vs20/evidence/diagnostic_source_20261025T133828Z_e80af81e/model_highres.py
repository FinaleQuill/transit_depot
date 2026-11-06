"""高分辨率 ConvNeXt-V2、多尺度类别注意力和 Study 级 BCE/排序损失。

本模块只定义接口；导入不会创建模型、下载权重或执行模型计算。仅显式调用
build_highres_model(pretrained=True) 时，timm 才可能下载预训练权重。
十类仍输出独立 logits；Study 概率均值由训练/推理入口负责，不在这里改写。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from runtime_compat import prepare_runtime

prepare_runtime()

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataset import NUM_CLASSES

HIGHRES_MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"
SUPPORTED_BACKBONES = (
    HIGHRES_MODEL_NAME,
    "convnextv2_tiny.fcmae_ft_in1k",
)
MODEL_CONFIG_KEYS = (
    "backbone_name", "attention_dim", "attention_heads", "attention_layers",
    "drop_rate", "drop_path_rate", "gradient_checkpointing",
)
_MODEL_DEFAULTS = {
    "backbone_name": HIGHRES_MODEL_NAME,
    "attention_dim": 256,
    "attention_heads": 8,
    "attention_layers": 2,
    "drop_rate": 0.2,
    "drop_path_rate": 0.1,
    "gradient_checkpointing": True,
}

__all__ = [
    "HIGHRES_MODEL_NAME", "SUPPORTED_BACKBONES", "MODEL_CONFIG_KEYS",
    "validate_model_config", "build_highres_model", "HighResolutionCXRModel",
    "StudyBCERankLoss",
]


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须为有限非负数。")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} 必须为有限非负数。")
    return float(value)


def validate_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """校验并补齐结构配置；不创建模型，也不接受元数据中的 pretrained 开关。

    训练可将返回值原样保存。推理应显式使用 pretrained=False，并将该字典
    传给构建函数，避免恢复已有权重时再次申请或下载预训练权重。
    """
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config 必须为映射。")
    unknown = set(model_config) - set(MODEL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"不支持的模型配置键：{', '.join(sorted(map(str, unknown)))}")
    config = {**_MODEL_DEFAULTS, **model_config}
    if config["backbone_name"] not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone_name 必须属于 {SUPPORTED_BACKBONES}。")
    for key in ("attention_dim", "attention_heads", "attention_layers"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} 必须为正整数。")
    if config["attention_dim"] % 4 != 0:
        raise ValueError("attention_dim 必须可被4整除，以构造二维正弦位置编码。")
    if config["attention_dim"] % config["attention_heads"] != 0:
        raise ValueError("attention_dim 必须可被 attention_heads 整除。")
    for key in ("drop_rate", "drop_path_rate"):
        config[key] = _finite_nonnegative(config[key], key)
        if config[key] >= 1.0:
            raise ValueError(f"{key} 必须小于1。")
    if type(config["gradient_checkpointing"]) is not bool:
        raise TypeError("gradient_checkpointing 必须为布尔值。")
    return config


def _spatial_position_encoding(
    height: int, width: int, dim: int, *, device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """按实际特征网格生成二维 sin/cos 编码，支持768/1024等不同输入尺寸。"""
    quarter = dim // 4
    frequency = torch.arange(quarter, device=device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * frequency / max(quarter - 1, 1))
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequency
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequency
    y_encoding = torch.cat((y.sin(), y.cos()), dim=-1)
    x_encoding = torch.cat((x.sin(), x.cos()), dim=-1)
    encoding = torch.cat((
        y_encoding[:, None, :].expand(height, width, -1),
        x_encoding[None, :, :].expand(height, width, -1),
    ), dim=-1)
    return encoding.reshape(1, height * width, dim).to(dtype=dtype)


class MultiScaleLabelAttentionHead(nn.Module):
    """两个尺度的空间 tokens、十个类别 queries 与全局分类分支。

    get_classifier 返回整个对象，投影、归一化、类别 queries、尺度编码、
    decoder 和两个输出分支都进入头部参数组。原 timm 的 pool/norm/dropout
    头用于全局分支，保留已加载的预训练归一化参数，不留下闲置可训练层。
    """

    def __init__(
        self, channels: tuple[int, int], global_preprocess: nn.Module,
        *, attention_dim: int, attention_heads: int, attention_layers: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, attention_dim, kernel_size=1)
            for in_channels in channels
        ])
        self.token_norms = nn.ModuleList([nn.LayerNorm(attention_dim) for _ in channels])
        self.scale_embeddings = nn.Parameter(torch.empty(2, attention_dim))
        self.class_queries = nn.Parameter(torch.empty(NUM_CLASSES, attention_dim))
        # 分别构建各层，避免 TransformerDecoder 的克隆层使用相同初始参数。
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=attention_dim, nhead=attention_heads,
                dim_feedforward=attention_dim * 4, dropout=drop_rate,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(attention_layers)
        ])
        self.decoder_norm = nn.LayerNorm(attention_dim)
        self.output_dropout = nn.Dropout(drop_rate)
        self.attention_weight = nn.Parameter(torch.empty(NUM_CLASSES, attention_dim))
        self.attention_bias = nn.Parameter(torch.zeros(NUM_CLASSES))
        self.global_preprocess = global_preprocess
        self.global_classifier = nn.Linear(channels[-1], NUM_CLASSES)
        # 只初始化新增参数，不对移入的预训练 pool/norm 头重新 apply 初始化。
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        nn.init.trunc_normal_(self.attention_weight, std=0.02)
        nn.init.trunc_normal_(self.global_classifier.weight, std=0.02)
        nn.init.zeros_(self.global_classifier.bias)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        memories = []
        for index, (feature, projection, norm) in enumerate(zip(
            features, self.projections, self.token_norms,
        )):
            projected = projection(feature)
            height, width = projected.shape[-2:]
            tokens = norm(projected.flatten(2).transpose(1, 2))
            position = _spatial_position_encoding(
                height, width, self.attention_dim, device=tokens.device, dtype=tokens.dtype,
            )
            scale = self.scale_embeddings[index].to(dtype=tokens.dtype)[None, None, :]
            memories.append(tokens + position + scale)
        memory = torch.cat(memories, dim=1)
        queries = self.class_queries.to(dtype=memory.dtype)[None, :, :]
        decoded = queries.expand(memory.shape[0], -1, -1)
        for layer in self.decoder_layers:
            decoded = layer(decoded, memory)
        decoded = self.output_dropout(self.decoder_norm(decoded))
        attention_logits = (decoded * self.attention_weight[None, :, :]).sum(dim=-1)
        attention_logits = attention_logits + self.attention_bias
        global_features = self.global_preprocess(features[-1])
        global_logits = self.global_classifier(global_features)
        # 两支均为未经过 Sigmoid 的十维 logits；不使用 softmax 类别互斥。
        return global_logits + attention_logits


class HighResolutionCXRModel(nn.Module):
    """手工取得最后两个 stage；训练时逐块使用非重入 activation checkpoint。

    本地 timm.ConvNeXtStage 的 forward 是 downsample -> blocks。这里保留
    同一计算顺序和参数，显式 use_reentrant=False，不依赖 timm 的全局设置。
    非重入模式允许输入图像 requires_grad=False，也支持入口另加的外层
    CPU-view checkpoint。外层需额外传 CUDA anchor，以记录 GPU 随机状态。
    模型不在成员或全局变量中缓存输入、特征等激活 Tensor。
    """

    def __init__(self, backbone: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_config = dict(config)
        self.num_classes = NUM_CLASSES
        self.gradient_checkpointing = config["gradient_checkpointing"]
        feature_info = backbone.feature_info
        if len(feature_info) != 4 or len(backbone.stages) != 4:
            raise ValueError("当前高分辨率模型要求四个 ConvNeXt stage。")
        channels = tuple(int(item["num_chs"]) for item in feature_info[-2:])
        if getattr(backbone.head, "num_features", None) != channels[-1]:
            raise ValueError("预训练全局头维度与最后一个 stage 不一致。")
        global_preprocess = backbone.head
        backbone.head = nn.Identity()
        # 禁止 timm 在 stage 内再嵌入默认策略；下方逐 block 明确指定非重入。
        backbone.set_grad_checkpointing(False)
        self.head = MultiScaleLabelAttentionHead(
            channels, global_preprocess,
            attention_dim=config["attention_dim"], attention_heads=config["attention_heads"],
            attention_layers=config["attention_layers"], drop_rate=config["drop_rate"],
        )

    def get_classifier(self) -> nn.Module:
        """返回完整自定义头，供冻结和差分学习率分组使用。"""
        return self.head

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if type(enable) is not bool:
            raise TypeError("checkpoint 开关必须为布尔值。")
        self.gradient_checkpointing = enable

    @staticmethod
    def _run_module(module: nn.Module, x: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
        if use_checkpoint:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("输入必须为 [batch,3,height,width] Tensor。")
        if images.shape[0] < 1 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError("图像 batch 必须非空、三通道，长宽至少32。")
        if not images.is_floating_point():
            raise TypeError("模型输入必须为归一化后的浮点图像。")
        use_checkpoint = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
            and any(parameter.requires_grad for parameter in self.backbone.parameters())
        )
        x = self._run_module(self.backbone.stem, images, use_checkpoint)
        features = []
        for index, stage in enumerate(self.backbone.stages):
            if use_checkpoint:
                x = self._run_module(stage.downsample, x, True)
                for block in stage.blocks:
                    x = self._run_module(block, x, True)
            else:
                x = stage(x)
            if index >= 2:
                features.append(x)
        features[-1] = self.backbone.norm_pre(features[-1])
        return features[0], features[1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))


def build_highres_model(
    *, pretrained: bool = True, backbone_name: str = HIGHRES_MODEL_NAME,
    attention_dim: int = 256, attention_heads: int = 8, attention_layers: int = 2,
    drop_rate: float = 0.2, drop_path_rate: float = 0.1,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """创建模型；已有 checkpoint 的恢复必须显式 pretrained=False。"""
    if type(pretrained) is not bool:
        raise TypeError("pretrained 必须为布尔值。")
    config = validate_model_config({
        "backbone_name": backbone_name, "attention_dim": attention_dim,
        "attention_heads": attention_heads, "attention_layers": attention_layers,
        "drop_rate": drop_rate, "drop_path_rate": drop_path_rate,
        "gradient_checkpointing": gradient_checkpointing,
    })
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError("无法导入 timm 或其依赖；本模块不会自动安装依赖。") from exc
    # num_classes=0 只移除原 ImageNet 分类器，已加载的 stem/stages/norm 仍复用。
    backbone = timm.create_model(
        config["backbone_name"], pretrained=pretrained, num_classes=0,
        drop_rate=config["drop_rate"], drop_path_rate=config["drop_path_rate"],
    )
    return HighResolutionCXRModel(backbone, config)


class StudyBCERankLoss(nn.Module):
    """原始硬标签的 Study BCE + 当前微批内分类别成对排序损失。

    study_logits 应为各视角概率均值对应的 logit。这里只接收 [Study,10]，
    不改变 Study 聚合。每类需在当前微批同时有正负 Study 才构造排序对，
    先对该类所有正负对求均值，再对有效类别等权平均。不跨微批储存图/标签。
    不叠加 ASL 或 pos_weight；返回 total、未加权 BCE、未加权 rank 供记录。
    """

    def __init__(self, rank_weight: float = 0.2) -> None:
        super().__init__()
        self.rank_weight = _finite_nonnegative(rank_weight, "rank_weight")

    def forward(
        self, study_logits: torch.Tensor, targets: torch.Tensor,
        *, rank_weight: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(study_logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("Study logits 和 targets 必须为 Tensor。")
        if study_logits.ndim != 2 or study_logits.shape != targets.shape:
            raise ValueError("Study logits 与 targets 必须为同形状 [Study,10]。")
        if study_logits.shape[0] < 1 or study_logits.shape[1] != NUM_CLASSES:
            raise ValueError("Study 微批必须非空，且每项有十个类别。")
        if not study_logits.is_floating_point() or not targets.is_floating_point():
            raise TypeError("Study logits 与 targets 必须为浮点 Tensor。")
        z, y = study_logits.float(), targets.float()
        if not bool(torch.all((y == 0.0) | (y == 1.0))):
            raise ValueError("StudyBCERankLoss 仅接受原始0/1硬标签，禁止软标签或伪标签。")
        weight = self.rank_weight if rank_weight is None else _finite_nonnegative(rank_weight, "rank_weight")
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="mean")
        class_losses = []
        for class_index in range(NUM_CLASSES):
            positive = z[:, class_index][y[:, class_index] == 1.0]
            negative = z[:, class_index][y[:, class_index] == 0.0]
            if positive.numel() and negative.numel():
                differences = positive[:, None] - negative[None, :]
                class_losses.append(F.softplus(-differences).mean())
        # 取一个有限标量乘0，避免用整批sum构造0时发生不必要的求和溢出。
        rank = torch.stack(class_losses).mean() if class_losses else z.reshape(-1)[0] * 0.0
        return bce + weight * rank, bce, rank
