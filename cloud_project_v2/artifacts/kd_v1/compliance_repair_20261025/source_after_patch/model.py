"""胸片十类多标签模型与训练损失，当前文件仅供人工审查。

模块导入时不会读取图片、创建模型、下载权重或启动训练。只有显式调用
``build_model`` 才会创建模型；``pretrained=True`` 在本地无权重时可能下载。

标签顺序严格沿用 dataset.py 中的原始 1～10 类映射。一个样本可以同时具有
多个阳性类别，所以模型输出十个独立的 logits，不能改成 softmax 单选分类。

接口依据：
https://huggingface.co/docs/timm/reference/models
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from runtime_compat import prepare_windows_runtime

# 直接导入 model.py 也必须先走本项目的 Windows 启动兼容层；不依赖
# train.py 的导入顺序，否则独立审查/推理仍可能停在 PyTorch 的 WMI 查询。
prepare_windows_runtime()

import torch
from torch import nn
from torch.nn import functional as F

from dataset import DEFAULT_TRAIN_MANIFEST, NUM_CLASSES, read_manifest

__all__ = [
    "build_model", "calculate_pos_weight", "build_criterion",
    "SoftTargetAsymmetricLoss", "SoftTargetFocalLoss",
]


def build_model(
    *, pretrained: bool = True, drop_rate: float = 0.3, drop_path_rate: float = 0.1
) -> nn.Module:
    """创建 EfficientNet-B3，返回未经 Sigmoid 的十维 logits。

    timm 按 ``num_classes=10`` 替换分类头。``drop_rate`` 在分类前随机丢弃
    特征，``drop_path_rate`` 对残差分支作随机深度正则；两者仅训练态生效，
    推理时关闭，因此不改变分类头参数名或历史权重的加载结构。

    冻结期若要固定主干 BN，训练层应单独把 BN 设为 eval，不能把整个模型
    设成 eval 后只让 classifier.train()：timm 在模型 forward_head 中的
    功能式 Dropout 读取的是整个模型的 training 标志。

    注意：本函数应在用户后续授权后才调用。显式选择 ``pretrained=True``
    可能触发 timm 下载预训练权重；仅导入本模块不会触发下载。
    """
    for name, value in (("drop_rate", drop_rate), ("drop_path_rate", drop_path_rate)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"{name} 必须为 [0, 1) 内的有限数。")
    try:
        # 延迟导入：便于先审查数据和损失代码，也避免导入文件即构建模型。
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "无法导入 timm 或其依赖。请在后续获授权的环境准备阶段检查依赖；"
            "本文件不会自动安装依赖或下载权重。"
        ) from exc

    # timm 的默认分类头直接输出 logits，这里严禁再追加 nn.Sigmoid()。
    return timm.create_model(
        "efficientnet_b3", pretrained=pretrained, num_classes=NUM_CLASSES,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate,
    )


def _binary_target(row: dict) -> tuple[float, ...]:
    """校验计算权重所用的原始多热标签；不允许拿增强后的软标签估计权重。"""
    raw_target = row.get("targets")
    image_id = row.get("image_id", "未知样本")
    if raw_target is None or len(raw_target) != NUM_CLASSES:
        raise ValueError(f"训练样本 {image_id} 必须提供 {NUM_CLASSES} 维多热标签。")
    target = tuple(float(value) for value in raw_target)
    if any(value not in (0.0, 1.0) for value in target):
        raise ValueError(f"训练样本 {image_id} 的权重统计标签必须全部为 0 或 1。")
    return target


def calculate_pos_weight(
    manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    *,
    unit: Literal["image", "study"] = "image",
    power: float = 0.5,
    max_weight: float | None = 10.0,
) -> torch.Tensor:
    """仅从训练清单估计十类正例权重，返回 CPU float32 张量。

    第 c 类的权重 = (第 c 类阴性样本数 / 第 c 类阳性样本数) ** power，
    然后按需限制上界。默认取平方根并封顶 10，缓和极稀有类别的梯度放大，
    避免把已经很大的原始比值再和激进重采样/Focal 叠加。
    这只放大/缩小 BCE 的正例项，并非给整张图片分配采样权重。

    参数说明：
        manifest_path: 默认使用 reviewed 版 train_clean.csv，绝不读取 valid/test。
        unit: 默认 ``image``，每个训练图像计一次，与逐图 BCE 的样本单位一致。
            可显式选择 ``study``：同一 Study 只计一次，且必须核对其标签一致。
            两者权重可能不同，不能在后续对比实验中无记录地切换。
        power: 默认 0.5；取 1 且 max_weight=None 复现原始比值加权 BCE；
            取 0 等价所有类别权重为 1。范围 [0, 1]，只改变训练损失。
        max_weight: 默认上限 10，可传 None 关闭上限；数值必须 >=1。
            原比值小于 1 时仍保留其值，不把多数阳性类强行改为权重 1。
            这些调整均为待验证方案，不能仅凭理论声称一定提高 AUC/F1。

    缺正例时比值无定义；缺负例时权重为零会消除该类正例梯度。因此这两种
    情况直接报错，要求人工检查数据/划分，不能悄悄补 epsilon 或伪造标签。
    """
    if unit not in ("image", "study"):
        raise ValueError("unit 只能为 'image' 或 'study'。")
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("power 必须为 [0, 1] 内的有限数。")
    if max_weight is not None and (
        not math.isfinite(max_weight) or max_weight < 1.0
    ):
        raise ValueError("max_weight 必须为 >=1 的有限数，或使用 None 表示不截断。")

    # 明确指定 train。read_manifest 还会检查每一行的 split，禁止误用验证/测试清单。
    # 这里只解析 CSV/JSON 多热标签，不打开任何原图或缓存，也不做随机增强。
    rows = read_manifest(manifest_path, split="train")
    if not rows:
        raise ValueError("训练清单为空，无法估计 pos_weight。")

    positive_counts = [0] * NUM_CLASSES
    sample_count = 0
    study_targets: dict[str, tuple[float, ...]] = {}
    for row in rows:
        target = _binary_target(row)
        if unit == "study":
            study_key = row.get("study_key")
            if not isinstance(study_key, str) or not study_key:
                raise ValueError("按 Study 计数时，每个样本都必须具有有效 study_key。")
            previous = study_targets.get(study_key)
            if previous is not None:
                if previous != target:
                    raise ValueError("同一 Study 出现不同标签，禁止静默选择一行计算权重。")
                continue
            study_targets[study_key] = target

        sample_count += 1
        for class_index, value in enumerate(target):
            positive_counts[class_index] += int(value)

    weights: list[float] = []
    for class_index, positives in enumerate(positive_counts):
        negatives = sample_count - positives
        if positives == 0 or negatives == 0:
            # 使用原始 1～10 类编号，便于对照清单；不把错误类别默认为“正常”。
            raise ValueError(
                f"第 {class_index + 1} 类无法计算有效 pos_weight："
                f"positive={positives}, negative={negatives}, unit={unit}。"
            )
        weight = (negatives / positives) ** power
        weights.append(min(weight, max_weight) if max_weight is not None else weight)

    return torch.tensor(weights, dtype=torch.float32)


class SoftTargetAsymmetricLoss(nn.Module):
    """兼容 Mixup/CutMix 软标签的多标签 ASL，不叠加类别正例权重。

    正、负分支分别计算损失与调制因子，再按 y 和 (1-y) 线性组合。
    这样软标签仍表示两个硬标签损失的插值，不使用依赖混合标签的统一
    p_t 或 gamma，也不对软标签取整。调制因子保留梯度，不擅自 detach。

    默认 gamma_pos=0、gamma_neg=4、clip=0.05；负例概率截断使容易的
    负例减小或消除梯度。该策略及其参数效果仍需后续完整验证，不能仅凭
    更小的损失数值判断提分；与原加权 BCE/Focal 的数值也不能直接比较。

    内部统一以 float32 计算。正分支使用 softplus(-z)，clip=0 时负分支
    直接使用 softplus(z)，避免极端 logits 经 sigmoid 后再取 log 不稳定。
    pos_weight 全1 buffer 仅保持日志、设备迁移及保存接口兼容；ASL 不读
    训练清单统计类别权重，不叠加 Nneg/Npos 或其开方权重。
    """

    def __init__(
        self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        for name, value in (("gamma_pos", gamma_pos), ("gamma_neg", gamma_neg)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"ASL {name} 必须为非负有限数。")
        if not math.isfinite(clip) or not 0.0 <= clip < 1.0:
            raise ValueError("ASL clip 必须为 [0, 1) 内的有限数。")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.register_buffer("pos_weight", torch.ones(NUM_CLASSES, dtype=torch.float32))

    @staticmethod
    def _focusing_power(base: torch.Tensor, exponent: float) -> torch.Tensor:
        """正指数下保持零底数为0，避免 0<gamma<1 时零点导数奇异。"""
        positive = base > 0.0
        # where 先替换零底数再幂运算，不能先 pow 再屏蔽可能无穷的梯度。
        # 正底数保留完整计算图；零底数的调制值及梯度都为0，不做 detach。
        safe_base = torch.where(positive, base, torch.ones_like(base))
        return torch.where(positive, safe_base.pow(exponent), torch.zeros_like(base))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("logits 和 targets 必须均为 Tensor。")
        if logits.shape != targets.shape or logits.ndim < 1:
            raise ValueError("logits 与 targets 必须同形状，且含类别维。")
        if logits.shape[-1] != self.pos_weight.numel():
            raise ValueError(f"输入最后一维必须为 {NUM_CLASSES} 类。")
        if not logits.is_floating_point() or not targets.is_floating_point():
            raise TypeError("logits 与软标签必须为浮点张量，不能取整或 argmax。")

        # 仅做与 Focal 一致的形状和类型检查；标签范围由数据层保证，
        # 非有限损失由训练层守卫，避免每批 .item() 强制 GPU 同步。
        z, y = logits.float(), targets.float()
        positive = F.softplus(-z)
        if self.gamma_pos != 0.0:
            positive = positive * self._focusing_power(torch.sigmoid(-z), self.gamma_pos)

        if self.clip == 0.0:
            negative = F.softplus(z)
        else:
            negative_probability = (torch.sigmoid(-z) + self.clip).clamp(max=1.0)
            negative = -negative_probability.clamp(min=torch.finfo(z.dtype).tiny).log()
        # 零指数时直接保留原分支，避免依赖 0**0 的行为。
        if self.gamma_neg != 0.0:
            negative_base = (torch.sigmoid(z) - self.clip).clamp(min=0.0)
            negative = negative * self._focusing_power(negative_base, self.gamma_neg)
        return (y * positive + (1.0 - y) * negative).mean()


class SoftTargetFocalLoss(nn.Module):
    """兼容 Mixup/CutMix 软标签的多标签 Focal，只作为后续消融选项。

    正例项和负例项分别按 sigmoid(-z)**gamma、sigmoid(z)**gamma 调制，
    再以 y 和 (1-y) 线性组合。这样混合标签仍表示两个硬标签损失的插值，
    不能先算混合 p_t 再统一乘调制项：后者改变软标签的训练语义。

    gamma=0 时退化为同 pos_weight 的 BCEWithLogitsLoss（数学等价）；
    本实现不另加 alpha，避免重复补偿类别。Focal 会强调难样本，胸片标签
    噪声也可能同时被强调，因此 Focal 保留为消融选项，不保证换 Focal
    就能提分。比较时须保持相同数据划分/seed/epoch 和其他参数。

    输入应为同形状浮点 logits 与 [0,1] 标签，且最后一维为十类。这里做
    静态形状/类型校验，不逐 batch 调 .item() 检查数值，避免迫使 GPU 同步；
    标签由数据层验证，非有限损失由训练层守卫。内部用 float32 和 softplus
    计算，防止半精度下直接 log(sigmoid()) 出现上溢/下溢。
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("focal gamma 必须为非负有限数。")
        if not isinstance(pos_weight, torch.Tensor):
            raise TypeError("pos_weight 必须为 Tensor。")
        if pos_weight.ndim != 1 or pos_weight.numel() != NUM_CLASSES:
            raise ValueError(f"pos_weight 必须为 {NUM_CLASSES} 维一维张量。")
        self.gamma = float(gamma)
        # 与原 BCE 一样以 buffer 保存；.to(device) 和 checkpoint 可统一处理。
        self.register_buffer("pos_weight", pos_weight.detach().clone().float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("logits 和 targets 必须均为 Tensor。")
        if logits.shape != targets.shape or logits.ndim < 1:
            raise ValueError("logits 与 targets 必须同形状，且含类别维。")
        if logits.shape[-1] != self.pos_weight.numel():
            raise ValueError(f"输入最后一维必须为 {NUM_CLASSES} 类。")
        if not logits.is_floating_point() or not targets.is_floating_point():
            raise TypeError("logits 与软标签必须为浮点张量，不能取整或 argmax。")
        z, y = logits.float(), targets.float()
        positive = self.pos_weight * y * F.softplus(-z)
        negative = (1.0 - y) * F.softplus(z)
        if self.gamma != 0.0:
            positive = positive * torch.sigmoid(-z).pow(self.gamma)
            negative = negative * torch.sigmoid(z).pow(self.gamma)
        return (positive + negative).mean()


def build_criterion(
    manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    *,
    loss_name: Literal["asl", "bce", "focal"] = "asl",
    focal_gamma: float = 2.0,
    asl_gamma_pos: float = 0.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    pos_weight_power: float = 0.5,
    unit: Literal["image", "study"] = "image",
    max_weight: float | None = 10.0,
    device: torch.device | str | None = None,
) -> nn.Module:
    """创建多标签 ASL 或原有加权 BCE/Focal，不运行模型或反向传播。

    使用时输入模型原始 logits 和同形状浮点标签，形状均为 [batch, 10]。
    BCEWithLogitsLoss 内部已经稳定地计算 Sigmoid，因此损失之前不能先做
    Sigmoid。Mixup/CutMix 得到的 [0, 1] 软标签可以直接传入，不做 argmax、
    取整或阈值化，也不把它们变回某一个硬类别。

    默认 ASL 分别调制正负分支，不读取清单统计 pos_weight，也不叠加类别
    权重；其全1 pos_weight buffer 仅保持既有日志和保存接口兼容。因此
    pos_weight_power、unit、max_weight 仅影响显式选择的 BCE/Focal。
    BCE/Focal 仍只从 train 估计权重，默认取平方根并封顶10；旧基线需显式
    loss_name='bce'、pos_weight_power=1、max_weight=None。不同权重/损失的
    原始 loss 不能直接比较，应看完整验证集 AUC/F1 或无权重参考 BCE。

    pos_weight 是损失模块的 buffer，必须和 logits/targets 位于同一设备。
    可传入 device；或在训练准备阶段将返回值与模型一起 ``.to(device)``。
    后续验证/推理若需要概率，应在模型 logits 之后仅调用一次 torch.sigmoid，
    再交给 dataset.py 中的 Study 汇总逻辑，不能把 logits 当作概率平均。
    """
    if loss_name not in ("asl", "bce", "focal"):
        raise ValueError("loss_name 只能为 'asl'、'bce' 或 'focal'。")
    criterion: nn.Module
    if loss_name == "asl":
        criterion = SoftTargetAsymmetricLoss(
            gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip
        )
    else:
        if not math.isfinite(focal_gamma) or focal_gamma < 0.0:
            raise ValueError("focal_gamma 必须为非负有限数。")
        pos_weight = calculate_pos_weight(
            manifest_path, unit=unit, power=pos_weight_power, max_weight=max_weight
        )
        if loss_name == "focal":
            criterion = SoftTargetFocalLoss(pos_weight=pos_weight, gamma=focal_gamma)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    if device is not None:
        criterion = criterion.to(device)
    return criterion
