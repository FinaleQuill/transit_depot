"""Class-specific spatial pooling for the existing ConvNeXt V1 head.

Installing the head does not construct a backbone or load weights. Load the GAP
checkpoint strictly first, then install: all old state-dict keys are retained and
only ``head.attention.weight`` is added. A full MIL checkpoint can instead be
loaded strictly after installation. Attention has no bias because a spatially
constant bias cancels under softmax.
"""
from __future__ import annotations

from statistics import median
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn


class SpatialMILHead(nn.Module):
    """Pool once per class, then apply the original post-pool norm and FC row.

    The backbone still runs once. Zero attention weights give uniform pooling;
    in evaluation mode its logits match the original GAP head to floating-point
    tolerance. ``pre_logits=True`` returns B x classes x channels, since each
    class now has its own pooled vector. Dropout remains the original module and
    is applied to these class-specific vectors during training.
    """

    def __init__(self, gap_head: nn.Module):
        super().__init__()
        required = ("global_pool", "norm", "flatten", "pre_logits", "drop", "fc")
        if not all(hasattr(gap_head, name) for name in required):
            raise ValueError("MIL requires a GAP -> norm -> dropout -> Linear head")
        if getattr(gap_head.global_pool, "pool_type", None) != "avg":
            raise ValueError("MIL installation requires average global pooling")
        if not isinstance(gap_head.fc, nn.Linear) or not isinstance(gap_head.pre_logits, nn.Identity):
            raise ValueError("MIL requires a Linear classifier without a hidden MLP")
        if isinstance(gap_head.norm, nn.Identity):
            raise ValueError("MIL requires the existing post-pooling normalization")
        self.in_features = gap_head.fc.in_features
        self.num_features = self.in_features
        self.num_classes = gap_head.fc.out_features
        self.hidden_size = None
        self.use_conv = False
        # Reuse, rather than copy, original modules so both parameter identities
        # and checkpoint key names remain unchanged.
        for name in required:
            setattr(self, name, getattr(gap_head, name))
        self.attention = nn.Conv2d(self.in_features, self.num_classes, kernel_size=1,
                                   bias=False, device=self.fc.weight.device,
                                   dtype=self.fc.weight.dtype)
        nn.init.zeros_(self.attention.weight)
        self.train(gap_head.training)

    def forward_attention(self, features: Tensor) -> Tensor:
        """Return B x classes x H x W maps; each map sums to one in FP32."""
        if features.ndim != 4 or features.shape[1] != self.in_features:
            raise ValueError(f"Expected NCHW features with {self.in_features} channels")
        logits = self.attention(features)
        return logits.flatten(2).float().softmax(dim=-1).reshape_as(logits)

    def _pool(self, features: Tensor, attention: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        weights = attention.flatten(2).to(features.dtype)
        # Centering on GAP preserves the original pooling reduction exactly at
        # zero initialization, while remaining algebraically weighted pooling.
        uniform = weights.new_tensor(1.0 / (height * width))
        delta = torch.bmm(weights - uniform, features.flatten(2).transpose(1, 2))
        pooled = self.global_pool(features).flatten(1).unsqueeze(1) + delta
        # ConvNeXt's LayerNorm2d is *after* pooling. Apply it to each class
        # vector, never to the feature map before attention pooling.
        pooled = self.norm(pooled.reshape(batch * self.num_classes, channels, 1, 1))
        pooled = self.flatten(pooled)
        pooled = self.pre_logits(pooled)
        return self.drop(pooled).reshape(batch, self.num_classes, channels)

    def _classify(self, pooled: Tensor) -> Tensor:
        # One batched dot product per class. baddbmm retains GEMM accumulation
        # and fused bias behavior in FP16; an elementwise product followed by
        # sum rounds every product first and needlessly perturbs the GAP logits.
        bias = (self.fc.bias[:, None, None] if self.fc.bias is not None
                else pooled.new_zeros(self.num_classes, 1, 1))
        logits = torch.baddbmm(bias, pooled.transpose(0, 1), self.fc.weight[:, :, None])
        return logits.squeeze(-1).transpose(0, 1)

    def forward_with_attention(self, features: Tensor) -> tuple[Tensor, Tensor]:
        attention = self.forward_attention(features)
        return self._classify(self._pool(features, attention)), attention

    def forward(self, features: Tensor, pre_logits: bool = False) -> Tensor:
        attention = self.forward_attention(features)
        pooled = self._pool(features, attention)
        return pooled if pre_logits else self._classify(pooled)


def install_spatial_mil(model: nn.Module) -> nn.Module:
    """Replace only ``model.head`` in place, before creating an optimizer/EMA.

    Idempotent installation avoids resetting already learned attention. Existing
    GAP state dictionaries should be loaded *before* this call; arbitrary
    ``strict=False`` checkpoint loading is unnecessary.
    """
    if isinstance(getattr(model, "head", None), SpatialMILHead):
        return model
    if not hasattr(model, "forward_features") or not hasattr(model, "head"):
        raise ValueError("MIL requires a ConvNeXt-style forward_features/head model")
    if not isinstance(getattr(model, "norm_pre", None), nn.Identity):
        raise ValueError("MIL requires ConvNeXt normalization after pooling")
    model.head = SpatialMILHead(model.head)
    return model


def forward_attention(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    """Return logits and maps with one backbone forward, without cached tensors."""
    if not isinstance(getattr(model, "head", None), SpatialMILHead):
        raise ValueError("Install spatial MIL before requesting attention")
    features = model.forward_features(images)
    return model.head.forward_with_attention(features)


def benchmark_spatial_mil(gap_head: nn.Module, mil_head: SpatialMILHead,
                          features: Tensor, *, warmup: int = 20,
                          repeats: int = 100) -> dict[str, Any]:
    """Measure both heads on identical features, excluding the shared backbone.

    Callers supply the device/precision and representative feature shape. CUDA
    uses synchronized events; CPU uses a monotonic wall clock. This function
    never transfers inputs or starts a GPU session for CPU features, and its
    results are a microbenchmark, not end-to-end Study latency or an admission
    gate. Module training flags are restored even if timing fails.
    """
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if not isinstance(mil_head, SpatialMILHead):
        raise TypeError("mil_head must be a SpatialMILHead")
    if features.device.type not in ("cpu", "cuda"):
        raise ValueError("The benchmark supports CPU or CUDA features")
    modules = {module: module.training for head in (gap_head, mil_head) for module in head.modules()}
    samples: dict[str, list[float]] = {"gap": [], "mil": []}
    heads = {"gap": gap_head, "mil": mil_head}
    try:
        gap_head.eval()
        mil_head.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                gap_head(features)
                mil_head(features)
            if features.is_cuda:
                torch.cuda.synchronize(features.device)
            for index in range(repeats):
                for name in (("gap", "mil") if index % 2 == 0 else ("mil", "gap")):
                    if features.is_cuda:
                        with torch.cuda.device(features.device):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            heads[name](features)
                            end.record()
                            end.synchronize()
                            elapsed = start.elapsed_time(end)
                    else:
                        start_time = perf_counter()
                        heads[name](features)
                        elapsed = (perf_counter() - start_time) * 1000.0
                    samples[name].append(elapsed)
    finally:
        for module, training in modules.items():
            module.training = training
    def percentile95(values: list[float]) -> float:
        ordered = sorted(values)
        position = .95 * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (position - low) * (ordered[high] - ordered[low])
    paired_delta = [mil - gap for gap, mil in zip(samples["gap"], samples["mil"])]
    return {
        "scope": "head_only; shared_backbone_excluded; not_a_study_latency_gate",
        "device": str(features.device), "dtype": str(features.dtype),
        "feature_shape": list(features.shape), "warmup": warmup, "repeats": repeats,
        "additional_parameters": sum(p.numel() for p in mil_head.attention.parameters()),
        "gap_ms": {"p50": median(samples["gap"]), "p95": percentile95(samples["gap"])},
        "mil_ms": {"p50": median(samples["mil"]), "p95": percentile95(samples["mil"])},
        "paired_delta_ms": {"p50": median(paired_delta), "p95": percentile95(paired_delta)},
    }
