"""Two complementary full-frame Tiny branches using one encoder per image.

The global branch preserves the original pooled classifier. The opacity branch
adds ten class-specific spatial attention pools from that same feature map,
with a zero-initialized residual gate preserving warm-start predictions.
Complete Studies retain the inherited probability-mean aggregation.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from reconstruction.model import NUM_CLASSES, StudyModel


class EnsembleStudyModel(StudyModel):
    def __init__(self, backbone_name="convnext_tiny.fb_in22k_ft_in1k_384",
                 pretrained_path=None, *, branch="global", **kwargs):
        self._validate_branch(branch, kwargs)
        kwargs.update(aggregation="mean", use_local=False)
        super().__init__(backbone_name, pretrained_path, **kwargs)
        self._configure_branch(branch)

    @classmethod
    def from_backbone(cls, backbone, *, branch="global", **kwargs):
        cls._validate_branch(branch, kwargs)
        kwargs.update(aggregation="mean", use_local=False)
        instance = super().from_backbone(backbone, **kwargs)
        instance._configure_branch(branch)
        return instance

    @staticmethod
    def _validate_branch(branch, kwargs):
        if branch not in {"global", "opacity"}:
            raise ValueError("branch must be global or opacity")
        if kwargs.get("aggregation", "mean") != "mean" or kwargs.get("use_local", False):
            raise ValueError("Ensemble members use full-frame complete-Study means")

    def _configure_branch(self, branch):
        self.branch = branch
        if branch == "opacity":
            channels = self.backbone.get_classifier().in_features
            self.spatial_attention = nn.Conv2d(channels, NUM_CLASSES, 1, bias=False)
            # Small nonuniform attention gives the zero gate a useful first-step
            # gradient. A zero attention kernel would duplicate GAP and make
            # both this difference and its gate gradient zero at initialization.
            nn.init.normal_(self.spatial_attention.weight, mean=0., std=.01)
            self.spatial_gate = nn.Parameter(torch.zeros(NUM_CLASSES))

    def _opacity_logits(self, spatial: Tensor, original: Tensor) -> Tensor:
        batch, channels, _, _ = spatial.shape
        scores = self.spatial_attention(spatial).flatten(2)
        # FP32 reductions keep normalization stable under autocast and FP16
        # exports. Only the compact pooled tensors reenter the original head.
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            attention = scores.float().softmax(-1)
            pooled = torch.einsum("nks,ncs->nkc", attention, spatial.float().flatten(2))
        head_input = pooled.reshape(batch * NUM_CLASSES, channels, 1, 1).to(spatial.dtype)
        class_features = self.backbone.forward_head(head_input, pre_logits=True)
        class_features = class_features.reshape(batch, NUM_CLASSES, channels)
        classifier = self.backbone.get_classifier()
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            focused = (class_features.float() * classifier.weight.float().unsqueeze(0)).sum(-1)
            if classifier.bias is not None:
                focused = focused + classifier.bias.float()
            # Bounded class-specific residual; zero gates reproduce the restored
            # GAP output exactly, without adding a second encoder forward.
            return original.float() + self.spatial_gate.float().tanh() * (focused - original.float())

    def _encode_chunk(self, pixels: Tensor):
        if self.branch == "global":
            return super()._encode_chunk(pixels)
        spatial = self.backbone.forward_features(pixels)
        if spatial.ndim != 4 or spatial.shape[1] != self.backbone.get_classifier().in_features:
            raise ValueError("Expected ConvNeXt NCHW feature maps")
        features = self.backbone.forward_head(spatial, pre_logits=True)
        original = self.backbone.get_classifier()(features)
        return self._opacity_logits(spatial, original), features

    def load_legacy_backbone(self, state: Mapping[str, Tensor], *, prefix=""):
        """Strict restoration of an explicitly selected task-trained backbone."""
        if not isinstance(state, Mapping) or not state:
            raise TypeError("Provide a nonempty, explicitly selected state dictionary")
        if not isinstance(prefix, str) or any(not isinstance(key, str) for key in state):
            raise TypeError("State keys and prefix must be strings")
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError("Legacy state includes keys outside the declared prefix")
        prepared = {key[len(prefix):]: value for key, value in state.items()}
        return self.backbone.load_state_dict(prepared, strict=True)
