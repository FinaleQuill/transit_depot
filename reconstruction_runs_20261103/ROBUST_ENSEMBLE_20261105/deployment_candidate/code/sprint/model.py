"""Full-frame Study classifier with a zero-initialized spatial residual.

The existing pooled ConvNeXt head is preserved. A 2x2 spatial summary adds
anatomical layout information without a second encoder or local image crops.
This module never downloads a model or discovers a checkpoint implicitly.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reconstruction.model import NUM_CLASSES, StudyModel


class SprintStudyModel(StudyModel):
    def __init__(self, backbone_name="convnext_base.fb_in22k_ft_in1k_384",
                 pretrained_path=None, *, spatial_dim=256, **kwargs):
        if kwargs.get("aggregation", "mean") != "mean" or kwargs.get("use_local", False):
            raise ValueError("Sprint uses complete-frame probability means, without local crops")
        kwargs.update(aggregation="mean", use_local=False)
        super().__init__(backbone_name, pretrained_path, **kwargs)
        self._add_spatial_head(spatial_dim)

    @classmethod
    def from_backbone(cls, backbone, *, spatial_dim=256, **kwargs):
        if kwargs.get("aggregation", "mean") != "mean" or kwargs.get("use_local", False):
            raise ValueError("Sprint uses complete-frame probability means, without local crops")
        kwargs.update(aggregation="mean", use_local=False)
        instance = super().from_backbone(backbone, **kwargs)
        instance._add_spatial_head(spatial_dim)
        return instance

    def _add_spatial_head(self, spatial_dim):
        if type(spatial_dim) is not int or spatial_dim < 1:
            raise ValueError("spatial_dim must be a positive integer")
        channels = self.backbone.get_classifier().in_features
        self.spatial_dim = spatial_dim
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(4 * channels), nn.Linear(4 * channels, spatial_dim),
            nn.GELU(), nn.Linear(spatial_dim, NUM_CLASSES))
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)

    def _encode_chunk(self, pixels: Tensor):
        spatial = self.backbone.forward_features(pixels)
        if spatial.ndim != 4 or spatial.shape[1] != self.backbone.get_classifier().in_features:
            raise ValueError("Expected ConvNeXt NCHW feature maps")
        features = self.backbone.forward_head(spatial, pre_logits=True)
        original = self.backbone.get_classifier()(features)
        summary = F.adaptive_avg_pool2d(spatial, (2, 2)).flatten(1)
        residual = self.spatial_head(summary)
        return original + residual, features

    def load_legacy_backbone(self, state: Mapping[str, Tensor], *, prefix=""):
        """Strictly restore one explicitly selected legacy state dictionary.

        The caller chooses the payload (e.g. checkpoint['state_dict']) and
        prefix (e.g. 'backbone.'); there is no guessing or missing-key fallback.
        A prefixed input must contain only that prefix so accidental new-model
        restoration cannot silently discard residual parameters.
        """
        if not isinstance(state, Mapping) or not state:
            raise TypeError("Provide a nonempty, explicitly selected state dictionary")
        if not isinstance(prefix, str) or any(not isinstance(k, str) for k in state):
            raise TypeError("State keys and prefix must be strings")
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError("Legacy state includes keys outside the declared prefix")
        prepared = {key[len(prefix):]: value for key, value in state.items()}
        return self.backbone.load_state_dict(prepared, strict=True)
