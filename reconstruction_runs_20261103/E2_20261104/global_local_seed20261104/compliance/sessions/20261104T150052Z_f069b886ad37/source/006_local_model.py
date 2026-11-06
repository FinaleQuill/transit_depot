"""Isolated E2 candidate: image-local feature fusion, then Study probability mean.

This module does not register or launch an experiment. Its caller must verify
the E0 checkpoint's new-chain ancestry and pre-register a matched continuation
control before use. Frozen E0/E1 modules remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn

from reconstruction.model import NUM_CLASSES, StudyModel, mean_probability_logits


class LocalStudyModel(StudyModel):
    """One shared backbone; each image has a global region and optional locals.

    Global features are encoded first with E0's chunk order. At zero residual
    initialization the same backbone, inputs and RNG state therefore preserve
    the E0 output, including its stochastic global path. Additional local
    encoding consumes RNG, so subsequent training randomness is not claimed to
    remain bitwise paired with E0 over multiple optimizer steps.

    Per-class attention has two groups: the global region and all locals.
    A log(count) correction gives both groups initial mass 1/2, independent of
    the number of locals. Attention remains image-local; Study aggregation is
    always the complete-view probability mean. With no locals it falls back
    exactly to the global path, even after the residual head has been trained.
    """
    def __init__(self, backbone_name, pretrained_path, *, drop_rate=.2, drop_path=.15,
                 gradient_checkpointing=True, view_chunk_size=4, attention_dim=256,
                 checkpoint_mode="chunks", channels_last=False):
        super().__init__(backbone_name, pretrained_path, aggregation="mean", use_local=True,
                         drop_rate=drop_rate, drop_path=drop_path,
                         gradient_checkpointing=gradient_checkpointing,
                         view_chunk_size=view_chunk_size, attention_dim=attention_dim,
                         checkpoint_mode=checkpoint_mode, channels_last=channels_last)

    @classmethod
    def from_backbone(cls, backbone: nn.Module, *, gradient_checkpointing=False,
                      view_chunk_size=4, attention_dim=256, checkpoint_mode="chunks",
                      channels_last=False):
        """CPU semantic-test construction; makes no public-weight provenance claim."""
        return super().from_backbone(backbone, aggregation="mean", use_local=True,
                                    gradient_checkpointing=gradient_checkpointing,
                                    view_chunk_size=view_chunk_size, attention_dim=attention_dim,
                                    checkpoint_mode=checkpoint_mode, channels_last=channels_last)

    def load_e0_state_dict(self, state: Mapping[str, Tensor]):
        """Load only an exact E0 backbone state into a fresh zero-residual model.

        No filenames are trusted here and old/extra aggregation parameters are
        rejected. The training boundary must independently verify checkpoint
        SHA, split hashes, E0 configuration and its public initialization chain.
        This method intentionally does not load optimizer/EMA counters.
        """
        expected = {k: v for k, v in self.state_dict().items() if k.startswith("backbone.")}
        if set(state) != set(expected):
            raise ValueError("E0 transfer requires exactly the global backbone state")
        for key, target in expected.items():
            value = state[key]
            if not isinstance(value, Tensor) or value.shape != target.shape:
                raise ValueError(f"E0 transfer shape mismatch: {key}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite E0 transfer tensor: {key}")
        if bool(self.region_residual_weight.detach().count_nonzero()) or bool(self.region_residual_bias.detach().count_nonzero()):
            raise ValueError("E0 transfer is allowed only before the local residual has trained")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != set(self.state_dict()) - set(expected):
            raise RuntimeError("Unexpected state transfer result")
        return result

    @staticmethod
    def _geometry_features(affines: Tensor | None, views: int, regions: int, device):
        if affines is None:
            if regions != 1:
                raise ValueError("Original-pixel local inputs require explicit region_affines")
            return None
        if not isinstance(affines, Tensor) or affines.shape != (views, regions, 2, 3):
            raise ValueError("region_affines must be [views, regions, 2, 3]")
        maps = affines.to(device=device, dtype=torch.float32)
        full = maps.new_tensor([[1., 0., 0.], [0., 1., 0.]]).expand(views, -1, -1)
        if not bool(torch.isfinite(maps).all()) or not torch.equal(maps[:, 0], full):
            raise ValueError("Region affines must be finite with identity global maps")
        determinant = maps[..., 0, 0]*maps[..., 1, 1] - maps[..., 0, 1]*maps[..., 1, 0]
        if bool((determinant <= 0).any()):
            raise ValueError("Tile-to-global maps require nonsingular consistent geometry")
        return maps.flatten(-2)

    def _aggregate_local(self, region_logits, features, geometry, return_components):
        views, regions = region_logits.shape[:2]
        projected = None
        if regions > 1:
            projected = self.feature_projection(features)
            projected = projected + self.coordinate_projection(geometry.to(projected.dtype))
        with torch.autocast(device_type=features.device.type, enabled=False):
            global_logits = region_logits[:, 0].float()
            global_study_logits = mean_probability_logits(global_logits)
            image_logits = global_logits
            region_attention = global_logits.new_ones((views, regions, NUM_CLASSES))
            if regions > 1:
                latent = projected.float()
                scores = torch.einsum("vrd,cd->vrc", latent, self.region_query.float()) / math.sqrt(self.attention_dim)
                # A branch represents one global image or one SET of locals.
                # Replicating the entire local set must not manufacture branch
                # mass. Duplicating only one distinct tile changes its weight.
                prior = scores.new_zeros(regions)
                prior[1:] = math.log(regions - 1)
                region_attention = (scores - prior[None, :, None]).softmax(dim=1)
                class_features = torch.einsum("vrc,vrd->vcd", region_attention, latent)
                residual = (class_features*self.region_residual_weight.float()).sum(-1)
                image_logits = global_logits + residual + self.region_residual_bias.float()
            study_logits = mean_probability_logits(image_logits)
            group_attention = torch.stack((region_attention[:, 0], region_attention[:, 1:].sum(1)), dim=1)
        if not return_components:
            return study_logits
        return {"study_logits": study_logits, "global_study_logits": global_study_logits,
                "image_logits": image_logits, "region_logits": region_logits,
                "region_attention": region_attention, "group_attention": group_attention,
                "view_attention": global_logits.new_full((views, NUM_CLASSES), 1/views)}

    def forward_studies(self, images_list, *, return_components=False, region_affines=None):
        """Encode complete Study packs in bounded chunks, without dropping views."""
        if not isinstance(images_list, (list, tuple)) or not images_list:
            raise ValueError("A pack requires at least one complete Study")
        affines = [None]*len(images_list) if region_affines is None else region_affines
        if not isinstance(affines, (list, tuple)) or len(affines) != len(images_list):
            raise ValueError("Supply one region affine tensor per Study")
        studies = [self._study_shape(images, None) for images in images_list]
        first = studies[0]
        if any(x.shape[-3:] != first.shape[-3:] or x.device != first.device or x.dtype != first.dtype for x in studies):
            raise ValueError("Packed inputs must share shape, device and dtype")
        geometries = [self._geometry_features(a, *s.shape[:2], s.device) for s, a in zip(studies, affines)]
        global_pixels = torch.cat([s[:, 0] for s in studies], dim=0)
        global_logits, global_features = self._encode(global_pixels)
        local_pixels = [s[:, 1:].reshape(-1, *s.shape[-3:]) for s in studies if s.shape[1] > 1]
        local_logits = local_features = None
        if local_pixels:
            local_logits, local_features = self._encode(torch.cat(local_pixels, dim=0))
        output, global_offset, local_offset = [], 0, 0
        for study, geometry in zip(studies, geometries):
            views, regions = study.shape[:2]
            end = global_offset + views
            logits = global_logits[global_offset:end, None]
            features = global_features[global_offset:end, None]
            count = views*(regions-1)
            if count:
                logits = torch.cat((logits, local_logits[local_offset:local_offset+count].reshape(views, regions-1, NUM_CLASSES)), dim=1)
                features = torch.cat((features, local_features[local_offset:local_offset+count].reshape(views, regions-1, -1)), dim=1)
            output.append(self._aggregate_local(logits.float(), features, geometry, return_components))
            global_offset, local_offset = end, local_offset + count
        return output

    def forward(self, images, *, return_components=False, region_affines=None):
        return self.forward_studies([images], return_components=return_components,
                                    region_affines=[region_affines])[0]
