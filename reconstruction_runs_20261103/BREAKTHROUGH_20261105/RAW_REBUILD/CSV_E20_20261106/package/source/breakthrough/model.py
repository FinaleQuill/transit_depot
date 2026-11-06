"""Fresh public DINOv2 features with complete-Study classification.

The caller verifies the local public checkpoint's source and SHA256 before
construction. Import and restoration never download weights. DINO register
tokens are excluded from the patch average; each image contributes one CLS
and one mean patch vector. Supervision acts on complete Study probability
means, with no per-image labels or cross-patient/context-dependent operation.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from reconstruction.model import mean_probability_logits

BACKBONE_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
NUM_CLASSES = 10
NOMINAL_IMAGE_SIZE = 518


class DinoStudyModel(nn.Module):
    """A token encoder, CLS/patch head and stable all-view probability mean.

    ``forward_studies`` accepts a list of complete ``[V,3,H,W]`` Studies and
    returns one ten-logit tensor per Study. Image sizes may change between
    calls, provided both dimensions are divisible by the 14-pixel patch size.
    The original 518 positional embedding is preserved and interpolated by
    timm's dynamic input path. Chunk checkpointing retains all image gradients
    without retaining activations for every image at once.
    """

    def __init__(self, pretrained_path: str | Path | None = None, *,
                 backbone_name: str = BACKBONE_NAME, image_size: int = 518,
                 drop_rate: float = .2, drop_path: float = .1,
                 gradient_checkpointing: bool = True, view_chunk_size: int = 4,
                 channels_last: bool = False):
        super().__init__()
        if backbone_name != BACKBONE_NAME:
            raise ValueError(f"This protocol requires the registered {BACKBONE_NAME}")
        if type(image_size) is not int or image_size < 14 or image_size % 14:
            raise ValueError("image_size must be a positive multiple of 14")
        if not isinstance(drop_path, (float, int)) or not math.isfinite(drop_path) or not 0 <= drop_path < 1:
            raise ValueError("drop_path must be finite and in [0,1)")
        import timm

        # Keep the public checkpoint's native position-grid shape. Dynamic
        # interpolation supports training/evaluation at other explicit sizes.
        backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="",
            img_size=NOMINAL_IMAGE_SIZE, dynamic_img_size=True,
            drop_path_rate=float(drop_path))
        self._configure(backbone, drop_rate=drop_rate,
                        gradient_checkpointing=gradient_checkpointing,
                        view_chunk_size=view_chunk_size, channels_last=channels_last)
        self.backbone_name = backbone_name
        self.image_size = image_size
        if pretrained_path is not None:
            self.load_public_pretrained(pretrained_path)

    @classmethod
    def from_backbone(cls, backbone: nn.Module, *, drop_rate: float = .2,
                      gradient_checkpointing: bool = False, view_chunk_size: int = 4,
                      channels_last: bool = False) -> "DinoStudyModel":
        """Dependency injection for CPU semantics; makes no provenance claim."""
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model._configure(backbone, drop_rate=drop_rate,
                         gradient_checkpointing=gradient_checkpointing,
                         view_chunk_size=view_chunk_size, channels_last=channels_last)
        model.backbone_name = type(backbone).__name__
        model.image_size = None
        return model

    def _configure(self, backbone, *, drop_rate, gradient_checkpointing,
                   view_chunk_size, channels_last):
        if not isinstance(drop_rate, (float, int)) or not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
            raise ValueError("drop_rate must be finite and in [0,1)")
        if type(gradient_checkpointing) is not bool or type(channels_last) is not bool:
            raise TypeError("Checkpointing and layout switches must be boolean")
        if type(view_chunk_size) is not int or view_chunk_size < 1:
            raise ValueError("view_chunk_size must be positive")
        dimension, prefixes = getattr(backbone, "num_features", None), getattr(backbone, "num_prefix_tokens", None)
        if not callable(getattr(backbone, "forward_features", None)):
            raise TypeError("The encoder must expose forward_features")
        if type(dimension) is not int or dimension < 1 or type(prefixes) is not int or prefixes < 1:
            raise ValueError("The encoder must declare feature width and CLS/register prefix count")
        if getattr(backbone, "cls_token", None) is None:
            raise ValueError("A real CLS token is required")
        self.backbone = backbone
        self.num_prefix_tokens = prefixes
        self.feature_width = dimension
        self.gradient_checkpointing = gradient_checkpointing
        self.view_chunk_size = view_chunk_size
        self.channels_last = channels_last
        # Outer chunk checkpointing already recomputes the complete encoder;
        # avoid nesting a second checkpoint policy inside transformer blocks.
        if callable(getattr(backbone, "set_grad_checkpointing", None)):
            backbone.set_grad_checkpointing(False)
        self.head = nn.Sequential(nn.LayerNorm(2 * dimension), nn.Dropout(float(drop_rate)),
                                  nn.Linear(2 * dimension, NUM_CLASSES))
        nn.init.trunc_normal_(self.head[-1].weight, std=.02)
        nn.init.zeros_(self.head[-1].bias)
        if channels_last:
            self.backbone.to(memory_format=torch.channels_last)

    def load_public_state(self, state: Mapping[str, Tensor]):
        """Load a full timm public encoder; only a 1000-class head may be removed.

        No prefix guessing, non-strict restore or partial legacy warm start is
        permitted. The caller records source/revision/hash, not this method.
        """
        if not isinstance(state, Mapping) or not state or any(not isinstance(k, str) for k in state):
            raise TypeError("Provide a nonempty public tensor state dictionary")
        prepared = dict(state)
        classifier_keys = {"head.weight", "head.bias"}.intersection(prepared)
        if classifier_keys:
            if classifier_keys != {"head.weight", "head.bias"}:
                raise ValueError("Incomplete public classification head")
            if (prepared["head.weight"].shape != (1000, self.feature_width)
                    or prepared["head.bias"].shape != (1000,)):
                raise ValueError("Only the original 1000-class public head can be discarded")
            del prepared["head.weight"], prepared["head.bias"]
        return self.backbone.load_state_dict(prepared, strict=True)

    def load_public_pretrained(self, path: str | Path):
        source = Path(path).resolve(strict=True)
        if source.suffix != ".safetensors":
            raise ValueError("The registered public initialization must be a safetensors file")
        from safetensors.torch import load_file

        return self.load_public_state(load_file(str(source), device="cpu"))

    def named_parameter_groups(self):
        groups = {"backbone": [], "head": []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                groups["backbone" if name.startswith("backbone.") else "head"].append((name, parameter))
        return groups

    def _encode_chunk(self, pixels: Tensor) -> Tensor:
        tokens = self.backbone.forward_features(pixels)
        if (not isinstance(tokens, Tensor) or tokens.ndim != 3
                or tokens.shape[-1] != self.feature_width
                or tokens.shape[1] <= self.num_prefix_tokens):
            raise ValueError("Expected CLS, optional registers and nonempty patch tokens")
        # Mean in FP32 avoids summing 1369 patch tokens in FP16.
        patches = tokens[:, self.num_prefix_tokens:].float().mean(dim=1)
        features = torch.cat((tokens[:, 0].float(), patches), dim=-1)
        return self.head(features.to(tokens.dtype))

    def image_logits(self, pixels: Tensor) -> Tensor:
        if (not isinstance(pixels, Tensor) or pixels.ndim != 4
                or pixels.shape[1] != 3 or min(pixels.shape) < 1
                or not pixels.is_floating_point()):
            raise ValueError("Provide nonempty floating point [images,3,H,W] pixels")
        outputs = []
        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for chunk in pixels.split(self.view_chunk_size, dim=0):
            if self.channels_last:
                chunk = chunk.contiguous(memory_format=torch.channels_last)
            logits = (checkpoint(self._encode_chunk, chunk, use_reentrant=False, preserve_rng_state=True)
                      if use_checkpoint else self._encode_chunk(chunk))
            outputs.append(logits.float())
        return torch.cat(outputs)

    def forward_studies(self, images_list: list[Tensor] | tuple[Tensor, ...], *,
                        return_components: bool = False):
        if not isinstance(images_list, (list, tuple)) or not images_list:
            raise ValueError("A pack must contain at least one complete Study")
        for images in images_list:
            if (not isinstance(images, Tensor) or images.ndim != 4
                    or images.shape[1] != 3 or min(images.shape) < 1
                    or not images.is_floating_point()):
                raise ValueError("Every complete Study must have shape [V,3,H,W]")
        first = images_list[0]
        if any(x.shape[1:] != first.shape[1:] or x.device != first.device or x.dtype != first.dtype
               for x in images_list):
            raise ValueError("Packed Studies must share image shape, device and dtype")
        logits = self.image_logits(torch.cat(images_list, dim=0))
        results, offset = [], 0
        for images in images_list:
            current = logits[offset:offset + len(images)]
            study_logits = mean_probability_logits(current)
            results.append({"study_logits": study_logits, "image_logits": current,
                            "global_study_logits": study_logits} if return_components else study_logits)
            offset += len(images)
        return results

    def forward(self, images: Tensor, *, return_components: bool = False):
        return self.forward_studies([images], return_components=return_components)[0]
