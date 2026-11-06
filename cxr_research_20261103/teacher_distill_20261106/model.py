"""Local public encoders with complete-Study supervision for the KD rebuild.

Construction never downloads weights.  The runner must verify the public
checkpoint's source, revision, license and SHA before passing its local
``.safetensors`` path.  A medical-task model is never accepted as a public
initialization merely because its shapes happen to match.

DINOv3 uses timm's dynamic rotary position embeddings, not a learned absolute
position table: changing 512 to 768 changes the patch grid without replacing
parameters or invalidating optimizer references.  Every original view remains
in its Study, including when the encoder is recomputed in bounded chunks.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from reconstruction.model import mean_probability_logits, study_bce_loss

NUM_CLASSES = 10
BACKBONES = {
    "dinov3_l": "vit_large_patch16_dinov3.lvd1689m",
    "convnext_l": "convnext_large.fb_in22k_ft_in1k_384",
}
_ALIASES = {
    "dino": "dinov3_l", "dinov3": "dinov3_l", "vit_l": "dinov3_l",
    "cnn": "convnext_l", "convnext": "convnext_l",
    **{name: family for family, name in BACKBONES.items()},
}


def _family(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("family must name a registered public encoder")
    result = _ALIASES.get(value, value)
    if result not in BACKBONES:
        raise ValueError(f"Unsupported encoder family: {value!r}")
    return result


def _probability(value, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return float(value)


class TeacherStudyModel(nn.Module):
    """Two public encoder families, one ten-label complete-Study interface.

    ``forward_studies`` returns one ten-logit tensor per complete Study.  Its
    sigmoid is the mean of all image probabilities, not the sigmoid of mean
    image logits.  ``return_components=True`` retains the existing trainer
    contract: study_logits, image_logits and global_study_logits.

    ``gradient_checkpointing`` controls whole-encoder chunk recomputation;
    ``block_checkpointing`` independently enables timm's block-level policy.
    Both may be enabled for the large model's 768 phase.  The runner still
    needs to measure peak memory using a complete highest-view-count Study.
    """

    def __init__(self, family: str, pretrained_path: str | Path | None = None, *,
                 image_size: int = 512, drop_rate: float = .2,
                 drop_path: float = .1, gradient_checkpointing: bool = True,
                 block_checkpointing: bool = True, view_chunk_size: int = 1,
                 channels_last: bool = False, role: str = "teacher"):
        super().__init__()
        family = _family(family)
        drop_path = _probability(drop_path, "drop_path")
        # A public checkpoint is loaded only after constructing the exact
        # registered timm topology; pretrained=False prohibits network access.
        import timm

        options = dict(pretrained=False, num_classes=0,
                       drop_path_rate=drop_path)
        if family == "dinov3_l":
            options.update(global_pool="", img_size=256,
                           dynamic_img_size=True)
        backbone = timm.create_model(BACKBONES[family], **options)
        self._configure(backbone, family=family, image_size=image_size,
            drop_rate=drop_rate, gradient_checkpointing=gradient_checkpointing,
            block_checkpointing=block_checkpointing,
            view_chunk_size=view_chunk_size, channels_last=channels_last,
            role=role)
        if pretrained_path is not None:
            self.load_public_pretrained(pretrained_path)

    @classmethod
    def from_backbone(cls, backbone: nn.Module, *, family: str,
                      image_size: int | None = None, drop_rate: float = 0.,
                      gradient_checkpointing: bool = False,
                      block_checkpointing: bool = False,
                      view_chunk_size: int = 1, channels_last: bool = False,
                      role: str = "teacher") -> "TeacherStudyModel":
        """Inject a small encoder for CPU semantics, without provenance claims."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._configure(backbone, family=_family(family),
            image_size=image_size, drop_rate=drop_rate,
            gradient_checkpointing=gradient_checkpointing,
            block_checkpointing=block_checkpointing,
            view_chunk_size=view_chunk_size, channels_last=channels_last,
            role=role)
        return instance

    def _configure(self, backbone, *, family, image_size, drop_rate,
                   gradient_checkpointing, block_checkpointing,
                   view_chunk_size, channels_last, role):
        drop_rate = _probability(drop_rate, "drop_rate")
        if any(type(value) is not bool for value in
               (gradient_checkpointing, block_checkpointing, channels_last)):
            raise TypeError("Checkpointing and memory-format switches must be bool")
        if type(view_chunk_size) is not int or view_chunk_size < 1:
            raise ValueError("view_chunk_size must be a positive integer")
        if role not in {"teacher", "student", "baseline"}:
            raise ValueError("role must be teacher, student or baseline")
        dimension = getattr(backbone, "num_features", None)
        if type(dimension) is not int or dimension < 1:
            raise ValueError("The encoder must declare a positive num_features")
        if not callable(getattr(backbone, "forward_features", None)):
            raise TypeError("The encoder must implement forward_features")
        if family == "dinov3_l":
            prefixes = getattr(backbone, "num_prefix_tokens", None)
            if (type(prefixes) is not int or prefixes < 1
                    or getattr(backbone, "cls_token", None) is None):
                raise ValueError("DINO requires CLS and its declared prefix count")
            self.num_prefix_tokens = prefixes
            head_width = 2 * dimension
            self.head = nn.Sequential(nn.LayerNorm(head_width),
                nn.Dropout(drop_rate), nn.Linear(head_width, NUM_CLASSES))
        else:
            if not callable(getattr(backbone, "forward_head", None)):
                raise TypeError("ConvNeXt must preserve its public pre-logit head")
            self.num_prefix_tokens = 0
            self.head = nn.Sequential(nn.Dropout(drop_rate),
                                      nn.Linear(dimension, NUM_CLASSES))
        nn.init.trunc_normal_(self.head[-1].weight, std=.02)
        nn.init.zeros_(self.head[-1].bias)
        self.backbone = backbone
        self.family = family
        self.backbone_name = BACKBONES[family]
        self.feature_width = dimension
        self.role = role
        self.gradient_checkpointing = gradient_checkpointing
        self.block_checkpointing = block_checkpointing
        self.view_chunk_size = view_chunk_size
        self.channels_last = channels_last
        self.image_size = None
        if image_size is not None:
            self.set_image_size(image_size)
        if block_checkpointing and not callable(getattr(backbone, "set_grad_checkpointing", None)):
            raise TypeError("Block checkpointing requested but unsupported by encoder")
        self._set_backbone_checkpointing()
        if channels_last:
            self.backbone.to(memory_format=torch.channels_last)

    def _set_backbone_checkpointing(self):
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if callable(setter):
            setter(self.block_checkpointing and self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_backbone_checkpointing()
        return self

    def set_image_size(self, image_size: int):
        """Change expected input size without replacing model parameters."""
        if type(image_size) is not int or image_size < 16:
            raise ValueError("image_size must be an integer >= 16")
        if self.family == "dinov3_l" and image_size % 16:
            raise ValueError("DINOv3 image_size must be divisible by 16")
        self.image_size = image_size
        return self

    def load_public_state(self, state: Mapping[str, Tensor]):
        """Strict timm encoder load; only the original ImageNet head is removed.

        No prefix rewriting, missing-key allowances or legacy medical head
        filtering is performed.  CNN's public head.norm is retained.  DINOv3
        is a feature encoder and must have no classification head in its file.
        """
        if (not isinstance(state, Mapping) or not state
                or any(not isinstance(k, str) or not isinstance(v, Tensor)
                       for k, v in state.items())):
            raise TypeError("Provide a nonempty public tensor state dictionary")
        prepared = dict(state)
        if self.family == "convnext_l":
            keys = {"head.fc.weight", "head.fc.bias"}.intersection(prepared)
            if keys:
                if keys != {"head.fc.weight", "head.fc.bias"}:
                    raise ValueError("Incomplete original ImageNet classifier")
                if (prepared["head.fc.weight"].shape != (1000, self.feature_width)
                        or prepared["head.fc.bias"].shape != (1000,)):
                    raise ValueError("Only the original 1000-class ImageNet head may be removed")
                del prepared["head.fc.weight"], prepared["head.fc.bias"]
        return self.backbone.load_state_dict(prepared, strict=True)

    def load_public_pretrained(self, path: str | Path):
        source = Path(path).resolve(strict=True)
        if not source.is_file() or source.suffix != ".safetensors":
            raise ValueError("Public initialization must be a local safetensors file")
        from safetensors.torch import load_file

        return self.load_public_state(load_file(str(source), device="cpu"))

    def named_parameter_groups(self):
        groups = {"backbone": [], "head": []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                groups["backbone" if name.startswith("backbone.") else "head"].append((name, parameter))
        return groups

    def _backbone_layer(self, name: str) -> tuple[int, int]:
        """Return (layer index, top encoder index), earliest layer first."""
        if self.family == "dinov3_l":
            depth = len(getattr(self.backbone, "blocks", ()))
            if depth < 1:
                raise ValueError("Layer decay requires the encoder's transformer blocks")
            match = re.match(r"blocks\.(\d+)\.", name)
            if match:
                return int(match.group(1)) + 1, depth
            if name.startswith(("norm.", "fc_norm.", "head.")):
                return depth, depth
            return 0, depth
        stages = getattr(self.backbone, "stages", None)
        if stages is None or not len(stages):
            raise ValueError("Layer decay requires ConvNeXt stages and blocks")
        depths = [len(stage.blocks) for stage in stages]
        depth = sum(depths)
        match = re.match(r"stages\.(\d+)\.(?:blocks\.(\d+)\.|downsample\.)", name)
        if match:
            stage = int(match.group(1))
            block = int(match.group(2)) if match.group(2) is not None else 0
            return sum(depths[:stage]) + block + 1, depth
        if name.startswith(("norm_pre.", "head.")):
            return depth, depth
        return 0, depth

    def optimizer_parameter_groups(self, *, backbone_lr: float, head_lr: float,
                                   weight_decay: float = .05,
                                   layer_decay: float = .85):
        """AdamW groups with exhaustive roles, layer LR and bias/norm exclusions.

        ``base_lr`` is the group's unscheduled LR; the runner should multiply
        it by the common warmup/cosine factor, preserving layer decay.  The
        serializable names/roles are included for audit and resume checks.
        """
        for name, value in (("backbone_lr", backbone_lr), ("head_lr", head_lr)):
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
                or not math.isfinite(weight_decay) or weight_decay < 0):
            raise ValueError("weight_decay must be finite and nonnegative")
        if (isinstance(layer_decay, bool) or not isinstance(layer_decay, (int, float))
                or not math.isfinite(layer_decay) or not 0 < layer_decay <= 1):
            raise ValueError("layer_decay must be finite and in (0, 1]")
        grouped = {}
        for role, named in self.named_parameter_groups().items():
            for name, parameter in named:
                short = name.removeprefix("backbone.")
                if role == "backbone":
                    layer, top = self._backbone_layer(short)
                    scale = float(layer_decay) ** (top - layer)
                    lr = float(backbone_lr) * scale
                else:
                    layer, scale, lr = -1, 1., float(head_lr)
                no_decay = (parameter.ndim <= 1 or name.endswith(".bias")
                    or short in {"cls_token", "reg_token", "pos_embed"})
                decay = 0. if no_decay else float(weight_decay)
                key = role, layer, decay
                if key not in grouped:
                    grouped[key] = dict(params=[], param_names=[], lr=lr,
                        base_lr=lr, lr_scale=scale, weight_decay=decay,
                        role=role, model_role=self.role, layer_id=layer,
                        group_name=f"{self.role}:{role}:{layer}:{decay:g}")
                grouped[key]["params"].append(parameter)
                grouped[key]["param_names"].append(name)
        return list(grouped.values())

    def _encode_chunk(self, pixels: Tensor) -> Tensor:
        features = self.backbone.forward_features(pixels)
        if self.family == "dinov3_l":
            if (not isinstance(features, Tensor) or features.ndim != 3
                    or features.shape[-1] != self.feature_width
                    or features.shape[1] <= self.num_prefix_tokens):
                raise ValueError("DINO must return CLS, registers and nonempty patch tokens")
            features = torch.cat((features[:, 0].float(),
                features[:, self.num_prefix_tokens:].float().mean(1)), dim=-1)
        else:
            features = self.backbone.forward_head(features, pre_logits=True)
            if (not isinstance(features, Tensor) or features.ndim != 2
                    or features.shape[-1] != self.feature_width):
                raise ValueError("ConvNeXt must return [images, features] pre-logits")
        # In real fp16 evaluation there is no autocast to convert a float32
        # patch mean.  Cast to the actual task head dtype before its forward.
        return self.head(features.to(dtype=self.head[-1].weight.dtype))

    def image_logits(self, pixels: Tensor) -> Tensor:
        if (not isinstance(pixels, Tensor) or pixels.ndim != 4
                or pixels.shape[1] != 3 or min(pixels.shape) < 1
                or not pixels.is_floating_point()):
            raise ValueError("Provide nonempty floating point [images,3,H,W]")
        if self.image_size is not None and pixels.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("Input resolution differs from set_image_size")
        if self.family == "dinov3_l" and (pixels.shape[-2] % 16 or pixels.shape[-1] % 16):
            raise ValueError("DINOv3 input dimensions must be divisible by 16")
        result = []
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for chunk in pixels.split(self.view_chunk_size, dim=0):
            if self.channels_last:
                chunk = chunk.contiguous(memory_format=torch.channels_last)
            logits = (checkpoint(self._encode_chunk, chunk, use_reentrant=False,
                       preserve_rng_state=True) if recompute else self._encode_chunk(chunk))
            result.append(logits.float())
        return torch.cat(result)

    def forward_studies(self, images_list: list[Tensor] | tuple[Tensor, ...], *,
                        return_components: bool = False):
        if not isinstance(images_list, (list, tuple)) or not images_list:
            raise ValueError("A pack must contain at least one complete Study")
        for images in images_list:
            if (not isinstance(images, Tensor) or images.ndim != 4
                    or images.shape[1] != 3 or min(images.shape) < 1
                    or not images.is_floating_point()):
                raise ValueError("Every complete Study must be [V,3,H,W]")
        first = images_list[0]
        if any(x.shape[1:] != first.shape[1:] or x.dtype != first.dtype
               or x.device != first.device for x in images_list):
            raise ValueError("Packed Studies must share image shape, dtype and device")
        logits = self.image_logits(torch.cat(images_list, dim=0))
        result, offset = [], 0
        for images in images_list:
            current = logits[offset:offset + len(images)]
            study_logits = mean_probability_logits(current)
            result.append(dict(study_logits=study_logits, image_logits=current,
                global_study_logits=study_logits) if return_components else study_logits)
            offset += len(images)
        return result

    def forward(self, images: Tensor, *, return_components: bool = False):
        return self.forward_studies([images], return_components=return_components)[0]
