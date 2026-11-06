"""Study supervision with a shared, locally initialized ConvNeXt encoder.

``StudyModel(images)`` returns ten FP32 Study logits. ``images`` contains every
view of ONE Study: [V, 3, H, W], or [V, R, 3, H, W] with region zero the full
image. Local regions require ``use_local=True`` and never receive patch labels.
With ``return_components=True`` a dictionary also exposes global_study_logits,
image_logits [V, 10], region_logits [V, R, 10], region_attention [V, R, 10],
and view_attention [V, 10]. Attention maps normalize within their own hierarchy.

Both new residual classifiers start at zero: E1 and local models initially
produce exactly the same aggregation as the full-image probability-mean model.
The residual classifiers then learn from the Study loss; their feature and
attention parameters receive gradients once those classifiers leave zero.

No download occurs on import or when pretrained_path=None. A supplied path is
passed to the existing approved model factory as a LOCAL public initialization.
The training boundary must verify that file's public-pretrain hash/provenance;
None is intended for strict restoration, not an implicit old-model warm start.
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

NUM_CLASSES = 10


def mean_probability_logits(logits: Tensor, dim: int = 0) -> Tensor:
    """Stable logit(mean(sigmoid(logits))); do not average the input logits.

    log(count) cancels between log-positive and log-negative probabilities.
    Unlike clamping probabilities this remains finite and differentiable for
    confidently incorrect finite inputs, including logits of magnitude 1000.
    """
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("Probability aggregation requires floating-point logits")
    if logits.ndim == 0 or not -logits.ndim <= dim < logits.ndim or logits.shape[dim] < 1:
        raise ValueError("Cannot aggregate an empty or invalid dimension")
    z = logits.float()
    return torch.logsumexp(F.logsigmoid(z), dim=dim) - torch.logsumexp(F.logsigmoid(-z), dim=dim)


def study_bce_loss(study_logits: Tensor, targets: Tensor,
                   pos_weight: Tensor | None = None, *,
                   global_study_logits: Tensor | None = None,
                   global_aux_weight: float = 0.0) -> Tensor:
    """Ten-class Study BCE, optionally plus the GLOBAL Study branch's BCE.

    Accepts precisely [10] labels for one Study. In particular [V,10] and
    [V,R,10] labels cannot accidentally broadcast into per-view/patch targets.
    Count pos_weight from only the active training split before calling this.
    """
    if study_logits.shape != (NUM_CLASSES,) or targets.shape != (NUM_CLASSES,):
        raise ValueError("Study BCE requires exactly ten Study logits and labels")
    if (not isinstance(global_aux_weight, (int, float))
            or not math.isfinite(global_aux_weight) or global_aux_weight < 0):
        raise ValueError("global_aux_weight must be finite and nonnegative")
    if not bool(torch.isfinite(study_logits).all()) or not bool(torch.isfinite(targets).all()):
        raise FloatingPointError("Nonfinite Study logits or targets")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("Study supervision requires original binary targets")
    y = targets.to(device=study_logits.device, dtype=torch.float32)
    weight = None
    if pos_weight is not None:
        if (pos_weight.shape != (NUM_CLASSES,)
                or not bool(torch.isfinite(pos_weight).all())
                or not bool((pos_weight > 0).all())):
            raise ValueError("pos_weight must contain ten finite positive values")
        weight = pos_weight.to(device=study_logits.device, dtype=torch.float32)
    with torch.autocast(device_type=study_logits.device.type, enabled=False):
        loss = F.binary_cross_entropy_with_logits(study_logits.float(), y, pos_weight=weight)
        if global_aux_weight:
            if global_study_logits is None or global_study_logits.shape != (NUM_CLASSES,):
                raise ValueError("The global auxiliary loss requires ten global Study logits")
            if not bool(torch.isfinite(global_study_logits).all()):
                raise FloatingPointError("Nonfinite global Study logits")
            loss = loss + global_aux_weight * F.binary_cross_entropy_with_logits(
                global_study_logits.float(), y, pos_weight=weight)
    return loss


class StudyModel(nn.Module):
    """Shared encoder plus an optional class-specific region/view hierarchy.

    ``aggregation='mean'`` is E0 (or E2 when use_local=True). ``'attention'``
    adds a residual feature branch after the complete-view probability mean.
    Region coordinates, when supplied, have shape [V,R,4] and contain normalized
    source-image (left, top, right, bottom), transformed with any image geometry.
    Full-image region zero must be (0,0,1,1). Without coordinates no positional
    features are inferred from tensor order, preserving region permutation
    invariance (the full-image slot itself must always remain first).

    View chunks are activation-checkpointed during training to retain every
    original view without retaining every chunk's backbone activations at once.
    Eval uses the same complete aggregation; no view is sampled or truncated.
    """

    def __init__(self, backbone_name: str, pretrained_path: str | Path | None,
                 aggregation: str = "mean", drop_rate: float = 0.2,
                 drop_path: float = 0.15, gradient_checkpointing: bool = True,
                 view_chunk_size: int = 4, use_local: bool = False,
                 attention_dim: int = 256):
        super().__init__()
        from model_overnight import build_overnight_model

        local_path = None if pretrained_path is None else str(Path(pretrained_path).resolve(strict=True))
        backbone = build_overnight_model(
            pretrained=local_path is not None, pretrained_weights_path=local_path,
            backbone_name=backbone_name, drop_rate=drop_rate, drop_path_rate=drop_path,
            gradient_checkpointing=gradient_checkpointing)
        self._configure(backbone, aggregation=aggregation, gradient_checkpointing=gradient_checkpointing,
                        view_chunk_size=view_chunk_size, use_local=use_local, attention_dim=attention_dim)
        self.backbone_name = backbone_name

    @classmethod
    def from_backbone(cls, backbone: nn.Module, *, aggregation: str = "mean",
                      gradient_checkpointing: bool = False, view_chunk_size: int = 4,
                      use_local: bool = False, attention_dim: int = 256) -> "StudyModel":
        """Dependency-injected construction for CPU semantic tests, no downloads.

        Training should use the public constructor and its approved-backbone
        factory; this hook deliberately makes no weight-provenance claim.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._configure(backbone, aggregation=aggregation,
                            gradient_checkpointing=gradient_checkpointing,
                            view_chunk_size=view_chunk_size, use_local=use_local,
                            attention_dim=attention_dim)
        instance.backbone_name = type(backbone).__name__
        return instance

    def _configure(self, backbone: nn.Module, *, aggregation: str,
                   gradient_checkpointing: bool, view_chunk_size: int,
                   use_local: bool, attention_dim: int) -> None:
        if aggregation not in {"mean", "attention"}:
            raise ValueError("aggregation must be mean or attention")
        if type(gradient_checkpointing) is not bool or type(use_local) is not bool:
            raise TypeError("Checkpointing and local-region switches must be boolean")
        if type(view_chunk_size) is not int or view_chunk_size < 1:
            raise ValueError("view_chunk_size must be a positive integer")
        if type(attention_dim) is not int or attention_dim < 1:
            raise ValueError("attention_dim must be a positive integer")
        if not all(callable(getattr(backbone, name, None)) for name in
                   ("forward_features", "forward_head", "get_classifier")):
            raise TypeError("The encoder must expose features, a pre-logit head and its classifier")
        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear) or classifier.out_features != NUM_CLASSES:
            raise ValueError("The encoder classifier must be a ten-class linear layer")
        self.backbone = backbone
        self.aggregation = aggregation
        self.gradient_checkpointing = gradient_checkpointing
        self.view_chunk_size = view_chunk_size
        self.use_local = use_local
        self.attention_dim = attention_dim
        if aggregation == "attention" or use_local:
            self.feature_projection = nn.Sequential(
                nn.LayerNorm(classifier.in_features), nn.Linear(classifier.in_features, attention_dim), nn.GELU())
        if use_local:
            self.coordinate_projection = nn.Linear(6, attention_dim, bias=False)
            self.region_query = nn.Parameter(torch.zeros(NUM_CLASSES, attention_dim))
            self.region_residual_weight = nn.Parameter(torch.zeros(NUM_CLASSES, attention_dim))
            self.region_residual_bias = nn.Parameter(torch.zeros(NUM_CLASSES))
        if aggregation == "attention":
            self.view_query = nn.Parameter(torch.zeros(NUM_CLASSES, attention_dim))
            self.view_residual_weight = nn.Parameter(torch.zeros(NUM_CLASSES, attention_dim))
            self.view_residual_bias = nn.Parameter(torch.zeros(NUM_CLASSES))

    def named_parameter_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """Split optimizer roles, retaining pretrained head.norm at backbone LR.

        The existing ten-class FC is a head even though its registered name is
        backbone.head.fc.*. All new region/view modules are also head parameters.
        Weight-decay/no-decay splitting can then be applied by the trainer.
        """
        classifier_ids = {id(p) for p in self.backbone.get_classifier().parameters()}
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {"backbone": [], "head": []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                role = "backbone" if name.startswith("backbone.") and id(parameter) not in classifier_ids else "head"
                groups[role].append((name, parameter))
        return groups

    def _encode_chunk(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.backbone.forward_features(pixels)
        # ConvNeXt's pretrained norm remains after pooling, as in its original
        # head. pre_logits also preserves its dropout policy during training.
        features = self.backbone.forward_head(spatial, pre_logits=True)
        logits = self.backbone.get_classifier()(features)
        return logits, features

    def _encode(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        all_logits, all_features = [], []
        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for chunk in pixels.split(self.view_chunk_size, dim=0):
            if use_checkpoint:
                logits, features = checkpoint(self._encode_chunk, chunk, use_reentrant=False,
                                              preserve_rng_state=True)
            else:
                logits, features = self._encode_chunk(chunk)
            all_logits.append(logits)
            all_features.append(features)
        return torch.cat(all_logits), torch.cat(all_features)

    @staticmethod
    def _coordinate_features(coordinates: Tensor, views: int, regions: int,
                             *, device: torch.device) -> Tensor:
        if coordinates.shape != (views, regions, 4):
            raise ValueError("region_coordinates must be [views, regions, 4] normalized xyxy")
        xyxy = coordinates.to(device=device, dtype=torch.float32)
        if (not bool(torch.isfinite(xyxy).all()) or bool(((xyxy < 0) | (xyxy > 1)).any())
                or bool((xyxy[..., 2:] <= xyxy[..., :2]).any())):
            raise ValueError("Region coordinates must be finite, nonempty boxes inside the image")
        full = xyxy.new_tensor([0., 0., 1., 1.]).expand(views, 4)
        if not torch.equal(xyxy[:, 0], full):
            raise ValueError("Region zero must be the complete global image")
        center = (xyxy[..., :2] + xyxy[..., 2:]) * 0.5
        size = xyxy[..., 2:] - xyxy[..., :2]
        return torch.cat((center, size, size.log()), dim=-1)

    def forward(self, images: Tensor, *, return_components: bool = False,
                region_coordinates: Tensor | None = None) -> Tensor | dict[str, Tensor]:
        if not isinstance(images, Tensor) or images.ndim not in (4, 5):
            raise ValueError("Provide all views of one Study as [V,3,H,W] or [V,R,3,H,W]")
        if not images.is_floating_point() or min(images.shape) < 1 or images.shape[-3] != 3:
            raise ValueError("Study pixels must be a nonempty floating-point RGB tensor")
        if images.ndim == 4:
            images = images.unsqueeze(1)
        views, regions = images.shape[:2]
        if regions != 1 and not self.use_local:
            raise ValueError("Multiple regions require use_local=True; region zero must be global")
        if region_coordinates is not None and not self.use_local:
            raise ValueError("Coordinates require an enabled local-region model")
        logits, features = self._encode(images.reshape(-1, *images.shape[-3:]))
        region_logits = logits.float().reshape(views, regions, NUM_CLASSES)
        global_logits = region_logits[:, 0]
        projected = None
        if self.use_local or self.aggregation == "attention":
            projected = self.feature_projection(features).reshape(views, regions, self.attention_dim)
            if region_coordinates is not None:
                positions = self._coordinate_features(region_coordinates, views, regions, device=images.device)
                projected = projected + self.coordinate_projection(positions.to(projected.dtype))
        # Explicitly disable autocast for all probability reductions and the
        # small class-specific residuals, not only for the final BCE.
        with torch.autocast(device_type=images.device.type, enabled=False):
            global_study_logits = mean_probability_logits(global_logits)
            region_attention = region_logits.new_full((views, regions, NUM_CLASSES), 1. / regions)
            image_logits = global_logits
            if self.use_local:
                assert projected is not None
                latent = projected.float()
                scores = torch.einsum("vrd,cd->vrc", latent, self.region_query.float()) / math.sqrt(self.attention_dim)
                region_attention = scores.softmax(dim=1)
                class_features = torch.einsum("vrc,vrd->vcd", region_attention, latent)
                image_logits = global_logits + (class_features * self.region_residual_weight.float()).sum(-1)
                image_logits = image_logits + self.region_residual_bias.float()
            elif self.aggregation == "attention":
                assert projected is not None
                class_features = projected[:, 0].float().unsqueeze(1).expand(-1, NUM_CLASSES, -1)
            study_logits = mean_probability_logits(image_logits)
            view_attention = region_logits.new_full((views, NUM_CLASSES), 1. / views)
            if self.aggregation == "attention":
                scores = (class_features * self.view_query.float()).sum(-1) / math.sqrt(self.attention_dim)
                view_attention = scores.softmax(dim=0)
                study_features = (view_attention.unsqueeze(-1) * class_features).sum(0)
                study_logits = study_logits + (study_features * self.view_residual_weight.float()).sum(-1)
                study_logits = study_logits + self.view_residual_bias.float()
        if not return_components:
            return study_logits
        return {"study_logits": study_logits, "global_study_logits": global_study_logits,
                "image_logits": image_logits, "region_logits": region_logits,
                "region_attention": region_attention, "view_attention": view_attention}

