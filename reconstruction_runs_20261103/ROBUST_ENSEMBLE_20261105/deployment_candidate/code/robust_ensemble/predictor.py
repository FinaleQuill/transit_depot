"""Fixed Sprint768 pair plus the existing final opacity512 model.

The original two-member Sprint probabilities are computed in their original
order and precision. A fixed 20% opacity contribution is then added for classes
2--10; class 1 keeps the original Sprint Base result exactly. Prediction, final
validation and timing all use this implementation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from deploy_kd import (MAX_MODEL_BYTES, MEAN, STD, StudyAccumulator,
                       _letterbox_image, aggregate_member_probabilities,
                       sha256_file)
from ensemble_sprint.predictor import TINY_BACKBONE, checked_member_payload, study_groups
from sprint.evaluate import MODEL_WEIGHTS

FUSION_WEIGHTS = [[1.0] + [0.8] * 9, [0.0] + [0.2] * 9]


def validate_recipe(config):
    """Reject any silent blend, epoch, input or member changes before loading."""
    weights = np.asarray(config.get("fusion_weights"), dtype=np.float64)
    if weights.shape != (2, 10) or not np.array_equal(weights, FUSION_WEIGHTS):
        raise ValueError("Require the single registered 80/20 Sprint/opacity fusion")
    sprint = config.get("sprint_config")
    if not isinstance(sprint, dict):
        raise ValueError("Declare the retained Sprint configuration")
    if (sprint.get("image_size", 768) != 768
            or sprint.get("tta", "hflip2") != "hflip2"
            or sprint.get("patient_smoothing", False)
            or sprint.get("fixed_classwise_probability_weights", MODEL_WEIGHTS) != MODEL_WEIGHTS):
        raise ValueError("The retained Sprint recipe must remain unchanged")
    member = config.get("opacity_member")
    if (not isinstance(member, dict) or member.get("id") != "opacity512"
            or member.get("branch") != "opacity" or member.get("image_size") != 512
            or member.get("epochs") != 6
            or member.get("backbone_name", TINY_BACKBONE) != TINY_BACKBONE):
        raise ValueError("Require the previously completed opacity512 final epoch 6")
    if config.get("tta", "hflip2") != "hflip2" or config.get("patient_smoothing", False):
        raise ValueError("Require deterministic hflip2 and no patient smoothing")
    for source, path_field, hash_field in (
            (sprint, "candidate_weights", "candidate_weights_sha256"),
            (member, "weights", "weights_sha256")):
        if not isinstance(source.get(path_field), str) or not source[path_field]:
            raise ValueError("Missing frozen asset path: " + path_field)
        identity = source.get(hash_field)
        if (not isinstance(identity, str) or len(identity) != 64
                or any(character not in "0123456789abcdef" for character in identity)):
            raise ValueError("Missing frozen asset SHA256: " + hash_field)
    if not isinstance(sprint.get("tiny_patch_config"), str) or not sprint["tiny_patch_config"]:
        raise ValueError("Declare the retained A_F2 Tiny patch configuration")
    return sprint, member


def _checked_path(path, identity, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require a regular frozen asset: " + description)
    path = path.resolve(strict=True)
    if sha256_file(path) != identity:
        raise ValueError("Frozen asset changed: " + description)
    return path


def inference_inventory(config, tiny_inventory, base_path, opacity_path):
    """Count actual unique loaded assets, including encoder and patch configs."""
    files = {}
    for item in tiny_inventory["files"]:
        path = Path(item["path"]).resolve(strict=True)
        files[str(path)] = dict(item, path=str(path))
    for identity, path in (("sprint_base768", base_path), ("opacity512", opacity_path)):
        path = Path(path).resolve(strict=True)
        if str(path) in files:
            raise ValueError("Retained and complementary members require distinct weights")
        files[str(path)] = {"id": identity, "path": str(path),
                            "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    recipe_bytes = len(json.dumps(config, ensure_ascii=False).encode("utf-8"))
    total = sum(item["bytes"] for item in files.values()) + recipe_bytes
    if total > MAX_MODEL_BYTES:
        raise ValueError("Actual loaded weights and necessary configurations exceed 500MB")
    return {"files": list(files.values()), "configuration_bytes": recipe_bytes,
            "total_bytes": total, "limit_bytes": MAX_MODEL_BYTES, "size_pass": True,
            "scope": "unique loaded weights, Tiny patch/configurations and serialized recipe"}


class RobustPredictor:
    """Three fixed encoders with shared decoding and unchanged Sprint arithmetic."""
    def __init__(self, config):
        from inference_runtime import activate_dependencies
        activate_dependencies()
        import torch
        from inference_inputs import ExactU8Input
        from inference_reproducibility import configure_inference
        from sprint.evaluate import _candidate, _tiny
        from ensemble_sprint.model import EnsembleStudyModel

        self.config = deepcopy(config)
        sprint, member = validate_recipe(self.config)
        base_path = _checked_path(sprint["candidate_weights"],
                                  sprint["candidate_weights_sha256"], "Sprint Base768")
        opacity_path = _checked_path(member["weights"], member["weights_sha256"], "opacity512")
        patch_config = Path(sprint["tiny_patch_config"]).resolve(strict=True)
        if sprint.get("tiny_patch_config_sha256") is not None:
            _checked_path(patch_config, sprint["tiny_patch_config_sha256"], "A_F2 configuration")
        if not torch.cuda.is_available():
            raise RuntimeError("The registered FP16 ensemble requires CUDA")
        configure_inference(torch_module=torch)
        torch.backends.cudnn.deterministic = True
        self.torch, self.device = torch, "cuda"
        self.tiny = _tiny(sprint)
        self.base, _, _ = _candidate(sprint, torch)
        payload = torch.load(opacity_path, map_location="cpu", weights_only=True)
        state = checked_member_payload(payload, member, torch)
        self.opacity = EnsembleStudyModel(backbone_name=TINY_BACKBONE, pretrained_path=None,
            branch="opacity", aggregation="mean", gradient_checkpointing=False,
            view_chunk_size=1, channels_last=True)
        self.opacity.load_state_dict(state, strict=True)
        self.opacity.eval().requires_grad_(False).to("cuda", dtype=torch.float16,
                                                   memory_format=torch.channels_last)
        del payload, state
        self.inventory = inference_inventory(self.config, self.tiny.inventory, base_path, opacity_path)
        for item in self.inventory["files"]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise RuntimeError("An inference asset changed during initialization")
        if (sha256_file(base_path) != sprint["candidate_weights_sha256"]
                or sha256_file(opacity_path) != member["weights_sha256"]):
            raise RuntimeError("Frozen encoder changed during initialization")
        self.image_sizes = (768, 512)
        self.input_lookup = ExactU8Input(torch, self.device, MEAN, STD, image_sizes=self.image_sizes)

    def synchronize(self):
        self.torch.cuda.synchronize()

    def predict_image_components(self, path):
        from inference_inputs import load_image_pixels, member_results_to_cpu
        torch = self.torch
        try:
            pixels = load_image_pixels(path, self.image_sizes, _letterbox_image)
            with torch.inference_mode():
                inputs = {}
                for size in self.image_sizes:
                    image = self.input_lookup.to_tensor(pixels[size])
                    inputs[size] = (image, torch.flip(image, dims=(-1,)).contiguous(
                        memory_format=torch.channels_last))
                means, finite = [], []
                for model, size, component_output in (
                        (self.tiny.model, 768, False), (self.base, 768, True),
                        (self.opacity, 512, True)):
                    probabilities = []
                    for image in inputs[size]:
                        logits = (model(image, return_components=True)["image_logits"]
                                  if component_output else model(image))
                        if tuple(logits.shape) != (1, 10):
                            raise ValueError("Require ten single-image logits from every encoder")
                        finite.append(torch.isfinite(logits).all())
                        probabilities.append(torch.sigmoid(logits.float())[0])
                    means.append(torch.stack(probabilities).mean(0))
                values = member_results_to_cpu(torch, means, finite)
            sprint_value = aggregate_member_probabilities(values[:2], MODEL_WEIGHTS)
            components = np.stack((sprint_value, values[2].astype(np.float64, copy=False)))
            fused = aggregate_member_probabilities(components, self.config["fusion_weights"])
            return fused, components
        except BaseException:
            self.input_lookup.sync_after_error()
            raise

    def predict_image(self, path):
        return self.predict_image_components(path)[0]

    def predict_study(self, records):
        if not records or len(study_groups(records)) != 1:
            raise ValueError("One prediction unit must contain exactly one complete Study")
        accumulator, per_image, result = StudyAccumulator(records), [], None
        for row in records:
            value = self.predict_image(row["path"])
            per_image.append((row, value))
            result = accumulator.add(row, value)
        if result is None or not accumulator.complete():
            raise ValueError("Incomplete Study prediction")
        return result, per_image


Predictor = RobustPredictor
