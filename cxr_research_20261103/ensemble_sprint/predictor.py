"""Shared fixed Base768 + two Tiny512 inference for valid, test and timing.

Only model initialization opens weight files. Images are decoded inside each
prediction call, with no cached predictions or cross-Study adaptation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deploy_kd import (MAX_MODEL_BYTES, MEAN, STD, StudyAccumulator,
                       _letterbox_image, aggregate_member_probabilities,
                       sha256_file)

MEMBER_IDS = ("base768", "context512", "opacity512")
TINY_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
FUSION_WEIGHTS = [[.5] * 10, [.25] * 10, [.25] * 10]
for _row, _class1, _class8 in zip(FUSION_WEIGHTS, (1., 0., 0.), (.4, .2, .4)):
    _row[0], _row[7] = _class1, _class8


def validate_recipe(config):
    weights = np.asarray(config.get("fusion_weights"), dtype=np.float64)
    if weights.shape != (3, 10) or not np.array_equal(weights, FUSION_WEIGHTS):
        raise ValueError("Require the single registered Base/context/opacity fusion")
    members = config.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("Exactly two trained Tiny members are required")
    for member, identity, branch, epochs in zip(
            members, MEMBER_IDS[1:], ("global", "opacity"), (4, 6)):
        if (member.get("id") != identity or member.get("branch") != branch
                or member.get("image_size") != 512 or member.get("epochs") != epochs
                or member.get("backbone_name", TINY_BACKBONE) != TINY_BACKBONE):
            raise ValueError("Member identity differs from the registered final ensemble")
    if config.get("tta", "hflip2") != "hflip2" or config.get("patient_smoothing", False):
        raise ValueError("Require frozen hflip2 without patient smoothing")
    return members


def member_weights_path(config, member):
    return Path(member.get("weights", Path(config["output"]) / member["id"] / "final_fp16.pt")).resolve(strict=True)


def checked_member_payload(payload, member, torch):
    if not isinstance(payload, dict) or not {"state_dict", "configuration", "epoch"} <= payload.keys():
        raise ValueError("Require a final EMA export with configuration and epoch")
    saved = payload["configuration"]
    if not isinstance(saved, dict):
        raise ValueError("Missing member training configuration")
    for field in ("id", "branch", "image_size", "epochs"):
        if saved.get(field) != member.get(field):
            raise ValueError("Export configuration differs from frozen member: " + field)
    if (payload["epoch"] != member["epochs"]
            or saved.get("backbone_name", TINY_BACKBONE) != TINY_BACKBONE):
        raise ValueError("Require exactly the registered final epoch and Tiny backbone")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Empty member state")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.is_complex()
                or value.is_floating_point() and
                (value.dtype != torch.float16 or not bool(torch.isfinite(value).all()))):
            raise ValueError("Require finite FP16 export tensors: " + str(key))
    return state


def study_groups(records):
    groups = {}
    for row in records:
        groups.setdefault(row["study_key"], []).append(row)
    return list(groups.values())


class EnsemblePredictor:
    """All three models use exactly the same image math in every entry point."""
    def __init__(self, config):
        from inference_runtime import activate_dependencies
        activate_dependencies()
        import torch
        from inference_inputs import ExactU8Input
        from inference_reproducibility import configure_inference
        from sprint.evaluate import _candidate
        from ensemble_sprint.model import EnsembleStudyModel

        self.config = dict(config)
        members = validate_recipe(config)
        if not torch.cuda.is_available():
            raise RuntimeError("The registered FP16 ensemble requires CUDA")
        configure_inference(torch_module=torch)
        torch.backends.cudnn.deterministic = True
        self.torch, self.device = torch, "cuda"
        base_path = Path(config["base_weights"]).resolve(strict=True)
        if sha256_file(base_path) != config["base_weights_sha256"]:
            raise ValueError("Frozen Sprint Base changed")
        base, _, base_sha = _candidate({"candidate_weights": str(base_path),
            "candidate_weights_sha256": config["base_weights_sha256"]}, torch)
        self.models = [base]
        self.image_sizes = [768, 512, 512]
        files = [{"id": MEMBER_IDS[0], "path": str(base_path),
                  "bytes": base_path.stat().st_size, "sha256": base_sha}]
        for member in members:
            path = member_weights_path(config, member)
            identity = sha256_file(path)
            if member.get("weights_sha256", identity) != identity:
                raise ValueError("Final member weights changed: " + member["id"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = checked_member_payload(payload, member, torch)
            model = EnsembleStudyModel(backbone_name=TINY_BACKBONE, pretrained_path=None,
                branch=member["branch"], aggregation="mean", gradient_checkpointing=False,
                view_chunk_size=1, channels_last=True)
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False).to("cuda", dtype=torch.float16,
                                                memory_format=torch.channels_last)
            self.models.append(model)
            files.append({"id": member["id"], "path": str(path),
                          "bytes": path.stat().st_size, "sha256": identity})
            del payload, state
        if len({item["path"] for item in files}) != 3:
            raise ValueError("The registered ensemble requires three distinct weight files")
        # The compact recipe is also a necessary inference asset. Deployment
        # packaging separately measures every actual copied config/asset.
        config_bytes = len(json.dumps(config, ensure_ascii=False).encode("utf-8"))
        total = sum(item["bytes"] for item in files) + config_bytes
        if total > MAX_MODEL_BYTES:
            raise ValueError("Registered inference assets exceed 500MB")
        self.inventory = {"files": files, "configuration_bytes": config_bytes,
                          "total_bytes": total, "size_pass": True,
                          "scope": "three loaded weights plus serialized recipe; final package measured separately"}
        self.input_lookup = ExactU8Input(torch, self.device, MEAN, STD, image_sizes=(512, 768))

    def synchronize(self):
        self.torch.cuda.synchronize()

    def predict_image_components(self, path):
        from inference_inputs import load_image_pixels, member_results_to_cpu
        torch = self.torch
        try:
            pixels = load_image_pixels(path, self.image_sizes, _letterbox_image)
            with torch.inference_mode():
                inputs = {}
                for size in sorted(set(self.image_sizes)):
                    image = self.input_lookup.to_tensor(pixels[size])
                    inputs[size] = (image, torch.flip(image, dims=(-1,)).contiguous(
                        memory_format=torch.channels_last))
                means, finite = [], []
                for model, size in zip(self.models, self.image_sizes):
                    probabilities = []
                    for image in inputs[size]:
                        logits = model(image, return_components=True)["image_logits"]
                        if tuple(logits.shape) != (1, 10):
                            raise ValueError("Require ten single-image logits from each member")
                        finite.append(torch.isfinite(logits).all())
                        probabilities.append(torch.sigmoid(logits.float())[0])
                    means.append(torch.stack(probabilities).mean(0))
                member_values = member_results_to_cpu(torch, means, finite)
            fused = aggregate_member_probabilities(member_values, self.config["fusion_weights"])
            return fused, member_values
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


# Compact public alias for callers that use the generic deployment name.
Predictor = EnsemblePredictor
