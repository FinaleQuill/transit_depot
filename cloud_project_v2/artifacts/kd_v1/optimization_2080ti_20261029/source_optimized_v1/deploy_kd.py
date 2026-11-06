"""Fixed FP16 Tiny deployment, deterministic TTA and probability ensembles.

Imports perform no model construction, weight download, or image inference.
The deployment package contains no training checkpoint or teacher model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

TINY_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
BASE_BACKBONE = "convnext_base.fb_in22k_ft_in1k_384"
CLASS_IDS = list(range(1, 11))
IMAGE_SIZE = 512
MAX_MODEL_BYTES = 500_000_000
TINY_PARAMETERS_10_CLASSES = 27_827_818
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["Subject_id", "Study_id", "Predict_class", "Probability"]
IMAGE_CSV_HEADER = ["Image_id", "Subject_id", "Study_id", "Predict_class", "Probability"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_preprocessing(config: Mapping[str, Any], *, image_size: int = IMAGE_SIZE,
                           allow_tta: bool = False) -> None:
    allowed_tta = {"identity", "hflip2"} if allow_tta else {"identity"}
    if (type(image_size) is not int or image_size not in {512, 768}
            or config.get("image_size") != image_size
            or config.get("validation_size", image_size) != image_size
            or config.get("resize") != "letterboxLANCZOS"
            or config.get("exif_transpose") is not True
            or config.get("color_mode") != "RGB"
            or list(config.get("mean", ())) != list(MEAN)
            or list(config.get("std", ())) != list(STD)
            or config.get("training_scales", [image_size]) != [image_size]
            or config.get("validation_tta", "identity") not in allowed_tta):
        raise ValueError("Expected fixed EXIF/RGB/LANCZOS-pad/ImageNet preprocessing at the declared size")


def validate_provenance(student: Mapping[str, Any], teachers: Any, *,
                        training_mode: str = "distillation") -> None:
    from kd_common import validate_pretrained_provenance

    validate_pretrained_provenance(student, TINY_BACKBONE)
    if training_mode == "supervised_bce":
        if not isinstance(teachers, list) or teachers:
            raise ValueError("An independent supervised student must declare no teacher ancestry")
        return
    if training_mode != "distillation":
        raise ValueError("Unknown student training mode")
    if not isinstance(teachers, list) or len(teachers) != 5:
        raise ValueError("All five OOF teacher provenance records are required")
    folds: set[int] = set()
    for teacher in teachers:
        if not isinstance(teacher, dict):
            raise ValueError("Malformed teacher provenance")
        fold = teacher.get("fold_index", teacher.get("fold"))
        if type(fold) is not int or fold not in range(5) or fold in folds:
            raise ValueError("Teacher provenance must contain folds 0..4 exactly once")
        folds.add(fold)
        backbone = teacher.get("model_config", {}).get("backbone_name", BASE_BACKBONE)
        if backbone != BASE_BACKBONE:
            raise ValueError("Only the approved V1 Base teachers are permitted")
        validate_pretrained_provenance(teacher.get("pretrained_provenance", {}), BASE_BACKBONE)
        if teacher.get("selection_protocol") != "fixed_epoch_final_ema":
            raise ValueError("Teacher selection must exclude the outer OOF fold")
        if teacher.get("training_complete") is not True:
            raise ValueError("Incomplete teacher cannot be a deployment ancestor")
        if (teacher.get("schema_version") != 6
                or teacher.get("artifact_type") != "kd_teacher"
                or teacher.get("model_name") != "convnext_v1_base_gap"
                or teacher.get("weights_source") != "ema"
                or teacher.get("outer_fold_used_for_selection") is not False
                or teacher.get("official_valid_used_for_training") is not False
                or teacher.get("test_used_for_training") is not False
                or teacher.get("epoch") != teacher.get("training_config", {}).get("epochs")):
            raise ValueError("Teacher ancestry does not prove the strict-OOF final-EMA protocol")
        trained, heldout = teacher.get("training_subject_ids"), teacher.get("oof_subject_ids")
        if (not isinstance(trained, list) or not trained or not isinstance(heldout, list)
                or not heldout or set(trained) & set(heldout)):
            raise ValueError("Teacher ancestry has missing or overlapping patient sets")
        digest = teacher.get("checkpoint_sha256", teacher.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Teacher checkpoint SHA-256 is required")


def validate_supervised_protocol(protocol: Mapping[str, Any], *, image_size: int,
                                 validation_tta: str,
                                 training: Mapping[str, Any] | None = None) -> None:
    """Teacher-free ancestry is accepted only for the explicit independent BCE mode."""
    if (not isinstance(protocol, Mapping) or protocol.get("enabled") is not False
            or type(protocol.get("alpha")) not in {int, float} or protocol["alpha"] != 0
            or protocol.get("loss") != "weighted_bce"
            or protocol.get("class_reduction") != "mean"
            or protocol.get("study_equal_weighting") is not True
            or protocol.get("student_training_size") != image_size
            or validation_tta not in {"identity", "hflip2"}
            or protocol.get("validation_tta") != validation_tta
            or protocol.get("deployment_tta") != validation_tta
            or protocol.get("deployment_ensemble") is not False):
        raise ValueError("Independent supervised student requires an explicit zero-KD weighted-BCE protocol")
    forbidden = {"soft_labels_sha256", "soft_metadata_sha256", "teacher_tta", "teacher_views",
                 "soft_target_aggregation"}
    if forbidden & set(protocol):
        raise ValueError("Supervised BCE cannot declare teacher-cache or teacher-view dependencies")
    if training is not None and (
            not isinstance(training, Mapping)
            or training.get("training_mode") != "supervised_bce"
            or type(training.get("alpha")) not in {int, float} or training["alpha"] != 0
            or "soft_labels" not in training or training["soft_labels"] is not None
            or training.get("image_size") != image_size
            or training.get("validation_tta") != validation_tta):
        raise ValueError("Supervised checkpoint training configuration contradicts its independent BCE mode")


def validate_student_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    schema = checkpoint.get("schema_version")
    size = checkpoint.get("image_size")
    mode = checkpoint.get("training_mode", "distillation")
    if (schema not in {7, 8, 9}
            or checkpoint.get("artifact_type") != "kd_student"
            or checkpoint.get("model_name") != "convnext_v1_tiny_gap"
            or checkpoint.get("model_config", {}).get("backbone_name") != TINY_BACKBONE
            or checkpoint.get("class_ids") != CLASS_IDS
            or type(size) is not int or size not in ({512} if schema == 7 else {512, 768})
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True):
        raise ValueError("Export requires a validated approved Tiny EMA student checkpoint")
    if ((schema == 9 and mode != "supervised_bce")
            or (schema in {7, 8} and mode != "distillation")):
        raise ValueError("Checkpoint schema and student training mode disagree")
    validate_preprocessing(checkpoint.get("preprocessing", {}), image_size=size, allow_tta=schema in {8, 9})
    validate_provenance(checkpoint.get("pretrained_provenance", {}), checkpoint.get("teacher_provenance"),
                        training_mode=mode)
    for teacher in checkpoint["teacher_provenance"]:
        if (teacher.get("source_manifest_sha256") != checkpoint.get("source_manifest_sha256")
                or teacher.get("fold_plan_sha256") != checkpoint.get("fold_plan_sha256")):
            raise ValueError("Student and teacher ancestry disagree on original train/fold plan")
    distillation = checkpoint.get("distillation", {})
    if schema == 9:
        tta = checkpoint.get("validation_tta")
        if (checkpoint["preprocessing"].get("validation_tta") != tta
                or checkpoint.get("deployment_tta") != tta
                or checkpoint.get("deployment_ensemble") is not False):
            raise ValueError("Supervised checkpoint validation and deployment declarations disagree")
        for field in ("source_manifest_sha256", "fold_plan_sha256", "official_valid_manifest_sha256"):
            if not _sha(checkpoint.get(field)):
                raise ValueError("Supervised checkpoint requires original train/fold/valid SHA256 bindings")
        validate_supervised_protocol(distillation, image_size=size, validation_tta=tta,
                                     training=checkpoint.get("training_config", {}))
        return
    if (checkpoint.get("training_config", {}).get("training_mode", "distillation") != "distillation"
            or distillation.get("enabled", True) is not True):
        raise ValueError("KD checkpoints cannot disguise an independent supervised training configuration")
    if schema == 7:
        if (distillation.get("deployment_tta", "identity") != "identity"
                or distillation.get("deployment_ensemble", False) is not False):
            raise ValueError("Legacy schema7 describes an identity single student")
    else:
        tta = checkpoint.get("validation_tta")
        training = checkpoint.get("training_config", {})
        teacher_views = distillation.get("teacher_views")
        expected_views = {"identity": ["identity"], "identity_hflip": ["identity", "hflip"]}
        if (tta not in {"identity", "hflip2"}
                or checkpoint["preprocessing"].get("validation_tta") != tta
                or training.get("image_size") != size or training.get("validation_tta") != tta
                or distillation.get("student_training_size") != size
                or distillation.get("validation_tta") != tta
                or distillation.get("deployment_tta") != tta
                or distillation.get("deployment_ensemble") is not False
                or teacher_views != expected_views.get(distillation.get("teacher_tta"))
                or distillation.get("soft_target_aggregation") !=
                    "mean_sigmoid_after_pos_weight_alignment_and_temperature"):
            raise ValueError("Schema8 resolution, validation TTA or view-logit KD protocol is inconsistent")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def deployment_members(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Represent legacy packages as one member without changing their declaration."""
    if config.get("schema_version") == 2:
        return [dict(member) for member in config["members"]]
    return [{**dict(config), "weights_file": "student_fp16.pt", "seed": config.get("seed")}]


def normalized_model_weights(values: Any, count: int) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("Provide one fixed weight or ten class weights per member")
    if any(isinstance(v, bool) or isinstance(v, (list, tuple)) and any(isinstance(x, bool) for x in v) for v in values):
        raise ValueError("Boolean ensemble weights are invalid")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed ensemble weights") from exc
    if array.shape == (count,):
        array = np.repeat(array[:, None], 10, axis=1)
    if (array.shape != (count, 10) or not np.isfinite(array).all() or (array < 0).any()
            or (array.max(axis=0) <= 0).any() or (array.max(axis=1) <= 0).any()):
        raise ValueError("Weights must be finite, nonnegative, with every member/class represented")
    # Scale before summing to avoid overflow for otherwise valid large fixed weights.
    scaled = array / array.max(axis=0, keepdims=True)
    return scaled / scaled.sum(axis=0, keepdims=True)


def aggregate_member_probabilities(values: Sequence[Any], weights: Any) -> np.ndarray:
    array = np.asarray([checked_probabilities(value) for value in values], dtype=np.float64)
    fixed = normalized_model_weights(weights, len(values))
    return checked_probabilities((array * fixed).sum(axis=0))


def deployment_inventory(model_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    names = config.get("inference_files")
    expected = ([m["weights_file"] for m in deployment_members(config)] + ["config.json"]
                if config.get("schema_version") == 2 else ["student_fp16.pt", "config.json"])
    if names != expected or len(set(expected)) != len(expected):
        raise ValueError("Inference inventory must contain every loaded unique weight and config.json")
    files = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Inference files must be plain filenames inside model-dir")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("Deployment files must be regular files inside model-dir")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    actual_hashes = {item["path"]: item["sha256"] for item in files}
    if config.get("schema_version") in {1, 2}:
        for member in deployment_members(config):
            if actual_hashes[member["weights_file"]] != member.get("weights_sha256"):
                raise ValueError("Inference inventory weight hash differs from its frozen config")
    total = sum(item["bytes"] for item in files)
    return {"files": files, "total_bytes": total, "limit_bytes": MAX_MODEL_BYTES,
            "size_pass": total <= MAX_MODEL_BYTES}


def validate_deployment_config(config: Mapping[str, Any]) -> None:
    schema = config.get("schema_version")
    if (schema not in {1, 2} or config.get("artifact_type") != "kd_student_deployment"
            or config.get("backbone_name") != TINY_BACKBONE or config.get("precision") != "float16"
            or config.get("class_ids") != CLASS_IDS or config.get("batch_size") != 1
            or config.get("aggregation") != "study_mean_probability"):
        raise ValueError("Expected fixed Tiny FP16 batch1 probability deployment")
    if schema == 1:
        if (config.get("image_size") != 512 or config.get("tta") != "identity"
                or config.get("ensemble") is not False
                or "scales" in config and config["scales"] != [512]):
            raise ValueError("Legacy deployments must retain single-Tiny512 identity semantics")
        if config.get("training_mode", "distillation") != "distillation":
            raise ValueError("Legacy deployment schema1 cannot declare a new supervised training mode")
        members = deployment_members(config)
    else:
        members = config.get("members")
        if not isinstance(members, list) or not 1 <= len(members) <= 4:
            raise ValueError("Deployment supports one to four distinct student members")
        if (config.get("member_aggregation") != "fixed_weighted_mean_probability"
                or config.get("tta_aggregation") != "mean_sigmoid_probability"):
            raise ValueError("Only fixed probability averages are permitted")
        if config.get("ensemble") is not (len(members) > 1):
            raise ValueError("Ensemble declaration disagrees with member count")
        if any(not isinstance(member, dict) or type(member.get("image_size")) is not int or member["image_size"] not in {512, 768} or member.get("tta") not in {"identity", "hflip2"} for member in members):
            raise ValueError("Each member must declare fixed Tiny512/768 and identity/hflip2")
        if config.get("training_mode", members[0].get("training_mode", "distillation")) != members[0].get("training_mode", "distillation"):
            raise ValueError("Representative training mode differs from the first deployment member")
        weights = normalized_model_weights(config.get("model_weights"), len(members))
        if np.asarray(config["model_weights"]).shape != weights.shape or not np.allclose(np.asarray(config["model_weights"]), weights, rtol=0, atol=1e-12):
            raise ValueError("Stored model_weights must be normalized Mx10 fixed weights")
        sizes = sorted({member.get("image_size") for member in members})
        tta_values = {member.get("tta") for member in members}
        if (config.get("image_sizes") != sizes
                or config.get("image_size") != (sizes[0] if len(sizes) == 1 else None)
                or config.get("tta") != (next(iter(tta_values)) if len(tta_values) == 1 else "mixed")
                or config.get("inference_files") != [m.get("weights_file") for m in members] + ["config.json"]):
            raise ValueError("Deployment summary/inventory disagrees with its members")
        for key in ("source_manifest_sha256", "fold_plan_sha256", "official_valid_manifest_sha256"):
            if not _sha(config.get(key)):
                raise ValueError("Shared original train/fold/valid SHA256 declarations are required")
        if (len({m.get("weights_file") for m in members}) != len(members)
                or len({m.get("weights_sha256") for m in members}) != len(members)
                or len({m.get("source_checkpoint_sha256") for m in members}) != len(members)):
            raise ValueError("Repeated files, weights or checkpoints do not constitute distinct student members")
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("Malformed deployment member")
        size = member.get("image_size")
        tta = member.get("tta")
        if (type(size) is not int or size not in ({512} if schema == 1 else {512, 768})
                or tta not in ({"identity"} if schema == 1 else {"identity", "hflip2"})):
            raise ValueError("Each member needs a fixed 512/768 size and identity/hflip2 TTA")
        model_config = member.get("model_config", {})
        if model_config.get("backbone_name") != TINY_BACKBONE or model_config.get("gradient_checkpointing") is not False:
            raise ValueError("Invalid deployment architecture metadata")
        validate_preprocessing(member.get("preprocessing", {}), image_size=size, allow_tta=schema == 2)
        mode = member.get("training_mode", "distillation")
        validate_provenance(member.get("pretrained_provenance", {}), member.get("teacher_provenance"),
                            training_mode=mode)
        if mode == "supervised_bce":
            if member.get("source_checkpoint_schema_version", 9) != 9:
                raise ValueError("Independent BCE member must originate from a schema9 supervised checkpoint")
            validate_supervised_protocol(member.get("distillation", {}), image_size=size,
                validation_tta=member.get("preprocessing", {}).get("validation_tta"))
        elif member.get("distillation", {}).get("enabled", True) is not True:
            raise ValueError("A KD member cannot disable teacher dependencies without a supervised mode declaration")
        elif member.get("source_checkpoint_schema_version", 8) not in {7, 8}:
            raise ValueError("KD member source schema contradicts its training mode")
        if schema == 2:
            if (not _sha(member.get("weights_sha256")) or not _sha(member.get("source_checkpoint_sha256"))
                    or member.get("weights_source") != "ema"
                    or not isinstance(member.get("weights_file"), str)
                    or Path(member["weights_file"]).name != member["weights_file"]
                    or member["weights_file"] in {"", ".", "..", "config.json"}):
                raise ValueError("Member needs safe FP16 file, EMA source and valid source/weight hashes")
            for field in ("source_manifest_sha256", "fold_plan_sha256", "official_valid_manifest_sha256"):
                if field in member and member[field] != config[field]:
                    raise ValueError("Member source data bindings differ from the shared deployment declaration")
            for teacher in member["teacher_provenance"]:
                if (teacher.get("source_manifest_sha256") != config["source_manifest_sha256"]
                        or teacher.get("fold_plan_sha256") != config["fold_plan_sha256"]):
                    raise ValueError("Members and teachers must share original train/fold declarations")
    gates = config.get("gates")
    if gates is not None and (len(gates) != 10 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in gates)):
        raise ValueError("Fixed Gate values must be ten finite probabilities")


def _letterbox_image(image: Image.Image, image_size: int, mode: str) -> Image.Image:
    scale = image_size / max(image.size)
    size = tuple(max(1, min(image_size, round(length * scale))) for length in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new(mode, (image_size, image_size), 0)
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return canvas


def letterbox_rgb(image: Image.Image, image_size: int = IMAGE_SIZE) -> Image.Image:
    """The same geometry and rounding as dataset.letterbox_rgb; no training imports."""
    return _letterbox_image(image, image_size, "RGB")


def image_array(image: Image.Image, image_size: int) -> np.ndarray:
    # Keep grayscale pixels single-channel through resize and padding. Normalizing
    # directly into contiguous CHW planes avoids RGB float32 intermediates and
    # strided three-channel broadcasting, while retaining each float32 operation.
    grayscale = image.mode == "L"
    pixels = np.asarray(_letterbox_image(image, image_size, "L" if grayscale else "RGB"))
    result = np.empty((3, image_size, image_size), dtype=np.float32)
    offset = np.asarray(MEAN, dtype=np.float32) * np.float32(255.0)
    denominator = np.asarray(STD, dtype=np.float32) * np.float32(255.0)
    for channel in range(3):
        np.copyto(result[channel], pixels if grayscale else pixels[:, :, channel])
        np.subtract(result[channel], offset[channel], out=result[channel])
        np.divide(result[channel], denominator[channel], out=result[channel])
    return result


def load_image_arrays(path: str | Path, image_sizes: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode/EXIF exactly once per source image; deterministic per-size preprocessing."""
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode in {"I", "F"} or source.mode.startswith("I;"):
            raise ValueError("Non-8-bit image requires an explicit intensity policy")
        # LANCZOS treats the identical RGB channels of an L image identically.
        # Keep it single-channel until image_array writes the normalized CHW planes.
        image = source if source.mode == "L" else source.convert("RGB")
        return {size: image_array(image, size) for size in sorted(set(image_sizes))}


def load_image_array(path: str | Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    return load_image_arrays(path, [image_size])[image_size]


class Predictor:
    """Shared fixed forward-only deployment: source batch1, serial members/views."""

    def __init__(self, model_dir: str | Path, *, device: str = "cuda") -> None:
        import torch
        from model_overnight import build_overnight_model

        self.torch = torch
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.config = read_json(self.model_dir / "config.json")
        validate_deployment_config(self.config)
        self.inventory = deployment_inventory(self.model_dir, self.config)
        if not self.inventory["size_pass"]:
            raise ValueError("Actual inference files exceed 500,000,000 bytes")
        self.members = deployment_members(self.config)
        self.model_weights = (self.config["model_weights"] if self.config.get("schema_version") == 2
                              else [[1.0] * 10])
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This FP16 deployment requires CUDA; no CPU fallback")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.models = []
        for member in self.members:
            path = self.model_dir / member["weights_file"]
            if sha256_file(path) != member.get("weights_sha256"):
                raise ValueError("Deployment weight SHA-256 mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ValueError("Expected a nonempty FP16 state dictionary")
            for value in state.values():
                if not isinstance(value, torch.Tensor) or (value.is_floating_point() and value.dtype != torch.float16):
                    raise ValueError("Disk inference weights must be FP16 tensors")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite deployment weight")
            model = build_overnight_model(pretrained=False, **member["model_config"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False).to(self.device, dtype=torch.float16,
                                                  memory_format=torch.channels_last)
            self.models.append(model)
            del state
        self.model = self.models[0]  # Existing diagnostic integrations retain their one-model handle.

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def predict_image(self, path: str | Path) -> np.ndarray:
        torch = self.torch
        arrays = load_image_arrays(path, [m["image_size"] for m in self.members])
        results = []
        finite_checks = []
        with torch.inference_mode():
            inputs = {size: torch.from_numpy(array).unsqueeze(0).to(
                self.device, dtype=torch.float16, memory_format=torch.channels_last)
                for size, array in arrays.items()}
            transformed_inputs = {(size, "identity"): image for size, image in inputs.items()}
            for member, model in zip(self.members, self.models):
                size = member["image_size"]
                views = ("identity", "hflip") if member["tta"] == "hflip2" else ("identity",)
                probabilities = []
                for view in views:
                    key = (size, view)
                    if key not in transformed_inputs:
                        transformed_inputs[key] = torch.flip(inputs[size], dims=(-1,)).contiguous(
                            memory_format=torch.channels_last)
                    transformed = transformed_inputs[key]
                    logits = model(transformed)
                    if tuple(logits.shape) != (1, 10):
                        raise FloatingPointError("Invalid single-image student logits")
                    finite_checks.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                results.append(torch.stack(probabilities).mean(0))
            # Check the original logits, since sigmoid can hide infinities. Delay
            # the host synchronization until all views finish, then transfer all
            # member means together without changing their probability arithmetic.
            if not bool(torch.stack(finite_checks).all()):
                raise FloatingPointError("Invalid single-image student logits")
            results = torch.stack(results).cpu().numpy()
        return aggregate_member_probabilities(results, self.model_weights)


def checked_probabilities(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (10,) or not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Expected ten finite independent probabilities")
    return result


def read_manifest(path: str | Path, *, split: str = "test") -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty inference manifest")
    ids: set[str] = set()
    for row in rows:
        required = ("image_id", "subject_id", "study_id", "study_key", "relative_path")
        if any(not row.get(key) for key in required) or row.get("split") != split:
            raise ValueError("Manifest identities or original split are invalid")
        if row["image_id"] in ids:
            raise ValueError("Duplicate manifest image ID")
        ids.add(row["image_id"])
        if row["study_key"] != f"{row['subject_id']}/{row['study_id']}":
            raise ValueError("Study identity mismatch")
        relative = Path(row["relative_path"])
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4
                or relative.parts[:3] != (split, row["subject_id"], row["study_id"])):
            raise ValueError("Manifest image path escapes its Study")
        if split == "test" and row.get("targets", "").strip() not in ("", "null", "None"):
            raise ValueError("Test targets must remain unknown")
    return rows


def resolve_image_root(image_root: str | Path, split: str = "test") -> Path:
    root = Path(image_root).resolve(strict=True)
    for candidate in (root, root / "cxr_first"):
        if (candidate / split).is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing {split} images under {root}; extraction may be incomplete")


def prediction_records(image_root: str | Path, manifest: str | Path | None, *,
                       split: str = "test") -> list[dict[str, Any]]:
    root = resolve_image_root(image_root, split)
    if manifest is not None:
        records: list[dict[str, Any]] = read_manifest(manifest, split=split)
        for row in records:
            path = (root / row["relative_path"]).resolve(strict=True)
            if not path.is_relative_to(root / split):
                raise ValueError("Image path leaves its original split root")
            row["path"] = path
        return records
    return discover_images(root / split)


def discover_images(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory).resolve(strict=True)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Timing image symlink escapes the supplied directory")
        subject, study = path.parent.parent.name, path.parent.name
        if not subject.startswith("subject_") or not study.startswith("study_"):
            raise ValueError("Expected subject_*/study_*/image layout; provide the original directory structure")
        records.append({"path": resolved, "image_id": f"{subject}/{study}/{path.name}",
                        "subject_id": subject, "study_id": study, "study_key": f"{subject}/{study}",
                        "filename": path.name})
    if not records:
        raise ValueError("No JPG/PNG images found")
    identities = [(r["subject_id"], r["study_id"], r["path"].name) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate supplied image identity")
    return records


class StudyAccumulator:
    """Fixed arithmetic aggregation, without learning cross-image statistics."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.expected = Counter(row["study_key"] for row in records)
        self.subject_studies = {row["study_key"]: (row["subject_id"], row["study_id"]) for row in records}
        self.sums: dict[str, np.ndarray] = {}
        self.counts: Counter = Counter()
        self.seen: set[str] = set()

    def add(self, record: Mapping[str, Any], probabilities: Any) -> tuple[str, str, np.ndarray] | None:
        key, image_id = record["study_key"], record["image_id"]
        if key not in self.expected or image_id in self.seen:
            raise ValueError("Unexpected or duplicate image during Study aggregation")
        self.seen.add(image_id)
        self.sums.setdefault(key, np.zeros(10, dtype=np.float64))
        self.sums[key] += checked_probabilities(probabilities)
        self.counts[key] += 1
        if self.counts[key] > self.expected[key]:
            raise ValueError("Too many images for Study")
        if self.counts[key] == self.expected[key]:
            subject, study = self.subject_studies[key]
            return subject, study, self.sums[key] / self.counts[key]
        return None

    def complete(self) -> bool:
        return self.counts == self.expected


def write_prediction(writer: Any, subject: str, study: str, probabilities: Any,
                     gates: Sequence[float] | None = None) -> None:
    probabilities = checked_probabilities(probabilities)
    for c, probability in zip(CLASS_IDS, probabilities):
        row: list[Any] = [subject, study, c, repr(float(probability))]
        if gates is not None:
            row.append(repr(float(gates[c - 1])))
        writer.writerow(row)


def write_image_prediction(writer: Any, record: Mapping[str, Any], probabilities: Any) -> None:
    """Write every image's final ten outputs before optional Study aggregation."""
    values = checked_probabilities(probabilities)
    for class_id, probability in zip(CLASS_IDS, values):
        writer.writerow([record["image_id"], record["subject_id"], record["study_id"],
                         class_id, repr(float(probability))])


def read_submission(path: str | Path) -> dict[tuple[str | None, str], np.ndarray]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Study_id", "Predict_class", "Probability"}.issubset(reader.fieldnames or []):
            raise ValueError("Invalid submission header")
        data: dict[tuple[str | None, str], dict[int, float]] = {}
        for row in reader:
            key = (row.get("Subject_id"), row["Study_id"])
            c = int(row["Predict_class"])
            if c not in CLASS_IDS or c in data.setdefault(key, {}):
                raise ValueError("Submission contains a duplicate or invalid class")
            probability = float(row["Probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Submission contains an invalid probability")
            data[key][c] = probability
    result = {}
    for key, values in data.items():
        if set(values) != set(CLASS_IDS):
            raise ValueError("Each submitted Study must have all ten classes")
        result[key] = np.asarray([values[c] for c in CLASS_IDS], dtype=np.float64)
    if not result:
        raise ValueError("Empty submission")
    return result


def match_timing_manifest(records: Sequence[Mapping[str, Any]], manifest: str | Path | None) -> dict[str, Any]:
    """Require evidence of complete Studies before claiming CSV reproducibility."""
    if manifest is None:
        return {"complete_studies_proven": False, "reason": "Full original test manifest not provided"}
    rows = read_manifest(manifest)
    full: dict[str, set[str]] = {}
    for row in rows:
        full.setdefault(row["study_key"], set()).add(Path(row["relative_path"]).name)
    selected: dict[str, set[str]] = {}
    for row in records:
        selected.setdefault(row["study_key"], set()).add(Path(row["path"]).name)
    for key, filenames in selected.items():
        if key not in full or not filenames.issubset(full[key]):
            raise ValueError("Timing images do not match the original test manifest")
    incomplete = [key for key, names in selected.items() if names != full[key]]
    return {"complete_studies_proven": not incomplete, "incomplete_studies": incomplete,
            "reason": "Timing directory omits views of one or more Studies" if incomplete else None,
            "manifest_sha256": sha256_file(manifest)}
