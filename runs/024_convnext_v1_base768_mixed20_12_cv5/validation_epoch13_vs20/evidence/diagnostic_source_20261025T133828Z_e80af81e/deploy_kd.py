"""Fixed single-Tiny FP16 deployment shared by predict.py and cost.py.

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


def validate_preprocessing(config: Mapping[str, Any]) -> None:
    if (config.get("image_size") != IMAGE_SIZE
            or config.get("validation_size", IMAGE_SIZE) != IMAGE_SIZE
            or config.get("resize") != "letterboxLANCZOS"
            or config.get("exif_transpose") is not True
            or config.get("color_mode") != "RGB"
            or list(config.get("mean", ())) != list(MEAN)
            or list(config.get("std", ())) != list(STD)):
        raise ValueError("Deployment preprocessing must be fixed EXIF/RGB/LANCZOS-pad/512/ImageNet")
    if config.get("training_scales", [512]) != [512]:
        raise ValueError("Student metadata must describe fixed 512 training, not legacy multi-scale")
    if config.get("validation_tta", "identity") != "identity":
        raise ValueError("Student deployment does not permit TTA")


def validate_provenance(student: Mapping[str, Any], teachers: Any) -> None:
    from kd_common import validate_pretrained_provenance

    validate_pretrained_provenance(student, TINY_BACKBONE)
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


def validate_student_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if (checkpoint.get("schema_version") != 7
            or checkpoint.get("artifact_type") != "kd_student"
            or checkpoint.get("model_name") != "convnext_v1_tiny_gap"
            or checkpoint.get("model_config", {}).get("backbone_name") != TINY_BACKBONE
            or checkpoint.get("class_ids") != CLASS_IDS
            or checkpoint.get("image_size") != IMAGE_SIZE
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True):
        raise ValueError("Export requires a validated approved single-Tiny EMA student checkpoint")
    validate_preprocessing(checkpoint.get("preprocessing", {}))
    validate_provenance(checkpoint.get("pretrained_provenance", {}), checkpoint.get("teacher_provenance"))
    for teacher in checkpoint["teacher_provenance"]:
        if (teacher.get("source_manifest_sha256") != checkpoint.get("source_manifest_sha256")
                or teacher.get("fold_plan_sha256") != checkpoint.get("fold_plan_sha256")):
            raise ValueError("Student and teacher ancestry disagree on original train/fold plan")
    distillation = checkpoint.get("distillation", {})
    if (distillation.get("deployment_tta", "identity") != "identity"
            or distillation.get("deployment_ensemble", False) is not False):
        raise ValueError("The approved deployment has no TTA or ensemble")


def deployment_inventory(model_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    names = config.get("inference_files")
    if names != ["student_fp16.pt", "config.json"]:
        raise ValueError("Only the fixed FP16 weights and configuration may be loaded")
    files = []
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("Deployment files must be regular files inside model-dir")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    total = sum(item["bytes"] for item in files)
    return {"files": files, "total_bytes": total, "limit_bytes": MAX_MODEL_BYTES,
            "size_pass": total <= MAX_MODEL_BYTES}


def validate_deployment_config(config: Mapping[str, Any]) -> None:
    if (config.get("schema_version") != 1
            or config.get("artifact_type") != "kd_student_deployment"
            or config.get("backbone_name") != TINY_BACKBONE
            or config.get("precision") != "float16"
            or config.get("image_size") != 512
            or config.get("class_ids") != CLASS_IDS
            or config.get("tta") != "identity"
            or config.get("ensemble") is not False
            or config.get("batch_size") != 1
            or config.get("aggregation") != "study_mean_probability"):
        raise ValueError("Expected single-Tiny/512/FP16/identity/batch1 deployment")
    model_config = config.get("model_config", {})
    if model_config.get("backbone_name") != TINY_BACKBONE or model_config.get("gradient_checkpointing") is not False:
        raise ValueError("Invalid deployment architecture metadata")
    if "scales" in config and config["scales"] != [512]:
        raise ValueError("Additional deployment scales are forbidden")
    validate_preprocessing(config.get("preprocessing", {}))
    validate_provenance(config.get("pretrained_provenance", {}), config.get("teacher_provenance"))
    gates = config.get("gates")
    if gates is not None and (len(gates) != 10 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in gates)):
        raise ValueError("Fixed Gate values must be ten finite probabilities")


def letterbox_rgb(image: Image.Image, image_size: int = IMAGE_SIZE) -> Image.Image:
    """The same geometry and rounding as dataset.letterbox_rgb; no training imports."""
    scale = image_size / max(image.size)
    size = tuple(max(1, min(image_size, round(length * scale))) for length in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return canvas


def load_image_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as source:
        source.load()
        upright = ImageOps.exif_transpose(source)
        if upright.mode in {"I", "F"} or upright.mode.startswith("I;"):
            raise ValueError("Non-8-bit image requires an explicit intensity policy")
        pixels = np.asarray(letterbox_rgb(upright.convert("RGB")), dtype=np.float32)
    # Albumentations Normalize computes multiplication and offset with float32.
    denominator = np.asarray(STD, dtype=np.float32) * np.float32(255.0)
    pixels = (pixels - np.asarray(MEAN, dtype=np.float32) * np.float32(255.0)) / denominator
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


class Predictor:
    """Load once outside timing; each call reads and predicts exactly one image."""

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
        if sha256_file(self.model_dir / "student_fp16.pt") != self.config.get("weights_sha256"):
            raise ValueError("Deployment weight SHA-256 mismatch")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This FP16 deployment requires CUDA; no CPU fallback")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state = torch.load(self.model_dir / "student_fp16.pt", map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("Expected a nonempty FP16 state dictionary")
        for value in state.values():
            if not isinstance(value, torch.Tensor) or (value.is_floating_point() and value.dtype != torch.float16):
                raise ValueError("Disk inference weights must be FP16 tensors")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite deployment weight")
        self.model = build_overnight_model(pretrained=False, **self.config["model_config"])
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device, dtype=torch.float16,
                                                 memory_format=torch.channels_last)
        del state

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def predict_image(self, path: str | Path) -> np.ndarray:
        torch = self.torch
        array = load_image_array(path)
        with torch.inference_mode():
            image = torch.from_numpy(array).unsqueeze(0).to(
                self.device, dtype=torch.float16, memory_format=torch.channels_last)
            logits = self.model(image)
            if tuple(logits.shape) != (1, 10) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Invalid single-image student logits")
            probabilities = torch.sigmoid(logits.float())[0].cpu().numpy()
        return checked_probabilities(probabilities)


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
