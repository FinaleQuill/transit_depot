"""Fixed N-member experimental student ensemble with class-wise probability weights.

The members are audited ``experimental_student_fp16`` exports.  Each member may
use identity or hflip TTA; the top-level decoder candidate is either the original
full decode or the separately registered JPEG draft-768 decoder.  No validation
or test observation changes the weights after construction.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from deploy_kd import (CLASS_IDS, MAX_MODEL_BYTES, MEAN, STD, Predictor,
                       _letterbox_image, aggregate_member_probabilities,
                       normalized_model_weights, sha256_file)
from student_experiment_evaluation import ExperimentalPredictor

ALLOWED_BACKBONES = {
    "convnext_tiny.fb_in22k_ft_in1k_384",
    "convnext_base.fb_in22k_ft_in1k_384",
}
ALLOWED_VIEWS = ("identity", "hflip2")
DECODERS = (None, "JPEG_draft_target768")


def _read_json(path):
    return json.loads(Path(path).read_text())


def _inventory(model_dir):
    model_dir = Path(model_dir)
    files = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(model_dir)),
                          "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    total = sum(item["bytes"] for item in files)
    return {"files": files, "total_bytes": total,
            "limit_bytes": MAX_MODEL_BYTES, "size_pass": total <= MAX_MODEL_BYTES}


class StudentEnsemble(Predictor):
    """Serial fixed ensemble over 2-3 exported FP16 students; shared image decode."""

    def __init__(self, model_dir, *, device="cuda"):
        model_dir = Path(model_dir).resolve(strict=True)
        config = _read_json(model_dir / "config.json")
        members = config.get("members")
        if (config.get("artifact_type") != "fixed_student_ensemble"
                or config.get("class_ids") != CLASS_IDS
                or config.get("precision") != "float16"
                or config.get("batch_size") != 1
                or config.get("image_size") != 768
                or not isinstance(members, list) or not 2 <= len(members) <= 3
                or config.get("decoder_candidate") not in DECODERS):
            raise ValueError("Malformed fixed student ensemble")
        normalized_model_weights(config.get("model_weights"), len(members))
        manifest_hashes = set()
        loaded_members, models, torch = [], [], None
        for entry in members:
            directory = entry.get("directory")
            if (not isinstance(directory, str) or Path(directory).name != directory
                    or directory in {"", ".", ".."}):
                raise ValueError("Ensemble member directory must be a plain name")
            source = (model_dir / directory).resolve(strict=True)
            if source.parent != model_dir:
                raise ValueError("Ensemble member must live inside the ensemble directory")
            if sha256_file(source / "config.json") != entry.get("config_sha256"):
                raise ValueError("Ensemble member config changed")
            tta = entry.get("tta")
            if tta not in ALLOWED_VIEWS:
                raise ValueError("Unknown ensemble member TTA")
            loaded = ExperimentalPredictor(source, tta=tta, device=device)
            member_config = loaded.config
            if (member_config.get("backbone_name") != entry.get("backbone")
                    or member_config.get("backbone_name") not in ALLOWED_BACKBONES
                    or member_config.get("weights_sha256") != entry.get("weights_sha256")
                    or member_config.get("image_size") != 768
                    or member_config.get("precision") != "float16"):
                raise ValueError("Ensemble member identity changed")
            manifest_hash = member_config.get("official_valid_manifest_sha256")
            if manifest_hash is not None:
                manifest_hashes.add(manifest_hash)
            if torch is None:
                torch = loaded.torch
            loaded_members.append({**loaded.members[0]})
            models.append(loaded.model)
            del loaded
        if len(manifest_hashes) > 1:
            raise ValueError("Ensemble members disagree on the official valid manifest")
        self.torch = torch
        self.device = self.torch.device(device)
        self.model_dir = model_dir
        self.models = models
        self.members = loaded_members
        self.model = models[0]
        self.model_weights = config["model_weights"]
        top_tta = "hflip2" if all(m["tta"] == "hflip2" for m in self.members) else "mixed"
        self.config = {**config, "tta": top_tta, "ensemble": True,
                       "members": self.members, "member_count": len(self.members)}
        from inference_inputs import ExactU8Input
        self.input_lookup = ExactU8Input(self.torch, self.device, MEAN, STD,
                                         image_sizes=[768])
        self.inventory = _inventory(model_dir)
        if not self.inventory["size_pass"]:
            raise ValueError("Actual ensemble inference files exceed 500,000,000 bytes")

    def _predict_image(self, path):
        from inference_inputs import member_results_to_cpu
        torch = self.torch
        if self.config.get("decoder_candidate") == "JPEG_draft_target768":
            from student_draft_pair import draft_image_pixels
            pixels = draft_image_pixels(path, [768])
        else:
            from inference_inputs import load_image_pixels
            pixels = load_image_pixels(path, [768], _letterbox_image)
        results, finite_checks = [], []
        with torch.inference_mode():
            inputs = {768: self.input_lookup.to_tensor(pixels[768])}
            transformed = {(768, "identity"): inputs[768]}
            for member, model in zip(self.members, self.models):
                probabilities = []
                views = ("identity", "hflip") if member["tta"] == "hflip2" else ("identity",)
                for view in views:
                    key = (768, view)
                    if key not in transformed:
                        transformed[key] = torch.flip(inputs[768], dims=(-1,)).contiguous(
                            memory_format=torch.channels_last)
                    logits = model(transformed[key])
                    if tuple(logits.shape) != (1, 10):
                        raise FloatingPointError("Invalid single-image student logits")
                    finite_checks.append(torch.isfinite(logits).all())
                    probabilities.append(torch.sigmoid(logits.float())[0])
                results.append(torch.stack(probabilities).mean(0))
            results = member_results_to_cpu(torch, results, finite_checks)
        return aggregate_member_probabilities(results, self.model_weights)
