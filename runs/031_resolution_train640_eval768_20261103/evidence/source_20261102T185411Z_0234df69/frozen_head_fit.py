"""Train registered FC patches from immutable train-only pre-logit caches on CPU.

No image, validation feature, test data, encoder or GPU is opened. Each training
example is one complete Study; all image/identity/hflip probabilities are averaged
before its hard BCE and pairwise rank score are computed. Fixed final-epoch heads
are retained, with source/cache/plan identities in a separate receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from training_hooks import compliance_run, start_run, epoch_completed, checkpoint_saved


HEAD_KEYS = ("head.fc.weight", "head.fc.bias")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_patch(path: Path, state: dict[str, torch.Tensor]) -> str:
    """Commit a final head patch durably before exposing its compliance event."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        with temporary.open("ab") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return sha256_file(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path: Path, value: Any):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _stamp():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class FitRecipe:
    epochs: int = 12
    batch_studies: int = 256
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    temperature: float = 1.0
    num_threads: int = 4
    bce_weight: float = 1.0
    rank_weight: float = 0.0
    anchor_weight: float = 0.1
    seed: int = 48

    def validate(self):
        for name in ("epochs", "batch_studies", "num_threads"):
            _require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                     name + " must be a positive integer")
        _require(self.num_threads <= 8, "CPU thread limit is eight")
        _require(type(self.seed) is int and 0 <= self.seed < 2**32, "Invalid fixed seed")
        _require(type(self.warmup_epochs) is int and 0 <= self.warmup_epochs < self.epochs,
                 "LR warmup must be shorter than the fixed horizon")
        for name in ("learning_rate", "min_learning_rate", "grad_clip", "temperature"):
            value = getattr(self, name)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     name + " must be positive and finite")
        _require(self.min_learning_rate <= self.learning_rate, "Minimum LR exceeds initial LR")
        for name in ("bce_weight", "rank_weight", "anchor_weight"):
            value = getattr(self, name)
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     name + " must be finite and nonnegative")
        _require(self.bce_weight + self.rank_weight > 0, "A supervised objective is required")


def recipe_from_plan(plan, name):
    _require(plan.get("test_used_for_design_or_fitting") is False
             and plan.get("registered_before_new_head_validation") is True,
             "Plan must explicitly exclude test use and precede new-head validation")
    common = plan.get("common")
    recipes = plan.get("recipes")
    _require(isinstance(common, dict) and isinstance(recipes, dict) and name in recipes,
             "Plan must provide common parameters and the selected named recipe")
    _require(common.get("weight_decay", 0) == 0 and common.get("pos_weight_power", .5) == .5,
             "Registered head fitting retains no weight decay and train-only sqrt positive weights")
    selected = recipes[name]
    _require(isinstance(selected, dict), "Malformed registered recipe")
    fields = set(FitRecipe.__dataclass_fields__)
    values = {key: value for key, value in common.items() if key in fields}
    values.update({key: value for key, value in selected.items() if key in fields})
    config = FitRecipe(**values)
    config.validate()
    return config


def learning_rate_at(config: FitRecipe, step: int, steps_per_epoch: int):
    total = config.epochs * steps_per_epoch
    warmup = config.warmup_epochs * steps_per_epoch
    if warmup and step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return config.min_learning_rate + .5 * (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * progress))


def study_log_probabilities(logits: torch.Tensor, mask: torch.Tensor):
    """Stable log(mean sigmoid(z)) and log(mean sigmoid(-z)) for complete bags."""
    _require(logits.ndim == 3 and logits.shape[-1] == 10
             and mask.shape == logits.shape[:2] and mask.dtype == torch.bool,
             "Expected BxVx10 logits and a boolean BxV view mask")
    _require(bool(mask.any(1).all()), "A complete Study needs at least one feature view")
    _require(bool(torch.isfinite(logits).all()), "Nonfinite cached-head logits")
    log_count = mask.sum(1).to(logits.dtype).log()[:, None]
    excluded = ~mask[:, :, None]
    positive = torch.logsumexp(F.logsigmoid(logits).masked_fill(excluded, -torch.inf), 1) - log_count
    negative = torch.logsumexp(F.logsigmoid(-logits).masked_fill(excluded, -torch.inf), 1) - log_count
    return positive, negative


def objective(logits, original_logits, mask, targets, pos_weights, config: FitRecipe):
    """Equal-Study BCE, equal-class pairwise ranking, and equal-Study logit anchor."""
    _require(targets.shape == (logits.shape[0], 10) and pos_weights.shape == (10,),
             "Expected ten original Study labels and ten train-only positive weights")
    _require(bool(((targets == 0) | (targets == 1)).all()), "Labels must remain original binary labels")
    _require(original_logits.shape == logits.shape, "Original head logits differ in shape")
    positive, negative = study_log_probabilities(logits, mask)
    hard = -(pos_weights[None] * targets * positive + (1 - targets) * negative).mean()
    per_study_anchor = ((logits - original_logits.detach()).square() * mask[:, :, None]).sum(1)
    anchor = (per_study_anchor / mask.sum(1)[:, None]).mean()
    rank_terms = []
    if config.rank_weight:
        scores = positive - negative
        for column in range(10):
            positives = scores[targets[:, column] == 1, column]
            negatives = scores[targets[:, column] == 0, column]
            if positives.numel() and negatives.numel():
                differences = positives[:, None] - negatives[None, :]
                rank_terms.append(F.softplus(-differences / config.temperature).mean())
    ranking = torch.stack(rank_terms).mean() if rank_terms else hard.new_zeros(())
    total = config.bce_weight * hard + config.rank_weight * ranking + config.anchor_weight * anchor
    _require(bool(torch.isfinite(total)), "Nonfinite cached-head objective")
    return total, hard, ranking, anchor, len(rank_terms)


def _cache_path(cache, name, *, train_only=False):
    _require(isinstance(name, str) and name, "Cache file name is missing")
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "Cache filenames must be relative without parent traversal")
    if train_only:
        _require(relative.parts[0] == "train", "Fit may open only declared train cache files")
    path = (cache / relative).resolve(strict=True)
    _require(path.is_relative_to(cache), "Cache filename escapes its immutable root")
    return path


def load_train_cache(cache, encoder):
    """Open only meta.json, the original head, and explicitly named train files."""
    cache = Path(cache).resolve(strict=True)
    meta_path = cache / "meta.json"
    meta = _json(meta_path)
    _require(meta.get("schema") == "frozen_fc_cache_v1", "Unsupported feature-cache schema")
    _require(meta.get("complete", True) is True, "Frozen feature cache is not finalized")
    info = meta.get("encoders", {}).get(encoder)
    _require(isinstance(info, dict), "Selected encoder is absent from the cache")
    dimension = info.get("feature_dim")
    _require(type(dimension) is int and dimension > 0, "Feature dimension is invalid")
    train = meta.get("splits", {}).get("train")
    _require(isinstance(train, dict) and _digest(train.get("manifest_sha256")),
             "Train cache must bind the original train manifest SHA256")
    records_path = _cache_path(cache, train.get("records_file"), train_only=True)
    features_info = train.get("features", {}).get(encoder)
    _require(isinstance(features_info, dict), "Selected train features are absent")
    features_path = _cache_path(cache, features_info.get("file"), train_only=True)
    head_path = _cache_path(cache, info.get("head_file"))
    bindings = {str(meta_path): sha256_file(meta_path)}
    for path, expected in [(records_path, train.get("records_sha256")),
                           (features_path, features_info.get("sha256")),
                           (head_path, info.get("head_sha256"))]:
        _require(_digest(expected) and sha256_file(path) == expected,
                 "Frozen cache file changed: " + str(path))
        bindings[str(path)] = expected
    for field in ("base_config_sha256", "base_weights_sha256"):
        _require(_digest(info.get(field)), "Cache omits original encoder binding: " + field)
    with records_path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    _require(features.dtype == np.float16 and features.shape == (len(records), 2, dimension)
             and list(features.shape) == features_info.get("shape"),
             "Train feature dtype/shape differs from metadata and original record order")
    study_indices, targets, subject_ids, image_ids = {}, {}, {}, set()
    for index, row in enumerate(records):
        _require(row.get("split", "train") == "train", "Only original train records may enter head fitting")
        image_id, key, subject = row.get("image_id"), row.get("study_key"), row.get("subject_id")
        _require(isinstance(image_id, str) and image_id and image_id not in image_ids,
                 "Train image identities must be unique")
        _require(isinstance(key, str) and key and isinstance(subject, str) and subject,
                 "Train Study/patient identity is missing")
        image_ids.add(image_id)
        values = row.get("targets")
        if isinstance(values, str):
            values = json.loads(values)
        _require(isinstance(values, (list, tuple)) and len(values) == 10 and all(v in (0, 1) for v in values),
                 "Cache must retain original ten-class hard labels")
        values = tuple(values)
        _require(key not in targets or targets[key] == values, "Conflicting labels within complete Study")
        _require(key not in subject_ids or subject_ids[key] == subject, "Study crosses patient identities")
        targets[key], subject_ids[key] = values, subject
        study_indices.setdefault(key, []).append(index)
    _require(bool(study_indices), "Empty train feature cache")
    _require(train.get("num_images", len(records)) == len(records)
             and train.get("num_studies", len(study_indices)) == len(study_indices),
             "Train cache coverage differs from its metadata counts")
    labels = np.asarray([targets[key] for key in study_indices], dtype=np.float32)
    positive = labels.sum(0, dtype=np.float64)
    negative = len(labels) - positive
    _require(bool((positive > 0).all()) and bool((negative > 0).all()),
             "Each train class requires positive and negative Studies")
    weights = np.sqrt(negative / positive).astype(np.float32)
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    _require(isinstance(head, dict) and set(head) == set(HEAD_KEYS),
             "Original head cache must contain exactly head.fc.weight and head.fc.bias")
    for name, shape in [(HEAD_KEYS[0], (10, dimension)), (HEAD_KEYS[1], (10,))]:
        tensor = head[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                 and tensor.dtype == torch.float32 and tuple(tensor.shape) == shape
                 and bool(torch.isfinite(tensor).all()), "Invalid original FP32 head tensor: " + name)
    return {"root": cache, "meta": meta, "encoder": info, "features": features,
            "groups": list(study_indices.values()), "labels": labels, "pos_weights": weights,
            "positive_studies": positive.astype(int).tolist(), "negative_studies": negative.astype(int).tolist(),
            "patients": len(set(subject_ids.values())), "head": head,
            "bindings": bindings, "train_manifest_sha256": train["manifest_sha256"]}


def _batch(cache, indices):
    groups = [cache["groups"][int(index)] for index in indices]
    max_views = max(2 * len(group) for group in groups)
    dimension = cache["features"].shape[-1]
    features = np.zeros((len(groups), max_views, dimension), dtype=np.float32)
    mask = np.zeros((len(groups), max_views), dtype=bool)
    for index, group in enumerate(groups):
        values = np.asarray(cache["features"][group]).reshape(-1, dimension)
        _require(bool(np.isfinite(values).all()), "Nonfinite train feature cache")
        features[index, :len(values)] = values
        mask[index, :len(values)] = True
    return torch.from_numpy(features), torch.from_numpy(mask), torch.from_numpy(cache["labels"][indices].copy())


@compliance_run
def fit(*, cache, plan, plan_sha256, encoder, recipe, output):
    plan = Path(plan).resolve(strict=True)
    _require(_digest(plan_sha256) and sha256_file(plan) == plan_sha256,
             "Training plan differs from its independently registered SHA256")
    registration = _json(plan)
    config = recipe_from_plan(registration, recipe)
    values = load_train_cache(cache, encoder)
    _require(values["meta"].get("plan_sha256", plan_sha256) == plan_sha256,
             "Feature cache was extracted under a different registered plan")
    expected_encoder = registration.get("encoders", {}).get(encoder)
    _require(isinstance(expected_encoder, dict), "Selected encoder is absent from the registered plan")
    for field in ("feature_dim", "base_model_dir", "base_config_sha256", "base_weights_sha256"):
        _require(values["encoder"].get(field) == expected_encoder.get(field),
                 "Cache encoder differs from the registered source: " + field)
    _require(values["train_manifest_sha256"] == registration.get("cache", {}).get("train_manifest_sha256"),
             "Feature cache is not the registered original train manifest")
    values["bindings"][str(plan)] = plan_sha256
    output = Path(output).resolve()
    _require(not output.exists(), "Use a fresh head-fit output directory")
    _require(output != values["root"] and not output.is_relative_to(values["root"]),
             "Output must not be inside immutable feature-cache inputs")
    torch.set_num_threads(config.num_threads)
    torch.random.default_generator.manual_seed(config.seed)  # CPU generator only.
    rng = np.random.default_rng(config.seed)
    original_weight = values["head"][HEAD_KEYS[0]].detach().clone()
    original_bias = values["head"][HEAD_KEYS[1]].detach().clone()
    weight = torch.nn.Parameter(original_weight.clone())
    bias = torch.nn.Parameter(original_bias.clone())
    optimizer = torch.optim.Adam([weight, bias], lr=config.learning_rate, weight_decay=0)
    positive_weights = torch.from_numpy(values["pos_weights"])
    count = len(values["groups"])
    steps_per_epoch = math.ceil(count / config.batch_studies)
    output.mkdir(parents=True, exist_ok=False)
    start_run(output, trainer="frozen_head_fit.fit", configuration={
        "encoder": encoder, "recipe": recipe, "training_config": asdict(config),
        "plan_sha256": plan_sha256, "cache_root": str(values["root"]),
        "device": "cpu", "valid_used_for_training": False, "test_used": False},
        source_files=[__file__, plan], resume=None)
    started = time.monotonic()
    receipt = {"schema_version": 1, "artifact_type": "frozen_fc_training_receipt",
        "state": "running", "encoder": encoder, "recipe": recipe, "started_hong_kong": _stamp(),
        "plan_id": registration.get("plan_id"), "training_plan_sha256": plan_sha256,
        "feature_cache_metadata_sha256": values["bindings"][str(values["root"] / "meta.json")],
        "source_files_sha256": values["bindings"], "base_model_dir": values["encoder"]["base_model_dir"],
        "base_config_sha256": values["encoder"]["base_config_sha256"],
        "base_weights_sha256": values["encoder"]["base_weights_sha256"],
        "feature_dim": original_weight.shape[1], "class_ids": list(range(1, 11)),
        "train_manifest_sha256": values["train_manifest_sha256"],
        "num_images": len(values["features"]), "num_studies": count, "num_patients": values["patients"],
        "training_config": asdict(config), "optimizer": "Adam", "weight_decay": 0,
        "pos_weight_formula": "sqrt(negative_train_Studies / positive_train_Studies)",
        "positive_studies": values["positive_studies"], "negative_studies": values["negative_studies"],
        "pos_weights": values["pos_weights"].tolist(),
        "hard_target_unit": "complete Study", "image_views": ["identity", "hflip"],
        "aggregation": "mean sigmoid over every original image and both views within Study before BCE/rank",
        "rank_score": "log(mean sigmoid(logits)) - log(mean sigmoid(-logits))",
        "rank_sampling": "all positive-negative Study pairs per class within uniform shuffled complete-Study batches; equal mean over eligible classes",
        "anchor": "mean per-Study, per-view, per-class squared raw-logit difference from the frozen original head",
        "feature_interface": "eval-mode forward_head(forward_features(input), pre_logits=True), including existing head norm",
        "frozen_backbone": True, "head_path": "head.fc", "weights_source": "fixed_final_epoch_head_parameters",
        "checkpoint_selection": "fixed_final_epoch", "early_stopping": False,
        "device": "cpu", "GPU_used": False, "test_used": False,
        "valid_used_for_training": False, "valid_used_for_checkpoint_selection": False,
        "validation_arrays_opened": False, "raw_images_opened": False,
        "completed_epochs": 0, "optimizer_updates": 0,
        "implementation_sha256": sha256_file(__file__),
        "limitations": ["Training computes the new FC in CPU FP32 on cached FP16 features; actual FP16 GPU inference requires separate validation.",
            "The inherited original encoder/head was selected historically on official valid; this fit does not make it independently validated."]}
    receipt_path = output / "training_receipt.json"
    _write_json(receipt_path, receipt)
    try:
        with (output / "history.jsonl").open("x") as history:
            step = 0
            for epoch in range(1, config.epochs + 1):
                epoch_start = time.monotonic()
                permutation = rng.permutation(count)
                seen, totals, eligible = 0, np.zeros(4, dtype=np.float64), 0
                first_lr = None
                for first in range(0, count, config.batch_studies):
                    indices = permutation[first:first + config.batch_studies]
                    features, mask, targets = _batch(values, indices)
                    with torch.no_grad():
                        reference = F.linear(features, original_weight, original_bias)
                    logits = F.linear(features, weight, bias)
                    total, hard, ranking, anchor, classes = objective(
                        logits, reference, mask, targets, positive_weights, config)
                    rate = learning_rate_at(config, step, steps_per_epoch)
                    first_lr = rate if first_lr is None else first_lr
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_([weight, bias], config.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    _require(bool(torch.isfinite(weight).all()) and bool(torch.isfinite(bias).all()),
                             "Head optimizer produced nonfinite parameters")
                    totals += np.asarray([float(v.detach()) for v in [total, hard, ranking, anchor]]) * len(indices)
                    seen += len(indices)
                    eligible += classes
                    step += 1
                _require(seen == count, "An epoch omitted original complete Studies")
                row = {"epoch": epoch, "studies": seen, "all_original_images_retained": True,
                    "loss": float(totals[0] / seen), "hard_bce": float(totals[1] / seen),
                    "pairwise_rank": float(totals[2] / seen), "anchor": float(totals[3] / seen),
                    "rank_eligible_classes_per_batch_mean": eligible / steps_per_epoch,
                    "optimizer_updates": step, "learning_rate_first": first_lr,
                    "learning_rate_last": rate, "seconds": time.monotonic() - epoch_start,
                    "valid_evaluated": False}
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                epoch_completed(epoch, row)
                receipt.update(completed_epochs=epoch, optimizer_updates=step)
                _write_json(receipt_path, receipt)
                print(f"frozen_head encoder={encoder} recipe={recipe} epoch={epoch}/{config.epochs} "
                      f"loss={row['loss']:.6f} bce={row['hard_bce']:.6f} "
                      f"rank={row['pairwise_rank']:.6f} anchor={row['anchor']:.6f}", flush=True)
        final_fp32 = {HEAD_KEYS[0]: weight.detach().cpu().contiguous(),
                      HEAD_KEYS[1]: bias.detach().cpu().contiguous()}
        final_fp16 = {name: value.half() for name, value in final_fp32.items()}
        _require(all(bool(torch.isfinite(value).all()) for value in final_fp16.values()), "FC patch overflows FP16")
        for precision, state in [("fp32", final_fp32), ("fp16", final_fp16)]:
            path = output / f"patch_{precision}.pt"
            checkpoint_saved(path, _save_patch(path, state), epoch=config.epochs, role="final")
        for path, digest in values["bindings"].items():
            _require(sha256_file(path) == digest, "A frozen training input changed: " + path)
        receipt.update(state="completed", completed_hong_kong=_stamp(),
            wall_seconds=time.monotonic() - started, final_epoch=config.epochs,
            head_patch_fp16_sha256=sha256_file(output / "patch_fp16.pt"),
            head_patch_fp32_sha256=sha256_file(output / "patch_fp32.pt"),
            history_sha256=sha256_file(output / "history.jsonl"),
            head_patch_fp16_bytes=(output / "patch_fp16.pt").stat().st_size,
            head_patch_fp32_bytes=(output / "patch_fp32.pt").stat().st_size,
            source_inputs_unchanged=True)
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(state="failed", failed_hong_kong=_stamp(), error=f"{type(error).__name__}: {error}")
        _write_json(receipt_path, receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--encoder", choices=("A", "B"), required=True)
    parser.add_argument("--recipe", choices=("F1", "F2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = vars(parser.parse_args(argv))
    execute = arguments.pop("execute")
    if not execute:
        _require(sha256_file(arguments["plan"]) == arguments["plan_sha256"], "Registered plan SHA256 changed")
        config = recipe_from_plan(_json(arguments["plan"]), arguments["recipe"])
        print(json.dumps({"preview_only": True, "training_started": False, "GPU_used": False,
                          "recipe": asdict(config), "encoder": arguments["encoder"]}, indent=2))
        return 0
    result = fit(**arguments)
    print(json.dumps({"state": result["state"], "encoder": result["encoder"], "recipe": result["recipe"],
                      "final_epoch": result["final_epoch"], "head_patch_fp16_sha256": result["head_patch_fp16_sha256"]}))
    return 0


if __name__ == "__main__":
    main()
