"""CPU proxy evaluation of the finite, registered frozen-classifier experiment.

This replays CPU FP32 Linear/sigmoid on immutable FP16 valid features. It never
loads an encoder, performs GPU work, opens test inputs or fits parameters. The
best two *registered pairs* are queued for actual FP16 full-valid inference;
cache scores do not certify deployment probabilities, accuracy or latency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from compare_student_candidates import normalized_weights, paired_intervals
from deploy_kd import sha256_file
from frozen_head_fit import HEAD_KEYS, recipe_from_plan
from student_experiment_evaluation import score_probabilities, validation_labels, write_new_json


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SHA256 = "f9a06316cfa8ab10ad4488e60a2dcf934cbc942727edd9ad2c31689823db6aef"
VALID_SHA256 = "c136b2d9f2646fdfa65e0db91d24b1ab5a021e7b038dcaf243855b8779bb5112"
PAIR_MEMBERS = {
    "A_F1_B": ["A_F1", "B"], "A_F2_B": ["A_F2", "B"],
    "A_B_F1": ["A", "B_F1"], "A_B_F2": ["A", "B_F2"],
    "A_F1_B_F1": ["A_F1", "B_F1"], "A_F2_B_F2": ["A_F2", "B_F2"],
}
PAIR_WEIGHTS = [[0.0] + [0.5] * 9, [1.0] + [0.5] * 9]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def bind(path, expected, bindings):
    path = Path(path).resolve(strict=True)
    require(digest(expected) and sha256_file(path) == expected, "Immutable file SHA256 differs: " + str(path))
    bindings[str(path)] = expected
    return path


def cache_path(root, relative, *, split=None):
    require(isinstance(relative, str) and relative, "Missing cache-relative file")
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Cache filename escapes its root")
    if split is not None:
        require(value.parts[0] == split, "Cache file has the wrong original split")
    path = (root / value).resolve(strict=True)
    require(path.is_relative_to(root), "Cache symlink escapes its immutable root")
    return path


def validate_plan(plan):
    require(plan.get("plan_id") == "frozen_head_auc_20261101"
        and plan.get("registered_before_new_head_validation") is True
        and plan.get("test_used_for_design_or_fitting") is False,
        "Require the registered frozen-head plan without test use")
    require(plan.get("cache", {}).get("schema") == "frozen_fc_cache_v1"
        and plan["cache"].get("train_manifest_sha256") == TRAIN_SHA256
        and plan["cache"].get("valid_manifest_sha256") == VALID_SHA256
        and plan["cache"].get("no_test_cache") is True, "Plan changed the original cache/data contract")
    require(plan.get("common", {}).get("epochs") == 12
        and plan["common"].get("validation_tta") == "hflip2", "Only fixed-final E12/hflip2 heads are registered")
    singles = ["A_F1", "A_F2", "B_F1", "B_F2"]
    require(plan.get("single_candidates") == singles, "Single-head candidate set differs from registration")
    pairs = plan.get("pair_candidates")
    require(isinstance(pairs, dict) and set(pairs) == set(PAIR_MEMBERS), "Pair set differs from the six registered rules")
    for name, members in PAIR_MEMBERS.items():
        require(pairs[name].get("members") == members, "Registered pair members changed: " + name)
        actual = normalized_weights(pairs[name].get("weights"), 2)
        require(np.array_equal(actual, np.asarray(PAIR_WEIGHTS)), "Unregistered pair/class weights: " + name)
    for name in ("F1", "F2"):
        recipe_from_plan(plan, name)


def validate_record_order(cached_records, original_records):
    require(len(cached_records) == len(original_records), "Valid cache omits or adds original images")
    seen = set()
    for cached, original in zip(cached_records, original_records):
        require(cached.get("split") == "valid", "Evaluation accepts only original valid cache records")
        for key in ("image_id", "study_key", "subject_id"):
            require(cached.get(key) == original[key], "Cache differs from original valid manifest/order: " + key)
        require(list(cached.get("targets", [])) == list(original["targets"]), "Valid cache labels differ from the immutable manifest")
        require(cached["image_id"] not in seen, "Valid cache repeats an image")
        seen.add(cached["image_id"])


def head_state(path, dimension, dtype):
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(value, dict) and set(value) == set(HEAD_KEYS), "A head must contain exactly the FC weight and bias")
    for name, shape in ((HEAD_KEYS[0], (10, dimension)), (HEAD_KEYS[1], (10,))):
        tensor = value[name]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
            and tensor.dtype == dtype and tuple(tensor.shape) == shape
            and bool(torch.isfinite(tensor).all()), "Head shape/dtype/finiteness differs: " + name)
    return value


def image_probabilities(features, state):
    """FP32 Linear/sigmoid and two-view mean, with explicit CPU-only tensors."""
    require(features.dtype == np.float16 and features.ndim == 3 and features.shape[1] == 2,
            "Expected immutable Nx2xD FP16 feature views")
    dimension = features.shape[2]
    require(tuple(state[HEAD_KEYS[0]].shape) == (10, dimension), "Classifier and feature dimensions differ")
    weight, bias = (state[name].float().cpu() for name in HEAD_KEYS)
    result = np.empty((len(features), 10), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            stop = min(start + 256, len(features))
            batch = np.array(features[start:stop], dtype=np.float32, copy=True)
            require(bool(np.isfinite(batch).all()), "Nonfinite valid features")
            logits = F.linear(torch.from_numpy(batch), weight, bias)
            probabilities = torch.sigmoid(logits).mean(dim=1)
            require(bool(torch.isfinite(probabilities).all()), "Nonfinite CPU classifier output")
            result[start:stop] = probabilities.numpy()
    return result


def study_probabilities(records, image_values):
    require(image_values.shape == (len(records), 10) and bool(np.isfinite(image_values).all())
        and bool((image_values >= 0).all()) and bool((image_values <= 1).all()), "Invalid ten-class image probabilities")
    sums, counts = {}, {}
    for row, probability in zip(records, image_values):
        key = (row["subject_id"], row["study_id"])
        sums.setdefault(key, np.zeros(10, dtype=np.float64))
        sums[key] += probability.astype(np.float64)
        counts[key] = counts.get(key, 0) + 1
    return {key: value / counts[key] for key, value in sums.items()}


def load_valid_cache(cache, plan, plan_sha256, bindings):
    from dataset import read_manifest
    root = Path(cache).resolve(strict=True)
    meta_path = root / "meta.json"
    metadata_sha = sha256_file(meta_path)
    bind(meta_path, metadata_sha, bindings)
    meta = read_json(meta_path)
    require(meta.get("complete") is True and meta.get("schema") == "frozen_fc_cache_v1"
        and meta.get("test_used") is False and meta.get("plan_sha256") == plan_sha256,
        "Require a complete frozen train/valid cache bound to this plan")
    require(meta.get("feature_contract") == plan["cache"].get("feature_contract"),
            "Cache changed the registered actual head.fc input feature contract")
    require(set(meta.get("splits", {})) == {"train", "valid"}
        and set(meta.get("encoders", {})) == {"A", "B"}, "Unexpected cache split or encoder")
    valid = meta["splits"]["valid"]
    train = meta["splits"]["train"]
    require(valid.get("manifest_sha256") == VALID_SHA256
        and train.get("manifest_sha256") == TRAIN_SHA256, "Frozen cache changed original train/valid bindings")
    require(valid.get("num_images") == 1422 and valid.get("num_studies") == 859
        and train.get("num_images") == 30328 and train.get("num_studies") == 18513,
        "Frozen cache does not cover the full original train/valid image/Study sets")
    manifest = bind(PROJECT_ROOT / "data/clean_v1_reviewed/valid_manifest.csv", VALID_SHA256, bindings)
    records = read_manifest(manifest, split="valid")
    labels, views = validation_labels(records)
    require(len(records) == 1422 and len(labels) == 859 and len({key[0] for key in labels}) == 228,
            "Expected all 1422 valid images, 859 Studies and 228 patients")
    record_path = bind(cache_path(root, valid.get("records_file"), split="valid"), valid.get("records_sha256"), bindings)
    with record_path.open() as stream:
        cached_records = [json.loads(line) for line in stream if line.strip()]
    validate_record_order(cached_records, records)
    values = {"root": root, "meta": meta, "meta_sha256": metadata_sha, "records": records,
        "labels": labels, "views": views, "features": {}, "original_heads": {}, "base_probabilities": {}}
    for name in ("A", "B"):
        declaration, actual = plan["encoders"][name], meta["encoders"][name]
        for field, expected in declaration.items():
            require(actual.get(field) == expected, "Cache encoder differs from registered source: " + name + "/" + field)
        dimension = declaration["feature_dim"]
        base = Path(declaration["base_model_dir"]).resolve(strict=True)
        config_path = bind(base / "config.json", declaration["base_config_sha256"], bindings)
        config = read_json(config_path)
        weight_name = config.get("weights_file")
        require(isinstance(weight_name, str) and Path(weight_name).name == weight_name,
                "Base weight filename escapes original export")
        bind(base / weight_name, declaration["base_weights_sha256"], bindings)
        require(config.get("weights_sha256") == declaration["base_weights_sha256"]
            and config.get("backbone_name") == declaration["backbone_name"]
            and config.get("source_epoch") == declaration["source_epoch"]
            and config.get("official_valid_manifest_sha256") == VALID_SHA256,
            "Original base export identity differs from plan/cache")
        head_path = bind(cache_path(root, actual.get("head_file")), actual.get("head_sha256"), bindings)
        values["original_heads"][name] = head_state(head_path, dimension, torch.float32)
        for field, dtype, shape in (("features", np.dtype("float16"), [1422, 2, dimension]),
                                    ("base_probabilities", np.dtype("float32"), [1422, 10])):
            info = valid.get(field, {}).get(name)
            require(isinstance(info, dict), "Cache lacks SHA-bound valid " + field + ": " + name)
            path = bind(cache_path(root, info.get("file"), split="valid"), info.get("sha256"), bindings)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            require(array.dtype == dtype and list(array.shape) == shape and info.get("shape") == shape
                and info.get("dtype") == dtype.name and bool(np.isfinite(array).all()),
                "Valid cache array shape/dtype/finiteness differs: " + field + "/" + name)
            values[field if field == "features" else "base_probabilities"][name] = array
        proof = valid.get("baseline_exact_reproduction", {}).get(name, {})
        require(proof.get("all_image_probabilities_bitwise_equal") is True and proof.get("max_absolute_error") == 0,
                "Cache has no exact original FP16 GPU baseline reproduction proof")
    return values


def load_fitted_head(directory, *, name, cache, plan, plan_sha256, bindings):
    root = Path(directory).resolve(strict=True)
    encoder, recipe = name.split("_")
    config = recipe_from_plan(plan, recipe)
    receipt_path = root / "training_receipt.json"
    bind(receipt_path, sha256_file(receipt_path), bindings)
    receipt = read_json(receipt_path)
    expected = {"artifact_type": "frozen_fc_training_receipt", "state": "completed",
        "encoder": encoder, "recipe": recipe, "training_plan_sha256": plan_sha256,
        "feature_cache_metadata_sha256": cache["meta_sha256"], "final_epoch": 12, "completed_epochs": 12,
        "class_ids": list(range(1, 11)), "training_config": asdict(config),
        "frozen_backbone": True, "head_path": "head.fc", "weights_source": "fixed_final_epoch_head_parameters",
        "checkpoint_selection": "fixed_final_epoch", "early_stopping": False, "device": "cpu", "GPU_used": False,
        "test_used": False, "valid_used_for_training": False, "valid_used_for_checkpoint_selection": False,
        "validation_arrays_opened": False, "source_inputs_unchanged": True, "num_images": 30328, "num_studies": 18513}
    for field in ("base_model_dir", "base_config_sha256", "base_weights_sha256", "feature_dim"):
        expected[field] = plan["encoders"][encoder][field]
    expected["train_manifest_sha256"] = TRAIN_SHA256
    for field, value in expected.items():
        require(receipt.get(field) == value, "Final head receipt differs from registered lineage: " + name + "/" + field)
    source_bindings = receipt.get("source_files_sha256")
    require(isinstance(source_bindings, dict) and source_bindings, "Head receipt omits frozen training input hashes")
    mandatory_sources = {
        str(cache["root"] / "meta.json"): cache["meta_sha256"],
        str(cache_path(cache["root"], cache["meta"]["encoders"][encoder]["head_file"])):
            cache["meta"]["encoders"][encoder]["head_sha256"],
    }
    train = cache["meta"]["splits"]["train"]
    for info in ({"file": train["records_file"], "sha256": train["records_sha256"]}, train["features"][encoder]):
        mandatory_sources[str(cache_path(cache["root"], info["file"], split="train"))] = info["sha256"]
    for path, value in mandatory_sources.items():
        require(source_bindings.get(path) == value, "Head training source/cache binding differs: " + name)
    require(plan_sha256 in source_bindings.values(), "Head training receipt omits the registered plan SHA256")
    for path, value in source_bindings.items():
        source = Path(path).resolve(strict=True)
        require(source.is_relative_to(cache["root"]) or value == plan_sha256,
                "Unexpected external source in train-only head receipt")
        require(not source.is_relative_to(cache["root"] / "valid"), "Head training receipt opened valid feature data")
        bind(source, value, bindings)
    patch_path = bind(root / "patch_fp16.pt", receipt.get("head_patch_fp16_sha256"), bindings)
    history_path = bind(root / "history.jsonl", receipt.get("history_sha256"), bindings)
    with history_path.open() as stream:
        history = [json.loads(line) for line in stream if line.strip()]
    steps = math.ceil(18513 / config.batch_studies)
    require(len(history) == 12 and receipt.get("optimizer_updates") == steps * 12,
            "Head did not complete the fixed twelve full-Study epochs")
    for epoch, row in enumerate(history, 1):
        require(row.get("epoch") == epoch and row.get("studies") == 18513
            and row.get("all_original_images_retained") is True and row.get("valid_evaluated") is False
            and row.get("optimizer_updates") == steps * epoch,
            "Head history omitted train coverage or used validation selection")
    state = head_state(patch_path, expected["feature_dim"], torch.float16)
    return state, {"training_receipt_sha256": bindings[str(receipt_path.resolve())],
        "patch_fp16_sha256": receipt["head_patch_fp16_sha256"], "history_sha256": receipt["history_sha256"],
        "encoder": encoder, "recipe": recipe, "final_epoch": 12,
        "base_config_sha256": expected["base_config_sha256"], "base_weights_sha256": expected["base_weights_sha256"]}


def select_actual_pairs(scores, pair_names):
    require(set(pair_names) == set(PAIR_MEMBERS), "Actual-evaluation shortlist must use exactly the registered pair set")
    return sorted(pair_names, key=lambda name: (-scores[name]["macro_auc"], name))[:2]


def evaluate(*, cache, plan, plan_sha256, fits_root, output):
    plan = Path(plan).resolve(strict=True)
    bindings = {}
    bind(plan, plan_sha256, bindings)
    registration = read_json(plan)
    validate_plan(registration)
    output = Path(output).resolve()
    fits_root = Path(fits_root).resolve(strict=True)
    cache_root = Path(cache).resolve(strict=True)
    require(not output.exists(), "Use a fresh frozen-head comparison output")
    require(not output.is_relative_to(cache_root) and not output.is_relative_to(fits_root),
            "Comparison output cannot alter immutable cache/fit inputs")
    values = load_valid_cache(cache_root, registration, plan_sha256, bindings)
    torch.set_num_threads(registration["common"]["num_threads"])
    images, scores, study_values, baseline_differences, fits = {}, {}, {}, {}, {}
    for encoder in ("A", "B"):
        cpu = image_probabilities(values["features"][encoder], values["original_heads"][encoder])
        saved = np.asarray(values["base_probabilities"][encoder])
        difference = np.abs(cpu.astype(np.float64) - saved.astype(np.float64))
        images[encoder] = cpu
        study_values[encoder] = study_probabilities(values["records"], cpu)
        saved_studies = study_probabilities(values["records"], saved)
        scores[encoder] = score_probabilities(values["labels"], study_values[encoder], values["views"])
        saved_scores = score_probabilities(values["labels"], saved_studies, values["views"])
        baseline_differences[encoder] = {"max_absolute_image_probability_error": float(difference.max()),
            "mean_absolute_image_probability_error": float(difference.mean()),
            "cpu_fp32_head_proxy_auc": scores[encoder]["macro_auc"],
            "saved_cuda_fp16_auc": saved_scores["macro_auc"],
            "macro_auc_delta": scores[encoder]["macro_auc"] - saved_scores["macro_auc"],
            "per_class_auc_delta": [a-b for a, b in zip(scores[encoder]["per_class_auc"], saved_scores["per_class_auc"])],
            "probabilities_bitwise_equal": bool(np.array_equal(cpu, saved)),
            "saved_cuda_fp16_scores": saved_scores,
            "explanation": "CPU FP32 FC accumulation/sigmoid can differ from actual CUDA FP16 FC on the same cached FP16 features; no equivalence is assumed."}
    for name in registration["single_candidates"]:
        state, identity = load_fitted_head(fits_root / name, name=name, cache=values,
            plan=registration, plan_sha256=plan_sha256, bindings=bindings)
        encoder = identity["encoder"]
        images[name] = image_probabilities(values["features"][encoder], state)
        study_values[name] = study_probabilities(values["records"], images[name])
        scores[name] = score_probabilities(values["labels"], study_values[name], values["views"])
        fits[name] = identity
    for name, rule in registration["pair_candidates"].items():
        weights = normalized_weights(rule["weights"], 2)
        blend = sum((images[member].astype(np.float64) * weights[index]
                     for index, member in enumerate(rule["members"])), np.zeros((1422, 10), dtype=np.float64))
        study_values[name] = study_probabilities(values["records"], blend)
        scores[name] = score_probabilities(values["labels"], study_values[name], values["views"])
    original_blend = (np.asarray(values["base_probabilities"]["A"], dtype=np.float64)
                      + np.asarray(values["base_probabilities"]["B"], dtype=np.float64)) / 2
    original_blend[:, 0] = values["base_probabilities"]["B"][:, 0]
    baseline_name = "original_A_B_class1_B_cuda_fp16_reference"
    study_values[baseline_name] = study_probabilities(values["records"], original_blend)
    scores[baseline_name] = score_probabilities(values["labels"], study_values[baseline_name], values["views"])
    keys = sorted(values["labels"])
    targets = np.asarray([values["labels"][key] for key in keys], dtype=np.int8)
    subjects = np.asarray([key[0] for key in keys])
    arrays = {name: np.asarray([predictions[key] for key in keys], dtype=np.float64)
              for name, predictions in study_values.items()}
    selection = registration["selection"]
    bootstrap = paired_intervals(targets, arrays, subjects, samples=selection["patient_bootstrap"],
        seed=selection["bootstrap_seed"], reference=baseline_name)
    current_best = selection["existing_best_valid"]
    for name, score in scores.items():
        score["macro_auc_delta_from_current_actual_best"] = score["macro_auc"] - current_best
        score["macro_auc_delta_from_original_AB_cuda_fp16"] = score["macro_auc"] - scores[baseline_name]["macro_auc"]
    shortlist = select_actual_pairs(scores, registration["pair_candidates"])
    for path, value in bindings.items():
        require(sha256_file(path) == value, "An immutable comparison input changed: " + path)
    report = {"complete": True, "time_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "comparison_kind": "provisional_CPU_FP32_FC_on_cached_FP16_valid_features",
        "plan_sha256": plan_sha256, "cache_meta_sha256": values["meta_sha256"],
        "num_images": 1422, "num_studies": 859, "num_patients": 228,
        "scores": scores, "old_head_cpu_vs_saved_cuda_fp16": baseline_differences,
        "current_best_actual_valid_reference": {"macro_auc": current_best,
            "source": "registered plan selection.existing_best_valid", "predictions_loaded": False,
            "paired_interval_against_this_scalar": False},
        "paired_patient_bootstrap": bootstrap, "fitted_heads": fits,
        "actual_gpu_full_valid_shortlist": [{"pair_name": name, "cached_proxy_macro_auc": scores[name]["macro_auc"],
            "rule": registration["pair_candidates"][name], "status": "pending_actual_GPU_FP16_full_valid_reproduction"}
            for name in shortlist],
        "selection_restricted_to_registered_pairs": True, "selected_for_deployment": False,
        "weights_fitted_in_this_evaluation": False, "thresholds_fitted": False,
        "test_used": False, "GPU_used": False, "encoder_loaded": False,
        "inference_latency_certified": False, "source_inputs_unchanged": True,
        "source_files_sha256": bindings, "implementation_sha256": sha256_file(__file__),
        "limitations": ["All new scores are CPU cache proxies and cannot be reported as actual deployment AUC.",
            "Original-head CPU/CUDA differences are reported rather than assumed zero; shortlisted pairs require actual fixed FP16 inference.",
            "Official valid informed earlier design and checkpoint/rule selection; patient intervals do not remove this selection bias.",
            "The current-best scalar has no paired uncertainty comparison; model size, actual latency and T4 certification remain separate checks."]}
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "registered_plan.json", registration)
    write_new_json(output / "comparison_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "plan", "fits-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    report = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps({"provisional_cached_auc": {name: score["macro_auc"] for name, score in report["scores"].items()},
        "actual_GPU_shortlist": report["actual_gpu_full_valid_shortlist"], "GPU_used": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
