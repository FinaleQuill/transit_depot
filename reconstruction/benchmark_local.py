"""Bounded real-train-input GPU mechanics benchmark for E2 chunk sizes.

One highest-view optimization Study, full global/local inputs, FP16 backward,
AdamW and EMA. Models are discarded; no weights or efficacy metrics are saved.
No development/H/official-valid/test manifest, predictions or images are read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from reconstruction import controller, train as base
from reconstruction.data import read_records
from reconstruction.local_data import LocalStudyDataset
from reconstruction.local_model import LocalStudyModel
from reconstruction.model import study_bce_loss
from training_compliance import read_events


def input_and_parent(config):
    """Bind the optimization input and E0 weights without reading dev outputs."""
    run_root = base.RUN_ROOT.resolve()
    parent_dir = run_root / "E0_base768_seed20261103"
    parent_path = Path(config["parent_checkpoint"]).resolve(strict=True)
    if parent_path != parent_dir / "best.pt":
        raise ValueError("Benchmark accepts only the registered E0 best EMA")
    certificate = controller._read_json(run_root / "controller/E0_verified.json")
    if (certificate["name"] != "E0" or certificate["state"] != "early_stopped"
            or certificate["best_epoch"] != config["parent_best_epoch"]
            or certificate["best_macro_auc"] != config["parent_best_macro_auc"]):
        raise ValueError("E0 completion certificate differs from the requested parent")
    for filename, field in [("history.jsonl", "history_sha256"),
                            ("compliance/events.jsonl", "ledger_sha256")]:
        if base.digest(parent_dir / filename) != certificate[field]:
            raise ValueError(f"E0 evidence differs from its completed certificate: {filename}")
    parent_config = controller._read_json(parent_dir / "configuration.json")
    status = controller._read_json(parent_dir / "status.json")
    if status["state"] != "early_stopped" or status["best_epoch"] != certificate["best_epoch"]:
        raise ValueError("E0 trainer status differs from its certificate")
    for key in ["backbone_name", "image_root", "image_size", "train_manifest", "split_plan",
                "drop_rate", "drop_path", "weight_decay"]:
        if config[key] != parent_config[key]:
            raise ValueError(f"Benchmark changed the registered E0 input: {key}")
    for relative, expected in parent_config["source_sha256"].items():
        if base.digest(base.ROOT / relative) != expected:
            raise ValueError(f"Frozen E0 source changed: {relative}")
    events = read_events(parent_dir / "compliance/events.jsonl")
    saved = [e for e in events if e["event_type"] == "checkpoint_saved"
             and e.get("role") == "best_inner_ema" and e.get("epoch") == config["parent_best_epoch"]]
    checkpoint_sha = base.digest(parent_path)
    if (not saved or Path(saved[-1]["checkpoint_path"]).resolve() != parent_path
            or checkpoint_sha != config["parent_checkpoint_sha256"]
            or checkpoint_sha != saved[-1]["checkpoint_sha256"]):
        raise ValueError("Actual E0 parent weight identity differs")
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (set(payload) != {"ema", "epoch", "configuration", "evidence", "macro_auc"}
            or payload["evidence"] != parent_config or payload["epoch"] != config["parent_best_epoch"]
            or payload["macro_auc"] != config["parent_best_macro_auc"]
            or any(payload["configuration"].get(k) != v for k, v in parent_config.items()
                   if k in payload["configuration"])):
        raise ValueError("Loaded E0 payload differs from its registered identity")
    split_path = Path(config["split_plan"]).resolve(strict=True)
    if split_path != run_root / "splits/split_plan.json" or base.digest(split_path) != parent_config["split_plan_sha256"]:
        raise ValueError("Registered split plan differs")
    split = controller._read_json(split_path)
    train_path = Path(config["train_manifest"]).resolve(strict=True)
    if (split["train_folds"] != [1, 2, 3] or train_path != run_root / "splits/train.csv"
            or str(train_path) != split["roles"]["train"]["path"]
            or base.digest(train_path) != split["roles"]["train"]["sha256"]):
        raise ValueError("Optimization manifest is not the fixed train role")
    records = read_records(train_path)
    if any(row["split"] != "train" for row in records):
        raise ValueError("Benchmark data must originate exclusively from official train")
    dataset = LocalStudyDataset(records, config["image_root"], image_size=config["image_size"],
                                training=True, seed=config["seed"], local_tiles=True, verify_hash=True)
    if len(dataset) != split["roles"]["train"]["studies"]:
        raise ValueError("Optimization Study coverage differs")
    targets = np.stack([records[group[0]]["targets"] for group in dataset.groups])
    positive = targets.sum(0)
    if np.any(positive == 0) or np.any(positive == len(targets)):
        raise ValueError("Optimization class weights lack both labels")
    weights = np.sqrt((len(targets) - positive) / positive).astype(np.float32)
    index = max(range(len(dataset)), key=lambda i: len(dataset.groups[i]))
    dataset.set_epoch(config["augmentation_epoch_offset"] + 1)
    item = dataset[index]
    paths = [dataset.paths[i].resolve(strict=True) for i in dataset.groups[index]]
    binding = {"train_manifest_path": str(train_path), "train_manifest_sha256": base.digest(train_path),
               "split_plan_sha256": base.digest(split_path), "parent_checkpoint_path": str(parent_path),
               "parent_checkpoint_sha256": checkpoint_sha, "parent_epoch": payload["epoch"],
               "selected_study_key": item["study_key"], "source_images":
               [{"path": str(path), "sha256": base.digest(path)} for path in paths],
               "images": len(item["image_ids"]), "regions_per_image": int(item["images"].shape[1]),
               "maximum_train_views": max(map(len, dataset.groups)),
               "tensor_shape": list(item["images"].shape), "positive_studies": positive.tolist()}
    return item, torch.from_numpy(weights), payload["ema"], binding


def run_trial(config, chunk, item, weights, initial_state, args, report, gpu_started):
    random.seed(config["seed"]); np.random.seed(config["seed"])
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    model = LocalStudyModel(config["backbone_name"], None, drop_rate=config["drop_rate"],
        drop_path=config["drop_path"], gradient_checkpointing=True, view_chunk_size=chunk,
        checkpoint_mode="chunks", channels_last=True)
    model.load_e0_state_dict(initial_state)
    model = model.cuda().to(memory_format=torch.channels_last).train()
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(base.parameter_groups(model, config), betas=(.9, .999))
    scaler = torch.amp.GradScaler("cuda")
    images, affines = item["images"].cuda(), item["region_affines"].cuda()
    targets, pos_weight = item["targets"].cuda(), weights.cuda()
    torch.manual_seed(config["seed"] + 1); torch.cuda.manual_seed_all(config["seed"] + 1)
    trial = {"chunk_size": chunk, "state": "running", "samples": [], "attempted_optimizer_updates": 0,
             "successful_optimizer_updates": 0, "skipped_optimizer_updates": 0,
             "same_initial_source_and_rng": True, "same_rng_trajectory_across_chunks_claimed": False}
    report["trials"].append(trial)
    base.write_json(args.output, report)
    for i in range(args.warmup + args.repeats):
        if time.monotonic() - gpu_started >= args.max_gpu_seconds:
            trial["state"] = "budget_exhausted"
            break
        phase = "warmup" if i < args.warmup else "timed"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trial["active_sample"] = {"phase": phase, "index": i + 1, "started_perf_counter": started}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            component = model(images, region_affines=affines, return_components=True)
            loss = study_bce_loss(component["study_logits"], targets, pos_weight,
                                  global_study_logits=component["global_study_logits"], global_aux_weight=0.)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite benchmark loss")
        scaler.scale(loss / config["effective_studies"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        trial["attempted_optimizer_updates"] += 1
        skipped = scaler.get_scale() < scale
        trial["skipped_optimizer_updates"] += int(skipped)
        if not skipped:
            trial["successful_optimizer_updates"] += 1
            base.update_ema(ema, model, trial["successful_optimizer_updates"], config["ema_decay"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        sample = {"phase": phase, "index": i + 1, "seconds": seconds, "loss": float(loss.detach()),
                  "optimizer_update_skipped": skipped, "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "optimizer_state_tensor_bytes": sum(value.numel() * value.element_size()
                      for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))}
        trial["samples"].append(sample)
        trial.pop("active_sample", None)
        base.write_json(args.output, report)
        print(json.dumps({"chunk_size": chunk, **sample}), flush=True)
        if phase == "warmup" and (skipped or not sample["optimizer_state_tensor_bytes"]):
            trial["state"] = "warmup_did_not_allocate_optimizer_state"
            break  # Preserve this failure; do not disguise a cold update as a timed warm result.
    else:
        trial["state"] = "completed"
    timed = [row for row in trial["samples"] if row["phase"] == "timed"]
    if timed:
        trial["mean_timed_seconds"] = statistics.mean(row["seconds"] for row in timed)
        trial["region_inputs_per_second"] = int(images.shape[0] * images.shape[1]) / trial["mean_timed_seconds"]
    trial["comparable_full_update_samples"] = (trial["state"] == "completed"
        and len(timed) == args.repeats and trial["skipped_optimizer_updates"] == 0)
    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-gpu-seconds", type=int, default=120)
    args = parser.parse_args()
    args.output = controller._inside(args.output, base.RUN_ROOT / "E2_20261104/preparation")
    if args.output.exists():
        raise FileExistsError("Preserve existing benchmark evidence; choose another output")
    if args.chunks != [4, 8] or args.warmup != 1 or args.repeats != 2 or not 30 <= args.max_gpu_seconds <= 180:
        parser.error("Registered benchmark: chunks 4 then 8; one warmup and two timed updates; 30..180 second budget")
    config = controller._read_json(args.config)
    if (config["experiment_arm"] != "global_local" or config["use_local"] is not True
            or config["image_size"] != 768 or config["global_aux_weight"] != 0
            or config["checkpoint_mode"] != "chunks" or config["channels_last"] is not True):
        raise ValueError("Benchmark requires the registered local Base768 mechanics")
    report = {"schema": "e2_real_train_chunk_mechanics_v1", "state": "preparing", "created_at": base.now(),
              "config_path": str(args.config.resolve()), "config_sha256": base.digest(args.config),
              "source_sha256": base.digest(__file__), "trials": [], "scientific_training_run": False,
              "efficacy_or_auc_evaluated": False, "model_weights_saved": False,
              "dev_accessed": False, "holdout_H_accessed": False, "official_validation_accessed": False,
              "test_accessed": False, "selection_basis": "execution throughput and memory only",
              "warmup_steps_per_mode": args.warmup, "timed_steps_per_mode": args.repeats,
              "max_gpu_seconds": args.max_gpu_seconds,
              "measurement_scope": "one whole high-view Study per optimizer step; includes forward/backward/AdamW/EMA; excludes data loading, initialization, validation and checkpoint I/O",
              "limitation": "Formal optimization accumulates 32 Studies per update; these six mechanics updates use one Study with loss/32 and are discarded. Not end-to-end epoch throughput or inference timing."}
    base.write_json(args.output, report)
    try:
        item, weights, initial_state, binding = input_and_parent(config)
        report["input_binding"] = binding
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = {"name": torch.cuda.get_device_name(),
                         "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory}
        report["state"] = "running"
        base.write_json(args.output, report)
        gpu_started = time.monotonic()
        for chunk in args.chunks:
            try:
                trial = run_trial(config, chunk, item, weights, initial_state, args, report, gpu_started)
            except torch.OutOfMemoryError as error:
                trial = report["trials"][-1] if report["trials"] and report["trials"][-1]["chunk_size"] == chunk else {"chunk_size": chunk, "samples": []}
                if trial not in report["trials"]:
                    report["trials"].append(trial)
                active = trial.pop("active_sample", None)
                if active is not None:
                    trial["samples"].append({"phase": active["phase"], "index": active["index"],
                        "state": "out_of_memory", "seconds": time.perf_counter() - active["started_perf_counter"],
                        "error": str(error), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                trial.update(state="out_of_memory", error=f"{type(error).__name__}: {error}", comparable_full_update_samples=False)
                base.write_json(args.output, report)
                print(json.dumps({"chunk_size": chunk, "state": "out_of_memory", "error": str(error)}), flush=True)
            gc.collect(); torch.cuda.empty_cache()
            if trial["state"] == "budget_exhausted" or time.monotonic() - gpu_started >= args.max_gpu_seconds:
                break
        report["state"] = "completed" if len(report["trials"]) == 2 and all(t["state"] == "completed" for t in report["trials"]) else "incomplete_or_failed_variant"
        report["comparable"] = len(report["trials"]) == 2 and all(t.get("comparable_full_update_samples", False) for t in report["trials"])
        report["gpu_phase_elapsed_seconds"] = time.monotonic() - gpu_started
        report["total_attempted_optimizer_updates"] = sum(t.get("attempted_optimizer_updates", 0) for t in report["trials"])
        report["total_successful_optimizer_updates"] = sum(t.get("successful_optimizer_updates", 0) for t in report["trials"])
        report["updated_at"] = base.now()
        for source in binding["source_images"]:
            if base.digest(source["path"]) != source["sha256"]:
                raise ValueError("Source train image changed during benchmark")
        base.write_json(args.output, report)
        print(json.dumps({"state": report["state"], "comparable": report["comparable"], "output": str(args.output)}), flush=True)
    except BaseException as error:
        if report["trials"]:
            active = report["trials"][-1].pop("active_sample", None)
            if active is not None:
                report["trials"][-1]["samples"].append({"phase": active["phase"], "index": active["index"],
                    "state": "failed", "seconds": time.perf_counter() - active["started_perf_counter"],
                    "error": f"{type(error).__name__}: {error}"})
        report.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=base.now())
        base.write_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
