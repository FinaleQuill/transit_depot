"""Score AP-selected best EMA checkpoints and fuse four saved A/B members.

The trainer owns the native E1 migration monitor. This module evaluates selected
weights in actual FP16 batch-1 inference precision. No test data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deploy_kd import StudyAccumulator, aggregate_member_probabilities
from kd_common import sha256_file
from tight_crop_ab_evaluate import (CROP, IMAGE_ROOT, PREPROCESS, ROOT, VALID, WEIGHTS,
                                    paired_bootstrap, paired_report, score_arm,
                                    validation_records)


OLD_PLAN = ROOT / "docs/TIGHT_CROP_AB_PLAN_20261102.json"
EARLY_PLAN = ROOT / "docs/TIGHT_CROP_AB_EARLYSTOP_PLAN_20261102.json"
INIT_SHA = {
    "tiny": "aa3c632e7764b57a522920369e16d9cb4768e43e008c0588439b4438aa88fd6f",
    "base": "9cb733eb65b71eeb0fe7633122dcb085b47fc4c13b4f46f139f5b773a5e805a3",
}


def _read_score(value):
    if isinstance(value, dict):
        return value
    path = Path(value)
    if path.is_dir():
        path /= "score.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _identities():
    for path in (OLD_PLAN, EARLY_PLAN, PREPROCESS, CROP, VALID):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"plan_sha256": sha256_file(OLD_PLAN),
            "earlystop_plan_sha256": sha256_file(EARLY_PLAN),
            "preprocess_sha256": sha256_file(PREPROCESS),
            "crop_sha256": sha256_file(CROP),
            "valid_sha256": sha256_file(VALID)}


def verify_epoch_source(source: Path, *, arm: str, member: str, epoch: int,
                        identities: dict, checkpoint: dict):
    """Check an AP-selected best.pt before constructing an inference model."""
    if arm not in ("A", "B") or member not in INIT_SHA or type(epoch) is not int or not 1 <= epoch <= 25:
        raise ValueError("Expected a registered A/B member and epoch 1..25")
    training = checkpoint.get("training_config", {})
    preprocessing = checkpoint.get("preprocessing", {})
    lineage = checkpoint.get("student_warm_start", {})
    progress = checkpoint.get("progress", {})
    expected_source = Path(json.loads(EARLY_PLAN.read_text())["orchestration"]["run_root"]) / f"{arm}_{member}" / "best.pt"
    if source != expected_source or source.name != "best.pt":
        raise ValueError("Only the registered earlystop25 AP-selected best.pt is eligible")
    if (checkpoint.get("artifact_type") != "kd_student"
            or checkpoint.get("epoch") != epoch
            or checkpoint.get("weights_source") != "ema"
            or checkpoint.get("validation_complete") is not True
            or checkpoint.get("inference_ready") is not True
            or checkpoint.get("official_valid_manifest_sha256") != identities["valid_sha256"]
            or progress.get("completed_epochs") != epoch
            or progress.get("validated_epochs") != epoch
            or training.get("arm") != arm or training.get("member") != member
            or training.get("epochs") != 25
            or training.get("patience") != 5
            or training.get("min_delta") != 1e-4
            or training.get("seed") != 48
            or training.get("erase_probability") != (0.5 if arm == "B" else 0.0)
            or training.get("image_size") != 768
            or training.get("validation_tta") != "hflip2"
            or training.get("ab_plan_sha256") != identities["plan_sha256"]
            or training.get("ab_preprocess_sha256") != identities["preprocess_sha256"]
            or training.get("ab_crop_sha256") != identities["crop_sha256"]
            or preprocessing.get("ab_arm") != arm or preprocessing.get("ab_member") != member
            or preprocessing.get("decoder_candidate") != "full_decode_exact_reference_v1"
            or preprocessing.get("training_random_erasing", {}).get("p") != (0.5 if arm == "B" else 0.0)
            or checkpoint.get("image_size") != 768
            or checkpoint.get("validation_tta") != "hflip2"
            or checkpoint.get("class_ids") != list(range(1, 11))
            or checkpoint.get("num_classes") != 10
            or lineage.get("source_snapshot_sha256") != INIT_SHA[member]):
        raise ValueError("AP-selected source identity or full EMA state differs")
    validation = checkpoint.get("validation", {})
    selected_auc = validation.get("ap_only_macro_auc")
    if (training.get("earlystop_plan_sha256") != identities["earlystop_plan_sha256"]
            or preprocessing.get("earlystop_plan_sha256") != identities["earlystop_plan_sha256"]
            or checkpoint.get("best_epoch") != epoch
            or validation.get("monitor") != "ap_only_macro_auc"
            or validation.get("macro_auc_scope") != "AP-only Studies; all_study_macro_auc retains fullvalid"
            or type(selected_auc) not in (int, float) or not np.isfinite(selected_auc)
            or type(validation.get("all_study_macro_auc")) not in (int, float)
            or not np.isclose(checkpoint.get("best_auc"), selected_auc, rtol=0, atol=1e-12)
            or not np.isclose(validation.get("macro_auc"), selected_auc, rtol=0, atol=1e-12)):
        raise ValueError("New best.pt is not selected by AP-only validation")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint lacks EMA tensor state")
    from model_overnight import (BASE_BACKBONE_NAME, TINY_BACKBONE_NAME,
                                 validate_model_config)
    model_config = validate_model_config(checkpoint.get("model_config", {}))
    expected = TINY_BACKBONE_NAME if member == "tiny" else BASE_BACKBONE_NAME
    if model_config["backbone_name"] != expected:
        raise ValueError("Checkpoint architecture differs from member")
    return state, {**model_config, "gradient_checkpointing": False}


def _score_images(model, arm, records):
    import torch
    from inference_inputs import ExactU8Input
    from deploy_kd import MEAN, STD
    from tight_crop_ab_preprocess import load_pixels

    lookup = ExactU8Input(torch, torch.device("cuda:0"), MEAN, STD, image_sizes=[])
    image_ids, values = [], []
    accumulator = StudyAccumulator(records)
    studies = {}
    with torch.inference_mode():
        for index, row in enumerate(records, 1):
            source = row["path"].read_bytes()
            if hashlib.sha256(source).hexdigest() != row["file_sha256"]:
                raise ValueError("Official valid source SHA256 changed")
            pixels = load_pixels(source, arm)
            image = lookup.to_tensor(pixels)
            flipped = torch.flip(image, dims=(-1,)).contiguous(memory_format=torch.channels_last)
            logits = [model(tensor) for tensor in (image, flipped)]
            if any(not bool(torch.isfinite(item).all()) for item in logits):
                raise FloatingPointError("Nonfinite validation logits")
            probability = torch.stack([torch.sigmoid(item.float())[0] for item in logits]).mean(0)
            value = probability.cpu().numpy().astype(np.float64)
            image_ids.append(row["image_id"])
            values.append(value)
            completed = accumulator.add(row, value)
            if completed is not None:
                subject, study, aggregate = completed
                studies[f"{subject}/{study}"] = aggregate
            if index % 100 == 0 or index == len(records):
                print(f"[{arm} epoch score] {index}/{len(records)}", flush=True)
    if not accumulator.complete() or len(studies) != 859:
        raise RuntimeError("Incomplete official valid Study coverage")
    return np.asarray(image_ids), np.stack(values), studies


def score_epoch(*, arm: str, member: str, source: Path,
                output: Path, image_root: Path = IMAGE_ROOT):
    """Evaluate one selected EMA in actual FP16 batch-1 inference precision."""
    import torch
    from inference_reproducibility import configure_inference
    from model_overnight import build_overnight_model

    identities = _identities()
    source = Path(source).resolve(strict=True)
    digest = sha256_file(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    epoch = checkpoint.get("epoch")
    if sha256_file(source) != digest:
        raise RuntimeError("Source checkpoint changed while loading")
    state, model_config = verify_epoch_source(source, arm=arm, member=member,
        epoch=epoch, identities=identities, checkpoint=checkpoint)
    if output.exists():
        raise FileExistsError(output)
    records, labels, views, _ = validation_records(VALID, image_root)
    configure_inference(torch_module=torch)
    if not torch.cuda.is_available():
        raise RuntimeError("Actual FP16 epoch scoring requires CUDA")
    torch.set_num_threads(1)
    model = build_overnight_model(pretrained=False, **model_config)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to("cuda:0", dtype=torch.float16,
                                          memory_format=torch.channels_last)
    output.mkdir(parents=True)
    export = output / "export"
    export.mkdir()
    weights = {name: value.detach().cpu().half() if value.is_floating_point()
               else value.detach().cpu() for name, value in state.items()}
    if any(not bool(torch.isfinite(value).all()) for value in weights.values()
           if value.is_floating_point()):
        raise FloatingPointError("EMA overflows FP16")
    torch.save(weights, export / "state_fp16.pt")
    export_config = {"artifact_type": "tight_crop_ab_epoch_fp16", "arm": arm,
        "member": member, "source_epoch": epoch, "source_kind": "ap_selected_best",
        "source_sha256": digest, "weights_sha256": sha256_file(export / "state_fp16.pt"),
        "model_config": model_config, "decoder": "full_decode_exact_reference_v1",
        "precision": "float16", "tta": "hflip2", **identities}
    (export / "config.json").write_text(json.dumps(export_config, indent=2) + "\n")
    image_ids, probabilities, studies = _score_images(model, arm, records)
    np.savez_compressed(output / "image_probs.npz", image_ids=image_ids,
                        probabilities=probabilities)
    scores = score_arm(sorted(labels), labels, views, studies)
    receipt = {"arm": arm, "member": member, "epoch": epoch,
               "source_kind": export_config["source_kind"],
               "source_sha256": digest, "image_probs_sha256": sha256_file(output / "image_probs.npz"),
               "export_weights_sha256": export_config["weights_sha256"],
               "ap_only_macro_auc": scores["ap_only"]["macro_auc"],
               "all_macro_auc": scores["all"]["macro_auc"],
               "scores": scores, "num_images": len(image_ids),
               "test_used": False, "selection_bias_corrected": False, **identities}
    (output / "score.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def evaluate_selected_pair(selection: Path, output: Path, *, image_root: Path = IMAGE_ROOT):
    """Fuse saved FP16 per-image probabilities with the original class rule."""
    identities = _identities()
    selected = json.loads(Path(selection).read_text(encoding="utf-8")).get("selected")
    names = [f"{arm}_{member}" for arm in ("A", "B") for member in ("tiny", "base")]
    if not isinstance(selected, dict) or set(selected) != set(names):
        raise ValueError("Selection must bind four A/B member score directories")
    records, labels, views, subjects = validation_records(VALID, image_root)
    expected_ids = [row["image_id"] for row in records]
    rows, images = {}, {}
    for name in names:
        directory = Path(selected[name]).resolve(strict=True)
        row = _read_score(directory)
        arm, member = name.split("_", 1)
        if row.get("arm") != arm or row.get("member") != member:
            raise ValueError("Selected member identity differs")
        if row.get("source_kind") != "ap_selected_best":
            raise ValueError("Final pair requires four AP-selected best.pt sources")
        if any(row.get(key) != value for key, value in identities.items()):
            raise ValueError("Selected score sources differ from frozen plans or valid")
        path = directory / "image_probs.npz"
        if sha256_file(path) != row.get("image_probs_sha256"):
            raise ValueError("Selected image probabilities changed")
        export = json.loads((directory / "export/config.json").read_text())
        if (export.get("source_epoch") != row.get("epoch")
                or export.get("weights_sha256") != row.get("export_weights_sha256")
                or sha256_file(directory / "export/state_fp16.pt") != row.get("export_weights_sha256")):
            raise ValueError("Selected FP16 export changed")
        with np.load(path, allow_pickle=False) as data:
            ids = data["image_ids"].tolist()
            values = data["probabilities"]
        if (ids != expected_ids or values.shape != (len(records), 10)
                or not np.isfinite(values).all()):
            raise ValueError("Selected probabilities do not cover official valid in order")
        rows[name], images[name] = row, values
    predictions = {}
    for arm in ("A", "B"):
        accumulator = StudyAccumulator(records)
        studies = {}
        for index, record in enumerate(records):
            probability = aggregate_member_probabilities(
                [images[f"{arm}_tiny"][index], images[f"{arm}_base"][index]], WEIGHTS)
            completed = accumulator.add(record, probability)
            if completed is not None:
                subject, study, values = completed
                studies[f"{subject}/{study}"] = values
        if not accumulator.complete() or len(studies) != 859:
            raise RuntimeError("Selected pair omitted official valid Studies")
        predictions[arm] = studies
    keys = sorted(labels)
    scores = {arm: score_arm(keys, labels, views, predictions[arm]) for arm in ("A", "B")}
    intervals = paired_bootstrap(keys, labels, subjects, views, predictions)
    deltas = {name: None if scores["A"][name]["macro_auc"] is None
              or scores["B"][name]["macro_auc"] is None else
              scores["B"][name]["macro_auc"] - scores["A"][name]["macro_auc"]
              for name in scores["A"]}
    metrics = {**identities, "num_images": len(records), "num_studies": len(keys),
        "num_patients": len(set(subjects.values())), "members": rows,
        "selected_epochs": {name: rows[name]["epoch"] for name in names},
        "arms": scores, "B_minus_A_macro_auc": deltas,
        "patient_bootstrap": intervals, "precision": "float16", "tta": "hflip2",
        "decoder": "full_decode_exact_reference_v1", "class_weights": WEIGHTS,
        "test_used": False, "selection_bias_corrected": False,
        "selection_metric": "AP-only Study macro-AUC", "T4_certified": False}
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    report = paired_report(metrics).replace("Fixed E2 final EMA",
        "AP-only early-stop selected best.pt EMA (up to E25)")
    report += "\nSelected epochs: " + ", ".join(f"{name}=E{metrics['selected_epochs'][name]}" for name in names)
    report += "\nAP-only official valid selected each member's epoch and then evaluated the pair. "
    report += "The patient intervals are conditional on this selection and do not correct selection bias or prove online generalization.\n"
    (output / "REPORT.md").write_text(report)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("score", "final"), required=True)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--member", choices=("tiny", "base"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "score":
        if any(value is None for value in (args.arm, args.member, args.source)) or args.selection:
            parser.error("score requires --arm --member --source and no --selection")
        score_epoch(arm=args.arm, member=args.member, source=args.source, output=args.output)
    else:
        if args.selection is None or any(value is not None for value in
                                         (args.arm, args.member, args.source)):
            parser.error("final requires only --selection --output")
        evaluate_selected_pair(args.selection, args.output)


if __name__ == "__main__":
    main()
