"""Close the user-truncated E0/E1 review without changing training evidence.

CPU only. Validates checkpoints as bytes, cached predictions and patient-paired
metrics; never loads a model, resumes training, or opens H/official valid/test.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from reconstruction import controller as ctl
from reconstruction.diagnostics import (
    align_reference, diagnose, load_bound_predictions, write_diagnostics,
)
from reconstruction.evaluate import evaluate_studies


def independent_auc(targets, probabilities):
    """Direct positive-versus-negative comparisons, independent of rank helpers."""
    values = []
    for c in range(10):
        positive = probabilities[targets[:, c] == 1, c]
        negative = probabilities[targets[:, c] == 0, c]
        if not len(positive) or not len(negative):
            raise ValueError("Every class must have both labels")
        wins = 0.0
        for block in np.array_split(positive, max(1, (len(positive) + 127) // 128)):
            wins += np.count_nonzero(block[:, None] > negative)
            wins += 0.5 * np.count_nonzero(block[:, None] == negative)
        values.append(wins / (len(positive) * len(negative)))
    return values


def equal_metrics(left, right, path="metrics"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f"Metric fields differ: {path}")
        for key in left:
            equal_metrics(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError(f"Metric lengths differ: {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            equal_metrics(a, b, f"{path}[{i}]")
    elif isinstance(left, float):
        if not isinstance(right, (int, float)) or not np.isclose(left, right, rtol=0, atol=1e-12):
            raise ValueError(f"Metric differs: {path}: {left} versus {right}")
    elif left != right:
        raise ValueError(f"Metric differs: {path}")


def verify_user_truncated(job, accepted_epochs):
    ctl.verify_resume(job)  # Checks closed lifecycle and the actual complete last checkpoint.
    output = Path(job["output"])
    status = ctl._read_json(output / "status.json")
    events = ctl._closed_sessions(output)
    history = ctl._history(output / "history.jsonl")
    if status["state"] != "interrupted" or len(history) != accepted_epochs:
        raise ValueError("User cutoff does not match the actual interrupted training")
    if [r["epoch"] for r in history] != list(range(1, accepted_epochs + 1)):
        raise ValueError("History must contain exactly the completed epochs")
    epoch_events = [e for e in events if e["event_type"] == "epoch_completed"]
    if [e["metrics"] for e in epoch_events] != history:
        raise ValueError("Complete epoch events and history differ")
    if any(not ctl._finite_auc(row["macro_auc"]) for row in history):
        raise ValueError("Incomplete/nonfinite history")
    best = max(history, key=lambda row: row["macro_auc"])
    if history[-1]["best_epoch"] != best["epoch"] or history[-1]["best_macro_auc"] != best["macro_auc"]:
        raise ValueError("History best selection differs")
    checkpoints = [e for e in events if e["event_type"] == "checkpoint_saved"
                   and e.get("role") == "best_inner_ema" and e.get("epoch") == best["epoch"]]
    if (not checkpoints or Path(checkpoints[-1]["checkpoint_path"]).resolve() != output / "best.pt"
            or ctl.sha(output / "best.pt") != checkpoints[-1]["checkpoint_sha256"]):
        raise ValueError("Best checkpoint differs from its saved evidence")
    return {"name": job["name"], "trainer_state": status["state"],
            "completed_epochs": len(history), "best_epoch": best["epoch"],
            "best_macro_auc": best["macro_auc"], "user_accepted_cutoff": accepted_epochs,
            "registered_early_stop_reached": False,
            "history_sha256": ctl.sha(output / "history.jsonl"),
            "ledger_sha256": ctl.sha(output / "compliance/events.jsonl"),
            "best_checkpoint_sha256": checkpoints[-1]["checkpoint_sha256"]}


def render_curves(destination, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, rows in histories.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["macro_auc"] for r in rows], marker="o", label=arm)
        axes[1].plot(epochs, [r["mean_loss"] for r in rows], marker="o", label=arm)
    for ax in axes:
        ax.set_xlabel("Completed epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Inner-development Macro-AUC")
    axes[1].set_ylabel("Training weighted BCE")
    fig.suptitle("E1: 12 complete epochs; epoch 13 interrupted; user declined continuation", fontsize=11)
    fig.tight_layout()
    fig.savefig(destination / "learning_curves.png", dpi=160)
    fig.savefig(destination / "learning_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-e1-completed-epochs", type=int, required=True)
    args = parser.parse_args()
    if args.accept_e1_completed_epochs != 12:
        parser.error("This closure records the user's explicit 12-complete-epoch cutoff")
    output = ctl._inside(args.output, ctl.RUN_ROOT)
    if output.exists():
        raise FileExistsError("Preserve existing reviews; use a new output directory")
    directory = ctl.RUN_ROOT / "controller"
    with (directory / ".controller.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                capture_output=True, text=True, check=True, timeout=20)
        if any(name.startswith("cxr-reconstruct-") for name in docker.stdout.splitlines()):
            raise RuntimeError("A research container is still running")
        plan = ctl._read_json(directory / "frozen_plan.json")
        if [j["name"] for j in plan["jobs"]] != ["E0", "E1"]:
            raise ValueError("Unexpected queue")
        original_status = ctl._read_json(directory / "STATUS.json")
        if original_status["state"] != "paused" or original_status.get("active_container") is not None:
            raise ValueError("Queue must remain truthfully paused without an active container")
        protected = [directory / name for name in ["STATUS.json", "PROGRESS.md", "STOP", "frozen_plan.json"]]
        for job in plan["jobs"]:
            d = Path(job["output"])
            protected.extend(d / name for name in ["status.json", "configuration.json", "history.jsonl",
                             "compliance/status.json", "compliance/events.jsonl"])
        protected.append(Path(plan["jobs"][1]["output"]) / "STOP")
        original_hashes = {str(p): ctl.sha(p) for p in protected}
        print("Verifying frozen sources, manifests and actual E0 checkpoints", flush=True)
        ctl.verify_plan(plan)
        e0 = ctl.verify_success(plan["jobs"][0])
        print("Verifying interrupted E1 lifecycle and actual epoch-12/epoch-7 checkpoints", flush=True)
        e1 = verify_user_truncated(plan["jobs"][1], args.accept_e1_completed_epochs)
        a = load_bound_predictions(Path(plan["jobs"][0]["output"]) / f"epoch_{e0['best_epoch']:02d}")
        b = load_bound_predictions(Path(plan["jobs"][1]["output"]) / f"epoch_{e1['best_epoch']:02d}")
        for item, verified in [(a, e0), (b, e1)]:
            if (item["report"].get("metadata", {}).get("epoch") != verified["best_epoch"]
                    or item["report"]["macro_auc"] != verified["best_macro_auc"]):
                raise ValueError("Selected prediction report differs from the checkpoint/history best")
            for field in ["prediction_path", "report_path"]:
                path = item["binding"][field]
                original_hashes[path] = ctl.sha(path)
        b_aligned = align_reference(a, b)
        for item in [a, b]:
            equal_metrics(independent_auc(item["targets"], item["probabilities"]), item["report"]["per_class_auc"])
        prior_path = directory / "comparison_poststop_20261104/report.json"
        prior = ctl._read_json(prior_path)
        prior_predictions = prior_path.parent / "predictions.npz"
        for path in [prior_path, prior_predictions]:
            original_hashes[str(path)] = ctl.sha(path)
        if ctl.sha(prior_predictions) != prior["predictions_sha256"]:
            raise ValueError("Prior paired prediction hash differs")
        for arm, item, epoch in [("E0", a, e0["best_epoch"]), ("E1", b, e1["best_epoch"])]:
            key = f"{arm}_epoch{epoch:02d}"
            if prior["metadata"]["source_prediction_sha256"][key] != item["binding"]["predictions_sha256"]:
                raise ValueError("Prior comparison used different source predictions")
        print("Replaying 1,000 patient-paired bootstrap samples and verifying prior numbers", flush=True)
        paired = evaluate_studies(b_aligned, a["targets"], a["study_keys"], a["patient_ids"], a["views"],
                                  reference_probabilities=a["probabilities"], bootstrap_repeats=1000, seed=20261103)
        for key, value in paired.items():
            equal_metrics(value, prior[key], key)
        print("Computing exact patient-removal sensitivity and weak-class review candidates", flush=True)
        diagnostics = diagnose(b, reference=a, model_stage="incomplete_training_epoch")
        # Keep the original diagnostic stage honest: selected_completed_run is invalid for interrupted E1.
        diagnostics["user_closure_context"] = {
            "selected_complete_epoch": e1["best_epoch"], "completed_training_epochs": 12,
            "training_state": "interrupted", "resume_requested": False,
            "interpretation": "Selected complete epoch from a user-truncated run; not normal early stopping"}
        histories = {j["name"]: ctl._history(Path(j["output"]) / "history.jsonl") for j in plan["jobs"]}
        matched = {arm: max(rows[:12], key=lambda r: r["macro_auc"])["macro_auc"]
                   for arm, rows in histories.items()}
        for path, value in original_hashes.items():
            if ctl.sha(path) != value:
                raise ValueError(f"Training evidence changed while reviewing: {path}")
        output.mkdir()
        ctl.write_json(output / "paired_report.json", paired)
        write_diagnostics(output / "patient_diagnostics", diagnostics)
        render_curves(output, histories)
        summary = {
            "schema": "e0_e1_user_cutoff_review_v1", "created_at": ctl.now(),
            "state": "evaluation_complete_user_cutoff", "evaluation_complete": True,
            "original_controller_state": original_status["state"], "original_queue_completed": False,
            "user_instruction": "E1第13轮不用跑了，前面12轮应该够了",
            "user_accepted_completed_epochs": {"E0": 14, "E1": 12},
            "partial_epoch_13_included": False, "training_resumed": False,
            "E0": e0, "E1": e1, "matched_first_12_epochs_best": matched,
            "same_epoch_auc": {str(epoch): {arm: rows[epoch-1]["macro_auc"] for arm, rows in histories.items()}
                               for epoch in [7, 8, 12]},
            "source_bindings": {"E0": a["binding"], "E1": b["binding"]},
            "prior_comparison": {"path": str(prior_path), "sha256": ctl.sha(prior_path),
                                 "all_recomputed_metrics_match_atol": 1e-12},
            "protected_evidence_sha256": original_hashes,
            "independent_direct_pair_auc_verified": True,
            "diagnostic_scope": "cached internal-development Study predictions only",
            "checkpoint_bytes_verified": True, "model_weights_loaded": False,
            "holdout_H_accessed": False, "official_validation_accessed": False, "test_accessed": False,
            "new_training_runs": 0, "source_sha256": ctl.sha(__file__),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
        }
        for path, value in original_hashes.items():
            if ctl.sha(path) != value:
                raise ValueError(f"Training/prediction evidence changed before publication: {path}")
        ctl.write_json(output / "STATUS.json", summary)
        print(json.dumps({"output": str(output), "state": summary["state"],
                          "macro_auc_delta": paired["overall"]["macro_auc_delta"],
                          "ci95": paired["overall"]["patient_bootstrap"]["paired_delta_macro_auc"]["ci95"]}), flush=True)


if __name__ == "__main__":
    main()
