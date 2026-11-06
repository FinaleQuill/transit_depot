"""CPU-only rolling backups for new student runs; never modify the source run."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

GUARD_BYTES = 80 * 1024**2
# Shared across sibling A/B outputs. Reserve the other 50 MiB of the stage's
# 100 MiB evidence budget for controller logs and evaluation reports.
SMALL_FILES_LIMIT = 50 * 1024**2
LIMITATIONS = [
    "Full local backups are on the same host and do not guarantee cross-instance persistence.",
    "Work EMA snapshots are inference-only; optimizer/raw model/scaler/RNG are absent and resume is unsupported.",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path, expected: str, *, guard: int = 0):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _sha(target) == expected:
        return
    if shutil.disk_usage(target.parent).free < source.stat().st_size + guard:
        raise OSError("Insufficient free space for atomic copy and guard")
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out, 4 * 1024**2)
            out.flush()
            os.fsync(out.fileno())
        if _sha(temporary) != expected or _sha(source) != expected:
            raise ValueError("Source changed or committed hash does not match")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _small_files(run: Path, persistent: Path) -> dict:
    """Only explicit metadata and evidence text/source; no recursive weight copies."""
    candidates = [run / name for name in (
        "config.json", "status.json", "history.jsonl", "checkpoint_audit.jsonl",
        "best_metrics.json", "pretrained_provenance.json")]
    evidence = run / "evidence"
    if evidence.is_dir():
        candidates.extend(p for p in evidence.rglob("*") if p.suffix.lower() in {
            ".py", ".sh", ".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml"})
    files = [p for p in candidates if p.is_file() and not p.is_symlink()]
    destinations = {persistent / "metadata" / p.relative_to(run): p.stat().st_size for p in files}
    old_sizes = {p: p.stat().st_size for sibling in persistent.parent.iterdir()
                 if sibling.is_dir() for p in (sibling / "metadata").rglob("*")
                 if p.is_file()}
    old_sizes.update(destinations)
    if sum(old_sizes.values()) > SMALL_FILES_LIMIT:
        return {"state": "skipped", "reason": "shared A/B backup metadata exceeds 50 MiB reservation"}
    count = 0
    for source in files:
        try:
            _atomic_copy(source, persistent / "metadata" / source.relative_to(run), _sha(source), guard=GUARD_BYTES)
            count += 1
        except (ValueError, OSError):
            return {"state": "retry", "copied_files": count}
    return {"state": "complete", "copied_files": count}


def refresh_run_metadata(run_dir: Path, persistent_dir: Path) -> dict:
    """Synchronize final logger evidence without inspecting any model weights."""
    run, persistent = Path(run_dir).resolve(), Path(persistent_dir).resolve()
    if run == persistent or run.is_relative_to(persistent) or persistent.is_relative_to(run):
        raise ValueError("Source run and persistent directory must be disjoint")
    if not persistent.is_dir():
        return {"state": "retry", "reason": "backup receipt unavailable"}
    with (persistent.parent / ".student_checkpoint_backup.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = persistent / "backup_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            return {"state": "retry", "reason": "backup receipt unavailable"}
        if receipt.get("run_dir") != str(run):
            return {"state": "retry", "reason": "backup receipt belongs to another run"}
        small_files = _small_files(run, persistent)
        # A metadata refresh cannot clear a prior model-backup failure.
        previous_complete = receipt.get("state") == "complete"
        receipt.update(small_files=small_files, metadata_refreshed=small_files["state"] == "complete",
                       updated_utc=datetime.now(timezone.utc).isoformat())
        if small_files["state"] != "complete":
            receipt.update(state="retry", reason="final metadata was not fully synchronized")
        elif previous_complete:
            receipt["state"] = "complete"
        _atomic_json(receipt_path, receipt)
        local_path = receipt.get("local_backup_dir")
        if isinstance(local_path, str) and Path(local_path).is_dir():
            _atomic_json(Path(local_path) / "backup_receipt.json", receipt)
        return receipt


def backup_run(run_dir: Path, local_backup_dir: Path, persistent_dir: Path) -> dict:
    """Back up committed epochs. Return retry on changing sources, never change run_dir.

    Execute with the fixed torch 2.10 training image. All work writes are serialized
    across sibling run directories. Paths must be dedicated, disjoint new outputs.
    """
    import torch

    run, local, persistent = map(lambda p: Path(p).resolve(), (run_dir, local_backup_dir, persistent_dir))
    paths = (run, local, persistent)
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Source run and backup directories must be disjoint")
    local.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(parents=True, exist_ok=True)
    receipt_path = persistent / "backup_receipt.json"
    with (persistent.parent / ".student_checkpoint_backup.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            status = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            return {"state": "retry", "reason": "status unavailable", "limitations": LIMITATIONS}
        old = {}
        if receipt_path.is_file():
            try:
                old = json.loads(receipt_path.read_text())
            except (OSError, ValueError):
                pass
        receipt = {"state": "complete", "run_dir": str(run), "local_backup_dir": str(local),
                   "persistent_dir": str(persistent), "limitations": LIMITATIONS,
                   "checkpoints": dict(old.get("checkpoints", {})),
                   "local_full_backups": dict(old.get("local_full_backups", {}))}
        changed = False
        for role, precision in (("best", "fp32"), ("last", "fp16")):
            digest = status.get(role + "_checkpoint_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                receipt.update(state="retry", reason="committed hashes unavailable")
                break
            source = run / (role + ".pt")
            target = local / source.name
            snapshot = persistent / f"{role}_ema_{precision}.pt"
            previous = receipt["checkpoints"].get(role, {})
            try:
                if _sha(source) != digest:
                    raise ValueError("Committed source hash mismatch")
                if (previous.get("source_checkpoint_sha256") == digest and target.is_file()
                        and snapshot.is_file() and _sha(target) == digest
                        and _sha(snapshot) == previous.get("snapshot_sha256")):
                    continue
                _atomic_copy(source, target, digest)
                receipt["local_full_backups"][role] = {
                    "path": str(target), "sha256": digest, "bytes": target.stat().st_size}
                if role in receipt["checkpoints"]:
                    # A later snapshot refusal must not leave an old receipt claiming
                    # the rolling local file still has its previous checkpoint hash.
                    receipt["checkpoints"][role] = {
                        **receipt["checkpoints"][role], "local_full_path": str(target),
                        "local_full_sha256": digest, "local_full_bytes": target.stat().st_size}
                checkpoint = torch.load(target, map_location="cpu", weights_only=True)
                if _sha(target) != digest:
                    raise ValueError("Local full backup changed while loading")
                receipt["local_full_backups"][role]["source_epoch"] = checkpoint.get("epoch")
                # Copy an explicit metadata allowlist: never propagate arbitrary training tensors.
                keys = ("schema_version", "model_name", "model_config", "class_ids", "num_classes",
                        "inference_ready", "validation_complete",
                        "image_size", "preprocessing", "epoch", "weights_source", "validation",
                        "training_precision", "validation_precision", "inference_precision",
                        "validation_tta", "deployment_tta", "aggregation", "distillation",
                        "loss_config", "source_manifest_sha256", "fold_plan_sha256",
                        "official_valid_manifest_sha256", "pretrained_provenance", "teacher_provenance",
                        "best_epoch", "best_auc", "training_config")
                minimal = {key: checkpoint[key] for key in keys if key in checkpoint}
                state = checkpoint["model_state_dict"]
                if not isinstance(state, dict) or not state:
                    raise ValueError("Missing EMA state")
                converted = {}
                for name, value in state.items():
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("EMA state contains a non-tensor")
                    if value.is_floating_point():
                        if not torch.isfinite(value).all():
                            raise ValueError("Nonfinite EMA state")
                        value = value.to(dtype=torch.float32 if precision == "fp32" else torch.float16)
                        if not torch.isfinite(value).all():
                            raise ValueError("FP16 EMA overflow")
                    converted[name] = value.detach().cpu().clone()
                minimal.update(artifact_type="student_ema_snapshot", full_training_state_included=False,
                               resume_supported=False, original_full_checkpoint_sha256=digest,
                               source_checkpoint_sha256=digest, snapshot_precision=precision,
                               model_state_dict=converted, limitations=LIMITATIONS)
                fd, staging_name = tempfile.mkstemp(prefix="ema_", suffix=".pt", dir=local)
                os.close(fd)
                staging = Path(staging_name)
                try:
                    torch.save(minimal, staging)
                    snapshot_hash = _sha(staging)
                    _atomic_copy(staging, snapshot, snapshot_hash, guard=GUARD_BYTES)
                finally:
                    staging.unlink(missing_ok=True)
                receipt["checkpoints"][role] = {
                    "source_checkpoint_sha256": digest, "source_epoch": checkpoint.get("epoch"),
                    "local_full_path": str(target), "local_full_sha256": digest,
                    "local_full_bytes": target.stat().st_size, "snapshot_path": str(snapshot),
                    "snapshot_sha256": snapshot_hash, "snapshot_bytes": snapshot.stat().st_size,
                    "snapshot_precision": precision, "resume_supported": False}
                changed = True
            except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
                receipt.update(state="retry", reason=f"{role}: {error}")
                break
        if receipt["state"] == "complete" and not changed and old.get("state") == "complete":
            return {**old, "no_op": True}
        receipt["small_files"] = _small_files(run, persistent)
        if receipt["small_files"]["state"] != "complete" and receipt["state"] == "complete":
            receipt.update(state="retry", reason="small metadata files were not fully synchronized")
        receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(receipt_path, receipt)
        _atomic_json(local / "backup_receipt.json", receipt)
        return receipt
