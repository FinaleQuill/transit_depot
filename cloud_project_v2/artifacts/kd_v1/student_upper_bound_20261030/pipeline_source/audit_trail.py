"""Append checkpoint evidence using already computed digests, without reading weights.

The ledger retains metadata for every save. It does not retain every checkpoint
binary: rolling last.pt/best.pt files can subsequently be replaced. Only a
truncated final JSONL fragment may be repaired, after preserving its exact bytes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def capture_source_version(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Hash only the explicitly supplied small code/config files, once per run."""
    files = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for p in paths}
    return {"files": files, "manifest_sha256": hashlib.sha256(_canonical(files)).hexdigest()}


def training_output_has_artifacts(directory: str | Path) -> bool:
    """The CLI logger may create evidence/ before a new training run starts.

    Every other entry, including any checkpoint, config, history or audit
    ledger, retains the original protection against accidental fresh starts.
    """
    directory = Path(directory)
    return directory.exists() and any(
        child.name != "evidence" or not child.is_dir() for child in directory.iterdir())


def _append(stream, event: dict[str, Any], previous: str | None) -> dict[str, Any]:
    event = {"audit_schema_version": 1, **event, "previous_record_sha256": previous}
    event["record_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
    stream.seek(0, os.SEEK_END)
    stream.write(_canonical(event) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())
    return event


def _read_locked(stream, ledger: Path) -> list[dict[str, Any]]:
    stream.seek(0)
    raw = stream.read()
    rows: list[dict[str, Any]] = []
    offset = 0
    previous = None
    for line in raw.splitlines(keepends=True):
        complete = line.endswith(b"\n")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if complete or offset + len(line) != len(raw):
                raise ValueError("Checkpoint audit ledger contains a corrupt complete record")
            # Preserve the original interrupted bytes before repairing only the tail.
            fragment = ledger.with_name(ledger.name + ".partial." + uuid.uuid4().hex)
            with fragment.open("xb") as evidence:
                evidence.write(line)
                evidence.flush()
                os.fsync(evidence.fileno())
            stream.seek(offset)
            stream.truncate()
            recovery = _append(stream, {
                "event_type": "truncated_tail_recovery", "recorded_at_utc": _utc_now(),
                "preserved_fragment": str(fragment), "fragment_bytes": len(line),
                "fragment_sha256": hashlib.sha256(line).hexdigest(),
                "explanation": "Interrupted final record preserved separately before tail repair",
            }, previous)
            rows.append(recovery)
            break
        if not isinstance(row, dict) or row.get("audit_schema_version") != 1:
            raise ValueError("Checkpoint audit ledger record schema is unsupported")
        digest = row.get("record_sha256")
        unhashed = {k: v for k, v in row.items() if k != "record_sha256"}
        if (row.get("previous_record_sha256") != previous
                or digest != hashlib.sha256(_canonical(unhashed)).hexdigest()):
            raise ValueError("Checkpoint audit ledger hash chain is inconsistent")
        if not complete:
            # A complete JSON record whose newline was interrupted needs no data removal.
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        previous = digest
        offset += len(line)
    return rows


def record_checkpoint(checkpoint_path: str | Path, sha256_digest: str, *, epoch: int,
                      role: str, ledger_path: str | Path | None = None,
                      source_version: Mapping[str, Any] | None = None,
                      evidence_source: str = "trainer", saved_at_utc: str | None = None,
                      observed_status_at: str | None = None,
                      checkpoint_metadata: Mapping[str, Any] | None = None,
                      historical: bool = False) -> dict[str, Any]:
    """Durably record one completed save; reuse its digest, never reopen weights.

    For historical/status-only recovery, set historical=True. That mode never
    associates a now-replaced file's stat values with an old hash. Any supplied
    metadata must itself come from the saved evidence, not a different version.
    Native trainer and external monitor may call concurrently; exact save events
    (path, epoch, role, digest) are deduplicated under the same advisory file lock.
    """
    if not isinstance(sha256_digest, str) or re.fullmatch(r"[0-9a-f]{64}", sha256_digest) is None:
        raise ValueError("A completed save's exact SHA256 is required")
    if type(epoch) is not int or epoch < 1 or role not in {"last", "best"}:
        raise ValueError("Checkpoint audit requires a positive epoch and last/best role")
    checkpoint = Path(checkpoint_path).absolute()
    ledger = Path(ledger_path).absolute() if ledger_path else checkpoint.parent / "checkpoint_audit.jsonl"
    metadata = dict(checkpoint_metadata or {})
    if checkpoint_metadata is None and not historical:
        stat = checkpoint.stat()
        metadata = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if saved_at_utc is None and metadata.get("mtime_ns") is not None:
        saved_at_utc = datetime.fromtimestamp(metadata["mtime_ns"] / 1_000_000_000,
                                               timezone.utc).isoformat()
    event = {
        "event_type": "checkpoint_saved", "checkpoint_path": str(checkpoint),
        "epoch": epoch, "role": role, "checkpoint_sha256": sha256_digest,
        "checkpoint_metadata": metadata, "saved_at_utc": saved_at_utc,
        "observed_status_at": observed_status_at, "recorded_at_utc": _utc_now(),
        "evidence_source": evidence_source, "historical_recovery": historical,
        "source_version": dict(source_version or {}),
        "retention": "metadata_only; rolling checkpoint binary may later be replaced",
    }
    _canonical(event)  # Reject invalid metadata before opening or changing the ledger.
    ledger.parent.mkdir(parents=True, exist_ok=True)
    key = (str(checkpoint), epoch, role, sha256_digest)
    with ledger.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            rows = _read_locked(stream, ledger)
            for prior in rows:
                prior_key = (prior.get("checkpoint_path"), prior.get("epoch"),
                             prior.get("role"), prior.get("checkpoint_sha256"))
                if prior.get("event_type") == "checkpoint_saved" and prior_key == key:
                    return prior
            return _append(stream, event, rows[-1]["record_sha256"] if rows else None)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_checkpoint_audit(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Read and verify complete records, recovering a preserved interrupted tail."""
    ledger = Path(ledger_path)
    with ledger.open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            return _read_locked(stream, ledger)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
