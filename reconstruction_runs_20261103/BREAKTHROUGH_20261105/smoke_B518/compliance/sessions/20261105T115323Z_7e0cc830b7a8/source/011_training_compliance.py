"""Durable, local-only lifecycle evidence. This module never trains or loads weights.

One writer per run holds an advisory lock. Each invocation has a separate session;
resuming never invents events for earlier training. A hard kill can leave an open
session, which the next invocation explicitly reports instead of calling complete.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from audit_trail import _append, _canonical, _read_locked

ROOT = Path(__file__).resolve().parent
HONG_KONG = timezone(timedelta(hours=8), "Asia/Hong_Kong")


def _normal(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _normal(value.item())
    raise TypeError(f"Unsupported compliance metadata type: {type(value).__name__}")


def _sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value):
    data = json.dumps(_normal(value), ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def read_events(path):
    """Strict read-only verification. Incomplete tails are never silently omitted."""
    rows, previous = [], None
    for line in Path(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("Incomplete compliance ledger tail; retain it for recovery")
        row = json.loads(line)
        digest = row.get("record_sha256")
        if (row.get("audit_schema_version") != 1
                or row.get("previous_record_sha256") != previous
                or digest != hashlib.sha256(_canonical(
                    {k: v for k, v in row.items() if k != "record_sha256"})).hexdigest()):
            raise ValueError("Compliance ledger hash chain is inconsistent")
        rows.append(row)
        previous = digest
    return rows


class ComplianceRecorder:
    """Capture four lifecycle points synchronously, with no GPU/network access.

    Checkpoint digests must come from the save operation; only stat is used here.
    events.jsonl is authoritative; status.json is a refreshable convenience view.
    Evidence collection errors propagate, rather than silently losing coverage.
    """

    def __init__(self, run_dir, *, trainer, configuration, source_files=(),
                 resume=None, demonstration=False):
        configuration = _normal(configuration)
        _canonical(configuration)
        sources = list(dict.fromkeys(Path(p).resolve() for p in source_files))
        for p in [Path(__file__), ROOT / "audit_trail.py", ROOT / "training_hooks.py"]:
            if p.is_file() and p.resolve() not in sources:
                sources.append(p.resolve())
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
        self.run_dir = Path(run_dir).resolve()
        self.directory = self.run_dir / "compliance"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / ".writer.lock").open("a+b")
        self._finished = False
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock.close()
            raise RuntimeError("Another compliance writer is active for this run") from None
        self.trainer = str(trainer)
        self.demonstration = bool(demonstration)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
        self.ledger = self.directory / "events.jsonl"
        self.session_dir = self.directory / "sessions" / self.session_id
        try:
            self.session_dir.mkdir(parents=True, exist_ok=False)
            config_sha = _write_json(self.session_dir / "configuration.json", configuration)
            source_dir = self.session_dir / "source"
            source_dir.mkdir()
            source_manifest = []
            for i, source in enumerate(sources):
                data = source.read_bytes()
                target = source_dir / f"{i:03d}_{source.name}"
                with target.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                source_manifest.append({"original_path": str(source),
                    "archived_path": str(target.relative_to(self.directory)),
                    "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            _sync_dir(source_dir)
            source_sha = _write_json(self.session_dir / "source_manifest.json", source_manifest)
            versions = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                               if d.metadata.get("Name")}, key=str.lower)
            env_sha = _write_json(self.session_dir / "environment.json", {
                "python_executable": sys.executable, "python_version": sys.version,
                "platform": platform.platform(), "packages": versions,
                "environment_variables_captured": False,
                "cuda_device_initialized": False})
            git = {}
            for key, args in [("commit", ["rev-parse", "HEAD"]),
                              ("status", ["status", "--porcelain=v1"]),
                              ("diff_sha256", ["diff", "--binary", "HEAD"])]:
                try:
                    result = subprocess.run(["git", "-C", str(ROOT), *args],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
                    git[key] = (hashlib.sha256(result.stdout).hexdigest() if key == "diff_sha256"
                                else result.stdout.decode("utf-8", errors="replace").strip())
                except (OSError, subprocess.SubprocessError) as error:
                    git[key] = {"unavailable": type(error).__name__}
            git_sha = _write_json(self.session_dir / "git.json", git)
            prior = self._read_for_append()
            started = {r["session_id"] for r in prior if r.get("event_type") == "training_started"}
            ended = {r["session_id"] for r in prior if r.get("event_type") == "training_ended"}
            self._record("training_started", {
                "configuration_sha256": config_sha, "source_manifest_sha256": source_sha,
                "environment_sha256": env_sha, "git_sha256": git_sha,
                "session_evidence": str(self.session_dir.relative_to(self.directory)),
                "resume_from": str(resume) if resume else None,
                "previous_unclosed_sessions": sorted(started - ended),
                "coverage": "this_invocation_only; no historical events fabricated",
                "final_members_frozen": False,
            })
            _sync_dir(self.directory / "sessions")
            _sync_dir(self.run_dir)
        except BaseException:
            self.close()
            raise

    def _read_for_append(self):
        with self.ledger.open("a+b") as stream:
            return _read_locked(stream, self.ledger)

    def _record(self, event_type, payload):
        if self._finished:
            raise RuntimeError("Compliance session is already closed")
        now = datetime.now(timezone.utc)
        event = _normal({"event_type": event_type, "session_id": self.session_id,
            "recorded_at_utc": now.isoformat(),
            "recorded_at_hong_kong": now.astimezone(HONG_KONG).isoformat(),
            "trainer": self.trainer, "run_dir": str(self.run_dir),
            "demonstration_only": self.demonstration, **payload})
        _canonical(event)
        with self.ledger.open("a+b") as stream:
            rows = _read_locked(stream, self.ledger)
            event = _append(stream, event, rows[-1]["record_sha256"] if rows else None)
        rows.append(event)
        current = [r for r in rows if r.get("session_id") == self.session_id]
        _write_json(self.directory / "status.json", {
            "schema_version": 1, "session_id": self.session_id,
            "state": payload.get("state", "running") if event_type == "training_ended" else "running",
            "last_event": event_type, "last_event_sha256": event["record_sha256"],
            "event_count": len(rows), "session_event_count": len(current),
            "completed_epochs_this_session": [r["epoch"] for r in current
                                                if r.get("event_type") == "epoch_completed"],
            "checkpoint_saves_this_session": sum(r.get("event_type") == "checkpoint_saved" for r in current),
            "demonstration_only": self.demonstration, "final_members_frozen": False,
            "updated_at_hong_kong": event["recorded_at_hong_kong"],
            "hard_kill_policy": "Missing end means unclosed, never completed; next start reports it.",
        })
        return event

    def epoch_completed(self, epoch, metrics):
        if type(epoch) is not int or epoch < 1:
            raise ValueError("A completed epoch must be a positive integer")
        rows = self._read_for_append()
        checkpoints = [{k: r[k] for k in ("checkpoint_path", "checkpoint_sha256", "role", "record_sha256")}
            for r in rows if r.get("session_id") == self.session_id
            and r.get("event_type") == "checkpoint_saved" and r.get("epoch") == epoch]
        return self._record("epoch_completed", {"epoch": epoch, "metrics": metrics,
                                                "checkpoint_events": checkpoints})

    def checkpoint_saved(self, path, digest, *, epoch, role):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("A completed checkpoint save's SHA256 is required")
        if type(epoch) is not int or epoch < 1 or not isinstance(role, str) or not role:
            raise ValueError("Checkpoint epoch and role are required")
        path = Path(path).resolve()
        if path.parent != self.run_dir:
            # Deliberately permit an explicit checkpoint subdirectory, not another run.
            try:
                path.relative_to(self.run_dir)
            except ValueError:
                raise ValueError("Checkpoint must belong to the active run") from None
        stat = path.stat()
        return self._record("checkpoint_saved", {"epoch": epoch, "role": role,
            "checkpoint_path": str(path), "checkpoint_sha256": digest,
            "checkpoint_metadata": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "saved_at_utc": datetime.fromtimestamp(stat.st_mtime_ns / 1e9, timezone.utc).isoformat(),
            "digest_source": "completed_save; reused_without_rereading_weights",
            "retention": "metadata_only; rolling checkpoint binary may later be replaced"})

    def finish(self, state, details=None):
        if self._finished:
            return None
        if not isinstance(state, str) or not state or state == "running":
            raise ValueError("An explicit terminal invocation state is required")
        try:
            event = self._record("training_ended", {"state": state, "details": details or {}})
            _write_json(self.session_dir / "end.json", event)
            return event
        finally:
            self.close()

    def close(self):
        """Release ownership without inventing an end event (e.g. storage failure)."""
        self._finished = True
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
