"""Reject only retired launches from an explicitly declared teacher run.

This standard-library guard never opens model weights or training data. The
already-running fold retains its in-memory code; later old-scheduler launches
can exit cleanly before logging/model construction with the distinct code 75.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HANDOFF_EXIT_CODE = 75
HANDOFF_ARTIFACT = "kd_planned_handoff"


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def reject_retired_fold(output_path: str | Path, fold: int, epochs: int) -> dict[str, Any] | None:
    """Return a persisted planned-handoff receipt for one exact retired launch.

    Fold 0 and schedules other than the retired 20-epoch schedule are never
    affected. A marker from another run is not transferable. Matching markers
    with malformed control fields fail visibly rather than silently training.
    Collection and plan paths are documentary references, never executed/read.
    """
    if type(fold) is not int or fold not in (1, 2, 3, 4) or type(epochs) is not int or epochs != 20:
        return None
    output = Path(output_path).resolve()
    original_run = output.parent
    marker = original_run / "evidence/planned_handoff.json"
    if not marker.is_file():
        return None
    raw = marker.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("artifact_type") != HANDOFF_ARTIFACT:
        return None
    declared_run = value.get("original_run")
    if (not isinstance(declared_run, str) or not Path(declared_run).is_absolute()
            or Path(declared_run).resolve() != original_run):
        return None
    blocked = value.get("blocked_folds")
    if (type(value.get("original_epochs")) is not int or value["original_epochs"] != 20
            or blocked != [1, 2, 3, 4] or any(type(item) is not int for item in blocked)):
        raise ValueError("Matching teacher handoff marker has invalid retired schedule declarations")
    receipt = {
        "artifact_type": "kd_planned_handoff_receipt",
        "state": "planned_handoff",
        "original_run": str(original_run),
        "output": str(output),
        "blocked_fold": fold,
        "requested_epochs": epochs,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_exit_code": HANDOFF_EXIT_CODE,
        "marker": str(marker),
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "training_started": False,
        "explanation": "Old 20-epoch scheduler launch intentionally retired; handoff controller owns the declared replacement plan.",
        "documentary_references": {
            key: value[key] for key in ("collection", "plan", "collection_path", "plan_path")
            if key in value
        },
    }
    _write_receipt(marker.with_name("planned_handoff_receipt.json"), receipt)
    return receipt
