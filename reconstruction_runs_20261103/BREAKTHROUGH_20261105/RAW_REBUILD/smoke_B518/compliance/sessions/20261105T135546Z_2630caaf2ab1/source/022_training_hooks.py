"""Small, dependency-free lifecycle adapter for explicit training invocations.

The adapter never starts training and does not import a model framework. Trainers
activate it only after their existing output/path guards. Outside an activated
training invocation checkpoint saves remain ordinary saves, not historical
training events. A killed process may have no end event; no end is fabricated.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import signal
import sys
import threading
from typing import Any, Callable, Iterable, Mapping

from training_compliance import ComplianceRecorder


_invocation: ContextVar[dict[str, Any] | None] = ContextVar("training_compliance_invocation", default=None)


def compliance_run(function: Callable) -> Callable:
    """Finalize a recorder on normal return or a catchable exception.

Existing trainer signal handlers keep their graceful epoch-boundary behavior.
Before those handlers are installed, a default SIGTERM becomes a catchable
interrupt. SIGKILL, machine loss and uncatchable exits cannot emit an end record.
"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        slot: dict[str, Any] = {"recorder": None}
        token = _invocation.set(slot)
        prior_term = None
        if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
            prior_term = signal.getsignal(signal.SIGTERM)
            def interrupted(signum, _frame):
                raise KeyboardInterrupt(f"Training interrupted by signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            recorder = slot["recorder"]
            if recorder is not None:
                state = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                try:
                    recorder.finish(state, {"error": f"{type(error).__name__}: {error}"})
                except BaseException as recording_error:
                    # Retain the training exception as the primary failure. A
                    # recording failure is still visible and never means pass.
                    note = f"Compliance end record failed: {type(recording_error).__name__}: {recording_error}"
                    if hasattr(error, "add_note"):
                        error.add_note(note)
                    print(note, file=sys.stderr, flush=True)
            raise
        else:
            recorder = slot["recorder"]
            if recorder is not None:
                details = dict(result) if isinstance(result, Mapping) else {"return_type": type(result).__name__}
                recorder.finish(str(details.get("state", "completed")), details)
            return result
        finally:
            if prior_term is not None:
                signal.signal(signal.SIGTERM, prior_term)
            _invocation.reset(token)
    return wrapped


def start_run(run_dir: str | Path, *, trainer: str, configuration: Mapping[str, Any],
              source_files: Iterable[str | Path] = (), resume: str | Path | None = None) -> None:
    """Activate exactly once, after a trainer's original output safety checks."""
    slot = _invocation.get()
    if slot is None:
        raise RuntimeError("start_run must be called inside @compliance_run")
    if slot["recorder"] is not None:
        raise RuntimeError("A training invocation already has an active compliance recorder")
    slot["recorder"] = ComplianceRecorder(run_dir, trainer=trainer,
        configuration=configuration, source_files=source_files, resume=resume)


def epoch_completed(epoch: int, metrics: Mapping[str, Any]) -> None:
    slot = _invocation.get()
    if slot is not None and slot["recorder"] is not None:
        slot["recorder"].epoch_completed(epoch, metrics)


def checkpoint_saved(path: str | Path, digest: str, *, epoch: int, role: str) -> None:
    """Notify only an activated invocation, after durable save and exact hash."""
    slot = _invocation.get()
    if slot is not None and slot["recorder"] is not None:
        slot["recorder"].checkpoint_saved(path, digest, epoch=epoch, role=role)
