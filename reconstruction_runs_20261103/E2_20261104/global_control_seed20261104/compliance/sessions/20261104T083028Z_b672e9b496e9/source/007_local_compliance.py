"""Narrow E2 lifecycle adapter for an explicitly untrained epoch-zero baseline.

The original recorder rejects epoch zero and remains unchanged. Positive
training epochs delegate to it. Baseline saves use the same hash-chained event
format, while declaring that no training epoch has been completed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from training_compliance import ComplianceRecorder
from training_hooks import _invocation


class LocalComplianceRecorder(ComplianceRecorder):
    def checkpoint_saved(self, path, digest, *, epoch, role):
        if epoch != 0 or type(epoch) is not int:
            return super().checkpoint_saved(path, digest, epoch=epoch, role=role)
        if self.trainer != 'reconstruction.train_local':
            raise ValueError('Epoch-zero evidence is restricted to the local continuation trainer')
        expected = {'best_inner_ema': 'best.pt', 'last_full_state': 'last.pt'}
        if role not in expected or not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Baseline checkpoint role and completed-save SHA256 are required')
        path = Path(path).resolve(strict=True)
        if path.parent != self.run_dir or path.name != expected[role] or not path.is_file():
            raise ValueError('Baseline checkpoint must be the declared best/last file in the active run')
        rows = self._read_for_append()
        current = [row for row in rows if row.get('session_id') == self.session_id]
        if any(row['event_type'] == 'epoch_completed' or
               (row['event_type'] == 'checkpoint_saved' and row.get('role') == role) for row in current):
            raise ValueError('Epoch-zero baseline may be saved once per role before any training epoch')
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError('Baseline checkpoint must be a nonempty completed file')
        return self._record('checkpoint_saved', {
            'epoch': 0, 'role': role, 'stage': 'pre_training_baseline', 'trained_epochs_completed': 0,
            'checkpoint_path': str(path), 'checkpoint_sha256': digest,
            'checkpoint_metadata': {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
            'saved_at_utc': datetime.fromtimestamp(stat.st_mtime_ns / 1e9, timezone.utc).isoformat(),
            'digest_source': 'completed_save; reused_without_rereading_weights',
            'retention': 'metadata_only; rolling checkpoint binary may later be replaced'})


def start_run(run_dir, *, trainer, configuration, source_files=(), resume=None):
    """Install only this invocation's recorder; preserve the shared end adapter."""
    if trainer != 'reconstruction.train_local':
        raise ValueError('Local lifecycle adapter only accepts reconstruction.train_local')
    slot = _invocation.get()
    if slot is None or slot['recorder'] is not None:
        raise RuntimeError('Activate exactly once inside the shared compliance_run invocation')
    slot['recorder'] = LocalComplianceRecorder(run_dir, trainer=trainer,
        configuration=configuration, source_files=source_files, resume=resume)
