"""Restore a best EMA snapshot as a new CPU-exported experimental student.

No training state is restored. Both the snapshot and original full-checkpoint
identities are retained; tensor equality does not imply serialized-file equality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from deploy_kd import MAX_MODEL_BYTES, sha256_file
from student_experiment_evaluation import (
    export_experimental_checkpoint, validate_checkpoint_metadata, write_new_json,
)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def restore_best_export(*, snapshot, receipt, output, historical_weights_sha256=None):
    import torch

    snapshot, receipt = Path(snapshot).resolve(strict=True), Path(receipt).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Choose a new export directory; existing outputs are never overwritten')
    if historical_weights_sha256 is not None and not _digest(historical_weights_sha256):
        raise ValueError('Historical weights SHA256 must be a lowercase hexadecimal digest')
    receipt_sha = sha256_file(receipt)
    backup = json.loads(receipt.read_text())
    entry = backup.get('checkpoints', {}).get('best', {})
    full_sha = entry.get('source_checkpoint_sha256')
    snapshot_sha = sha256_file(snapshot)
    if (backup.get('state') != 'complete' or not _digest(full_sha)
            or entry.get('snapshot_sha256') != snapshot_sha
            or entry.get('snapshot_bytes') != snapshot.stat().st_size
            or entry.get('snapshot_precision') != 'fp32'
            or entry.get('resume_supported') is not False
            or type(entry.get('source_epoch')) is not int or entry['source_epoch'] < 1):
        raise ValueError('Receipt does not identify a complete best FP32 EMA snapshot')
    # The recorded absolute snapshot path is deliberately not required: snapshots
    # can be relocated, while content SHA, size and epoch remain binding.
    checkpoint = torch.load(snapshot, map_location='cpu', weights_only=True)
    if (checkpoint.get('artifact_type') != 'student_ema_snapshot'
            or checkpoint.get('snapshot_precision') != 'fp32'
            or checkpoint.get('resume_supported') is not False
            or checkpoint.get('full_training_state_included') is not False
            or checkpoint.get('epoch') != entry['source_epoch']
            or checkpoint.get('original_full_checkpoint_sha256') != full_sha
            or checkpoint.get('source_checkpoint_sha256') != full_sha):
        raise ValueError('Snapshot epoch, precision or checkpoint lineage differs from receipt')
    validate_checkpoint_metadata(checkpoint)
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        raise ValueError('Snapshot has no EMA tensor state')
    for tensor in state.values():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                or tensor.is_floating_point() and (
                    tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()))):
            raise ValueError('Require finite CPU FP32 snapshot tensors')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-student-', dir=output.parent) as temp:
        staging = Path(temp) / 'export'
        export_experimental_checkpoint(snapshot, staging)
        config_path = staging / 'config.json'
        config = json.loads(config_path.read_text())
        exported = torch.load(staging / config['weights_file'], map_location='cpu', weights_only=True)
        if set(exported) != set(state):
            raise ValueError('Exported tensor names differ from snapshot')
        for name, source in state.items():
            expected = source.half() if source.is_floating_point() else source
            actual = exported[name]
            if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                    or actual.shape != expected.shape or actual.device.type != 'cpu'
                    or actual.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    != expected.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()):
                raise ValueError('Exported tensor bytes differ: ' + name)
        if sha256_file(snapshot) != snapshot_sha or sha256_file(receipt) != receipt_sha:
            raise RuntimeError('Recovery inputs changed while exporting')
        config.update(source_checkpoint_sha256=full_sha,
                      original_full_checkpoint_sha256=full_sha,
                      source_snapshot_sha256=snapshot_sha,
                      source_backup_receipt_sha256=receipt_sha,
                      restored_from_best_ema_snapshot=True)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        inventory_bytes = (staging / config['weights_file']).stat().st_size + config_path.stat().st_size
        if inventory_bytes > MAX_MODEL_BYTES:
            raise ValueError('Restored inference assets exceed the size limit')
        report = {'source_checkpoint_sha256': full_sha, 'source_snapshot_sha256': snapshot_sha,
                  'source_backup_receipt_sha256': receipt_sha, 'source_epoch': entry['source_epoch'],
                  'weights_sha256': sha256_file(staging / config['weights_file']),
                  'config_sha256': sha256_file(config_path), 'tensor_bytes_equal_to_snapshot_fp16': True,
                  'historical_weights_sha256': historical_weights_sha256,
                  'historical_weights_file_equal': None if historical_weights_sha256 is None else
                      historical_weights_sha256 == sha256_file(staging / config['weights_file']),
                  'actual_inference_bytes': inventory_bytes, 'size_pass': True,
                  'resume_supported': False, 'model_forward_executed': False}
        # Replace the exporter report's snapshot-as-source wording with explicit
        # full-checkpoint and snapshot fields before publishing this new artifact.
        (staging / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
        write_new_json(staging / 'restoration_report.json', report)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Output appeared during recovery')
        staging.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='Persistent best_ema_fp32.pt')
    parser.add_argument('--receipt', type=Path, required=True, help='Associated backup_receipt.json')
    parser.add_argument('--output', type=Path, required=True, help='New experimental export directory')
    parser.add_argument('--historical-weights-sha256', help='Optional old exported FP16 file identity')
    print(json.dumps(restore_best_export(**vars(parser.parse_args(argv))), indent=2))


if __name__ == '__main__':
    main()
