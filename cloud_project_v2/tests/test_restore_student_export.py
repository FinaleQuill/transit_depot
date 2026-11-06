"""Synthetic CPU recovery checks; no real student weights or GPU are used."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import restore_student_export as recovery


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot, self.receipt, self.output = [self.root / x for x in ('best.pt', 'receipt.json', 'export')]
        self.full_sha = 'a' * 64
        self.checkpoint = {'artifact_type': 'student_ema_snapshot', 'snapshot_precision': 'fp32',
            'resume_supported': False, 'full_training_state_included': False, 'epoch': 10,
            'original_full_checkpoint_sha256': self.full_sha, 'source_checkpoint_sha256': self.full_sha,
            'model_state_dict': {'weight': torch.tensor([1.125, -0.0, 0.123456]),
                                 'counter': torch.tensor(4, dtype=torch.int64)}}
        self.publish()

    def publish(self):
        torch.save(self.checkpoint, self.snapshot)
        self.backup = {'state': 'complete', 'checkpoints': {'best': {
            'snapshot_sha256': recovery.sha256_file(self.snapshot),
            'snapshot_bytes': self.snapshot.stat().st_size, 'snapshot_precision': 'fp32',
            'resume_supported': False, 'source_epoch': 10, 'source_checkpoint_sha256': self.full_sha}}}
        self.receipt.write_text(json.dumps(self.backup))

    def fake_export(self, snapshot, output):
        output.mkdir()
        state = torch.load(snapshot, map_location='cpu', weights_only=True)['model_state_dict']
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in state.items()}, output / 'student_fp16.pt')
        (output / 'config.json').write_text(json.dumps({'weights_file': 'student_fp16.pt',
            'source_checkpoint_sha256': recovery.sha256_file(snapshot),
            'weights_sha256': recovery.sha256_file(output / 'student_fp16.pt')}))
        (output / 'export_report.json').write_text('{}')

    def call(self, exporter=None, **kwargs):
        with patch.object(recovery, 'validate_checkpoint_metadata') as validate, patch.object(
                recovery, 'export_experimental_checkpoint', side_effect=exporter or self.fake_export):
            result = recovery.restore_best_export(snapshot=self.snapshot, receipt=self.receipt,
                                                  output=self.output, **kwargs)
            validate.assert_called_once()
            return result

    def test_preserves_two_sha_lineage_and_distinguishes_file_equality(self):
        result = self.call(historical_weights_sha256='b' * 64)
        config = json.loads((self.output / 'config.json').read_text())
        self.assertEqual(config['source_checkpoint_sha256'], self.full_sha)
        self.assertEqual(config['original_full_checkpoint_sha256'], self.full_sha)
        self.assertEqual(config['source_snapshot_sha256'], recovery.sha256_file(self.snapshot))
        self.assertTrue(result['tensor_bytes_equal_to_snapshot_fp16'])
        self.assertFalse(result['historical_weights_file_equal'])
        self.assertEqual(result['config_sha256'], recovery.sha256_file(self.output / 'config.json'))
        self.assertEqual(json.loads((self.output / 'export_report.json').read_text()), result)

    def test_rejects_wrong_receipt_hash_before_loading(self):
        self.backup['checkpoints']['best']['snapshot_sha256'] = 'c' * 64
        self.receipt.write_text(json.dumps(self.backup))
        with patch.object(torch, 'load') as load, self.assertRaisesRegex(ValueError, 'Receipt'):
            self.call()
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_epoch_or_lineage_mismatch(self):
        for field, value in [('epoch', 9), ('original_full_checkpoint_sha256', 'd' * 64)]:
            with self.subTest(field=field):
                previous = self.checkpoint[field]
                self.checkpoint[field] = value
                self.publish()
                with self.assertRaisesRegex(ValueError, 'epoch, precision or checkpoint lineage'):
                    self.call()
                self.assertFalse(self.output.exists())
                self.checkpoint[field] = previous

    def test_rejects_corrupt_export_tensor_and_does_not_publish(self):
        def corrupt(snapshot, output):
            self.fake_export(snapshot, output)
            path = output / 'student_fp16.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            # Numerically equal signed zeros still have different bytes.
            state['weight'][1] = 0.0
            torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'tensor bytes differ'):
            self.call(corrupt)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('original')
        with self.assertRaises(FileExistsError):
            self.call()
        self.assertEqual(marker.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
