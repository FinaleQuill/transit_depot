"""Exercise actual tiny checkpoint files at the best-before-last crash boundary."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train_distill import DistillConfig, repair_student_resume_files
from kd_common import sha256_file


class ResumeCommitTests(unittest.TestCase):
    def payload(self, epoch):
        return {"epoch": epoch, "progress": {"completed_epochs": epoch, "validated_epochs": epoch,
                    "optimizer_updates": 7 * epoch, "ema_updates": 7 * epoch},
                "early_stopping": {"bad_epochs": 0}, "best_auc": .8, "best_epoch": epoch,
                "model_state_dict": {"ema": torch.tensor([float(epoch)])},
                "raw_model_state_dict": {"raw": torch.tensor([float(epoch + 1)])},
                "optimizer_state_dict": {"step": epoch}, "rng_state": {"torch": torch.tensor([epoch])}}

    def config(self, root, **changes):
        return replace(DistillConfig(fold_plan=root/'fold.json', soft_labels=root/'cache',
                       image_root=root/'images', output=root, epochs=3), **changes)

    def test_newer_best_commits_full_state_before_nonterminal_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(self.payload(1), root/'last.pt')
            recovered = self.payload(2)
            torch.save(recovered, root/'best.pt')
            with patch('train_distill.record_checkpoint') as recorded:
                result = repair_student_resume_files(root, recovered, self.config(root),
                    saved_epoch=1, had_last=True, source_version={"fixture": True})
            self.assertIsNone(result)
            last = torch.load(root/'last.pt', weights_only=True)
            self.assertEqual(last['optimizer_state_dict']['step'], 2)
            torch.testing.assert_close(last['raw_model_state_dict']['raw'], recovered['raw_model_state_dict']['raw'])
            torch.testing.assert_close(last['rng_state']['torch'], recovered['rng_state']['torch'])
            self.assertEqual(recorded.call_args.kwargs['epoch'], 2)
            self.assertEqual(recorded.call_args.kwargs['role'], 'last')

    def test_terminal_newer_best_repairs_last_and_completion_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(self.payload(1), root/'last.pt')
            recovered = self.payload(2)
            torch.save(recovered, root/'best.pt')
            with patch('train_distill.record_checkpoint'):
                result = repair_student_resume_files(root, recovered, self.config(root, epochs=2),
                    saved_epoch=1, had_last=True, source_version={})
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(result['state'], 'already_terminal')
            self.assertEqual(status['state'], 'completed')
            self.assertEqual(status['completed_epochs'], 2)
            self.assertEqual(status['last_checkpoint_sha256'], sha256_file(root/'last.pt'))
            self.assertEqual(status['best_checkpoint_sha256'], sha256_file(root/'best.pt'))

    def test_existing_terminal_last_only_repairs_stale_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovered = self.payload(2)
            torch.save(recovered, root/'last.pt')
            torch.save(recovered, root/'best.pt')
            before = sha256_file(root/'last.pt')
            with patch('train_distill.record_checkpoint') as recorded:
                repair_student_resume_files(root, recovered, self.config(root, epochs=2),
                    saved_epoch=2, had_last=True, source_version={})
            recorded.assert_not_called()
            self.assertEqual(sha256_file(root/'last.pt'), before)
            self.assertEqual(json.loads((root/'status.json').read_text())['state'], 'completed')

    def test_first_best_without_last_is_committed_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovered = self.payload(1)
            torch.save(recovered, root/'best.pt')
            with patch('train_distill.record_checkpoint'):
                result = repair_student_resume_files(root, recovered, self.config(root),
                    saved_epoch=1, had_last=False, source_version={})
            self.assertIsNone(result)
            self.assertEqual(torch.load(root/'last.pt', weights_only=True)['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
