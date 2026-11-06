"""CPU recovery/selection tests, independent of the already tested GPU kernels."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from raw_rebuild import train as trainer


class LifecycleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(3, 10))

    def cuda(self):
        return self


class LifecycleDataset:
    def __init__(self):
        self.groups = [[0], [1]]

    def __len__(self):
        return len(self.groups)


class LifecycleScaler:
    def state_dict(self):
        return {'fixture_scale': 1024.}

    def load_state_dict(self, value):
        if value != self.state_dict():
            raise ValueError('Synthetic scaler state changed')


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, root, stack, scores=(.71, .75, .73)):
        source = root / 'fixture_source.py'
        source.write_text('# frozen source\n')
        config = dict(output=str(root / 'output'), checkpoint_dir=str(root / 'state'),
            seed=42, image_size=518, drop_rate=.2, drop_path=.1, view_chunk_size=4,
            pretrained_path='synthetic-public', image_root='synthetic-originals',
            epochs=len(scores), min_epochs=1, patience=5, min_delta=.0001,
            comparison_bootstrap_repeats=300, reference_predictions='synthetic-reference',
            reference_sha256='a' * 64)
        records = [dict(subject_id='p0'), dict(subject_id='p1')]
        stack.enter_context(patch.object(trainer, 'ROOT', root))
        stack.enter_context(patch.object(trainer, 'source_files', return_value=[source]))
        stack.enter_context(patch.object(trainer, 'verify_inputs', return_value=(records, records, {})))
        stack.enter_context(patch.object(trainer, 'start_run'))
        stack.enter_context(patch.object(trainer.shutil, 'disk_usage',
                                        return_value=SimpleNamespace(free=20_000_000_000)))
        models = stack.enter_context(patch.object(trainer, 'DinoStudyModel',
            side_effect=lambda *a, **kw: LifecycleModel()))
        stack.enter_context(patch.object(trainer, 'StudyDataset',
            side_effect=lambda *a, **kw: LifecycleDataset()))
        stack.enter_context(patch.object(trainer, 'training_weights',
            return_value=(np.array([[0] * 10, [1] * 10]), np.ones(10))))
        stack.enter_context(patch.object(trainer, 'optimizer_groups',
            side_effect=lambda model, _: [dict(params=list(model.parameters()), lr=.01, base_lr=.01)]))
        stack.enter_context(patch.object(torch.amp, 'GradScaler',
            side_effect=lambda *a, **kw: LifecycleScaler()))
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return original_tensor(*args, **kwargs)

        stack.enter_context(patch.object(torch, 'tensor', side_effect=cpu_tensor))

        def train_epoch(model, ema, dataset, optimizer, scaler, pos_weight,
                        configuration, epoch, progress, output):
            # A real CPU optimizer update gives last.pt meaningful Adam state.
            optimizer.zero_grad(set_to_none=True)
            loss = model.head(torch.ones(2, 3)).square().mean()
            loss.backward()
            optimizer.step()
            ema.load_state_dict(model.state_dict())
            progress['attempted_updates'] += 1
            progress['optimizer_updates'] += 1
            return dict(epoch=epoch, loss=float(loss.detach()), studies=2,
                        images=2, **progress)

        def evaluate(model, dataset, configuration, epoch, output):
            (output / f'epoch_{epoch:02d}').mkdir(exist_ok=True)
            return dict(overall=dict(macro_auc=scores[epoch - 1]),
                        patient_equal_macro_auc=scores[epoch - 1])

        def compare(candidate_path, reference_path, *, output_path, **kwargs):
            epoch = int(candidate_path.parent.name.split('_')[-1])
            report = dict(candidate_epoch=epoch, diagnostic_only=True)
            trainer.write_json(output_path, report)
            return report

        training = stack.enter_context(patch.object(trainer, 'train_epoch', side_effect=train_epoch))
        stack.enter_context(patch.object(trainer, 'evaluate', side_effect=evaluate))
        comparisons = stack.enter_context(patch.object(trainer, 'compare_files', side_effect=compare))
        decision = stack.enter_context(patch.object(trainer, 'stage_decision',
            side_effect=lambda report, training_complete: dict(
                state='review', candidate_epoch=report['candidate_epoch'],
                training_complete=training_complete, automatic_submission=False)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return config, source, models, training, comparisons, decision

    def test_complete_local_checkpoint_precedes_failed_network_publication(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, _, _, training, _, decision = self.fixture(root, stack, scores=(.71,))

            def failed_publication(source, destination):
                saved = torch.load(root / 'state/last.pt', map_location='cpu', weights_only=False)
                self.assertEqual(saved['epoch'], 1)
                self.assertEqual(saved['best_epoch'], 1)
                self.assertEqual(saved['progress']['optimizer_updates'], 1)
                self.assertTrue(saved['optimizer']['state'])
                for field in ('model', 'ema', 'scaler', 'rng', 'evidence'):
                    self.assertIn(field, saved)
                raise OSError('synthetic persistent filesystem failure')

            stack.enter_context(patch.object(trainer, 'publish_file', side_effect=failed_publication))
            with self.assertRaisesRegex(OSError, 'persistent filesystem failure'):
                trainer.run(config)
            self.assertEqual(training.call_count, 1)
            decision.assert_not_called()
            self.assertTrue((root / 'state/best_fp16.pt').is_file())
            self.assertFalse((root / 'output/best_fp16.pt').exists())

    def test_resume_rejects_changed_configuration_or_source_before_model_creation(self):
        for changed in ('configuration', 'source'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                config, source, models, training, _, _ = self.fixture(root, stack)
                trainer.run(config, smoke=True)
                before = (root / 'state/last.pt').read_bytes()
                if changed == 'configuration':
                    config = dict(config, seed=config['seed'] + 1)
                else:
                    source.write_text('# altered source\n')
                with self.assertRaisesRegex(ValueError, 'configuration/source/data mismatch'):
                    trainer.run(config, resume=True, smoke=True)
                self.assertEqual(models.call_count, 1)
                self.assertEqual(training.call_count, 1)
                self.assertEqual((root / 'state/last.pt').read_bytes(), before)

    def test_resume_repairs_best_publication_without_repeating_completed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, _, _, training, _, decision = self.fixture(root, stack, scores=(.75, .73))
            with patch.object(trainer, 'publish_file', side_effect=OSError('publication failed')):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    trainer.run(config)
            # Simulate interruption between committing last and saving best.
            (root / 'state/best.pt').unlink()
            (root / 'state/best_fp16.pt').unlink()
            result = trainer.run(config, resume=True)
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(training.call_count, 2)
            self.assertEqual([call.args[7] for call in training.call_args_list], [1, 2])
            best = torch.load(root / 'output/best_fp16.pt', map_location='cpu', weights_only=False)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(result['optimizer_updates'], 2)
            decision.assert_called_once_with(dict(candidate_epoch=1, diagnostic_only=True),
                                             training_complete=True)

    def test_final_stage_decision_uses_only_selected_best_epoch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, _, _, training, comparisons, decision = self.fixture(root, stack)
            result = trainer.run(config)
            self.assertEqual(result['state'], 'completed')
            self.assertEqual(result['best_epoch'], 2)
            self.assertEqual(training.call_count, 3)
            self.assertEqual(comparisons.call_count, 3)
            decision.assert_called_once_with(dict(candidate_epoch=2, diagnostic_only=True),
                                             training_complete=True)
            saved = torch.load(root / 'state/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual((saved['epoch'], saved['best_epoch']), (3, 2))
            best = torch.load(root / 'output/best_fp16.pt', map_location='cpu', weights_only=False)
            self.assertEqual(best['epoch'], 2)
            history = [json.loads(line) for line in (root / 'output/history.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in history], [1, 2, 3])
            self.assertEqual(result['optimizer_updates'], 3)

    def test_stop_preserves_completed_epoch_and_cannot_advance_stage(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, _, _, training, comparisons, decision = self.fixture(root, stack)
            original = comparisons.side_effect

            def request_stop(*args, **kwargs):
                report = original(*args, **kwargs)
                (root / 'output/STOP').touch()
                return report

            comparisons.side_effect = request_stop
            result = trainer.run(config)
            self.assertEqual(result['state'], 'paused')
            self.assertEqual(training.call_count, 1)
            decision.assert_called_once_with(dict(candidate_epoch=1, diagnostic_only=True),
                                             training_complete=False)
            saved = torch.load(root / 'state/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 1)
            self.assertFalse(result['stage_decision']['training_complete'])


if __name__ == '__main__':
    unittest.main()
