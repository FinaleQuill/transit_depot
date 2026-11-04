"""Exercise E2 transport, baseline selection and continuation on CPU."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reconstruction import train_local as trainer
from reconstruction.local_model import LocalStudyModel
from reconstruction.model import StudyModel
from reconstruction.test_model import DummyBackbone
from reconstruction.local_compliance import LocalComplianceRecorder, start_run
from training_compliance import read_events
from training_hooks import compliance_run


class SyntheticLocalStudies:
    def __init__(self):
        self.batches, self.groups, self.records = [], [], []
        offset = 0
        maps = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                             [[.6, 0., 0.], [0., .6, 0.]],
                             [[.6, 0., .4], [0., .6, 0.]],
                             [[.6, 0., 0.], [0., .6, .4]],
                             [[.6, 0., .4], [0., .6, .4]]])
        for index, views in enumerate([1, 2, 4]):
            self.groups.append(list(range(offset, offset + views)))
            self.records.extend({'study_key': f'p{index}/s{index}'} for _ in range(views))
            offset += views
            self.batches.append({'images': torch.randn(views, 5, 3, 6, 6),
                'region_affines': maps[None].repeat(views, 1, 1, 1),
                'targets': ((torch.arange(10) + index) % 3 == 0).float(),
                'study_key': f'p{index}/s{index}', 'subject_id': f'p{index}',
                'views': ['AP'] * views, 'image_ids': [f'{index}_{j}' for j in range(views)]})

    def __len__(self):
        return len(self.groups)

    def set_epoch(self, epoch):
        self.epoch = epoch


class CPUTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.dataset = SyntheticLocalStudies()
        self.model = LocalStudyModel.from_backbone(DummyBackbone(), attention_dim=12, view_chunk_size=3)
        with torch.no_grad():
            self.model.region_residual_weight.normal_(std=.01)
        self.config = {'experiment_arm': 'global_local', 'workers': 0, 'seed': 31,
            'effective_studies': 2, 'pack_images': 8, 'eval_pack_images': 1,
            'epochs': 6, 'augmentation_epoch_offset': 8, 'clip_grad': 1.,
            'ema_decay': .999, 'smoke_studies': 3, 'bootstrap_repeats': 0}

    def cpu_transport(self):
        stack = ExitStack()
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and str(args[0]) == 'cuda':
                args = ('cpu',) + args[1:]
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
        stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()))
        stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
        stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
        stack.enter_context(patch.object(trainer, 'make_loader', return_value=self.dataset.batches))
        return stack

    def test_training_and_evaluation_pass_geometry_and_retain_complete_studies(self):
        ema = deepcopy(self.model).eval().requires_grad_(False)
        optimizer = torch.optim.SGD([{'params': self.model.parameters(), 'lr': .01, 'base_lr': .01}])
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        progress = {'attempted_updates': 0, 'optimizer_updates': 0, 'skipped_updates': 0}
        before = self.model.region_residual_weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder, self.cpu_transport():
            with patch.object(self.model, 'forward_studies', wraps=self.model.forward_studies) as forward:
                metrics = trainer.train_epoch(self.model, ema, self.dataset, optimizer, scaler,
                    torch.ones(10), self.config, 1, progress, Path(folder))
                self.assertTrue(forward.called)
                for call in forward.call_args_list:
                    self.assertIn('region_affines', call.kwargs)
                    self.assertEqual(len(call.kwargs['region_affines']), len(call.args[0]))
                    self.assertTrue(call.kwargs['return_components'])
            self.assertEqual((metrics['studies'], metrics['images'], metrics['regions']), (3, 7, 35))
            self.assertEqual(progress['optimizer_updates'], 2)
            self.assertEqual(self.dataset.epoch, 9)
            self.assertFalse(torch.equal(before, self.model.region_residual_weight))
            with patch.object(ema, 'forward_studies', wraps=ema.forward_studies) as forward:
                report = trainer.evaluate(ema, self.dataset, self.config, 1, Path(folder))
                self.assertEqual(len(forward.call_args_list), 3)
                for call in forward.call_args_list:
                    self.assertIn('region_affines', call.kwargs)
                    self.assertFalse(call.kwargs['return_components'])
            self.assertEqual(report['scope'], 'train_diagnostic')
            self.assertEqual((report['num_studies'], report['num_images']), (3, 7))
            self.assertIsNone(trainer.selection_score(report, self.config))
            with np.load(Path(folder) / 'epoch_01/predictions.npz', allow_pickle=False) as stored:
                actual = stored['probabilities']
            expected = np.stack([ema(b['images'], region_affines=b['region_affines']).sigmoid().detach().numpy()
                                 for b in self.dataset.batches])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_global_control_uses_ordinary_global_path_without_geometry_keyword(self):
        model = StudyModel.from_backbone(DummyBackbone()).eval()
        batches = [{**b, 'images': b['images'][:, :1]} for b in self.dataset.batches]
        actual = trainer.forward_pack(model, batches, {'experiment_arm': 'global_control'}, device='cpu')
        expected = [model(b['images']) for b in batches]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_highest_view_smoke_preserves_complete_group_and_selection_is_deterministic(self):
        trainer.select_smoke_groups(self.dataset, 2, True)
        self.assertEqual(list(map(len, self.dataset.groups)), [4, 2])
        self.assertEqual(self.dataset.groups[0], [3, 4, 5, 6])

    def test_finite_smoke_score_never_enters_selection(self):
        report = {'overall': {'macro_auc': .99}}
        self.assertIsNone(trainer.selection_score(report, {'smoke_studies': 1}))
        self.assertEqual(trainer.selection_score(report, {'smoke_studies': 0}), .99)


class EvidenceTests(unittest.TestCase):
    def test_actual_epoch_zero_checkpoint_callback_and_closed_hash_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            @compliance_run
            def invocation():
                start_run(output, trainer='reconstruction.train_local', configuration={'synthetic_test': True},
                          source_files=[Path(__file__)])
                trainer.base.save_checkpoint(output/'best.pt', {'ema': {'weight': torch.ones(2)}}, 0, 'best_inner_ema')
                trainer.base.save_checkpoint(output/'last.pt', {'model': {'weight': torch.ones(2)}}, 0, 'last_full_state')
                return {'state': 'paused'}
            invocation()
            events = read_events(output/'compliance/events.jsonl')
            self.assertEqual([e['event_type'] for e in events],
                             ['training_started', 'checkpoint_saved', 'checkpoint_saved', 'training_ended'])
            for event in events[1:3]:
                self.assertEqual(event['epoch'], 0)
                self.assertEqual(event['stage'], 'pre_training_baseline')
                self.assertEqual(event['trained_epochs_completed'], 0)
                self.assertEqual(event['checkpoint_sha256'], hashlib.sha256(Path(event['checkpoint_path']).read_bytes()).hexdigest())
            self.assertEqual(events[-1]['state'], 'paused')
            status = json.loads((output/'compliance/status.json').read_text())
            self.assertEqual(status['completed_epochs_this_session'], [])

    def test_baseline_recorder_rejects_duplicate_or_foreign_roles_and_preserves_positive_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            recorder = LocalComplianceRecorder(output, trainer='reconstruction.train_local', configuration={})
            try:
                path = output/'best.pt'; path.write_bytes(b'synthetic completed save')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    recorder.checkpoint_saved(path, digest, epoch=0, role='unknown')
                recorder.checkpoint_saved(path, digest, epoch=0, role='best_inner_ema')
                with self.assertRaises(ValueError):
                    recorder.checkpoint_saved(path, digest, epoch=0, role='best_inner_ema')
                with self.assertRaises(ValueError):
                    recorder.epoch_completed(0, {})
                event = recorder.checkpoint_saved(path, digest, epoch=1, role='best_inner_ema')
                self.assertNotIn('stage', event)
                self.assertEqual(event['epoch'], 1)
                recorder.finish('paused')
            finally:
                recorder.close()

    def test_epoch_zero_probability_guard_fails_before_optimization_if_local_output_drifted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control, output = root/'control', root/'local'
            (control/'epoch_00').mkdir(parents=True)
            output.mkdir()
            config = {'smoke_studies': 0, 'experiment_arm': 'global_local', 'use_local': True,
                      'output': str(output), 'parent_checkpoint': str(root/'parent/best.pt'),
                      'parent_best_epoch': 8, 'reference_epoch_zero': str(control/'epoch_00'),
                      'epoch_zero_max_abs_tolerance': .000002}
            reference_config = {**config, 'experiment_arm': 'global_control', 'use_local': False,
                                'output': str(control), 'reference_epoch_zero': None}
            (control/'configuration.json').write_text(json.dumps(reference_config))
            item = {'probabilities': np.full((2, 10), .5), 'targets': np.zeros((2, 10)),
                    'study_keys': np.asarray(['p/s1', 'q/s2']), 'patient_ids': np.asarray(['p', 'q']),
                    'image_counts': np.asarray([1, 1]), 'views': [['AP'], ['AP']],
                    'report': {'macro_auc': .5, 'metadata': {'epoch': 0}}, 'binding': {}}
            changed = {**item, 'probabilities': item['probabilities'] + .0001}
            with patch.object(trainer, 'RUN_ROOT', root), patch.object(trainer, 'load_bound_predictions',
                    side_effect=[changed, item, item]):
                with self.assertRaisesRegex(ValueError, 'beyond tolerance'):
                    trainer.verify_epoch_zero(config, output)
            report = json.loads((output/'epoch_zero_equivalence.json').read_text())
            self.assertFalse(report['passed'])
            self.assertGreater(report['max_abs_probability_difference'], report['tolerance'])

    def test_parent_payload_rejects_mismatched_epoch_evidence_and_extra_heads(self):
        binding = {'verified_parent': {'best_epoch': 8, 'best_macro_auc': .8},
                   'configuration': {'seed': 1}, 'evidence': {'split': 'clean'}}
        payload = {'ema': {'backbone.fc.weight': torch.ones(10, 3)}, 'epoch': 8,
                   'macro_auc': .8, 'configuration': {'seed': 1}, 'evidence': {'split': 'clean'}}
        self.assertEqual(set(trainer.parent_state(payload, binding)), {'backbone.fc.weight'})
        for changed in [{**payload, 'epoch': 7}, {**payload, 'evidence': {}},
                        {**payload, 'ema': {'view_query': torch.zeros(10, 3)}}]:
            with self.assertRaises(ValueError):
                trainer.parent_state(changed, binding)

    def test_resume_retains_epoch_zero_if_all_trained_epochs_are_worse(self):
        config, evidence = {'epochs': 6, 'smoke_studies': 0}, {'source': 'abc'}
        prior = {'configuration': config, 'evidence': evidence, 'epoch': 2,
                 'progress': {'attempted_updates': 4}, 'best_epoch': 0, 'best': .8}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'baseline.json').write_text(json.dumps({'epoch': 0, 'macro_auc': .8}))
            rows = [{'epoch': 1, 'macro_auc': .79, 'attempted_updates': 2},
                    {'epoch': 2, 'macro_auc': .78, 'attempted_updates': 4}]
            (output/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            trainer.validate_resume_payload(prior, config, evidence, output)
            with self.assertRaises(ValueError):
                trainer.validate_resume_payload({**prior, 'best_epoch': 1, 'best': .79}, config, evidence, output)
            with self.assertRaises(ValueError):
                trainer.validate_resume_payload(prior, config, {'source': 'changed'}, output)
            with self.assertRaises(ValueError):
                trainer.validate_resume_payload({**prior, 'epoch': 3}, config, evidence, output)

    def test_epoch_zero_checkpoint_can_resume_without_any_completed_training(self):
        config, evidence = {'epochs': 6, 'smoke_studies': 0}, {'source': 'abc'}
        prior = {'configuration': config, 'evidence': evidence, 'epoch': 0,
                 'progress': {'attempted_updates': 0}, 'best_epoch': 0, 'best': .8}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'baseline.json').write_text(json.dumps({'epoch': 0, 'macro_auc': .8}))
            trainer.validate_resume_payload(prior, config, evidence, output)

    def test_loader_uses_bounded_prefetch_and_identical_seed(self):
        with patch.object(trainer, 'DataLoader') as loader:
            trainer.make_loader([], training=True, workers=2, seed=97)
            kwargs = loader.call_args.kwargs
            self.assertEqual(kwargs['prefetch_factor'], 1)
            self.assertEqual(kwargs['generator'].initial_seed(), 97)
            self.assertEqual(kwargs['multiprocessing_context'], 'spawn')


if __name__ == '__main__':
    unittest.main()
