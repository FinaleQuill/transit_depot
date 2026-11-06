"""CPU tests for E2 execution-only loader adaptation and real evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reconstruction import train_local as frozen
from reconstruction import train_local_performance as performance
from reconstruction import controller_local
from reconstruction import controller_performance
from training_compliance import read_events
from training_hooks import compliance_run


def sample_profile(workers=3, prefetch=2):
    return {'schema': performance.PROFILE_SCHEMA, 'profile_id': 'cpu_test_profile',
            'loader': {**performance.LOADER_FIXED, 'num_workers': workers,
                       'prefetch_factor': prefetch},
            'applies_to_arms': ['global_control', 'global_local'], 'scope': 'loader_execution_only'}


class DeterministicStudies:
    """Image-keyed augmentation and deliberately out-of-order worker finish."""
    epoch = 10

    def __len__(self):
        return 9

    def __getitem__(self, index):
        time.sleep(.001 * (index % 3))
        rng = random.Random(int.from_bytes(hashlib.sha256(f'20261104/{self.epoch}/{index}'.encode()).digest()[:8], 'big'))
        views = index % 3 + 1
        return {'study_key': f'study_{index}', 'image_ids': [f'{index}_{v}' for v in range(views)],
                'images': torch.tensor([rng.random() for _ in range(views * 12)]).reshape(views, 3, 2, 2),
                'targets': torch.tensor([(index + c) % 2 for c in range(10)])}


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # These tests exercise CPU ordering, not pinned-memory/H2D behavior.
        # Also prevent a host with a visible GPU from creating a CUDA context.
        availability = patch.object(torch.cuda, 'is_available', return_value=False)
        availability.start()
        self.addCleanup(availability.stop)

    def write_profile(self, root, profile=None):
        path = root / 'profile.json'
        path.write_text(json.dumps(sample_profile() if profile is None else profile))
        return path

    def test_profile_rejects_scientific_or_order_changes_and_checks_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.write_profile(root)
            bound = performance.load_profile(path)
            self.assertEqual(bound['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            performance.load_profile(path, bound['sha256'])
            for key, value in [('in_order', False), ('persistent_workers', True),
                               ('pin_memory', False), ('multiprocessing_context', 'fork'),
                               ('timeout', 10), ('num_workers', True), ('prefetch_factor', 0)]:
                profile = sample_profile()
                profile['loader'][key] = value
                with self.assertRaises(ValueError):
                    performance.validate_profile(profile)
            for changed in [{**sample_profile(), 'seed': 1},
                            {**sample_profile(), 'applies_to_arms': ['global_local']}]:
                with self.assertRaises(ValueError):
                    performance.validate_profile(changed)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                performance.load_profile(path, bound['sha256'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                performance.verify_binding(bound)

    def test_real_workers_preserve_order_pixels_targets_and_main_rng(self):
        dataset = DeterministicStudies()
        for training in [True, False]:
            random.seed(119); np.random.seed(120); torch.manual_seed(121)
            py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            original = list(frozen.make_loader(dataset, training=training, workers=2, seed=531))
            adapted, actual = performance.build_loader(dataset, training=training, workers=2,
                                                       seed=531, profile=sample_profile())
            candidate = list(adapted)
            self.assertEqual(actual['num_workers'], 3)
            self.assertEqual(actual['prefetch_factor'], 2)
            self.assertTrue(actual['in_order'])
            self.assertEqual(len(original), len(candidate))
            for before, after in zip(original, candidate):
                self.assertEqual(before['study_key'], after['study_key'])
                self.assertEqual(before['image_ids'], after['image_ids'])
                self.assertTrue(torch.equal(before['images'], after['images']))
                self.assertTrue(torch.equal(before['targets'], after['targets']))
            self.assertEqual(py_rng, random.getstate())
            self.assertEqual(np_rng[0], np.random.get_state()[0])
            np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
            self.assertEqual(np_rng[2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_actual_lifecycle_records_overrides_but_frozen_evidence_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.write_profile(root)
            output = root / 'run'
            split_path = root / 'split.json'
            split_path.write_text(json.dumps({'roles': {'train': {'studies': 9}, 'dev': {'studies': 9}}}))
            config = {'workers': 2, 'experiment_arm': 'global_control', 'epochs': 6,
                      'smoke_studies': 0, 'output': str(output), 'seed': 529,
                      'augmentation_epoch_offset': 8, 'split_plan': str(split_path)}
            science = {**config, 'source_sha256': {'frozen.py': 'registered'}}
            expected_config, expected_science = deepcopy(config), deepcopy(science)
            output.mkdir()
            (output / 'configuration.json').write_text(json.dumps(science))
            science_file_sha = frozen.base.digest(output / 'configuration.json')
            original_sources = frozen.source_files()
            original_loader, original_start = frozen.make_loader, frozen.start_run
            expected_payload = {'configuration': deepcopy(config), 'evidence': deepcopy(science),
                                'epoch': 0, 'best_epoch': 0, 'best': .8,
                                'progress': {'attempted_updates': 0}, 'model': {'w': torch.ones(2)}}

            @compliance_run
            def invocation(resume=False):
                frozen.start_run(output, trainer='reconstruction.train_local', configuration=science,
                                 source_files=original_sources, resume=output / 'last.pt' if resume else None)
                loader = frozen.make_loader(DeterministicStudies(), training=True, workers=2, seed=531)
                self.assertEqual(loader.num_workers, 3)
                if not resume:
                    frozen.base.save_checkpoint(output / 'best.pt', {'ema': {'w': torch.ones(2)}}, 0, 'best_inner_ema')
                    frozen.base.save_checkpoint(output / 'last.pt', expected_payload, 0, 'last_full_state')
                return {'state': 'paused'}

            with performance.profile_adapter(config, path) as binding:
                invocation()
                first_checkpoint_sha = frozen.base.digest(output / 'last.pt')
                invocation(resume=True)
                self.assertEqual(first_checkpoint_sha, frozen.base.digest(output / 'last.pt'))
            self.assertIs(frozen.make_loader, original_loader)
            self.assertIs(frozen.start_run, original_start)
            self.assertEqual(frozen.source_files(), original_sources)
            self.assertEqual(config, expected_config)
            self.assertEqual(science, expected_science)
            self.assertEqual(frozen.base.digest(output / 'configuration.json'), science_file_sha)
            loaded = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(set(loaded), set(expected_payload))
            self.assertEqual(loaded['configuration'], expected_config)
            self.assertEqual(loaded['evidence'], expected_science)
            (output / 'baseline.json').write_text(json.dumps({'epoch': 0, 'macro_auc': .8}))
            frozen.validate_resume_payload(loaded, config, science, output)
            events = read_events(output / 'compliance/events.jsonl')
            starts = [e for e in events if e['event_type'] == 'training_started']
            self.assertEqual(len(starts), 2)
            self.assertIsNone(starts[0]['resume_from'])
            self.assertEqual(starts[1]['resume_from'], str(output / 'last.pt'))
            for start in starts:
                session = output / 'compliance' / start['session_evidence']
                actual = json.loads((session / 'configuration.json').read_text())
                self.assertEqual(actual.pop('execution_profile'), binding)
                self.assertEqual(actual, expected_science)
                self.assertEqual(start['configuration_sha256'], frozen.base.digest(session / 'configuration.json'))
                sources = json.loads((session / 'source_manifest.json').read_text())
                indexed = {x['original_path']: x for x in sources}
                for source, digest in [(str(path), binding['sha256']),
                                       (binding['adapter_source'], binding['adapter_source_sha256'])]:
                    self.assertEqual(indexed[source]['sha256'], digest)
                    self.assertEqual(frozen.base.digest(output / 'compliance' / indexed[source]['archived_path']), digest)
            created = [e for e in events if e['event_type'] == 'data_loader_created']
            self.assertEqual(len(created), 2)
            for event in created:
                self.assertEqual(event['execution_profile_sha256'], binding['sha256'])
                self.assertEqual(event['registered_workers'], 2)
                self.assertEqual(event['loader']['num_workers'], 3)
                self.assertEqual(event['loader']['prefetch_factor'], 2)
                self.assertEqual(event['seed'], 531)
                self.assertEqual(event['dataset_epoch'], 10)
            self.assertEqual(events[-1]['state'], 'paused')
            (output / 'status.json').write_text(json.dumps({'state': 'paused'}))
            frozen.verify_resume_lifecycle(output, config)
            self.assertEqual(controller_local.closed_sessions(output), events)
            performance_plan = {
                'execution_profile': {key: binding[key] for key in ['path', 'sha256', 'profile']},
                'source_sha256': {binding['adapter_source']: binding['adapter_source_sha256']},
                'transition_arms': {'global_control': {'session_ids': []}}}
            job = {'name': 'global_control', 'output': str(output), 'config': config}
            verified = controller_performance.verify_execution_sessions(job, performance_plan)
            self.assertEqual(len(verified['performance_sessions']), 2)

    def test_binding_restored_on_error_and_uncovered_loader_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_profile(Path(folder))
            config = {'workers': 2, 'experiment_arm': 'global_local'}
            original = frozen.make_loader, frozen.start_run
            with self.assertRaisesRegex(RuntimeError, 'active lifecycle'):
                with performance.profile_adapter(config, path):
                    frozen.make_loader(DeterministicStudies(), training=True, workers=2, seed=19)
            self.assertEqual((frozen.make_loader, frozen.start_run), original)
            with self.assertRaisesRegex(RuntimeError, 'Nested'):
                with performance.profile_adapter(config, path):
                    with performance.profile_adapter(config, path):
                        pass
            self.assertEqual((frozen.make_loader, frozen.start_run), original)

    def test_preflight_does_not_run_model_or_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.write_profile(root)
            config_path = root / 'science.json'
            config = {'workers': 2, 'experiment_arm': 'global_control', 'output': str(root / 'unused')}
            config_path.write_text(json.dumps(config))
            argv = ['entrypoint', '--config', str(config_path), '--execution-profile', str(path), '--resume']
            with patch.object(sys, 'argv', argv), patch.object(frozen, 'validate_config') as validate, \
                    patch.object(frozen, 'verify_parent') as parent, patch.object(frozen, 'run') as run:
                performance.main()
            validate.assert_called_once_with(config)
            parent.assert_called_once_with(config)
            run.assert_not_called()
            self.assertFalse((root / 'unused').exists())


if __name__ == '__main__':
    unittest.main()
