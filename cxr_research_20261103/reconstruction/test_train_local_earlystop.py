"""CPU-only real lifecycle tests for synchronous E2 stopping-policy hooks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconstruction import early_stop_policy as policy
from reconstruction import train_local as frozen
from reconstruction import train_local_earlystop as adapter
from reconstruction import train_local_performance as performance
from reconstruction import controller_performance
from reconstruction.test_train_local_performance import sample_profile, DeterministicStudies
from training_compliance import read_events
from training_hooks import compliance_run


class AdapterTests(unittest.TestCase):
    def setUp(self):
        availability = patch.object(frozen.torch.cuda, 'is_available', return_value=False)
        availability.start(); self.addCleanup(availability.stop)
        frozen.torch.set_num_threads(1)

    def setup_run(self, root):
        output = root / 'run'; output.mkdir()
        policy_path = root / 'policy.json'; policy_path.write_text(json.dumps(policy.POLICY))
        profile = root / 'profile.json'; profile.write_text(json.dumps(sample_profile(workers=1)))
        config = {'output': str(output), 'epochs': 6, 'min_delta': .0001,
                  'smoke_studies': 0, 'experiment_arm': 'global_control', 'workers': 2}
        (output / 'baseline.json').write_text(json.dumps({'epoch': 0, 'macro_auc': .8}))
        (output / 'configuration.json').write_text(json.dumps(config))
        return output, config, policy.load_policy(policy_path), profile

    def invocation(self, output, config, scores, *, external_stop=False):
        @compliance_run
        def run():
            frozen.start_run(output, trainer='reconstruction.train_local', configuration=config,
                             source_files=frozen.source_files())
            for epoch, score in enumerate(scores, 1):
                metrics = {'epoch': epoch, 'macro_auc': score}
                if epoch == 3 and external_stop:
                    (output / 'STOP').write_text('external user stop\n')
                elif epoch < 6:
                    self.assertFalse((output / 'STOP').exists())
                # Model-free fixture uses the actual durable-save hook and ledger.
                frozen.base.save_checkpoint(output / 'last.pt', {'fixture_epoch': epoch}, epoch, 'last_full_state')
                with (output / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics) + '\n')
                frozen.epoch_completed(epoch, metrics)
                if (output / 'STOP').exists():
                    return {'state': 'paused'}
            return {'state': 'completed'}
        return run()

    def test_real_lifecycle_synchronous_stop_and_bound_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, profile = self.setup_run(Path(folder))
            original = frozen.start_run, frozen.epoch_completed
            evidence_before = (output / 'configuration.json').read_bytes()
            with performance.profile_adapter(config, profile) as execution:
                with adapter.stopping_adapter(config, binding):
                    result = self.invocation(output, config, [.79, .78, .77, .95])
            self.assertEqual(result['state'], 'paused')
            self.assertEqual((frozen.start_run, frozen.epoch_completed), original)
            self.assertEqual((output / 'configuration.json').read_bytes(), evidence_before)
            events = read_events(output / 'compliance/events.jsonl')
            starts = [e for e in events if e['event_type'] == 'training_started']
            self.assertEqual(len(starts), 1)
            session = output / 'compliance' / starts[0]['session_evidence']
            saved_config = json.loads((session / 'configuration.json').read_text())
            self.assertEqual(saved_config.pop('execution_profile'), execution)
            self.assertEqual(saved_config, config)
            indexed = {s['original_path']: s for s in json.loads((session / 'source_manifest.json').read_text())}
            for source in [*adapter.source_files(), Path(binding['path'])]:
                self.assertEqual(indexed[str(source)]['sha256'], policy.sha(source))
                self.assertEqual(policy.sha(output / 'compliance' / indexed[str(source)]['archived_path']), policy.sha(source))
            checks = [e for e in events if e['event_type'] == 'early_stopping_check']
            self.assertEqual([e['decision']['bad_epochs'] for e in checks], [1, 2, 3])
            self.assertEqual(events[-1]['state'], 'paused')
            completed = [e['epoch'] for e in events if e['event_type'] == 'epoch_completed']
            self.assertEqual(completed, [1, 2, 3])
            for check in checks:
                index = events.index(check)
                self.assertEqual(events[index - 1]['event_type'], 'epoch_completed')
                self.assertEqual(events[index - 2]['event_type'], 'checkpoint_saved')
            self.assertEqual(json.loads((output / 'STOP').read_text())['epoch'], 3)
            self.assertTrue(json.loads((output / 'early_stopping.json').read_text())['stop_created'])

    def test_external_stop_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, _ = self.setup_run(Path(folder))
            with adapter.stopping_adapter(config, binding):
                self.invocation(output, config, [.79, .78, .77], external_stop=True)
            self.assertEqual((output / 'STOP').read_text(), 'external user stop\n')
            record = json.loads((output / 'early_stopping.json').read_text())
            self.assertFalse(record['stop_created'])
            self.assertTrue(record['decision']['stop'])

    def test_maximum_epoch_finishes_without_pause_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, _ = self.setup_run(Path(folder))
            with adapter.stopping_adapter(config, binding):
                result = self.invocation(output, config, [.81, .82, .83, .84, .85, .86])
            self.assertEqual(result['state'], 'completed')
            self.assertFalse((output / 'STOP').exists())
            record = json.loads((output / 'early_stopping.json').read_text())
            self.assertEqual(record['decision']['stop_reason'], 'max_epochs_reached')

    def test_resume_after_exhausted_budget_is_rejected_before_new_session(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, _ = self.setup_run(Path(folder))
            with adapter.stopping_adapter(config, binding):
                self.invocation(output, config, [.79, .78, .77])
                before = (output / 'compliance/events.jsonl').read_bytes()
                @compliance_run
                def resume():
                    frozen.start_run(output, trainer='reconstruction.train_local', configuration=config,
                                     resume=output / 'last.pt')
                with self.assertRaisesRegex(ValueError, 'budget already completed'):
                    resume()
                self.assertEqual(before, (output / 'compliance/events.jsonl').read_bytes())

    @unittest.skipUnless(performance.IN_ORDER_ARGUMENT_SUPPORTED, 'Requires the official PyTorch DataLoader API')
    def test_profile_verifier_accepts_real_two_session_resume_and_policy_events(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, config, binding, profile = self.setup_run(root)
            split = root / 'split.json'
            split.write_text(json.dumps({'roles': {'train': {'studies': 9}, 'dev': {'studies': 9}}}))
            config.update(seed=530, augmentation_epoch_offset=8, split_plan=str(split))
            (output / 'configuration.json').write_text(json.dumps(config))

            @compliance_run
            def run(epochs, resume=False):
                frozen.start_run(output, trainer='reconstruction.train_local', configuration=config,
                    source_files=frozen.source_files(), resume=output / 'last.pt' if resume else None)
                for epoch in epochs:
                    training = DeterministicStudies(); training.epoch = 8 + epoch
                    evaluation = DeterministicStudies(); evaluation.epoch = 0
                    frozen.make_loader(training, training=True, workers=2, seed=530 + epoch)
                    frozen.make_loader(evaluation, training=False, workers=2, seed=530)
                    metrics = {'epoch': epoch, 'macro_auc': .79 - .01 * epoch}
                    frozen.base.save_checkpoint(output / 'last.pt', {'fixture_epoch': epoch}, epoch, 'last_full_state')
                    with (output / 'history.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics) + '\n')
                    frozen.epoch_completed(epoch, metrics)
                return {'state': 'paused'}

            with performance.profile_adapter(config, profile) as execution:
                with adapter.stopping_adapter(config, binding):
                    run([1, 2])
                    self.assertFalse((output / 'STOP').exists())
                    run([3], resume=True)
            plan = {'execution_profile': {k: execution[k] for k in ['path', 'sha256', 'profile']},
                    'source_sha256': {execution['adapter_source']: execution['adapter_source_sha256']},
                    'transition_arms': {'global_control': {'session_ids': []}}}
            job = {'name': 'global_control', 'output': str(output), 'config': config}
            verified = controller_performance.verify_execution_sessions(job, plan)
            self.assertEqual([s['completed_epochs'] for s in verified['performance_sessions']], [[1, 2], [3]])
            record = json.loads((output / 'early_stopping.json').read_text())
            self.assertEqual(record['decision']['bad_epochs'], 3)
            self.assertTrue(record['stop_created'])

    def test_tampered_policy_fails_and_hooks_are_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, _ = self.setup_run(Path(folder))
            original = frozen.start_run, frozen.epoch_completed
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                with adapter.stopping_adapter(config, binding):
                    Path(binding['path']).write_text(json.dumps({**deepcopy(policy.POLICY), 'patience': 4}))
                    self.invocation(output, config, [.79])
            self.assertEqual((frozen.start_run, frozen.epoch_completed), original)
            self.assertFalse((output / 'compliance/events.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
