"""CPU-only transition, bounded-probe and frozen execution evidence contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from reconstruction import controller_checkpoint_speed as queue
from reconstruction.local_compliance import LocalComplianceRecorder


def binding_fixture(root):
    path = root / 'checkpoint_profile.json'
    queue.base.write_json(path, queue.PROFILE)
    return queue.profile_binding(path, queue.base.sha(path))


def result_fixture(binding, *, adopt=True, oom=False):
    return {'schema': queue.BENCHMARK_SCHEMA, 'complete': True, 'baseline_valid': True,
        'adopt': adopt, 'candidate_oom': oom, 'profile_binding': binding,
        'checkpoint_sha256': 'a' * 64, 'weighted_speedup': 1.2,
        'exact_equivalence': True, 'candidate_min_device_headroom_bytes': 1610612736,
        'predicted_seconds_saved_per_epoch': 1000., 'total_benchmark_seconds': 400.}


class RequestAndProbeTests(unittest.TestCase):
    def test_cancel_is_checked_even_if_the_lock_is_already_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / queue.CANCEL_FILE).touch()
            request = {'schema': queue.REQUEST_SCHEMA, 'boundary_epoch': 2, 'previous_controller_pid': 1}
            with (directory / '.lock').open('a+b') as lock, self.assertRaisesRegex(RuntimeError, 'canceled'):
                queue.wait_for_lock(lock, directory, request)

    def test_changed_user_stop_is_never_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'STOP').write_text('new stop')
            request = {'schema': queue.REQUEST_SCHEMA, 'boundary_epoch': 2,
                       'previous_controller_pid': 1, 'requested_stop_sha256': 'a' * 64}
            with self.assertRaisesRegex(ValueError, 'STOP changed'):
                queue.verify_request(directory, request)
            self.assertEqual((directory / 'STOP').read_text(), 'new stop')

    def test_wait_is_bounded_without_launching_a_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with (directory / '.lock').open('a+b') as lock, \
                 mock.patch.object(queue, 'verify_request'), \
                 mock.patch.object(queue.fcntl, 'flock', side_effect=BlockingIOError), \
                 mock.patch.object(queue.time, 'monotonic', side_effect=[0, 2]), \
                 self.assertRaisesRegex(TimeoutError, 'bounded wait'):
                queue.wait_for_lock(lock, directory, {}, timeout=1)

    def test_kill_helper_refuses_training_container_names(self):
        with self.assertRaisesRegex(ValueError, 'Only this bounded temporary probe'):
            queue.kill_probe_only('cxr-e2-global-local-p3-123', mock.Mock())

    def test_gpu_probe_refuses_other_compute_processes(self):
        with mock.patch.object(queue.subprocess, 'run', return_value=mock.Mock(stdout='65432\n')), \
             self.assertRaisesRegex(RuntimeError, 'exclusive access'):
            queue.assert_gpu_idle()

    def test_probe_timeout_kills_only_its_unique_probe_and_never_restarts_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'controller'; directory.mkdir()
            speed = root / 'checkpoint_speed_20261104'; speed.mkdir()
            binding = binding_fixture(root)
            request = {'benchmark_output': str(speed / 'benchmark')}
            job = {'config_path': 'original.json', 'output': str(root / 'local')}
            child = mock.Mock()
            child.pid = 123
            child.poll.return_value = None
            with mock.patch.object(queue, 'verify_request'), \
                 mock.patch.object(queue, 'assert_no_e2_container'), \
                 mock.patch.object(queue, 'assert_gpu_idle'), \
                 mock.patch.object(queue.runtime_local, 'command', return_value=['fake-probe']), \
                 mock.patch.object(queue.subprocess, 'Popen', return_value=child), \
                 mock.patch.object(queue.time, 'monotonic', side_effect=[0, 0, 601]), \
                 mock.patch.object(queue, 'kill_probe_only') as kill, \
                 mock.patch.object(queue, 'run_selected_controller') as train:
                with self.assertRaisesRegex(TimeoutError, '600 seconds'):
                    queue.run_probe(directory, request, job, binding, 'a' * 64)
                self.assertEqual(kill.call_count, 1)
                self.assertTrue(kill.call_args.args[0].startswith('cxr-e2-checkpoint-probe-'))
                train.assert_not_called()

    def test_duplicate_probe_directory_preserves_prior_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'controller'; directory.mkdir()
            output = root / 'checkpoint_speed_20261104/benchmark'; output.mkdir(parents=True)
            with mock.patch.object(queue.subprocess, 'Popen') as popen, \
                 self.assertRaisesRegex(FileExistsError, 'automatically retried'):
                queue.run_probe(directory, {'benchmark_output': str(output)}, {}, {}, 'x')
            popen.assert_not_called()


class BenchmarkDecisionTests(unittest.TestCase):
    def test_valid_candidate_oom_is_a_fallback_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binding = binding_fixture(root)
            result = result_fixture(binding, adopt=False, oom=True)
            path = root / 'RESULT.json'; queue.base.write_json(path, result)
            self.assertFalse(queue.benchmark_result(path, binding, 'a' * 64)['adopt'])

    def test_incomplete_or_baseline_failed_or_wrong_weights_cannot_fallback(self):
        for field, value in [('complete', False), ('baseline_valid', False), ('checkpoint_sha256', 'b' * 64)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); binding = binding_fixture(root)
                result = result_fixture(binding, adopt=False); result[field] = value
                path = root / 'RESULT.json'; queue.base.write_json(path, result)
                with self.assertRaisesRegex(ValueError, 'incomplete, failed, or bound'):
                    queue.benchmark_result(path, binding, 'a' * 64)

    def test_bound_worker_results_replay_and_cannot_drift(self):
        from reconstruction import benchmark_checkpoint_speed as probe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binding = binding_fixture(root)
            plan = {'checkpoint_path': '/synthetic/last.pt', 'checkpoint_sha256': 'a' * 64,
                'profile_binding': binding, 'input_sha256': {'/synthetic/last.pt': 'a' * 64},
                'workload': {'study_image_count_distribution': {'1': 100, '2': 100, '7': 1},
                             'study_count': 201}, 'baseline_training_epoch_seconds': 2000,
                'effective_studies': 32}
            baseline = {'complete': True, 'outcome': 'valid', 'weights_sha256': 'a' * 64, 'shapes': {}}
            for count in ['1', '2', '7']:
                baseline['shapes'][count] = {'input_sha256': 'pixels', 'initial_rng_sha256': 'rng',
                    'initial_state_sha256': 'state', 'equivalence': {'skipped_update': False},
                    'timing_final_rng_sha256': 'same', 'median_forward_backward_seconds': 2.,
                    'median_optimizer_ema_seconds': .1, 'estimated_min_device_headroom_bytes': 2 * 1024**3}
            candidate = deepcopy(baseline)
            for count in ['1', '2']:
                candidate['shapes'][count]['median_forward_backward_seconds'] = 1.
            for name, content in [('PLAN.json', plan), ('original.json', baseline), ('retain_small.json', candidate)]:
                queue.base.write_json(root / name, content)
            result = probe.compare(plan, baseline, candidate, 100.)
            result['worker_result_sha256'] = {mode: queue.base.sha(root / f'{mode}.json') for mode in probe.MODES}
            for name in ['RESULT.json', 'result.json']:
                queue.base.write_json(root / name, result)
            acceptance = {'adopt': True, 'controller_observed_seconds': 110.,
                          'benchmark_sha256': queue.base.sha(root / 'RESULT.json')}
            queue.base.write_json(root / 'EXTERNAL_ACCEPTANCE.json', acceptance)
            benchmark = {'path': str(root / 'RESULT.json'), 'adopt': True,
                         'controller_observed_seconds': 110., 'acceptance': acceptance,
                         'artifact_sha256': {str(p): queue.base.sha(p) for p in root.iterdir()}}
            queue.verify_benchmark_artifacts(benchmark)
            candidate['shapes']['1']['median_forward_backward_seconds'] = .01
            queue.base.write_json(root / 'retain_small.json', candidate)
            with self.assertRaisesRegex(ValueError, 'Frozen probe artifact'):
                queue.verify_benchmark_artifacts(benchmark)

    def test_adoption_requires_speed_and_positive_first_epoch_net_saving(self):
        for field, value in [('weighted_speedup', 1.079), ('predicted_seconds_saved_per_epoch', 700.),
                             ('candidate_oom', True), ('exact_equivalence', False),
                             ('candidate_min_device_headroom_bytes', 1610612735)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); binding = binding_fixture(root)
                result = result_fixture(binding); result[field] = value
                path = root / 'RESULT.json'; queue.base.write_json(path, result)
                with self.assertRaisesRegex(ValueError, 'speed and net-time gates'):
                    queue.benchmark_result(path, binding, 'a' * 64)


class TransitionDecisionTests(unittest.TestCase):
    def run_transition(self, root, *, adopt):
        directory = root / 'controller'; directory.mkdir()
        archive = directory / 'checkpoint_speed_transition'; archive.mkdir()
        (directory / 'STOP').write_text('authorized transition')
        binding = binding_fixture(root)
        jobs = [{'name': name, 'config_path': str(root / f'{name}.json'), 'output': str(root / name)}
                for name in ['global_control', 'global_local']]
        plan, perf, amendment = {'jobs': jobs}, {}, {}
        request = {'profile_path': binding['path'], 'profile_sha256': binding['sha256']}
        benchmark = {'path': str(root / 'RESULT.json'), 'sha256': 'b' * 64,
                     'result': result_fixture(binding, adopt=adopt), 'adopt': adopt}
        execution = {'adopted': True} if adopt else None
        certificates = [{}, {'checkpoint_sha256': {'last.pt': 'a' * 64}}]
        with mock.patch.object(queue, 'load_plans', return_value=(plan, perf, amendment)), \
             mock.patch.object(queue, 'verify_closed_boundary', return_value=certificates) as verify, \
             mock.patch.object(queue, 'assert_no_e2_container'), \
             mock.patch.object(queue, 'snapshot_transition', return_value={}), \
             mock.patch.object(queue, 'run_probe', return_value=benchmark) as probe, \
             mock.patch.object(queue, 'create_execution_plan', return_value=execution) as create, \
             mock.patch.object(queue, 'verify_execution_plan'), \
             mock.patch.object(queue, 'verify_request'), \
             mock.patch.object(queue, 'run_selected_controller', return_value={'state': 'completed'}) as run:
            queue.transition(directory, request)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(create.call_count, int(adopt))
        self.assertIs(run.call_args.args[4], execution)
        self.assertEqual(run.call_args.kwargs['resumes'], [jobs[1]['config_path']])
        self.assertFalse((directory / 'STOP').exists())
        self.assertEqual((archive / 'consumed_controller.STOP').read_text(), 'authorized transition')

    def test_valid_negative_probe_resumes_original_trainer_at_local3(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_transition(Path(tmp), adopt=False)

    def test_valid_positive_probe_creates_new_plan_then_resumes_local3(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_transition(Path(tmp), adopt=True)

    def test_probe_failure_preserves_stop_and_does_not_resume_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / 'controller'; directory.mkdir()
            (directory / 'STOP').write_text('original')
            binding = binding_fixture(root)
            jobs = [{}, {'config_path': 'local', 'output': 'local'}]
            request = {'profile_path': binding['path'], 'profile_sha256': binding['sha256']}
            with mock.patch.object(queue, 'load_plans', return_value=({'jobs': jobs}, {}, {})), \
                 mock.patch.object(queue, 'verify_closed_boundary', return_value=[{}, {'checkpoint_sha256': {'last.pt': 'x'}}]), \
                 mock.patch.object(queue, 'assert_no_e2_container'), \
                 mock.patch.object(queue, 'snapshot_transition', return_value={}), \
                 mock.patch.object(queue, 'run_probe', side_effect=RuntimeError('baseline failed')) as probe, \
                 mock.patch.object(queue, 'run_selected_controller') as run:
                with self.assertRaisesRegex(RuntimeError, 'baseline failed'):
                    queue.transition(directory, request)
                self.assertEqual(probe.call_count, 1)
                run.assert_not_called()
                self.assertEqual((directory / 'STOP').read_text(), 'original')


def execution_session(root, *, wrong_count=False, missing_activation=False, historical=False):
    output = root / 'local'; output.mkdir()
    binding = binding_fixture(root)
    adapter = Path(queue.__file__).with_name('train_local_checkpoint_speed.py').resolve()
    execution = {'checkpoint_profile': binding, 'source_sha256': {str(adapter): queue.base.sha(adapter)},
                 'transition_arms': {'global_local': {'session_ids': []}}}
    recorder = LocalComplianceRecorder(output, trainer='reconstruction.train_local', configuration={},
                                      source_files=[adapter, binding['path']])
    if not missing_activation:
        recorder._record('checkpoint_execution_profile_activated', {
            'checkpoint_profile': binding, 'adapter_source': str(adapter),
            'adapter_source_sha256': queue.base.sha(adapter), 'scientific_configuration_changed': False,
            'forward_packing_chunking_and_rng_order_unchanged': True, 'activation_retention_only': True})
    counts = {'retained_packs': 1, 'retained_regions': 10, 'checkpointed_packs': 1,
              'checkpointed_regions': 15, 'total_packs': 2, 'total_regions': 25,
              'max_regions': 15, 'pack_region_counts': {'10': 1, '15': 1}}
    if wrong_count:
        counts['retained_regions'] = 11
    recorder._record('checkpoint_execution_epoch', {'epoch': 3,
        'checkpoint_profile_sha256': binding['sha256'], 'checkpoint_profile_id': queue.PROFILE['profile_id'],
        'counts': counts, 'complete_training_epoch': True, 'evaluation_and_checkpoint_save_pending': True})
    row = {'epoch': 3, 'regions': 25}
    recorder.epoch_completed(3, row)
    (output / 'history.jsonl').write_text('\n'.join(json.dumps(r) for r in
        [{'epoch': 1, 'regions': 25}, {'epoch': 2, 'regions': 25}, row]) + '\n')
    recorder.finish('paused', {'state': 'paused'})
    events = queue.local.closed_sessions(output)
    if historical:
        execution['transition_arms']['global_local']['session_ids'] = [events[0]['session_id']]
    return {'name': 'global_local', 'output': str(output)}, execution


class ExecutionSessionTests(unittest.TestCase):
    def test_actual_region_distribution_and_archives_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, execution = execution_session(Path(tmp))
            result = queue.verify_checkpoint_sessions(job, execution)
            self.assertEqual(result['checkpoint_execution_sessions'][0]['epochs'], [3])

    def test_self_consistent_chain_cannot_hide_wrong_retained_region_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, execution = execution_session(Path(tmp), wrong_count=True)
            with self.assertRaisesRegex(ValueError, 'Activation counts'):
                queue.verify_checkpoint_sessions(job, execution)

    def test_missing_activation_rejects_unprofiled_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, execution = execution_session(Path(tmp), missing_activation=True)
            with self.assertRaisesRegex(ValueError, 'activate exactly one'):
                queue.verify_checkpoint_sessions(job, execution)

    def test_historical_sessions_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, execution = execution_session(Path(tmp), historical=True)
            with self.assertRaisesRegex(ValueError, 'Historical session was relabelled'):
                queue.verify_checkpoint_sessions(job, execution)


class QueueTests(unittest.TestCase):
    def test_external_stop_before_queue_launches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ['performance_plan.json', 'early_stopping_plan.json', 'checkpoint_execution_plan.json']:
                queue.base.write_json(directory / name, {})
            (directory / 'STOP').touch()
            jobs = [{'name': 'global_control', 'config_path': 'global.json'}]
            ctl = queue.CheckpointSpeedController(directory, {'jobs': jobs}, {'execution_profile': {}},
                                                 {'policy_binding': {}}, {'checkpoint_profile': {}})
            with mock.patch.object(ctl, 'verify_frozen'), mock.patch.object(ctl, 'launch') as launch:
                self.assertEqual(ctl.run()['state'], 'paused')
                launch.assert_not_called()

    def test_training_failure_never_retries_or_restarts_completed_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ['performance_plan.json', 'early_stopping_plan.json', 'checkpoint_execution_plan.json']:
                queue.base.write_json(directory / name, {})
            jobs = [{'name': n, 'config_path': str(directory / f'{n}.json'),
                     'output': str(directory / n)} for n in ['global_control', 'global_local']]
            ctl = queue.CheckpointSpeedController(directory, {'jobs': jobs}, {'execution_profile': {}},
                {'policy_binding': {}}, {'checkpoint_profile': {}}, resume_runs=[jobs[1]['config_path']])
            certificates = [{'policy_complete': True}, {'policy_complete': False}]
            with mock.patch.object(ctl, 'verify_frozen'), \
                 mock.patch.object(ctl, 'verify_job', side_effect=certificates), \
                 mock.patch.object(ctl, 'launch', return_value=9) as launch:
                with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                    ctl.run()
                self.assertEqual(launch.call_count, 1)
                self.assertIn(jobs[1]['config_path'], launch.call_args.args[0])


class SummaryCompletionTests(unittest.TestCase):
    def test_extra_verification_failure_keeps_summary_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            execution = {'checkpoint_profile': {}}
            queue.base.write_json(directory / 'checkpoint_execution_plan.json', execution)
            report_path = directory / 'comparison/report.json'
            def old_summary(_):
                result = {'complete': True}
                queue.base.write_json(report_path, result)
                return result
            with mock.patch.object(queue, 'load_plans', return_value=({'jobs': []}, {}, {})), \
                 mock.patch.object(queue, 'verify_execution_plan', side_effect=[None, ValueError('postcheck failed')]), \
                 mock.patch.object(queue.early, 'summarize', side_effect=old_summary):
                with self.assertRaisesRegex(ValueError, 'postcheck failed'):
                    queue.summarize(directory)
            self.assertFalse(queue.base._read_json(report_path)['complete'])


if __name__ == '__main__':
    unittest.main()
