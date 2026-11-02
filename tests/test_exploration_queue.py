import hashlib
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from run_exploration_queue import (Queue, active_run, preregistered,
                                  require_timing_evidence, verify_job_binding,
                                  verify_run)
from training_compliance import ComplianceRecorder

class QueueSafety(unittest.TestCase):
    def test_missing_registration_stops(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                preregistered({'name': 'fixture', 'preregistration': d+'/absent', 'preregistration_sha256': '0'*64})

    def test_incomplete_lifecycle_never_advances(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            c = ComplianceRecorder(p, trainer='synthetic queue test', configuration={}, demonstration=True)
            hashes = {}
            for role in ('best', 'last'):
                f = p/(role+'.pt')
                f.write_bytes(b'synthetic metadata fixture, not weights')
                hashes[role+'_checkpoint_sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
                c.checkpoint_saved(f, hashes[role+'_checkpoint_sha256'], epoch=1, role=role)
            c.epoch_completed(1, {'synthetic': True})
            (p/'history.jsonl').write_text(json.dumps({'epoch': 1})+'\n')
            (p/'status.json').write_text(json.dumps({'state': 'completed', 'completed_epochs': 1, **hashes}))
            with self.assertRaisesRegex(RuntimeError, 'incomplete compliance'):
                verify_run(p)
            c.finish('completed')
            self.assertEqual(verify_run(p)['completed_epochs'], 1)
            (p/'last.pt').write_bytes(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                verify_run(p)

    def test_controller_persists_failed_state_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prereg = root/'prereg.txt'
            prereg.write_text('Hypothesis: fixture\nExpected effect: fixture\nCost: fixture\nFailure: fixture\n')
            plan = root/'plan.json'
            plan.write_text(json.dumps({
                'output': str(root/'queue'), 'status_file': str(root/'STATUS.json'),
                'jobs': [{'name':'missing_run_prereg', 'kind':'train',
                          'preregistration':str(root/'absent.txt'), 'preregistration_sha256':'0'*64}],
            }))
            with self.assertRaisesRegex(RuntimeError, 'missing or changed preregistration'):
                Queue(plan).run()
            status = json.loads((root/'STATUS.json').read_text())
            self.assertEqual(status['state'], 'FAILED')
            self.assertIn('missing or changed preregistration', status['failure_reason'])
            self.assertTrue((root/'queue'/'FAILED').is_file())

    def test_controller_completes_empty_plan_with_stable_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = root/'plan.json'
            plan.write_text(json.dumps({'output':str(root/'queue'),
                'status_file':str(root/'STATUS.json'), 'jobs':[]}))
            Queue(plan).run()
            status = json.loads((root/'STATUS.json').read_text())
            self.assertEqual(status['state'], 'COMPLETED')
            self.assertEqual(status['completed_jobs'], [])

    def test_invalid_plan_schema_is_persisted_as_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = root/'plan.json'
            plan.write_text(json.dumps({'output':str(root/'queue'),
                'jobs':[{'name':'missing_kind'}]}))
            queue = Queue(plan)
            with self.assertRaisesRegex(RuntimeError, 'requires status_file'):
                queue.run()
            status = json.loads((root/'queue'/'STATUS.json').read_text())
            self.assertEqual(status['state'], 'FAILED')
            self.assertTrue((root/'queue'/'FAILED').is_file())

    def test_terminal_run_configuration_is_bound_to_preregistration(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            expected = {'seed': 44, 'image_size': 768, 'roi': True, 'mil': False}
            source = run_dir/'trainer.py'
            source.write_text('synthetic source\n')
            observed = {**expected, 'model_config':{'backbone':'tiny'}, 'preprocessing':{'size':768}}
            (run_dir/'config.json').write_text(json.dumps(observed))
            recorder = ComplianceRecorder(run_dir, trainer='synthetic binding test',
                configuration={**expected, 'fold_plan_sha256':'c'*64, 'derived_pos_weight':[1.0]*10},
                source_files=[source], demonstration=True)
            recorder.finish('completed')
            source_manifest = run_dir/'compliance'/f'sessions/{recorder.session_id}/source_manifest.json'
            source_map = {row['original_path']:row['sha256'] for row in json.loads(source_manifest.read_text())}
            job = {'run_dir':str(run_dir), 'expected_configuration':expected,
                   'expected_source_sha256':source_map}
            binding = verify_job_binding(job, {})
            self.assertEqual(binding['session_id'], recorder.session_id)
            config_path = run_dir/'compliance'/f'sessions/{recorder.session_id}/configuration.json'
            self.assertEqual(binding['configuration_sha256'], hashlib.sha256(config_path.read_bytes()).hexdigest())
            self.assertEqual(binding['source_manifest_sha256'], hashlib.sha256(source_manifest.read_bytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError, 'source files differ'):
                verify_job_binding({**job, 'expected_source_sha256':{**source_map, str(source):'0'*64}}, {})
            (run_dir/'config.json').write_text(json.dumps({**observed, 'roi':False}))
            with self.assertRaisesRegex(RuntimeError, 'differs from registration for roi'):
                verify_job_binding(job, {})

    def test_all_resumed_sessions_must_match_registered_config_and_sources(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            expected = {'seed': 44, 'image_size': 768}
            source = run_dir/'trainer.py'
            source.write_text('version one\n')
            (run_dir/'config.json').write_text(json.dumps(expected))
            first = ComplianceRecorder(run_dir, trainer='synthetic resumed binding test',
                configuration={'seed': 99, 'image_size': 768}, source_files=[source], demonstration=True)
            first.finish('paused')
            source.write_text('version two\n')
            second = ComplianceRecorder(run_dir, trainer='synthetic resumed binding test',
                configuration=expected, source_files=[source], demonstration=True, resume=run_dir/'last.pt')
            second.finish('completed')
            latest_manifest = run_dir/'compliance'/f'sessions/{second.session_id}/source_manifest.json'
            expected_sources = {row['original_path']:row['sha256'] for row in json.loads(latest_manifest.read_text())}
            job = {'run_dir':str(run_dir), 'expected_configuration':expected,
                   'expected_source_sha256':expected_sources}
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                verify_job_binding(job, {})

    def test_preregistration_requires_exactly_four_nonempty_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'prereg.txt'
            for content, valid in (
                    ('hypothesis\neffect\ncost\nfailure\n', True),
                    ('hypothesis\neffect\ncost\n', False),
                    ('hypothesis\neffect\ncost\nfailure\nextra\n', False),
                    ('hypothesis\n\neffect\ncost\nfailure\n', True)):
                path.write_text(content)
                job = {'name':'fixture', 'preregistration':str(path),
                       'preregistration_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                if valid:
                    self.assertEqual(preregistered(job)['sha256'], job['preregistration_sha256'])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'four nonempty lines'):
                        preregistered(job)

    def test_preregistration_changes_are_detected_during_queue(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prereg = root/'prereg.txt'
            prereg.write_text('four lines\n')
            plan = root/'plan.json'
            plan.write_text(json.dumps({'output':str(root/'queue'),
                'status_file':str(root/'STATUS.json'), 'jobs':[]}))
            queue = Queue(plan)
            queue.status['current_preregistration'] = {
                'path':str(prereg), 'sha256':hashlib.sha256(prereg.read_bytes()).hexdigest()}
            prereg.write_text('changed after registration\n')
            with self.assertRaisesRegex(RuntimeError, 'registration changed'):
                queue.check_stop()

    def test_active_run_returns_a_real_matching_pid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc = root/'proc'/'4242'
            proc.mkdir(parents=True)
            run_dir = root/'run_031'
            (proc/'cmdline').write_bytes(b'/usr/bin/python3\0/home/work/train_distill.py\0--output\0' +
                                         str(run_dir).encode() + b'\0--execute\0')
            self.assertEqual(active_run(run_dir, proc_root=root/'proc'), 4242)

    def test_adopted_run_stop_signals_matching_pid_not_boolean_pid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prereg = root/'prereg.txt'
            prereg.write_text('hypothesis\neffect\ncost\nfailure\n')
            plan = root/'plan.json'
            plan.write_text(json.dumps({'output':str(root/'queue'),
                'status_file':str(root/'STATUS.json'), 'jobs':[]}))
            queue = Queue(plan)
            registration = {'path':str(prereg), 'sha256':hashlib.sha256(prereg.read_bytes()).hexdigest()}
            queue.status['current_preregistration'] = registration
            (root/'queue'/'STOP').write_text('stop\n')
            job = {'run_dir':str(root/'run_031')}
            with patch('run_exploration_queue.active_run', side_effect=[4242, None]), \
                    patch('run_exploration_queue.os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError, 'STOP marker'):
                    queue.training(job, registration)
            kill.assert_called_once_with(4242, signal.SIGTERM)

    def test_timing_evidence_requires_finite_ordered_metrics_and_samples(self):
        report = {'timing':{'modes':{'hflip2':{'p50_ms':50., 'p95_ms':70., 'observations':90}}}}
        self.assertEqual(require_timing_evidence(report, 'hflip2', 'fixture')['p95_ms'], 70.)
        for timing in (
                {'p50_ms':50., 'p95_ms':None, 'observations':90},
                {'p50_ms':80., 'p95_ms':70., 'observations':90},
                {'p50_ms':float('nan'), 'p95_ms':70., 'observations':90},
                {'p50_ms':50., 'p95_ms':70., 'observations':0},
                {'p50_ms':50., 'p95_ms':70., 'observations':89}):
            with self.assertRaisesRegex(RuntimeError, 'timing metrics are invalid'):
                require_timing_evidence({'timing':{'modes':{'hflip2':timing}}}, 'hflip2', 'fixture')

if __name__ == '__main__':
    unittest.main()
