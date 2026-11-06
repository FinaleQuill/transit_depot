"""Approved phase ordering, failure gates, and bounded backup scheduling."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run_student_upper_bound as controller


class ControllerTests(unittest.TestCase):
    def run_sequence(self, directory, fail_a_evaluation=False):
        events = []
        def child(command, *, label, name, interval):
            events.append((label, name))
            if fail_a_evaluation and (label, name) == ('A', 'evaluation'):
                raise RuntimeError('evaluation did not complete')
        report = directory / 'reports/A/REPORT.md'
        report.parent.mkdir(parents=True)
        report.write_text('A vs 033 complete report')
        with (patch.object(controller, 'EVIDENCE', directory),
              patch.object(controller, 'LOCAL', directory),
              patch.object(controller, 'commands', return_value=(['a'], ['b'], {'A':['ae'], 'B':['be']})),
              patch.object(controller, 'freeze_sources', return_value={}),
              patch.object(controller, 'require_sources'),
              patch('training_evidence_monitor.snapshot_source'),
              patch.object(controller.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*1024**3)),
              patch.object(controller, 'wait_existing_a', side_effect=lambda _:events.append(('A','attached_training'))),
              patch.object(controller, 'run_child', side_effect=child)):
            controller.execute(attach_a=True, interval=1)
        return events

    def test_a_report_precedes_b_training_and_b_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = self.run_sequence(Path(tmp))
            self.assertEqual(events, [('A','attached_training'), ('A','evaluation'),
                                      ('B','training'), ('B','evaluation')])
            self.assertEqual(json.loads((Path(tmp)/'pipeline_status.json').read_text())['stage'],'completed')

    def test_failed_a_evaluation_never_launches_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'evaluation did not complete'):
                self.run_sequence(Path(tmp), fail_a_evaluation=True)
            status=json.loads((Path(tmp)/'pipeline_status.json').read_text())
            self.assertEqual(status['stage'],'failed')
            self.assertIn('evaluation did not complete',status['error'])

    def test_source_change_is_rejected_before_next_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.py'
            source.write_text('original')
            expected={'source.py':controller.digest(source)}
            with patch.object(controller,'ROOT',Path(tmp)):
                controller.require_sources(expected)
                source.write_text('modified')
                with self.assertRaisesRegex(RuntimeError,'source.py'):
                    controller.require_sources(expected)

    def test_committed_backup_is_only_rehashed_after_status_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            status={'best_checkpoint_sha256':'a'*64,'last_checkpoint_sha256':'b'*64,'state':'running'}
            (run/'status.json').write_text(json.dumps(status))
            with (patch.object(controller,'EVIDENCE',root/'evidence'),
                  patch.object(controller,'LOCAL',root/'local'),
                  patch.object(controller,'RUNS',{'A':run}),
                  patch.object(controller,'BACKUP_COMMITTED',{}),
                  patch('student_checkpoint_backup.backup_run',return_value={'state':'complete'}) as actual):
                controller.backup('A');controller.backup('A')
                self.assertEqual(actual.call_count,1)
                status['last_checkpoint_sha256']='c'*64
                (run/'status.json').write_text(json.dumps(status))
                controller.backup('A')
                self.assertEqual(actual.call_count,2)

    def test_external_a_exit_is_checked_even_when_status_remains_running(self):
        for exit_code in (137, 0):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                run=Path(tmp);(run/'evidence').mkdir()
                (run/'status.json').write_text(json.dumps({'state':'running','completed_epochs':0}))
                (run/'evidence/command_1.json').write_text(json.dumps({'exit_code':exit_code}))
                with (patch.object(controller,'RUNS',{'A':run}),
                      patch.object(controller,'stage'),
                      patch.object(controller,'backup',side_effect=AssertionError('Must detect exit first')),
                      patch.object(controller.time,'sleep',side_effect=AssertionError('Must not wait forever'))):
                    with self.assertRaises(RuntimeError):
                        controller.wait_existing_a(1)

    def test_final_backup_refreshes_receipt_after_unchanged_model_hash(self):
        with (patch.object(controller,'backup',return_value={'state':'complete'}),
              patch('student_checkpoint_backup.refresh_run_metadata',return_value={
                  'state':'complete','metadata_refreshed':True}) as refresh,
              patch.object(controller,'write_json')):
            result=controller.require_final_backup('A')
            self.assertTrue(result['metadata_refreshed'])
            refresh.assert_called_once()


if __name__=='__main__':
    unittest.main()
