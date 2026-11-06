"""Background handoff refuses duplicate jobs and insufficient training budget."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_bounded_pipeline as pipeline


class BoundedPipelineTests(unittest.TestCase):
    def test_missing_existing_run_never_silently_restarts_x1(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(pipeline, 'EVIDENCE', Path(temporary)), \
                patch.object(pipeline, 'run_directory', return_value=Path(temporary) / 'missing'), \
                patch.object(pipeline, 'launcher_alive', return_value=False), \
                patch.object(pipeline.subprocess, 'run') as inspect:
            inspect.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'absent'):
                pipeline.wait_training('X1')

    def test_final_export_does_not_make_incomplete_epoch_count_successful(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'run').mkdir()
            (root / 'run/status.json').write_text(json.dumps({'state': 'completed', 'completed_epochs': 3}))
            (root / 'X1.final_ema.json').write_text(json.dumps({
                'registered_plan_sha256': pipeline.PLAN_SHA, 'packing_addendum_sha256': pipeline.PACKING_SHA,
                'source_epoch': 4}))
            with patch.object(pipeline, 'EVIDENCE', root), \
                    patch.object(pipeline, 'run_directory', return_value=root / 'run'), \
                    patch.object(pipeline, 'registered', return_value=({}, {'epochs': 4}, None)):
                with self.assertRaisesRegex(ValueError, 'completion'):
                    pipeline.wait_training('X1')

    def test_x2_never_launches_when_its_budget_and_validation_reserve_do_not_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            deadline = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
            plan = {'budget': {'deadline_hong_kong': deadline}}
            with patch.object(pipeline, 'EVIDENCE', Path(temporary)), \
                    patch.object(pipeline, 'run_directory', side_effect=lambda stage: Path(temporary) / stage), \
                    patch.object(pipeline, 'registered', return_value=(plan, {}, None)), \
                    patch.object(pipeline, 'wait_training') as waiting, \
                    patch.object(pipeline, 'evaluate_and_review') as evaluation, \
                    patch.object(pipeline.subprocess, 'Popen', side_effect=AssertionError('Unexpected training')):
                pipeline.main()
                waiting.assert_called_once_with('X1')
                evaluation.assert_called_once_with('X1')
                result = json.loads((Path(temporary) / 'pipeline_status.json').read_text())
                self.assertEqual(result['state'], 'budget_exhausted')


if __name__ == '__main__':
    unittest.main()
