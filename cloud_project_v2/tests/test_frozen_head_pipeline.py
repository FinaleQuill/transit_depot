"""Background classification-head jobs fail closed at incomplete inputs/budget."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import run_frozen_head_pipeline as pipeline


class FrozenHeadPipelineTests(unittest.TestCase):
    def test_dead_cache_is_not_silently_restarted_or_used_for_fitting(self):
        registration = {'budget': {'cache_seconds_max': 5400}}
        launch = {'started_hong_kong': '2026-11-01T20:40:00+08:00', 'container': 'cache-only'}
        with patch.object(pipeline, 'REGISTRATION', registration, create=True), \
                patch.object(pipeline, 'DEADLINE', 9999999999, create=True), \
                patch.object(pipeline, 'read', side_effect=[launch, {'complete': False}]), \
                patch.object(pipeline, 'container_exists', return_value=False), \
                patch.object(pipeline.time, 'time', return_value=0), \
                patch.object(pipeline, 'update'), \
                patch.object(pipeline.subprocess, 'Popen', side_effect=AssertionError('Unexpected training')):
            with self.assertRaisesRegex(RuntimeError, 'without complete'):
                pipeline.await_cache()

    def test_head_config_refuses_e11_even_if_receipt_claims_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'A_F1').mkdir()
            (root/'A_F1/patch_fp16.pt').write_bytes(b'fixture only')
            receipt = {'state':'completed','final_epoch':11,'training_plan_sha256':pipeline.PLAN_SHA,
                       'head_patch_fp16_sha256':pipeline.sha(root/'A_F1/patch_fp16.pt')}
            with patch.object(pipeline, 'FITS', root), \
                    patch.object(pipeline, 'REGISTRATION', {'encoders':{'A':{}}}, create=True), \
                    patch.object(pipeline, 'read', return_value=receipt):
                with self.assertRaisesRegex(ValueError, 'registered final epoch'):
                    pipeline.write_patch_config('A_F1')

    def test_stage_timeout_kills_only_the_named_container_and_never_marks_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = MagicMock()
            process.poll.return_value = None
            with patch.object(pipeline, 'EVIDENCE', Path(temporary)), \
                    patch.object(pipeline, 'DEADLINE', 0, create=True), \
                    patch.object(pipeline, 'require_sources'), \
                    patch.object(pipeline, 'update'), \
                    patch.object(pipeline, 'container_exists', return_value=False), \
                    patch.object(pipeline.subprocess, 'Popen', return_value=process), \
                    patch.object(pipeline, 'kill_container') as kill:
                with self.assertRaisesRegex(TimeoutError, 'Budget reached'):
                    pipeline.run_stage('fit_A_F1', ['head.py'], 100)
                kill.assert_called_once_with('codex-frozen-fit-a-f1-20261101')
                process.wait.assert_called_once_with(timeout=60)

    def test_final_comparison_is_valid_python_and_uses_actual_saved_outputs(self):
        with patch.object(pipeline, 'run_stage') as stage:
            pipeline.final_actual_comparison()
            arguments = stage.call_args.args[1]
            self.assertEqual(arguments[0], '-c')
            compile(arguments[1], '<actual-comparison>', 'exec')
            self.assertIn('validation_report.json', arguments[1])
            self.assertIn('predictions_sha256', arguments[1])


if __name__ == '__main__':
    unittest.main()
