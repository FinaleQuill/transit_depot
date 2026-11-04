"""CPU contract checks for winner-only publication and shared Study inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deploy_kd import read_submission, sha256_file
from sprint.deploy import (MODEL_WEIGHTS, SprintPredictor, _write_studies, inspect_package,
                          prepare_package, publish, winning_selection)


class DeploymentTests(unittest.TestCase):
    def _fixture(self, root):
        comparison = root / 'final_comparison'
        comparison.mkdir()
        base = root / 'original_tiny'
        base.mkdir()
        (base / 'weights.pt').write_bytes(b'original tiny weights')
        (base / 'config.json').write_text(json.dumps({'weights_file': 'weights.pt'}))
        patch_dir = root / 'original_patch'
        patch_dir.mkdir()
        (patch_dir / 'patch.pt').write_bytes(b'original classifier')
        tiny = {'base_model_dir': str(base), 'base_config_sha256': sha256_file(base / 'config.json'),
                'base_weights_sha256': sha256_file(base / 'weights.pt'), 'patch_file': 'patch.pt',
                'patch_sha256': sha256_file(patch_dir / 'patch.pt')}
        (patch_dir / 'config.json').write_text(json.dumps(tiny))
        weights = root / 'candidate.pt'
        weights.write_bytes(b'final epoch weights')
        for name in ['candidate_predictions.csv', 'candidate.image_predictions.csv']:
            (comparison / name).write_text(name)
        report = {'complete': True, 'candidate_won': True,
                  'candidate_weights_sha256': sha256_file(weights), 'manifest_sha256': 'a' * 64,
                  'fixed_classwise_probability_weights': MODEL_WEIGHTS,
                  'candidate_predictions_sha256': sha256_file(comparison / 'candidate_predictions.csv'),
                  'candidate_image_predictions_sha256': sha256_file(comparison / 'candidate.image_predictions.csv')}
        selection = {'candidate_won': True, 'selected': 'sprint_candidate',
                     'report': str(comparison / 'report.json'), 'candidate_weights': str(weights),
                     'candidate_weights_sha256': sha256_file(weights),
                     'tiny_patch_config': str(patch_dir / 'config.json'), 'tta': 'hflip2',
                     'fixed_classwise_probability_weights': MODEL_WEIGHTS,
                     'selection_checks': {'beats_baseline': True}}
        (comparison / 'report.json').write_text(json.dumps(report))
        (comparison / 'selection.json').write_text(json.dumps(selection))
        return comparison, selection

    def test_losing_or_changed_selection_refused_before_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison, selection = self._fixture(root)
            selection['candidate_won'] = False
            (comparison / 'selection.json').write_text(json.dumps(selection))
            with patch('sprint.deploy.prediction_records') as records:
                with self.assertRaises(ValueError):
                    publish({'output': str(root)})
                records.assert_not_called()
            self.assertFalse((root / 'deployment_candidate').exists())
            selection['candidate_won'] = True
            (comparison / 'selection.json').write_text(json.dumps(selection))
            (root / 'candidate.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'weights changed'):
                winning_selection(comparison)

    def test_package_portable_inventory_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison, _ = self._fixture(root)
            destination = root / 'deployment_candidate'
            with patch('sprint.deploy._copy_runtime_sources', return_value=[]):
                report = prepare_package(comparison, destination)
            self.assertTrue(report['size_pass'])
            patch_config = json.loads((destination / 'model/tiny_patch/config.json').read_text())
            self.assertEqual(patch_config['base_model_dir'], '../tiny')
            moved = root / 'relocated'
            destination.rename(moved)
            self.assertTrue(inspect_package(moved / 'model')['size_pass'])
            (moved / 'model/tiny/weights.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                inspect_package(moved / 'model')

    def test_predict_and_timing_share_complete_study_mean_and_csv(self):
        records = [dict(study_key=f'subject_1/study_{study}', subject_id='subject_1',
                        study_id=f'study_{study}', image_id=f'image_{image}', path=Path(str(image)))
                   for study, image in [(1, 1), (2, 2), (1, 3)]]
        predictor = SprintPredictor.__new__(SprintPredictor)
        seen = []
        def predict_image(path):
            seen.append(str(path))
            return np.full(10, int(str(path)) / 10., dtype=np.float64)
        predictor.predict_image = predict_image
        predictor.synchronize = lambda: None
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / 'predict.csv', Path(temporary) / 'cost.csv'
            _, predicted = _write_studies(predictor, records, first)
            times, timed = _write_studies(predictor, records, second, timing=True)
            self.assertEqual(len(times), 2)
            self.assertEqual(len(seen), 6)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            np.testing.assert_array_equal(predicted[('subject_1', 'study_1')], np.full(10, .2))
            self.assertEqual(set(read_submission(first)), set(timed))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                predictor.predict_study(records)
            with self.assertRaisesRegex(ValueError, 'duplicate image'):
                predictor.predict_study([records[0], records[0]])


if __name__ == '__main__':
    unittest.main()
