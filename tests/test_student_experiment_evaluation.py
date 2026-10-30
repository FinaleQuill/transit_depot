"""Validation arithmetic, immutable output/cache guards and report behavior.

No model is constructed and no GPU is queried in these tests.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import student_experiment_evaluation as evaluation
from deploy_kd import sha256_file


class EvaluationTests(unittest.TestCase):
    def fixtures(self, root):
        records = []
        for index in range(6):
            subject, study = f'subject_{index//2}', f'study_{index}'
            for view in range(2 if index % 2 == 0 else 1):
                records.append({'image_id': f'{index}_{view}', 'split': 'valid',
                    'subject_id': subject, 'study_id': study, 'study_key': f'{subject}/{study}',
                    'targets': [index % 2] * 10, 'relative_path': f'valid/{subject}/{study}/{view}.jpg',
                    'path': root / f'{index}_{view}.jpg'})
        manifest = root / 'valid.csv'
        with manifest.open('w', newline='') as stream:
            fields = ['image_id', 'split', 'subject_id', 'study_id', 'study_key', 'targets', 'relative_path']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({key: json.dumps(row[key]) if key == 'targets' else row[key] for key in fields})
        return records, manifest

    def test_study_probability_average_and_exact_image_csv_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, manifest = self.fixtures(root)
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'config.json').write_text('{}')
            # Two-view Study0 receives .9 and .1; must aggregate to .5.
            # Other classes repeat the same values solely to keep the fixture small.
            probabilities = {row['path']: np.full(10, .2 if row['targets'][0] == 0 else .8) for row in records}
            probabilities[records[0]['path']] = np.full(10, .9)
            probabilities[records[1]['path']] = np.full(10, .1)
            fake = SimpleNamespace(model_dir=model_dir, models=[object()],
                config={'official_valid_manifest_sha256': sha256_file(manifest), 'image_size': 768, 'tta': 'hflip2'},
                predict_image=lambda path: probabilities[path])
            output = root / 'evaluation'
            with patch.object(evaluation, 'prediction_records', return_value=records):
                report = evaluation.evaluate_predictor(fake, image_root=root / 'images', manifest=manifest,
                    output_dir=output, environment={'gpu_name': 'CPU_fixture_no_GPU_query'})
            saved = evaluation.read_submission(output / 'valid_predictions.csv')
            np.testing.assert_array_equal(saved[('subject_0', 'study_0')], np.full(10, .5))
            self.assertEqual(report['num_images'], 9)
            self.assertEqual(report['num_studies'], 6)
            self.assertEqual(report['macro_auc'], 1.)
            with (output / 'valid.image_predictions.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 90)
            self.assertFalse(report['test_images_used'])

    def test_labels_reject_test_and_inconsistent_study(self):
        row = {'split': 'test', 'targets': [0] * 10, 'subject_id': 'p', 'study_id': 's'}
        with self.assertRaisesRegex(ValueError, 'only official valid'):
            evaluation.validation_labels([row])
        row['split'] = 'valid'
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            evaluation.validation_labels([row, {**row, 'targets': [1] * 10}])

    def test_exact_study_coverage_guard_and_slices(self):
        keys = [(f'p{i}', f's{i}') for i in range(4)]
        labels = {key: [i % 2] * 10 for i, key in enumerate(keys)}
        values = {key: np.full(10, .1 + .8 * (i % 2)) for i, key in enumerate(keys)}
        views = {key: 1 if i < 2 else 2 for i, key in enumerate(keys)}
        result = evaluation.score_probabilities(labels, values, views)
        self.assertEqual(result['slices']['single_image_study']['macro_auc'], 1.)
        self.assertEqual(result['slices']['multiple_image_study']['num_studies'], 2)
        with self.assertRaisesRegex(ValueError, 'exact original Study coverage'):
            evaluation.score_probabilities(labels, dict(list(values.items())[1:]), views)

    def test_cache_guard_detects_manifest_and_prediction_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            (root / 'valid.csv').write_text('fixture')
            (root / 'valid_predictions.csv').write_text('fixture probabilities')
            report = {'complete': True, 'manifest_sha256': sha256_file(root / 'valid.csv'),
                'deployment_config_sha256': sha256_file(model / 'config.json'),
                'predictions_sha256': '0'*64}
            (root / 'validation_report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Cached reference validation identity differs'):
                evaluation.checked_cached_validation(root, model_dir=model, image_root=root / 'images', manifest=root / 'valid.csv')

    def test_history_reports_class1_late_decline_instead_of_assuming_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            rows = [{'epoch': i, 'validation': {'macro_auc': macro,
                     'per_class_auc': [class1] + [.8] * 9}}
                    for i, macro, class1 in [(1, .79, .75), (2, .81, .73), (3, .80, .68)]]
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            result = evaluation.history_diagnostics(path)
            self.assertEqual(result['best_macro_epoch'], 2)
            self.assertAlmostEqual(result['class_1_last_minus_best_macro_epoch'], -.05)
            self.assertAlmostEqual(result['class_1_last_minus_peak'], -.07)

    def test_report_highlights_target_classes_and_latency_warning(self):
        scores = {'macro_auc': .81, 'per_class_auc': [.81]*10, 'macro_f1_fixed_0_5': .4,
                  'class_1_4_8_mean_auc': .81}
        metrics = {'candidate': scores, 'reference_scores': {**scores, 'macro_auc': .80},
                   'macro_auc_delta': .01, 'paired_patient_bootstrap': {},
                   'class_1_training_trajectory': {'available': False},
                   'latency': {'modes': {'identity': {'mean_ms': 51.}, 'hflip2': {'mean_ms': 80.}},
                               'single_model_without_TTA_above_50ms': True}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'REPORT.md'
            evaluation.write_report(output, metrics, label='B')
            text = output.read_text()
            self.assertIn('**1**', text)
            self.assertIn('**4**', text)
            self.assertIn('**8**', text)
            self.assertIn('超过50ms', text)
            with self.assertRaises(FileExistsError):
                evaluation.write_report(output, metrics, label='B')


if __name__ == '__main__':
    unittest.main()
