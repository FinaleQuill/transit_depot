"""CPU checks for baseline identity and the single final selection decision."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from sprint.evaluate import _reference, selection_checks
from deploy_kd import CSV_HEADER, sha256_file, write_prediction
from student_experiment_evaluation import score_probabilities


class FinalComparisonTests(unittest.TestCase):
    def test_a_win_over_submission_best_alone_does_not_replace_it(self):
        online = {"macro_auc": .809, "per_class_auc": [.690] * 10}
        local = {"macro_auc": .8176, "per_class_auc": [.746] * 10}
        candidate = {"macro_auc": .815, "per_class_auc": [.750] * 10}
        checks = selection_checks(candidate, online, local)
        self.assertTrue(checks["beats_submission_best_macro"])
        self.assertFalse(all(checks.values()))

    def test_class1_regression_blocks_a_macro_win(self):
        online = {"macro_auc": .809, "per_class_auc": [.690] * 10}
        local = {"macro_auc": .8176, "per_class_auc": [.746] * 10}
        candidate = {"macro_auc": .819, "per_class_auc": [.740] * 10}
        checks = selection_checks(candidate, online, local)
        self.assertTrue(checks["beats_historical_local_best_macro"])
        self.assertFalse(checks["class1_not_below_historical_local_best"])
        self.assertFalse(all(checks.values()))
        candidate["per_class_auc"][0] = .746
        self.assertTrue(all(selection_checks(candidate, online, local).values()))
        candidate["macro_auc"] = .8176
        self.assertFalse(all(selection_checks(candidate, online, local).values()))

    def test_reference_requires_hash_report_labels_and_complete_coverage(self):
        labels = {(str(i), str(i)): [i % 2] * 10 for i in range(4)}
        predictions = {key: [float(target[0]) * .8 + .1] * 10 for key, target in labels.items()}
        counts = {key: 1 + int(key[0]) // 2 for key in labels}
        expected = score_probabilities(labels, predictions, counts)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, report_path = Path(tmp) / "valid.csv", Path(tmp) / "report.json"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_HEADER)
                for (subject, study), values in predictions.items():
                    write_prediction(writer, subject, study, values)
            identity = sha256_file(csv_path)
            report = dict(expected, complete=True, predictions_sha256=identity, manifest_sha256="manifest")
            report_path.write_text(json.dumps(report))
            arguments = dict(expected_sha=identity, expected_macro=1., manifest_sha="manifest")
            actual, scores, digest = _reference(csv_path, report_path, labels, counts, **arguments)
            self.assertEqual(set(actual), set(labels))
            self.assertEqual(scores["macro_auc"], 1.)
            self.assertEqual(digest, identity)
            with self.assertRaisesRegex(ValueError, "provenance"):
                _reference(csv_path, report_path, labels, counts, **dict(arguments, expected_sha="wrong"))
            with self.assertRaisesRegex(ValueError, "coverage"):
                _reference(csv_path, report_path, dict(list(labels.items())[:-1]), counts, **arguments)
            report["per_class_auc"][0] = .9
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "frozen scores"):
                _reference(csv_path, report_path, labels, counts, **arguments)


if __name__ == "__main__":
    unittest.main()
