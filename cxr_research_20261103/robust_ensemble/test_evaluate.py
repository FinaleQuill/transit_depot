from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from deploy_kd import IMAGE_CSV_HEADER
from robust_ensemble.evaluate import image_cache, sensitivity, uncertainty, weighted_auc_many, selection_checks
from sprint.evaluate import _write_json
from student_experiment_evaluation import score_probabilities


class EvaluationTests(unittest.TestCase):
    def test_numpy_prediction_decisions_serialize_through_report_writer(self):
        keys = [('p' + str(i // 2), str(i)) for i in range(8)]
        labels = {key: [i % 2] * 10 for i, key in enumerate(keys)}
        old = {key: np.full(10, (i * 3 % 8) / 8., dtype=np.float64)
               for i, key in enumerate(keys)}
        new = {key: value.copy() for key, value in old.items()}
        counts = {key: 1 if i < 4 else 2 for i, key in enumerate(keys)}
        records = [{'subject_id': key[0], 'study_id': key[1], 'view_position': 'AP'}
                   for key in keys]
        score = score_probabilities(labels, old, counts)
        sensitive = sensitivity(labels, old, counts, records)
        for change in (False, True):
            if change:
                new[keys[0]][0] = np.nextafter(new[keys[0]][0], np.inf)
            # Match the original failing path: max over NumPy scalar differences.
            difference = max(abs(new[k][0] - old[k][0]) for k in keys)
            self.assertIsInstance(difference, np.floating)
            checks = selection_checks(score_probabilities(labels, new, counts), score,
                                      sensitivity(labels, new, counts, records), sensitive, difference)
            self.assertTrue(all(type(value) is bool for value in checks.values()))
            self.assertEqual(checks['class1_exactly_preserved'], not change)
            report = {'candidate': score, 'selection_checks': checks,
                      'candidate_won': all(checks.values()),
                      'class1_max_abs_probability_difference': float(difference)}
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'report.json'
                _write_json(path, report)
                self.assertEqual(json.loads(path.read_text()), report)

    def test_patient_equal_slices_recompute_patient_counts_within_slice(self):
        # Patient a has more total Studies but only one single-image Study.
        # All three single-image patients must therefore have equal weight.
        keys = [('a', '1'), ('a', '2'), ('a', '3'), ('b', '1'), ('c', '1')]
        labels = {key: [target] * 10 for key, target in zip(keys, (1, 1, 0, 0, 1))}
        predictions = {key: [value] * 10 for key, value in zip(keys, (.2, .9, .8, .5, .7))}
        counts = {key: count for key, count in zip(keys, (1, 2, 2, 1, 1))}
        records = [{'subject_id': key[0], 'study_id': key[1], 'view_position': 'AP'}
                   for key in keys]
        result = sensitivity(labels, predictions, counts, records)
        self.assertAlmostEqual(result['patient_equal_single_image']['macro_auc'], .5)
        self.assertNotAlmostEqual(result['patient_equal_single_image']['macro_auc'], .75)
        expected = roc_auc_score([1, 1, 0, 0, 1], [.2, .9, .8, .5, .7],
                                 sample_weight=[1/3, 1/3, 1/3, 1, 1])
        self.assertAlmostEqual(result['patient_equal']['macro_auc'], expected)

    def test_cluster_auc_matches_sklearn_with_ties_and_zero_weight(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([.1, .2, .2, .2, .8, .9])
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 0, 2, 3, 1, 0], [0, 1, 0, 1, 1, 0]], float)
        actual = weighted_auc_many(y, p, weights)
        for i in range(2):
            self.assertAlmostEqual(actual[i], roc_auc_score(y, p, sample_weight=weights[i]), places=14)
        self.assertTrue(np.isnan(actual[2]))

    def test_overall_gain_cannot_hide_single_image_regression(self):
        old = dict(macro_auc=.81, per_class_auc=[.8]*10,
                   slices={'single_image_study': {'macro_auc': .75}})
        new = dict(macro_auc=.82, per_class_auc=[.81]*10,
                   slices={'single_image_study': {'macro_auc': .749}})
        sensitive = {'patient_equal': {'macro_auc': .85}}
        checks = selection_checks(new, old, sensitive, sensitive, 0.)
        self.assertTrue(checks['macro_strictly_better'])
        self.assertFalse(checks['single_image_macro_not_worse'])
        self.assertFalse(all(checks.values()))

    def test_each_frozen_acceptance_gate_is_required(self):
        old = dict(macro_auc=.8, per_class_auc=[.8]*10,
                   slices={'single_image_study': {'macro_auc': .75}})
        new = deepcopy(old)
        new['macro_auc'] = .801
        new['per_class_auc'][7] = .81
        new['slices']['single_image_study']['macro_auc'] = .751
        old_sensitive = {'patient_equal': {'macro_auc': .85}}
        new_sensitive = {'patient_equal': {'macro_auc': .851}}
        self.assertTrue(all(selection_checks(new, old, new_sensitive, old_sensitive, 0.).values()))
        for field in ('macro', 'class8', 'single', 'balanced', 'class1'):
            candidate, sensitive = deepcopy(new), deepcopy(new_sensitive)
            class1_difference = 0.
            if field == 'macro':
                candidate['macro_auc'] = old['macro_auc']
            elif field == 'class8':
                candidate['per_class_auc'][7] = old['per_class_auc'][7]
            elif field == 'single':
                candidate['slices']['single_image_study']['macro_auc'] = .75 - 1e-9
            elif field == 'balanced':
                sensitive['patient_equal']['macro_auc'] = .85 - 1e-9
            else:
                class1_difference = 1e-16
            with self.subTest(field=field):
                self.assertFalse(all(selection_checks(candidate, old, sensitive,
                                                      old_sensitive, class1_difference).values()))

    def test_image_cache_aligns_identity_and_rejects_duplicate_missing_or_wrong_patient(self):
        records = [{'image_id': 'image-b', 'subject_id': 'patient-b', 'study_id': 'study-b'},
                   {'image_id': 'image-a', 'subject_id': 'patient-a', 'study_id': 'study-a'}]
        rows = [[r['image_id'], r['subject_id'], r['study_id'], c, c / 11.]
                for r in records for c in range(1, 11)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'images.csv'
            def write(values):
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(IMAGE_CSV_HEADER)
                    writer.writerows(values)
            write(list(reversed(rows)))
            result = image_cache(path, records)
            np.testing.assert_array_equal(result['image-a'], np.arange(1, 11) / 11.)
            wrong_patient = deepcopy(rows)
            wrong_patient[0][1] = 'wrong-patient'
            for malformed in (rows + [rows[0]], rows[:-1], wrong_patient):
                write(malformed)
                with self.assertRaises(ValueError):
                    image_cache(path, records)

    def test_cluster_bootstrap_intervals_match_independent_pair_enumeration(self):
        patient_names = ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd']
        keys = [(patient, f'{i:02d}') for i, patient in enumerate(patient_names)]
        labels = {key: [int((i + c) % 3 == 0) for c in range(10)] for i, key in enumerate(keys)}
        old = {key: [((i * 7 + c * 3) % 13) / 12. for c in range(10)] for i, key in enumerate(keys)}
        new = {key: [((i * 5 + c * 2) % 11) / 10. for c in range(10)] for i, key in enumerate(keys)}
        counts = {key: 1 if i % 2 else 2 for i, key in enumerate(keys)}
        samples, seed = 80, 915
        actual = uncertainty(labels, new, old, counts, samples=samples, seed=seed)
        patients = sorted(set(patient_names))
        frequencies = Counter(patient_names)
        draws = np.random.default_rng(seed).multinomial(len(patients),
                    np.full(len(patients), 1 / len(patients)), size=samples)

        def independent_auc(subset, probabilities, weights, cls):
            numerator = denominator = 0.
            for i in subset:
                if labels[keys[i]][cls] != 1:
                    continue
                for j in subset:
                    if labels[keys[j]][cls] != 0:
                        continue
                    pair_weight = weights[i] * weights[j]
                    left, right = probabilities[keys[i]][cls], probabilities[keys[j]][cls]
                    numerator += pair_weight * (1. if left > right else .5 if left == right else 0.)
                    denominator += pair_weight
            return numerator / denominator if denominator else np.nan

        full, single, balanced = [], [], []
        all_indices = list(range(len(keys)))
        single_indices = [i for i, key in enumerate(keys) if counts[key] == 1]
        for draw in draws:
            weights = np.asarray([draw[patients.index(k[0])] for k in keys], dtype=float)
            balanced_weights = weights / np.asarray([frequencies[k[0]] for k in keys])
            for destination, subset, w in ((full, all_indices, weights),
                                            (single, single_indices, weights),
                                            (balanced, all_indices, balanced_weights)):
                destination.append([independent_auc(subset, new, w, c) -
                                    independent_auc(subset, old, w, c) for c in range(10)])
        full, single, balanced = map(np.asarray, (full, single, balanced))
        expected = {'macro': full.mean(1), 'class8': full[:, 7],
                    'single_image_macro': single.mean(1), 'patient_equal_macro': balanced.mean(1)}
        for name, values in expected.items():
            finite = values[np.isfinite(values)]
            self.assertEqual(actual[name]['valid_samples'], len(finite))
            np.testing.assert_allclose(actual[name]['delta_95_percentile_interval'],
                                       np.quantile(finite, [.025, .975]), atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
