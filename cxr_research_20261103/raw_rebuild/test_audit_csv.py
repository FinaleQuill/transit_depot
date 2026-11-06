"""Synthetic completeness, identity and serialization rejection checks."""
import csv
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from raw_rebuild.audit_csv import HEADER, audit_csv, main


class AuditCSVTests(unittest.TestCase):
    def fixture(self, directory):
        test = directory / 'test.csv'
        submission = directory / 'submission.csv'
        predictions = directory / 'predictions.npz'
        with test.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Dicom_id', 'Subject_id', 'Study_id'])
            writer.writerows([['pic_001', 'subject_001', 'study_001'],
                              ['pic_002', 'subject_001', 'study_001'],
                              ['pic_003', 'subject_002', 'study_002']])
        values = np.arange(20, dtype=np.float64).reshape(2, 10) / 23.
        rows = [[f'subject_{i+1:03}', f'study_{i+1:03}', str(c+1), repr(float(values[i, c]))]
                for i in range(2) for c in range(10)]
        self.write(submission, rows)
        np.savez(predictions, probabilities=values[::-1],
                 study_keys=np.asarray(['subject_002/study_002', 'subject_001/study_001']),
                 patient_ids=np.asarray(['subject_002', 'subject_001']),
                 class_ids=np.arange(1, 11))
        return test, submission, predictions, rows

    def write(self, path, rows, header=HEADER):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def audit(self, test, submission, predictions=None):
        return audit_csv(submission, test, predictions, expected_images=3, expected_studies=2)

    def test_exact_values_align_by_string_study_identity_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            test, submission, predictions, rows = self.fixture(Path(tmp))
            self.write(submission, rows[::-1])
            before = {p: p.read_bytes() for p in (test, submission, predictions)}
            report = self.audit(test, submission, predictions)
            self.assertTrue(report['passed'])
            self.assertEqual((report['images'], report['studies'], report['submission_rows']), (3, 2, 20))
            self.assertEqual(report['study_prediction_agreement']['max_absolute_error'], 0.)
            self.assertTrue(report['study_prediction_agreement']['exact_float64_values'])
            for entry in report['inputs'].values():
                path = Path(entry['path'])
                self.assertEqual(entry['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(entry['bytes'], path.stat().st_size)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse(self.audit(test, submission)['study_prediction_agreement']['checked'])

    def test_probabilities_accept_endpoints_and_at_most_one_trillionth_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            test, submission, predictions, rows = self.fixture(Path(tmp))
            values = np.asarray([[float(row[3]) for row in rows[i:i+10]] for i in (0, 10)])
            values[0, 1] = 1.
            rows[1][3] = '1.0'
            np.savez(predictions, probabilities=values,
                     study_keys=np.asarray(['subject_001/study_001', 'subject_002/study_002']),
                     patient_ids=np.asarray(['subject_001', 'subject_002']))
            rows[0][3] = '1e-12'
            self.write(submission, rows)
            report = self.audit(test, submission, predictions)
            self.assertEqual(report['study_prediction_agreement']['max_absolute_error'], 1e-12)
            self.assertFalse(report['study_prediction_agreement']['exact_float64_values'])
            rows[0][3] = repr(float(np.nextafter(1e-12, np.inf)))
            self.write(submission, rows)
            with self.assertRaisesRegex(ValueError, 'probability mismatch'):
                self.audit(test, submission, predictions)

    def test_tolerance_cannot_be_invalid_or_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            test, submission, predictions, _ = self.fixture(Path(tmp))
            for tolerance in (-1., float('nan'), float('inf'), 1e-11, True, None, '1e-12'):
                with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, 'tolerance'):
                    audit_csv(submission, test, predictions, expected_images=3, expected_studies=2,
                              absolute_tolerance=tolerance)
            report = audit_csv(submission, test, predictions, expected_images=3, expected_studies=2,
                               absolute_tolerance=0.)
            self.assertTrue(report['study_prediction_agreement']['exact_float64_values'])

    def test_rejects_missing_class_missing_study_duplicate_and_wrong_patient(self):
        for fault in ('missing_class', 'missing_study', 'duplicate', 'wrong_patient', 'numeric_id'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                test, submission, _, rows = self.fixture(Path(tmp))
                if fault == 'missing_class':
                    rows.pop()
                elif fault == 'missing_study':
                    rows = rows[:10]
                elif fault == 'duplicate':
                    rows.append(rows[0])
                elif fault == 'wrong_patient':
                    rows[0][0] = 'subject_002'
                else:
                    rows[0][1] = '1'
                self.write(submission, rows)
                with self.assertRaises(ValueError):
                    self.audit(test, submission)

    def test_rejects_invalid_probabilities_and_noncanonical_classes(self):
        for value in ('NaN', 'nan', 'inf', '-inf', '1.00001', '-0.00001', '', 'bad'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                test, submission, _, rows = self.fixture(Path(tmp))
                rows[0][3] = value
                self.write(submission, rows)
                with self.assertRaises(ValueError):
                    self.audit(test, submission)
        for category in ('0', '11', '01', '1.0', ' 1'):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                test, submission, _, rows = self.fixture(Path(tmp))
                rows[0][2] = category
                self.write(submission, rows)
                with self.assertRaises(ValueError):
                    self.audit(test, submission)

    def test_rejects_extra_columns_and_npz_probability_or_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            test, submission, predictions, rows = self.fixture(Path(tmp))
            self.write(submission, [row + ['0.5'] for row in rows], header=HEADER + ['Gate'])
            with self.assertRaisesRegex(ValueError, 'four-column'):
                self.audit(test, submission)
            self.write(submission, rows)
            rows[0][3] = '0.000001'
            self.write(submission, rows)
            with self.assertRaisesRegex(ValueError, 'probability mismatch'):
                self.audit(test, submission, predictions)
            rows[0][3] = '0.0'
            self.write(submission, rows)
            np.savez(predictions, probabilities=np.zeros((2, 10)),
                     study_keys=np.asarray(['subject_001/study_001'] * 2),
                     patient_ids=np.asarray(['subject_001'] * 2))
            with self.assertRaisesRegex(ValueError, 'repeats a Study'):
                self.audit(test, submission, predictions)

    def test_rejects_malformed_submission_headers_and_rows(self):
        for fault in ('column_order', 'duplicate_header', 'short_row', 'long_row', 'empty'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                test, submission, _, rows = self.fixture(Path(tmp))
                header = HEADER.copy()
                if fault == 'column_order':
                    header[:2] = header[1::-1]
                elif fault == 'duplicate_header':
                    header[1] = header[0]
                elif fault == 'short_row':
                    rows[0].pop()
                elif fault == 'long_row':
                    rows[0].append('extra')
                else:
                    rows = []
                self.write(submission, rows, header=header)
                with self.assertRaises(ValueError):
                    self.audit(test, submission)

    def test_rejects_invalid_official_identity_manifests(self):
        for fault in ('duplicate_image', 'conflicting_patient', 'missing_header',
                      'duplicate_header', 'short_row', 'long_row', 'bad_identity', 'empty',
                      'wrong_counts'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                test, submission, _, _ = self.fixture(Path(tmp))
                with test.open(newline='') as stream:
                    header, *rows = list(csv.reader(stream))
                if fault == 'duplicate_image':
                    rows[1][0] = rows[0][0]
                elif fault == 'conflicting_patient':
                    rows[1][1] = 'subject_999'
                elif fault == 'missing_header':
                    header[0] = 'Image_id'
                elif fault == 'duplicate_header':
                    header[0] = 'Study_id'
                elif fault == 'short_row':
                    rows[0].pop()
                elif fault == 'long_row':
                    rows[0].append('extra')
                elif fault == 'bad_identity':
                    rows[0][1] = ' subject_001'
                elif fault == 'empty':
                    rows = []
                else:
                    rows.pop(0)
                self.write(test, rows, header=header)
                with self.assertRaises(ValueError):
                    self.audit(test, submission)

    def test_rejects_invalid_npz_arrays_and_identities(self):
        faults = ('missing_array', 'probability_shape', 'key_shape', 'patient_shape',
                  'non_numeric', 'nan', 'inf', 'negative', 'above_one', 'byte_strings',
                  'object_strings', 'class_order', 'key_patient_mismatch', 'invalid_key',
                  'unknown_study', 'bad_patient')
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                test, submission, predictions, _ = self.fixture(Path(tmp))
                with np.load(predictions, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                if fault == 'missing_array':
                    del arrays['patient_ids']
                elif fault == 'probability_shape':
                    arrays['probabilities'] = arrays['probabilities'][:, :-1]
                elif fault == 'key_shape':
                    arrays['study_keys'] = arrays['study_keys'][:, None]
                elif fault == 'patient_shape':
                    arrays['patient_ids'] = arrays['patient_ids'][:1]
                elif fault == 'non_numeric':
                    arrays['probabilities'] = arrays['probabilities'].astype(str)
                elif fault in ('nan', 'inf', 'negative', 'above_one'):
                    arrays['probabilities'][0, 0] = {
                        'nan': np.nan, 'inf': np.inf, 'negative': -1e-10, 'above_one': 1.01}[fault]
                elif fault == 'byte_strings':
                    arrays['study_keys'] = arrays['study_keys'].astype('S')
                elif fault == 'object_strings':
                    arrays['patient_ids'] = arrays['patient_ids'].astype(object)
                elif fault == 'class_order':
                    arrays['class_ids'] = arrays['class_ids'][::-1]
                elif fault == 'key_patient_mismatch':
                    arrays['patient_ids'] = arrays['patient_ids'][::-1]
                elif fault == 'invalid_key':
                    arrays['study_keys'] = np.asarray(['subject_002/study_002/x',
                                                      'subject_001/study_001'])
                elif fault == 'unknown_study':
                    arrays['study_keys'][0] = 'subject_002/study_999'
                else:
                    arrays['patient_ids'][0] = ' subject_002'
                np.savez(predictions, **arrays)
                with self.assertRaises(ValueError):
                    self.audit(test, submission, predictions)

    def test_cli_writes_json_counts_and_hashes_for_complete_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test, submission = root / 'test.csv', root / 'submission.csv'
            report_path = root / 'audit' / 'report.json'
            official_rows = [[f'pic_{i}', f'subject_{i % 1414}', f'study_{i % 1414}']
                             for i in range(2218)]
            self.write(test, official_rows, header=['Dicom_id', 'Subject_id', 'Study_id'])
            rows = [[f'subject_{i}', f'study_{i}', str(c), '0.5']
                    for i in range(1414) for c in range(1, 11)]
            self.write(submission, rows)
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(['--csv', str(submission), '--test-csv', str(test), '--output', str(report_path)])
            report = json.loads(report_path.read_text())
            self.assertEqual(report, json.loads(stdout.getvalue()))
            self.assertEqual((report['images'], report['studies'], report['submission_rows']),
                             (2218, 1414, 14140))
            self.assertTrue(report['passed'])
            self.assertEqual(report['inputs']['submission']['sha256'],
                             hashlib.sha256(submission.read_bytes()).hexdigest())

    def test_cli_cannot_overwrite_csv_or_an_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test, submission, _, _ = self.fixture(root)
            report = root / 'report.json'
            report.write_text('preserve')
            before = submission.read_bytes()
            for output in (submission, test, report):
                with self.assertRaises(FileExistsError):
                    main(['--csv', str(submission), '--test-csv', str(test), '--output', str(output)])
            self.assertEqual(submission.read_bytes(), before)
            self.assertEqual(report.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
