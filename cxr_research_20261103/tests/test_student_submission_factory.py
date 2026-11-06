"""Check the common writer retains all images, Study means and failed-output safety."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import predict
from submit_student_candidate import SubmissionStudentPredictor


class StudentSubmissionFactoryTests(unittest.TestCase):
    def test_injected_student_uses_shared_writer_and_exact_study_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            records = [{'image_id': name, 'subject_id': 'subject_1', 'study_id': 'study_1',
                        'study_key': 'subject_1/study_1', 'path': root / name}
                       for name in ('a.jpg', 'b.jpg')]
            values = [np.arange(10) / 20, np.arange(10) / 20 + .2]

            class FakeStudent:
                def __init__(self, directory):
                    self.config = {'schema_version': 1, 'tta': 'hflip2', 'ensemble': False,
                                   'image_size': 768}
                    self.inventory = {'size_pass': True}
                    self.inputs = []

                def predict_image(self, path):
                    self.inputs.append(path)
                    return values[len(self.inputs) - 1]

            output = root / 'out/submission.csv'
            with patch.object(predict, 'prediction_records', return_value=records), \
                    patch.object(predict, 'configure_inference', return_value={'seed': 43}), \
                    patch.object(predict, 'Predictor', side_effect=AssertionError('Wrong loader')):
                result = predict.predict_submission(model_dir=model, image_root=root / 'images',
                    output=output, predictor_factory=FakeStudent)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            np.testing.assert_array_equal([float(r['Probability']) for r in rows],
                                          (values[0] + values[1]) / 2)
            self.assertEqual(result['images'], 2)
            self.assertEqual(result['data_rows'], 10)
            self.assertEqual(result['member_count'], 1)
            with output.with_suffix('.image_predictions.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 20)
            self.assertFalse(output.with_name('submission.csv.tmp').exists())

    def test_fixed_delivery_rejects_tta_change_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps({'tta': 'identity'}))
            with self.assertRaisesRegex(ValueError, 'fixed validated hflip2'):
                SubmissionStudentPredictor(root)


if __name__ == '__main__':
    unittest.main()
