"""The two public CLIs share loader selection, including the draft JPEG path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cost
import predict
from submission_runtime import load_submission_predictor


class SubmissionRuntimeTests(unittest.TestCase):
    def test_both_clis_pass_the_identical_model_factory(self):
        with patch.object(predict, 'predict_submission', return_value={}) as prediction:
            predict.main(['--model-dir', '/model', '--image-root', '/images', '--output', '/out/submission.csv'])
        with patch.object(cost, 'run_cost', return_value={'official_compliance_pass': False}) as timing:
            self.assertEqual(cost.main(['/out/submission.csv', '/100', '/5',
                                        '--model-dir', '/model', '--output-dir', '/cost']), 2)
        self.assertIs(prediction.call_args.kwargs['predictor_factory'], load_submission_predictor)
        self.assertIs(timing.call_args.kwargs['predictor_factory'], load_submission_predictor)

    def test_declared_draft_decoder_cannot_silently_use_full_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'artifact_type': 'fixed_classwise_student_pair',
                      'decoder_candidate': 'JPEG_draft_target768'}
            (root / 'config.json').write_text(json.dumps(config))
            expected = object()
            with patch('student_draft_pair.DraftStudentPair', return_value=expected) as loader, \
                    patch('student_fixed_pair.FixedStudentPair', side_effect=AssertionError('Wrong pixels')):
                self.assertIs(load_submission_predictor(root), expected)
                loader.assert_called_once_with(root)
            config['decoder_candidate'] = 'unknown_pixel_transform'
            (root / 'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Unknown fixed-pair decoder'):
                load_submission_predictor(root)


if __name__ == '__main__':
    unittest.main()
