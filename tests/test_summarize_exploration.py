import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_exploration


class ExplorationSummaryTests(unittest.TestCase):
    def test_report_reads_checkpoint_sha_from_metadata_and_ledger_from_compliance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root/'eval'
            evaluation.mkdir()
            report = {
                'complete': True,
                'scores': {'hflip2': {
                    'all': {'macro_auc': .81, 'per_class_auc': [.81] * 10},
                    'ap_only': {'macro_auc': .79, 'per_class_auc': [.79] * 10},
                }},
                'sampled_train_gap': {'num_studies':1000,
                    'sampled_train_minus_full_valid_macro_auc': .0123},
                'timing': {'modes': {'hflip2': {'p50_ms': 50., 'p95_ms': 75.}}},
            }
            (evaluation/'report.json').write_text(json.dumps(report))
            (evaluation/'metadata.json').write_text(json.dumps({
                'source_checkpoint_sha256': 'a' * 64,
                'selection_kind': 'fixed_final_E3',
                'training_compliance': {'ledger_sha256': 'b' * 64},
            }))
            def per_class(delta):
                return [{'class_id':index+1, 'delta':delta, 'ci95':[delta-.01,delta+.01]}
                        for index in range(10)]
            comparison = root/'resolution.json'
            comparison.write_text(json.dumps({
                'resolution_sensitivity':[{'class_id':1, 'status':'higher_training_resolution_supported',
                    'train896_benefit_at_infer768':True,
                    'train640_noninferior_at_infer768_margin_0_01':False}],
                'comparisons':{
                    'res896_hflip2_minus_res768_hflip2':{'slices':{
                        'all':{'macro_auc_delta':.02, 'delta_macro':{'ci95':[.01,.03]}, 'per_class':per_class(.02)},
                        'ap_only':{'per_class':per_class(.03)}}},
                    'res640_hflip2_minus_res768_hflip2':{'slices':{
                        'all':{'macro_auc_delta':-.01, 'delta_macro':{'ci95':[-.02,0.]}, 'per_class':per_class(-.01)},
                        'ap_only':{'per_class':per_class(-.02)}}},
                },
            }))
            status_path = root/'STATUS.json'
            status_path.write_text(json.dumps({'state':'COMPLETED', 'current_job':None,
                'phase':'done', 'estimated_finish_utc':None, 'failure_reason':None}))
            output = root/'morning.md'
            plan = root/'plan.json'
            plan.write_text(json.dumps({
                'status_file':str(status_path), 'roi_gallery':str(root/'missing-gallery'),
                'evaluation_reports':{'tiny_a':str(evaluation/'report.json')},
                'comparison_reports':{'resolution':str(comparison)}, 'report_file':str(output),
            }))
            with patch('sys.argv', ['summarize_exploration.py', '--plan', str(plan)]):
                summarize_exploration.main()
            markdown = output.read_text()
            self.assertIn('checkpoint SHA ' + 'a' * 64, markdown)
            self.assertIn('lifecycle ledger SHA ' + 'b' * 64, markdown)
            self.assertNotIn('checkpoint SHA unavailable', markdown)
            self.assertIn('fixed_final_E3', markdown)
            self.assertIn('1000 train Studies: 0.012300', markdown)
            self.assertIn('Enlarged Cardiomediastinum | +0.020000 [+0.010000, +0.030000]', markdown)
            self.assertIn('+0.030000 [+0.020000, +0.040000] | -0.010000 [-0.020000, +0.000000] | '
                          '-0.020000 [-0.030000, -0.010000]', markdown)
            self.assertIn('train896 benefit=True', markdown)


if __name__ == '__main__':
    unittest.main()
