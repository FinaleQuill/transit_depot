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

    def test_resolution_direction_uses_zero_four_and_majority_cutoffs(self):
        def resolution(count):
            return {'resolution_sensitivity': [
                {'class_id': index + 1, 'train896_benefit_at_infer768': index < count,
                 'train640_noninferior_at_infer768_margin_0_01': False}
                for index in range(10)]}

        self.assertIn('No class is supported', summarize_exploration.resolution_direction(resolution(0))[1])
        self.assertIn('Only 4/10 classes', summarize_exploration.resolution_direction(resolution(4))[1])
        self.assertIn('majority (5/10)', summarize_exploration.resolution_direction(resolution(5))[1])
        self.assertIsNone(summarize_exploration.resolution_direction({'resolution_sensitivity': []})[0])

    def test_dual_tiny_review_threshold_requires_best_selection_and_fixed_members(self):
        def item(auc, members, selection='valid_selected_best'):
            return {'selection_kind':selection, 'members':members, 'probability_weights':[.5, .5],
                    'scores':{'all':{'macro_auc':auc}}}
        comparison = {'candidates': {
            'tiny_a_hflip2': item(.80, ['tiny_a']),
            'tiny_b_hflip2': item(.81, ['tiny_b']),
            'tiny_a_tiny_b_equal_hflip2': item(.814, ['tiny_a','tiny_b']),
        }, 'comparisons': {
            'tiny_a_tiny_b_equal_hflip2_minus_tiny_b_hflip2':
                {'slices':{'all':{'delta_macro':{'ci95':[.001,.007]}}}},
        }}
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertAlmostEqual(result['gain'], .004)
        self.assertEqual(result['best_single_name'], 'tiny_b')
        self.assertEqual(result['paired_ci95'], [.001,.007])
        self.assertTrue(result['recommend_for_human_review'])
        comparison['candidates']['tiny_a_tiny_b_equal_hflip2']['probability_weights'] = [.6, .4]
        self.assertIsNone(summarize_exploration.best_dual_tiny_recommendation(comparison))
        comparison['candidates']['tiny_a_tiny_b_equal_hflip2']['probability_weights'] = [.5, .5]
        comparison['candidates']['tiny_a_hflip2']['selection_kind'] = 'fixed_final_E3'
        self.assertIsNone(summarize_exploration.best_dual_tiny_recommendation(comparison))

    def test_summary_prefers_resolution_labels_uses_timing_fallback_and_reviews_best_dual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def scores(value):
                return {'all': {'macro_auc': value, 'per_class_auc': [value] * 10},
                        'ap_only': {'macro_auc': value - .01, 'per_class_auc': [value - .01] * 10}}
            def candidate(value, members, selection='valid_selected_best', timing=None):
                return {'scores': scores(value), 'members': members, 'selection_kind': selection,
                        'probability_weights': [.5, .5] if len(members) == 2 else [1.], 'timing': timing}
            final = root/'final.json'
            with final.open('w') as stream:
                json.dump({'candidates': {
                'res768_hflip2': candidate(.80, ['res768']),
                'res896_hflip2': candidate(.81, ['res896']),
                'res768_res896_equal_hflip2': candidate(.815, ['res768', 'res896'],
                                                          selection=['valid_selected_best', 'valid_selected_best']),
                'res768_roi_equal_hflip2': candidate(.812, ['res768', 'roi'],
                                                       selection=['fixed_final_E3', 'fixed_final_E3']),
            }, 'comparisons': {
                'res896_hflip2_minus_res768_hflip2': {'slices': {
                    'all': {'macro_auc_delta': .01, 'delta_macro': {'ci95': [.0, .02]},
                            'per_class': [{'delta': .01, 'ci95': [.0, .02]}] * 10},
                    'ap_only': {'macro_auc_delta': .01, 'delta_macro': {'ci95': [.0, .02]},
                                'per_class': [{'delta': .01, 'ci95': [.0, .02]}] * 10}}},
                }}, stream)
            resolution = root/'resolution.json'
            with resolution.open('w') as stream:
                json.dump({'candidates': {
                'res768_hflip2': {'timing': {'p50_ms': 61., 'p95_ms': 81., 'observations': 90}},
                'res768_res896_equal_hflip2': {'timing': {'p50_ms': 95., 'p95_ms': 120., 'observations': 90}},
                }, 'resolution_sensitivity': [], 'comparisons': {}}, stream)
            roi_report = root/'roi.json'
            with roi_report.open('w') as stream:
                json.dump({'candidates': {
                    'res768_roi_equal_hflip2': {'timing': {'p50_ms': 88., 'p95_ms': 99., 'observations': 90}},
                }, 'comparisons': {}}, stream)
            resolution_best = root/'resolution_best.json'
            with resolution_best.open('w') as stream:
                json.dump({'candidates': {
                'res640_hflip2': candidate(.79, ['res640']),
                'res768_hflip2': candidate(.80, ['res768']),
                'res896_hflip2': candidate(.81, ['res896']),
                'roi_hflip2': candidate(.78, ['roi']),
                'mil_hflip2': candidate(.805, ['mil']),
                'tiny_a_hflip2': candidate(.80, ['tiny_a']),
                'tiny_b_hflip2': candidate(.81, ['tiny_b']),
                'tiny_a_tiny_b_equal_hflip2': {
                    **candidate(.815, ['tiny_a', 'tiny_b']),
                    'probability_weights': [.5, .5]},
                'res768_res896_equal_hflip2': candidate(.815, ['res768', 'res896'],
                                                          selection=['valid_selected_best', 'valid_selected_best']),
                }}, stream)
            with resolution_best.open() as stream:
                best_payload = json.load(stream)
            best_payload['comparisons'] = {
                'tiny_a_tiny_b_equal_hflip2_minus_tiny_b_hflip2':
                    {'slices':{'all':{'delta_macro':{'ci95':[.001,.009]}}}}
            }
            with resolution_best.open('w') as stream:
                json.dump(best_payload, stream)
            status_path = root/'STATUS.json'
            status_path.write_text(json.dumps({'state': 'COMPLETED', 'current_job': 'report', 'phase': 'done',
                'estimated_finish_utc': None, 'failure_reason': None, 'compliance_records': {
                    'compare_final': {'receipt_path': '/tmp/receipt.json', 'preregistration_sha256': 'c' * 64,
                                      'run_ledger_sha256': 'd' * 64, 'receipt_sha256': 'e' * 64}}}))
            output = root/'morning.md'
            plan = root/'plan.json'
            plan.write_text(json.dumps({'status_file': str(status_path), 'roi_gallery': str(root/'missing'),
                'evaluation_reports': {}, 'comparison_reports': {
                    'final': str(final), 'resolution': str(resolution), 'roi': str(roi_report),
                    'resolution_best': str(resolution_best)},
                'report_file': str(output)}))
            with patch('sys.argv', ['summarize_exploration.py', '--plan', str(plan)]):
                summarize_exploration.main()
            markdown = output.read_text()
            self.assertIn('| res768 | 0.800000 | 0.790000 | 61.00/81.00 ms |', markdown)
            self.assertIn('| res896 | 0.810000 | 0.800000 | not measured |', markdown)
            self.assertIn('| res768_res896_equal | 0.815000 | 0.805000 | 95.00/120.00 ms |', markdown)
            self.assertIn('| res768_roi_equal | 0.812000 | 0.802000 | 88.00/99.00 ms |', markdown)
            self.assertIn('tiny_a_tiny_b_equal_hflip2: dual Macro-AUC=0.815000; best matched single tiny_b=0.810000; gain=+0.005000', markdown)
            self.assertIn('patient-paired 95% CI=[0.001, 0.009]', markdown)
            self.assertIn('fixed >=0.004 local threshold for human final-member review', markdown)
            self.assertIn('Best-checkpoint comparison (supplementary)', markdown)
            self.assertIn('| mil | valid_selected_best | 0.805000 | 0.795000 |', markdown)
            self.assertIn('compare_final', markdown)
            self.assertIn('c' * 64, markdown)


if __name__ == '__main__':
    unittest.main()
