import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_exploration


def review_candidate(auc, members, ap_auc=None, selection=None):
    if selection is None:
        selection = ('initialization_reference' if members == ['champion041']
                     else ['valid_selected_best'] * 2 if len(members) == 2 else 'valid_selected_best')
    ap_auc = auc - .01 if ap_auc is None else ap_auc
    return {'selection_kind': selection, 'members': members,
            'probability_weights': [.5, .5] if len(members) == 2 else [1.],
            'scores': {'all': {'macro_auc': auc, 'per_class_auc': [auc] * 10},
                       'ap_only': {'macro_auc': ap_auc, 'per_class_auc': [ap_auc] * 10}}}


def review_comparison():
    values = {'champion041': .80, 'res640': .79, 'res768': .805, 'res896': .81, 'roi': .80, 'mil': .82}
    return {'candidates': {name + '_hflip2': review_candidate(value, [name])
                           for name, value in values.items()}, 'comparisons': {}}


def add_pair(comparison, members, auc, ap_auc=None, reference='mil', ci=(.001, .007), ap_ci=(-.002, .006)):
    key = '_'.join(members) + '_equal_hflip2'
    comparison['candidates'][key] = review_candidate(auc, list(members), ap_auc=ap_auc)
    comparison['comparisons'][key + '_minus_' + reference + '_hflip2'] = {
        'slices': {'all': {'delta_macro': {'ci95': list(ci)}},
                   'ap_only': {'delta_macro': {'ci95': list(ap_ci)}}}}
    return key


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
        self.assertIn('Half (5/10)', summarize_exploration.resolution_direction(resolution(5))[1])
        self.assertIn('majority (6/10)', summarize_exploration.resolution_direction(resolution(6))[1])
        self.assertIsNone(summarize_exploration.resolution_direction({'resolution_sensitivity': []})[0])

    def test_actual_pair_compares_all_singles_and_requires_fixed_best_members(self):
        comparison = review_comparison()
        key = add_pair(comparison, ('res768', 'res896'), .824)
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertAlmostEqual(result['gain'], .004)
        self.assertEqual(result['best_single_name'], 'mil')
        self.assertEqual(result['paired_ci95'], [.001,.007])
        self.assertTrue(result['recommend_for_human_review'])
        comparison['candidates'][key]['probability_weights'] = [.6, .4]
        self.assertIsNone(summarize_exploration.best_dual_tiny_recommendation(comparison))
        comparison['candidates'][key]['probability_weights'] = [.5, .5]
        comparison['candidates']['res768_hflip2']['selection_kind'] = 'fixed_final_E3'
        self.assertIsNone(summarize_exploration.best_dual_tiny_recommendation(comparison))

    def test_both_registered_pairs_reported_and_ap_safe_pair_preferred(self):
        comparison = review_comparison()
        add_pair(comparison, ('res768', 'res896'), .83, ap_auc=.800)
        roi_key = add_pair(comparison, ('res768', 'roi'), .825, ap_auc=.813)
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertEqual(result['candidate'], roi_key)
        self.assertEqual(len(result['pair_assessments']), 2)
        bad = result['pair_assessments'][0]
        self.assertFalse(bad['recommend_for_human_review'])
        self.assertIn('AP-only point decline exceeds 0.003', bad['reasons'])
        self.assertEqual(summarize_exploration.final_local_recommendation(comparison, result)['candidate'], roi_key)

    def test_exact_margins_pass_without_rounding_near_misses_up(self):
        for auc, ap_auc, expected in ((.824, .807, True), (.8239996, .807, False),
                                      (.824, .8069996, False)):
            with self.subTest(auc=auc, ap_auc=ap_auc):
                comparison = review_comparison()
                add_pair(comparison, ('res768', 'roi'), auc, ap_auc=ap_auc, ap_ci=(-.009, .003))
                result = summarize_exploration.best_dual_tiny_recommendation(comparison)
                self.assertEqual(result['recommend_for_human_review'], expected)
                self.assertTrue(result['ap_only_ci_allows_decline_over_margin'])

    def test_ci_crossing_zero_and_missing_ap_ci_do_not_establish_reliable_gain(self):
        for ci in ((-.002, .01), (0., .01)):
            comparison = review_comparison()
            add_pair(comparison, ('res768', 'roi'), .825, ci=ci)
            result = summarize_exploration.best_dual_tiny_recommendation(comparison)
            self.assertTrue(result['passes_registered_local_gates'])
            self.assertFalse(result['recommend_for_human_review'])
        comparison = review_comparison()
        key = add_pair(comparison, ('res768', 'roi'), .825)
        del comparison['comparisons'][key + '_minus_mil_hflip2']['slices']['ap_only']
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertFalse(result['recommend_for_human_review'])
        self.assertIn('paired AP-only CI is unavailable', result['reasons'])

    def test_original_snapshot_prevents_recommending_internal_winner_after_regression(self):
        comparison = review_comparison()
        comparison['candidates']['champion041_hflip2'] = review_candidate(.84, ['champion041'])
        add_pair(comparison, ('res768', 'res896'), .83, reference='champion041', ci=(-.02, -.001))
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertEqual(result['best_single_name'], 'champion041')
        self.assertFalse(result['recommend_for_human_review'])
        final = summarize_exploration.final_local_recommendation(comparison, result)
        self.assertTrue(final['retain_original_baseline'])
        self.assertEqual(final['candidate'], 'champion041_hflip2')

    def test_incomplete_registered_single_set_is_not_recommended(self):
        comparison = review_comparison()
        add_pair(comparison, ('res768', 'roi'), .825)
        del comparison['candidates']['res640_hflip2']
        result = summarize_exploration.best_dual_tiny_recommendation(comparison)
        self.assertFalse(result['all_registered_singles_available'])
        self.assertFalse(result['recommend_for_human_review'])

    def test_final_suggestion_can_choose_single_with_paired_gain_over_original(self):
        comparison = review_comparison()
        comparison['comparisons']['champion041_hflip2_minus_mil_hflip2'] = {
            'slices': {'all': {'delta_macro': {'ci95': [-.03, -.01]}},
                       'ap_only': {'delta_macro': {'ci95': [-.03, -.01]}}}}
        final = summarize_exploration.final_local_recommendation(comparison, None)
        self.assertFalse(final['retain_original_baseline'])
        self.assertEqual(final['candidate'], 'mil_hflip2')
        self.assertEqual(final['assessment']['paired_ci95'], [.01, .03])

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
                'champion041_hflip2': candidate(.799, ['champion041'], selection='initialization_reference'),
                'res640_hflip2': candidate(.79, ['res640']),
                'res768_hflip2': candidate(.80, ['res768']),
                'res896_hflip2': candidate(.81, ['res896']),
                'roi_hflip2': candidate(.78, ['roi']),
                'mil_hflip2': candidate(.805, ['mil']),
                'res768_res896_equal_hflip2': candidate(.815, ['res768', 'res896'],
                                                          selection=['valid_selected_best', 'valid_selected_best']),
                'res768_roi_equal_hflip2': candidate(.812, ['res768', 'roi'],
                                                       selection=['valid_selected_best', 'valid_selected_best']),
                }}, stream)
            with resolution_best.open() as stream:
                best_payload = json.load(stream)
            best_payload['comparisons'] = {
                'res768_res896_equal_hflip2_minus_res896_hflip2':
                    {'slices':{'all':{'delta_macro':{'ci95':[.001,.009]}},
                              'ap_only':{'delta_macro':{'ci95':[-.004,.01]}}}},
                'res768_roi_equal_hflip2_minus_res896_hflip2':
                    {'slices':{'all':{'delta_macro':{'ci95':[-.002,.006]}},
                              'ap_only':{'delta_macro':{'ci95':[-.002,.006]}}}},
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
            self.assertIn('res768_res896_equal_hflip2: dual Macro-AUC=0.815000; best single across the original reference and five registered arms res896=0.810000; gain=+0.005000', markdown)
            self.assertIn('patient-paired 95% CI=[0.001, 0.009]', markdown)
            self.assertIn('registered local point gates: gain >=0.004 and AP-only delta >=-0.003', markdown)
            self.assertIn('One candidate for owner choice: **res768_res896_equal_hflip2**', markdown)
            self.assertIn('AP-only risk: the paired interval still includes a decline greater than 0.003', markdown)
            self.assertIn('Online performance is unknown', markdown)
            self.assertIn('report does not freeze members', markdown)
            self.assertNotIn('tiny_a_tiny_b', markdown)
            self.assertIn('Best-checkpoint comparison (supplementary)', markdown)
            self.assertIn('| mil | valid_selected_best | 0.805000 | 0.795000 |', markdown)
            self.assertIn('| champion041 | initialization_reference | 0.799000 | 0.789000 |', markdown)
            # The metrics table remains contiguous instead of being split by per-class prose.
            self.assertIn('| res640 | valid_selected_best | 0.790000 | 0.780000 | not measured |\n'
                          '| res768 | valid_selected_best | 0.800000 | 0.790000 | 61.00/81.00 |', markdown)
            self.assertIn('compare_final', markdown)
            self.assertIn('c' * 64, markdown)

    def test_report_explicitly_retains_original_when_no_comparison_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'report.md'
            plan = root/'plan.json'
            plan.write_text(json.dumps({'status_file': str(root/'missing-status'),
                'roi_gallery': str(root/'missing-gallery'), 'report_file': str(output)}))
            with patch('sys.argv', ['summarize_exploration.py', '--plan', str(plan)]):
                summarize_exploration.main()
            markdown = output.read_text()
            self.assertIn('retain the original champion041 baseline', markdown)
            self.assertIn('do not claim a stronger model from this round', markdown)


if __name__ == '__main__':
    unittest.main()
