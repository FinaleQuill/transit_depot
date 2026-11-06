"""CPU tests for replayable patience-three decisions and exact policy binding."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from reconstruction import early_stop_policy as policy


def history(scores):
    return [{'epoch': i, 'macro_auc': score} for i, score in enumerate(scores, 1)]


class PolicyTests(unittest.TestCase):
    def test_baseline_does_not_count_and_three_bad_epochs_stop(self):
        self.assertEqual(policy.decision(.8, [], policy.POLICY)['bad_epochs'], 0)
        two = policy.decision(.8, history([.79, .78]), policy.POLICY)
        self.assertFalse(two['stop'])
        three = policy.decision(.8, history([.79, .78, .8]), policy.POLICY)
        self.assertEqual((three['bad_epochs'], three['first_stop_epoch']), (3, 3))
        self.assertEqual(three['stop_reason'], 'patience_exhausted')
        self.assertEqual((three['best_epoch'], three['best_macro_auc']), (0, .8))

    def test_exact_delta_does_not_reset_but_accumulated_improvement_does(self):
        baseline, delta = .8, policy.POLICY['min_delta']
        exact = policy.decision(baseline, history([baseline + delta]), policy.POLICY)
        self.assertEqual(exact['bad_epochs'], 1)
        self.assertEqual(exact['best_epoch'], 1)
        self.assertEqual(exact['significant_best_epoch'], 0)
        cumulative = policy.decision(baseline, history([.80006, .80012, .80015]), policy.POLICY)
        self.assertEqual(cumulative['bad_epochs'], 1)
        self.assertEqual(cumulative['significant_best_epoch'], 2)
        self.assertEqual(cumulative['best_epoch'], 3)
        self.assertFalse(cumulative['stop'])

    def test_complete_history_replay_never_resets_patience_on_resume(self):
        first = history([.79, .78])
        self.assertEqual(policy.decision(.8, first, policy.POLICY)['bad_epochs'], 2)
        resumed = policy.decision(.8, first + [{'epoch': 3, 'macro_auc': .77}], policy.POLICY)
        self.assertTrue(resumed['stop'])
        with self.assertRaisesRegex(ValueError, 'exceeds first policy stop'):
            policy.decision(.8, history([.79, .78, .77, .9]), policy.POLICY)

    def test_improvement_resets_and_maximum_six_is_binding(self):
        decision = policy.decision(.8, history([.79, .81, .80, .82, .83, .84]), policy.POLICY)
        self.assertTrue(decision['stop'])
        self.assertEqual(decision['stop_reason'], 'max_epochs_reached')
        self.assertEqual(decision['first_stop_epoch'], 6)
        self.assertEqual(decision['bad_epochs'], 0)
        with self.assertRaises(ValueError):
            policy.decision(.8, history([.81] * 7), policy.POLICY)

    def test_missing_epochs_nonfinite_and_unregistered_options_fail(self):
        for rows in [[{'epoch': 2, 'macro_auc': .8}], [{'epoch': True, 'macro_auc': .8}],
                     history([float('nan')]), history([None]), history([True]), history([1.1])]:
            with self.assertRaises(ValueError):
                policy.decision(.8, rows, policy.POLICY)
        for changes in [{'patience': 5}, {'patience': True}, {'min_delta': 0.0}, {'max_epochs': 3},
                        {'applies_to_arms': ['global_local']}, {'extra': 'unregistered'}]:
            with self.assertRaises(ValueError):
                policy.validate_policy({**deepcopy(policy.POLICY), **changes})

    def test_policy_is_bound_to_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.json'
            path.write_text(json.dumps(policy.POLICY))
            binding = policy.load_policy(path)
            self.assertEqual(binding['sha256'], policy.sha(path))
            policy.verify_binding(binding)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                policy.verify_binding(binding)


if __name__ == '__main__':
    unittest.main()
