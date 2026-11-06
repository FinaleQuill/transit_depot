"""Reject member/weight/view substitutions before loading or copying models."""
import copy
import json
from pathlib import Path
import unittest

from make_ensemble import registered_ensemble_rule


class RegisteredEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((Path(__file__).resolve().parents[1] /
            'docs/BOUNDED_STUDY_FINETUNE_20261101.json').read_text())
        self.kwargs = dict(rule_name='X1_B_class1_B', names=['X1', 'B'],
            ttas=['hflip2', 'hflip2'], weights=[[0.] + [.5]*9, [1.] + [.5]*9], decoder=None)

    def test_named_weights_follow_members_even_if_the_input_order_changes(self):
        result = registered_ensemble_rule(self.plan, **self.kwargs)
        self.assertEqual(result['expected_member_identities']['X1']['epoch'], 4)
        swapped = {**self.kwargs, 'names': ['B', 'X1'], 'weights': self.kwargs['weights'][::-1]}
        result = registered_ensemble_rule(self.plan, **swapped)
        self.assertEqual(result['normalized_weights'], swapped['weights'])
        with self.assertRaisesRegex(ValueError, 'weights differ'):
            registered_ensemble_rule(self.plan, **{**swapped, 'weights': self.kwargs['weights']})

    def test_unregistered_rule_member_view_or_decoder_is_rejected(self):
        for change in [dict(rule_name='new_sweep'), dict(names=['X2', 'B']),
                       dict(ttas=['identity', 'hflip2']), dict(decoder='JPEG_draft_target768')]:
            with self.assertRaises(ValueError):
                registered_ensemble_rule(self.plan, **{**self.kwargs, **change})
        registered_ensemble_rule(self.plan, **{**self.kwargs, 'decoder': 'JPEG_draft_target768',
                                              'allow_latency_fallback': True})

    def test_old_plan_has_no_implicit_draft_decoder_exception(self):
        old = json.loads((Path(__file__).resolve().parents[1] /
            'docs/auc820_20261031/registered_rules.json').read_text())
        kwargs = dict(rule_name='R1_B_B2_equal', names=['B', 'B2'], ttas=['hflip2']*2,
                      weights=[[.5]*10]*2, decoder='JPEG_draft_target768', allow_latency_fallback=True)
        with self.assertRaisesRegex(ValueError, 'Decoder differs'):
            registered_ensemble_rule(old, **kwargs)


if __name__ == '__main__':
    unittest.main()
