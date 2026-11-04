"""CPU-only tests for exact-state hashing and the bounded adoption gates."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from reconstruction import benchmark_checkpoint_speed as benchmark


def plan():
    return {'checkpoint_sha256': 'abc', 'profile_binding': {}, 'input_sha256': {},
        'effective_studies': 32, 'workload': {'study_count': 10000,
        'study_image_count_distribution': {'1': 7000, '2': 2000, '3': 900, '7': 100}},
        'baseline_training_epoch_seconds': 8000}


def report(small_multiplier=1.):
    return {'complete': True, 'outcome': 'valid', 'weights_sha256': 'same', 'shapes': {
        str(n): {'input_sha256': 'same', 'initial_rng_sha256': 'same', 'initial_state_sha256': 'same',
            'equivalence': {'skipped_update': False}, 'timing_final_rng_sha256': 'same',
            'median_forward_backward_seconds': n * (small_multiplier if n <= 2 else 1.),
            'median_optimizer_ema_seconds': .02,
            'estimated_min_device_headroom_bytes': 2 * 1024**3}
        for n in benchmark.VIEWS}}


class BenchmarkTests(unittest.TestCase):
    def test_profitable_exact_candidate_is_accepted(self):
        result = benchmark.compare(plan(), report(), report(.7), 120)
        self.assertTrue(result['adopt'])
        self.assertGreater(result['weighted_speedup'], 1.08)
        self.assertGreater(result['net_first_epoch_saving'], 300)

    def test_any_gradient_or_update_difference_is_rejected(self):
        candidate = report(.7)
        candidate['shapes']['1']['equivalence']['gradients_sha256'] = 'changed'
        result = benchmark.compare(plan(), report(), candidate, 120)
        self.assertFalse(result['adopt'])
        self.assertIn('exact_equivalence_failed', result['reason'])

    def test_rng_change_is_rejected(self):
        candidate = report(.7)
        candidate['shapes']['2']['timing_final_rng_sha256'] = 'changed'
        self.assertFalse(benchmark.compare(plan(), report(), candidate, 120)['adopt'])

    def test_checkpointed_maximum_shape_must_also_match(self):
        candidate = report(.7)
        candidate['shapes']['7']['equivalence']['skipped_update'] = True
        self.assertFalse(benchmark.compare(plan(), report(), candidate, 120)['adopt'])

    def test_candidate_cuda_oom_is_complete_rejection(self):
        result = benchmark.compare(plan(), report(), {'outcome': 'out_of_memory'}, 120)
        self.assertTrue(result['complete'])
        self.assertTrue(result['candidate_oom'])
        self.assertFalse(result['adopt'])

    def test_baseline_or_non_oom_candidate_failures_raise(self):
        for candidate in [{'complete': False, 'outcome': 'failed'}, {'complete': False}]:
            with self.assertRaises(ValueError):
                benchmark.compare(plan(), report(), candidate, 120)
        with self.assertRaises(ValueError):
            benchmark.compare(plan(), {'complete': False}, report(.7), 120)

    def test_small_speedup_is_rejected(self):
        result = benchmark.compare(plan(), report(), report(.99), 120)
        self.assertFalse(result['adopt'])
        self.assertIn('below_1_08', result['reason'])

    def test_probe_cost_and_five_minute_margin_must_be_recovered(self):
        result = benchmark.compare(plan(), report(), report(.7), 4000)
        self.assertFalse(result['adopt'])
        self.assertIn('probe_plus_300_seconds', result['reason'])

    def test_memory_headroom_is_a_hard_gate(self):
        candidate = report(.7)
        candidate['shapes']['2']['estimated_min_device_headroom_bytes'] = 1024**3
        result = benchmark.compare(plan(), report(), candidate, 120)
        self.assertFalse(result['adopt'])
        self.assertIn('headroom', result['reason'])

    def test_nested_state_hash_is_exact_and_layout_independent(self):
        for value in [torch.tensor(2.), torch.arange(10), torch.arange(6).reshape(2, 3).T,
                      np.arange(3), {'x': [torch.randn(2, 3), None, True, 3.5], 'y': (1, 2)}]:
            self.assertEqual(benchmark.value_sha(value), benchmark.value_sha(deepcopy(value)))
        tensor = torch.arange(6).reshape(2, 3).T
        self.assertEqual(benchmark.value_sha(tensor), benchmark.value_sha(tensor.contiguous()))

    def test_hash_includes_dtype_and_container_type(self):
        self.assertNotEqual(benchmark.value_sha(torch.tensor(2.)),
                            benchmark.value_sha(torch.tensor(2., dtype=torch.float64)))
        self.assertNotEqual(benchmark.value_sha([1, 2]), benchmark.value_sha((1, 2)))

    def test_deepcopied_adam_restore_preserves_original_checkpoint_payload(self):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(2, 3)).sum().backward(); optimizer.step()
        checkpoint_state = deepcopy(optimizer.state_dict())
        before = benchmark.value_sha(checkpoint_state)
        optimizer.load_state_dict(deepcopy(checkpoint_state)); optimizer.zero_grad()
        model(torch.ones(2, 3)).sum().backward(); optimizer.step()
        self.assertEqual(benchmark.value_sha(checkpoint_state), before)


if __name__ == '__main__':
    unittest.main()
