"""Independent loss/gradient equivalence and full-Study packing boundaries."""
import unittest

import torch
from torch import nn

from study_packing import complete_study_packs
from study_student import accumulation_window_size, study_probability_bce


class StudyPackingTests(unittest.TestCase):
    def studies(self, counts):
        generator = torch.Generator().manual_seed(8)
        return [{'images': torch.randn(count, 3, generator=generator),
                 'targets': torch.tensor([[index % 2] * 10], dtype=torch.float32).repeat(count, 1)}
                for index, count in enumerate(counts)]

    def test_all_views_retained_and_packs_never_cross_optimizer_windows(self):
        loader = self.studies([1, 2, 3] * 11 + [12, 2])
        packs = list(complete_study_packs(loader, max_images=8, accumulation=32))
        self.assertEqual([i for pack in packs for i, _ in pack], list(range(35)))
        for pack in packs:
            self.assertEqual(len({i // 32 for i, _ in pack}), 1)
            count = sum(len(batch['images']) for _, batch in pack)
            self.assertTrue(count <= 8 or len(pack) == 1)
            for index, batch in pack:
                self.assertIs(batch, loader[index])
        self.assertTrue(any(len(pack) == 1 and len(pack[0][1]['images']) == 12 for pack in packs))

    def test_packed_gradients_equal_independent_equal_study_objective_including_tail(self):
        loader = self.studies([1, 2, 3] * 11 + [2, 1])
        torch.manual_seed(9)
        sequential = nn.Linear(3, 10)
        packed = nn.Linear(3, 10)
        packed.load_state_dict(sequential.state_dict())
        weights = torch.arange(1., 11.)
        sequential_gradients = []
        for index, batch in enumerate(loader):
            loss = study_probability_bce(sequential(batch['images']), batch['targets'], weights)
            (loss / accumulation_window_size(index, len(loader), 32)).backward()
            if (index + 1) % 32 == 0 or index + 1 == len(loader):
                sequential_gradients.append([p.grad.clone() for p in sequential.parameters()])
                sequential.zero_grad(set_to_none=True)
        packed_gradients = []
        for entries in complete_study_packs(loader, max_images=8, accumulation=32):
            values = packed(torch.cat([batch['images'] for _, batch in entries]))
            offset, losses = 0, []
            for _, batch in entries:
                count = len(batch['images'])
                losses.append(study_probability_bce(values[offset:offset+count], batch['targets'], weights))
                offset += count
            (torch.stack(losses).sum() / accumulation_window_size(entries[0][0], len(loader), 32)).backward()
            last = entries[-1][0]
            if (last + 1) % 32 == 0 or last + 1 == len(loader):
                packed_gradients.append([p.grad.clone() for p in packed.parameters()])
                packed.zero_grad(set_to_none=True)
        self.assertEqual(len(packed_gradients), 2)
        for expected, actual in zip(sequential_gradients, packed_gradients):
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=2e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
