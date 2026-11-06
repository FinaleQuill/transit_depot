from types import SimpleNamespace

import numpy as np
import torch

from one_week.train import batch_objective, class_weights
from train_distill import distillation_losses, supervised_bce_loss
from train_overnight import set_learning_rates


def test_raw_missing_kd_retains_full_bce_and_reference_gradients():
    logits = torch.tensor([[.2] * 10, [-.3] * 10, [.7] * 10], requires_grad=True)
    labels = torch.tensor([[0.] * 10, [1.] * 10, [0.] * 10])
    weights = torch.arange(1, 11, dtype=torch.float32).sqrt()
    cache = SimpleNamespace(index={'known0': 0, 'known1': 1},
        logits=np.arange(40, dtype=np.float32).reshape(2, 2, 10) / 20.,
        teacher_pos_weights=np.ones((2, 10), dtype=np.float32) * 3.)
    actual, _, _, mask = batch_objective(logits, labels, weights,
                                        ['known0', 'missing', 'known1'], cache, .2)
    hard = supervised_bce_loss(logits, labels, weights)
    _, soft = distillation_losses(logits[[0, 2]], labels[[0, 2]], torch.tensor(cache.logits),
                                  torch.tensor(cache.teacher_pos_weights), weights)
    expected = torch.stack([.8 * hard[0] + .2 * soft[0], hard[1], .8 * hard[2] + .2 * soft[1]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mask, torch.tensor([1., 0., 1.]))
    a_grad, = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
    e_grad, = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(a_grad, e_grad)


def test_no_teacher_target_is_exact_full_supervision():
    logits = torch.zeros(2, 10, requires_grad=True)
    labels = torch.zeros_like(logits)
    cache = SimpleNamespace(index={})
    actual, hard, soft, mask = batch_objective(logits, labels, torch.ones(10), ['a', 'b'], cache, .2)
    torch.testing.assert_close(actual, hard)
    assert not soft.any() and not mask.any()
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_positive_weights_count_original_studies_once_after_valid_merge():
    rows = [dict(study_key='a', targets=[1] * 10, split='train'),
            dict(study_key='a', targets=[1] * 10, split='train'),
            dict(study_key='b', targets=[0] * 10, split='train'),
            dict(study_key='c', targets=[0] * 10, split='valid')]
    weights, stats = class_weights(rows)
    np.testing.assert_allclose(weights, np.sqrt(2))
    assert stats['studies'] == 3 and stats['images'] == 4 and stats['counted_splits'] == ['train', 'valid']


def test_epoch9_cutoff_preserves_historical_25_epoch_cosine():
    cfg = SimpleNamespace(epochs=25, warmup_epochs=1, min_lr_ratio=.05,
                          backbone_lr=5e-5, head_lr=5e-4)
    optimizer = torch.optim.AdamW([{'params': [torch.nn.Parameter(torch.tensor(1.))],
                                  'role': 'backbone', 'name': 'backbone'}])
    rate = set_learning_rates(optimizer, cfg, 9 * 949 - 1, 949)['backbone']
    assert 3.8e-5 < rate < 3.82e-5
    cfg.epochs = 9
    compressed = set_learning_rates(optimizer, cfg, 9 * 949 - 1, 949)['backbone']
    assert compressed == 2.5e-6 and rate > compressed * 15


def test_tail_window_gradient_uses_actual_image_count():
    values = torch.tensor([1., 2., 3.])
    parameter = torch.nn.Parameter(torch.tensor(2.))
    # Last accumulation window contains 2+1 images, not two full microbatches.
    for chunk in (values[:2], values[2:]):
        ((chunk * parameter).square().sum() / len(values)).backward()
    expected = (2. * parameter.detach() * values.square()).mean()
    torch.testing.assert_close(parameter.grad, expected)
