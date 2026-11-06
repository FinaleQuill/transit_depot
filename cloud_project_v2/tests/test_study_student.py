"""Independent CPU checks of Study probability supervision and full-view grouping."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import study_student as study


def rows():
    labels = {"a": [1, 0] * 5, "b": [0, 1] * 5, "c": [1] * 10}
    return [{"image_id": f"i{i}", "study_key": key, "targets": labels[key], "split": "train"}
            for i, key in enumerate(["a", "b", "a", "c", "c", "a"])]


class ProbabilityMeanLossTests(unittest.TestCase):
    def test_weighted_bce_matches_independent_float64_probability_formula(self):
        torch.manual_seed(4)
        logits = torch.randn(5, 10, dtype=torch.float64) * 2
        targets = torch.tensor([[1., 0.] * 5], dtype=torch.float64).repeat(5, 1)
        weights = torch.linspace(1., 9., 10, dtype=torch.float64)
        probabilities = 1 / (1 + torch.exp(-logits))
        mean = probabilities.mean(0)
        expected = -(weights * targets[0] * mean.log()
                     + (1 - targets[0]) * (1 - mean).log()).mean()
        self.assertAlmostEqual(float(study.study_probability_bce(logits, targets, weights)),
                               float(expected), places=6)

    def test_aggregation_happens_before_bce_and_is_not_logit_mean(self):
        logits = torch.tensor([[-4.] * 10, [2.] * 10])
        targets, weights = torch.ones_like(logits), torch.ones(10)
        actual = study.study_probability_bce(logits, targets, weights)
        mean_probability = (1 / (1 + math.exp(4)) + 1 / (1 + math.exp(-2))) / 2
        self.assertAlmostEqual(float(actual), -math.log(mean_probability), places=6)
        self.assertGreater(abs(float(actual - F.binary_cross_entropy_with_logits(logits, targets))), .5)
        logit_average_loss = F.binary_cross_entropy_with_logits(logits.mean(0), targets[0])
        self.assertGreater(abs(float(actual - logit_average_loss)), .1)

    def test_single_view_matches_original_weighted_image_bce(self):
        logits = torch.linspace(-3, 3, 10)[None]
        targets, weights = torch.tensor([[0., 1.] * 5]), torch.arange(1., 11.)
        expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
        self.assertTrue(torch.allclose(study.study_probability_bce(logits, targets, weights), expected, atol=1e-6))

    def test_gradient_matches_analytic_derivative_through_probability_mean(self):
        logits = torch.tensor(np.random.default_rng(8).normal(size=(4, 10)), dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([[0., 1.] * 5]).repeat(4, 1)
        weights = torch.arange(1., 11.)
        loss = study.study_probability_bce(logits, targets, weights)
        loss.backward()
        p = 1 / (1 + np.exp(-logits.detach().numpy().astype(np.float64)))
        mean = p.mean(axis=0)
        y, weight = targets[0].numpy(), weights.numpy()
        analytic = ((1 - y) / (1 - mean) - weight * y / mean)[None] * p * (1 - p) / 4 / 10
        np.testing.assert_allclose(logits.grad.numpy(), analytic, rtol=3e-6, atol=3e-7)

    def test_extreme_logits_have_finite_loss_and_nonzero_opposite_label_gradients(self):
        logits = torch.tensor([[-1000., 1000.] * 5] * 3, requires_grad=True)
        targets, weights = torch.tensor([[1., 0.] * 5] * 3), torch.arange(1., 11.)
        loss = study.study_probability_bce(logits, targets, weights)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool((logits.grad.abs() > 0).all()))

    def test_kd_aligns_each_raw_teacher_view_then_mean_probabilities_and_images(self):
        torch.manual_seed(12)
        logits = torch.randn(3, 10, requires_grad=True)
        teacher = torch.randn(3, 2, 10, requires_grad=True)
        targets = torch.tensor([[0., 1.] * 5] * 3)
        teacher_weights = torch.rand(3, 10) * 5 + 1
        student_weights = torch.rand(10) * 2 + 1
        hard, soft = study.study_distillation_losses(logits, targets, teacher, teacher_weights, student_weights)
        q = torch.sigmoid((teacher.detach().double() - teacher_weights.double().log()[:, None]
                          + student_weights.double().log()[None, None]) / 2).mean(1)
        p = torch.sigmoid(logits.detach().double() / 2)
        expected = (q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()).mean() * 4
        self.assertAlmostEqual(float(soft.detach()), float(expected), places=6)
        (hard + soft).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_conflicting_labels_nonbinary_labels_and_empty_views_are_rejected(self):
        logits, targets, weights = torch.zeros(2, 10), torch.zeros(2, 10), torch.ones(10)
        targets[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            study.study_probability_bce(logits, targets, weights)
        with self.assertRaisesRegex(ValueError, "binary"):
            study.study_probability_bce(logits, torch.full((2, 10), .5), weights)
        with self.assertRaises(ValueError):
            study.study_probability_bce(logits[:0], targets[:0], weights)


class CompleteStudyDataTests(unittest.TestCase):
    def test_noncontiguous_views_are_complete_and_preserve_first_seen_study_order(self):
        grouped = study.complete_study_groups(rows())
        self.assertEqual(grouped, [[0, 2, 5], [1], [3, 4]])
        self.assertEqual(sorted(index for group in grouped for index in group), list(range(6)))

    def test_duplicates_test_records_and_conflicting_study_labels_fail(self):
        records = rows()
        with self.assertRaisesRegex(ValueError, "unique"):
            study.complete_study_groups(records + [records[0]])
        records[0] = {**records[0], "split": "test"}
        with self.assertRaisesRegex(ValueError, "train"):
            study.complete_study_groups(records)
        records = rows()
        records[2] = {**records[2], "targets": [0] * 10}
        with self.assertRaisesRegex(ValueError, "disagree"):
            study.complete_study_groups(records)

    def test_real_dataloader_default_conversion_preserves_all_views_per_study(self):
        class FakeImages:
            def __init__(self, records, *args, **kwargs):
                self.records = records
            def reseed(self, seed):
                self.seed = seed
            def __getitem__(self, index):
                row = self.records[index]
                return {"image": torch.full((3, 2, 2), float(index)), "targets": torch.tensor(row['targets']),
                    "image_id": row['image_id'], "study_key": row['study_key'],
                    "subject_id": 'subject', "study_id": row['study_key'], 'split': 'train', 'loss_weight': .01}
        expected = {key: {r['image_id'] for r in rows() if r['study_key'] == key} for key in ['a', 'b', 'c']}
        with patch('dataset_overnight.OvernightImageDataset', FakeImages):
            loader = study.make_study_loader(rows(), '/never-open-images', image_size=768,
                num_workers=0, prefetch_factor=2, seed=44, pin_memory=False)
            self.assertEqual(len(loader), 3)
            seen = {}
            for batch in loader:
                self.assertEqual(len(set(batch['study_keys'])), 1)
                key = batch['study_keys'][0]
                seen[key] = set(batch['image_ids'])
                self.assertEqual(batch['images'].shape[0], len(expected[key]))
            self.assertEqual(seen, expected)

    def test_tail_window_uses_actual_study_count_and_unit_total_weight(self):
        denominators = [study.accumulation_window_size(i, 35, 32) for i in range(35)]
        self.assertEqual(denominators, [32] * 32 + [3] * 3)
        self.assertAlmostEqual(sum(1 / x for x in denominators[:32]), 1.)
        self.assertAlmostEqual(sum(1 / x for x in denominators[32:]), 1.)
        with self.assertRaises(ValueError):
            study.accumulation_window_size(35, 35, 32)

    def test_study_config_is_base_and_cannot_silently_change_batch_semantics(self):
        config = study.StudyConfig(fold_plan=Path('/folds'), soft_labels=Path('/cache'), output=Path('/out'),
            image_root=Path('/images'), pretrained_provenance_path=Path('/provenance'), pretrained_weights_path=Path('/weights'))
        config.validate()
        self.assertIn('convnext_base', config.model_config()['backbone_name'])
        with self.assertRaisesRegex(ValueError, "one complete Study"):
            replace(config, batch_size=2, accumulation_steps=16).validate()


class StudyContinuationTests(unittest.TestCase):
    def config(self):
        return study.StudyConfig(fold_plan=Path('/folds'), soft_labels=Path('/cache'), output=Path('/out'),
            image_root=Path('/images'), pretrained_provenance_path=Path('/provenance'), pretrained_weights_path=Path('/weights'))

    def checkpoint(self, config):
        checkpoint = {"schema_version": 10, "model_name": study.MODEL_NAME,
            "training_config": study._configuration(config), "epoch": 2, "best_epoch": 1,
            "best_auc": .81, "weights_source": "ema", "inference_ready": True, "validation_complete": True,
            "validation": {"macro_auc": .8},
            "progress": {"completed_epochs": 2, "validated_epochs": 2, "optimizer_updates": 12, "ema_updates": 12,
                         "backward_batches": 350, "amp_skipped_steps": 0},
            "early_stopping": {"reference_auc": .81, "bad_epochs": 1}}
        for name in ('model_state_dict', 'raw_model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'rng_state'):
            checkpoint[name] = {}
        return checkpoint

    def test_resume_requires_complete_states_and_unchanged_protocol(self):
        config = self.config()
        expected = {'schema_version': 10, 'model_name': study.MODEL_NAME}
        checkpoint = self.checkpoint(config)
        study.validate_study_resume(checkpoint, config, expected)
        checkpoint.pop('rng_state')
        with self.assertRaisesRegex(ValueError, 'rng_state'):
            study.validate_study_resume(checkpoint, config, expected)
        checkpoint = self.checkpoint(config)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            study.validate_study_resume(checkpoint, replace(config, seed=45), expected)

    def test_resume_rejects_corrupt_auc_and_progress_counters(self):
        config = self.config()
        expected = {'schema_version': 10, 'model_name': study.MODEL_NAME}
        for location, key, value in [('validation', 'macro_auc', .9),
                                     ('early_stopping', 'reference_auc', 1.5),
                                     ('progress', 'backward_batches', -1),
                                     ('progress', 'amp_skipped_steps', 0.5)]:
            checkpoint = self.checkpoint(config)
            checkpoint[location][key] = value
            with self.subTest(location=location, key=key), self.assertRaises(ValueError):
                study.validate_study_resume(checkpoint, config, expected)

    def test_preview_never_opens_images_or_constructs_model_or_queries_cuda(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            output = Path(folder) / 'new-run'
            args = ['--fold-plan', '/folds', '--soft-labels', '/cache', '--output', str(output),
                    '--pretrained-provenance-path', '/provenance', '--pretrained-weights-path', '/weights']
            with (patch.object(study, 'run_training', side_effect=AssertionError('training forbidden')),
                  patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA forbidden'))):
                self.assertEqual(study.main(args), 0)
            self.assertFalse(output.exists())


class StudyEpochTests(unittest.TestCase):
    def fixture(self, count=35):
        records, batches = [], []
        for index in range(count):
            views = 1 + index % 3
            ids = [f'image{index}_{v}' for v in range(views)]
            key = f'study{index}'
            targets = [[index % 2, (index + 1) % 2] * 5] * views
            records.extend({'image_id': value, 'study_key': key} for value in ids)
            batches.append({'image_ids': ids, 'study_keys': [key] * views,
                'images': torch.full((views, 3, 2, 2), .1 + index / 100),
                'targets': torch.tensor(targets, dtype=torch.float32)})
        class Loader:
            dataset = SimpleNamespace(records=records)
            def __len__(self):
                return len(batches)
            def __iter__(self):
                return iter(batches)
        cache = SimpleNamespace(index={r['image_id']: i for i, r in enumerate(records)},
            logits=np.zeros((len(records), 2, 10), dtype=np.float32),
            teacher_pos_weights=np.ones((len(records), 10), dtype=np.float32))
        return Loader(), cache, batches

    def execute(self, loader, cache):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'role': 'head', 'name': 'head'}], lr=.001)
        progress = {'optimizer_updates': 0, 'ema_updates': 0, 'backward_batches': 0, 'amp_skipped_steps': 0}
        def step(*args):
            progress['optimizer_updates'] += 1
            progress['ema_updates'] += 1
        hook = optimizer.register_step_post_hook(step)
        config = study.StudyConfig(fold_plan=Path('/folds'), soft_labels=Path('/cache'), output=Path('/out'),
            image_root=Path('/images'), pretrained_provenance_path=Path('/provenance'), pretrained_weights_path=Path('/weights'))
        try:
            with (patch.object(torch.cuda, 'reset_peak_memory_stats'),
                  patch.object(torch.cuda, 'max_memory_allocated', return_value=0),
                  patch.object(torch, 'autocast', side_effect=lambda **kwargs: nullcontext()),
                  redirect_stdout(io.StringIO())):
                result = study.train_study_epoch(model, None, loader, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), torch.device('cpu'), torch.float32,
                    1, config, progress, cache, torch.ones(10))
        finally:
            hook.remove()
        return result, progress

    def test_epoch_keeps_all_views_and_steps_once_per_full_or_tail_study_window(self):
        loader, cache, _ = self.fixture()
        result, progress = self.execute(loader, cache)
        self.assertEqual(result['studies'], 35)
        self.assertEqual(result['images'], len(loader.dataset.records))
        self.assertTrue(result['full_manifest_coverage'])
        self.assertEqual(progress['optimizer_updates'], 2)
        self.assertEqual(progress['ema_updates'], 2)
        self.assertEqual(progress['backward_batches'], 35)
        self.assertEqual(progress['amp_skipped_steps'], 0)

    def test_epoch_rejects_incomplete_study_before_silently_training_partial_views(self):
        loader, cache, batches = self.fixture()
        batches[1] = {**batches[1], 'image_ids': batches[1]['image_ids'][:1],
                      'study_keys': batches[1]['study_keys'][:1]}
        with self.assertRaisesRegex(ValueError, 'omitted or added'):
            self.execute(loader, cache)


if __name__ == '__main__':
    unittest.main()
