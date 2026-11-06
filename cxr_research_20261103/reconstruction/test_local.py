"""CPU semantic checks for E2 source crops, geometry and Study-only learning."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from reconstruction.data import MEAN, STD, StudyDataset, letterbox, tile_boxes
from reconstruction.local_data import LocalStudyDataset, tile_to_global_affines
from reconstruction.local_model import LocalStudyModel
from reconstruction.model import StudyModel, mean_probability_logits, study_bce_loss


class TinyEncoder(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.norm = nn.LayerNorm(8)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8, 10)

    def forward_features(self, pixels):
        return self.conv(pixels).tanh()

    def forward_head(self, features, pre_logits=False):
        x = self.drop(self.norm(features.mean((-2, -1))))
        return x if pre_logits else self.fc(x)

    def get_classifier(self):
        return self.fc


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = []
        self.sources = []
        for i, (height, width) in enumerate(((37, 53), (47, 31))):
            y, x = np.mgrid[:height, :width]
            pixels = np.stack(((x*31+y*7) % 256, (y*17) % 256, (x*13) % 256), -1).astype(np.uint8)
            path = self.root / 'train' / 'patient' / 'study' / f'image{i}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            source = Image.fromarray(pixels)
            source.save(path)
            self.sources.append(source)
            self.records.append({'split': 'train', 'image_id': f'train:{i:06d}',
                                 'subject_id': 'patient', 'study_id': 'study', 'study_key': 'patient/study',
                                 'relative_path': str(path.relative_to(self.root)), 'targets': [i % 2 for i in range(10)],
                                 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 'training_eligible': 'true', 'status': 'ok'})

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_pixels_are_cropped_before_downsampling_and_all_views_retained(self):
        data = LocalStudyDataset(self.records, self.root, image_size=17)
        sample = data[0]
        self.assertEqual(sample['images'].shape, (2, 5, 3, 17, 17))
        self.assertEqual(sample['targets'].shape, (10,))
        self.assertEqual(len(sample['image_ids']), 2)
        for v, source in enumerate(self.sources):
            boxes = tile_boxes(*source.size, True)
            coverage = np.zeros((source.height, source.width), dtype=bool)
            for r, box in enumerate(boxes):
                expected = np.asarray(letterbox(source.crop(box), 17), dtype=np.float32).transpose(2, 0, 1)/255
                expected = (expected-np.asarray(MEAN, np.float32)[:, None, None])/np.asarray(STD, np.float32)[:, None, None]
                torch.testing.assert_close(sample['images'][v, r], torch.from_numpy(expected), rtol=0, atol=0)
                if r:
                    a, b, c, d = box
                    coverage[b:d, a:c] = True
                    self.assertLessEqual(abs((c-a)-source.width*.6), .5)
                    self.assertLessEqual(abs((d-b)-source.height*.6), .5)
            self.assertTrue(coverage.all())
        for row in self.records:
            self.assertEqual(hashlib.sha256((self.root/row['relative_path']).read_bytes()).hexdigest(), row['file_sha256'])

    def test_augmented_global_tensor_remains_bitwise_identical_to_e0(self):
        baseline = StudyDataset(self.records, self.root, image_size=23, seed=20261103, training=True)
        local = LocalStudyDataset(self.records, self.root, image_size=23, seed=20261103, training=True)
        triggered = set()
        for epoch in range(18):
            baseline.set_epoch(epoch)
            local.set_epoch(epoch)
            sample = local[0]
            torch.testing.assert_close(sample['images'][:, 0], baseline[0]['images'], rtol=0, atol=0)
            for flip, brightness, contrast, angle in sample['augmentation']:
                if flip: triggered.add('flip')
                if brightness or contrast != 1: triggered.add('intensity')
                if angle: triggered.add('rotation')
        self.assertEqual(triggered, {'flip', 'intensity', 'rotation'})

    def test_source_crop_affine_tracks_rotation_reflection_and_letterbox(self):
        boxes = tile_boxes(100, 100, True)
        plain = tile_to_global_affines(100, 100, 20, boxes)
        torch.testing.assert_close(plain[1], torch.tensor([[.6, 0, 0], [0, .6, 0.]]))
        flipped = tile_to_global_affines(100, 100, 20, boxes, flip=True)
        torch.testing.assert_close(flipped[1], torch.tensor([[.6, 0, .4], [0, .6, 0.]]))
        rotated = tile_to_global_affines(100, 100, 20, boxes, angle=90)
        torch.testing.assert_close(rotated[1], torch.tensor([[.6, 0, 0], [0, .6, .4]]), atol=1e-7, rtol=1e-6)
        # A tall source's global and crop have different rounded padding; the
        # global-frame map must include that padding, not just normalized xyxy.
        tall = tile_to_global_affines(31, 47, 17, tile_boxes(31, 47, True))
        self.assertFalse(torch.allclose(tall[1], plain[1]))
        torch.testing.assert_close(tall[0], torch.tensor([[1., 0, 0], [0, 1., 0]]), rtol=0, atol=0)

    def test_hash_change_and_test_records_rejected(self):
        data = LocalStudyDataset(self.records, self.root, image_size=17)
        (self.root/self.records[0]['relative_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            data[0]
        records = copy.deepcopy(self.records)
        records[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            LocalStudyDataset(records, self.root)


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.encoder = TinyEncoder()
        self.images = torch.randn(3, 5, 3, 7, 7)
        self.geometry = tile_to_global_affines(100, 100, 20, tile_boxes(100, 100, True)).expand(3, -1, -1, -1)
        self.targets = torch.tensor([1., 0., 1., 0., 0., 1., 0., 1., 0., 0.])

    def model(self, **kwargs):
        return LocalStudyModel.from_backbone(copy.deepcopy(self.encoder), attention_dim=12, **kwargs)

    def active(self, model):
        with torch.no_grad():
            model.region_query.normal_()
            model.region_residual_weight.normal_(std=.2)
            model.region_residual_bias.normal_(std=.1)
        return model

    def test_e0_transfer_and_zero_residual_exact_global_equality(self):
        e0 = StudyModel.from_backbone(copy.deepcopy(self.encoder), view_chunk_size=2).eval()
        local = self.model(view_chunk_size=2).eval()
        local.load_e0_state_dict(e0.state_dict())
        result = local(self.images, region_affines=self.geometry, return_components=True)
        torch.testing.assert_close(result['study_logits'], e0(self.images[:, 0]), rtol=0, atol=0)
        torch.testing.assert_close(result['study_logits'], result['global_study_logits'], rtol=0, atol=0)
        invalid = dict(e0.state_dict(), view_residual_bias=torch.zeros(10))
        with self.assertRaises(ValueError):
            local.load_e0_state_dict(invalid)
        missing = dict(e0.state_dict())
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            local.load_e0_state_dict(missing)

    def test_stochastic_global_path_matches_e0_at_same_rng_state(self):
        encoder = TinyEncoder(dropout=.4)
        e0 = StudyModel.from_backbone(copy.deepcopy(encoder), view_chunk_size=2).train()
        local = LocalStudyModel.from_backbone(copy.deepcopy(encoder), view_chunk_size=2).train()
        rng = torch.get_rng_state()
        expected = e0(self.images[:, 0])
        torch.set_rng_state(rng)
        actual = local(self.images, region_affines=self.geometry)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_group_mass_and_duplicate_local_invariance(self):
        model = self.model().eval()
        initial = model(self.images, region_affines=self.geometry, return_components=True)
        torch.testing.assert_close(initial['group_attention'], torch.full((3, 2, 10), .5))
        model = self.active(model)
        # Duplicating the entire local set must not increase its total mass or
        # influence; this differs from flattening all tiles into one softmax.
        order = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4])
        original = model(self.images, region_affines=self.geometry, return_components=True)
        repeated = model(self.images[:, order], region_affines=self.geometry[:, order], return_components=True)
        torch.testing.assert_close(original['study_logits'], repeated['study_logits'], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(original['group_attention'], repeated['group_attention'], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(original['region_attention'].sum(1), torch.ones(3, 10))
        torch.testing.assert_close(original['view_attention'], torch.full((3, 10), 1/3))
        torch.testing.assert_close(original['study_logits'], mean_probability_logits(original['image_logits']))

    def test_region_and_view_permutation_preserve_prediction(self):
        model = self.active(self.model()).eval()
        expected = model(self.images, region_affines=self.geometry)
        v, r = torch.tensor([2, 0, 1]), torch.tensor([0, 4, 1, 3, 2])
        actual = model(self.images[v][:, r], region_affines=self.geometry[v][:, r])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        wrong = self.geometry.clone()
        wrong[:, 0, 0, 2] = .2
        with self.assertRaises(ValueError):
            model(self.images, region_affines=wrong)
        with self.assertRaises(ValueError):
            model(self.images)

    def test_no_local_fallback_and_single_view_after_head_has_trained(self):
        model = self.active(self.model()).eval()
        result = model(self.images[:, 0], return_components=True)
        torch.testing.assert_close(result['study_logits'], result['global_study_logits'], atol=0, rtol=0)
        one = model(self.images[:1], region_affines=self.geometry[:1], return_components=True)
        torch.testing.assert_close(one['study_logits'], one['image_logits'][0], atol=1e-6, rtol=1e-6)

    def test_every_local_receives_only_study_loss_gradient(self):
        model = self.active(self.model()).train()
        images = self.images.clone().requires_grad_()
        components = model(images, region_affines=self.geometry, return_components=True)
        loss = study_bce_loss(components['study_logits'], self.targets,
                              global_study_logits=components['global_study_logits'], global_aux_weight=.2)
        loss.backward()
        self.assertTrue(bool((images.grad.abs().sum((2, 3, 4)) > 0).all()))
        self.assertGreater(float(model.region_query.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            study_bce_loss(components['study_logits'], self.targets.expand(3, 5, -1))

    def test_packed_checkpointed_chunks_match_independent_study_gradients(self):
        separate = self.active(self.model(view_chunk_size=3)).train()
        packed = copy.deepcopy(separate)
        packed.gradient_checkpointing = True
        packed.view_chunk_size = 2
        studies = [self.images[:1], self.images[1:]]
        geometry = [self.geometry[:1], self.geometry[1:]]
        targets = [self.targets, 1-self.targets]
        expected = torch.stack([study_bce_loss(separate(x, region_affines=a), y)
                                for x, a, y in zip(studies, geometry, targets)]).mean()
        with patch.object(packed, '_encode_chunk', wraps=packed._encode_chunk) as encoder:
            outputs = packed.forward_studies(studies, region_affines=geometry)
            self.assertEqual(sum(call.args[0].shape[0] for call in encoder.call_args_list), 15)
            self.assertLessEqual(max(call.args[0].shape[0] for call in encoder.call_args_list), 2)
        actual = torch.stack([study_bce_loss(z, y) for z, y in zip(outputs, targets)]).mean()
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        for (name, a), (_, b) in zip(separate.named_parameters(), packed.named_parameters()):
            self.assertIsNotNone(b.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
