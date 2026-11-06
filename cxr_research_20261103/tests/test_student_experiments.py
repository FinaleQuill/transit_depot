"""Train-only weighting, KD alignment, and fixed offline initialization."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from student_experiments import compute_student_loss_config, load_local_pretrained_provenance
from train_distill import DistillConfig, distillation_losses
from train_overnight import compute_loss_config
from kd_common import TINY_BACKBONE, text_sha256


def records():
    return [{'split': 'train', 'training_eligible': 'true', 'subject_id': str(i),
             'study_id': str(i), 'study_key': f'{i}/{i}', 'image_id': f'image{i}',
             'targets': [int(i == 0)] * 10} for i in range(5)]


class StudentExperimentTests(unittest.TestCase):
    def test_weights_count_studies_not_views_and_original_is_unchanged(self):
        rows = records()
        rows.append({**rows[0], 'image_id': 'second_view'})
        original = compute_loss_config(rows, source_manifest_sha256='a' * 64, fold=None)
        self.assertEqual(original['pos_weight'], [4.] * 10)
        self.assertEqual(compute_student_loss_config(rows, source_manifest_sha256='a' * 64,
                         pos_weight_power=1.), original)
        new = compute_student_loss_config(rows, source_manifest_sha256='a' * 64)
        self.assertEqual(new['pos_weight'], [2.] * 10)
        self.assertEqual(new['positive_studies'], [1] * 10)
        self.assertEqual(new['original_pos_weight'], [4.] * 10)
        rows[-1]['split'] = 'valid'
        with self.assertRaises(ValueError):
            compute_student_loss_config(rows, source_manifest_sha256='a' * 64)

    def test_sqrt_weight_is_used_by_both_kd_alignment_and_bce_gradient(self):
        weights = torch.full((10,), 2.)
        student = torch.full((1, 10), 2.0 ** 0).log() + weights.log()
        student.requires_grad_()
        teacher = torch.full((1, 2, 10), 4.).log()
        target = torch.ones_like(student)
        hard, kl = distillation_losses(student, target, teacher,
                                      torch.full((1, 10), 4.), weights)
        torch.testing.assert_close(kl, torch.zeros(1), atol=2e-7, rtol=0)
        gradient, = torch.autograd.grad(hard.sum(), student)
        torch.testing.assert_close(gradient, 2. * (student.detach().sigmoid() - 1.) / 10)

    def test_config_rejects_unregistered_power_and_partial_offline_source(self):
        base = DistillConfig(Path('f'), Path('s'), Path('o'), Path('i'))
        replace(base, pos_weight_power=.5).validate()
        for invalid in (0., .7, True, float('nan')):
            with self.assertRaises(ValueError):
                replace(base, pos_weight_power=invalid).validate()
        with self.assertRaises(ValueError):
            replace(base, pretrained_weights_path=Path('w')).validate()

    def test_offline_source_checks_original_bytes_and_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'model.safetensors'
            weights.write_bytes(b'original initialization fixture')
            texts = {'model_card': 'license: apache-2.0\nImageNet',
                     'upstream_license': 'MIT License', 'apache_license': 'Apache License'}
            revision = 'a' * 40
            source = {'backbone_name': TINY_BACKBONE, 'repo_id': f'timm/{TINY_BACKBONE}',
                      'license': 'apache-2.0', 'pretrain_dataset': 'ImageNet-22k',
                      'finetune_dataset': 'ImageNet-1k', 'revision': revision,
                      'weights_path': '/previous/environment/model.safetensors',
                      'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                      'weights_url': f'https://huggingface.co/timm/{TINY_BACKBONE}/resolve/{revision}/model.safetensors',
                      'materials': {k: {'text': v, 'sha256': text_sha256(v)} for k,v in texts.items()}}
            path = root / 'provenance.json'
            path.write_text(json.dumps(source))
            result = load_local_pretrained_provenance(path, weights, TINY_BACKBONE)
            self.assertEqual(result['weights_path'], str(weights))
            self.assertEqual(json.loads(path.read_text()), source)
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                load_local_pretrained_provenance(path, weights, TINY_BACKBONE)


if __name__ == '__main__':
    unittest.main()
