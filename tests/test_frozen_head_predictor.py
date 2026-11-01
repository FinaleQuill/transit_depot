"""CPU-only checks of a fixed head overlay; no images, GPU or source model mutation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from frozen_head_predictor import (apply_classifier_patch, validate_patch_config, BASE_BACKBONE,
                                   TINY_BACKBONE, FrozenHeadPredictor, ExperimentalPredictor,
                                   export_merged_for_validation, sha256_file)


class FrozenHeadTests(unittest.TestCase):
    def model(self, feature_dim=1024):
        torch.manual_seed(33)
        return SimpleNamespace(head=SimpleNamespace(fc=torch.nn.Linear(feature_dim, 10).half()))

    def patch(self, model, rows=None):
        rows = list(range(10)) if rows is None else rows
        return {'head.fc.weight': model.head.fc.weight.detach().cpu()[rows].clone(),
                'head.fc.bias': model.head.fc.bias.detach().cpu()[rows].clone()}

    def test_original_head_overlay_preserves_tensor_bytes_and_logits(self):
        model = self.model()
        features = torch.randn(3, 1024).half()
        expected = model.head.fc(features).detach().clone()
        weights = model.head.fc.weight.detach().clone()
        patch = self.patch(model)
        apply_classifier_patch(model, patch, class_ids=list(range(1, 11)))
        self.assertTrue(torch.equal(weights, model.head.fc.weight))
        self.assertTrue(torch.equal(expected, model.head.fc(features)))

    def test_selected_rows_change_and_other_rows_stay_bit_identical(self):
        model = self.model()
        source = self.patch(model)
        patch = {'head.fc.weight': torch.full((2, 1024), .125, dtype=torch.float16),
                 'head.fc.bias': torch.tensor([.25, -.25], dtype=torch.float16)}
        apply_classifier_patch(model, patch, class_ids=[1, 8])
        self.assertTrue(torch.equal(model.head.fc.weight[[0, 7]], patch['head.fc.weight']))
        untouched = [1, 2, 3, 4, 5, 6, 8, 9]
        self.assertTrue(torch.equal(source['head.fc.weight'][untouched], model.head.fc.weight[untouched]))
        self.assertTrue(torch.equal(source['head.fc.bias'][untouched], model.head.fc.bias[untouched]))
        self.assertTrue(torch.equal(patch['head.fc.weight'], torch.full((2, 1024), .125, dtype=torch.float16)))

    def test_invalid_patch_is_rejected_before_any_head_row_changes(self):
        for mutation in ('shape', 'fp32', 'nan', 'missing', 'extra'):
            model = self.model()
            original = self.patch(model)
            patch = copy.deepcopy(original)
            if mutation == 'shape':
                patch['head.fc.bias'] = torch.ones(9, dtype=torch.float16)
            elif mutation == 'fp32':
                patch['head.fc.weight'] = patch['head.fc.weight'].float()
            elif mutation == 'nan':
                patch['head.fc.bias'][3] = float('nan')
            elif mutation == 'missing':
                del patch['head.fc.bias']
            else:
                patch['encoder.weight'] = torch.zeros(1)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                apply_classifier_patch(model, patch, class_ids=list(range(1, 11)))
            self.assertTrue(torch.equal(model.head.fc.weight, original['head.fc.weight']))
            self.assertTrue(torch.equal(model.head.fc.bias, original['head.fc.bias']))

    def test_wrong_feature_dimension_and_duplicate_rows_are_rejected(self):
        model = self.model()
        with self.assertRaises(ValueError):
            apply_classifier_patch(model, self.patch(model), class_ids=list(range(1, 11)), feature_dim=768)
        with self.assertRaises(ValueError):
            apply_classifier_patch(model, self.patch(model), class_ids=[1, 1])

    def test_config_binds_base_and_patch_without_querying_runtime(self):
        config = {'schema_version': 1, 'artifact_type': 'frozen_head_student_fp16',
            'backbone_name': BASE_BACKBONE, 'image_size': 768, 'feature_dim': 1024,
            'class_ids': list(range(1, 11)), 'replaced_class_ids': list(range(1, 11)),
            'precision': 'float16', 'tta': 'hflip2', 'base_model_dir': '/immutable/B',
            'patch_file': 'head.pt', 'base_config_sha256': 'a'*64,
            'base_weights_sha256': 'b'*64, 'patch_sha256': 'c'*64}
        validate_patch_config(config)
        for key, value in [('patch_file', '../escape.pt'), ('replaced_class_ids', [1, 1]),
                           ('feature_dim', 768), ('base_weights_sha256', 'missing')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_patch_config({**config, key: value})

    def test_loader_binds_external_encoder_inventory_and_never_changes_source_files(self):
        with tempfile.TemporaryDirectory() as folder:
            base, output = Path(folder)/'B', Path(folder)/'patch'
            base.mkdir()
            output.mkdir()
            model = self.model()
            torch.save(self.patch(model), base/'student_fp16.pt')
            base_config = {'backbone_name': BASE_BACKBONE, 'image_size': 768,
                'weights_file': 'student_fp16.pt', 'weights_sha256': sha256_file(base/'student_fp16.pt')}
            (base/'config.json').write_text(json.dumps(base_config))
            torch.save(self.patch(model), output/'head.pt')
            config = {'schema_version': 1, 'artifact_type': 'frozen_head_student_fp16',
                'backbone_name': BASE_BACKBONE, 'image_size': 768, 'feature_dim': 1024,
                'class_ids': list(range(1, 11)), 'replaced_class_ids': list(range(1, 11)),
                'precision': 'float16', 'tta': 'hflip2', 'base_model_dir': '../B',
                'patch_file': 'head.pt', 'base_config_sha256': sha256_file(base/'config.json'),
                'base_weights_sha256': base_config['weights_sha256'],
                'patch_sha256': sha256_file(output/'head.pt')}
            (output/'config.json').write_text(json.dumps(config))
            initial_files = {str(path): sha256_file(path) for path in Path(folder).rglob('*') if path.is_file()}
            def fixture_init(instance, directory, *, tta, device):
                self.assertEqual(Path(directory), base)
                self.assertEqual(tta, 'hflip2')
                instance.torch, instance.model = torch, model
                instance.models, instance.members = [model], [{'image_size': 768, 'tta': 'hflip2'}]
            with patch.object(ExperimentalPredictor, '__init__', fixture_init):
                predictor = FrozenHeadPredictor(output, device='CPU_FIXTURE_ONLY')
            self.assertEqual(predictor.model_dir, output)
            self.assertEqual(len(predictor.inventory['files']), 4)
            self.assertEqual(predictor.inventory['total_bytes'], sum(Path(p).stat().st_size for p in initial_files))
            self.assertEqual(initial_files,
                {str(path): sha256_file(path) for path in Path(folder).rglob('*') if path.is_file()})
            # A changed patch cannot be accepted under an unchanged declared SHA.
            (output/'head.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'patch SHA256'):
                FrozenHeadPredictor(output, device='CPU_FIXTURE_ONLY')

    def test_tiny_original_head_patch_preserves_output_and_has_768_features(self):
        model = self.model(768)
        features = torch.randn(3, 768).half()
        expected = model.head.fc(features).detach().clone()
        apply_classifier_patch(model, self.patch(model), class_ids=list(range(1, 11)), feature_dim=768)
        self.assertTrue(torch.equal(expected, model.head.fc(features)))

    def test_merged_exports_preserve_encoder_bytes_key_order_and_full_parent_identity(self):
        for backbone, dimension in ((TINY_BACKBONE, 768), (BASE_BACKBONE, 1024)):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory(dir='/tmp') as folder:
                base, patch_dir, output = (Path(folder)/name for name in ('base', 'patch', 'merged'))
                base.mkdir()
                patch_dir.mkdir()
                model = self.model(dimension)
                state = {'encoder.synthetic_fixture': torch.randn(3, 2).half(),
                         **self.patch(model), 'encoder.synthetic_count': torch.tensor(3, dtype=torch.int64)}
                torch.save(state, base/'student_fp16.pt')
                original_weights_sha = sha256_file(base/'student_fp16.pt')
                parent = {'artifact_type': 'experimental_student_fp16', 'backbone_name': backbone,
                    'model_config': {'backbone_name': backbone, 'drop_rate': .2, 'drop_path_rate': .15,
                                     'gradient_checkpointing': False},
                    'class_ids': list(range(1, 11)), 'image_size': 768, 'precision': 'float16',
                    'weights_file': 'student_fp16.pt', 'weights_sha256': original_weights_sha,
                    'source_checkpoint_sha256': 'd'*64, 'pretrained_provenance': {'fixture': 'nonmedical'},
                    'teacher_provenance': [{'fixture': 'strict_oof'}]}
                (base/'config.json').write_text(json.dumps(parent))
                patch = {'head.fc.weight': torch.full((2, dimension), .125, dtype=torch.float16),
                         'head.fc.bias': torch.tensor([.25, -.25], dtype=torch.float16)}
                torch.save(patch, patch_dir/'head.pt')
                config = {'schema_version': 1, 'artifact_type': 'frozen_head_student_fp16',
                    'backbone_name': backbone, 'image_size': 768, 'feature_dim': dimension,
                    'class_ids': list(range(1, 11)), 'replaced_class_ids': [1, 8],
                    'precision': 'float16', 'tta': 'hflip2', 'base_model_dir': '../base',
                    'patch_file': 'head.pt', 'base_config_sha256': sha256_file(base/'config.json'),
                    'base_weights_sha256': original_weights_sha, 'patch_sha256': sha256_file(patch_dir/'head.pt'),
                    'epochs': 12, 'training_config': {'seed': 48, 'recipe': 'CPU_FIXTURE_ONLY'}}
                (patch_dir/'config.json').write_text(json.dumps(config))
                receipt = export_merged_for_validation(patch_dir/'config.json', output)
                merged = torch.load(output/'student_fp16.pt', weights_only=True)
                self.assertEqual(list(merged), list(state))
                for name in ('encoder.synthetic_fixture', 'encoder.synthetic_count'):
                    self.assertTrue(torch.equal(merged[name], state[name]))
                self.assertTrue(torch.equal(merged['head.fc.weight'][[0, 7]], patch['head.fc.weight']))
                for row in [1, 2, 3, 4, 5, 6, 8, 9]:
                    self.assertTrue(torch.equal(merged['head.fc.weight'][row], state['head.fc.weight'][row]))
                    self.assertTrue(torch.equal(merged['head.fc.bias'][row], state['head.fc.bias'][row]))
                merged_config = json.loads((output/'config.json').read_text())
                self.assertEqual(merged_config['source_checkpoint_sha256'], config['patch_sha256'])
                self.assertEqual(merged_config['source_patch_checkpoint_sha256'], config['patch_sha256'])
                self.assertEqual(merged_config['source_parent_checkpoint_sha256'], parent['source_checkpoint_sha256'])
                self.assertEqual(merged_config['pretrained_provenance'], parent['pretrained_provenance'])
                self.assertEqual(merged_config['teacher_provenance'], parent['teacher_provenance'])
                self.assertEqual(merged_config['source_epoch'], 12)
                self.assertEqual(merged_config['source_training_config'], config['training_config'])
                self.assertTrue(receipt['frozen_encoder_tensor_bytes_preserved'])
                self.assertEqual(sha256_file(base/'student_fp16.pt'), original_weights_sha)
                with self.assertRaises(FileExistsError):
                    export_merged_for_validation(patch_dir/'config.json', output)

    def test_merged_export_refuses_persistent_output_before_reading_sources(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            config = Path(folder)/'config.json'
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'must remain in /tmp'):
                export_merged_for_validation(config, '/home/featurize/work/forbidden_large_export')


if __name__ == '__main__':
    unittest.main()
