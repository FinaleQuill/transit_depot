"""CPU publication guards and portable source/asset closure checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from deploy_kd import sha256_file
from robust_ensemble.deploy import (ROOT, _copy_sources, inspect_package,
                                    package_candidate, publish, require_winner,
                                    selected_assets)
from robust_ensemble.predictor import FUSION_WEIGHTS


class PublicationTests(unittest.TestCase):
    def fixture(self, root):
        comparison = root / 'final_comparison'
        comparison.mkdir()
        manifest = root / 'valid.csv'
        manifest.write_text('unchanged original valid manifest')
        base = root / 'tiny'
        base.mkdir()
        (base / 'weights.pt').write_bytes(b'original tiny')
        (base / 'config.json').write_text(json.dumps({'weights_file': 'weights.pt'}))
        patch_dir = root / 'tiny_patch'
        patch_dir.mkdir()
        (patch_dir / 'patch.pt').write_bytes(b'tiny original classifier patch')
        patch_config = {'base_model_dir': str(base), 'patch_file': 'patch.pt',
                        'base_config_sha256': sha256_file(base / 'config.json'),
                        'base_weights_sha256': sha256_file(base / 'weights.pt'),
                        'patch_sha256': sha256_file(patch_dir / 'patch.pt')}
        (patch_dir / 'config.json').write_text(json.dumps(patch_config))
        for name in ('base768', 'opacity512'):
            (root / (name + '.pt')).write_bytes(name.encode())
        config = {'output': str(root), 'valid_manifest': str(manifest),
                  'fusion_weights': FUSION_WEIGHTS,
                  'sprint_config': {'candidate_weights': str(root / 'base768.pt'),
                      'candidate_weights_sha256': sha256_file(root / 'base768.pt'),
                      'tiny_patch_config': str(patch_dir / 'config.json'),
                      'tiny_patch_config_sha256': sha256_file(patch_dir / 'config.json')},
                  'opacity_member': {'id': 'opacity512', 'branch': 'opacity',
                      'image_size': 512, 'epochs': 6,
                      'backbone_name': 'convnext_tiny.fb_in22k_ft_in1k_384',
                      'weights': str(root / 'opacity512.pt'),
                      'weights_sha256': sha256_file(root / 'opacity512.pt')}}
        for name in ('candidate_predictions.csv', 'candidate.image_predictions.csv'):
            (comparison / name).write_text(name)
        report = {'complete': True, 'candidate_won': True, 'fusion_weights': FUSION_WEIGHTS,
                  'manifest_sha256': sha256_file(manifest),
                  'candidate_inference_assets': selected_assets(config),
                  'candidate_predictions_sha256': sha256_file(comparison / 'candidate_predictions.csv'),
                  'candidate_image_predictions_sha256': sha256_file(comparison / 'candidate.image_predictions.csv')}
        selection = {'candidate_won': True, 'report': str(comparison / 'report.json'),
                     'fusion_weights': FUSION_WEIGHTS, 'selection_checks': {'macro_better': True}}
        (comparison / 'report.json').write_text(json.dumps(report))
        (comparison / 'selection.json').write_text(json.dumps(selection))
        return config, comparison, selection

    def test_loser_and_empty_checks_never_access_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, comparison, selection = self.fixture(root)
            for changes in ({'candidate_won': False}, {'candidate_won': True, 'selection_checks': {}}):
                selection.update(changes)
                (comparison / 'selection.json').write_text(json.dumps(selection))
                with patch('robust_ensemble.deploy.prediction_records') as read:
                    with self.assertRaises(ValueError):
                        publish(config)
                    read.assert_not_called()
                self.assertFalse((root / 'deployment_candidate').exists())

    def test_changed_member_and_saved_prediction_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, comparison, _ = self.fixture(root)
            require_winner(config)
            (comparison / 'candidate_predictions.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'probabilities changed'):
                require_winner(config)
            (comparison / 'candidate_predictions.csv').write_text('candidate_predictions.csv')
            (root / 'opacity512.pt').write_bytes(b'another model')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                require_winner(config)

    def test_package_survives_move_and_refuses_external_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = self.fixture(root)
            destination = root / 'deployment_candidate'
            with patch('robust_ensemble.deploy._copy_sources', return_value=[]):
                result = package_candidate(config, destination)
            self.assertTrue(result['size_pass'])
            moved = root / 'relocated'
            destination.rename(moved)
            self.assertTrue(inspect_package(moved / 'model')['size_pass'])
            runtime_path = moved / 'model/runtime_config.json'
            runtime = json.loads(runtime_path.read_text())
            self.assertEqual(runtime['sprint_config']['candidate_weights'], 'base768.pt')
            self.assertEqual(runtime['opacity_member']['weights'], 'opacity512.pt')
            self.assertEqual(json.loads((moved / 'model/tiny_patch/config.json').read_text())['base_model_dir'], '../tiny')
            runtime['opacity_member']['weights'] = str(root / 'opacity512.pt')
            runtime_path.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, 'relative and contained'):
                inspect_package(moved / 'model')

    def test_copied_sources_win_over_original_project_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'code'
            files = _copy_sources(copied)
            names = {item['path'] for item in files}
            self.assertIn('robust_ensemble/__init__.py', names)
            self.assertIn('ensemble_sprint/__init__.py', names)
            self.assertIn('reconstruction/__init__.py', names)
            script = '''import importlib.util, pathlib, sys
copied, original = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(copied), str(original)]
import robust_ensemble.deploy, robust_ensemble.predictor
for name in ('robust_ensemble.deploy', 'robust_ensemble.predictor', 'ensemble_sprint.predictor', 'sprint.deploy', 'reconstruction.model'):
    origin = importlib.util.find_spec(name).origin
    assert pathlib.Path(origin).resolve().is_relative_to(copied), origin
print('portable source closure passed')
'''
            result = subprocess.run([sys.executable, '-c', script, str(copied), str(ROOT)],
                cwd=temporary, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT)},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
