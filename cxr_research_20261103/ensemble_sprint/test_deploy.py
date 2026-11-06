"""CPU checks for immutable winner-only ensemble publication."""
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
from ensemble_sprint.deploy import (ROOT, _copy_sources, inspect_package, package_candidate,
                                   publish, require_winner)
from ensemble_sprint.predictor import FUSION_WEIGHTS


class EnsembleDeploymentTests(unittest.TestCase):
    def test_copied_regular_packages_win_with_original_project_on_sys_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'code'
            inventory = _copy_sources(copied)
            names = {item['path'] for item in inventory}
            self.assertIn('ensemble_sprint/__init__.py', names)
            self.assertIn('reconstruction/__init__.py', names)
            self.assertNotIn('sprint/__init__.py', names)
            for name in ('ensemble_sprint/__init__.py', 'reconstruction/__init__.py'):
                self.assertEqual((copied / name).read_bytes(), (ROOT / name).read_bytes())
            script = '''import importlib.util, pathlib, sys
copied, original = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(copied), str(original)]
import ensemble_sprint.deploy, ensemble_sprint.predictor, reconstruction
for module in (ensemble_sprint.deploy, ensemble_sprint.predictor, reconstruction):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(copied), module.__file__
for name in ('reconstruction.model', 'sprint.deploy'):
    origin = importlib.util.find_spec(name).origin
    assert pathlib.Path(origin).resolve().is_relative_to(copied), origin
print('copied package isolation passed')
'''
            result = subprocess.run([sys.executable, '-c', script, str(copied), str(ROOT)],
                cwd=temporary, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT)},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('copied package isolation passed', result.stdout)

    def fixture(self, root):
        comparison = root / 'final_comparison'
        comparison.mkdir()
        manifest = root / 'valid.csv'
        manifest.write_text('original validation identity')
        base = root / 'base.pt'
        base.write_bytes(b'validated base')
        members, assets = [], [{'id': 'base768', 'path': str(base),
                                'bytes': base.stat().st_size, 'sha256': sha256_file(base)}]
        for identity, branch, epochs in [('context512', 'global', 4), ('opacity512', 'opacity', 6)]:
            parent = root / identity
            parent.mkdir()
            weights = parent / 'final_fp16.pt'
            weights.write_bytes(identity.encode())
            members.append({'id': identity, 'branch': branch, 'epochs': epochs,
                            'image_size': 512, 'backbone_name': 'convnext_tiny.fb_in22k_ft_in1k_384',
                            'initial_weights': '/unneeded-training-state'})
            assets.append({'id': identity, 'path': str(weights), 'bytes': weights.stat().st_size,
                           'sha256': sha256_file(weights)})
        config = {'output': str(root), 'base_weights': str(base), 'base_weights_sha256': sha256_file(base),
                  'members': members, 'valid_manifest': str(manifest), 'fusion_weights': FUSION_WEIGHTS}
        for name in ('candidate_predictions.csv', 'candidate.image_predictions.csv'):
            (comparison / name).write_text(name)
        report = {'complete': True, 'candidate_won': True, 'candidate_inference_assets': assets,
                  'manifest_sha256': sha256_file(manifest), 'fusion_weights': FUSION_WEIGHTS,
                  'candidate_predictions_sha256': sha256_file(comparison / 'candidate_predictions.csv'),
                  'candidate_image_predictions_sha256': sha256_file(comparison / 'candidate.image_predictions.csv')}
        selection = {'candidate_won': True, 'report': str(comparison / 'report.json'),
                     'fusion_weights': FUSION_WEIGHTS, 'selection_checks': {'macro_strictly_better': True}}
        (comparison / 'report.json').write_text(json.dumps(report))
        (comparison / 'selection.json').write_text(json.dumps(selection))
        return config, comparison, selection

    def test_loser_never_reads_test_or_creates_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, comparison, selection = self.fixture(root)
            selection['candidate_won'] = False
            (comparison / 'selection.json').write_text(json.dumps(selection))
            with patch('ensemble_sprint.deploy.prediction_records') as records:
                with self.assertRaisesRegex(ValueError, 'winning'):
                    publish(config)
                records.assert_not_called()
            self.assertFalse((root / 'deployment_candidate').exists())

    def test_member_change_cannot_replace_validated_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = self.fixture(root)
            require_winner(config)
            (root / 'context512/final_fp16.pt').write_bytes(b'another epoch')
            with self.assertRaisesRegex(ValueError, 'validated assets'):
                require_winner(config)

    def test_package_moves_with_three_local_weight_paths_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = self.fixture(root)
            destination = root / 'deployment_candidate'
            with patch('ensemble_sprint.deploy._copy_sources', return_value=[]):
                package_candidate(config, destination)
            moved = root / 'relocated'
            destination.rename(moved)
            runtime = json.loads((moved / 'model/runtime_config.json').read_text())
            self.assertEqual(runtime['base_weights'], 'base768.pt')
            self.assertEqual([member['weights'] for member in runtime['members']],
                             ['context512.pt', 'opacity512.pt'])
            self.assertFalse(any('initial_weights' in member for member in runtime['members']))
            self.assertTrue(inspect_package(moved / 'model')['size_pass'])
            (moved / 'model/opacity512.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                inspect_package(moved / 'model')


if __name__ == '__main__':
    unittest.main()
