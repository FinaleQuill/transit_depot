"""CPU semantic and lifecycle checks of small-pack activation retention."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from reconstruction import train_local as frozen
from reconstruction import train_local_checkpoint_speed as adapter
from reconstruction import train_local_performance as performance
from reconstruction import train_local_earlystop as stopping
from reconstruction import early_stop_policy
from reconstruction.local_model import LocalStudyModel
from reconstruction.model import study_bce_loss
from training_compliance import read_events
from training_hooks import compliance_run


class StochasticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.LayerNorm(8)
        self.dropout = nn.Dropout(.3)
        self.fc = nn.Linear(8, 10)

    def forward_features(self, images):
        features = torch.nn.functional.gelu(self.conv(images))
        if self.training:
            mask = torch.empty(len(images), 1, 1, 1, device=features.device).bernoulli_(.8).div_(.8)
            features = features * mask
        return features

    def forward_head(self, features, pre_logits=False):
        values = self.dropout(self.norm(features.mean((-2, -1))))
        return values if pre_logits else self.fc(values)

    def get_classifier(self):
        return self.fc


def make_pack(views, regions=5):
    if regions == 1:
        pixels = torch.randn(views, 3, 6, 6)
    else:
        pixels = torch.randn(views, regions, 3, 6, 6)
    affines = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).expand(views, regions, -1, -1).clone()
    if regions > 1:
        affines[:, 1:, 0, 0] = .6
        affines[:, 1:, 1, 1] = .6
    return [{'images': pixels, 'region_affines': affines,
             'targets': torch.tensor([1., 0., 1., 0., 0., 1., 0., 1., 0., 0.])}]


class CheckpointExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(513)
        availability = patch.object(frozen.torch.cuda, 'is_available', return_value=False)
        availability.start(); self.addCleanup(availability.stop)

    def fixture(self, root):
        output = root / 'run'; output.mkdir()
        profile = root / 'checkpoint.json'
        profile.write_text(json.dumps(adapter.PROFILE))
        config = {'experiment_arm': 'global_local', 'use_local': True,
                  'checkpoint_mode': 'chunks', 'view_chunk_size': 8,
                  'output': str(output)}
        model = LocalStudyModel.from_backbone(StochasticBackbone(),
            gradient_checkpointing=True, view_chunk_size=8, attention_dim=12,
            checkpoint_mode='chunks')
        with torch.no_grad():
            model.region_query.normal_(std=.2)
            model.region_residual_weight.normal_(std=.1)
        return output, config, adapter.load_profile(profile), model

    def assert_tree_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), f'Tensor differs, max delta {(left-right).abs().max()}')
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def step(self, model, pack, config):
        model = deepcopy(model).train()
        ema = deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
        torch.manual_seed(937)
        # A pre-existing accumulation contribution checks additive gradients,
        # rather than just a clean first backward.
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        result = frozen.forward_pack(model, pack, config, return_components=True, device='cpu')
        loss = torch.stack([study_bce_loss(c['study_logits'], b['targets'])
                            for c, b in zip(result, pack)]).sum() / 32
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() if parameter.grad is not None else None
                     for name, parameter in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        optimizer.step()
        frozen.base.update_ema(ema, model, 1, .999)
        self.assertTrue(model.gradient_checkpointing)
        return {'logits': [c['study_logits'].detach() for c in result], 'loss': loss.detach(),
                'gradients': gradients, 'rng': rng, 'final_rng': torch.get_rng_state().clone(),
                'parameters': deepcopy(model.state_dict()), 'ema': deepcopy(ema.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict())}

    def test_real_dropout_gradients_adam_ema_and_rng_equal_for_small_and_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config, binding, model = self.fixture(Path(folder))
            for views in [1, 2, 3, 7]:
                with self.subTest(views=views):
                    pack = make_pack(views)
                    baseline = self.step(model, pack, config)
                    with adapter.checkpoint_adapter(config, binding):
                        actual = self.step(model, pack, config)
                    self.assert_tree_equal(baseline, actual)

    def test_boundary_counts_dimensions_and_state_dict_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config, binding, model = self.fixture(Path(folder))
            before = deepcopy(model.state_dict())
            seen = []
            def forward(m, pack, cfg, **kwargs):
                seen.append((adapter.pack_regions(pack), m.gradient_checkpointing))
                return 'sentinel'
            with patch.object(frozen, 'forward_pack', side_effect=forward):
                with adapter.checkpoint_adapter(config, binding):
                    model.train()
                    for pack in [make_pack(1), make_pack(2), make_pack(3), make_pack(7), make_pack(3, 1),
                                 make_pack(1) + make_pack(1)]:
                        self.assertEqual(frozen.forward_pack(model, pack, config), 'sentinel')
                        self.assertTrue(model.gradient_checkpointing)
            self.assertEqual(seen, [(5, False), (10, False), (15, True), (35, True), (3, False), (10, False)])
            self.assert_tree_equal(before, model.state_dict())

    def test_eval_no_grad_and_forward_exception_restore_original_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config, binding, model = self.fixture(Path(folder))
            flags = []
            def forward(m, *args, **kwargs):
                flags.append(m.gradient_checkpointing)
                if len(flags) == 3:
                    raise RuntimeError('synthetic forward failure')
                return None
            original_hooks = frozen.start_run, frozen.forward_pack, frozen.train_epoch
            with patch.object(frozen, 'forward_pack', side_effect=forward):
                with adapter.checkpoint_adapter(config, binding):
                    frozen.forward_pack(model.eval(), make_pack(1), config)
                    with torch.no_grad():
                        frozen.forward_pack(model.train(), make_pack(1), config)
                    with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                        frozen.forward_pack(model, make_pack(1), config)
                    self.assertTrue(model.gradient_checkpointing)
            self.assertEqual(flags, [True, True, False])
            self.assertEqual((frozen.start_run, frozen.forward_pack, frozen.train_epoch), original_hooks)

    def test_real_lifecycle_archives_profile_and_counts_without_mutating_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            output, config, binding, model = self.fixture(Path(folder))
            original_config = deepcopy(config)
            returned_metrics = {'regions': 20, 'images': 4, 'studies': 2, 'mean_loss': .4}
            def epoch(m, *args):
                m.train()
                for views in [1, 3]:
                    result = frozen.forward_pack(m, make_pack(views), config, device='cpu')
                    result[0].sum().backward()
                return returned_metrics
            @compliance_run
            def invocation():
                frozen.start_run(output, trainer='reconstruction.train_local', configuration=config,
                                 source_files=frozen.source_files())
                metrics = frozen.train_epoch(model, None, None, None, None, None,
                                             config, 3, {}, output)
                self.assertIs(metrics, returned_metrics)
                return {'state': 'paused'}
            with patch.object(frozen, 'train_epoch', side_effect=epoch):
                with adapter.checkpoint_adapter(config, binding):
                    invocation()
            self.assertEqual(config, original_config)
            self.assertEqual(returned_metrics, {'regions': 20, 'images': 4, 'studies': 2, 'mean_loss': .4})
            events = read_events(output / 'compliance/events.jsonl')
            activation = [e for e in events if e['event_type'] == 'checkpoint_execution_profile_activated']
            self.assertEqual(len(activation), 1)
            self.assertEqual(activation[0]['checkpoint_profile'], binding)
            epoch = [e for e in events if e['event_type'] == 'checkpoint_execution_epoch'][0]
            self.assertEqual(epoch['epoch'], 3)
            self.assertEqual(epoch['counts'], {'retained_packs': 1, 'retained_regions': 5,
                'checkpointed_packs': 1, 'checkpointed_regions': 15,
                'total_packs': 2, 'total_regions': 20, 'max_regions': 15,
                'pack_region_counts': {'5': 1, '15': 1}})
            session = output / 'compliance' / events[0]['session_evidence']
            self.assertEqual(json.loads((session / 'configuration.json').read_text()), config)
            manifest = {r['original_path']: r for r in json.loads((session / 'source_manifest.json').read_text())}
            for path in [*adapter.source_files(), Path(binding['path'])]:
                self.assertEqual(manifest[str(path)]['sha256'], adapter.sha(path))

    def test_profile_sha_and_model_policy_drift_fail_without_state_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config, binding, model = self.fixture(Path(folder))
            with adapter.checkpoint_adapter(config, binding):
                model.gradient_checkpointing = False
                with self.assertRaisesRegex(ValueError, 'checkpoint policy'):
                    frozen.forward_pack(model.train(), make_pack(1), config)
                self.assertFalse(model.gradient_checkpointing)
            Path(binding['path']).write_text(json.dumps({**adapter.PROFILE, 'max_retained_regions': 15}))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                with adapter.checkpoint_adapter(config, binding):
                    pass

    def test_profile_rejects_unregistered_fields_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, binding, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                adapter.validate_profile({**adapter.PROFILE, 'tf32': True})
            with self.assertRaises(ValueError):
                adapter.validate_profile({**adapter.PROFILE, 'max_retained_regions': True})
            link = root / 'link.json'; link.symlink_to(binding['path'])
            with self.assertRaisesRegex(ValueError, 'regular'):
                adapter.load_profile(link)
            with self.assertRaisesRegex(ValueError, 'frozen E2'):
                with adapter.checkpoint_adapter({**config, 'experiment_arm': 'global_control'}, binding):
                    pass

    def test_three_nested_adapters_keep_original_configuration_and_stopping(self):
        from reconstruction.test_train_local_performance import sample_profile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, config, binding, model = self.fixture(root)
            config.update(epochs=6, min_delta=.0001, smoke_studies=0, workers=2)
            loader_profile = root / 'loader.json'
            loader_profile.write_text(json.dumps(sample_profile(workers=1)))
            stop_policy = root / 'stop.json'
            stop_policy.write_text(json.dumps(early_stop_policy.POLICY))
            stop_binding = early_stop_policy.load_policy(stop_policy)
            (output / 'baseline.json').write_text(json.dumps({'epoch': 0, 'macro_auc': .8}))
            def epoch(m, *args):
                result = frozen.forward_pack(m.train(), make_pack(1), config, device='cpu')
                result[0].sum().backward()
                return {'regions': 5}
            @compliance_run
            def invocation():
                frozen.start_run(output, trainer='reconstruction.train_local', configuration=config,
                                 source_files=frozen.source_files())
                for number in [1, 2, 3]:
                    metrics = frozen.train_epoch(model, None, None, None, None, None,
                                                 config, number, {}, output)
                    metrics.update(epoch=number, macro_auc=.79)
                    frozen.base.save_checkpoint(output / 'last.pt', {'fixture_epoch': number}, number, 'last_full_state')
                    with (output / 'history.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics) + '\n')
                    frozen.epoch_completed(number, metrics)
                return {'state': 'paused'}
            with patch.object(frozen, 'train_epoch', side_effect=epoch):
                with performance.profile_adapter(config, loader_profile) as execution:
                    with stopping.stopping_adapter(config, stop_binding):
                        with adapter.checkpoint_adapter(config, binding):
                            invocation()
            events = read_events(output / 'compliance/events.jsonl')
            session = output / 'compliance' / events[0]['session_evidence']
            session_config = json.loads((session / 'configuration.json').read_text())
            self.assertEqual(session_config.pop('execution_profile'), execution)
            self.assertEqual(session_config, config)
            self.assertEqual([e['epoch'] for e in events if e['event_type'] == 'checkpoint_execution_epoch'], [1, 2, 3])
            self.assertEqual([e['epoch'] for e in events if e['event_type'] == 'epoch_completed'], [1, 2, 3])
            marker = json.loads((output / 'STOP').read_text())
            self.assertEqual(marker['reason'], 'patience_exhausted')
            self.assertEqual(marker['epoch'], 3)
            manifest = {r['original_path'] for r in json.loads((session / 'source_manifest.json').read_text())}
            for source in [*adapter.source_files(), *stopping.source_files(), *performance.source_files(),
                           Path(binding['path']), loader_profile, stop_policy]:
                self.assertIn(str(source), manifest)


if __name__ == '__main__':
    unittest.main()
