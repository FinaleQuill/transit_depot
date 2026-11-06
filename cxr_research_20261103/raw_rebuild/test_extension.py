"""Bounded continuation tests with synthetic checkpoints and real CPU Adam.

These checks never load competition images or checkpoints. The CUDA calls are
replaced using the established CPU fixtures, while the real training loop,
learning-rate updates, state restoration and persistence code are exercised.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from breakthrough import test_train as kernel_fixtures
from raw_rebuild import test_train as lifecycle_fixtures
from raw_rebuild import train as trainer
from raw_rebuild.extension import (compatible_best_configuration, identity,
                                   only_epoch_extension, validate_extension)


CHANGED_SOURCES = ('raw_rebuild/train.py', 'raw_rebuild/launch.py', 'breakthrough/train.py')
ADDED_SOURCE = 'raw_rebuild/extension.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_for(prior, evidence, checkpoint, path):
    receipt = dict(schema='authorized_raw_dino_18_to_23_v1',
        authorization='User explicitly requested five additional epochs, all other settings unchanged',
        completed_parent_epoch=18, parent_evidence_sha256=identity(prior['evidence']),
        new_evidence_sha256=identity(evidence), parent_checkpoint_sha256=sha(checkpoint))
    path.write_text(json.dumps(receipt))
    return receipt


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def receipt_fixture(self, root):
        old = dict(epochs=18, seed=43, patience=4, min_epochs=4, backbone_lr=1e-5,
                   input_sha256={'train': 'a'*64})
        new = dict(old, epochs=23, lr_schedule_epochs=18)
        sources = {name: hashlib.sha256(('old-'+name).encode()).hexdigest()
                   for name in CHANGED_SOURCES + ('reconstruction/data.py',)}
        parent = dict(configuration=old, source_sha256=sources,
                      initialization={'source':'public'}, test_used=False)
        next_sources = dict(sources)
        next_sources.update({name: hashlib.sha256(('new-'+name).encode()).hexdigest()
                             for name in CHANGED_SOURCES + (ADDED_SOURCE,)})
        evidence = dict(parent, configuration=new, source_sha256=next_sources)
        prior = dict(epoch=18, evidence=parent, optimizer={'fixture_step':8334},
                     progress={'attempted_updates':8334}, stale=1)
        checkpoint, receipt_path = root/'last.pt', root/'extension.json'
        checkpoint.write_bytes(b'synthetic immutable full epoch18 checkpoint')
        receipt_for(prior, evidence, checkpoint, receipt_path)
        return prior, evidence, checkpoint, receipt_path

    def test_valid_authorization_is_read_only_and_binds_the_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            prior, evidence, checkpoint, receipt = self.receipt_fixture(Path(tmp))
            prior_before, evidence_before, bytes_before = deepcopy(prior), deepcopy(evidence), checkpoint.read_bytes()
            lineage = validate_extension(prior, evidence, checkpoint, receipt)
            self.assertEqual(lineage['parent_configuration'], prior['evidence']['configuration'])
            self.assertEqual(lineage['receipt_sha256'], sha(receipt))
            self.assertEqual(prior, prior_before)
            self.assertEqual(evidence, evidence_before)
            self.assertEqual(checkpoint.read_bytes(), bytes_before)

    def test_wrong_epoch_other_configuration_and_source_inventory_are_rejected(self):
        faults = ('epoch17', 'epoch19', 'seed', 'patience', 'learning_rate', 'input',
                  'wrong_limit', 'wrong_schedule', 'source_extra', 'source_unrelated',
                  'source_missing', 'evidence_usage')
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                prior, evidence, checkpoint, receipt = self.receipt_fixture(Path(tmp))
                if fault.startswith('epoch'):
                    prior['epoch'] = int(fault[5:])
                elif fault == 'seed':
                    evidence['configuration']['seed'] += 1
                elif fault == 'patience':
                    evidence['configuration']['patience'] += 1
                elif fault == 'learning_rate':
                    evidence['configuration']['backbone_lr'] *= 2
                elif fault == 'input':
                    evidence['configuration']['input_sha256'] = {'train':'b'*64}
                elif fault == 'wrong_limit':
                    evidence['configuration']['epochs'] = 24
                elif fault == 'wrong_schedule':
                    evidence['configuration']['lr_schedule_epochs'] = 23
                elif fault == 'source_extra':
                    evidence['source_sha256']['unreviewed.py'] = 'a'*64
                elif fault == 'source_unrelated':
                    evidence['source_sha256']['reconstruction/data.py'] = 'b'*64
                elif fault == 'source_missing':
                    del evidence['source_sha256'][ADDED_SOURCE]
                else:
                    evidence['test_used'] = True
                # Even a receipt with updated digests cannot authorize changes
                # outside the narrow structural continuation contract.
                receipt_for(prior, evidence, checkpoint, receipt)
                with self.assertRaises(ValueError):
                    validate_extension(prior, evidence, checkpoint, receipt)

    def test_stale_receipt_rejects_changed_source_checkpoint_and_authorization(self):
        for fault in ('source_digest', 'checkpoint_bytes', 'parent_digest', 'new_digest', 'authorization'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                prior, evidence, checkpoint, receipt = self.receipt_fixture(Path(tmp))
                if fault == 'source_digest':
                    evidence['source_sha256']['raw_rebuild/train.py'] = 'c'*64
                elif fault == 'checkpoint_bytes':
                    checkpoint.write_bytes(checkpoint.read_bytes() + b' changed')
                else:
                    value = json.loads(receipt.read_text())
                    key = {'parent_digest':'parent_evidence_sha256',
                           'new_digest':'new_evidence_sha256', 'authorization':'authorization'}[fault]
                    value[key] = 'invalid'
                    receipt.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    validate_extension(prior, evidence, checkpoint, receipt)

    def test_old_best_configuration_is_accepted_only_through_exact_lineage(self):
        old = dict(epochs=18, seed=41, patience=4)
        current = dict(old, epochs=23, lr_schedule_epochs=18)
        lineage = dict(parent_configuration=deepcopy(old))
        self.assertTrue(only_epoch_extension(old, current))
        self.assertTrue(compatible_best_configuration(old, current, lineage))
        self.assertTrue(compatible_best_configuration(current, current, None))
        for bad in (dict(old, seed=42), dict(old, epochs=17), dict(old, patience=5)):
            self.assertFalse(compatible_best_configuration(bad, current, lineage))
        self.assertFalse(compatible_best_configuration(old, current, None))
        self.assertFalse(compatible_best_configuration(old, current, dict(parent_configuration={})))

    def test_real_kernel_retains_every_original_lr_and_clamps_extra_epochs(self):
        helper = kernel_fixtures.TrainerTests()
        helper.setUp()
        first = kernel_fixtures.TinyStudy(dropout=.25)
        second = deepcopy(first)
        ema_first, ema_second = deepcopy(first), deepcopy(second)
        opt_first, opt_second = helper.optimizer(first), helper.optimizer(second)
        data = kernel_fixtures.FakeDataset()
        old = dict(helper.config(), epochs=18)
        new = dict(old, epochs=23, lr_schedule_epochs=18)
        first_progress = dict(attempted_updates=0, optimizer_updates=0, skipped_updates=0)
        second_progress = deepcopy(first_progress)
        rates = []
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for epoch in range(1, 24):
                saved_rng = trainer.rng_state()
                first_rates, second_rates = [], []
                first_step, second_step = opt_first.step, opt_second.step
                def step_first(*args, **kwargs):
                    first_rates.append([g['lr'] for g in opt_first.param_groups])
                    return first_step(*args, **kwargs)
                def step_second(*args, **kwargs):
                    second_rates.append([g['lr'] for g in opt_second.param_groups])
                    return second_step(*args, **kwargs)
                with patch.object(opt_first, 'step', side_effect=step_first):
                    helper.cpu_epoch(first, ema_first, data, opt_first, kernel_fixtures.CPUScaler(),
                                     old, epoch, first_progress, output)
                trainer.restore_rng(saved_rng)
                with patch.object(opt_second, 'step', side_effect=step_second):
                    helper.cpu_epoch(second, ema_second, data, opt_second, kernel_fixtures.CPUScaler(),
                                     new, epoch, second_progress, output)
                self.assertEqual(first_rates, second_rates)
                self.assert_tree_equal(first.state_dict(), second.state_dict())
                self.assert_tree_equal(ema_first.state_dict(), ema_second.state_dict())
                self.assert_tree_equal(opt_first.state_dict(), opt_second.state_dict())
                self.assertEqual(first_progress, second_progress)
                if epoch >= 19:
                    for step_rates in second_rates:
                        self.assertEqual(step_rates, [g['base_lr'] * .1 for g in opt_second.param_groups])
                rates.extend(second_rates)
            status = json.loads((output/'progress.json').read_text())
            self.assertEqual(status['epochs'], 23)
            self.assertEqual(status['epoch'], 23)
        self.assertEqual(len(rates), 46)  # two optimizer boundaries per tiny epoch

    def lifecycle(self, root, stack, *, early_stop_at_18=False):
        scores = tuple(.5+i*.001 for i in range(18)) + (.4,)*5
        if early_stop_at_18:
            scores = tuple(.5+i*.001 for i in range(14)) + (.4,)*9
        helper = lifecycle_fixtures.LifecycleTests()
        config, _, models, training, comparisons, decision = helper.fixture(root, stack, scores=scores)
        config.update(epochs=18, min_epochs=4, patience=4)
        old_sources = []
        for name in CHANGED_SOURCES + ('reconstruction/data.py',):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# old synthetic source: '+name+'\n')
            old_sources.append(path)
        sources = stack.enter_context(patch.object(trainer, 'source_files', return_value=old_sources))
        trainer.run(config)
        checkpoint = root/'state/last.pt'
        prior = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.assertEqual(prior['epoch'], 18)
        for name in CHANGED_SOURCES + (ADDED_SOURCE,):
            path = root/name
            path.write_text('# reviewed extension fixture: '+name+'\n')
        new_sources = old_sources + [root/ADDED_SOURCE]
        sources.return_value = new_sources
        extended = dict(config, epochs=23, lr_schedule_epochs=18)
        evidence = deepcopy(prior['evidence'])
        evidence.update(configuration=extended,
                        source_sha256={str(p.relative_to(root)):sha(p) for p in new_sources})
        receipt = root/'extension.json'
        receipt_for(prior, evidence, checkpoint, receipt)
        return extended, receipt, prior, models, training, comparisons, decision

    def test_resume_preserves_old_best_and_every_full_state_before_epoch19(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, receipt, prior, models, training, comparisons, decision = self.lifecycle(root, stack)
            paths = [root/'state/last.pt', root/'state/best.pt', root/'state/best_fp16.pt']
            original = {str(path):path.read_bytes() for path in paths}
            archive = root/'archived_epoch18.pt'
            archive.write_bytes(paths[0].read_bytes())
            inner_train, inner_compare = training.side_effect, comparisons.side_effect
            first_update_seen = []
            def inspect_resume(model, ema, dataset, optimizer, scaler, weights, cfg, epoch, progress, output):
                first_update_seen.append(epoch)
                self.assertEqual(epoch, 19)
                for path in paths:
                    self.assertEqual(path.read_bytes(), original[str(path)])
                self.assert_tree_equal(model.state_dict(), prior['model'])
                self.assert_tree_equal(ema.state_dict(), prior['ema'])
                self.assert_tree_equal(optimizer.state_dict(), prior['optimizer'])
                self.assert_tree_equal(scaler.state_dict(), prior['scaler'])
                self.assert_tree_equal(progress, prior['progress'])
                self.assert_tree_equal(trainer.rng_state(), prior['rng'])
                return inner_train(model, ema, dataset, optimizer, scaler, weights, cfg, epoch, progress, output)
            def pause_after_comparison(*args, **kwargs):
                result = inner_compare(*args, **kwargs)
                (root/'output/STOP').touch()
                return result
            training.side_effect = inspect_resume
            comparisons.side_effect = pause_after_comparison
            result = trainer.run(config, resume=True, extension_receipt=receipt)
            self.assertEqual(first_update_seen, [19])
            self.assertEqual(result['state'], 'paused')
            resumed = torch.load(paths[0], map_location='cpu', weights_only=False)
            self.assertEqual(resumed['epoch'], 19)
            self.assertEqual(resumed['best_epoch'], 18)
            self.assertEqual(resumed['stale'], prior['stale']+1)
            self.assertEqual(resumed['meaningful'], prior['meaningful'])
            self.assertEqual(resumed['progress']['optimizer_updates'], prior['progress']['optimizer_updates']+1)
            self.assertEqual(archive.read_bytes(), original[str(paths[0])])
            for path in paths[1:]:
                self.assertEqual(path.read_bytes(), original[str(path)])
            self.assertEqual((root/'output/best_fp16.pt').read_bytes(), original[str(paths[2])])
            # A later ordinary resume carries the bound lineage; it must not
            # need or reapply the one-time epoch18 exception.
            (root/'output/STOP').unlink()
            training.side_effect = inner_train
            result = trainer.run(config, resume=True)
            self.assertEqual(result['optimizer_updates'], 20)
            self.assertEqual(training.call_args.args[7], 20)
            self.assertEqual(paths[2].read_bytes(), original[str(paths[2])])
            self.assertEqual(archive.read_bytes(), original[str(paths[0])])

    def test_already_satisfied_early_stop_does_not_run_an_extra_epoch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            config, receipt, prior, _, training, _, _ = self.lifecycle(root, stack, early_stop_at_18=True)
            self.assertEqual(prior['stale'], 4)
            checkpoint_bytes = (root/'state/last.pt').read_bytes()
            calls = training.call_count
            result = trainer.run(config, resume=True, extension_receipt=receipt)
            self.assertEqual(result['state'], 'early_stopped')
            self.assertEqual(training.call_count, calls)
            self.assertEqual((root/'state/last.pt').read_bytes(), checkpoint_bytes)


if __name__ == '__main__':
    unittest.main()
