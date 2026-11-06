"""Persisted student status across CPU-only simulated training lifecycles."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_distill as student
from kd_common import sha256_file


class FakeModel:
    def to(self, *args, **kwargs):
        return self

    def requires_grad_(self, enabled):
        return self

    def state_dict(self):
        return {'fixture': torch.tensor([1.])}

    def load_state_dict(self, state, **kwargs):
        return None


class FakeOptimizer:
    def __init__(self):
        self.hook = SimpleNamespace(remove=lambda: setattr(self, 'removed', True))
        self.removed = False

    def register_step_post_hook(self, function):
        return self.hook

    def state_dict(self):
        return {'fixture': True}

    def load_state_dict(self, state):
        pass


class StudentStatusLifecycleTests(unittest.TestCase):
    def fixture(self, *, epochs=3, patience=0, schema=9):
        self.addCleanup(student.random.setstate, student.random.getstate())
        self.addCleanup(student.np.random.set_state, student.np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(setattr, torch.backends.cudnn, 'benchmark', torch.backends.cudnn.benchmark)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root/'images').mkdir()
        for name in ('fold.json', 'train.csv', 'valid.csv'):
            (root/name).write_text('{}')
        config = student.DistillConfig(root/'fold.json', None, root/'out', root/'images',
            training_mode='supervised_bce', alpha=0., epochs=epochs, patience=patience)
        if schema == 8:
            config = replace(config, training_mode='distillation', alpha=.2, soft_labels=root/'cache')
        train = [{'subject_id': 'train', 'study_key': 'train/a', 'image_id': 'a'}]
        valid = [{'subject_id': 'valid', 'study_key': 'valid/b', 'image_id': 'b'}]
        plan = {'source_train_manifest': {'path': str(root/'train.csv')},
            'source_manifest_sha256': sha256_file(root/'train.csv'),
            'official_holdout': {'path': str(root/'valid.csv'), 'sha256': sha256_file(root/'valid.csv')}}
        return SimpleNamespace(root=root, config=config, plan=plan, train=train, valid=valid,
                               schema=schema, observed=[], evaluations=[], models=0)

    def run_fixture(self, case, aucs, *, stop=None, fail=False, resume=False):
        models_before = case.models
        config = replace(case.config, resume=case.config.output/'last.pt') if resume else case.config
        if resume:
            (config.output/'STOP').unlink(missing_ok=True)
        optimizer = FakeOptimizer()
        handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
        originals = dict(handlers)

        def register(sig, handler):
            previous = handlers[sig]
            handlers[sig] = handler
            return previous

        def build_model(**kwargs):
            case.models += 1
            if stop == 'before_first_epoch':
                (config.output/'STOP').touch()
            return FakeModel()

        def train_epoch(model, ema, loader, opt, scaler, device, dtype, epoch,
                        supplied, progress, cache, weights):
            case.observed.append(json.loads((config.output/'status.json').read_text()))
            if fail:
                raise RuntimeError('simulated epoch failure')
            progress['backward_batches'] += 2
            progress['optimizer_updates'] += 1
            progress['ema_updates'] += 1
            if stop == 'signal':
                handlers[signal.SIGINT](signal.SIGINT, None)
            if stop == 'file':
                (config.output/'STOP').touch()
            return {'loss': .5, 'images': 2}

        def validate(*args):
            value = aucs[len(case.evaluations)]
            case.evaluations.append(value)
            return {'metrics': {'macro_auc': value}}

        objective = {'cache': None, 'protocol': student.supervised_protocol(config)
                     if case.schema == 9 else {'fixture': 'KD'},
                     'schema_version': case.schema, 'teacher_provenance': [], 'cache_bindings': []}
        scaler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
        with ExitStack() as stack:
            for target, replacement in (
                ('train_distill.require_training_runtime', lambda: None),
                ('train_distill.capture_source_version', lambda *args: {}),
                ('cv_splits.read_fold_plan', lambda *args: case.plan),
                ('dataset.read_manifest', lambda path, **kwargs: case.train if kwargs['split']=='train' else case.valid),
                ('train_distill.prepare_student_objective', lambda *args: objective),
                ('train_overnight.compute_loss_config', lambda *args, **kwargs: {'pos_weight': [1.]*10}),
                ('train_distill.prepare_pretrained_provenance', lambda *args: {'weights_path': 'unused'}),
                ('train_distill.validate_pretrained_provenance', lambda *args: None),
                ('model_overnight.build_overnight_model', build_model),
                ('timm.utils.ModelEmaV3', lambda *args, **kwargs: SimpleNamespace(module=FakeModel())),
                ('train_overnight.build_optimizer', lambda *args, **kwargs: optimizer),
                ('train_overnight.capture_rng', lambda: {}),
                ('train_overnight.restore_rng', lambda *args: None),
                ('dataset_overnight.make_overnight_loader', lambda *args, **kwargs: object()),
                ('train_distill._train_epoch', train_epoch),
                ('train_distill.evaluate_student', validate),
                ('train_distill.record_checkpoint', lambda *args, **kwargs: {'saved_at_utc': 'fixture', 'record_sha256': 'a'*64}),
                ('torch.cuda.is_available', lambda: True),
                ('torch.cuda.is_bf16_supported', lambda: True),
                ('torch.cuda.set_device', lambda *args: None),
                ('torch.cuda.manual_seed_all', lambda *args: None),
                ('torch.amp.GradScaler', lambda *args, **kwargs: scaler),
                ('signal.signal', register),
            ):
                stack.enter_context(patch(target, replacement))
            # Redirect only this orchestration's device tensor; no CUDA APIs run.
            tensor = torch.tensor
            stack.enter_context(patch('torch.tensor', lambda *args, **kwargs:
                                     tensor(*args, **{k:v for k,v in kwargs.items() if k!='device'})))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                result = student.run_training(config)
            finally:
                self.assertEqual(handlers, originals)
                if case.models > models_before:
                    self.assertTrue(optimizer.removed)
        return result, json.loads((config.output/'status.json').read_text())

    def test_initial_and_between_epoch_status_is_running_for_kd_and_bce(self):
        for schema in (8, 9):
            with self.subTest(schema=schema):
                case = self.fixture(schema=schema)
                result, status = self.run_fixture(case, [.7, .8, .81])
                self.assertEqual([s['state'] for s in case.observed], ['running']*3)
                self.assertEqual([s['completed_epochs'] for s in case.observed], [0, 1, 2])
                self.assertEqual(result['state'], 'completed')
                self.assertEqual(status['state'], 'completed')
                self.assertEqual(status['best_epoch'], 3)

    def test_early_stop_preserves_best_and_last_without_false_pause(self):
        case = self.fixture(epochs=4, patience=1)
        result, status = self.run_fixture(case, [.8, .7])
        self.assertEqual(status['state'], 'early_stopped')
        self.assertEqual(status['completed_epochs'], 2)
        self.assertEqual(status['best_epoch'], 1)
        best = torch.load(case.config.output/'best.pt', weights_only=True)
        last = torch.load(case.config.output/'last.pt', weights_only=True)
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(last['epoch'], 2)

    def test_user_stop_is_paused_only_after_the_completed_epoch_is_saved(self):
        for stop in ('signal', 'file'):
            with self.subTest(stop=stop):
                case = self.fixture()
                result, status = self.run_fixture(case, [.8], stop=stop)
                self.assertEqual(result['state'], 'paused')
                self.assertEqual(status['completed_epochs'], 1)
                self.assertEqual(status['last_checkpoint_sha256'], sha256_file(case.config.output/'last.pt'))
                self.assertEqual(case.observed[0]['state'], 'running')

    def test_stop_during_initialization_does_not_start_an_epoch(self):
        case = self.fixture()
        result, status = self.run_fixture(case, [], stop='before_first_epoch')
        self.assertEqual(status['state'], 'paused')
        self.assertEqual(status['completed_epochs'], 0)
        self.assertEqual(case.observed, [])

    def test_epoch_failure_is_failed_and_cleans_up_handlers(self):
        case = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'simulated epoch failure'):
            self.run_fixture(case, [], fail=True)
        status = json.loads((case.config.output/'status.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertTrue(status['incomplete_epoch'])

    def test_resume_replaces_stale_pause_and_terminal_resume_stays_terminal(self):
        case = self.fixture(epochs=3)
        self.run_fixture(case, [.7], stop='file')
        case.evaluations.clear()
        result, status = self.run_fixture(case, [.8, .81], resume=True)
        self.assertEqual([s['state'] for s in case.observed], ['running']*3)
        self.assertEqual([s['completed_epochs'] for s in case.observed], [0, 1, 2])
        self.assertEqual(status['state'], 'completed')
        models_before = case.models
        # A completed checkpoint repairs/retains terminal status before any model build.
        result, status = self.run_fixture(case, [], resume=True)
        self.assertEqual(result['state'], 'already_terminal')
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(case.models, models_before)


if __name__ == '__main__':
    unittest.main()
