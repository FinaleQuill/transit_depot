"""Small CSV/receipt fixtures; all GPU queries and process starts are mocked."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

import build_training_evidence as evidence_builder
import compare_student_candidates as comparison
from deploy_kd import sha256_file
import run_student_experiments as experiments


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class StudentExperimentTests(unittest.TestCase):
    def arguments(self, root, *, execute=False):
        images = root / "images"
        images.mkdir(exist_ok=True)
        return argparse.Namespace(stage="compare", output_root=root / "students", soft_labels=root / "cache",
            fold_plan=root / "folds.json", image_root=images, python=Path("/test/python"),
            epochs=25, num_workers=0, execute=execute, teacher_collection=root / "teachers.json")

    def test_compare_stage_retains_paired_seed_resolution_and_effective_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            jobs = experiments.experiment_commands(stage=args.stage, output_root=args.output_root,
                soft_labels=args.soft_labels, fold_plan=args.fold_plan, image_root=args.image_root,
                python=args.python, epochs=args.epochs, num_workers=args.num_workers)
            self.assertEqual([(j['image_size'], j['seed']) for j in jobs], [(512, 43), (768, 43)])
            for job in jobs:
                command = job['command']
                self.assertEqual(command[command.index('--validation-tta')+1], 'hflip2')
                batch = int(command[command.index('--batch-size')+1])
                accumulation = int(command[command.index('--accumulation-steps')+1])
                self.assertEqual(batch*accumulation, 32)
                self.assertNotIn('--resume', command)
            self.assertFalse(args.output_root.exists())

    def test_preview_never_queries_gpu_or_starts_a_process(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(directory))
            with (patch.object(experiments.subprocess, 'Popen', side_effect=AssertionError('No process')),
                  patch.object(experiments.subprocess, 'check_output', side_effect=AssertionError('No GPU query'))):
                self.assertEqual(experiments.run_experiments(args), 0)
            self.assertFalse(args.output_root.exists())

    def test_extra_seed_stages_are_explicit_and_existing_best_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            old = args.output_root / 'tiny512_seed47'
            old.mkdir(parents=True)
            (old / 'best.pt').write_bytes(b'fixture marker, never loaded')
            jobs = experiments.experiment_commands(stage='extra512', output_root=args.output_root,
                soft_labels=args.soft_labels, fold_plan=args.fold_plan, image_root=args.image_root)
            self.assertEqual([(j['image_size'], j['seed']) for j in jobs], [(512, 47)])
            command = jobs[0]['command']
            self.assertEqual(command[command.index('--resume')+1], str(old / 'last.pt'))

    def test_incomplete_teacher_rejection_precedes_gpu_and_student_start(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory), execute=True)
            with (patch('teacher_collection.read_teacher_collection', side_effect=ValueError('unfinished teacher')),
                  patch.object(experiments.subprocess, 'Popen', side_effect=AssertionError('No process')),
                  patch.object(experiments.subprocess, 'check_output', side_effect=AssertionError('No GPU query'))):
                with self.assertRaisesRegex(ValueError, 'unfinished teacher'):
                    experiments.run_experiments(args)
            self.assertFalse(args.output_root.exists())

    def test_paused_first_student_does_not_start_second(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(directory), execute=True)
            fake_cache = SimpleNamespace(metadata={'tta': 'identity_hflip',
                'teachers': [{'checkpoint_sha256': 'a'*64}]})
            def spawn(command, **kwargs):
                output = Path(command[command.index('--output')+1])
                write_json(output / 'status.json', {'state': 'paused'})
                return SimpleNamespace(wait=lambda: 0)
            with ExitStack() as stack:
                stack.enter_context(patch('teacher_collection.read_teacher_collection',
                    return_value={'teachers': [{'checkpoint': 'fake_teacher.pt'}]}))
                stack.enter_context(patch('cv_splits.read_fold_plan',
                    return_value={'source_train_manifest': {'path': 'fake_train.csv'}}))
                stack.enter_context(patch('dataset.read_manifest', return_value=[]))
                stack.enter_context(patch('distill_tta_data.load_soft_labels', return_value=fake_cache))
                stack.enter_context(patch('distill_data.sha256_file', return_value='a'*64))
                stack.enter_context(patch.object(experiments.subprocess, 'check_output', return_value=''))
                process = stack.enter_context(patch.object(experiments.subprocess, 'Popen', side_effect=spawn))
                self.assertEqual(experiments.run_experiments(args), 2)
                self.assertEqual(process.call_count, 1)
            self.assertFalse((args.output_root / 'tiny768_seed43').exists())


class SavedProbabilityTests(unittest.TestCase):
    def fixtures(self, root):
        manifest = root / 'valid.csv'
        fields = ['image_id', 'split', 'study_key', 'targets', 'cache_relative_path', 'subject_id', 'study_id']
        keys = [(f'p{i//2}', f's{i}') for i in range(6)]
        targets = np.asarray([[i % 2]*10 for i in range(6)])
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, (subject, study) in enumerate(keys):
                for view in range(2 if i == 0 else 1):
                    writer.writerow({'image_id': f'i{i}_{view}', 'split': 'valid',
                        'study_key': f'{subject}/{study}', 'targets': json.dumps(targets[i].tolist()),
                        'cache_relative_path': f'valid/{subject}/{study}/{view}.jpg',
                        'subject_id': subject, 'study_id': study})
        vectors = ([.1, .9, .8, .3, .4, .6], [.6, .7, .3, .9, .2, .4])
        roots, values = [], []
        for name, vector in zip(['tiny512', 'tiny768'], vectors):
            directory = root / name
            directory.mkdir()
            matrix = np.repeat(np.asarray(vector)[:, None], 10, axis=1)
            with (directory / 'valid_predictions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Subject_id', 'Study_id', 'Predict_class', 'Probability'])
                for key, row in zip(keys, matrix):
                    for c, value in enumerate(row, 1):
                        writer.writerow([*key, c, repr(float(value))])
            write_json(directory / 'validation_report.json', {'complete': True,
                'manifest_sha256': sha256_file(manifest),
                'predictions_sha256': sha256_file(directory / 'valid_predictions.csv'),
                'macro_auc': float(roc_auc_score(targets[:, 0], matrix[:, 0])), 'tta': 'hflip2'})
            roots.append(directory)
            values.append(matrix)
        return manifest, roots, targets, values

    def test_saved_soft_ensemble_auc_and_patient_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, roots, targets, values = self.fixtures(root)
            weights = root / 'weights.json'
            write_json(weights, {'weighted': [1, 3]})
            result = comparison.compare(validation_dirs=roots, manifest=manifest, output=root/'comparison',
                                        combinations_json=weights, bootstrap=20, seed=43)
            expected_mean = roc_auc_score(targets[:, 0], ((values[0]+values[1])/2)[:, 0])
            expected_weighted = roc_auc_score(targets[:, 0], (.25*values[0]+.75*values[1])[:, 0])
            self.assertAlmostEqual(result['candidates']['equal_mean']['macro_auc'], expected_mean)
            self.assertAlmostEqual(result['candidates']['weighted']['macro_auc'], expected_weighted)
            self.assertEqual(result['num_patients'], 3)
            self.assertEqual(result['num_studies'], 6)
            self.assertEqual(result['reference'], 'tiny512')
            self.assertEqual(result['bootstrap']['reference'], 'tiny512')
            self.assertNotIn('tiny512', result['bootstrap']['results'])
            self.assertAlmostEqual(result['candidates']['tiny768']['macro_auc_delta_from_reference'],
                roc_auc_score(targets[:, 0], values[1][:, 0]) -
                roc_auc_score(targets[:, 0], values[0][:, 0]))
            self.assertTrue(result['bootstrap']['paired'])
            self.assertEqual(result['bootstrap']['sampling_unit'], 'patient_with_all_Studies')
            self.assertEqual(result['bootstrap']['results']['tiny768']['macro_auc_delta']['valid_resamples'], 20)
            self.assertFalse(result['weights_fitted'])
            self.assertFalse(result['deployment_selected'])
            self.assertFalse(result['T4_latency_certified'])

    def test_cli_second_member_reference_preserves_weights_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, roots, targets, values = self.fixtures(root)
            weights = root / 'weights.json'
            write_json(weights, {'weighted': [1, 3]})
            output = root / 'comparison'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(comparison.main(['--validation-dirs', *map(str, roots),
                    '--names', 'seed43', 'seed44', '--reference', 'seed44',
                    '--manifest', str(manifest), '--combinations-json', str(weights),
                    '--bootstrap', '20', '--seed', '43', '--output', str(output)]), 0)
            result = json.loads((output / 'comparison_report.json').read_text())
            self.assertEqual(result['reference'], 'seed44')
            self.assertEqual(result['bootstrap']['reference'], 'seed44')
            self.assertEqual(list(result['candidates']), ['seed43', 'seed44', 'weighted', 'equal_mean'])
            self.assertEqual([p['name'] for p in result['source_validations']], ['seed43', 'seed44'])
            self.assertEqual([p['validation_dir'] for p in result['source_validations']],
                             [str(p.resolve()) for p in roots])
            np.testing.assert_array_equal(result['fixed_probability_weights']['weighted'],
                                          [[.25] * 10, [.75] * 10])
            matrices = {'seed43': values[0], 'seed44': values[1],
                        'weighted': .25 * values[0] + .75 * values[1],
                        'equal_mean': (values[0] + values[1]) / 2}
            reference_auc = roc_auc_score(targets, values[1], average=None)
            for name, matrix in matrices.items():
                aucs = roc_auc_score(targets, matrix, average=None)
                score = result['candidates'][name]
                np.testing.assert_allclose(score['per_class_auc'], aucs)
                np.testing.assert_allclose(score['per_class_delta_from_reference'], aucs - reference_auc)
                self.assertAlmostEqual(score['macro_auc_delta_from_reference'],
                                       float(np.mean(aucs - reference_auc)))
            self.assertEqual(list(result['bootstrap']['results']), ['seed43', 'weighted', 'equal_mean'])
            # Independently replay patient draws with sklearn, without calling the helper under test.
            draws = np.random.default_rng(43).integers(0, 3, (20, 3))
            for name in result['bootstrap']['results']:
                rows = []
                for draw in draws:
                    indices = np.asarray([i for patient in draw for i in (2 * patient, 2 * patient + 1)])
                    delta = (roc_auc_score(targets[indices], matrices[name][indices], average=None) -
                             roc_auc_score(targets[indices], values[1][indices], average=None))
                    rows.append([*delta, float(np.mean(delta))])
                expected = np.quantile(rows, [.025, .975], axis=0).T
                actual = result['bootstrap']['results'][name]
                intervals = actual['per_class_delta'] + [actual['macro_auc_delta']]
                self.assertEqual([v['valid_resamples'] for v in intervals], [20] * 11)
                np.testing.assert_allclose([v['ci_95'] for v in intervals], expected, atol=1e-15)

    def test_reference_must_be_an_individual_name_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, roots, _, _ = self.fixtures(root)
            weights = root / 'weights.json'
            write_json(weights, {'weighted': [1, 3]})
            for reference in ('missing', 'weighted', 'equal_mean', '', 'tiny768'):
                with self.subTest(reference=reference):
                    output = root / 'not_created' / 'comparison'
                    with self.assertRaisesRegex(ValueError, 'individual'):
                        comparison.compare(validation_dirs=roots, names=['seed43', 'seed44'],
                            manifest=manifest, output=output, combinations_json=weights,
                            reference=reference)
                    self.assertFalse(output.parent.exists())
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'individual'):
                comparison.main(['--validation-dirs', *map(str, roots), '--reference', 'equal_mean',
                    '--manifest', str(manifest), '--output', str(root / 'cli_invalid')])
            self.assertFalse((root / 'cli_invalid').exists())

    def test_wrong_manifest_or_missing_study_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, roots, _, _ = self.fixtures(root)
            report_path = roots[0]/'validation_report.json'
            report = json.loads(report_path.read_text())
            report['manifest_sha256'] = 'b'*64
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                comparison.compare(validation_dirs=roots, manifest=manifest, output=root/'wrong')
            self.assertFalse((root/'wrong').exists())
            report['manifest_sha256'] = sha256_file(manifest)
            prediction_path = roots[0]/'valid_predictions.csv'
            lines = prediction_path.read_text().splitlines(keepends=True)
            prediction_path.write_text(''.join(lines[:-10]))
            report['predictions_sha256'] = sha256_file(prediction_path)
            write_json(report_path, report)
            with self.assertRaises(ValueError):
                comparison.compare(validation_dirs=roots, manifest=manifest, output=root/'missing')
            self.assertFalse((root/'missing').exists())

    def test_cluster_bootstrap_is_paired_and_reproducible(self):
        targets = np.asarray([[0]*10, [1]*10, [0]*10, [1]*10])
        a = targets * .8 + .1
        subjects = np.asarray(['p0', 'p0', 'p1', 'p1'])
        inputs = dict(reference=a, identical=a.copy())
        first = comparison.paired_intervals(targets, inputs, subjects, samples=12, seed=7)
        second = comparison.paired_intervals(targets, inputs, subjects, samples=12, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first['results']['identical']['macro_auc_delta']['ci_95'], [0., 0.])
        explicit = comparison.paired_intervals(targets, inputs, subjects, samples=12, seed=7,
                                               reference='identical')
        self.assertEqual(explicit['reference'], 'identical')
        self.assertEqual(list(explicit['results']), ['reference'])
        self.assertEqual(explicit['results']['reference']['macro_auc_delta']['ci_95'], [0., 0.])
        with self.assertRaisesRegex(ValueError, 'reference'):
            comparison.paired_intervals(targets, inputs, subjects, samples=0, seed=7, reference='missing')

    def test_same_auc_cannot_hide_changed_probability_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, roots, targets, values = self.fixtures(root)
            prediction = roots[0]/'valid_predictions.csv'
            with prediction.open(newline='') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row['Probability'] = repr(.1 + .8*float(row['Probability']))
            with prediction.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            self.assertEqual(roc_auc_score(targets[:, 0], values[0][:, 0]),
                             roc_auc_score(targets[:, 0], .1+.8*values[0][:, 0]))
            with self.assertRaisesRegex(ValueError, 'SHA256 binding'):
                comparison.compare(validation_dirs=roots, manifest=manifest, output=root/'changed')
            self.assertFalse((root/'changed').exists())


class TrainingEvidenceAssemblyTests(unittest.TestCase):
    def fixtures(self, root, *, resume_only=False):
        model = root/'model'
        model.mkdir()
        teacher = root/'teacher_evidence'
        teacher.mkdir()
        write_json(teacher/'evidence_manifest.json', {'artifact_type': 'kd_training_evidence',
            'schema_version': 1, 'runs': [], 'source_archives': [], 'console_logs': [],
            'supporting_files': [], 'source_version_timeline': [], 'console_coverage': []})
        students, members = [], []
        for index, seed in enumerate((43, 47)):
            student = root/f'student{index}'
            student.mkdir()
            (student/'best.pt').write_bytes(f'small fake checkpoint {index}'.encode())
            write_json(student/'status.json', {'state': 'completed', 'completed_epochs': 2})
            protocol = {'alpha': .2, 'temperature': 2, 'teacher_tta': 'identity_hflip'}
            write_json(student/'config.json', {'seed': seed, 'image_size': 512, 'distillation': protocol})
            native = student/'evidence'
            archive = native/'source_0'
            archive.mkdir(parents=True)
            (archive/'train_distill.py').write_text('# native fixture source\n')
            (archive/'environment_lock.txt').write_text('torch==fixture\n')
            write_json(archive/'source_manifest.json', {'fixture': True})
            console = native/'console_0.log'
            console.write_text('student epoch=2 valid_AUC=.8\n')
            write_json(native/'command_0.json', {'cwd': str(root), 'source_archive': str(archive),
                'console_log': str(console), 'started_at': '2026-10-26T01:00:00+00:00',
                'ended_at': '2026-10-26T02:00:00+00:00', 'exit_code': 0,
                'console_bytes': console.stat().st_size, 'coverage_from_start': 'complete',
                'preexisting_run_artifacts': ['last.pt', 'config.json'] if resume_only else []})
            members.append({'seed': seed, 'image_size': 512, 'distillation': protocol,
                            'source_checkpoint_sha256': sha256_file(student/'best.pt')})
            students.append(student)
        write_json(model/'config.json', {'members': members})
        return model, teacher, students

    def build(self, model, teacher, students, output, *, execute=True):
        with (patch.object(evidence_builder, 'validate_deployment_config'),
              patch.object(evidence_builder, 'deployment_inventory', return_value={'size_pass': True})):
            return evidence_builder.build_evidence(model_dir=model, teacher_evidence=teacher,
                student_runs=students, output=output, execute=execute)

    def test_multiple_students_keep_distinct_roles_and_native_log_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, teacher, students = self.fixtures(root)
            self.build(model, teacher, students, root/'combined')
            manifest = json.loads((root/'combined/evidence_manifest.json').read_text())
            self.assertEqual([r['id'] for r in manifest['runs']], ['student_0', 'student_1'])
            self.assertTrue(all(r['role'] == 'student' for r in manifest['runs']))
            self.assertTrue(all(r['coverage_from_start'] == 'invocation_only' for r in manifest['console_logs']))
            self.assertTrue(all(r['coverage_from_start'] == 'complete' for r in manifest['console_coverage']))
            self.assertFalse(manifest['assembly']['source_runs_modified'])

    def test_resume_only_receipts_cannot_claim_complete_training_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, teacher, students = self.fixtures(root, resume_only=True)
            self.build(model, teacher, students, root/'combined')
            manifest = json.loads((root/'combined/evidence_manifest.json').read_text())
            self.assertTrue(all(r['coverage_from_start'] == 'partial' for r in manifest['console_coverage']))
            self.assertTrue(all(r['coverage_from_start'] == 'invocation_only' for r in manifest['console_logs']))

    def test_wrong_student_checkpoint_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, teacher, students = self.fixtures(root)
            (students[1]/'best.pt').write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'differs from the deployed checkpoint'):
                self.build(model, teacher, students, root/'combined')
            self.assertFalse((root/'combined').exists())

    def test_preview_does_not_create_an_evidence_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, teacher, students = self.fixtures(root)
            result = self.build(model, teacher, students, root/'combined', execute=False)
            self.assertFalse(result['execute'])
            self.assertFalse((root/'combined').exists())

    def test_wrong_resolution_or_distillation_cannot_supply_member_evidence(self):
        for change, message in [({'image_size': 768}, 'resolution'),
                                ({'distillation': {'alpha': 0}}, 'distillation protocol')]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, teacher, students = self.fixtures(root)
                path = students[1]/'config.json'
                value = json.loads(path.read_text())
                value.update(change)
                write_json(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.build(model, teacher, students, root/'combined')
                self.assertFalse((root/'combined').exists())

    def test_unfinalized_or_truncated_fresh_logs_remain_partial(self):
        for change in ({'ended_at': None}, {'console_bytes': 99999}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, teacher, students = self.fixtures(root)
                receipt = students[1]/'evidence/command_0.json'
                value = json.loads(receipt.read_text())
                value.update(change)
                write_json(receipt, value)
                self.build(model, teacher, students, root/'combined')
                manifest = json.loads((root/'combined/evidence_manifest.json').read_text())
                coverage = {c['run_id']: c['coverage_from_start'] for c in manifest['console_coverage']}
                self.assertEqual(coverage['student_0'], 'complete')
                self.assertEqual(coverage['student_1'], 'partial')


if __name__ == '__main__':
    unittest.main()
