"""Pure-supervision ancestry stays teacher-free through actual evidence packaging."""
import json
import shutil
import unittest

from build_training_evidence import build_evidence
from export_student import make_deployment_config
from package_kd_submission import collect_training_evidence, package_submission, unique_pretrains
from tests import test_kd_ensemble_package as fixtures


class SupervisedPackageTests(unittest.TestCase):
    def fixture(self, mixed=False):
        case = fixtures.EnsemblePackageTests('test_two_member_mixed_resolution_package_keeps_all_runtime_and_sources')
        case.setUp()
        self.addCleanup(case.doCleanups)
        member = case.members[0]
        member['training_mode'] = 'supervised_bce'
        member['source_checkpoint_schema_version'] = 9
        member['teacher_provenance'] = []
        member['distillation'] = {'enabled': False, 'alpha': 0.0, 'loss': 'weighted_bce',
            'class_reduction': 'mean', 'study_equal_weighting': True, 'student_training_size': 512,
            'validation_tta': 'hflip2', 'deployment_tta': 'hflip2', 'deployment_ensemble': False}
        if not mixed:
            (case.model / case.members[1]['weights_file']).unlink()
            case.members = case.members[:1]
        case.config = make_deployment_config(case.members, source_manifest_sha256=case.train_hash,
            fold_plan_sha256=case.fold_hash, official_valid_manifest_sha256=case.valid_hash)
        (case.model/'config.json').write_text(json.dumps(case.config))
        teachers, students = case.evidence()
        run_config = json.loads((students[0]/'config.json').read_text())
        run_config.update(training_mode='supervised_bce', alpha=0.0, soft_labels=None)
        (students[0]/'config.json').write_text(json.dumps(run_config))
        if not mixed:
            shutil.rmtree(teachers)
            # The supervised execution source need not contain unused cache loaders.
            for name in ('distill_data.py', 'distill_tta_data.py'):
                (case.root/'student_archive_0'/name).unlink()
        return case, teachers, students

    def test_supervised_package_requires_no_teacher_files_or_pretraining_source(self):
        case, teachers, students = self.fixture()
        self.assertFalse(teachers.exists())
        self.assertEqual(len(unique_pretrains(case.config)), 1)
        evidence = case.root/'supervised_evidence'
        preview = build_evidence(model_dir=case.model, student_runs=students, output=evidence)
        self.assertIsNone(preview['teacher_evidence'])
        self.assertFalse(evidence.exists())
        build_evidence(model_dir=case.model, student_runs=students, output=evidence, execute=True)
        manifest = json.loads((evidence/'evidence_manifest.json').read_text())
        self.assertEqual([r['id'] for r in manifest['runs']], ['student'])
        self.assertIsNone(manifest['assembly']['teacher_evidence'])
        result = package_submission(model_dir=case.model, project_root=case.source,
            training_evidence=evidence, output=case.root/'package')
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['training_modes'], ['supervised_bce'])
        self.assertTrue(result['training_evidence']['complete'], result['training_evidence']['pending_items'])
        self.assertEqual(len(result['pretraining_sources']), 1)
        self.assertEqual({r['id'] for r in result['training_evidence']['runs']}, {'student'})

    def test_mixed_bce_and_kd_preserve_all_required_teacher_and_student_runs(self):
        case, teachers, students = self.fixture(mixed=True)
        # The first, representative root member is teacher-free; KD's ancestry is still required.
        self.assertEqual(case.config['teacher_provenance'], [])
        evidence = case.root/'mixed_evidence'
        with self.assertRaisesRegex(ValueError, 'require.*teacher'):
            build_evidence(model_dir=case.model, student_runs=students, output=evidence)
        build_evidence(model_dir=case.model, teacher_evidence=teachers,
            student_runs=students, output=evidence, execute=True)
        result = package_submission(model_dir=case.model, project_root=case.source,
            training_evidence=evidence, output=case.root/'package')
        self.assertTrue(result['training_evidence']['complete'], result['training_evidence']['pending_items'])
        self.assertEqual(result['training_modes'], ['supervised_bce', 'distillation'])
        self.assertEqual(len(result['pretraining_sources']), 2)
        self.assertEqual({r['id'] for r in result['training_evidence']['runs']},
                         {f'teacher_fold_{f}' for f in range(5)} | {'student_0', 'student_1'})

    def test_packaging_itself_rejects_mislabeled_training_mode(self):
        case, _, students = self.fixture()
        evidence = case.root/'evidence'
        build_evidence(model_dir=case.model, student_runs=students, output=evidence, execute=True)
        run_config = json.loads((students[0]/'config.json').read_text())
        run_config['training_mode'] = 'distillation'
        (students[0]/'config.json').write_text(json.dumps(run_config))
        with self.assertRaisesRegex(ValueError, 'Training mode'):
            collect_training_evidence(evidence, case.root/'copied', case.config)


if __name__ == '__main__':
    unittest.main()
