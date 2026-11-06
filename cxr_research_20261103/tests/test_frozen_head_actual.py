"""CPU fixtures for immutable actual-pair preparation and complete valid audits."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import frozen_head_actual as actual
from audit_student_submission import IMAGE_HEADER, SUBMISSION_HEADER


class ActualHeadTests(unittest.TestCase):
    def plan(self):
        return json.loads((actual.ROOT / 'docs/FROZEN_HEAD_PLAN_20261101.json').read_text())

    def test_registered_pair_rejects_unregistered_members_or_weights(self):
        value = self.plan()
        self.assertEqual(actual.registered_pair(value, 'A_F1_B')['members'], ['A_F1', 'B'])
        with self.assertRaises(ValueError):
            actual.registered_pair(value, 'arbitrary_new_pair')
        modified = copy.deepcopy(value)
        modified['pair_candidates']['A_F1_B']['weights'][0][0] = .5
        with self.assertRaises(ValueError):
            actual.registered_pair(modified, 'A_F1_B')

    def test_probability_identity_requires_all_ten_exact_float_bytes(self):
        value = np.linspace(.1, .9, 10, dtype=np.float64)
        self.assertTrue(actual.probability_identity(value, value.copy()))
        changed = value.copy()
        changed[0] = np.nextafter(changed[0], 1)
        self.assertFalse(actual.probability_identity(value, changed))
        self.assertFalse(actual.probability_identity(value, value.astype(np.float32)))
        self.assertFalse(actual.probability_identity(value, value[:9]))

    def test_prepare_pair_keeps_sources_unchanged_and_binds_merged_patch_identity(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root = Path(folder)
            plan = self.plan()
            fits = root/'fits'
            fits.mkdir()
            for encoder in ('A', 'B'):
                base = root/encoder
                base.mkdir()
                entry = plan['encoders'][encoder]
                dimension = entry['feature_dim']
                state = {'CPU_fixture_encoder': torch.ones(3).half(),
                    'head.fc.weight': torch.zeros(10, dimension).half(),
                    'head.fc.bias': torch.zeros(10).half()}
                torch.save(state, base/'student_fp16.pt')
                config = {'schema_version': 1, 'artifact_type': 'experimental_student_fp16',
                    'backbone_name': entry['backbone_name'],
                    'model_config': {'backbone_name': entry['backbone_name'], 'drop_rate': .2,
                        'drop_path_rate': .1 if encoder == 'A' else .15, 'gradient_checkpointing': False},
                    'weights_file': 'student_fp16.pt', 'weights_sha256': actual.sha256_file(base/'student_fp16.pt'),
                    'class_ids': list(range(1, 11)), 'image_size': 768, 'precision': 'float16',
                    'source_checkpoint_sha256': 'e'*64,
                    'official_valid_manifest_sha256': plan['cache']['valid_manifest_sha256']}
                (base/'config.json').write_text(json.dumps(config))
                entry.update(base_model_dir=str(base), base_config_sha256=actual.sha256_file(base/'config.json'),
                             base_weights_sha256=config['weights_sha256'])
            plan_path = root/'plan.json'
            plan_path.write_text(json.dumps(plan))
            plan_sha = actual.sha256_file(plan_path)
            head = fits/'A_F1'
            head.mkdir()
            torch.save({'head.fc.weight': torch.full((10, 768), .125).half(),
                        'head.fc.bias': torch.zeros(10).half()}, head/'patch_fp16.pt')
            patch_sha = actual.sha256_file(head/'patch_fp16.pt')
            cfg = {'schema_version': 1, 'artifact_type': 'frozen_head_student_fp16',
                **{key: plan['encoders']['A'][key] for key in ('backbone_name', 'base_model_dir',
                    'base_config_sha256', 'base_weights_sha256', 'feature_dim')},
                'class_ids': list(range(1, 11)), 'replaced_class_ids': list(range(1, 11)),
                'image_size': 768, 'precision': 'float16', 'tta': 'hflip2', 'epochs': 12,
                'patch_file': 'patch_fp16.pt', 'patch_sha256': patch_sha,
                'training_plan_sha256': plan_sha, 'feature_cache_metadata_sha256': 'f'*64}
            (head/'config.json').write_text(json.dumps(cfg))
            receipt = {'state': 'completed', 'encoder': 'A', 'recipe': 'F1',
                'training_plan_sha256': plan_sha, 'feature_cache_metadata_sha256': 'f'*64,
                'final_epoch': 12, 'completed_epochs': 12, 'head_patch_fp16_sha256': patch_sha,
                'valid_used_for_training': False, 'valid_used_for_checkpoint_selection': False, 'test_used': False}
            (head/'training_receipt.json').write_text(json.dumps(receipt))
            before = {str(p): actual.sha256_file(p) for p in root.rglob('*') if p.is_file()}
            model_dir, config, sources = actual.prepare_pair(plan=plan_path, plan_sha256=plan_sha,
                fits_root=fits, pair_name='A_F1_B', output=root/'actual', cache_metadata_sha256='f'*64)
            self.assertEqual(config['model_weights'], plan['pair_candidates']['A_F1_B']['weights'])
            self.assertEqual(config['rule_name'], 'A_F1_B')
            self.assertIsNone(config['decoder_candidate'])
            merged = json.loads((model_dir/'A_F1/config.json').read_text())
            self.assertEqual(merged['source_checkpoint_sha256'], patch_sha)
            self.assertEqual(sources['A_F1']['patch_sha256'], patch_sha)
            self.assertEqual(sources['B']['kind'], 'original_frozen_encoder')
            self.assertEqual(before, {name: actual.sha256_file(name) for name in before})
            with self.assertRaises(FileExistsError):
                actual.prepare_pair(plan=plan_path, plan_sha256=plan_sha, fits_root=fits,
                    pair_name='A_F1_B', output=root/'actual', cache_metadata_sha256='f'*64)

    def test_independent_valid_replay_rejects_changed_Study_means(self):
        from student_experiment_evaluation import score_probabilities, validation_labels
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records, images, studies = [], {}, {}
            for index in range(6):
                subject, study = f'p{index}', f's{index}'
                views = 2 if index % 2 == 0 else 1
                values = []
                for view in range(views):
                    image = f'i{index}_{view}'
                    records.append({'image_id': image, 'subject_id': subject, 'study_id': study,
                        'study_key': f'{subject}/{study}', 'split': 'valid', 'targets': [index % 2]*10})
                    probability = np.full(10, .2 + .6*(index % 2) + .05*view)
                    images[image] = probability
                    values.append(probability)
                studies[(subject, study)] = sum(values, np.zeros(10))/views
            with (root/'valid.image_predictions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(IMAGE_HEADER)
                for row in records:
                    for column, value in enumerate(images[row['image_id']], 1):
                        writer.writerow([row['image_id'], row['subject_id'], row['study_id'], column, value])
            def write_studies():
                with (root/'valid_predictions.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(SUBMISSION_HEADER)
                    for (subject, study), values in studies.items():
                        for column, value in enumerate(values, 1):
                            writer.writerow([subject, study, column, value])
            write_studies()
            labels, counts = validation_labels(records)
            scores = score_probabilities(labels, studies, counts)
            audit = actual.audit_complete_valid(root, records, scores)
            self.assertEqual(audit['num_images'], 9)
            self.assertTrue(audit['all_Study_means_bitwise_equal_to_original_order_FP64_replay'])
            studies[('p0', 's0')][0] += .01
            write_studies()
            with self.assertRaisesRegex(ValueError, 'independent original-order'):
                actual.audit_complete_valid(root, records, scores)

    def test_evidence_copy_excludes_models_weights_and_feature_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, report = root/'output', root/'report'
            (output/'model').mkdir(parents=True)
            (output/'valid').mkdir()
            (output/'model/config.json').write_text('{}')
            (output/'model/large_weights.pt').write_bytes(b'CPU_FIXTURE_WEIGHTS')
            (output/'valid/validation_report.json').write_text('{"complete":true}')
            (output/'actual_audit.json').write_text('{"complete":true}')
            actual.persist_reports(output, report)
            self.assertTrue((report/'model_top_config.json').is_file())
            self.assertTrue((report/'valid/validation_report.json').is_file())
            self.assertFalse(list(report.rglob('*.pt')))


if __name__ == '__main__':
    unittest.main()
