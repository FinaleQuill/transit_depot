import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_exploration_queue as prepare


class PrepareExplorationQueueTest(unittest.TestCase):
    def test_retry_evaluation_directory_is_registered_in_resolution_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            root = temp / 'cloud_project_v2'
            base = root / 'artifacts/resolution_explore_20261103'
            queue = base / 'queue'
            run640 = temp / 'runs/031_resolution_train640_eval768_20261103'

            source_names = (
                'train_distill.py', 'exploration_train.py', 'roi_scale.py', 'tight_crop.py',
                'spatial_mil.py', 'exploration_evaluate.py', 'run_exploration_queue.py',
                'summarize_exploration.py', 'inference_inputs.py',
                'inference_reproducibility.py', 'prepare_exploration_queue.py',
            )
            for name in source_names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'test source: {name}\n')

            source = root / 'train_distill.py'
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            session = run640 / 'compliance/sessions/test_session'
            session.mkdir(parents=True)
            (session / 'source_manifest.json').write_text(json.dumps([
                {'original_path': str(source), 'sha256': source_digest},
            ]))
            (run640 / 'compliance').mkdir(parents=True, exist_ok=True)
            (run640 / 'compliance/events.jsonl').write_text(json.dumps({
                'event_type': 'training_started', 'session_evidence': 'sessions/test_session',
            }) + '\n')
            run640.mkdir(parents=True, exist_ok=True)
            (run640 / 'config.json').write_text(json.dumps({
                'validation_size': 768, 'output': str(run640), 'image_size': 640,
                'seed': 44, 'epochs': 3,
            }))

            original_jobs = []
            for size, name in ((640, 'res640'), (768, 'res768'), (896, 'res896')):
                prereg = base / 'prereg' / f'train{size}_eval768.txt'
                prereg.parent.mkdir(parents=True, exist_ok=True)
                prereg.write_text('hypothesis\neffect\ncost\nfailure\n')
                output = run640 if size == 640 else temp / f'runs/{name}'
                arguments = ['--image-size', str(size), '--output', str(output)]
                original_jobs.append({
                    'name': name,
                    'run_dir': str(output),
                    'script': 'train_distill.py',
                    'arguments': arguments,
                    'preregistration': str(prereg),
                    'preregistration_sha256': hashlib.sha256(prereg.read_bytes()).hexdigest(),
                })
            plan_path = base / 'queue_plan.json'
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps({'jobs': original_jobs}))

            retry_prereg = base / 'prereg/eval_res640_retry1.txt'
            retry_prereg.write_text('hypothesis\neffect\ncost\nfailure\n')
            recovery_path = queue / 'attempts/active_recovery.json'
            recovery_path.parent.mkdir(parents=True, exist_ok=True)
            recovery_path.write_text(json.dumps({
                'retry_job': 'eval_res640_retry1',
                'retry_preregistration': str(retry_prereg),
            }))

            roi_gallery = root / 'artifacts/roi_magnify_explore_20261103_v2_2/gallery'
            roi_gallery.mkdir(parents=True)
            (roi_gallery / 'metrics.json').write_text(json.dumps({
                'count': 48,
                'method_version': 'body_envelope_magnify_only_v2_2',
                'code_sha256': {},
                'items': [{'diagnostics': {'effective_magnification': 1.0}}],
            }))
            (roi_gallery / 'SELF_REVIEW.md').write_text('reviewed fixture\n')

            config_dir = base / 'configs'
            config_dir.mkdir(parents=True)
            for name, roi, mil in (('roi', True, False), ('mil', False, True)):
                (config_dir / f'{name}_seed44.json').write_text(json.dumps({
                    'output': str(temp / f'runs/{name}'),
                    'image_size': 768,
                    'validation_size': 768,
                    'validation_tta': 'hflip2',
                    'roi': roi,
                    'mil': mil,
                    'init_student_ema_sha256': 'a' * 64,
                }))
            anchor = root / 'artifacts/kd_v1/student_upper_bound_20261030/models/A/best_ema_fp32.pt'
            anchor.parent.mkdir(parents=True)
            anchor.write_bytes(b'anchor fixture')
            (anchor.parent / 'backup_receipt.json').write_text('{}\n')

            prereg_names = (
                'compare_resolution.txt', 'morning_report.txt', 'eval_champion041.txt',
                'roi_magnify_seed44.txt', 'compare_roi.txt', 'mil_seed44.txt',
                'compare_final.txt', 'compare_resolution_best.txt', 'compare_final.txt',
                'eval_res640.txt', 'eval_res640_best.txt', 'eval_res768.txt',
                'eval_res768_best.txt', 'eval_res896.txt', 'eval_res896_best.txt',
                'eval_roi.txt', 'eval_roi_best.txt', 'eval_mil.txt', 'eval_mil_best.txt',
            )
            for name in set(prereg_names):
                path = base / 'prereg' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('hypothesis\neffect\ncost\nfailure\n')

            with patch.object(prepare, 'ROOT', root), \
                    patch.object(prepare, 'BASE', base), \
                    patch.object(prepare, 'QUEUE', queue), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare.main()

            plan = json.loads(plan_path.read_text())
            retry_job = next(job for job in plan['jobs'] if job['name'] == 'eval_res640_retry1')
            expected_dir = str(base / 'evaluations/res640_retry1')
            self.assertEqual(retry_job['output'], expected_dir)
            self.assertEqual(plan['evaluation_reports']['res640'],
                             str(base / 'evaluations/res640_retry1/report.json'))
            resolution_compare = next(job for job in plan['jobs']
                                      if job['name'] == 'compare_resolution')
            self.assertIn(f'res640={expected_dir}', resolution_compare['inputs'])
            self.assertNotIn(f'res640={base / "evaluations/res640"}',
                             resolution_compare['inputs'])


if __name__ == '__main__':
    unittest.main()
