import unittest
from run_a_submission_interlude import safe_compute_stack


class PauseBoundaryTests(unittest.TestCase):
    def stack(self, extra):
        return [{'pid': 123, 'thread_name': 'MainThread', 'frames': extra + [
            {'name': 'train_study_epoch', 'filename': '/project/study_student.py', 'line': 200}]}]

    def test_accepts_positive_forward_and_backward_evidence(self):
        for name, filename in [('forward','/opt/timm/models/convnext.py'),
                               ('backward','/site-packages/torch/autograd/__init__.py')]:
            self.assertTrue(safe_compute_stack(self.stack([{'name':name,'filename':filename}]),123)[0])

    def test_rejects_waiting_even_if_train_and_compute_frames_are_present(self):
        rows=self.stack([{'name':'forward','filename':'/opt/timm/model.py'},
                         {'name':'_get_data','filename':'/site-packages/torch/utils/data/dataloader.py'}])
        self.assertFalse(safe_compute_stack(rows,123)[0])

    def test_rejects_unknown_main_thread_pid_or_absent_positive_evidence(self):
        self.assertFalse(safe_compute_stack(self.stack([]),123)[0])
        self.assertFalse(safe_compute_stack(self.stack([{'name':'forward','filename':'/opt/timm/model.py'}]),999)[0])
        self.assertFalse(safe_compute_stack([{'pid':123,'thread_name':'worker','frames':[]}],123)[0])
