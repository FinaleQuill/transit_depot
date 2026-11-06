"""Select the same declared model loader for the prediction and timing CLIs."""
from __future__ import annotations

import json
from pathlib import Path


def load_submission_predictor(model_dir):
    config = json.loads((Path(model_dir) / 'config.json').read_text())
    kind = config.get('artifact_type')
    if kind == 'fixed_student_ensemble':
        from student_ensemble import StudentEnsemble
        return StudentEnsemble(model_dir)
    if kind == 'experimental_student_fp16':
        from submit_student_candidate import SubmissionStudentPredictor
        return SubmissionStudentPredictor(model_dir)
    if kind == 'fixed_classwise_student_pair':
        if config.get('decoder_candidate') == 'JPEG_draft_target768':
            from student_draft_pair import DraftStudentPair
            return DraftStudentPair(model_dir)
        if config.get('decoder_candidate') is not None:
            raise ValueError('Unknown fixed-pair decoder')
        from student_fixed_pair import FixedStudentPair
        return FixedStudentPair(model_dir)
    from deploy_kd import Predictor
    return Predictor(model_dir)
