"""Submit an explicitly declared Tiny/Base student through the shared CSV writer.

The loader changes for Base; pixel processing, fixed TTA, single-image inference,
Study averaging and CSV output remain shared with predict.py and cost.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy_kd import deployment_inventory
from predict import predict_submission
from student_experiment_evaluation import ExperimentalPredictor


class SubmissionStudentPredictor(ExperimentalPredictor):
    def __init__(self, model_dir):
        declared = json.loads((Path(model_dir) / 'config.json').read_text())
        if declared.get('tta') != 'hflip2':
            raise ValueError('This delivery requires the fixed validated hflip2 configuration')
        super().__init__(model_dir, tta='hflip2')
        self.config = {**self.config, 'inference_files': ['student_fp16.pt', 'config.json']}
        self.inventory = deployment_inventory(self.model_dir, self.config)
        if not self.inventory['size_pass']:
            raise ValueError('Student inference assets exceed the size limit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = predict_submission(**vars(args), predictor_factory=SubmissionStudentPredictor)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
