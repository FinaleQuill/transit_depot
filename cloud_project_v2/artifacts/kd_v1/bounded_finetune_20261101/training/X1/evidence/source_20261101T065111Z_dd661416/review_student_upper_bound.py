"""Bounded official-valid review and fixed multiscale student experiments.

No teachers are loaded, no parameters are trained, and test is never accepted.
The source FP16 exports and production inference code remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from deploy_kd import MEAN, STD, read_submission, sha256_file
from student_experiment_evaluation import (
    ExperimentalPredictor, checked_cached_validation, evaluate_predictor,
    score_probabilities, validation_labels,
)


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def original_labels(manifest):
    from dataset import read_manifest
    records = read_manifest(manifest, split='valid')
    return records, *validation_labels(records)


def read_bound_predictions(directory, manifest, labels):
    directory = Path(directory)
    report = json.loads((directory / 'validation_report.json').read_text())
    source = directory / 'valid_predictions.csv'
    if (report.get('complete') is not True
            or report['manifest_sha256'] != sha256_file(manifest)
            or report.get('predictions_sha256') != sha256_file(source)):
        raise ValueError('Require complete, SHA-bound original-valid predictions')
    values = read_submission(source)
    if set(values) != set(labels):
        raise ValueError('Exact original-valid Study coverage is required')
    return values


def fixed_mean(sources, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (len(sources),) or not np.isfinite(weights).all()
            or (weights < 0).any() or not weights.sum() > 0):
        raise ValueError('Use fixed nonnegative global probability weights')
    keys = set(sources[0])
    if any(set(source) != keys for source in sources):
        raise ValueError('Cannot combine mismatched Study coverage')
    weights = weights / weights.sum()
    return {key: sum(weight * np.asarray(source[key], dtype=np.float64)
                     for source, weight in zip(sources, weights)) for key in sorted(keys)}


def compare(args):
    from compare_student_candidates import compare as run_comparison
    initial = json.loads(args.initial_plan.read_text())
    second = json.loads(args.second_plan.read_text())
    output = args.output
    combinations = {**initial['initial_candidates_before_computing_combinations'],
                    **second['coarse_probability_blends']}
    config = output.parent / 'fixed_combinations.json'
    if config.exists():
        if json.loads(config.read_text()) != combinations:
            raise ValueError('Previously registered combinations changed')
    else:
        write_new(config, combinations)
    result = run_comparison(validation_dirs=args.validations, names=['A', 'B', '033'],
        reference='033', manifest=args.manifest, output=output,
        combinations_json=config, bootstrap=args.bootstrap, seed=20261031)
    print(json.dumps({name: row['macro_auc'] for name, row in result['candidates'].items()}, indent=2), flush=True)


class ResolutionPredictor(ExperimentalPredictor):
    """An explicit inference-resolution override of unchanged student weights."""
    def __init__(self, model_dir, *, image_size):
        if image_size not in (640, 768, 896):
            raise ValueError('Only the three predeclared resolutions are accepted')
        super().__init__(model_dir, tta='hflip2')
        from inference_inputs import ExactU8Input
        self.config = {**self.config, 'training_image_size': self.config['image_size'],
                       'image_size': image_size, 'resolution_override': True}
        self.members = [{**self.members[0], 'image_size': image_size}]
        self.config['members'] = self.members
        self.input_lookup = ExactU8Input(self.torch, self.device, MEAN, STD, image_sizes=[image_size])


def multiscale(args):
    import torch
    from inference_reproducibility import configure_inference
    records, labels, counts = original_labels(args.manifest)
    if len(records) != 1422 or len(labels) != 859:
        raise ValueError('This review requires the unchanged full official holdout')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    plan = json.loads(args.second_plan.read_text())
    sizes = plan['multiscale']['resolutions']
    if sizes != [640, 768, 896]:
        raise ValueError('Registered resolution set changed')
    models, all_scores, sources = {}, {}, []
    for name, model_dir, original in zip(('A', 'B'), args.models, args.original_validations):
        checked_cached_validation(original, model_dir=model_dir,
                                  image_root=args.image_root, manifest=args.manifest)
        scales = []
        for size in sizes:
            if size == 768:
                directory = original
            else:
                directory = args.output / f'{name}_{size}_hflip2'
                configure_inference()
                start = time.monotonic()
                predictor = ResolutionPredictor(model_dir, image_size=size)
                result = evaluate_predictor(predictor, image_root=args.image_root,
                    manifest=args.manifest, output_dir=directory)
                result.update(training_image_size=768, evaluation_image_size=size,
                    evaluation_script_sha256=sha256_file(__file__),
                    elapsed_seconds=time.monotonic()-start,
                    source_weights_sha256=predictor.config['weights_sha256'])
                write_new(directory/'resolution_override.json', result)
                del predictor
                gc.collect()
                torch.cuda.empty_cache()
            values = read_bound_predictions(directory, args.manifest, labels)
            scales.append(values)
            all_scores[f'{name}_{size}'] = score_probabilities(labels, values, counts)
            sources.append({'name':name, 'inference_size':size, 'directory':str(directory),
                'predictions_sha256':sha256_file(directory/'valid_predictions.csv')})
            print(name, size, all_scores[f'{name}_{size}']['macro_auc'], flush=True)
        models[name] = fixed_mean(scales, [1,1,1])
        all_scores[f'{name}_multiscale'] = score_probabilities(labels, models[name], counts)
    combined = fixed_mean([models['A'], models['B']], [1,1])
    all_scores['A_B_multiscale_equal'] = score_probabilities(labels, combined, counts)
    write_new(args.output/'multiscale_report.json', {'sources':sources,'scores':all_scores,
        'manifest_sha256':sha256_file(args.manifest), 'second_plan_sha256':sha256_file(args.second_plan),
        'teacher_models_loaded':False, 'new_training':False, 'test_used':False,
        'formal_submission_ready':False,
        'note':'Linear fixed probability means commute with Study averaging; real combined predictor still requires direct verification.'})
    print(json.dumps({name:r['macro_auc'] for name,r in all_scores.items()}, indent=2), flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='mode',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--manifest',type=Path,required=True)
    common.add_argument('--second-plan',type=Path,required=True)
    common.add_argument('--output',type=Path,required=True)
    c=subs.add_parser('compare',parents=[common])
    c.add_argument('--initial-plan',type=Path,required=True)
    c.add_argument('--validations',type=Path,nargs=3,required=True)
    c.add_argument('--bootstrap',type=int,default=1000)
    m=subs.add_parser('multiscale',parents=[common])
    m.add_argument('--models',type=Path,nargs=2,required=True)
    m.add_argument('--original-validations',type=Path,nargs=2,required=True)
    m.add_argument('--image-root',type=Path,required=True)
    args=parser.parse_args(argv)
    {'compare':compare,'multiscale':multiscale}[args.mode](args)


if __name__=='__main__':
    main()
