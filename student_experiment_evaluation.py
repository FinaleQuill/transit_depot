"""Independent Tiny/Base student evaluation; frozen production inference is untouched.

Only the original official validation split is accepted. Experimental FP16
exports have their own schema and never pretend that a Base model is Tiny.
The actual per-image math is inherited from the current production Predictor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np

from deploy_kd import (
    BASE_BACKBONE, TINY_BACKBONE, CLASS_IDS, CSV_HEADER, IMAGE_CSV_HEADER,
    MEAN, STD, MAX_MODEL_BYTES, Predictor, StudyAccumulator,
    prediction_records, read_json, read_submission, sha256_file,
    validate_preprocessing, write_image_prediction, write_prediction,
)
from inference_reproducibility import configure_inference


def write_new_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def validation_labels(records):
    labels, counts = {}, Counter()
    for row in records:
        if row.get('split') != 'valid':
            raise ValueError('Experiment evaluation accepts only official valid')
        target = row['targets']
        if isinstance(target, str):
            target = json.loads(target)
        target = list(target)
        if len(target) != 10 or any(x not in (0, 1) for x in target):
            raise ValueError('Preserve ten original validation labels')
        key = (row['subject_id'], row['study_id'])
        if key in labels and labels[key] != target:
            raise ValueError('Inconsistent Study labels')
        labels[key] = list(target)
        counts[key] += 1
    return labels, counts


def score_probabilities(labels, predictions, view_counts):
    from sklearn.metrics import f1_score, roc_auc_score
    if set(labels) != set(predictions) or set(labels) != set(view_counts):
        raise ValueError('Evaluation requires exact original Study coverage')
    keys = sorted(labels)
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    values = np.asarray([predictions[key] for key in keys], dtype=np.float64)
    if (values.shape != (len(keys), 10) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError('Invalid ten-class probabilities')

    def aucs(indices):
        return [float(roc_auc_score(targets[indices, c], values[indices, c]))
                if len(np.unique(targets[indices, c])) == 2 else None for c in range(10)]

    per_class = aucs(np.arange(len(keys)))
    if any(x is None for x in per_class):
        raise ValueError('Each validation class needs positive and negative Studies')
    slices = {}
    for name, multiple in [('single_image_study', False), ('multiple_image_study', True)]:
        indices = np.asarray([i for i, key in enumerate(keys)
                              if (view_counts[key] > 1) == multiple], dtype=int)
        scores = aucs(indices)
        slices[name] = {'num_studies': len(indices), 'per_class_auc': scores,
                        'macro_auc': None if any(x is None for x in scores) else float(np.mean(scores))}
    return {'macro_auc': float(np.mean(per_class)), 'per_class_auc': per_class,
            'class_1_4_8_mean_auc': float(np.mean([per_class[c-1] for c in (1, 4, 8)])),
            'macro_f1_fixed_0_5': float(f1_score(targets, values >= .5, average='macro', zero_division=0)),
            'num_studies': len(keys), 'num_patients': len({key[0] for key in keys}), 'slices': slices}


def validate_checkpoint_metadata(checkpoint):
    from kd_common import validate_pretrained_provenance
    from model_overnight import validate_model_config
    declared = checkpoint.get('model_config', {})
    if declared.get('backbone_name') not in (TINY_BACKBONE, BASE_BACKBONE):
        raise ValueError('Checkpoint must explicitly name its actual student backbone')
    config = validate_model_config(declared)
    backbone = config['backbone_name']
    if (backbone not in (TINY_BACKBONE, BASE_BACKBONE)
            or checkpoint.get('weights_source') != 'ema'
            or checkpoint.get('validation_complete') is not True
            or checkpoint.get('inference_ready') is not True
            or checkpoint.get('class_ids') != CLASS_IDS
            or checkpoint.get('image_size') not in (512, 768)):
        raise ValueError('Require a validated, explicitly declared Tiny/Base EMA student')
    validate_preprocessing(checkpoint.get('preprocessing', {}), image_size=checkpoint['image_size'], allow_tta=True)
    validate_pretrained_provenance(checkpoint.get('pretrained_provenance', {}), backbone)
    for field in ('source_manifest_sha256', 'fold_plan_sha256', 'official_valid_manifest_sha256'):
        digest = checkpoint.get(field, '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing immutable input binding: ' + field)
    teachers = checkpoint.get('teacher_provenance')
    if not isinstance(teachers, list) or len(teachers) != 5:
        raise ValueError('Student must retain all five strict-OOF teacher provenance records')
    folds = set()
    for teacher in teachers:
        fold = teacher.get('fold_index', teacher.get('fold'))
        if type(fold) is not int or fold not in range(5) or fold in folds:
            raise ValueError('OOF ancestry requires folds 0..4 exactly once')
        folds.add(fold)
        validate_pretrained_provenance(teacher.get('pretrained_provenance', {}), BASE_BACKBONE)
        if (teacher.get('selection_protocol') != 'fixed_epoch_final_ema'
                or teacher.get('training_complete') is not True
                or teacher.get('outer_fold_used_for_selection') is not False
                or teacher.get('official_valid_used_for_training') is not False
                or teacher.get('test_used_for_training') is not False):
            raise ValueError('Invalid strict-OOF teacher ancestry')
        for field in ('source_manifest_sha256', 'fold_plan_sha256'):
            if teacher.get(field) != checkpoint[field]:
                raise ValueError('Teacher/student source binding differs')
        trained, heldout = teacher.get('training_subject_ids'), teacher.get('oof_subject_ids')
        if not trained or not heldout or set(trained) & set(heldout):
            raise ValueError('OOF teacher has missing or overlapping patient identities')
    return config


def export_experimental_checkpoint(checkpoint_path: Path, output: Path):
    import torch
    from model_overnight import build_overnight_model
    if output.exists():
        raise FileExistsError('Choose a new experimental export directory')
    source_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_config = {**validate_checkpoint_metadata(checkpoint), 'gradient_checkpointing': False}
    if sha256_file(checkpoint_path) != source_sha:
        raise RuntimeError('Checkpoint changed while exporting')
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing EMA tensor state')
    half_state = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('Invalid EMA state')
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite EMA tensor')
            tensor = tensor.half()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('EMA tensor overflows FP16')
        half_state[key] = tensor
    model = build_overnight_model(pretrained=False, **model_config)
    model.load_state_dict(half_state, strict=True)
    del model
    output.mkdir(parents=True, exist_ok=False)
    torch.save(half_state, output / 'student_fp16.pt')
    copied = ('preprocessing', 'pretrained_provenance', 'teacher_provenance', 'distillation',
              'source_manifest_sha256', 'fold_plan_sha256', 'official_valid_manifest_sha256',
              'student_warm_start')
    config = {key: checkpoint[key] for key in copied if key in checkpoint}
    config.update(schema_version=1, artifact_type='experimental_student_fp16',
                  backbone_name=model_config['backbone_name'], model_config=model_config,
                  image_size=checkpoint['image_size'], class_ids=CLASS_IDS, precision='float16',
                  weights_source='ema', weights_file='student_fp16.pt',
                  weights_sha256=sha256_file(output / 'student_fp16.pt'),
                  source_checkpoint_sha256=source_sha, source_epoch=checkpoint.get('epoch'),
                  source_validation=checkpoint.get('validation', {}),
                  source_training_config=checkpoint.get('training_config', {}),
                  tta='hflip2', gates=None, formal_submission_ready=False)
    write_new_json(output / 'config.json', config)
    actual_bytes = sum((output / name).stat().st_size for name in ('student_fp16.pt', 'config.json'))
    if actual_bytes > MAX_MODEL_BYTES:
        raise ValueError('Experimental single-model export exceeds the 500MB limit')
    write_new_json(output / 'export_report.json', {'source_checkpoint_sha256': source_sha,
        'actual_inference_bytes': actual_bytes, 'size_limit_bytes': MAX_MODEL_BYTES,
        'size_pass': True, 'backbone_name': model_config['backbone_name'], 'model_forward_executed': False})
    del checkpoint, state, half_state
    gc.collect()
    return config


class ExperimentalPredictor(Predictor):
    """Different explicit loader, identical current production per-image math."""
    def __init__(self, model_dir, *, tta='hflip2', device='cuda'):
        from inference_runtime import activate_dependencies
        activate_dependencies()
        import torch
        from model_overnight import build_overnight_model, validate_model_config
        from inference_inputs import ExactU8Input
        if tta not in ('identity', 'hflip2'):
            raise ValueError('Only fixed identity/hflip2 is permitted')
        self.torch = torch
        self.model_dir = Path(model_dir).resolve(strict=True)
        config = read_json(self.model_dir / 'config.json')
        model_config = validate_model_config(config['model_config'])
        if (config.get('artifact_type') != 'experimental_student_fp16'
                or config.get('backbone_name') != model_config['backbone_name']
                or config.get('precision') != 'float16' or config.get('class_ids') != CLASS_IDS):
            raise ValueError('Malformed experimental export')
        validate_preprocessing(config['preprocessing'], image_size=config['image_size'], allow_tta=True)
        weights_path = self.model_dir / config['weights_file']
        if weights_path.parent != self.model_dir or sha256_file(weights_path) != config['weights_sha256']:
            raise ValueError('Experimental FP16 weight identity mismatch')
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('FP16 student evaluation requires CUDA')
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        for tensor in state.values():
            if not isinstance(tensor, torch.Tensor) or (tensor.is_floating_point()
                    and (tensor.dtype != torch.float16 or not bool(torch.isfinite(tensor).all()))):
                raise ValueError('Require finite on-disk FP16 state')
        model = build_overnight_model(pretrained=False, **model_config)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False).to(self.device, dtype=torch.float16, memory_format=torch.channels_last)
        self.models = [model]
        self.model = model
        self.members = [{**config, 'tta': tta}]
        self.model_weights = [[1.] * 10]
        self.config = {**config, 'tta': tta, 'members': self.members, 'ensemble': False}
        self.input_lookup = ExactU8Input(torch, self.device, MEAN, STD, image_sizes=[config['image_size']])


def evaluate_predictor(predictor, *, image_root, manifest, output_dir, environment=None):
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new validation output directory')
    if destination.is_relative_to(Path(image_root).resolve()) or destination.is_relative_to(predictor.model_dir):
        raise ValueError('Evaluation outputs cannot overwrite model/image inputs')
    manifest = Path(manifest)
    manifest_sha = sha256_file(manifest)
    if predictor.config.get('official_valid_manifest_sha256') != manifest_sha:
        raise ValueError('Validation manifest is not the declared official holdout')
    records = prediction_records(image_root, manifest, split='valid')
    labels, view_counts = validation_labels(records)
    accumulator = StudyAccumulator(records)
    predictions = {}
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'valid_predictions.csv').open('x', newline='', encoding='utf-8') as stream, \
         (destination / 'valid.image_predictions.csv').open('x', newline='', encoding='utf-8') as image_stream:
        writer, image_writer = csv.writer(stream), csv.writer(image_stream)
        writer.writerow(CSV_HEADER)
        image_writer.writerow(IMAGE_CSV_HEADER)
        for index, row in enumerate(records, 1):
            probability = predictor.predict_image(row['path'])
            write_image_prediction(image_writer, row, probability)
            result = accumulator.add(row, probability)
            if result is not None:
                subject, study, probability = result
                write_prediction(writer, subject, study, probability)
                predictions[(subject, study)] = probability.copy()
            if index % 100 == 0:
                print(f'[VALID] {index}/{len(records)} images', flush=True)
    if not accumulator.complete():
        raise RuntimeError('Incomplete validation image coverage')
    scores = score_probabilities(labels, predictions, view_counts)
    if environment is None:
        from cost import timing_environment
        environment = timing_environment(predictor)
    report = {**scores, 'complete': True, 'precision': 'float16',
              'image_size': predictor.config['image_size'], 'tta': predictor.config['tta'],
              'member_count': len(predictor.models), 'batch_size': 1, 'num_images': len(records),
              'manifest_sha256': manifest_sha,
              'deployment_config_sha256': sha256_file(predictor.model_dir / 'config.json'),
              'predictions_sha256': sha256_file(destination / 'valid_predictions.csv'),
              'image_predictions_sha256': sha256_file(destination / 'valid.image_predictions.csv'),
              'environment': environment, 'thresholds_fitted': False,
              'test_images_used': False, 'formal_T4_certification': False}
    write_new_json(destination / 'validation_report.json', report)
    return report


def checked_cached_validation(directory, *, model_dir, image_root, manifest):
    directory, model_dir = Path(directory), Path(model_dir)
    report = read_json(directory / 'validation_report.json')
    if (report.get('complete') is not True
            or report.get('manifest_sha256') != sha256_file(manifest)
            or report.get('deployment_config_sha256') != sha256_file(model_dir / 'config.json')
            or report.get('predictions_sha256') != sha256_file(directory / 'valid_predictions.csv')):
        raise ValueError('Cached reference validation identity differs')
    config = read_json(model_dir / 'config.json')
    from deploy_kd import deployment_members
    for member in deployment_members(config):
        if sha256_file(model_dir / member['weights_file']) != member['weights_sha256']:
            raise ValueError('Reference model differs from cached evaluation')
    records = prediction_records(image_root, manifest, split='valid')
    labels, counts = validation_labels(records)
    scores = score_probabilities(labels, read_submission(directory / 'valid_predictions.csv'), counts)
    if not np.isclose(scores['macro_auc'], report['macro_auc'], atol=1e-12, rtol=0):
        raise ValueError('Cached reference AUC does not match saved probabilities')
    return report


def history_diagnostics(history_path):
    if history_path is None or not Path(history_path).exists():
        return {'available': False}
    rows = [json.loads(line) for line in Path(history_path).read_text().splitlines() if line.strip()]
    points = []
    for row in rows:
        validation = row.get('validation', {})
        aucs = validation.get('per_class_auc')
        if isinstance(aucs, list) and len(aucs) == 10:
            points.append({'epoch': row['epoch'], 'macro_auc': validation['macro_auc'],
                           'class_1_auc': aucs[0], 'class_4_auc': aucs[3], 'class_8_auc': aucs[7]})
    if not points:
        return {'available': False, 'reason': 'No complete epoch AUCs'}
    best = max(points, key=lambda row: row['macro_auc'])
    last = points[-1]
    class1_peak = max(points, key=lambda row: row['class_1_auc'])
    return {'available': True, 'history_sha256': sha256_file(history_path), 'epochs': points,
            'best_macro_epoch': best['epoch'], 'last_epoch': last['epoch'],
            'class_1_last_minus_best_macro_epoch': last['class_1_auc'] - best['class_1_auc'],
            'class_1_last_minus_peak': last['class_1_auc'] - class1_peak['class_1_auc'],
            'class_1_peak_epoch': class1_peak['epoch'],
            'interpretation': 'Descriptive training-precision trajectory; not causal proof or independent confirmation.'}


def compare_validation_dirs(candidate_dir, reference_dir, *, manifest, history_path=None, bootstrap=1000, seed=20261030):
    from compare_student_candidates import paired_intervals
    from dataset import read_manifest
    records = read_manifest(manifest, split='valid')
    labels, views = validation_labels(records)
    keys = sorted(labels)
    candidate = read_submission(Path(candidate_dir) / 'valid_predictions.csv')
    reference = read_submission(Path(reference_dir) / 'valid_predictions.csv')
    candidate_scores = score_probabilities(labels, candidate, views)
    reference_scores = score_probabilities(labels, reference, views)
    targets = np.asarray([labels[key] for key in keys], dtype=np.int8)
    candidates = {'033': np.asarray([reference[key] for key in keys]),
                  'student': np.asarray([candidate[key] for key in keys])}
    subjects = np.asarray([key[0] for key in keys])
    paired = paired_intervals(targets, candidates, subjects, samples=bootstrap, seed=seed, reference='033')
    return {'reference': '033', 'candidate': candidate_scores, 'reference_scores': reference_scores,
            'macro_auc_delta': candidate_scores['macro_auc'] - reference_scores['macro_auc'],
            'per_class_auc_delta': [a-b for a, b in zip(candidate_scores['per_class_auc'], reference_scores['per_class_auc'])],
            'paired_patient_bootstrap': paired,
            'class_1_training_trajectory': history_diagnostics(history_path),
            'manifest_sha256': sha256_file(manifest), 'thresholds_fitted': False, 'test_used': False,
            'inference_config': 'same full official valid, FP16, fixed identity+hflip, Study mean',
            'interpretation': '033 is a historical two-student ensemble; comparing against a new single student is a performance comparison, not an isolated causal effect. Hardware/training precision, seed and checkpoint selection uncertainty remain.'}


def run_latency(model_dir, *, inputs, manifest, output_dir):
    from benchmark_local_resources import validate_inputs, validate_round_output
    import cost
    import torch
    inputs, output_dir = Path(inputs), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Choose new latency output')
    selection_path = inputs / 'selection.portable.json'
    if not selection_path.is_file():
        selection_path = inputs / 'selection.json'
    selection = read_json(selection_path)
    if selection.get('manifest_sha256') != sha256_file(manifest):
        raise ValueError('Latency selection is not this official valid manifest')
    # Only relocate the original manifest pointer; identities/selection stay fixed.
    output_dir.mkdir(parents=True, exist_ok=False)
    relocated = {**selection, 'manifest_path': str(Path(manifest).resolve())}
    relocated_path = output_dir / 'selection.relocated.json'
    write_new_json(relocated_path, relocated)
    measured, warmup, coverage = validate_inputs(inputs / 'pictures_100', inputs / 'pictures_5', relocated_path)
    result = {'T4_certified': False, 'development_diagnostic_only': True,
              'protocol': {'images': 100, 'warmup': 5, 'rounds': 3, 'batch': 1,
                           'execution': 'serial', 'timing': 'read_file_to_actual_CSV_write',
                           'model_load_excluded': True}, 'coverage': coverage, 'modes': {}}
    for tta in ('identity', 'hflip2'):
        configure_inference()
        predictor = ExperimentalPredictor(model_dir, tta=tta)
        environment = cost.timing_environment(predictor)
        mode_dir = output_dir / tta
        mode_dir.mkdir()
        for record in warmup:
            predictor.predict_image(record['path'])
        predictor.synchronize()
        durations, outputs, comparisons, first = [], [], [], None
        for index in range(3):
            path = mode_dir / f'round_{index+1}.csv'
            times, probabilities = cost.time_round(predictor, measured, path)
            durations.append(times)
            outputs.append(validate_round_output(measured, path, probabilities))
            if first is None:
                first = probabilities
            comparisons.append(cost.compare_predictions(probabilities, first, atol=0))
        rounds, per_image, mean_ms = cost.summarize_timings(durations)
        if not all(item['match'] for item in comparisons):
            raise RuntimeError('Latency rounds changed deterministic predictions')
        result['modes'][tta] = {'mean_ms': mean_ms, 'round_mean_ms': rounds.mean(1).tolist(),
            'per_image_round_ms': rounds.T.tolist(), 'per_image_mean_ms': per_image.tolist(),
            'p95_ms': float(np.percentile(per_image, 95)), 'environment': environment,
            'above_50ms': mean_ms > 50., 'round_outputs': outputs, 'round_comparisons': comparisons}
        write_new_json(mode_dir / 'latency_report.json', result['modes'][tta])
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    result['single_model_without_TTA_above_50ms'] = result['modes']['identity']['above_50ms']
    write_new_json(output_dir / 'latency_report.json', result)
    return result


def write_report(path, metrics, *, label):
    candidate, reference = metrics['candidate'], metrics['reference_scores']
    lines = [f'# {label} vs 冻结033：官方valid成绩单', '',
             f"Macro-AUC：{candidate['macro_auc']:.9f} vs {reference['macro_auc']:.9f}，差值 {metrics['macro_auc_delta']:+.9f}。", '',
             '| 类别 | 学生 | 033 | 差值 |', '|---|---:|---:|---:|']
    for index, (a, b) in enumerate(zip(candidate['per_class_auc'], reference['per_class_auc']), 1):
        name = f'**{index}**' if index in (1, 4, 8) else str(index)
        lines.append(f'| {name} | {a:.6f} | {b:.6f} | {a-b:+.6f} |')
    lines += ['', f"固定0.5 Macro-F1：{candidate['macro_f1_fixed_0_5']:.6f} vs {reference['macro_f1_fixed_0_5']:.6f}。",
              f"类1/4/8平均AUC：{candidate['class_1_4_8_mean_auc']:.6f} vs {reference['class_1_4_8_mean_auc']:.6f}。"]
    bootstrap = metrics['paired_patient_bootstrap']
    if bootstrap:
        interval = bootstrap['results']['student']['macro_auc_delta']['ci_95']
        lines += [f'患者聚类配对95%区间：{interval}（{bootstrap["samples"]}次；不消除选epoch/训练环境混杂）。']
    history = metrics['class_1_training_trajectory']
    if history['available']:
        lines += ['', f"类1末轮−整体最佳轮：{history['class_1_last_minus_best_macro_epoch']:+.6f}；末轮−类1峰值：{history['class_1_last_minus_peak']:+.6f}。"]
    latency = metrics.get('latency')
    if latency:
        lines += ['', '2080Ti端到端开发诊断（100图、另5预热、batch1串行三轮；并非T4认证）：', '',
                  '| 单模型模式 | ms/图 |', '|---|---:|']
        for tta, result in latency['modes'].items():
            lines.append(f'| {tta} | {result["mean_ms"]:.3f} |')
        if latency['single_model_without_TTA_above_50ms']:
            lines += ['', '**耗时预警：单模型无TTA已超过50ms，后续需重点研究瘦身。**']
    lines += ['', '仅使用官方valid；033为双模型、新候选为单模型。训练端硬件/精度及选模因素仍有差异，不能把成绩差值全部归因于某一配方。未重生成test，未修改冻结资产。', '']
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def evaluate_experiment(*, checkpoint, image_root, manifest, reference_model_dir,
                        reference_validation_dir, output_dir, bootstrap=1000, latency_inputs=None, label='Student'):
    import torch
    checkpoint, image_root, manifest, reference_model_dir, reference_validation_dir, output_dir = map(Path,
        (checkpoint, image_root, manifest, reference_model_dir, reference_validation_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Choose a new experiment evaluation directory')
    if output_dir.resolve().is_relative_to(image_root.resolve()):
        raise ValueError('Outputs cannot be inside original images')
    output_dir.mkdir(parents=True, exist_ok=False)
    export_experimental_checkpoint(checkpoint, output_dir / 'fp16')
    if reference_validation_dir.exists():
        checked_cached_validation(reference_validation_dir, model_dir=reference_model_dir,
                                  image_root=image_root, manifest=manifest)
    else:
        configure_inference()
        baseline = Predictor(reference_model_dir)
        evaluate_predictor(baseline, image_root=image_root, manifest=manifest, output_dir=reference_validation_dir)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
    configure_inference()
    predictor = ExperimentalPredictor(output_dir / 'fp16', tta='hflip2')
    evaluate_predictor(predictor, image_root=image_root, manifest=manifest, output_dir=output_dir / 'valid_hflip2')
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    metrics = compare_validation_dirs(output_dir / 'valid_hflip2', reference_validation_dir,
        manifest=manifest, history_path=checkpoint.parent / 'history.jsonl', bootstrap=bootstrap)
    metrics.update(label=label, checkpoint_sha256=sha256_file(checkpoint),
                   reference_validation_dir=str(reference_validation_dir), candidate_validation_dir=str(output_dir / 'valid_hflip2'))
    # Save the complete A-vs-033 table immediately, before optional B timing.
    write_new_json(output_dir / 'metrics.before_latency.json', metrics)
    write_report(output_dir / 'REPORT.before_latency.md', metrics, label=label)
    print((output_dir / 'REPORT.before_latency.md').read_text(), flush=True)
    if latency_inputs is not None:
        metrics['latency'] = run_latency(output_dir / 'fp16', inputs=latency_inputs,
                                       manifest=manifest, output_dir=output_dir / 'latency_2080ti')
    write_new_json(output_dir / 'metrics.json', metrics)
    write_report(output_dir / 'REPORT.md', metrics, label=label)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'image-root', 'manifest', 'reference-model-dir', 'reference-validation-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--latency-inputs', type=Path)
    parser.add_argument('--label', default='Student')
    args = vars(parser.parse_args(argv))
    metrics = evaluate_experiment(**args)
    print(json.dumps({'macro_auc': metrics['candidate']['macro_auc'], 'delta_vs_033': metrics['macro_auc_delta']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
