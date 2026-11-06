"""Verify frozen A assets or start an editable A-derived student experiment."""
from dataclasses import fields
import json
from pathlib import Path
import sys

BUNDLE = Path('/bundle')
PARENT = BUNDLE / 'parent_run'


def original_config():
    from train_distill import DistillConfig
    raw = json.loads((PARENT / 'config.json').read_text())
    names = {field.name for field in fields(DistillConfig)}
    values = {key: value for key, value in raw.items() if key in names}
    for name in ('fold_plan', 'soft_labels', 'output', 'image_root', 'resume',
                 'pretrained_provenance_path', 'pretrained_weights_path', 'student_init_checkpoint'):
        if values.get(name) is not None:
            values[name] = Path(values[name])
    return DistillConfig(**values)


def verify():
    import gc
    import torch
    from cv_splits import read_fold_plan
    from dataset import read_manifest
    from kd_common import require_training_runtime, sha256_file
    from model_overnight import TINY_MODEL_NAME
    from train_distill import (prepare_student_objective, student_preprocessing_config,
                               validate_student_resume)
    from student_experiments import compute_student_loss_config
    require_training_runtime()
    config = original_config()
    plan = read_fold_plan(config.fold_plan)
    records = read_manifest(plan['source_train_manifest']['path'], split='train')
    objective = prepare_student_objective(config, records)
    loss = compute_student_loss_config(records, source_manifest_sha256=plan['source_manifest_sha256'],
                                       pos_weight_power=config.pos_weight_power)
    expected = {'schema_version':8, 'artifact_type':'kd_student', 'model_name':TINY_MODEL_NAME,
                'fold_plan_sha256':sha256_file(config.fold_plan),
                'source_manifest_sha256':plan['source_manifest_sha256'],
                'official_valid_manifest_sha256':sha256_file(plan['official_holdout']['path']),
                'distillation':objective['protocol'], 'loss_config':loss,
                'model_config':config.model_config(), 'image_size':config.image_size,
                'preprocessing':student_preprocessing_config(config), 'training_precision':'float16',
                'teacher_provenance':objective['teacher_provenance'], 'validation_tta':config.validation_tta}
    checked = {}
    for role in ('best', 'last'):
        checkpoint = torch.load(PARENT / (role+'.pt'), map_location='cpu', weights_only=True)
        validate_student_resume(checkpoint, config, expected)
        checked[role] = {'epoch':checkpoint['epoch'], 'early_bad_epochs':checkpoint['early_stopping']['bad_epochs'],
                         'complete_resume_state':True, 'sha256':sha256_file(PARENT / (role+'.pt'))}
        del checkpoint
        gc.collect()
    from train_distill import DistillConfig
    options = dict(config.__dict__)
    options.update(output=Path('/experiments/verify_preview_only'), resume=None,
                   student_init_checkpoint=PARENT/'best.pt')
    fresh = DistillConfig(**options)
    fresh.validate()
    from train_distill import load_student_initialization
    initialized, lineage = load_student_initialization(PARENT/'best.pt', fresh.model_config())
    assert lineage['student_warm_start']['source_epoch'] == 9
    del initialized
    gc.collect()
    print(json.dumps({'verified':True, 'train_images':len(records),
        'oof_shape':list(objective['cache'].logits.shape), 'full_checkpoints':checked,
        'editable_warm_start_config_valid':True, 'teacher_weights_loaded':False,
        'models_constructed':False, 'gpu_work':False, 'training_started':False},indent=2))


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('verify','train','exact-resume'):
        raise SystemExit('Usage: portable_a.py verify | train [options] [--execute] | exact-resume [--execute]')
    if sys.argv[1]=='verify':
        verify()
        return 0
    from train_distill import main as train_main
    if sys.argv[1]=='exact-resume':
        if any(value!='--execute' for value in sys.argv[2:]):
            raise SystemExit('Exact resume keeps the original recipe; use train to change parameters')
        config = original_config()
        values = dict(config.__dict__)
        values['resume'] = Path('/tmp/cloud_student_runs_20261030/039_tiny768_p2_seed44/last.pt')
        arguments=[]
        for name,value in values.items():
            if value is None or name=='align_teacher_pos_weight':
                continue
            arguments.extend(['--'+name.replace('_','-'),str(value)])
        arguments.extend(sys.argv[2:])
        return train_main(arguments)
    # Team members may append any original CLI parameter. argparse's final
    # occurrence wins, so overriding LR, loss weights, augmentation code, etc.
    # is explicit. No teacher extraction or training command is invoked.
    arguments = ['--fold-plan','data/cv5_overnight/folds.json', '--soft-labels','/bundle/oof',
        '--output','/experiments/A_finetune', '--image-root','/home/featurize/data/cxr_first',
        '--image-size','768', '--seed','44', '--amp-dtype','float16', '--pos-weight-power','0.5',
        '--student-init-checkpoint','/bundle/parent_run/best.pt', *sys.argv[2:]]
    return train_main(arguments)


if __name__=='__main__':
    raise SystemExit(main())
