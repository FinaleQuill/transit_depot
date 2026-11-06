"""Freeze the six authorized first-stage teacher configurations, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from teacher_distill_20261106.runtime import EVIDENCE, EXPERIMENT, ROOT, TRAINING_SOURCES


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def execution_policy(family):
    path = EXPERIMENT/f'probes/{family}_768/summary.json'
    report = json.loads(path.read_text())
    candidates = report['results']
    reference = 'blocks_chunk1_pack1' if family == 'dinov3_l' else 'blocks_chunk2_pack4'
    candidate = 'blocks_chunk2_pack4' if family == 'dinov3_l' else 'blocks_chunk2_pack4_channels_last'
    if candidates[reference]['state'] != 'completed':
        raise ValueError('Required complete-Study 768 execution reference failed')
    current = candidates[candidate]
    choose = (current['state'] == 'completed'
              and current.get('numerical_alignment', {}).get('passed') is True
              and current['peak_allocated_bytes'] < 19_000_000_000
              and current['steady_mean_seconds'] < candidates[reference]['steady_mean_seconds'])
    selected = candidate if choose else reference
    options = candidates[selected]['options']
    return dict(gradient_checkpointing=options['chunk_checkpointing'],
        block_checkpointing=options['block_checkpointing'], view_chunk_size=options['view_chunk_size'],
        pack_images=options['pack_images'], channels_last=options.get('channels_last', False),
        execution_probe=dict(path=str(path), sha256=sha(path), selected_case=selected,
                             complete_maximum_view_Study=True))


def build():
    plan_path = EXPERIMENT/'plan.json'
    plan = json.loads(plan_path.read_text())
    success = json.loads((EXPERIMENT/'_SUCCESS.json').read_text())
    if (success.get('complete') is not True or success.get('plan_sha256') != sha(plan_path)
            or not plan['all_original_images_retained']):
        raise ValueError('Complete original data preparation is required')
    assets = json.loads((EXPERIMENT/'assets/teacher_source_manifest.json').read_text())
    by_family = {'dinov3_l': next(x for x in assets['models'] if x['key'] == 'dinov3_l'),
                 'convnext_l': next(x for x in assets['models'] if x['key'] == 'convnext_large')}
    for family, asset in by_family.items():
        folder = 'dinov3_l' if family == 'dinov3_l' else 'convnext_large'
        if sha(EXPERIMENT/f'assets/{folder}/model.safetensors') != asset['sha256']:
            raise ValueError('Downloaded public weight integrity failed')
    frozen = []
    for run in plan['teacher_runs']:
        family = run['family']; asset = by_family[family]
        folder = 'dinov3_l' if family == 'dinov3_l' else 'convnext_large'
        dino = family == 'dinov3_l'
        epochs = 24 if dino else 20
        config = dict(schema='kd_raw_teacher_stage1_v1', run_id=run['run_id'], family=family,
            inner_fold=run['inner_fold'], seed=20261106 + run['inner_fold'],
            source_manifest=str(EXPERIMENT/'source_train.csv'), R_manifest=str(EXPERIMENT/'R.csv'),
            H_manifest=str(EXPERIMENT/'H.csv'), train_manifest=run['train_manifest'],
            oof_manifest=run['oof_manifest'], fold_plan=str(plan_path),
            image_root='/home/featurize/data/cxr_first',
            pretrained_path=str(EXPERIMENT/f'assets/{folder}/model.safetensors'),
            provenance_path=str(EXPERIMENT/f'assets/{folder}/source.json'),
            pretrained_revision=asset['revision'], epochs=epochs,
            resolution_schedule=[dict(through_epoch=18 if dino else 8, image_size=512),
                                 dict(through_epoch=epochs, image_size=768)],
            oof_image_sizes=[512, 768], effective_studies=32, pack_images=1,
            view_chunk_size=1, workers=4, backbone_lr=1e-5 if dino else 2e-5,
            head_lr=2e-4, layer_decay=.9 if dino else .95,
            weight_decay=.05, drop_rate=.2, drop_path=.1, clip_grad=1., ema_decay=.999,
            selection_policy='fixed_final_ema_no_H_selection',
            source_sha256={name: sha(ROOT/name) for name in TRAINING_SOURCES},
            environment_manifest=str(EXPERIMENT/'environment/manifest.json'),
            environment_manifest_sha256=sha(EXPERIMENT/'environment/manifest.json'),
            automatic_submission=False, original_data_policy='all raw official train, no exclusions',
            H_is_independent=False, H_used_for_early_stopping=False,
            output=str(EVIDENCE/run['run_id']),
            checkpoint_dir=str(EXPERIMENT/'state'/run['run_id']))
        config.update(execution_policy(family))
        keys = ('source_manifest', 'R_manifest', 'H_manifest', 'train_manifest',
                'oof_manifest', 'fold_plan', 'pretrained_path', 'provenance_path')
        config['input_sha256'] = {k: asset['sha256'] if k == 'pretrained_path' else sha(config[k]) for k in keys}
        path = EVIDENCE/'configs'/f"{run['run_id']}.json"
        if (path.exists() and json.loads(path.read_text()) != config
                and ((EVIDENCE/'launch.json').exists()
                     or (Path(config['output'])/'configuration.json').exists())):
            raise FileExistsError('Existing frozen configuration differs')
        write(path, config)
        frozen.append(dict(run_id=run['run_id'], family=family, config=str(path),
                           config_sha256=sha(path), epochs=epochs))
        if run['inner_fold'] == 0:
            smoke = dict(config, run_id=f'smoke_{family}', epochs=2, effective_studies=4,
                workers=0, resolution_schedule=[dict(through_epoch=1, image_size=512),
                                              dict(through_epoch=2, image_size=768)],
                output=str(EVIDENCE/'mechanics'/family),
                checkpoint_dir=str(EXPERIMENT/'state/mechanics'/family))
            write(EVIDENCE/'configs'/f'smoke_{family}.json', smoke)
    queue = dict(schema='kd_first_stage_teacher_queue_v1', runs=frozen,
        teacher_count=6, total_teacher_epochs=sum(x['epochs'] for x in frozen),
        data_plan=str(plan_path), data_plan_sha256=sha(plan_path),
        after_teachers='awaiting_teacher_review', automatic_students=False,
        automatic_second_outer_split=False, automatic_full_train_valid=False,
        automatic_submission=False, user_authorization='2026-11-06: 按照你的计划，启动训练')
    path = EVIDENCE/'teacher_queue.json'
    if path.exists() and json.loads(path.read_text()) != queue and (EVIDENCE/'launch.json').exists():
        raise FileExistsError('Existing teacher queue differs')
    write(path, queue)
    return queue


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    queue = build()
    print(json.dumps(dict(queue=str(EVIDENCE/'teacher_queue.json'),
                         teachers=queue['teacher_count'], epochs=queue['total_teacher_epochs'])))
