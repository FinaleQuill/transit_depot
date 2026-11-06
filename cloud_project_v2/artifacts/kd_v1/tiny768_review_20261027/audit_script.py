"""CPU-only, read-only audit of 029 completion. Never construct a model."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone, timedelta
from collections import Counter

ROOT = Path('/home/featurize/work/cloud_project_v2')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
import torch
torch.set_num_threads(1)
from cv_splits import read_fold_plan
from dataset import read_manifest
from kd_common import sha256_file, validate_pretrained_provenance, TINY_BACKBONE
from model_overnight import TINY_MODEL_NAME
from train_overnight import compute_loss_config
from train_distill import (DistillConfig, prepare_student_objective,
    student_preprocessing_config, reconcile_student_checkpoints, validate_student_resume)

run = ROOT.parent / 'runs/029_tiny768_kd_seed43'
def snapshot():
    return {str(f): sha256_file(f) for f in sorted(run.rglob('*')) if f.is_file()}
before = snapshot()
raw = json.loads((run / 'config.json').read_text())
kwargs = {k: v for k, v in raw.items() if k in DistillConfig.__dataclass_fields__}
for k in ('fold_plan', 'soft_labels', 'output', 'image_root', 'resume'):
    if kwargs[k] is not None:
        kwargs[k] = Path(kwargs[k])
config = DistillConfig(**kwargs)
config.validate()
plan = read_fold_plan(config.fold_plan)
train_path = Path(plan['source_train_manifest']['path'])
train = read_manifest(train_path, split='train')
valid_path = Path(plan['official_holdout']['path'])
valid = read_manifest(valid_path, split='valid')
assert len(train) == 30328 and len(valid) == 1422
for key in ('subject_id', 'study_key', 'image_id', 'file_sha256', 'pixel_sha256'):
    assert not (({r.get(key) for r in train} - {None, ''}) &
                ({r.get(key) for r in valid} - {None, ''})), key
input_hashes = {str(ROOT/'artifacts/kd_v1/short12_completed_review_20261027/audit_script.py'): sha256_file(ROOT/'artifacts/kd_v1/short12_completed_review_20261027/audit_script.py'), str(config.fold_plan): sha256_file(config.fold_plan),
                str(train_path): sha256_file(train_path), str(valid_path): sha256_file(valid_path)}
assert input_hashes[str(train_path)] == plan['source_manifest_sha256']
assert input_hashes[str(valid_path)] == plan['official_holdout']['sha256']
# Native strict-OOF loader validates identity, source-image hash fields, fold exclusion,
# teacher checkpoint provenance metadata and finite per-view logits, without models.
objective = prepare_student_objective(config, train)
for path, expected_hash in objective['cache_bindings']:
    input_hashes[str(path)] = sha256_file(path)
    assert input_hashes[str(path)] == expected_hash
assert objective['protocol'] == raw['distillation']
expected = dict(schema_version=objective['schema_version'], artifact_type='kd_student',
    model_name=TINY_MODEL_NAME, fold_plan_sha256=sha256_file(config.fold_plan),
    source_manifest_sha256=plan['source_manifest_sha256'],
    official_valid_manifest_sha256=sha256_file(valid_path), distillation=objective['protocol'],
    loss_config=compute_loss_config(train, source_manifest_sha256=plan['source_manifest_sha256'], fold=None),
    model_config=config.model_config(), image_size=config.image_size,
    preprocessing=student_preprocessing_config(config), training_precision='bfloat16',
    teacher_provenance=objective['teacher_provenance'], validation_tta=config.validation_tta)
last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
validate_student_resume(last, config, expected)
validate_student_resume(best, config, expected)
assert reconcile_student_checkpoints(last, best, config, expected) is last
assert (last['epoch'], best['epoch'], last['best_epoch']) == (13, 8, 8)
assert best['best_auc'] == last['best_auc'] == 0.8041769633268219
assert last['validation']['macro_auc'] == 0.7951635392608315
status = json.loads((run / 'status.json').read_text())
history = [json.loads(s) for s in (run / 'history.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in history] == list(range(1, 14))
assert status['state'] == 'early_stopped'
progress = dict(completed_epochs=13, validated_epochs=13, backward_batches=49283,
                optimizer_updates=12324, ema_updates=12324, amp_skipped_steps=0)
assert last['progress'] == progress
assert {k: status[k] for k in progress} == progress
assert last['early_stopping']['bad_epochs'] == config.patience == 5
assert last['epoch'] == 13 and config.epochs == 25
hashes = {name: sha256_file(run / name) for name in ('last.pt', 'best.pt')}
assert hashes['last.pt'] == '8bb5a1cf10ca8986bb3b14eb5b30cbd803e55a54e0be6570e6ee00fcd7e3d4c6'
assert hashes['best.pt'] == '46e06022d963b58d262eaa99ba376de40c3ef182f50a9619f0d09dcf2a0870b7'
for role in ('last', 'best'):
    assert hashes[role + '.pt'] == status[role + '_checkpoint_sha256'] == history[-1][role + '_checkpoint_sha256']
assert status['best_epoch'] == 8 and status['best_auc'] == best['best_auc']
assert json.loads((run / 'best_metrics.json').read_text()) == dict(epoch=8, **best['validation'])
for row in history:
    assert row['training']['images'] == 30328 and row['training']['full_manifest_coverage'] is True
    assert row['validation']['num_studies'] == 859
    assert row['validation']['validation_tta'] == 'hflip2'
assert math.ceil(len(train) / config.batch_size) * 13 == progress['backward_batches']
assert math.ceil(math.ceil(len(train) / config.batch_size) / config.accumulation_steps) * 13 == progress['optimizer_updates']

# Deliberately do not call audit_trail.read_checkpoint_audit (it may write repairs).
ledger = [json.loads(s) for s in (run / 'checkpoint_audit.jsonl').read_text().splitlines()]
previous = None
for record in ledger:
    assert record['audit_schema_version'] == 1
    assert record['previous_record_sha256'] == previous
    canonical = json.dumps({k: v for k, v in record.items() if k != 'record_sha256'},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == record['record_sha256']
    previous = record['record_sha256']
    for name, digest in record['source_version']['files'].items():
        assert sha256_file(name) == digest
    source_canonical = json.dumps(record['source_version']['files'], sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(source_canonical).hexdigest() == record['source_version']['manifest_sha256']
    assert record['historical_recovery'] is False and record['evidence_source'] == 'trainer'
assert len(ledger) == 21
assert Counter(r['role'] for r in ledger) == {'best': 8, 'last': 13}
for row in history:
    candidates = [r for r in ledger if r['epoch'] == row['epoch'] and r['role'] == 'last']
    assert len(candidates) == 1
    assert candidates[0]['checkpoint_sha256'] == row['last_checkpoint_sha256']
    assert candidates[0]['record_sha256'] == row['checkpoint_audit_record_sha256']
    earlier_best = [r for r in ledger if r['epoch'] <= row['epoch'] and r['role'] == 'best'][-1]
    assert earlier_best['checkpoint_sha256'] == row['best_checkpoint_sha256']
for role in ('last', 'best'):
    item = [r for r in ledger if r['role'] == role][-1]
    assert item['checkpoint_sha256'] == hashes[role + '.pt']
    assert item['checkpoint_metadata']['size_bytes'] == (run / (role + '.pt')).stat().st_size

archives = []
source_hashes = {}
for archive in sorted((run / 'evidence').glob('source_*')):
    manifest = json.loads((archive / 'source_manifest.json').read_text())
    assert isinstance(manifest['files'], list) and len(manifest['files']) == 38
    assert len({f['path'] for f in manifest['files']}) == 38
    for item in manifest['files']:
        p = Path(item['path'])
        assert not p.is_absolute() and '..' not in p.parts
        saved = archive / p
        assert saved.stat().st_size == item['bytes']
        assert sha256_file(saved) == item['sha256'] == sha256_file(ROOT / p)
        source_hashes[str(ROOT / p)] = item['sha256']
    archives.append(dict(directory=str(archive), files=len(manifest['files']),
        python_files=sum(i['path'].endswith('.py') for i in manifest['files']),
        manifest_sha256=sha256_file(archive / 'source_manifest.json'),
        all_archived_files_match_manifest_and_current=True))
assert len(archives) == 1

receipts = [json.loads(f.read_text()) for f in sorted((run / 'evidence').glob('command_*.json'))]
assert len(receipts) == 1 and receipts[0]['exit_code'] == 0
assert '--resume' not in receipts[0]['command']
assert receipts[0]['started_at'] == '2026-10-27T10:06:45.715790+00:00'
assert receipts[0]['ended_at'] == '2026-10-27T12:07:29.820569+00:00'
logs = [Path(r['console_log']).read_text().splitlines() for r in receipts]
batch_re = re.compile(r'^student epoch=(\d+) batch=(\d+)/(\d+) ')
valid_re = re.compile(r'^student epoch=(\d+) valid_AUC=([0-9.]+) best=([0-9.]+)@(\d+)$')
coverage = []
for row in history:
    epoch = row['epoch']
    index = 0
    batch = [tuple(map(int, m.groups())) for line in logs[index] if (m := batch_re.match(line)) and int(m[1]) == epoch]
    expected_marks = [1] + list(range(100, 3791, 100))
    assert [b[1] for b in batch] == expected_marks and all(b[2] == 3791 for b in batch)
    validation = [m for line in logs[index] if (m := valid_re.match(line)) and int(m[1]) == epoch]
    assert len(validation) == 1
    assert abs(float(validation[0][2]) - row['validation']['macro_auc']) <= 5.01e-9
    coverage.append(dict(epoch=epoch, receipt_index=index + 1, batch_progress_marks=len(batch),
                         validation_lines=1, images=30328))
assert receipts[0]['console_bytes'] == Path(receipts[0]['console_log']).stat().st_size
terminal_text = '\n'.join(logs[0][next(i for i, s in enumerate(logs[0]) if s == '{'):])
assert json.loads(terminal_text) == status

oldconfig = json.loads((ROOT.parent / 'runs/027_tiny512_kd_seed43/config.json').read_text())
config_diff = {k: dict(run027=oldconfig.get(k), run029=raw.get(k)) for k in set(oldconfig) | set(raw)
               if oldconfig.get(k) != raw.get(k)}
assert set(config_diff) == {'image_size', 'output', 'preprocessing', 'distillation'}
assert oldconfig['image_size'] == 512 and raw['image_size'] == 768
assert {k for k in oldconfig['preprocessing'] if oldconfig['preprocessing'][k] != raw['preprocessing'][k]} == {'image_size', 'validation_size', 'training_scales'}
assert {k for k in oldconfig['distillation'] if oldconfig['distillation'][k] != raw['distillation'][k]} == {'student_training_size'}
assert raw['epochs'] == oldconfig['epochs'] == 25 and raw['seed'] == oldconfig['seed'] == 43
oldhistory = [json.loads(s) for s in (ROOT.parent / 'runs/027_tiny512_kd_seed43/history.jsonl').read_text().splitlines()]
assert all(a['training']['learning_rates'] == b['training']['learning_rates'] for a,b in zip(history, oldhistory))
provenance = validate_pretrained_provenance(last['pretrained_provenance'], TINY_BACKBONE)
assert best['pretrained_provenance'] == provenance == json.loads((run / 'pretrained_provenance.json').read_text())
pretrain_path = Path(provenance['weights_path'])
assert sha256_file(pretrain_path) == provenance['weights_sha256']
input_hashes[str(pretrain_path)] = provenance['weights_sha256']
finite = {}
for role, checkpoint in [('best', best), ('last', last)]:
    finite[role] = {}
    for key in ('raw_model_state_dict', 'model_state_dict'):
        state = checkpoint[key]
        assert state and all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' for t in state.values())
        assert all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in state.values())
        finite[role][key] = dict(tensors=len(state), numel=sum(t.numel() for t in state.values()), finite=True)
delivery_path = ROOT.parent / 'deliveries/029_tiny768_kd_seed43'
assert not delivery_path.exists()
assert not any((run / p).exists() for p in ('predictions.npz', 'predictions.csv', 'valid_predictions.npz', 'valid_predictions.csv'))
after = snapshot()
assert before == after
for p, digest in source_hashes.items():
    assert sha256_file(p) == digest
for p, digest in input_hashes.items():
    assert sha256_file(p) == digest
started = datetime.fromisoformat(receipts[0]['started_at'])
ended = datetime.fromisoformat(receipts[0]['ended_at'])
result = dict(audited_at_hk=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    run=str(run), conclusion='completed_13_committed_epochs_single_run_exit0',
    status=status, progress=progress, best_epoch=8, best_auc=best['best_auc'], last_auc=last['validation']['macro_auc'],
    checkpoint_sha256=hashes, native_resume_validation='both_passed_reconciliation_selects_last',
    expected_metadata_rebuilt_from_native_fold_manifest_objective_and_loss=True,
    early_stopping=last['early_stopping'], epoch_cap_also_reached=False,
    ledger_records=len(ledger), ledger_role_counts=dict(Counter(r['role'] for r in ledger)), ledger_chain='passed_pure_read',
    all_history_hash_bindings='passed', source_archives=archives, source_hashes=source_hashes,
    input_hashes=input_hashes, train_valid_isolation='passed_subject_study_image_file_pixel',
    strict_oof_native_loader='passed', strict_oof_shape=list(objective['cache'].logits.shape),
    teacher_weight_files_not_rehashed=True, pretrained_provenance_native='passed',
    original_pretrained_weights_sha256=provenance['weights_sha256'], finite_cpu_checkpoint_states=finite,
    command_receipts=receipts, successful_epoch_log_coverage=coverage,
    committed_image_exposures=13*30328,
    elapsed_first_start_to_final_exit_seconds=(ended-started).total_seconds(),
    mean_end_to_end_seconds_per_epoch=(ended-started).total_seconds()/13,
    committed_training_seconds=sum(row['training']['seconds'] for row in history),
    config027_to029_differences=config_diff,
    lr_schedule_identical_to027_for_all_common_epochs=True,
    training_seconds_per_epoch=[row['training']['seconds'] for row in history],
    mean_training_seconds_per_epoch=sum(row['training']['seconds'] for row in history)/13,
    gpu_peak_allocated_gib=max(row['training']['gpu_peak_allocated_gib'] for row in history),
    original_run_files_unchanged=True, original_run_files_count=len(before), original_run_hashes=before,
    original_sources_and_bound_inputs_unchanged=True, no_model_construction_or_forward=True,
    no_training_or_gpu_work=True, test_data_usage=False,
    deployment_review=dict(canonical_delivery_directory=str(delivery_path),
        fp16_export='absent_pending', per_study_predictions='absent_pending',
        actual_fp16_auc='pending', official_t4_timing='pending'),
    checkpoint_interval_seconds=[(datetime.fromisoformat(row['checkpoint_saved_at_utc'])-
        (datetime.fromisoformat(history[i-1]['checkpoint_saved_at_utc']) if i else started)).total_seconds()
        for i,row in enumerate(history)],
    limitations=['Native cache verifies source-image hash fields, not a new full raw-image rehash.',
        'Teacher large checkpoint binaries were not rehashed; native cache metadata provenance remains bound.',
        'No new predictions, patient confidence intervals, FP16 deployment, or T4 timing produced.'])
Path(__file__).with_name('integrity.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('original_run_hashes','source_hashes','command_receipts','source_archives')}, indent=2, ensure_ascii=False))
