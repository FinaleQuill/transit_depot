"""CPU-only, read-only audit of committed 028 completion. Never construct a model."""
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

run = ROOT.parent / 'runs/028_tiny512_kd_short12_seed43'
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
input_hashes = {str(config.fold_plan): sha256_file(config.fold_plan),
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
assert (last['epoch'], best['epoch'], last['best_epoch']) == (12, 7, 7)
assert best['best_auc'] == last['best_auc'] == 0.8038905628482913
assert last['validation']['macro_auc'] == 0.7985410099003226
status = json.loads((run / 'status.json').read_text())
history = [json.loads(s) for s in (run / 'history.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in history] == list(range(1, 13))
assert status['state'] == 'early_stopped'
progress = dict(completed_epochs=12, validated_epochs=12, backward_batches=45492,
                optimizer_updates=11376, ema_updates=11376, amp_skipped_steps=0)
assert last['progress'] == progress
assert {k: status[k] for k in progress} == progress
assert last['early_stopping']['bad_epochs'] == config.patience == 5
assert last['epoch'] == config.epochs == 12
hashes = {name: sha256_file(run / name) for name in ('last.pt', 'best.pt')}
assert hashes['last.pt'] == 'ec0815e5957268f23f3e5ad2f99785a09f9cb23ac8d69cbdc437f04d214e1a1f'
assert hashes['best.pt'] == '4e08a64bb236431c40b0e8e05ad91feee628c5b63385c3f7eceb2b4ccb380b70'
for role in ('last', 'best'):
    assert hashes[role + '.pt'] == status[role + '_checkpoint_sha256'] == history[-1][role + '_checkpoint_sha256']
assert status['best_epoch'] == 7 and status['best_auc'] == best['best_auc']
assert json.loads((run / 'best_metrics.json').read_text()) == dict(epoch=7, **best['validation'])
for row in history:
    assert row['training']['images'] == 30328 and row['training']['full_manifest_coverage'] is True
    assert row['validation']['num_studies'] == 859
    assert row['validation']['validation_tta'] == 'hflip2'
assert math.ceil(len(train) / config.batch_size) * 12 == progress['backward_batches']
assert math.ceil(math.ceil(len(train) / config.batch_size) / config.accumulation_steps) * 12 == progress['optimizer_updates']

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
assert len(ledger) == 19
assert Counter(r['role'] for r in ledger) == {'best': 7, 'last': 12}
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
assert len(archives) == 3

receipts = [json.loads(f.read_text()) for f in sorted((run / 'evidence').glob('command_*.json'))]
assert len(receipts) == 3 and [r['exit_code'] for r in receipts] == [None, None, 0]
assert '--resume' not in receipts[0]['command']
assert all('--resume' in r['command'] for r in receipts[1:])
assert receipts[2]['ended_at'] == '2026-10-27T09:55:37.475889+00:00'
logs = [Path(r['console_log']).read_text().splitlines() for r in receipts]
assert logs[0][-1] == 'student epoch=8 batch=2000/3791 loss=0.625755 alpha=0.2000'
assert logs[1][-1] == 'student epoch=8 batch=1300/3791 loss=0.621672 alpha=0.2000'
batch_re = re.compile(r'^student epoch=(\d+) batch=(\d+)/(\d+) ')
valid_re = re.compile(r'^student epoch=(\d+) valid_AUC=([0-9.]+) best=([0-9.]+)@(\d+)$')
coverage = []
for row in history:
    epoch = row['epoch']
    index = 0 if epoch <= 7 else 2
    batch = [tuple(map(int, m.groups())) for line in logs[index] if (m := batch_re.match(line)) and int(m[1]) == epoch]
    expected_marks = [1] + list(range(100, 3791, 100))
    assert [b[1] for b in batch] == expected_marks and all(b[2] == 3791 for b in batch)
    validation = [m for line in logs[index] if (m := valid_re.match(line)) and int(m[1]) == epoch]
    assert len(validation) == 1
    assert abs(float(validation[0][2]) - row['validation']['macro_auc']) <= 5.01e-9
    coverage.append(dict(epoch=epoch, receipt_index=index + 1, batch_progress_marks=len(batch),
                         validation_lines=1, images=30328))
assert receipts[2]['console_bytes'] == Path(receipts[2]['console_log']).stat().st_size
terminal_text = '\n'.join(logs[2][next(i for i, s in enumerate(logs[2]) if s == '{'):])
assert json.loads(terminal_text) == status

oldconfig = json.loads((ROOT.parent / 'runs/027_tiny512_kd_seed43/config.json').read_text())
config_diff = {k: dict(run027=oldconfig.get(k), run028=raw.get(k)) for k in set(oldconfig) | set(raw)
               if oldconfig.get(k) != raw.get(k)}
assert set(config_diff) == {'epochs', 'output', 'resume'}
provenance = validate_pretrained_provenance(last['pretrained_provenance'], TINY_BACKBONE)
assert best['pretrained_provenance'] == provenance == json.loads((run / 'pretrained_provenance.json').read_text())
pretrain_path = Path(provenance['weights_path'])
assert sha256_file(pretrain_path) == provenance['weights_sha256']
finite = {}
for role, checkpoint in [('best', best), ('last', last)]:
    finite[role] = {}
    for key in ('raw_model_state_dict', 'model_state_dict'):
        state = checkpoint[key]
        assert state and all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' for t in state.values())
        assert all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in state.values())
        finite[role][key] = dict(tensors=len(state), numel=sum(t.numel() for t in state.values()), finite=True)
after = snapshot()
assert before == after
for p, digest in source_hashes.items():
    assert sha256_file(p) == digest
for p, digest in input_hashes.items():
    assert sha256_file(p) == digest
started = datetime.fromisoformat(receipts[0]['started_at'])
ended = datetime.fromisoformat(receipts[2]['ended_at'])
result = dict(audited_at_hk=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    run=str(run), conclusion='completed_12_committed_epochs_final_resume_exit0',
    status=status, progress=progress, best_epoch=7, best_auc=best['best_auc'], last_auc=last['validation']['macro_auc'],
    checkpoint_sha256=hashes, native_resume_validation='both_passed_reconciliation_selects_last',
    expected_metadata_rebuilt_from_native_fold_manifest_objective_and_loss=True,
    early_stopping=last['early_stopping'], epoch_cap_also_reached=True,
    ledger_records=len(ledger), ledger_role_counts=dict(Counter(r['role'] for r in ledger)), ledger_chain='passed_pure_read',
    all_history_hash_bindings='passed', source_archives=archives, source_hashes=source_hashes,
    input_hashes=input_hashes, train_valid_isolation='passed_subject_study_image_file_pixel',
    strict_oof_native_loader='passed', strict_oof_shape=list(objective['cache'].logits.shape),
    teacher_weight_files_not_rehashed=True, pretrained_provenance_native='passed',
    original_pretrained_weights_sha256=provenance['weights_sha256'], finite_cpu_checkpoint_states=finite,
    command_receipts=receipts, successful_epoch_log_coverage=coverage,
    partial_epoch8_logs=[dict(receipt_index=i+1, last_line=logs[i][-1], exit_code=None,
        exit_reason='unknown_not_retroactively_reclassified') for i in (0,1)],
    committed_image_exposures=12*30328, committed_counts_exclude_abandoned_partial_epoch8_work=True,
    total_physical_training_compute='not_exactly_recoverable_from_sparse_logs_and_unknown_exits',
    elapsed_first_start_to_final_exit_seconds=(ended-started).total_seconds(),
    final_resume_elapsed_seconds=(ended-datetime.fromisoformat(receipts[2]['started_at'])).total_seconds(),
    committed_training_seconds=sum(row['training']['seconds'] for row in history),
    config027_to028_differences=config_diff,
    original_run_files_unchanged=True, original_run_files_count=len(before), original_run_hashes=before,
    original_sources_and_bound_inputs_unchanged=True, no_model_construction_or_forward=True,
    no_training_or_gpu_work=True,
    limitations=['First two historical exits and causes remain unknown.',
        'Committed checkpoint counters omit abandoned partial E8 execution.',
        'Native cache verifies source-image hash fields, not a new full raw-image rehash.',
        'Teacher large checkpoint binaries were not rehashed; native cache metadata provenance remains bound.',
        'No new predictions, patient confidence intervals, FP16 deployment, or T4 timing produced.'])
Path(__file__).with_name('integrity.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('original_run_hashes','source_hashes','command_receipts','source_archives')}, indent=2, ensure_ascii=False))
