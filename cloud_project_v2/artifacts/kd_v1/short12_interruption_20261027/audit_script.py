"""CPU-only interruption/resume audit; never build a model or modify the run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path('/home/featurize/work/cloud_project_v2')
sys.path.insert(0, str(ROOT))
import torch
from cv_splits import read_fold_plan
from dataset import read_manifest
from kd_common import sha256_file
from model_overnight import TINY_MODEL_NAME
from train_overnight import compute_loss_config
from train_distill import (DistillConfig, _configuration, prepare_student_objective,
                           student_preprocessing_config, reconcile_student_checkpoints)

run = ROOT.parent / 'runs/028_tiny512_kd_short12_seed43'
before = {str(f): sha256_file(f) for f in run.rglob('*') if f.is_file()}
raw = json.loads((run / 'config.json').read_text())
fields = DistillConfig.__dataclass_fields__
kwargs = {k: v for k, v in raw.items() if k in fields}
for k in ('fold_plan', 'soft_labels', 'output', 'image_root'):
    if kwargs[k] is not None:
        kwargs[k] = Path(kwargs[k])
kwargs['resume'] = Path('../runs/028_tiny512_kd_short12_seed43/last.pt')
config = DistillConfig(**kwargs)
config.validate()
plan = read_fold_plan(config.fold_plan)
train = read_manifest(plan['source_train_manifest']['path'], split='train')
valid_path = Path(plan['official_holdout']['path'])
valid = read_manifest(valid_path, split='valid')
for key in ('subject_id', 'study_key', 'image_id', 'file_sha256', 'pixel_sha256'):
    assert not (({r.get(key) for r in train} - {None, ''}) &
                ({r.get(key) for r in valid} - {None, ''})), key
assert sha256_file(plan['source_train_manifest']['path']) == plan['source_manifest_sha256']
assert sha256_file(valid_path) == plan['official_holdout']['sha256']
objective = prepare_student_objective(config, train)
for path, expected_hash in objective['cache_bindings']:
    assert sha256_file(path) == expected_hash
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
selected = reconcile_student_checkpoints(last, best, config, expected)
assert selected is last
assert last['epoch'] == best['epoch'] == 7
status = json.loads((run / 'status.json').read_text())
history = [json.loads(s) for s in (run / 'history.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in history] == list(range(1, 8))
assert last['progress']['completed_epochs'] == status['completed_epochs'] == 7
assert sha256_file(run / 'last.pt') == status['last_checkpoint_sha256'] == history[-1]['last_checkpoint_sha256']
assert sha256_file(run / 'best.pt') == status['best_checkpoint_sha256'] == history[-1]['best_checkpoint_sha256']
ledger = [json.loads(s) for s in (run / 'checkpoint_audit.jsonl').read_text().splitlines()]
previous = None
for record in ledger:
    assert record['previous_record_sha256'] == previous
    canonical = json.dumps({k: v for k, v in record.items() if k != 'record_sha256'},
                           sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == record['record_sha256']
    previous = record['record_sha256']
    for name, digest in record['source_version']['files'].items():
        assert sha256_file(name) == digest
assert ledger[-1]['checkpoint_sha256'] == status['last_checkpoint_sha256']
archive = next((run / 'evidence').glob('source_*'))
archive_count = 0
for f in archive.glob('*.py'):
    assert sha256_file(f) == sha256_file(ROOT / f.name), f.name
    archive_count += 1
receipt_path = next((run / 'evidence').glob('command_*.json'))
receipt = json.loads(receipt_path.read_text())
log = Path(receipt['console_log']).read_text().splitlines()
assert log[-1] == 'student epoch=8 batch=2000/3791 loss=0.625755 alpha=0.2000'
assert receipt['exit_code'] is None
assert not Path('/proc', str(receipt['child_pid'])).exists()
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
active = [s for s in processes if 'train_distill.py' in s and '--execute' in s]
assert not active, active
after = {str(f): sha256_file(f) for f in run.rglob('*') if f.is_file()}
assert before == after
result = dict(
    audited_at_hk=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    conclusion='interrupted_in_epoch_8_not_terminal', exit_reason='not_recorded_unknown',
    run=str(run), status_is_stale=True, status=status, command_receipt=receipt,
    last_console_line=log[-1], validated_epochs=7, next_epoch_on_resume=8,
    remaining_max_epochs=5, best_epoch=best['epoch'], best_auc=best['best_auc'],
    progress=last['progress'], early_stopping=last['early_stopping'],
    checkpoint_sha256={k: sha256_file(run/k) for k in ('best.pt', 'last.pt')},
    native_resume_validation='passed_both_checkpoints', ledger_chain='passed', ledger_records=len(ledger),
    archived_python_files_equal_current=archive_count, raw_ema_optimizer_scaler_rng_present=True,
    input_hashes_and_train_valid_isolation='passed', soft_cache_bindings='passed',
    original_run_files_unchanged=True, no_model_construction_or_forward=True, no_training_started=True,
    resume_config=_configuration(config), history=[dict(epoch=r['epoch'],
        auc=r['validation']['macro_auc'], seconds=r['training']['seconds']) for r in history],
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                                 '--format=csv,noheader'], text=True).strip())
output = Path(__file__).with_name('audit.json')
output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('command_receipt','resume_config','status')},
                 ensure_ascii=False, indent=2))
