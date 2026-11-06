"""Compare saved scalar metrics and preview 029; no model or training execution."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess

ROOT = Path('/home/featurize/work/cloud_project_v2')
OUT = Path(__file__).parent
RUNS = ROOT.parent / 'runs'
names = dict(bce='026_tiny512_bce_seed43', original='027_tiny512_kd_seed43',
             short12='028_tiny512_kd_short12_seed43')
hist = {k: [json.loads(s) for s in (RUNS / n / 'history.jsonl').read_text().splitlines()]
        for k, n in names.items()}
best = {k: json.loads((RUNS / n / 'best_metrics.json').read_text()) for k,n in names.items()}
rows = []
for i in range(10):
    rows.append(dict(class_id=i+1, original_best_auc=best['original']['per_class_auc'][i],
        short12_best_auc=best['short12']['per_class_auc'][i],
        short12_minus_original=best['short12']['per_class_auc'][i]-best['original']['per_class_auc'][i],
        short12_last_minus_best=hist['short12'][-1]['validation']['per_class_auc'][i]-best['short12']['per_class_auc'][i]))
analysis = dict(checked_at_hk=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    metric_context='native BF16 Tiny512 hflip2 Study859; each checkpoint selected by same official valid; one seed43',
    best_auc={k: v['macro_auc'] for k,v in best.items()},
    best_epoch={k:v['epoch'] for k,v in best.items()},
    short12_minus_original=best['short12']['macro_auc']-best['original']['macro_auc'],
    classes_improved=sum(r['short12_minus_original']>0 for r in rows),
    classes_declined=sum(r['short12_minus_original']<0 for r in rows), per_class=rows,
    short12_curve=[dict(epoch=r['epoch'], auc=r['validation']['macro_auc'],
        macro_ap=r['validation']['macro_ap'], valid_bce=r['validation']['loss'],
        train_loss=r['training']['loss'], hard_bce=r['training']['hard_bce'],
        kl_T2=r['training']['soft_kl_T2'], backbone_lr=r['training']['learning_rates']['backbone_decay'])
        for r in hist['short12']],
    short12_best_to_last_auc=hist['short12'][-1]['validation']['macro_auc']-best['short12']['macro_auc'],
    original_best_to_last_auc=hist['original'][-1]['validation']['macro_auc']-best['original']['macro_auc'],
    short12_successful_epoch_train_seconds=sum(r['training']['seconds'] for r in hist['short12']),
    successful_epoch_time_excludes='validation/save/load, pauses, and discarded partial E8 work',
    uncertainty=dict(paired_patient_ci_available=False, class_or_error_slices_available=False,
        reason='028 has saved aggregate metrics but no deployed/per-Study probability validation output',
        seed_replicates=1, statistical_equivalence_proved=False, pure_noise_proved=False,
        selection_bias='both best epochs selected on same finite valid; ten classes correlated'),
    decision='keep027; retain028 as negative experiment; no current evidence to adopt short12',
    model_forward_or_training_started=False, test_used=False)
(OUT / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2)+'\n')

python = '/home/featurize/.conda/envs/cxr-kd-v1/bin/python'
argv = [python, '-u', 'train_distill.py', '--fold-plan', 'data/cv5_overnight/folds.json',
    '--soft-labels', '../runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2',
    '--training-mode','distillation','--alpha','0.2','--temperature','2',
    '--image-size','768','--seed','43','--validation-tta','hflip2',
    '--batch-size','8','--accumulation-steps','4','--epochs','25','--patience','5',
    '--output','../runs/029_tiny768_kd_seed43']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
    NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
preview = json.loads(subprocess.check_output(argv, cwd=ROOT, env=env, text=True))
old = json.loads((RUNS / names['original'] / 'config.json').read_text())
diff = {k:dict(original=old[k], proposed=v) for k,v in preview['config'].items() if old[k] != v}
assert set(diff) == {'image_size','output'}, diff
assert preview['dry_run'] is True and preview['training_started'] is False
assert not (RUNS / '029_tiny768_kd_seed43').exists()
rules_hash = hashlib.sha256((ROOT.parent / 'rules.pdf').read_bytes()).hexdigest()
assert rules_hash == 'ff6aa3adbfd3d000194253055a93dbce8a140031c8432f892046a49de8136fb6'
proposal = dict(differences_from_027=diff, preview=preview, preview_only_argv=argv,
    user_training_argv=argv+['--execute'], proposal_not_executed=True, output_does_not_exist=True,
    time_estimate='2.5-4h if ~12 epochs; 6-7h if all25; recalibrate after first epoch',
    time_estimate_basis='027 actual12epochs85m46s and imagearea(768/512)^2=2.25; not a measured768benchmark',
    retained_batch_size=8, accumulation_steps=4, effective_batch=32,
    memory_status='feasible hypothesis on4090; no768forward or measured guarantee',
    oof_reusable=True, no_teacher_training_or_reextraction=True,
    fp16_size_estimate='same architecture, ~67MB withprovenance; exactexport pending',
    t4_latency_pending=True, official_local_pdf_sha256=rules_hash,
    source_model_card='https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384',
    local_frozen_pretrain_revision='32c741ba106f9e062174ef829354a16719c3dcac',
    source_paper='https://arxiv.org/abs/2201.03545')
(OUT / 'tiny768_proposal.json').write_text(json.dumps(proposal, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(best_auc=analysis['best_auc'], delta=analysis['short12_minus_original'],
    class_deltas=rows, late_delta=analysis['short12_best_to_last_auc'],
    successful_epoch_train_seconds=analysis['short12_successful_epoch_train_seconds'],
    next_differences=diff, preview_only=True), indent=2))
