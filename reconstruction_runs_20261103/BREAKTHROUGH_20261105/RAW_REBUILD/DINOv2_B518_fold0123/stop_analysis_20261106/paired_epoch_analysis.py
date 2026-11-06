"""CPU-only fixed comparisons of saved development predictions after user stop.

No model import, weights, image decoding, inference, training or test access.
Exactly E20-E18, E21-E20 and fixed E20 50:50 CNN blend-E20 are compared.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cxr_stop_analysis_mpl_20261106')

ROOT = Path('/home/featurize/work/cxr_research_20261103')
RUN = Path('/home/featurize/work/reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD/DINOv2_B518_fold0123')
OUTPUT = RUN / 'stop_analysis_20261106'
sys.path.insert(0, str(ROOT))

import numpy as np
from breakthrough.generalization_monitor import compare, read_prediction, sha, write
from raw_rebuild.comparison import align_predictions

REPEATS = 2000
SEED = 20261105  # Fixed by the registered compare() implementation.
PAIRS = ('e20_minus_e18', 'e21_minus_e20', 'e20_fixed_blend_minus_e20_single')
SLICES = ('all', 'ap_projection_only', 'single_image_study',
          'multiple_image_study', 'at_least_5_studies_patient',
          'at_least_10_studies_patient', 'one_study_patient', 'multiple_study_patient')


def compare_fixed(name, config):
    started = time.monotonic()
    if name == 'e20_fixed_blend_minus_e20_single':
        current_path = RUN / 'epoch_20/predictions.npz'
        old_path = Path(config['reference_predictions'])
        if sha(old_path) != config['reference_sha256']:
            raise ValueError('Registered CNN prediction SHA mismatch')
        single, cnn, coverage = align_predictions(read_prediction(current_path), read_prediction(old_path))
        candidate = dict(single, probabilities=.5*single['probabilities']+.5*cnn['probabilities'])
        reference = single
        policy = dict(candidate='Fixed 0.5 DINO E20 + 0.5 historical CNN probabilities',
                      reference='DINO E20 alone', blend_weights_fitted=False,
                      reference_input_coverage=coverage)
    else:
        current_epoch, old_epoch = (20,18) if name == 'e20_minus_e18' else (21,20)
        current_path = RUN / f'epoch_{current_epoch:02d}/predictions.npz'
        old_path = RUN / f'epoch_{old_epoch:02d}/predictions.npz'
        candidate, reference, coverage = align_predictions(read_prediction(current_path), read_prediction(old_path))
        if coverage['image_count_difference'] or coverage['studies_with_view_metadata_changes']:
            raise ValueError('Epoch prediction metadata changed')
        policy = dict(candidate=f'DINO E{current_epoch}', reference=f'DINO E{old_epoch}',
                      same_model_architecture=True, same_image_coverage=True)
    before = {str(p):sha(p) for p in (current_path,old_path)}
    report = compare(candidate, reference, repeats=REPEATS)
    # compare() adds historical DINO-vs-CNN limitations that do not apply to
    # every fixed contrast; replace them with precise limitations here.
    report['limitations'] = [
        'This is the development cohort repeatedly used for checkpoint selection.',
        'Percentile intervals are exploratory and not corrected for checkpoint selection or multiple slices.',
        'No hidden-test or official-validation result is computed or implied.',
        'An interval crossing zero does not establish equivalence or non-inferiority.',
    ]
    if name == 'e20_fixed_blend_minus_e20_single':
        report['limitations'].append('CNN reference used 6064 historical images; DINO uses all 6067 original images of the same Studies.')
    report.update(comparison_name=name, policy=policy, input_sha256=before,
                  cpu_only=True, model_weights_loaded=False, training_started=False,
                  prediction_inference_started=False, compared_epochs_fixed_in_advance=True,
                  epoch_or_blend_search=False, bootstrap_seed=SEED,
                  elapsed_seconds=time.monotonic()-started)
    if any(sha(path) != value for path,value in before.items()):
        raise ValueError('Saved predictions changed during analysis')
    write(OUTPUT / f'{name}.json', report)
    summary = {}
    for key in SLICES:
        result = report['overall'] if key == 'all' else report['slices'][key]
        interval = result.get('patient_bootstrap',{}).get('paired_delta_macro_auc')
        if interval is None:
            interval = result['paired_patient_bootstrap']
        summary[key] = dict(candidate_macro_auc=result['macro_auc'],
            reference_macro_auc=result['reference']['macro_auc'],
            delta=result['macro_auc_delta'], **interval,
            studies=result.get('num_studies',result.get('studies')),
            patients=result.get('num_patients',result.get('patients')))
    return name, summary


def plot_history(history, partial):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    epochs = np.asarray([r['epoch'] for r in history])
    losses = np.asarray([r['loss'] for r in history])
    aucs = np.asarray([r['macro_auc'] for r in history])
    if epochs.tolist() != list(range(1,22)):
        raise ValueError('Expected exactly complete epochs 1-21')
    if epochs[int(np.argmax(aucs))] != 20:
        raise ValueError('Recorded best complete epoch is not E20')
    with plt.rc_context({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':150}):
        fig,axes = plt.subplots(1,2,figsize=(11.8,4.4))
        axes[0].plot(epochs, losses, color='#16697a', marker='o', markersize=3,
                     linewidth=1.8, label='Complete-epoch training loss')
        axes[0].set(ylabel='Mean weighted BCE loss', title='Training loss')
        axes[1].plot(epochs, aucs, color='#2c6eaa', marker='o', markersize=3,
                     linewidth=1.8, label='Development macro AUC (EMA)')
        axes[1].scatter([20],[aucs[19]],color='#bf4e30',s=55,zorder=5)
        axes[1].annotate(f'Best E20: {aucs[19]:.6f}', xy=(20,aucs[19]),
                         xytext=(8,aucs[19]-.030),
                         arrowprops={'arrowstyle':'->','color':'#bf4e30'},
                         color='#bf4e30', fontsize=10)
        axes[1].set(ylabel='10-class mean ROC AUC', title='Patient-separated development set')
        for ax in axes:
            ax.axvline(18,color='#777777',linestyle='--',linewidth=1.2,
                       label='Original 18-epoch limit')
            ax.axvspan(18,21.4,color='#d8e5ed',alpha=.35,zorder=-1)
            ax.set(xlabel='Complete epoch',xlim=(.6,21.4))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
            ax.grid(alpha=.22)
            ax.legend(loc='lower left' if ax is axes[1] else 'upper right',fontsize=8)
        fig.suptitle('Raw-data DINOv2: completed epochs 1-21',fontsize=13)
        fig.text(.5,.015,
            f'Partial E22 ({partial["studies_done"]:,}/{partial["studies_total"]:,} Studies) excluded. '
            'Development results are not hidden-test scores.', ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.05,1,.94))
        fig.savefig(OUTPUT/'training_loss_dev_auc_epochs_01_21.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    with (OUTPUT/'completed_epoch_curve.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['epoch','loss','macro_auc','patient_equal_macro_auc'])
        writer.writeheader()
        for row in history:
            writer.writerow({key:row[key] for key in writer.fieldnames})


def main():
    OUTPUT.mkdir(exist_ok=True)
    config = json.loads((RUN/'configuration.json').read_text())['configuration']
    history = [json.loads(line) for line in (RUN/'history.jsonl').read_text().splitlines() if line]
    partial_path = OUTPUT / 'progress.json.before_stop_summary'
    partial = json.loads(partial_path.read_text())
    inputs = [RUN/'history.jsonl', partial_path, RUN/'configuration.json',
              ROOT/'reconstruction/evaluate.py', ROOT/'raw_rebuild/comparison.py',
              ROOT/'breakthrough/generalization_monitor.py', Path(__file__)]
    before = {str(path):sha(path) for path in inputs}
    plot_history(history, partial)
    summaries = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(compare_fixed,name,config) for name in PAIRS]
        for future in as_completed(futures):
            name, summary = future.result()
            summaries[name] = summary
            print(json.dumps({'comparison':name,'overall':summary['all']},ensure_ascii=False),flush=True)
    result = dict(schema='fixed_stopped_run_development_comparisons_v1',
                  created_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
                  comparisons=summaries, bootstrap_repeats=REPEATS, bootstrap_seed=SEED,
                  resampling='Patient clusters, all Studies and sampled patient multiplicity; paired draws.',
                  interval='Percentile 95%, exploratory; not selection-bias or multiplicity corrected.',
                  complete_epochs=21, best_complete_epoch=20, partial_epoch_excluded=partial,
                  figure=str(OUTPUT/'training_loss_dev_auc_epochs_01_21.png'),
                  input_and_source_sha256=before, cpu_only=True,
                  training_started=False, model_inference_started=False,
                  test_accessed=False, official_validation_accessed=False,
                  hidden_test_improvement_established=False)
    if any(sha(path) != value for path,value in before.items()):
        raise ValueError('Analysis input/source changed during execution')
    write(OUTPUT/'paired_comparison_summary.json',result)
    print(json.dumps({'complete':True,'summary':str(OUTPUT/'paired_comparison_summary.json')}),flush=True)


if __name__ == '__main__':
    main()
