"""Prepare and benchmark an independent copy of the actual submission project."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '038_dual768_optimized_project_20261030'
PROJECT = Path('/tmp/dual768_2080ti_diagnosis/final_project_isolated')
INPUTS = Path('/tmp/dual768_2080ti_diagnosis/inputs')

def read_studies(path):
    import numpy as np
    result = {}
    for row in csv.DictReader(path.open()):
        key = row['Subject_id'], row['Study_id']
        result.setdefault(key, np.zeros(10, dtype=np.float64))[int(row['Predict_class'])-1] = float(row['Probability'])
    return result

def execute_project():
    sys.path.insert(0, str(PROJECT/'code'))
    from inference_runtime import activate_dependencies
    activate_dependencies()
    import torch
    import numpy as np
    import timm
    import cost
    import deploy_kd
    import model_overnight
    import inference_inputs
    import fast_lanczos
    from inference_reproducibility import configure_inference
    for module in [cost, deploy_kd, model_overnight, inference_inputs, fast_lanczos]:
        assert Path(module.__file__).resolve().is_relative_to(PROJECT/'code'), module.__file__
    assert Path(timm.__file__).resolve().is_relative_to(PROJECT/'code/_runtime_addons')
    configure_inference(torch_module=torch)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/props.total_memory)
    measured = deploy_kd.discover_images(INPUTS/'pictures_100')
    warmup = deploy_kd.discover_images(INPUTS/'pictures_5')
    assert len(measured) == 100 and len(warmup) == 5
    selection = json.loads((INPUTS/'selection.portable.json').read_text())
    expected_sha = {(r['subject_id'], r['study_id'], Path(r['relative_path']).name):r['sha256']
                    for group in ['measured','warmup'] for r in selection[group]}
    for row in measured+warmup:
        key = row['subject_id'], row['study_id'], row['path'].name
        assert deploy_kd.sha256_file(row['path']) == expected_sha[key]
    predictor = deploy_kd.Predictor(PROJECT/'model')
    reference = read_studies(ROOT/'local_baseline_v1/round_1.csv')
    expected_image_bytes = (ROOT/'local_baseline_v1/round_1.image_predictions.csv').read_bytes()
    expected_study_bytes = (ROOT/'local_baseline_v1/round_1.csv').read_bytes()
    output = PROJECT/'local_2080ti_benchmark'
    output.mkdir()
    for row in warmup:
        predictor.predict_image(row['path'])
    predictor.synchronize()
    rounds, comparisons = [], []
    for index in range(3):
        path = output/f'round_{index+1}.csv'
        durations, actual = cost.time_round(predictor, measured, path)
        rounds.append(durations)
        comparison = cost.compare_predictions(actual, reference)
        assert comparison['match'], comparison
        image_bytes = path.with_suffix('.image_predictions.csv').read_bytes()
        assert image_bytes == expected_image_bytes
        assert path.read_bytes() == expected_study_bytes
        comparisons.append(comparison)
    times, per_image, mean = cost.summarize_timings(rounds)
    report = {'project_root': str(PROJECT), 'model_dir': str(PROJECT/'model'),
        'official_T4_certified': False, 'gpu': props.name, 'sm': props.multi_processor_count,
        'allocator_gib': 8, 'external_addons_removed': True, 'same_105_image_hashes': True,
        'protocol': {'measured':100, 'warmup':5, 'rounds':3, 'batch':1, 'serial':True,
                     'timing':'file_read_to_actual_CSV_write', 'model_initialization_timed':False},
        'environment':cost.timing_environment(predictor), 'mean_ms':mean,
        'round_mean_ms':times.mean(axis=1).tolist(), 'per_image_round_ms':times.T.tolist(),
        'p95_ms':float(np.percentile(per_image,95)), 'max_ms':float(per_image.max()),
        'image_and_study_CSV_byte_equal_all_rounds':True, 'study_comparisons':comparisons,
        'inference_inventory':predictor.inventory,
        'module_paths':{m.__name__:m.__file__ for m in [cost,deploy_kd,model_overnight,inference_inputs,fast_lanczos,timm]}}
    (output/'project_runtime_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_image_round_ms','study_comparisons','inference_inventory']},indent=2))
    return 0

def main():
    if '--execute-project' in sys.argv:
        return execute_project()
    started = time.perf_counter()
    shutil.copytree(SOURCE, PROJECT)
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    with (PROJECT/'isolated_preparation.log').open('x') as log:
        subprocess.run([sys.executable,str(PROJECT/'code/prepare_inference.py'),'--force-build',
                        '--output',str(PROJECT/'isolated_preparation.json')],
                       cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'packaged_project_benchmark.log').open('x') as log:
        result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--execute-project'],
                                cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt = {'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'exit_code':result.returncode,
               'copy_prepare_and_benchmark_wall_seconds':time.perf_counter()-started,
               'note':'copy and preparation precede the timed three-round loop; no downloads or other GPU jobs overlap timing'}
    (ROOT/'packaged_project_execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if (PROJECT/'local_2080ti_benchmark').exists():
        shutil.copytree(PROJECT/'local_2080ti_benchmark',ROOT/'packaged_project_benchmark')
    for name in ['isolated_preparation.log','isolated_preparation.json']:
        shutil.copyfile(PROJECT/name,ROOT/('packaged_'+name))
    print(json.dumps(receipt,indent=2))
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
