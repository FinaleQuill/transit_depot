"""Run isolated MPS resource profiles; these are not virtual or certified T4s.

Preview by default. --execute starts a private, user-owned MPS controller,
verifies every quota, runs the shared local benchmark, and stops the controller.
No GPU power limit, clock, compute mode, or deployment source is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CALIBRATION = r'''
import json,os,torch
torch.set_num_threads(1)
torch.cuda.init()
p=torch.cuda.get_device_properties(0)
requested=int(os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
if requested<100 and p.multi_processor_count>=128:
    raise RuntimeError('MPS quota ineffective')
torch.cuda.set_per_process_memory_fraction(16*1024**3/p.total_memory)
n=4096
a=torch.randn((n,n),device='cuda',dtype=torch.float16)
b=torch.randn_like(a);c=torch.empty_like(a)
for _ in range(10):torch.mm(a,b,out=c)
start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(30):torch.mm(a,b,out=c)
end.record();end.synchronize()
gemm_ms=start.elapsed_time(end)
del a,b,c
elements=256*1024**2
x=torch.empty(elements,device='cuda',dtype=torch.uint8)
y=torch.empty_like(x)
for _ in range(5):y.copy_(x)
start.record()
for _ in range(30):y.copy_(x)
end.record();end.synchronize()
copy_ms=start.elapsed_time(end)
print(json.dumps(dict(gpu=p.name,sm_count=p.multi_processor_count,
    requested_thread_percentage=requested,fp16_gemm_n=n,fp16_gemm_repeats=30,
    fp16_gemm_ms=gemm_ms,fp16_gemm_tflops=2*n**3*30/(gemm_ms/1000)/1e12,
    copy_payload_bytes=elements,copy_repeats=30,copy_ms=copy_ms,
    copy_read_plus_write_gbps=2*elements*30/(copy_ms/1000)/1e9,
    T4_equivalence_proved=False)))
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def control(binary, env, command):
    return subprocess.run([binary], input=command+'\n', env=env,
                          text=True, capture_output=True, timeout=30)


def run(*, model_dirs, names, inputs, output, percentages, execute=False):
    root = Path(__file__).resolve().parent
    inputs, output = Path(inputs).resolve(strict=True), Path(output).resolve()
    models = [Path(p).resolve(strict=True) for p in model_dirs]
    if (not models or len(models) != len(names) or len(set(names)) != len(names)
            or any(not name or not name.replace('_','').isalnum() for name in names)):
        raise ValueError('Supply matching, unique simple candidate names and model directories')
    if (not percentages or len(set(percentages)) != len(percentages)
            or any(type(p) is not int or not 1<=p<=100 for p in percentages)):
        raise ValueError('Unique integer percentages in1..100 required')
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in [inputs,*models]):
        raise ValueError('Results must not overlap model or input directories')
    binary = shutil.which('nvidia-cuda-mps-control')
    if binary is None:
        raise FileNotFoundError('NVIDIA MPS controller is unavailable')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model_dirs={name:str(path) for name,path in zip(names,models)},
        inputs=str(inputs), output=str(output), percentages=percentages,
        memory_allocator_limit_gib=16, actual_T4_test=False,
        hardware_clock_or_power_changes=False, executed=execute,
        limitation='MPS quotas do not reproduce T4 architecture, bandwidth, kernels or latency')
    if not execute:
        return plan
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True).strip()
    if active:
        raise RuntimeError('GPU compute processes are active; benchmark requires an idle GPU')
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(root/name):digest(root/name) for name in
        ['run_resource_stress.py','benchmark_local_resources.py','prepare_latency_inputs.py',
         'cost.py','predict.py','deploy_kd.py','model_overnight.py']}
    hashes = {str(f):digest(f) for directory in models for f in directory.iterdir()
              if f.is_file() and (f.suffix=='.pt' or f.name=='config.json')}
    plan.update(source_hashes=sources, inference_input_hashes=hashes,
                selection_sha256=digest(inputs/'selection.json'), cases=[], calibration=[])
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    try:
        with tempfile.TemporaryDirectory(prefix='cxr-resource-mps-') as temp:
            pipe, logs = Path(temp)/'pipe', Path(temp)/'logs'
            pipe.mkdir(); logs.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_MPS_PIPE_DIRECTORY=str(pipe),
                CUDA_MPS_LOG_DIRECTORY=str(logs), CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING='1',
                CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='0=16G', PYTHONDONTWRITEBYTECODE='1',
                NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            started = subprocess.run([binary,'-d'],env=env,capture_output=True,text=True)
            plan['controller_start'] = dict(returncode=started.returncode,
                                            stdout=started.stdout,stderr=started.stderr)
            try:
                started.check_returncode()
                for percentage in percentages:
                    client_env = dict(env, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(percentage))
                    calibration = subprocess.run([sys.executable,'-u','-c',CALIBRATION],cwd=root,
                        env=client_env,text=True,capture_output=True,timeout=120)
                    (output/f'calibration_{percentage}.log').write_text(calibration.stdout+calibration.stderr)
                    calibration.check_returncode()
                    calibrated = json.loads(calibration.stdout)
                    plan['calibration'].append(calibrated)
                    print(f"[PROFILE] {percentage}%: actualSM={calibrated['sm_count']}, "
                          f"FP16 GEMM={calibrated['fp16_gemm_tflops']:.2f}TF/s",flush=True)
                    for name, model in zip(names,models):
                        case = output/f'mps{percentage}_{name}'
                        command = [sys.executable,'-u',str(root/'benchmark_local_resources.py'),
                            '--model-dir',str(model),'--pictures-100',str(inputs/'pictures_100'),
                            '--pictures-5',str(inputs/'pictures_5'),'--selection-json',str(inputs/'selection.json'),
                            '--output-dir',str(case),'--profile',f'mps{percentage}_{name}',
                            '--requested-thread-percentage',str(percentage),
                            '--baseline-sm-count','128','--memory-limit-gib','16','--execute']
                        print(f'[START] mps{percentage} {name}',flush=True)
                        with (output/f'mps{percentage}_{name}.log').open('w') as log:
                            completed = subprocess.run(command,cwd=root,env=client_env,stdout=log,
                                stderr=subprocess.STDOUT,text=True,timeout=600)
                        if completed.returncode:
                            raise RuntimeError(f'{name}/{percentage} failed; see its preserved log')
                        report=json.loads((case/'local_resource_report.json').read_text())
                        plan['cases'].append(dict(candidate=name,percentage=percentage,
                            visible_sm_count=report['resources']['cuda_visible_sm_count'],
                            mean_ms=report['mean_ms'],p95_ms=report['p95_ms'],max_ms=report['max_ms'],
                            round_mean_ms=report['round_mean_ms'],
                            actual_inference_bytes=report['inference_files']['total_bytes'],
                            comparison=report['comparison']['status'],
                            report=str(case/'local_resource_report.json'),
                            report_sha256=digest(case/'local_resource_report.json')))
                        print(f"[DONE] mps{percentage} {name}: {report['mean_ms']:.3f}ms/image",flush=True)
                        (output/'progress.json').write_text(json.dumps(plan,indent=2)+'\n')
            finally:
                stopped = control(binary,env,'quit')
                plan['controller_stop'] = dict(returncode=stopped.returncode,
                                              stdout=stopped.stdout,stderr=stopped.stderr)
                shutil.copytree(logs,output/'mps_logs')
        if any(digest(path)!=value for path,value in {**sources,**hashes}.items()):
            raise RuntimeError('Source or inference artifact changed during measurement')
        plan['complete']=True
        plan['original_sources_and_inference_files_unchanged']=True
    except BaseException as exc:
        plan.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        plan['ended_utc']=datetime.now(timezone.utc).isoformat()
        (output/'resource_stress_summary.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dirs',nargs='+',type=Path,required=True)
    parser.add_argument('--names',nargs='+',required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--percentages',nargs='+',type=int,default=[100,30,20,10])
    parser.add_argument('--execute',action='store_true')
    result=run(**vars(parser.parse_args()))
    print(json.dumps({key:value for key,value in result.items()
                      if key not in ('source_hashes','inference_input_hashes')},indent=2))


if __name__=='__main__':
    main()
