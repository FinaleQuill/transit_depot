"""Probe private MPS quotas only, with guaranteed controller cleanup."""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone, timedelta

OUT = Path(__file__).parent
PYTHON = '/home/featurize/.conda/envs/cxr-kd-v1/bin/python'
CONTROL = '/usr/bin/nvidia-cuda-mps-control'
ROOT = Path('/home/featurize/work/cloud_project_v2')
code = '''import json,os,torch
torch.cuda.init()
x=torch.ones(1,device="cuda")
torch.cuda.synchronize()
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(pid=os.getpid(),name=p.name,sm_count=p.multi_processor_count,
    total_memory=p.total_memory,compute_capability=[p.major,p.minor],torch=torch.__version__,
    cuda=torch.version.cuda,requested=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))),flush=True)
input()
'''
rows = []
result = dict(started_hk=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    gpu_clock_or_power_changes=False, training_started=False, model_loaded=False,
    T4_emulation_validated=False)
with tempfile.TemporaryDirectory(prefix='cxr-resource-mps-') as directory:
    base = Path(directory)
    pipe, logs = base/'pipe', base/'logs'
    pipe.mkdir(); logs.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_MPS_PIPE_DIRECTORY=str(pipe),
        CUDA_MPS_LOG_DIRECTORY=str(logs), CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING='1',
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='0=16G', PYTHONDONTWRITEBYTECODE='1',
        NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)
    started = subprocess.run([CONTROL,'-d'],env=env,capture_output=True,text=True)
    result['start_control'] = dict(returncode=started.returncode,stdout=started.stdout,stderr=started.stderr)
    print('MPS_START',json.dumps(result['start_control']),flush=True)
    try:
        started.check_returncode()
        for quota in [100,30,20,10]:
            childenv = dict(env, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(quota))
            child = subprocess.Popen([PYTHON,'-u','-c',code],env=childenv,cwd=ROOT,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                line=child.stdout.readline()
                if not line:
                    stderr=child.stderr.read()
                    raise RuntimeError(f'MPS probe child failed: {stderr}')
                row=json.loads(line)
                server=subprocess.run([CONTROL],input='get_server_list\n',env=env,capture_output=True,text=True)
                row['server_list']=server.stdout
                memberships=[]
                for pid in re.findall(r'(?m)^\s*(\d+)\s*$',server.stdout):
                    clients=subprocess.run([CONTROL],input=f'get_client_list {pid}\n',env=env,
                        capture_output=True,text=True)
                    memberships.append(dict(server_pid=pid,clients=clients.stdout))
                row['memberships']=memberships
                row['membership_proven']=any(str(row['pid']) in m['clients'].split() for m in memberships)
                assert row['membership_proven'],row
                if quota<100:
                    assert 0<row['sm_count']<rows[0]['sm_count'],row
                rows.append(row)
                print('MPS_PROBE',json.dumps(row),flush=True)
            finally:
                if child.poll() is None:
                    child.communicate('\n',timeout=30)
                else:
                    child.communicate(timeout=30)
            assert child.returncode==0,child.returncode
        result['quota_restriction_proven']=True
    finally:
        stopped=subprocess.run([CONTROL],input='quit\n',env=env,capture_output=True,text=True,timeout=30)
        result['stop_control']=dict(returncode=stopped.returncode,stdout=stopped.stdout,stderr=stopped.stderr)
        result['rows']=rows
        result['logs']={f.name:f.read_text(errors='replace') for f in logs.glob('*') if f.is_file()}
        (OUT/'mps_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('MPS_STOP',json.dumps(result['stop_control']),flush=True)
