"""Fixed-eight-image, CPU-only comparison to current v1, not old033.

Run after both prototype validation jobs have stopped. This is a diagnostic,
not a formal end-to-end GPU benchmark, and makes no deployment change.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

from parallel_lanczos_both_axes import load_image_arrays as load_parallel

PROJECT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT))
import deploy_kd as current_v1
BUNDLE=PROJECT.parent/'deliveries/034_dual768_proxy_timing_20261029/bundle'


def cgroup_cpu():
    rel=Path('/proc/self/cgroup').read_text()
    current=Path('/sys/fs/cgroup')
    leaf=None
    for line in rel.splitlines():
        if line.startswith('0::'):
            leaf=current/line.split('::',1)[1].lstrip('/')
    result={'proc_self_cgroup':rel,'affinity':sorted(os.sched_getaffinity(0)),'entries':[]}
    if leaf is not None:
        for p in [leaf,*list(leaf.parents)]:
            if not p.is_relative_to(current):continue
            entry={'path':str(p)}
            for name in ('cpu.max','cpu.stat','cpu.cfs_quota_us','cpu.cfs_period_us'):
                q=p/name
                entry[name]=q.read_text() if q.is_file() else None
            result['entries'].append(entry)
    return result


def main():
    before=cgroup_cpu()
    paths=sorted((BUNDLE/'inputs/pictures_100').rglob('*.jpg'))[:8]
    rows=[]
    for index,path in enumerate(paths):
        samples={'current_v1_load768_ms':[],'parallel_both_axes4_load768_ms':[]}
        methods={'current_v1_load768_ms':lambda:current_v1.load_image_arrays(path,[768]),
                 'parallel_both_axes4_load768_ms':lambda:load_parallel(path,[768],workers=4,min_pixels=1_000_000)}
        with Image.open(path)as image: dimensions=list(image.size);mode=image.mode
        for repeat in range(3):
            names=list(methods)
            if(index+repeat)%2:names.reverse()
            values={}
            for name in names:
                start=time.perf_counter();values[name]=methods[name]();elapsed=(time.perf_counter()-start)*1000
                samples[name].append(elapsed)
            assert values[names[0]][768].tobytes()==values[names[1]][768].tobytes()
        rows.append({'sample_index':index,'source_mode':mode,'source_size':dimensions,
                     'all_ms':samples,'median_ms':{key:statistics.median(value)for key,value in samples.items()}})
    mean={key:statistics.mean(row['median_ms'][key]for row in rows)for key in methods}
    result={
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'CPU-only diagnostic after both artifact validation jobs stopped; fixed8 images,3 alternating-order observations per implementation; no formal GPU/end-to-end benchmark',
        'comparison':'current_v1 includes CHW normalization and deferred finite checks; CPU load path only,not old033',
        'python':platform.python_version(),'numpy':np.__version__,'Pillow':pillow_version,
        'max_workers_per_source_image':4,'cross_image_prefetch':False,
        'current_v1_source_sha256':hashlib.sha256((PROJECT/'deploy_kd.py').read_bytes()).hexdigest(),
        'parallel_source_sha256':hashlib.sha256(Path(__file__).with_name('parallel_lanczos_both_axes.py').read_bytes()).hexdigest(),
        'all_float32_byte_parity_checks_pass':True,
        'mean_sample_median_ms':mean,
        'net_cpu_ms_saved':mean['current_v1_load768_ms']-mean['parallel_both_axes4_load768_ms'],
        'cgroup_before':before,'cgroup_after':cgroup_cpu(),
        'loadavg':Path('/proc/loadavg').read_text(),'samples':rows,
        'training_or_gpu_executed':False,'production_source_changed':False,
    }
    with Path(__file__).with_name('net_vs_v1_report.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({key:value for key,value in result.items()if key not in ('samples','cgroup_before','cgroup_after')},indent=2))


if __name__=='__main__':main()
