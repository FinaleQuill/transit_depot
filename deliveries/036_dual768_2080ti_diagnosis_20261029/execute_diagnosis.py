"""Run prepared diagnostics sequentially after pinned-image restoration completes."""
from pathlib import Path
import datetime as dt
import json
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/featurize/work/cloud_project_v2')
LOCAL = Path('/tmp/dual768_2080ti_diagnosis')
MODEL = Path('/home/featurize/work/deliveries/033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2')
BUNDLE = Path('/home/featurize/work/deliveries/034_dual768_proxy_timing_20261029/bundle')
RUNNER = ROOT / 'run_container.py'

def run(name, args):
    print(f'START {name}', flush=True)
    start = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with (ROOT / (name + '.console.log')).open('x') as log:
        result = subprocess.run([sys.executable, str(RUNNER), *map(str, args)], stdout=log, stderr=subprocess.STDOUT)
    receipt = {'started_utc': started, 'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'wall_seconds': time.perf_counter() - start, 'python_arguments': list(map(str, args)),
               'exit_code': result.returncode}
    (ROOT / (name + '.execution.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'END {name} exit={result.returncode} seconds={receipt["wall_seconds"]:.1f}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

def main():
    assert (ROOT / 'environment/docker_pull.exit_code.txt').read_text().strip() == '0'
    probe = subprocess.run([sys.executable, str(RUNNER), str(REPO / 'artifacts/kd_v1/optimization_2080ti_20261029/environment/probe_container.py')], capture_output=True, text=True)
    (ROOT / 'environment/competition_probe.json').write_text(probe.stdout)
    (ROOT / 'environment/competition_probe.stderr.txt').write_text(probe.stderr)
    assert probe.returncode == 0, probe.stderr
    info = json.loads(probe.stdout)
    assert info['torch']['cuda_available']
    assert info['module_runtime_versions']['torchvision'] == '0.25.0+cu128'
    assert info['module_runtime_versions']['numpy'] == '1.26.4'
    assert info['module_runtime_versions']['PIL'] == '10.4.0'
    assert all(x == 'ok' for x in info['import_checks'].values()), info
    print('Pinned image CUDA/import probe passed', flush=True)
    extra = subprocess.run([sys.executable, str(RUNNER), '-c',
        'import json,os,torch; from PIL import features; print(json.dumps({"libjpeg_turbo": features.check_feature("libjpeg_turbo"), "libjpeg_turbo_version": features.version_feature("libjpeg_turbo"), "jpeg_version": features.version_codec("jpg"), "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads_before_predictor": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads()}))'], capture_output=True, text=True)
    (ROOT / 'environment/image_features.json').write_text(extra.stdout)
    (ROOT / 'environment/image_features.stderr.txt').write_text(extra.stderr)
    assert extra.returncode == 0, extra.stderr
    # No installation, download, extraction, CPU profiler or other GPU job overlaps the baseline.
    stage = REPO / 'artifacts/kd_v1/optimization_2080ti_20261029/run_stage.py'
    for name, inputs_bundle, output in (
        ('benchmark_nfs_v1', BUNDLE, ROOT / 'benchmark_nfs_v1'),
        ('benchmark_local_v1', LOCAL, LOCAL / 'benchmark_local_v1')):
        run(name, [stage, 'benchmark', '--source-root', REPO, '--model-dir', MODEL,
                   '--bundle-root', inputs_bundle, '--output-dir', output])
    for name, inputs, output in (
        ('diagnostic_nfs_v1', BUNDLE / 'inputs', ROOT / 'diagnostic_nfs_v1'),
        ('diagnostic_local_v1', LOCAL / 'inputs', LOCAL / 'diagnostic_local_v1')):
        run(name, [ROOT / 'diagnose_current_v1.py', '--source-root', REPO, '--model-dir', MODEL,
                   '--inputs-root', inputs, '--selection-json', inputs / 'selection.portable.json',
                   '--output-dir', output])
    for name in ('benchmark_local_v1', 'diagnostic_local_v1'):
        shutil.copytree(LOCAL / name, ROOT / name)
    receipt = (LOCAL / 'benchmark_local_v1.receipt.json')
    shutil.copyfile(receipt, ROOT / 'benchmark_local_v1.receipt.json')
    print('All diagnostic runs complete; local reports copied to persistent work', flush=True)

if __name__ == '__main__':
    main()
