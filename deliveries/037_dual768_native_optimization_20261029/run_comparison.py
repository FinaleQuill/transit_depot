"""Run frozen source candidates sequentially with the unchanged three-round timer."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / '036_dual768_2080ti_diagnosis_20261029/run_container.py'
STAGE = Path('/home/featurize/work/cloud_project_v2/artifacts/kd_v1/optimization_2080ti_20261029/run_stage.py')
LOCAL = Path('/tmp/dual768_2080ti_diagnosis')
MODEL = ROOT.parent / '033_dual_tiny768_seed43_25_seed44_75/fp16_hflip2'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', action='append', required=True, help='label=source_directory')
    parser.add_argument('--nfs', action='store_true')
    args = parser.parse_args()
    for value in args.stage:
        label, source = value.split('=', 1)
        assert label and '/' not in label
        source = Path(source).resolve(strict=True)
        local_output = ROOT / label if args.nfs else LOCAL / label
        assert not local_output.exists() and not (ROOT / (label + '.execution.json')).exists()
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        assert not processes.strip(), processes
        inputs = ROOT.parent / '034_dual768_proxy_timing_20261029/bundle' if args.nfs else LOCAL
        cmd = [sys.executable, str(RUNNER), str(STAGE), 'benchmark', '--source-root', str(source),
               '--model-dir', str(MODEL), '--bundle-root', str(inputs), '--output-dir', str(local_output)]
        receipt = {'source': str(source), 'argv': cmd, 'disk': 'NFS' if args.nfs else 'local_ext4',
                   'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
        started = time.perf_counter()
        print('START ' + label, flush=True)
        with (ROOT / (label + '.console.log')).open('x') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(exit_code=result.returncode, wall_seconds=time.perf_counter() - started,
                       ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (ROOT / (label + '.execution.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'END {label} exit={result.returncode} seconds={receipt["wall_seconds"]:.1f}', flush=True)
        if local_output.exists() and not args.nfs:
            shutil.copytree(local_output, ROOT / label)
        stage_receipt = local_output.with_name(local_output.name + '.receipt.json')
        if stage_receipt.exists() and not args.nfs:
            shutil.copyfile(stage_receipt, ROOT / stage_receipt.name)
        if result.returncode:
            return result.returncode
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
