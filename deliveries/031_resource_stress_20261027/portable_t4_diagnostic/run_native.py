"""CPU metadata preview by default; --execute enables native CUDA timing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['all', 'kd512', 'kd768',
                        'equal512_768', 'two768_timing_only'], default='all')
    parser.add_argument('--output-base', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)  # selection.manifest_path is intentionally relative to bundle root.
    for key in list(os.environ):
        if key.startswith('CUDA_MPS_'):
            os.environ.pop(key)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    for item in inventory['files']:
        path = ROOT / item['path']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']
                or digest(path) != item['sha256']):
            raise ValueError('Bundle file is missing or changed: ' + item['path'])
    from benchmark_local_resources import run_benchmark
    from deploy_kd import read_json, validate_deployment_config
    names = list(inventory['candidates']) if args.candidate == 'all' else [args.candidate]
    output = args.output_base or Path('reports') / ('native_' + datetime.now(
        timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    reports = {}
    for name in names:
        model_dir = ROOT / inventory['candidates'][name]['model_dir']
        validate_deployment_config(read_json(model_dir / 'config.json'))
        report = run_benchmark(model_dir=model_dir,
            pictures_100=ROOT/'inputs/pictures_100', pictures_5=ROOT/'inputs/pictures_5',
            selection_json=ROOT/'inputs/selection.json', output_dir=output/name,
            profile='actual_t4_diagnostic', requested_thread_percentage=None,
            memory_limit_gib=14, execute=args.execute)
        assert report['T4_latency_certified'] is False
        assert report['official_compliance_pass'] is False
        reports[name] = report
    if not args.execute and 'torch' in sys.modules:
        raise AssertionError('Metadata preview unexpectedly imported torch')
    print(json.dumps({'bundle_hashes_verified': True, 'executed': args.execute,
        'actual_hardware_must_be_read_from_execution_reports': True,
        'T4_latency_certified': False, 'official_compliance_pass': False,
        'candidates': reports}, indent=2, allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
