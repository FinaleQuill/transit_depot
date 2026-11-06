"""Pinned offline runtime; originals read-only, new state on local disk."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from reconstruction.runtime import ROOT, IMAGE
from reconstruction.runtime_local import ADDON_ROOT, verify_addons

def command(arguments, config, *, gpu=True, name=None):
    verify_addons()
    output = Path(config['output']).resolve()
    state = Path(config['checkpoint_dir']).resolve()
    assets = Path(config['pretrained_path']).resolve().parent
    if not output.is_relative_to(ROOT.parent / 'reconstruction_runs_20261103/BREAKTHROUGH_20261105'):
        raise ValueError('Unregistered persistent output')
    if not state.is_relative_to(Path('/home/featurize/data/cxr_breakthrough_20261105/state')):
        raise ValueError('Mutable state must be separate from original data')
    for p in (output, state):
        p.mkdir(parents=True, exist_ok=True)
    mounts = [(ROOT.parent, ROOT.parent, 'ro'),
              (Path('/home/featurize/data'), Path('/home/featurize/data'), 'ro'),
              (output, output, 'rw'), (state, state, 'rw'),
              (assets, assets, 'ro'), (ADDON_ROOT / 'site', Path('/opt/cxr_addons'), 'ro')]
    cmd = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--shm-size=4g']
    if gpu:
        cmd += ['--gpus', 'all']
    if name:
        cmd += ['--name', name]
    cmd += ['--user', f'{os.getuid()}:{os.getgid()}']
    for source, target, mode in mounts:
        cmd += ['-v', f'{source}:{target}:{mode}']
    env = dict(PYTHONPATH=f'/opt/cxr_addons:{ROOT}', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TZ='Asia/Hong_Kong',
               PYTHONHASHSEED=str(config['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    return cmd + ['-w', str(ROOT), '--entrypoint', 'python3', IMAGE, '-u', *arguments]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not argv:
        parser.error('Provide Python arguments after --')
    config = json.loads(args.config.read_text())
    raise SystemExit(subprocess.call(command(argv, config, gpu=not args.cpu, name=args.name)))

if __name__ == '__main__':
    main()
