"""Pinned official runtime for one sprint; immutable inputs and two output roots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from reconstruction.runtime import IMAGE, ROOT
from reconstruction.runtime_local import ADDON_ROOT, verify_addons


def command(arguments, config, *, gpu=True, name=None):
    verify_addons()
    work = ROOT.parent.resolve()
    output = Path(config['output']).resolve()
    state = Path(config['checkpoint_dir']).resolve()
    data = Path('/home/featurize/data').resolve()
    if not output.is_relative_to(work / 'reconstruction_runs_20261103'):
        raise ValueError('Persistent sprint outputs must be inside the registered research run root')
    if not state.is_relative_to(data) or state == data or state.is_relative_to(Path(config['image_root']).resolve()):
        raise ValueError('Local mutable sprint state must be separate from original images')
    for path in (output, state):
        if not path.is_dir():
            raise FileNotFoundError(f'Create the explicit sprint output directory before launch: {path}')
    mounts = [(work, work, 'ro'), (data, data, 'ro'),
              (output, output, 'rw'), (state, state, 'rw'),
              (ADDON_ROOT / 'site', Path('/opt/cxr_sprint_addons'), 'ro')]
    args = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--shm-size=4g']
    if gpu:
        args += ['--gpus', 'all']
    if name:
        args += ['--name', name]
    args += ['--user', f'{os.getuid()}:{os.getgid()}']
    for source, target, mode in mounts:
        args += ['--volume', f'{source}:{target}:{mode}']
    for key, value in {
        'PYTHONPATH': f'/opt/cxr_sprint_addons:{ROOT}',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_XET': '1',
        'NO_ALBUMENTATIONS_UPDATE': '1', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'TZ': 'Asia/Hong_Kong', 'PYTHONHASHSEED': str(config['seed']),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    }.items():
        args += ['--env', f'{key}={value}']
    return args + ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE, '-u', *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'sprint/plan.json')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('output', 'checkpoint_dir'):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not argv:
        parser.error('Python arguments required after --')
    raise SystemExit(subprocess.call(command(argv, config, gpu=not args.cpu, name=args.name)))


if __name__ == '__main__':
    main()
