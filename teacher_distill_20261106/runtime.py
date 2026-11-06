"""Pinned official image; original images and public initialization read-only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from reconstruction.runtime import IMAGE, ROOT

EXPERIMENT = Path('/home/featurize/data/kd_rebuild_20261106')
EVIDENCE = ROOT.parent / 'reconstruction_runs_20261103/KD_REBUILD_20261106'
ADDONS = EXPERIMENT / 'environment/site'
TRAINING_SOURCES = (
    'teacher_distill_20261106/__init__.py', 'teacher_distill_20261106/train.py',
    'teacher_distill_20261106/model.py', 'teacher_distill_20261106/data.py',
    'teacher_distill_20261106/runtime.py', 'breakthrough/train.py',
    'breakthrough/model.py', 'reconstruction/data.py', 'reconstruction/train.py',
    'reconstruction/model.py', 'training_hooks.py', 'training_compliance.py',
)


def command(arguments, config=None, *, gpu=True, name=None):
    if not ADDONS.is_dir():
        raise FileNotFoundError('Restore the registered lightweight official-image addons first')
    for name_ in ('torch', 'numpy', 'PIL', 'torchvision', 'nvidia', 'triton'):
        if (ADDONS / name_).exists() or (ADDONS / f'{name_}.py').exists():
            raise ValueError(f'An addon shadows the official image core: {name_}')
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    args = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--shm-size=4g']
    if gpu:
        args += ['--gpus', 'all']
    if name:
        args += ['--name', name]
    args += ['--user', f'{os.getuid()}:{os.getgid()}']
    # The read-only data parent is overlaid only by this new experiment's writable tree.
    mounts = [(ROOT.parent, ROOT.parent, 'ro'),
              (Path('/home/featurize/data'), Path('/home/featurize/data'), 'ro'),
              (EXPERIMENT, EXPERIMENT, 'rw'), (EVIDENCE, EVIDENCE, 'rw'),
              (EXPERIMENT / 'assets', EXPERIMENT / 'assets', 'ro'),
              (EXPERIMENT / 'environment', EXPERIMENT / 'environment', 'ro'),
              (ADDONS, Path('/opt/kd_addons'), 'ro')]
    for source, target, mode in mounts:
        args += ['-v', f'{source}:{target}:{mode}']
    for key, value in dict(PYTHONPATH=f'/opt/kd_addons:{ROOT}', PYTHONNOUSERSITE='1',
                          PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                          NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1',
                          MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                          TZ='Asia/Hong_Kong', PYTHONHASHSEED='20261106',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8').items():
        args += ['-e', f'{key}={value}']
    return args + ['-w', str(ROOT), '--entrypoint', 'python3', IMAGE, '-u', *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not argv:
        parser.error('Python arguments required after --')
    raise SystemExit(subprocess.call(command(argv, gpu=not args.cpu, name=args.name)))


if __name__ == '__main__':
    main()
