"""Run research code in the pinned official 3.0 image, with read-only inputs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'cloud_project_v2'
OUTPUT = ROOT.parent / 'reconstruction_runs_20261103'
PRETRAIN = Path('/tmp/cloud_student_runs_20261030/tight_crop_ab_assets/038_dual768_optimized_project_20261030/pretrain_model')
IMAGE = 'crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:125d8bcb4ce5a2a9acec6d9838b37e2ae1e2014c09b848700f9530f2f7856a0b'


def command(arguments, *, gpu=True, name=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    args = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--shm-size=4g']
    if gpu:
        args += ['--gpus', 'all']
    if name:
        args += ['--name', name]
    args += ['--user', f'{os.getuid()}:{os.getgid()}']
    for source, target, mode in [
        (ROOT, ROOT, 'ro'), (ORIGINAL, ORIGINAL, 'ro'),
        (OUTPUT, OUTPUT, 'rw'), (OUTPUT / 'assets', OUTPUT / 'assets', 'ro'),
        (OUTPUT / 'splits', OUTPUT / 'splits', 'ro'),
        ('/home/featurize/data', '/home/featurize/data', 'ro'),
        (PRETRAIN, PRETRAIN, 'ro'),
        ('/tmp/cxr_train_addons_20261103', '/opt/cxr_train_addons', 'ro'),
    ]:
        args += ['--volume', f'{source}:{target}:{mode}']
    for key, value in {
        'PYTHONPATH': f'/opt/cxr_train_addons:{ROOT}', 'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
        'NO_ALBUMENTATIONS_UPDATE': '1', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'TZ': 'Asia/Hong_Kong', 'PYTHONHASHSEED': '20261103',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    }.items():
        args += ['--env', f'{key}={value}']
    return args + ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE, '-u', *arguments]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    argv = ns.arguments[1:] if ns.arguments[:1] == ['--'] else ns.arguments
    if not argv:
        parser.error('Python arguments required after --')
    raise SystemExit(subprocess.call(command(argv, gpu=not ns.cpu, name=ns.name)))
