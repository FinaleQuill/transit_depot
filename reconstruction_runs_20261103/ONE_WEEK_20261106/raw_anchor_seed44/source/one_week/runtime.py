"""Pinned offline student runtime with only this run's output/state writable.

Public initialization, original data, prior models and registered dependencies
are read-only. Constructing a command never launches a process or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from reconstruction.runtime import IMAGE, ROOT

OUTPUT_ROOT = ROOT.parent / 'reconstruction_runs_20261103/ONE_WEEK_20261106'
STATE_ROOT = Path('/home/featurize/data/one_week_20261106/state')
ASSET_ROOT = Path('/home/featurize/data/one_week_20261106/assets')
ENVIRONMENT_ROOT = Path('/home/featurize/data/kd_rebuild_20261106/environment')
ADDONS = ENVIRONMENT_ROOT / 'site'
ENVIRONMENT_MANIFEST = ENVIRONMENT_ROOT / 'manifest.json'
ENVIRONMENT_SHA256 = 'f212ba9c9bf2b0d23e6e62efca75df8281983805e254d01e2049e588ee1fc42e'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_environment():
    """Verify the durable addon inventory without shadowing image core modules."""
    if sha256_file(ENVIRONMENT_MANIFEST) != ENVIRONMENT_SHA256:
        raise ValueError('The registered offline environment manifest changed')
    manifest = json.loads(ENVIRONMENT_MANIFEST.read_text())
    if (manifest['schema'] != 'cxr_kd_rebuild_runtime_addons_v1'
            or manifest['base_image'] != IMAGE):
        raise ValueError('The offline addons belong to another registered image')
    for core in ('torch', 'numpy', 'PIL', 'torchvision', 'nvidia', 'triton'):
        if (ADDONS / core).exists() or (ADDONS / f'{core}.py').exists():
            raise ValueError(f'Addon shadows the official image core: {core}')
    registered = manifest['files']
    observed = {str(path.relative_to(ADDONS)) for path in ADDONS.rglob('*') if path.is_file()}
    if observed != set(registered):
        raise ValueError('Offline addons contain missing or unregistered files')
    resolved = ADDONS.resolve(strict=True)
    for relative, expected in registered.items():
        path = ADDONS / relative
        if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved)
                or path.stat().st_size != expected['bytes']
                or sha256_file(path) != expected['sha256']):
            raise ValueError(f'Offline addon changed: {relative}')
    return dict(manifest=str(ENVIRONMENT_MANIFEST), sha256=ENVIRONMENT_SHA256,
                base_image=IMAGE, packages=manifest['packages'], files=len(registered))


def _registered_path(value, root, description):
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'{description} must remain under {root}')
    return path


def command(arguments, config, *, gpu=True, name=None):
    """Return an argv list for the pinned, disconnected official Docker image."""
    verify_environment()
    if not arguments or not isinstance(arguments, (list, tuple)):
        raise ValueError('Provide a nonempty list of Python arguments')
    output = _registered_path(config['output'], OUTPUT_ROOT, 'Persistent output')
    state = _registered_path(config['checkpoint_dir'], STATE_ROOT, 'Checkpoint state')
    pretrained = _registered_path(config['pretrained_path'], ASSET_ROOT, 'Public initialization')
    if not pretrained.is_file():
        raise FileNotFoundError(f'Public initialization is absent: {pretrained}')
    seed = config['seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for destination in (output, state):
        destination.mkdir(parents=True, exist_ok=True)
    mounts = [
        (ROOT.parent, ROOT.parent, 'ro'),
        (Path('/home/featurize/data'), Path('/home/featurize/data'), 'ro'),
        (output, output, 'rw'),
        (state, state, 'rw'),
        (ASSET_ROOT, ASSET_ROOT, 'ro'),
        (ADDONS, Path('/opt/one_week_addons'), 'ro'),
    ]
    for source, _, _ in mounts:
        if not source.is_dir():
            raise FileNotFoundError(f'Required runtime directory is absent: {source}')
    args = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--shm-size=4g']
    if gpu:
        args += ['--gpus', 'all']
    if name:
        args += ['--name', name]
    args += ['--user', f'{os.getuid()}:{os.getgid()}']
    for source, target, mode in mounts:
        args += ['--volume', f'{source}:{target}:{mode}']
    for key, value in dict(
        PYTHONPATH=f'/opt/one_week_addons:{ROOT}', PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_XET='1',
        NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TZ='Asia/Hong_Kong', PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
    ).items():
        args += ['--env', f'{key}={value}']
    return args + ['--workdir', str(ROOT), '--entrypoint', 'python3', IMAGE, '-u', *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    argv = options.arguments[1:] if options.arguments[:1] == ['--'] else options.arguments
    if not argv:
        parser.error('Provide Python arguments after --')
    config = json.loads(options.config.read_text())
    raise SystemExit(subprocess.call(command(argv, config, gpu=not options.cpu, name=options.name)))


if __name__ == '__main__':
    main()
