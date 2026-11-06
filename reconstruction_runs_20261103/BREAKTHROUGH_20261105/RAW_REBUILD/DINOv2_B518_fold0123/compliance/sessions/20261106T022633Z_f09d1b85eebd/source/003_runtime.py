"""Offline official runtime with original images and manifests mounted read-only."""
from pathlib import Path

from breakthrough.runtime import command as historical_command
from reconstruction.runtime import ROOT


def command(arguments, config, *, gpu=True, name=None):
    output, state = Path(config['output']).resolve(), Path(config['checkpoint_dir']).resolve()
    if not output.is_relative_to(ROOT.parent / 'reconstruction_runs_20261103/BREAKTHROUGH_20261105/RAW_REBUILD'):
        raise ValueError('Raw experiment requires a distinct registered output')
    if not state.is_relative_to(Path('/home/featurize/data/cxr_breakthrough_20261105/state/RAW_REBUILD')):
        raise ValueError('Raw experiment requires a distinct local state directory')
    return historical_command(arguments, config, gpu=gpu, name=name)
