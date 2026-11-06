"""CPU-only guards and controller-cleanup checks; no CUDA or MPS is started."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_resource_stress as stress


def inputs(tmp_path):
    model, data = tmp_path/'model', tmp_path/'inputs'
    model.mkdir(); data.mkdir()
    (model/'config.json').write_text('{}')
    (model/'fake.pt').write_bytes(b'not loaded')
    (data/'selection.json').write_text('{}')
    return dict(model_dirs=[model],names=['candidate'],inputs=data,
                output=tmp_path/'results',percentages=[100,20])


def test_preview_never_starts_processes(tmp_path):
    kwargs=inputs(tmp_path)
    with patch.object(stress.shutil,'which',return_value='/mps'), \
            patch.object(stress.subprocess,'run',side_effect=AssertionError('process started')):
        report=stress.run(**kwargs)
    assert report['executed'] is False
    assert not kwargs['output'].exists()


def test_private_controller_stops_after_calibration_failure(tmp_path):
    kwargs=inputs(tmp_path)
    commands=[]
    def fake_run(command,**options):
        commands.append((command,options.get('input')))
        return stress.subprocess.CompletedProcess(command,1 if '-c' in command else 0,
                                                  stdout='',stderr='expected fake failure')
    with patch.object(stress.shutil,'which',return_value='/mps'), \
            patch.object(stress.subprocess,'check_output',return_value=''), \
            patch.object(stress.subprocess,'run',side_effect=fake_run):
        with pytest.raises(stress.subprocess.CalledProcessError):
            stress.run(**kwargs,execute=True)
    assert commands[-1]==(['/mps'],'quit\n')
    result=json.loads((kwargs['output']/'resource_stress_summary.json').read_text())
    assert result['complete'] is False
    assert result['controller_stop']['returncode']==0
    assert (kwargs['model_dirs'][0]/'fake.pt').read_bytes()==b'not loaded'


def test_rejects_overwrite_and_overlap(tmp_path):
    kwargs=inputs(tmp_path)
    with patch.object(stress.shutil,'which',return_value='/mps'):
        with pytest.raises(ValueError,match='overlap'):
            stress.run(**{**kwargs,'output':kwargs['inputs']/'results'})
        kwargs['output'].mkdir()
        with pytest.raises(FileExistsError):
            stress.run(**kwargs)
