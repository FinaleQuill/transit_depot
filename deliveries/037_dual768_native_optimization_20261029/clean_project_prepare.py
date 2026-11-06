"""Rebuild an isolated submitted-code layout with no external addons path."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/featurize/work/cloud_project_v2')
PROJECT = Path('/tmp/dual768_2080ti_diagnosis/submission_runtime_smoke')

def main():
    code = PROJECT / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for path in REPO.glob('*.py'):
        shutil.copyfile(path, code / path.name)
    shutil.copyfile(REPO / 'requirements_inference.txt', code / 'requirements_inference.txt')
    shutil.copytree(REPO / 'native_lanczos', code / 'native_lanczos',
                    ignore=shutil.ignore_patterns('*.so', '__pycache__', 'build', 'dist'))
    environment = REPO / 'artifacts/kd_v1/optimization_2080ti_20261029/environment'
    (code / 'vendor').mkdir()
    for name in ['competition_addons_cp312_linux_x86_64.tar.gz', 'competition_addons_archive_manifest.json']:
        shutil.copyfile(environment / name, code / 'vendor' / name)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    receipts = []
    for flag, output in [('--force-build', 'prepared.json'), ('--check-only', 'checked.json')]:
        command = [sys.executable, str(code / 'prepare_inference.py'), flag, '--output', str(PROJECT / output)]
        with (PROJECT / (output + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipts.append({'flag': flag, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('Isolated project preparation failed: ' + output)
    check = '''import json, torch, timm, numpy as np, PIL, fast_lanczos
from PIL import Image
from inference_runtime import activate_dependencies
assert activate_dependencies() is not None
assert '_runtime_addons' in timm.__file__
image=Image.fromarray(np.random.default_rng(43).integers(0,256,(511,739),dtype=np.uint8))
assert fast_lanczos.resize(image,(311,257)).tobytes() == image.resize((311,257),Image.Resampling.LANCZOS).tobytes()
print(json.dumps({'native':fast_lanczos.implementation(),'native_available':fast_lanczos.available(),
 'timm_module':timm.__file__,'numpy_module':np.__file__,'torch_module':torch.__file__,
 'cuda_initialized':torch.cuda.is_initialized(),'pixel_bytes_equal':True}))
'''
    # Activation precedes all addon imports in the fresh interpreter.
    check = 'from inference_runtime import activate_dependencies\nactivate_dependencies()\n' + check
    result = subprocess.run([sys.executable, '-c', check], cwd=code, env=env, capture_output=True, text=True, check=True)
    report = {'project_root': str(PROJECT), 'external_addons_environment_removed': True,
              'preparation_runs': receipts, 'probe': json.loads(result.stdout), 'model_inference_executed': False,
              'native_so_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (code / 'native_lanczos').glob('*.so')}}
    (ROOT / 'clean_project_prepare.json').write_text(json.dumps(report, indent=2) + '\n')
    for name in ['prepared.json', 'checked.json', 'prepared.json.log', 'checked.json.log']:
        shutil.copyfile(PROJECT / name, ROOT / ('clean_project_' + name))
    print(json.dumps(report, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
