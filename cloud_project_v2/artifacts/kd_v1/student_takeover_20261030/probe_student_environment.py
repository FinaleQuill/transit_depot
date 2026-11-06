"""Import and CUDA arithmetic checks only; never constructs a model."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

expected = {
    'torch': '2.10.0', 'torchvision': '0.25.0', 'timm': '1.0.30',
    'numpy': '1.26.4', 'Pillow': '10.4.0',
    'opencv-python-headless': '4.11.0.86', 'albumentations': '2.0.8',
    'scikit-learn': '1.6.1', 'scipy': '1.15.3', 'safetensors': '0.8.0',
}
report = {
    'time_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
    'python': sys.version, 'packages': {}, 'modules': {},
    'model_constructed': False, 'model_forward_executed': False,
    'training_executed': False, 'test_prediction_executed': False,
}
assert sys.version_info[:2] == (3, 12)
shared_memory = os.statvfs('/dev/shm')
report['shared_memory_bytes'] = shared_memory.f_frsize * shared_memory.f_blocks
assert report['shared_memory_bytes'] >= 4 * 1024 ** 3
report['read_only_mounts'] = {}
for protected in (
    '/home/featurize/work/runs', '/home/featurize/work/deliveries',
    '/home/featurize/work/deliveries/038_dual768_optimized_project_20261030',
    '/home/featurize/work/cloud_project_v2/data', '/home/featurize/data',
):
    readonly = bool(os.statvfs(protected).f_flag & os.ST_RDONLY)
    assert readonly, protected
    report['read_only_mounts'][protected] = readonly
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    assert actual.split('+')[0] == version, (name, actual, version)
    report['packages'][name] = actual
for name in ('torch', 'torchvision', 'timm', 'numpy', 'PIL', 'cv2',
             'albumentations', 'scipy', 'sklearn', 'safetensors',
             'train_distill', 'distill_tta_data', 'fast_lanczos'):
    module = importlib.import_module(name)
    report['modules'][name] = {'file': module.__file__, 'import': 'ok'}

import torch
import fast_lanczos
assert torch.cuda.is_available()
assert torch.version.cuda == '12.8'
properties = torch.cuda.get_device_properties(0)
tensor = torch.tensor([1., 2., 3.], device='cuda', dtype=torch.float16)
assert float((tensor * 2).sum().cpu()) == 12.
torch.cuda.synchronize()
report['cuda'] = {
    'build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
    'device': properties.name, 'capability': [properties.major, properties.minor],
    'memory_bytes': properties.total_memory, 'sm_count': properties.multi_processor_count,
    'fp16_arithmetic_check': 'passed', 'recommended_training_dtype': 'float16',
    'hardware_native_bf16': properties.major >= 8,
}
report['native'] = {
    'available': fast_lanczos.available(),
    'implementation': fast_lanczos.implementation(),
    'max_threads': fast_lanczos.default_threads(),
}
assert report['native']['available']
for name in ('torch', 'torchvision', 'numpy', 'PIL'):
    assert 'addons' not in report['modules'][name]['file']
for name in ('cv2', 'albumentations', 'scipy', 'sklearn'):
    assert report['modules'][name]['file'].startswith('/opt/student_training_addons/')

out = Path(__file__).with_name('student_environment_probe.json')
with out.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report, indent=2))
