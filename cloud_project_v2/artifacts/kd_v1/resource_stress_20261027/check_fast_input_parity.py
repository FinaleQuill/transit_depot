"""CPU-only paired preprocessing parity/timing on the frozen 105 selected valid JPEGs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import deploy_kd as candidate

ARTIFACT = Path(__file__).resolve().parent
FROZEN = ARTIFACT / 'deploy_before_fast_inputs.py'
EXPECTED_ORIGINAL_SHA256 = 'fd70c10e83a7fc0daa01782d62d0c941f0b2f06b87e59c0b2af89d9a22395849'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(FROZEN) == EXPECTED_ORIGINAL_SHA256
spec = importlib.util.spec_from_file_location('frozen_deploy_before_fast_inputs', FROZEN)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
selection_path = PROJECT.parent / 'deliveries/031_resource_stress_20261027/inputs/selection.json'
selection_bytes = selection_path.read_bytes()
selection = json.loads(selection_bytes)
assert selection['origin_split'] == 'valid' and not selection['test_used']
assert len(selection['measured']) == 100 and len(selection['warmup']) == 5
rows = [('warmup', row) for row in selection['warmup']] + [('measured', row) for row in selection['measured']]
timings = []
for index, (group, row) in enumerate(rows):
    path = selection_path.parent / row['output_relative_path']
    before = (sha(path), path.stat().st_mtime_ns, path.stat().st_size)
    assert before[0] == row['file_sha256']
    for size in (512, 768):
        arrays, elapsed = {}, {}
        # Alternate execution order to reduce consistent filesystem-cache order bias.
        names = ('original', 'candidate') if index % 2 == 0 else ('candidate', 'original')
        for name in names:
            module = original if name == 'original' else candidate
            start = time.perf_counter_ns()
            arrays[name] = module.load_image_arrays(path, [size])[size]
            elapsed[name] = (time.perf_counter_ns() - start) / 1e6
        assert np.array_equal(arrays['original'], arrays['candidate']), (index, size)
        assert arrays['original'].tobytes() == arrays['candidate'].tobytes(), (index, size)
        timings.append({'group': group, 'index': index, 'size': size, 'original_ms': elapsed['original'],
                        'candidate_ms': elapsed['candidate'], 'array_equal': True, 'byte_equal': True})
    assert (sha(path), path.stat().st_mtime_ns, path.stat().st_size) == before
    if (index + 1) % 20 == 0:
        print(f'Completed {index + 1}/105 selected image parity checks at both sizes', flush=True)
assert selection_path.read_bytes() == selection_bytes
summary = {}
for group in ('all_105', 'measured_100', 'warmup_5'):
    summary[group] = {}
    for size in (512, 768):
        subset = [r for r in timings if r['size'] == size and (group == 'all_105' or r['group'] == group.split('_')[0])]
        old = np.asarray([r['original_ms'] for r in subset])
        new = np.asarray([r['candidate_ms'] for r in subset])
        summary[group][str(size)] = {'images': len(subset), 'original_mean_ms': float(old.mean()),
                                    'candidate_mean_ms': float(new.mean()),
                                    'original_median_ms': float(np.median(old)),
                                    'candidate_median_ms': float(np.median(new)),
                                    'speedup_from_means': float(old.mean() / new.mean())}
report = {'status': 'passed', 'comparisons': len(timings), 'images': len(rows), 'sizes': [512, 768],
          'normalized_arrays_byte_identical': True, 'input_files_unchanged': True, 'selection_file_unchanged': True,
          'original_sha256': sha(FROZEN), 'candidate_sha256': sha(PROJECT / 'deploy_kd.py'),
          'selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
          'python': platform.python_version(), 'pillow': pillow_version, 'numpy': np.__version__,
          'cpu_only': True, 'gpu_used': False, 'model_constructed': False, 'labels_read': False, 'test_read': False,
          'timing_scope': 'CPU load_image_arrays(path, [size]) only; paired per image with alternating order; no model, GPU, or CSV writes',
          'timing_is_official_latency': False, 'summary': summary, 'timings': timings}
(ARTIFACT / 'fast_input_parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('status', 'comparisons', 'normalized_arrays_byte_identical', 'summary')}, indent=2))
