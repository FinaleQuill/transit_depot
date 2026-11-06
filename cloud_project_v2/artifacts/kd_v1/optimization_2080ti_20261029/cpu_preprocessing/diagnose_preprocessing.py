"""CPU-only, non-certifying stage diagnostic and byte parity against frozen 033."""
from pathlib import Path
import importlib.util
import json
import platform
import statistics
import sys
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

PROJECT = Path(__file__).resolve().parents[4]
BUNDLE = PROJECT.parent / 'deliveries/034_dual768_proxy_timing_20261029/bundle'
sys.path.insert(0, str(PROJECT))
import deploy_kd as candidate
spec = importlib.util.spec_from_file_location('frozen_033_deploy', BUNDLE / 'deploy_kd.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


def stages(path):
    values = {}
    start = time.perf_counter()
    with Image.open(path) as source:
        values['read_header_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        source.load()
        values['decode_ms'] = elapsed_ms(start)
        source_mode, source_size = source.mode, source.size
        start = time.perf_counter()
        ImageOps.exif_transpose(source, in_place=True)
        image = source if source.mode == 'L' else source.convert('RGB')
        values['exif_and_color_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        canvas = frozen.letterbox_rgb(image, 768)
        values['original_lanczos_and_rgb_canvas_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        pixels = np.asarray(canvas, dtype=np.float32)
        values['original_rgb_float32_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        pixels = (pixels - np.asarray(frozen.MEAN, dtype=np.float32) * np.float32(255)) / (
            np.asarray(frozen.STD, dtype=np.float32) * np.float32(255))
        values['original_normalize_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        expected = np.ascontiguousarray(pixels.transpose(2, 0, 1))
        values['original_chw_copy_ms'] = elapsed_ms(start)
        start = time.perf_counter()
        actual = candidate.image_array(image, 768)
        values['candidate_resized_and_normalized_ms'] = elapsed_ms(start)
        assert actual.tobytes() == expected.tobytes()
    return {'source_mode': source_mode, 'source_size': list(source_size), **values}


def main():
    paths = sorted((BUNDLE / 'inputs/pictures_100').rglob('*.jpg'))[:8]
    samples = []
    for i, path in enumerate(paths):
        row = {'sample_index': i, **stages(path)}
        for name, module in [('original', frozen), ('candidate', candidate)]:
            start = time.perf_counter()
            arrays = module.load_image_arrays(path, [512, 768])
            row[name + '_two_sizes_total_ms'] = elapsed_ms(start)
            if name == 'original':
                expected = arrays
            else:
                assert all(arrays[size].tobytes() == expected[size].tobytes() for size in expected)
        samples.append(row)
    all_paths = sorted((BUNDLE / 'inputs/pictures_100').rglob('*.jpg')) + sorted(
        (BUNDLE / 'inputs/pictures_5').rglob('*.jpg'))
    parity = []
    for index, path in enumerate(all_paths):
        before = path.stat()
        expected = frozen.load_image_arrays(path, [512, 768])
        actual = candidate.load_image_arrays(path, [512, 768])
        assert all(actual[size].tobytes() == expected[size].tobytes() for size in expected)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        parity.append(index)
    result = {
        'scope': 'CPU-only diagnostic; concurrent dataset extraction/host load may affect timings; not a formal benchmark',
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version,
        'sample_selection': 'first 8 sorted timing-image paths, chosen independently of timings or predictions',
        'sizes': [512, 768], 'samples': samples,
        'mean_stage_ms': {key: statistics.mean(row[key] for row in samples)
                          for key in samples[0] if key.endswith('_ms')},
        'all_105_images_two_size_float32_byte_parity': len(parity) == 105,
        'image_count': len(parity), 'inference_or_training_executed': False,
    }
    out = Path(__file__).with_name('cpu_preprocessing_report.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
