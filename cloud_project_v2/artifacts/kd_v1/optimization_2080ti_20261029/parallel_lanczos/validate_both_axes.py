"""CPU-only exact-pixel test and nonformal timing for the artifact prototype."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from parallel_lanczos_both_axes import (load_image_arrays, resize_single_image,
                                        image_array, letterbox)

PROJECT = Path(__file__).resolve().parents[4]
BUNDLE = PROJECT.parent / 'deliveries/034_dual768_proxy_timing_20261029/bundle'
spec = importlib.util.spec_from_file_location('frozen_033_deploy', BUNDLE / 'deploy_kd.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def fixture(mode, shape):
    height, width = shape
    y, x = np.indices(shape)
    gray = ((x * 79 + y * 53 + (x * y) % 251) % 256).astype(np.uint8)
    rgb = np.stack([gray, np.roll(gray, 1, 0), 255 - gray], axis=-1)
    if mode == 'L':
        return Image.fromarray(gray)
    if mode == 'RGB':
        return Image.fromarray(rgb)
    if mode == '1':
        return Image.fromarray(gray).convert('1', dither=Image.Dither.NONE)
    if mode == 'P':
        image = Image.fromarray(gray).convert('P')
        image.putpalette([value for i in range(256) for value in (i, 255-i, i//2)])
        image.info['transparency'] = 17
        return image
    if mode == 'LA':
        return Image.fromarray(np.stack([gray, 255-gray], axis=-1), 'LA')
    return Image.fromarray(np.concatenate([rgb, gray[..., None]], axis=-1), 'RGBA')


def milliseconds(fn):
    started = time.perf_counter()
    value = fn()
    return (time.perf_counter()-started)*1000, value


def main():
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    checks = {'direct_resize_cases': 0, 'mode_geometry_exif_size_cases': 0,
              'real_image_size_cases': 0, 'non8bit_rejections': 0}
    for mode in ('L', 'RGB'):
        rng = np.random.default_rng(43)
        for width, height in ((1,1), (1,17), (17,1), (37,61), (503,719), (2049,1701)):
            shape = (height, width) + (() if mode == 'L' else (3,))
            image = Image.fromarray(rng.integers(0,256,size=shape,dtype=np.uint8))
            for size in ((1,1), (11,17), (3,5), (768,512), (width,height), (width,5), (5,height)):
                expected = image.resize(size, Image.Resampling.LANCZOS)
                for workers in (2,4):
                    actual = resize_single_image(image, size, workers=workers, min_pixels=0)
                    assert actual.mode == expected.mode and actual.size == expected.size
                    assert actual.tobytes() == expected.tobytes()
                    checks['direct_resize_cases'] += 1
    with tempfile.TemporaryDirectory(prefix='single-image-lanczos-tests-') as temporary:
        root = Path(temporary)
        for mode in ('L','RGB','P','1','LA','RGBA'):
            for shape in ((1,1),(1,1027),(1027,1),(37,61),(519,1031)):
                for orientation in range(1,9):
                    image = fixture(mode,shape)
                    exif = Image.Exif()
                    exif[274] = orientation
                    path = root/'fixture.png'
                    image.save(path,exif=exif)
                    expected = frozen.load_image_arrays(path,[512,768])
                    actual = load_image_arrays(path,[512,768],workers=4,min_pixels=0)
                    for size in (512,768):
                        assert actual[size].tobytes() == expected[size].tobytes()
                        checks['mode_geometry_exif_size_cases'] += 1
                    with Image.open(path) as source:
                        source.load(); ImageOps.exif_transpose(source,in_place=True)
                        converted = source if source.mode == 'L' else source.convert('RGB')
                        for size in (512,768):
                            gray_or_rgb = letterbox(converted,size,workers=4,min_pixels=0)
                            reference = frozen.letterbox_rgb(converted,size)
                            assert gray_or_rgb.convert('RGB').tobytes() == reference.tobytes()
        for mode,dtype in (('I',np.int32),('F',np.float32),('I;16',np.uint16)):
            path=root/'fixture.tiff'
            Image.fromarray(np.arange(35,dtype=dtype).reshape(5,7),mode).save(path)
            try: load_image_arrays(path,[768])
            except ValueError: checks['non8bit_rejections'] += 1
            else: raise AssertionError('non8bit image was accepted')
    paths = sorted((BUNDLE/'inputs/pictures_100').rglob('*.jpg')) + sorted(
        (BUNDLE/'inputs/pictures_5').rglob('*.jpg'))
    for path in paths:
        before = (hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_mtime_ns)
        expected = frozen.load_image_arrays(path,[512,768])
        actual = load_image_arrays(path,[512,768],workers=4,min_pixels=0)
        for size in (512,768):
            assert actual[size].tobytes() == expected[size].tobytes()
            checks['real_image_size_cases'] += 1
        assert before == (hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_mtime_ns)
    print('All byte-parity tests passed:',checks,flush=True)
    rows=[]
    for index,path in enumerate(paths[:8]):
        with Image.open(path) as source:
            source.load();ImageOps.exif_transpose(source,in_place=True)
            image=source if source.mode=='L' else source.convert('RGB')
            scale=768/max(image.size)
            size=tuple(max(1,min(768,round(length*scale))) for length in image.size)
            methods = {
                'original_2d':lambda:image.resize(size,Image.Resampling.LANCZOS),
                'parallel_2':lambda:resize_single_image(image,size,workers=2,min_pixels=0),
                'parallel_4':lambda:resize_single_image(image,size,workers=4,min_pixels=0),
            }
            samples={name:[]for name in methods}
            for repeat in range(3):
                names=list(methods)
                if (index+repeat)%2:names.reverse()
                for name in names:
                    elapsed,actual=milliseconds(methods[name]);samples[name].append(elapsed)
                    assert actual.tobytes()==image.resize(size,Image.Resampling.LANCZOS).tobytes()
            row={'sample_index':index,'source_mode':image.mode,'source_size':list(image.size),
                 'target_size':list(size),'median_resize_ms':{key:statistics.median(value)for key,value in samples.items()},
                 'all_resize_ms':samples}
        row['original_load768_ms'],expected=milliseconds(lambda:frozen.load_image_arrays(path,[768]))
        row['parallel4_load768_ms'],actual=milliseconds(lambda:load_image_arrays(path,[768],workers=4,min_pixels=1_000_000))
        assert expected[768].tobytes()==actual[768].tobytes()
        rows.append(row)
    result={
        'started_utc':started,'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'CPU-only prototype; concurrent fullvalid GPU job and host activity; timings are diagnostic,not formal benchmark',
        'production_source_changed':False,'gpu_or_training_executed':False,
        'python':platform.python_version(),'numpy':np.__version__,'Pillow':pillow_version,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'max_executor_workers_per_image':4,
        'pool_lifetime':'one decoded source image; all futures joined before returning; no cross-image work/prefetch',
        'checks':checks,'all_byte_parity_checks_pass':True,
        'real_images':len(paths),'real_sizes':[512,768],
        'reference':'Frozen033 deploy_kd.py and original Pillow2D resize',
        'prototype_sha256':hashlib.sha256(Path(__file__).with_name('parallel_lanczos_both_axes.py').read_bytes()).hexdigest(),
        'samples':rows,
        'mean_sample_median_resize_ms':{key:statistics.mean(row['median_resize_ms'][key]for row in rows)for key in ('original_2d','parallel_2','parallel_4')},
        'mean_sample_load768_ms':{key:statistics.mean(row[key]for row in rows)for key in ('original_load768_ms','parallel4_load768_ms')},
        'source_references':['https://github.com/python-pillow/Pillow/blob/10.4.0/src/libImaging/Resample.c#L604-L653'],
        'deployment_status':'not_deployed;requires_root_review_and_isolated_real_gpu_benchmark/fullvalid_byte_parity_before_any_acceptance',
    }
    out=Path(__file__).with_name('both_axes_report.json')
    with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({key:value for key,value in result.items()if key!='samples'},indent=2),flush=True)


if __name__=='__main__':main()
