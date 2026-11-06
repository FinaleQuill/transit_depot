"""Run the relocatable A handoff inside its pinned image; host stdlib only."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

IMAGE = 'crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6'
ORIGINAL_PROJECT = '/home/featurize/work/cloud_project_v2'
ORIGINAL_RUN = '/tmp/cloud_student_runs_20261030/039_tiny768_p2_seed44'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--exact-run', type=Path, help='New writable clone made by prepare_exact_resume.py')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent.parent
    output = (args.output_root or bundle / 'experiments').resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not parameters:
        parser.error('Supply Python arguments after --, for example: -- portable_a.py verify')
    if not args.cpu and args.data_root is None:
        parser.error('Training requires --data-root pointing to official cxr_first images')
    command = ['docker', 'run', '--rm', '--pull=never', '--network', 'none', '--shm-size=4g', '-i',
               '--user', f'{os.getuid()}:{os.getgid()}']
    if args.cpu:
        command += ['--runtime=runc']
    elif shutil.which('nvidia-container-runtime'):
        command += ['--runtime=nvidia', '--gpus', 'all']
    else:
        command += ['--runtime=runc']
        for name in ('nvidia0', 'nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools'):
            device = Path('/dev') / name
            if not device.is_char_device():
                parser.error(f'Missing GPU device {device}; configure NVIDIA Container Toolkit')
            command += ['--device', str(device)]
        cuda = Path('/lib/x86_64-linux-gnu/libcuda.so.1').resolve(strict=True)
        version = cuda.name.removeprefix('libcuda.so.')
        for name in ('libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-ptxjitcompiler.so.1',
                     'libnvidia-gpucomp.so.' + version):
            source = Path('/lib/x86_64-linux-gnu') / name
            if not source.is_file():
                parser.error(f'Missing installed driver library {source}')
            command += ['-v', f'{source}:/usr/local/nvidia/lib64/{name}:ro']
    parent_mount = bundle / 'parent_run'
    parent_mode = 'ro'
    if args.exact_run:
        parent_mount = args.exact_run.resolve(strict=True)
        if not (parent_mount / 'clone_receipt.json').is_file():
            parser.error('--exact-run must be a dedicated clone from prepare_exact_resume.py')
        parent_mode = 'rw'
    mounts = [
        (bundle, '/bundle', 'ro'), (bundle / 'project', ORIGINAL_PROJECT, 'ro'),
        (parent_mount, ORIGINAL_RUN, parent_mode),
        (bundle / 'oof', '/home/featurize/work/runs/024_convnext_v1_base768_mixed20_12_cv5/oof_logits_tta_v2', 'ro'),
        (bundle / 'provenance/original_tiny.json', '/home/featurize/work/runs/030_tiny768_kd_seed44/pretrained_provenance.json', 'ro'),
        (bundle / 'pretrain/model.safetensors', '/home/featurize/work/deliveries/038_dual768_optimized_project_20261030/pretrain_model/timm__convnext_tiny.fb_in22k_ft_in1k_384_a5cbe10cfa2e/model.safetensors', 'ro'),
        (bundle / 'environment/training_addons', '/opt/a_training_addons', 'ro'),
        (bundle / 'environment/inference_addons', '/opt/a_inference_addons', 'ro'),
        (output, '/experiments', 'rw'),
    ]
    if args.data_root:
        mounts.append((args.data_root.resolve(strict=True), '/home/featurize/data/cxr_first', 'ro'))
    for source, target, mode in mounts:
        if not source.exists():
            parser.error(f'Bundle asset missing: {source}')
        command += ['-v', f'{source}:{target}:{mode}']
    for name, value in {
        'PYTHONPATH': '/opt/a_training_addons:/opt/a_inference_addons',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '44',
        'HF_HUB_OFFLINE': '1', 'NO_ALBUMENTATIONS_UPDATE': '1', 'TZ': 'Asia/Hong_Kong',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    }.items():
        command += ['-e', f'{name}={value}']
    command += ['-w', ORIGINAL_PROJECT, '--entrypoint', 'python', IMAGE, '-B', *parameters]
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
