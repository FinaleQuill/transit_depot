"""Build a CPU-only portable diagnostic bundle, preserving all original bytes.

Run only after the current GPU timing job has finished. Preview is the default;
--execute copies and hashes data but never initializes a model or CUDA.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT.parent / 'deliveries/031_resource_stress_20261027'
DEST = BASE / 'portable_t4_diagnostic'
SOURCE_NAMES = ['deploy_kd.py', 'model_overnight.py', 'runtime_compat.py',
                'kd_common.py', 'predict.py', 'cost.py',
                'benchmark_local_resources.py', 'prepare_latency_inputs.py',
                'run_resource_stress.py', 'requirements_inference.txt']
CANDIDATES = {
    'kd512': ROOT.parent / 'deliveries/027_tiny512_kd_seed43/fp16_hflip2',
    'kd768': ROOT.parent / 'deliveries/029_tiny768_kd_seed43/fp16_hflip2',
    'equal512_768': BASE / 'models/kd512_kd768_equal_hflip2',
    'two768_timing_only': BASE / 'models/two768_E8_E13_timing_only',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + '\n').encode('utf-8')


LAUNCHER = r'''"""CPU metadata preview by default; --execute enables native CUDA timing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['all', 'kd512', 'kd768',
                        'equal512_768', 'two768_timing_only'], default='all')
    parser.add_argument('--output-base', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)  # selection.manifest_path is intentionally relative to bundle root.
    for key in list(os.environ):
        if key.startswith('CUDA_MPS_'):
            os.environ.pop(key)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    for item in inventory['files']:
        path = ROOT / item['path']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']
                or digest(path) != item['sha256']):
            raise ValueError('Bundle file is missing or changed: ' + item['path'])
    from benchmark_local_resources import run_benchmark
    from deploy_kd import read_json, validate_deployment_config
    names = list(inventory['candidates']) if args.candidate == 'all' else [args.candidate]
    output = args.output_base or Path('reports') / ('native_' + datetime.now(
        timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    reports = {}
    for name in names:
        model_dir = ROOT / inventory['candidates'][name]['model_dir']
        validate_deployment_config(read_json(model_dir / 'config.json'))
        report = run_benchmark(model_dir=model_dir,
            pictures_100=ROOT/'inputs/pictures_100', pictures_5=ROOT/'inputs/pictures_5',
            selection_json=ROOT/'inputs/selection.json', output_dir=output/name,
            profile='actual_t4_diagnostic', requested_thread_percentage=None,
            memory_limit_gib=14, execute=args.execute)
        assert report['T4_latency_certified'] is False
        assert report['official_compliance_pass'] is False
        reports[name] = report
    if not args.execute and 'torch' in sys.modules:
        raise AssertionError('Metadata preview unexpectedly imported torch')
    print(json.dumps({'bundle_hashes_verified': True, 'executed': args.execute,
        'actual_hardware_must_be_read_from_execution_reports': True,
        'T4_latency_certified': False, 'official_compliance_pass': False,
        'candidates': reports}, indent=2, allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
'''


README = '''# 可迁移 T4 诊断材料

这是四个候选的本地计时材料合集，不是最终提交包。当前没有真实 T4 验收结果。
本包不包含训练集、教师权重或训练断点；config 中的教师/预训练来源文字是核验材料，
旧机器绝对路径仅作来源证据，推理不读取旧机器路径或下载原始预训练权重。

四候选：kd512（027 E7）、kd768（029 E8）、equal512_768（两者固定等权），
two768_timing_only（同一次 029 训练的 E8/E13，仅诊断两个 768 模型的计时开销）。
最后一项不是两个独立 seed/fold，不作精度或集成收益声明。各候选保持原始 FP16
权重、配置及导出/组合报告字节不变。单独候选的实际调用权重和必要配置均小于
500,000,000 bytes；合集总大小不能当成最终提交模型大小。

输入来自原始 valid：100 张、53 个完整 Study、40 位患者；另 5 张预热、4 个完整
Study、4 位患者。两组患者不相交。所有 105 张都是独立文件拷贝，未重编码。
valid_manifest.csv 是原始清单逐字节副本；inputs/selection.original.json 保留原选样记录。
inputs/selection.json 只将 manifest_path 改为 valid_manifest.csv，并增加
original_selection_sha256 / original_manifest_path 两项来源证据；身份、原图 SHA、
Study 完整性及其它原字段保留。output_relative_path 仍相对于 inputs/。

请先切换到本包根目录。以下默认仅 CPU 元数据核验，隐藏 CUDA 亦可运行：

```sh
CUDA_VISIBLE_DEVICES='' python run_native.py
```

依赖见 requirements_inference.txt。在指定 T4 环境使用已有兼容 CUDA/PyTorch；
不要用训练环境 wheel 替换官方环境。这里不自动安装依赖。明确执行时，每个命令
使用全新报告目录；启动器清除 CUDA_MPS_*，不设置 MPS 或线程百分比，使用
14 GiB 的 PyTorch allocator 上限（它不模拟带宽/架构，也不限制整个进程显存）：

```sh
python run_native.py --candidate kd512 --output-base reports/t4_kd512 --execute
python run_native.py --candidate kd768 --output-base reports/t4_kd768 --execute
python run_native.py --candidate equal512_768 --output-base reports/t4_equal --execute
python run_native.py --candidate two768_timing_only --output-base reports/t4_two768 --execute
```

启动器先核验 inventory.json 所列文件 SHA/大小，再校验部署配置和完整 Study 输入。
也可直接使用原包装器（必须在本包根目录运行；增加 --execute 才会启用 CUDA）：

```sh
python benchmark_local_resources.py --model-dir models/kd512 --pictures-100 inputs/pictures_100 --pictures-5 inputs/pictures_5 --selection-json inputs/selection.json --output-dir reports/direct_kd512 --profile actual_t4_diagnostic --memory-limit-gib 14
```

协议为 FP16、batch=1 串行、固定 hflip2 TTA、5 图预热、100 图连续 3 轮，
读图到预处理/前向/汇总/实际 CSV 写出计时。没有 DALI、CLAHE、跨图预取或预测缓存；
仅在同一张图内共享解码/相同尺寸预处理及 TTA 输入。权重/模型初始化不计时。
原生设备名称、SM 数、环境和 mean_ms 等由执行报告保存，必须核对实际硬件。
profile 名称本身不能证明使用 T4。包装器始终保留 T4_latency_certified=false、
official_compliance_pass=false；真实 T4 上的这些 valid 诊断也不等于官方选定
100 张 test 的验收。4090/MPS/allocator 上限结果都不能换算或宣称 T4 已通过。

最终仍须在官方指定 T4/输入/环境，独立运行 predict.py 与 cost.py，核对完整
submission.csv、全部实际部署成员体积、三轮时延和输出一致性。完整 test 清单和
图片不在本包中，须明确提供 --manifest / --image-root，不能使用旧机器默认路径。
预训练许可证/模型卡另存于 LICENSES/，并保留在原配置；未给本项目源码新增许可证。
inventory.json 的 files 记录本包固定文件及来源；它不列自身（避免自引用哈希），
执行后新生成的 reports/ 也不属于固定原始材料。

diagnostics/ 保留本机既有报告：results_v1 为旧预处理的 MPS 压测，native_controls
为旧原生结果，optimized_native_controls / optimized_native_repeat 为新预处理的
原生结果；其原始绝对路径是来源证据。provenance/deploy_before_fast_inputs.py
保留加速前源码，运行入口始终是本包根目录的最新版 deploy_kd.py。
这些都是本机 4090 诊断，不是真实 T4 统计。新版单 768 首轮完整均值 109.235 ms，
三轮均值 80.024 / 79.693 / 167.990 ms，异常慢轮原样保留；复测均值 77.335 ms。
双 768 E8/E13 计时代理两次均值 79.244 / 79.917 ms，不声明两个独立训练的精度。
CLAHE 与 DALI 均未启用；具体完整精度数字、每轮均值和环境以原始 JSON 为准。
'''


def build(*, execute=False, tar=False):
    selection_path = BASE / 'inputs/selection.json'
    selection = json.loads(selection_path.read_text())
    plan = {'destination': str(DEST), 'candidates': {k:str(v) for k,v in CANDIDATES.items()},
            'input_counts': selection['counts'], 'execute': execute,
            'gpu_or_model_execution': False}
    if not execute:
        print(json.dumps(plan, indent=2)); return
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    records, source_digests = {}, {}

    def copy(source, relative):
        source = Path(source).resolve(strict=True)
        target = DEST / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(source)
        before = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != before or target.stat().st_ino == source.stat().st_ino:
            raise ValueError('Copy mismatch or hardlink: ' + str(source))
        source_digests[str(source)] = before
        records[str(relative)] = {'origin': str(source), 'source_sha256': before,
                                  'source_bytes_unchanged': True}

    def generated(relative, data, origin):
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records[str(relative)] = {'origin': origin, 'derived': True}

    for name in SOURCE_NAMES:
        copy(ROOT/name, name)
    copy(ROOT/'artifacts/kd_v1/resource_stress_20261027/deploy_before_fast_inputs.py',
         'provenance/deploy_before_fast_inputs.py')
    summaries = {'results_v1': 'resource_stress_summary.json',
                 'native_controls': 'native_summary.json',
                 'optimized_native_controls': 'optimized_native_summary.json',
                 'optimized_native_repeat': 'optimized_native_summary.json'}
    for directory, summary in summaries.items():
        parent = BASE/directory
        if (parent/summary).is_file():
            copy(parent/summary, Path('diagnostics')/directory/summary)
        for report in sorted(parent.glob('*/local_resource_report.json')):
            copy(report, Path('diagnostics')/directory/report.relative_to(parent))
    for name, directory in CANDIDATES.items():
        for source in sorted(directory.iterdir()):
            if source.is_file() and (source.suffix == '.pt' or source.name in
                    {'config.json', 'export_report.json', 'composition_report.json'}):
                copy(source, Path('models')/name/source.name)
    manifest = Path(selection['manifest_path'])
    copy(manifest, 'valid_manifest.csv')
    if sha(DEST/'valid_manifest.csv') != selection['manifest_sha256']:
        raise ValueError('Manifest identity mismatch')
    copy(selection_path, 'inputs/selection.original.json')
    for item in selection['measured'] + selection['warmup']:
        relative = Path(item['output_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe image relative path')
        copy(BASE/'inputs'/relative, Path('inputs')/relative)
        if sha(DEST/'inputs'/relative) != item['sha256']:
            raise ValueError('Original image mismatch')
    portable = dict(selection, manifest_path='valid_manifest.csv',
                    original_selection_sha256=sha(selection_path),
                    original_manifest_path=selection['manifest_path'])
    generated('inputs/selection.json', json_bytes(portable),
              'inputs/selection.original.json; only manifest_path and two provenance fields derived')
    generated('run_native.py', LAUNCHER.encode(), str(Path(__file__).resolve())+'#LAUNCHER')
    generated('README_ZH.md', README.encode(), str(Path(__file__).resolve())+'#README')
    material_hashes = set()
    for candidate in CANDIDATES:
        config_path = DEST/'models'/candidate/'config.json'
        config = json.loads(config_path.read_text())
        provenances = [('tiny', config['pretrained_provenance'])]
        provenances += [('base', t['pretrained_provenance']) for t in config.get('teacher_provenance', [])]
        for label, provenance in provenances:
            for kind, material in provenance['materials'].items():
                digest = material['sha256']
                content = material['text'].encode('utf-8')
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError('Embedded license/material mismatch')
                if digest not in material_hashes:
                    generated(f'LICENSES/{label}_{kind}_{digest[:12]}.txt', content,
                              f'models/{candidate}/config.json embedded pretrained_provenance materials.{kind}; sha256={digest}')
                    material_hashes.add(digest)
    sys.path.insert(0, str(DEST))
    from deploy_kd import deployment_inventory, validate_deployment_config
    candidates = {}
    for name in CANDIDATES:
        relative = Path('models')/name
        config = json.loads((DEST/relative/'config.json').read_text())
        validate_deployment_config(config)
        inventory = deployment_inventory(DEST/relative, config)
        if not inventory['size_pass']:
            raise ValueError('Oversize inference candidate')
        candidates[name] = {'model_dir': str(relative), 'inference_files': inventory,
                            'member_count': len(config['members']),
                            'distinct_weights_used': len({m['weights_sha256'] for m in config['members']}),
                            'timing_only': name == 'two768_timing_only',
                            'independent_seed_or_fold_claim': False}
    # Exercise the unmodified wrapper directly with CUDA hidden, from bundle root.
    env = {key:value for key,value in os.environ.items() if not key.startswith('CUDA_MPS_')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name in CANDIDATES:
        command = [sys.executable, '-B', 'benchmark_local_resources.py', '--model-dir',
                   f'models/{name}', '--pictures-100', 'inputs/pictures_100',
                   '--pictures-5', 'inputs/pictures_5', '--selection-json', 'inputs/selection.json',
                   '--output-dir', f'reports/preview_{name}', '--profile', 'actual_t4_diagnostic',
                   '--memory-limit-gib', '14']
        done = subprocess.run(command, cwd=DEST, env=env, text=True, capture_output=True, check=True)
        report = json.loads(done.stdout)
        assert report['executed'] is False and report['T4_latency_certified'] is False
        generated(f'previews/{name}.json', json_bytes(report),
                  {'command': command, 'cuda_visible_devices': '', 'cwd': 'bundle root'})
    if 'torch' in sys.modules:
        raise AssertionError('Builder unexpectedly imported torch')
    for source, before in source_digests.items():
        if sha(source) != before:
            raise ValueError('Original source changed during packaging: ' + source)
    inventory = {'schema_version': 1, 'purpose': 'portable native T4 diagnostic materials',
                 'final_submission': False, 'GPU_used_in_build': False,
                 'source_files_unchanged_verified': True,
                 'original_selection_sha256': sha(selection_path),
                 'original_manifest_sha256': selection['manifest_sha256'],
                 'candidate_size_limit_bytes': 500_000_000, 'candidates': candidates,
                 'files': [dict(path=relative, bytes=(DEST/relative).stat().st_size,
                                sha256=sha(DEST/relative), **origin)
                           for relative, origin in sorted(records.items())]}
    (DEST/'inventory.json').write_bytes(json_bytes(inventory))
    launcher = subprocess.run([sys.executable, '-B', 'run_native.py'], cwd=DEST,
                              env=env, text=True, capture_output=True, check=True)
    validated = json.loads(launcher.stdout)
    assert validated['bundle_hashes_verified'] and not validated['executed']
    result = dict(plan, complete=True, files=len(inventory['files']),
                  bytes=sum(item['bytes'] for item in inventory['files']),
                  inventory_sha256=sha(DEST/'inventory.json'),
                  portable_preview_candidates=list(validated['candidates']),
                  sources_unchanged=True, gpu_used=False)
    if tar and (result['bytes'] <= 700_000_000
                and shutil.disk_usage(DEST).free >= result['bytes'] + 100_000_000):
        archive = DEST.with_suffix('.tar.gz')
        if archive.exists():
            raise FileExistsError(archive)
        with archive.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as bundle:
                for path in sorted(DEST.rglob('*')):
                    if not path.is_file():
                        continue
                    info = bundle.gettarinfo(str(path), arcname=str(Path(DEST.name)/path.relative_to(DEST)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open('rb') as stream:
                        bundle.addfile(info, stream)
        result['archive'] = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}
    elif tar:
        result['archive_skipped'] = 'Bundle over 700MB or insufficient free disk for archive plus 100MB margin'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tar', action='store_true')
    build(**vars(parser.parse_args()))
