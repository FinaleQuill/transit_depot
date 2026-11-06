"""Bounded CPU-only comparison of frozen E2 training input loaders.

Run through runtime_local --cpu, after coordinating CPU capacity with training.
Only the registered optimization/train manifest and its selected source images
are opened. No model, optimizer, dev/H/test images, cache mutation, or training.
Tensor equivalence and throughput are separate passes so SHA work cannot hide
loader scaling. The benchmark does not measure pinned transfer or GPU overlap.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from reconstruction.data import StudyDataset, read_records, study_groups
from reconstruction.local_data import LocalStudyDataset
from reconstruction import runtime_local


RUN_ROOT = runtime_local.OUTPUT
PERFORMANCE_ROOT = RUN_ROOT / 'E2_20261104/performance_20261104'
IDENTITY_FIELDS = ('study_key', 'subject_id', 'study_id', 'image_ids', 'views')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def text_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def system_snapshot():
    root = Path('/sys/fs/cgroup')
    names = ('cpu.max', 'cpu.stat', 'cpu.pressure', 'cpuset.cpus.effective',
             'memory.max', 'memory.current', 'memory.events', 'memory.stat',
             'io.stat', 'io.pressure')
    shm = os.statvfs('/dev/shm')
    return {'at': now(), 'cgroup': {name: text_file(root / name) for name in names},
            'meminfo': text_file('/proc/meminfo'), 'affinity': sorted(os.sched_getaffinity(0)),
            'shm_used_bytes': (shm.f_blocks-shm.f_bfree)*shm.f_frsize,
            'shm_total_bytes': shm.f_blocks*shm.f_frsize}


def process_snapshot(pid):
    root = Path('/proc') / str(pid)
    raw = text_file(root / 'stat')
    if raw is None:
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    io = {}
    for line in (text_file(root / 'io') or '').splitlines():
        key, value = line.split(':', 1)
        io[key] = int(value)
    pss = None
    for line in (text_file(root / 'smaps_rollup') or '').splitlines():
        if line.startswith('Pss:'):
            pss = int(line.split()[1]) * 1024
    return {'pid': pid, 'rss_bytes': int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
            'pss_bytes': pss, 'cpu_seconds': (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
            'io': io}


class Monitor:
    """Low-frequency diagnostic sampling; sums of RSS include shared pages."""
    def __init__(self, pids):
        self.pids = pids
        self.stop = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        processes = [p for pid in self.pids if (p := process_snapshot(pid)) is not None]
        self.rows.append({'seconds': time.monotonic()-self.started,
                          'processes': processes, 'system': system_snapshot()})

    def run(self):
        while not self.stop.wait(1.):
            self.sample()

    def __enter__(self):
        self.started = time.monotonic()
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def summary(self):
        rss = [sum(p['rss_bytes'] for p in r['processes']) for r in self.rows]
        pss = [sum(p['pss_bytes'] or 0 for p in r['processes']) for r in self.rows]
        return {'sampling_seconds': 1., 'peak_sampled_process_rss_sum_bytes': max(rss),
                'peak_sampled_process_pss_sum_bytes': max(pss),
                'peak_sampled_shm_used_bytes': max(r['system']['shm_used_bytes'] for r in self.rows),
                'rss_sum_double_counts_shared_pages': True, 'samples': self.rows}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def prepare(config_path, count, selection):
    plan_path = RUN_ROOT / 'E2_20261104/controller/frozen_plan.json'
    plan = json.loads(plan_path.read_text())
    jobs = [j for j in plan['jobs'] if Path(j['config_path']).resolve() == config_path.resolve()]
    if len(jobs) != 1 or digest(config_path) != jobs[0]['config_sha256']:
        raise ValueError('Use one exact frozen formal E2 config')
    config = json.loads(config_path.read_text())
    if config != jobs[0]['config']:
        raise ValueError('Config does not match the frozen queue')
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    train_path = Path(config['train_manifest']).resolve(strict=True)
    split_path = Path(config['split_plan']).resolve(strict=True)
    if train_path != RUN_ROOT / 'splits/train.csv' or split_path != RUN_ROOT / 'splits/split_plan.json':
        raise ValueError('Only registered optimization inputs are permitted')
    for path in (train_path, split_path):
        if digest(path) != jobs[0]['inputs_sha256'][str(path)]:
            raise ValueError('Registered optimization input has changed')
    split = json.loads(split_path.read_text())
    if split['train_folds'] != [1, 2, 3] or digest(train_path) != split['roles']['train']['sha256']:
        raise ValueError('Optimization role binding differs')
    records = read_records(train_path)
    if any(r['split'] != 'train' for r in records):
        raise ValueError('Every input must originate from official train')
    groups = study_groups(records)
    if len(groups) != split['roles']['train']['studies'] or count > len(groups):
        raise ValueError('Invalid optimization coverage or requested subset size')
    # Metadata-only quantile coverage of view count and source pixel workload.
    workload = lambda g: sum(int(records[i]['width'])*int(records[i]['height']) for i in g)
    ranked = sorted(groups, key=lambda g: (len(g), workload(g), records[g[0]]['study_key']))
    if selection == 'highest-view':
        picked = [ranked[-1]]
    elif selection == 'highest-workload':
        picked = ranked[-count:]
    else:
        picked = [ranked[min(len(ranked)-1, int((i+.5)*len(ranked)/count))] for i in range(count)]
    picked.sort(key=lambda g: g[0])
    selected = [records[i] for group in picked for i in group]
    if len({records[g[0]]['study_key'] for g in picked}) != count:
        raise ValueError('Subset selection produced a duplicate Study')
    identity = [{k: row[k] for k in ('image_id', 'study_key', 'relative_path', 'file_sha256')} for row in selected]
    binding = {'config_sha256': digest(config_path), 'frozen_plan_sha256': digest(plan_path),
        'train_manifest_sha256': digest(train_path), 'split_plan_sha256': digest(split_path),
        'selection': ('single highest-view Study; separate mechanics stress' if selection == 'highest-view'
                      else f'highest {count} distinct complete Studies by (view_count,total_source_pixels)'
                      if selection == 'highest-workload'
                      else 'metadata workload quantiles over (view_count,total_source_pixels); no forced rare tail'),
        'selection_is_random_representative_sample': False,
        'train_view_histogram': dict(Counter(map(len, groups))),
        'selected_view_histogram': dict(Counter(map(len, picked))),
        'studies': count, 'images': len(selected), 'selected_images': identity,
        'selected_identity_sha256': hashlib.sha256(encoded(identity)).hexdigest(),
        'source_megapixels': {'min': min(workload(g) for g in picked)/1e6,
                            'median': statistics.median(workload(g) for g in picked)/1e6,
                            'max': max(workload(g) for g in picked)/1e6}}
    return config, selected, binding


def item_digest(item):
    value = hashlib.sha256()
    value.update(encoded({k: item[k] for k in IDENTITY_FIELDS}))
    for key in sorted(k for k, value in item.items() if isinstance(value, torch.Tensor)):
        tensor = item[key].detach().cpu().contiguous()
        value.update(encoded([key, str(tensor.dtype), list(tensor.shape)]))
        value.update(memoryview(tensor.numpy()).cast('B'))
    return value.hexdigest()


def run_pass(config, records, workers, prefetch, args, *, verify, deadline):
    dataset_class = LocalStudyDataset if config['experiment_arm'] == 'global_local' else StudyDataset
    dataset = dataset_class(records, config['image_root'], image_size=config['image_size'],
        training=True, seed=config['seed'], local_tiles=config['use_local'], verify_hash=True)
    dataset.set_epoch(config['augmentation_epoch_offset'] + args.epoch)
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=workers,
        pin_memory=False, generator=torch.Generator().manual_seed(config['seed'] + args.epoch),
        worker_init_fn=seed_worker, prefetch_factor=prefetch, multiprocessing_context='spawn',
        timeout=180, persistent_workers=False, in_order=True)
    start = time.monotonic()
    iterator = iter(loader)
    pids = [os.getpid(), *(p.pid for p in iterator._workers)]
    identities, tensors, arrivals, image_counts = [], [], [], []
    first = None
    steady_start = start if args.warmup_studies == 0 else None
    try:
        with Monitor(pids) as monitor:
            for index in range(len(dataset)):
                if time.monotonic() >= deadline:
                    raise TimeoutError('CPU benchmark budget exhausted; no automatic retry')
                before = time.monotonic()
                item = next(iterator)
                arrived = time.monotonic()
                arrivals.append(arrived-before)
                first = arrived-start if first is None else first
                identities.append({k: item[k] for k in IDENTITY_FIELDS})
                image_counts.append(len(item['image_ids']))
                if verify:
                    tensors.append(item_digest(item))
                if args.consumer_delay_seconds:
                    time.sleep(args.consumer_delay_seconds)
                if index + 1 == args.warmup_studies:
                    steady_start = time.monotonic()
                del item
            ended = time.monotonic()
    finally:
        # Explicit diagnostic cleanup, including exceptions; no lingering workers.
        iterator._shutdown_workers()
        del iterator, loader, dataset
        gc.collect()
    if len({r['study_key'] for r in identities}) != len(identities):
        raise ValueError('Duplicate Study in delivered order')
    expected = Counter(r['image_id'] for r in records)
    if Counter(i for r in identities for i in r['image_ids']) != expected:
        raise ValueError('Incomplete or duplicate image coverage')
    steady_seconds = ended-steady_start
    timed_studies = len(identities)-args.warmup_studies
    timed_images = sum(image_counts[args.warmup_studies:])
    return {'phase': 'tensor_verification' if verify else 'throughput', 'workers': workers,
        'prefetch_factor': prefetch, 'pin_memory': False, 'training_shuffle': True,
        'augmentation_epoch': config['augmentation_epoch_offset']+args.epoch,
        'dataset_class': dataset_class.__name__,
        'consumer_delay_seconds_per_study': args.consumer_delay_seconds,
        'throughput_interpretation': ('delayed-consumer memory stress; not maximum loader throughput'
                                     if args.consumer_delay_seconds else 'fast CPU consumer; no GPU compute'),
        'studies': len(identities), 'images': sum(image_counts), 'first_item_seconds': first,
        'total_seconds_including_spawn': ended-start, 'steady_seconds': steady_seconds,
        'steady_studies': timed_studies, 'steady_images': timed_images,
        'steady_studies_per_second': timed_studies/steady_seconds,
        'steady_images_per_second': timed_images/steady_seconds,
        'next_item_wait_seconds': sum(arrivals),
        'order_sha256': hashlib.sha256(encoded(identities)).hexdigest(),
        'tensor_sha256': hashlib.sha256(encoded(tensors)).hexdigest() if verify else None,
        'tensor_hash_included_in_this_pass_timing': verify,
        'memory_and_process_observations': monitor.summary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--studies', type=int, default=64)
    parser.add_argument('--selection', choices=['representative', 'highest-view', 'highest-workload'], default='representative')
    parser.add_argument('--warmup-studies', type=int, default=8)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, 4, 6])
    parser.add_argument('--prefetch', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--nice', type=int, default=10)
    parser.add_argument('--concurrent-training', action='store_true')
    parser.add_argument('--consumer-delay-seconds', type=float, default=0.)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    valid_subset = ((args.selection in {'representative', 'highest-workload'} and 16 <= args.studies <= 256
                     and 1 <= args.warmup_studies < args.studies//2)
                    or (args.selection == 'highest-view' and args.studies == 1 and args.warmup_studies == 0))
    if (not valid_subset
            or not set(args.workers).issubset({2, 4, 6}) or not args.workers
            or not set(args.prefetch).issubset({1, 2}) or not args.prefetch
            or len(set(args.workers)) != len(args.workers) or len(set(args.prefetch)) != len(args.prefetch)
            or not 1 <= args.epoch <= 6 or not 30 <= args.max_seconds <= 1800
            or not 0 <= args.nice <= 19 or not 0 <= args.consumer_delay_seconds <= 2.):
        parser.error('Use 16..256 representative Studies or one highest-view/no-warmup Study; workers 2/4/6 and prefetch 1/2')
    args.output = args.output.resolve()
    if PERFORMANCE_ROOT.resolve() not in args.output.parents:
        raise ValueError('Evidence output must stay in the isolated E2 performance directory')
    if args.output.exists():
        raise FileExistsError('Existing measurement evidence must not be overwritten')
    if any(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('Run with runtime_local --cpu; this diagnostic must not expose a GPU')
    os.nice(args.nice)
    runtime_local.verify_addons()
    torch.set_num_threads(1)
    config, records, binding = prepare(args.config, args.studies, args.selection)
    cases = [(w, p) for w in args.workers for p in args.prefetch]
    random.Random(config['seed']).shuffle(cases)
    report = {'schema': 'e2_cpu_input_scaling_v1', 'state': 'preview', 'created_at': now(),
        'config_path': str(args.config.resolve()), 'source_sha256': digest(__file__),
        'binding': binding, 'case_order': cases, 'warmup_studies': args.warmup_studies,
        'max_seconds': args.max_seconds, 'system_before': system_snapshot(), 'results': [],
        'process_nice': os.getpriority(os.PRIO_PROCESS, 0),
        'consumer_delay_seconds_per_study': args.consumer_delay_seconds,
        'concurrent_training_declared': args.concurrent_training,
        'exclusive_resources_claimed': not args.concurrent_training,
        'scientific_training_run': False, 'holdout_H_accessed': False,
        'official_validation_accessed': False, 'test_accessed': False,
        'limitations': ['CPU-only loader consumption; no model, pinning, GPU transfer, or compute overlap',
            'Tensor verification and throughput are independent fresh-worker passes',
            'Selected source files are read repeatedly; operating-system caches are not flushed',
            'Quantile subset and separate highest-view stress are not unbiased epoch estimates',
            'PSS/RSS/SHM sampled each second; brief peaks may be missed',
            'Concurrent-training measurements describe contested CPU capacity, not isolated epoch speed',
            'Frozen formal source/config remain unchanged; these results do not activate any loader setting']}
    if not args.execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report['state'] = 'running'
    save(args.output, report)
    deadline = time.monotonic()+args.max_seconds
    expected = None
    try:
        for workers, prefetch in cases:
            report['active_case'] = {'workers': workers, 'prefetch_factor': prefetch}
            save(args.output, report)
            verification = run_pass(config, records, workers, prefetch, args, verify=True, deadline=deadline)
            actual = (verification['order_sha256'], verification['tensor_sha256'])
            if expected is None:
                expected = actual
            row = {'workers': workers, 'prefetch_factor': prefetch, 'verification': verification,
                   'matches_reference_exactly': actual == expected}
            report['results'].append(row)
            save(args.output, report)
            if actual != expected:
                raise ValueError('Tensor bytes or Study order differ across loader settings')
            row['throughput'] = run_pass(config, records, workers, prefetch, args, verify=False, deadline=deadline)
            if row['throughput']['order_sha256'] != expected[0]:
                raise ValueError('Throughput pass delivered another order')
            save(args.output, report)
            print(json.dumps({'workers': workers, 'prefetch': prefetch,
                'tensor_order_exact': True, 'studies_per_second': row['throughput']['steady_studies_per_second'],
                'images_per_second': row['throughput']['steady_images_per_second'],
                'peak_sampled_pss_bytes': row['throughput']['memory_and_process_observations']['peak_sampled_process_pss_sum_bytes']}), flush=True)
        report['state'] = 'completed'
        report.pop('active_case', None)
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['finished_at'] = now()
        report['system_after'] = system_snapshot()
        save(args.output, report)


if __name__ == '__main__':
    main()
