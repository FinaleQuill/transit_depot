"""One-shot read-only observation of E2 loader speed after audited resume."""
import json
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/home/featurize/work/reconstruction_runs_20261103/E2_20261104')
OUT = ROOT / 'performance_20261104/LIVE_SPEEDUP.json'

def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()

def save(value):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(OUT)

def main():
    if OUT.exists():
        raise FileExistsError('Preserve the original live observation: ' + str(OUT))
    launch = json.loads((ROOT / 'performance_launch.json').read_text())
    baseline = json.loads((ROOT / 'performance_20261104/baseline_training_interval.json').read_text())['baseline_interval']
    report = {'state': 'observing', 'started_at': now(), 'launch': launch,
              'baseline': baseline, 'snapshots': [], 'gpu_samples': [],
              'limitation': 'Observed different epoch intervals, not a matched stochastic replay; excludes epoch validation and saves.'}
    save(report)
    started = time.monotonic()
    first = None
    try:
        while time.monotonic() - started < 900:
            controller = json.loads((ROOT / 'controller/STATUS.json').read_text())
            if controller['controller_pid'] != launch['pid'] or controller['state'] in {'failed', 'paused', 'stopping'}:
                raise RuntimeError('Controller changed or stopped before observation completed')
            p = ROOT / 'global_control_seed20261104/progress.json'
            progress = json.loads(p.read_text())
            if progress.get('state') == 'training' and progress.get('epoch') == 2 and progress.get('studies_done', 0) >= 128:
                if first is None:
                    first = progress
                report['snapshots'].append({'observed_at': now(), 'progress': progress})
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], text=True, timeout=10)
                a,b,c = [float(x.strip()) for x in raw.split(',')]
                report['gpu_samples'].append({'observed_at': now(), 'utilization_percent':a, 'used_mib':b, 'power_watts':c})
                studies = progress['studies_done'] - first['studies_done']
                seconds = progress['elapsed_seconds'] - first['elapsed_seconds']
                if studies >= 640 and seconds > 0:
                    wait = progress['data_wait_seconds'] - first['data_wait_seconds']
                    report.update(state='completed', finished_at=now(),
                        interval={'first': first, 'last': progress, 'studies': studies, 'seconds': seconds,
                                  'studies_per_second': studies / seconds, 'data_wait_seconds': wait, 'data_wait_fraction':wait/seconds},
                        observed_speedup=(studies/seconds)/baseline['studies_per_second'],
                        gpu_mean_percent=statistics.mean(x['utilization_percent'] for x in report['gpu_samples']),
                        gpu_peak_used_mib=max(x['used_mib'] for x in report['gpu_samples']))
                    save(report)
                    print(json.dumps({k:report[k] for k in ['state','finished_at','observed_speedup','gpu_mean_percent','gpu_peak_used_mib','interval']}, ensure_ascii=False), flush=True)
                    return
            save(report)
            time.sleep(5)
        raise TimeoutError('No full 640-Study steady observation within 15 minutes; no auto-retry')
    except BaseException as error:
        report.update(state='failed', finished_at=now(), error=f'{type(error).__name__}: {error}')
        save(report)
        raise

if __name__ == '__main__':
    main()
