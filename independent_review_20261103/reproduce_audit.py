"""Read-only audit of source, Git, and explicitly named train/valid artifacts.

Writes only next to this script. No model imports, image reads, or test predictions.
"""
from pathlib import Path
import ast
import collections
import csv
from datetime import datetime
import hashlib
import json
import statistics
import subprocess
from zoneinfo import ZoneInfo

WORK = Path('/home/featurize/work')
REPO = WORK / 'cloud_project_v2'
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


facts = {'observed_hong_kong': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()}
facts['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
facts['git_commit_count'] = int(subprocess.check_output(['git', 'rev-list', '--count', 'HEAD'], cwd=REPO))
facts['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True)
facts['rules_sha256'] = {p.name: sha(p) for p in (WORK/'rules.pdf', WORK/'rules_1.pdf')}
banks = {}
facts['data'] = {}
for split, filename in [('train', 'train_clean.csv'), ('valid', 'valid_manifest.csv')]:
    path = REPO/'data/clean_v1_reviewed'/filename
    with path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    studies = collections.defaultdict(list)
    patients = collections.defaultdict(list)
    for row in rows:
        studies[row['study_key']].append(row)
    for group in studies.values():
        patients[group[0]['subject_id']].append(json.loads(group[0]['targets']))
    sizes = sorted(map(len, patients.values()), reverse=True)
    views = collections.defaultdict(list)
    for group in studies.values():
        key = '+'.join(sorted(set((r.get('view_position') or 'MISSING').upper() for r in group)))
        views[key].append(json.loads(group[0]['targets']))
    facts['data'][split] = {
        'manifest': str(path), 'manifest_sha256': sha(path),
        'images': len(rows), 'studies': len(studies), 'patients': len(patients),
        'study_image_count_distribution': dict(sorted(collections.Counter(map(len, studies.values())).items())),
        'positive_studies': [sum(json.loads(g[0]['targets'])[c] for g in studies.values()) for c in range(10)],
        'positive_patients': [sum(any(y[c] for y in ys) for ys in patients.values()) for c in range(10)],
        'maximum_studies_per_patient': max(sizes), 'top10_studies': sum(sizes[:10]),
        'top10_share': sum(sizes[:10])/len(studies),
        'median_dimensions': {k: statistics.median(int(r[k]) for r in rows) for k in ('width', 'height')},
        'study_views': {k: {'studies': len(v), 'class10_positive': sum(y[9] for y in v)} for k, v in views.items()},
    }
    banks[split] = {k: {r[k] for r in rows} for k in ('subject_id','study_key','file_sha256','pixel_sha256')}
facts['train_valid_manifest_overlaps'] = {k: len(banks['train'][k]&banks['valid'][k]) for k in banks['train']}
facts['stop_files'] = {}
for relative in ['artifacts/resolution_explore_20261103/STOP', 'artifacts/resolution_explore_20261103/queue/STOP', 'artifacts/resolution_explore_20261103/roi_queue_20261103/STOP']:
    path = REPO/relative
    facts['stop_files'][relative] = path.read_text() if path.exists() else None
save('audit_facts.json', facts)

tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
paths = [REPO/p for p in tracked if p and Path(p).suffix in ('.py','.sh','.c','.h')]
paths.extend(REPO/p for p in ('fixed_student_candidate.py','review_student_upper_bound.py','student_probability_fusion.py'))
paths.extend((WORK/'archive_v1/cloud_project').glob('*.py'))
source_index = []
for path in sorted(set(paths)):
    if not path.is_file():
        continue
    content = path.read_text()
    item = {'path': str(path), 'sha256': sha(path), 'lines': len(content.splitlines())}
    if path.suffix == '.py':
        try:
            tree = ast.parse(content)
            item['definitions'] = [{'name': n.name, 'line': n.lineno} for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            item['imports'] = sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module})
        except SyntaxError as exc:
            item['parse_error'] = str(exc)
    source_index.append(item)
save('source_inventory.json', source_index)
(OUT/'git_history.txt').write_text(subprocess.check_output(['git','log','--reverse','--format=%h %aI %s'], cwd=REPO, text=True))

history_roots = [WORK/'runs', WORK/'archive_v1/runs', REPO/'artifacts/kd_v1/student_upper_bound_20261030', REPO/'artifacts/kd_v1/bounded_finetune_20261101', REPO/'artifacts/kd_v1/frozen_head_20261101', REPO/'artifacts/tight_crop_ab_20261102']
history_index = []
for root in history_roots:
    for path in sorted(root.rglob('history.jsonl')):
        records = []
        errors = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({'line': line_number, 'error': str(exc)})
        history_index.append({'path': str(path), 'sha256': sha(path), 'records': records, 'errors': errors})
save('training_histories.json', history_index)
print(json.dumps({'git_head': facts['git_head'], 'commits': facts['git_commit_count'], 'source_files_indexed': len(source_index), 'python_files': sum(p['path'].endswith('.py') for p in source_index), 'history_files_read': len(history_index), 'history_records': sum(len(p['records']) for p in history_index), 'parse_errors': sum('parse_error' in p for p in source_index), 'data': {s: {k: v[k] for k in ('images','studies','patients','top10_share')} for s,v in facts['data'].items()}}, ensure_ascii=False, indent=2))
