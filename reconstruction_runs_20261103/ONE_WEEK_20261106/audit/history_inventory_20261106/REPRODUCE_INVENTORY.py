from __future__ import annotations
from pathlib import Path
import os,json,hashlib,re,collections,math,concurrent.futures,time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/home/featurize/work')
OUT=ROOT/'reconstruction_runs_20261103/ONE_WEEK_20261106/audit/history_inventory_20261106'
OUT.mkdir(parents=True,exist_ok=False)
START=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
SUFFIX={'.md','.txt','.rst','.json','.jsonl','.log','.yaml','.yml','.toml'}
SKIP={'node_modules','site-packages','site','wheels','__pycache__','vendor','cache','images512','images','信息文件'}
PRIVATE=('private_key','id_rsa','id_ed25519','secret','credentials','token.json','.env')
METRIC=re.compile(r'(?:^|\.)(?:macro_auc|val_macro_auc|auc_macro|best_macro_auc|best_auc|best_metric|macro_ap|macro_f1|best_epoch|epoch|completed_epochs|last_epoch|optimizer_updates|skipped_updates|attempted_updates|updates_this_epoch|images|studies|num_studies|num_images|seconds|state|status|stage|phase|stop_reason|error|exception|reason|selection_protocol|outer_fold_evaluated|is_best|full_manifest_coverage)$',re.I)
ERROR=re.compile(r'Traceback \(most recent call last\)|(?:RuntimeError|ValueError|TypeError|OSError|FileNotFoundError|KeyError|AssertionError|PermissionError|MemoryError|ModuleNotFoundError):[^\r\n]{0,400}|CUDA out of memory|loss[=: ]+nan\b|loss[=: ]+inf\b',re.I)
paths=[]; excluded=collections.Counter()
for directory,dirs,files in os.walk(ROOT):
 kept=[]
 for name in dirs:
  p=Path(directory)/name
  if name.startswith('.') or name in SKIP or p.is_relative_to(ROOT/'reconstruction_runs_20261103/ONE_WEEK_20261106/audit'):
   excluded['directory:'+name]+=1
  else:kept.append(name)
 dirs[:]=kept
 for name in files:
  p=Path(directory)/name
  if p.is_symlink() or name.startswith('.') or any(x in name.lower() for x in PRIVATE):
   excluded['sensitive_hidden_or_symlink']+=1;continue
  if p.suffix.lower() not in SUFFIX:excluded['not_document_or_structured_log']+=1;continue
  paths.append(p)
paths.sort()


def selected_scalars(value,prefix='',depth=0):
 out={}
 if isinstance(value,dict):
  for key,item in value.items():
   name=f'{prefix}.{key}' if prefix else str(key)
   if isinstance(item,(str,int,float,bool)) or item is None:
    if METRIC.search(name):out[name]=item[:450] if isinstance(item,str) else item
   elif isinstance(item,list):
    if METRIC.search(name) and len(item)<=100 and all(isinstance(x,(str,int,float,bool)) for x in item):out[name]=item
    elif depth<10:
     for i,x in enumerate(item):
      if isinstance(x,dict):out.update(selected_scalars(x,f'{name}[{i}]',depth+1))
   elif isinstance(item,dict) and depth<10:out.update(selected_scalars(item,name,depth+1))
 elif isinstance(value,list) and depth<10:
  for i,x in enumerate(value):
   if isinstance(x,dict):out.update(selected_scalars(x,f'{prefix}[{i}]',depth+1))
 return out


def history_stats(records):
 epochs=[];last={};best=None;best_epoch=None;best_path=None;rows=[]
 for n,row in enumerate(records,1):
  if not isinstance(row,dict):continue
  fields=selected_scalars(row)
  e=row.get('epoch')
  if isinstance(e,int):epochs.append(e)
  candidates=[(k,v) for k,v in fields.items() if k.lower() in {'macro_auc','validation.macro_auc','val_macro_auc','valid.macro_auc','auc_macro','metrics.macro_auc','val_auc_macro'} and isinstance(v,(int,float)) and math.isfinite(v)]
  if candidates:
   key,score=candidates[0]
   if best is None or score>best:best=float(score);best_epoch=e;best_path=key
  last=fields
  rows.append(dict(line=n,fields=fields))
 return dict(json_records=len(records),epoch_records=len(epochs),recorded_complete_epochs=sorted(set(epochs)),last_complete_epoch=max(epochs) if epochs else None,best_recorded_selection_macro_auc=best,best_recorded_selection_epoch=best_epoch,best_metric_path=best_path,last_record=last,records=rows)


def inspect(path):
 relative=str(path.relative_to(ROOT));before=path.stat();raw=path.read_bytes();after=path.stat()
 digest=hashlib.sha256(raw).hexdigest();text=raw.decode('utf-8-sig',errors='replace')
 entry=dict(path=relative,bytes=len(raw),sha256=digest,suffix=path.suffix.lower(),mtime_ns=after.st_mtime_ns,changed_during_read=(before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size),utf8_replacement_characters=text.count('\ufffd'),lines=len(text.splitlines()))
 result=dict(inventory=entry)
 suffix=path.suffix.lower()
 if suffix=='.json':
  try:
   obj=json.loads(text);fields=selected_scalars(obj)
   entry['json_parsed']=True
   result['structured']=dict(path=relative,sha256=digest,kind='json',top_level_keys=list(obj)[:60] if isinstance(obj,dict) else None,fields=fields)
  except (json.JSONDecodeError,ValueError) as exc:entry['json_parsed']=False;entry['parse_error']=str(exc)[:400]
 elif suffix=='.jsonl':
  records=[];errors=[]
  for i,line in enumerate(text.splitlines(),1):
   if not line.strip():continue
   try:records.append(json.loads(line))
   except (ValueError,TypeError) as exc:errors.append(dict(line=i,error=str(exc)[:220]))
  entry.update(jsonl_records=len(records),jsonl_parse_errors=errors)
  result['structured']=dict(path=relative,sha256=digest,kind='jsonl',records=len(records),parse_errors=errors)
  if 'history' in path.name.lower():result['history']=dict(path=relative,sha256=digest,**history_stats(records))
  else:
   result['structured']['first_fields']=selected_scalars(records[0]) if records else {}
   result['structured']['last_fields']=selected_scalars(records[-1]) if records else {}
 elif suffix=='.log':
  # Read every log byte; exact exception matches are evidence, not proof that a
  # whole experiment failed (historical/recovered/manual-stop cases stay distinct).
  matches=[m.group(0) for m in ERROR.finditer(text)]
  embedded=[];bad=0
  for line in text.splitlines():
   value=line.strip()
   if value.startswith('{') and value.endswith('}'):
    try:embedded.append(json.loads(value))
    except ValueError:bad+=1
  result['log']=dict(path=relative,sha256=digest,exception_match_count=len(matches),exception_samples=matches[:10]+(matches[-10:] if len(matches)>10 else []),embedded_json_records=len(embedded),embedded_json_parse_errors=bad,last_fields=selected_scalars(embedded[-1]) if embedded else {})
 else:
  headers=[line.strip() for line in text.splitlines() if line.lstrip().startswith('#')]
  result['document']=dict(path=relative,sha256=digest,title=headers[0] if headers else None,headings=headers[:100],characters=len(text))
 return result

all_results=[];failures=[]
print(json.dumps(dict(started_at=START,files_to_read=len(paths),total_bytes=sum(p.stat().st_size for p in paths))),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures={pool.submit(inspect,p):p for p in paths}
 for n,future in enumerate(concurrent.futures.as_completed(futures),1):
  path=futures[future]
  try:all_results.append(future.result())
  except BaseException as exc:failures.append(dict(path=str(path.relative_to(ROOT)),error=repr(exc)))
  if n%250==0:print(json.dumps(dict(read_files=n,total_files=len(paths),read_failures=len(failures))),flush=True)
all_results.sort(key=lambda x:x['inventory']['path'])
for kind,name in [('inventory','FILES_SHA256.jsonl'),('structured','STRUCTURED_SUMMARIES.jsonl'),('history','HISTORY_DETAILS.jsonl'),('log','LOG_SUMMARIES.jsonl'),('document','DOCUMENT_HEADINGS.jsonl')]:
 with (OUT/name).open('x') as stream:
  for r in all_results:
   if kind in r:stream.write(json.dumps(r[kind],ensure_ascii=False,allow_nan=True)+'\n')
histories=[r['history'] for r in all_results if 'history' in r]
groups=collections.defaultdict(list)
for h in histories:groups[h['sha256']].append(h)
def priority(h):
 p=h['path'];return (0 if p.startswith('runs/') else 1 if p.startswith('reconstruction_runs_20261103/') else 2 if '/training/' in p else 3,len(p),p)
unique=[]
for digest,items in groups.items():
 selected=min(items,key=priority);record={k:v for k,v in selected.items() if k!='records'}
 record['identical_history_paths']=[x['path'] for x in sorted(items,key=priority)]
 unique.append(record)
unique.sort(key=lambda x:x['path'])
(OUT/'UNIQUE_HISTORY_RUNS.json').write_text(json.dumps(unique,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
counts=collections.Counter();sizes=collections.Counter()
for r in all_results:
 e=r['inventory'];counts[e['suffix']]+=1;sizes[e['suffix']]+=e['bytes']
# Explicit statuses and failure artefacts are separate from prose mentions.
status_records=[];failure_records=[]
for r in all_results:
 p=r['inventory']['path'];base=Path(p).name.lower()
 if 'structured' in r:
  if any(x in base for x in ['status','progress','failure','stop','decision']):status_records.append(r['structured'])
  if 'failure' in base or 'failed' in base:failure_records.append(r['structured'])
(OUT/'STATUS_STOP_FAILURE_RECORDS.json').write_text(json.dumps(status_records,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
summary=dict(started_at=START,completed_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),root=str(ROOT),inventory_scope='Human documents .md/.txt/.rst and structured/log .json/.jsonl/.log/.yaml/.yml/.toml; all matched file contents read and SHA256 computed; no Python source or binaries are claimed line-by-line reviewed.',files=len(all_results),bytes=sum(sizes.values()),counts_by_suffix=dict(counts),bytes_by_suffix=dict(sizes),unique_content_hashes=len({r['inventory']['sha256'] for r in all_results}),history_files=len(histories),unique_history_content_sets=len(unique),recorded_positive_epoch_entries_across_unique_histories=sum(len([e for e in h['recorded_complete_epochs'] if e>0]) for h in unique),warning_epoch_total_not_independent_runs='Snapshots/continuations can overlap without identical full-file hashes; no sum is a count of independently executed training.',status_stop_decision_files=len(status_records),explicit_failure_named_json_files=len(failure_records),logs_with_exception_matches=sum(bool(r['log']['exception_match_count']) for r in all_results if 'log' in r),json_parse_failures=[r['inventory'] for r in all_results if r['inventory'].get('json_parsed') is False],jsonl_with_parse_failures=[r['inventory'] for r in all_results if r['inventory'].get('jsonl_parse_errors')],changed_during_read=[r['inventory']['path'] for r in all_results if r['inventory']['changed_during_read']],read_failures=failures,excluded_counts=dict(excluded),exclusions='Hidden dirs/files (.git/.ssh/.config etc), symlinks, known secret filenames, node_modules/site/site-packages/wheels/vendor/cache/images; binary weights/images/archives/PDF; this audit output. No external tool logs outside work.',limitations=['Filesystem inventory is not atomic; active new student logs may change after snapshot.','All included bytes were accessed mechanically; headings/scalars/exception extraction is not a claim of manual line-by-line review or causal validation.','Missing/disappeared temp files and outside-work local state are not reconstructed.','A history line is recorded-complete epoch evidence; an interrupted next epoch is not counted.','Duplicate history hashes identify exact copies only; nonidentical continuations and mechanical smokes are not independent full experiments.','Status failure may reflect user STOP/recoverable reporting errors; logs and stop receipts must decide interpretation.'])
(OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False),flush=True)
