from pathlib import Path
import os, json, hashlib, shutil, subprocess, stat, time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/home/featurize/work/reconstruction_runs_20261103/BREAKTHROUGH_20261105')
AUDIT=ROOT/'audit'
SOURCE=Path('/tmp/cxr_cleanup_duplicate_candidates_20261105.json')
RECEIPT=ROOT/'cleanup_receipt.json'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
def snapshot(p):
 s=p.stat()
 return {'path':str(p),'bytes':s.st_size,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'sha256':sha(p)}
def df():
 s=os.statvfs(ROOT)
 return {'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize,'df_text':subprocess.check_output(['df','-B1',str(ROOT)],text=True)}
def save(r):
 t=RECEIPT.with_suffix('.json.tmp')
 with t.open('w') as f:
  json.dump(r,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,RECEIPT)

ROOT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(exist_ok=True)
if RECEIPT.exists():raise RuntimeError('Existing receipt; refusing duplicate execution')
files=['cxr_rules_data_audit_20261105.md','cxr_rules_data_audit_20261105.json','cxr_cleanup_duplicate_candidates_20261105.json']
copied=[]
for name in files:
 source=Path('/tmp')/name;dest=AUDIT/name
 if dest.exists() and sha(dest)!=sha(source):raise RuntimeError('Conflicting audit copy')
 if not dest.exists():shutil.copy2(source,dest)
 if sha(source)!=sha(dest):raise RuntimeError('Audit copy mismatch')
 copied.append({'source':str(source),'destination':str(dest),'sha256':sha(dest)})
plan=json.loads(SOURCE.read_text())
r={'started_at_hong_kong':now(),'status':'verifying_before_mutation','authorized_scope':'Only three exact-byte weight duplicate groups; preserve all paths and canonical source files; no other removals','audit_saved_before_replacement':copied,'script_sha256':sha(__file__),'groups':[],'operations':[],'limitations':['Hard links share inode metadata; never modify retained checkpoints in place. To restore independent copies, run the saved restore script.','Filesystem free-space delta is global and can include concurrent unrelated activity.'],'restore_script':str(AUDIT/'restore_independent_weight_copies.py')}
r['process_precheck']={'nvidia_compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'docker_running':subprocess.check_output(['docker','ps','--format','{{.ID}} {{.Names}} {{.Status}}'],text=True)}
if r['process_precheck']['nvidia_compute'].strip() or r['process_precheck']['docker_running'].strip():raise RuntimeError('GPU/container activity found; inspect before replacement')
for group in plan['verified_groups']:
 before=[]
 for entry in group:
  p=Path(entry['path'])
  if not p.is_file() or p.is_symlink():raise RuntimeError('Expected normal file')
  snap=snapshot(p)
  if snap['sha256']!=entry['sha256'] or snap['bytes']!=entry['bytes']:raise RuntimeError('Content changed')
  before.append(snap)
 if len({x['sha256'] for x in before})!=1 or len({x['device'] for x in before})!=1:raise RuntimeError('Group differs or filesystem mismatch')
 r['groups'].append({'canonical_verified_backup':before[0]['path'],'sha256':before[0]['sha256'],'before':before})
r['independent_inodes_before']=len({(x['device'],x['inode']) for g in r['groups'] for x in g['before']})
if r['independent_inodes_before']!=8:raise RuntimeError('Expected 8 independent inodes; inspect concurrency')
r['logical_duplicate_bytes_removed']=sum(x['bytes'] for g in r['groups'] for x in g['before'][1:])
r['allocated_duplicate_bytes_removed']=sum(x['allocated_bytes'] for g in r['groups'] for x in g['before'][1:] if x['nlink']==1)
r['filesystem_before']=df();r['status']='verified_backups_and_audit_saved';save(r)
restore='''from pathlib import Path\nimport os, json, shutil, hashlib, argparse\np=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()\nr=json.loads((Path(__file__).resolve().parent.parent/"cleanup_receipt.json").read_text())\ndef sha(path):\n h=hashlib.sha256()\n with open(path,"rb") as f:\n  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)\n return h.hexdigest()\nfor g in r["groups"]:\n source=Path(g["canonical_verified_backup"])\n if sha(source)!=g["sha256"]:raise RuntimeError("Canonical hash mismatch")\n for old in g["before"][1:]:\n  target=Path(old["path"])\n  if sha(target)!=g["sha256"]:raise RuntimeError("Target changed; refusing overwrite")\n  print("Restore independent copy:",target)\n  if not a.execute:continue\n  temp=target.with_name(target.name+".restore-independent.tmp")\n  if temp.exists():raise RuntimeError("Temporary restore path exists")\n  shutil.copy2(source,temp)\n  if sha(temp)!=g["sha256"]:raise RuntimeError("Copy mismatch")\n  os.chmod(temp,old["mode"]);os.utime(temp,ns=(old["mtime_ns"],old["mtime_ns"]))\n  os.replace(temp,target)\nprint("Completed" if a.execute else "Preview only; add --execute to restore independent copies")\n'''
(AUDIT/'restore_independent_weight_copies.py').write_text(restore)
shutil.copy2(__file__,AUDIT/Path(__file__).name)
# Parent has exclusive ownership of these mutations. Re-check activity immediately before swaps.
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU training started during verification')
for group in r['groups']:
 canonical=Path(group['canonical_verified_backup'])
 for old in group['before'][1:]:
  target=Path(old['path']);st=target.stat()
  if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(old['device'],old['inode'],old['bytes'],old['mtime_ns']):raise RuntimeError('Target metadata changed since verified')
  temporary=target.with_name(target.name+'.dedup-link-'+str(os.getpid())+'.tmp')
  if temporary.exists():raise RuntimeError('Unexpected temporary path')
  os.link(canonical,temporary)
  try:
   if temporary.stat().st_ino!=canonical.stat().st_ino:raise RuntimeError('Hardlink inode mismatch')
   os.replace(temporary,target)
  finally:
   if temporary.exists():temporary.unlink()
  r['operations'].append({'target':str(target),'canonical':str(canonical),'original_inode':old['inode'],'new_inode':target.stat().st_ino,'sha256':group['sha256'],'timestamp':now()})
  r['status']='replacing_verified_duplicates';save(r)
for group in r['groups']:
 group['after']=[snapshot(Path(x['path'])) for x in group['before']]
 if any(x['sha256']!=group['sha256'] for x in group['after']):raise RuntimeError('After hash mismatch')
 if len({(x['device'],x['inode']) for x in group['after']})!=1:raise RuntimeError('After inode mismatch')
r['independent_inodes_after']=len({(x['device'],x['inode']) for g in r['groups'] for x in g['after']})
if r['independent_inodes_after']!=3:raise RuntimeError('Expected 3 final inodes')
r['filesystem_after']=df();r['observed_available_bytes_delta']=r['filesystem_after']['available_bytes']-r['filesystem_before']['available_bytes'];r['all_original_paths_preserved']=True;r['all_original_sha256_preserved']=True;r['canonical_files_unmodified']=True;r['status']='completed';r['completed_at_hong_kong']=now();save(r)
print(json.dumps({k:r[k] for k in ['status','independent_inodes_before','independent_inodes_after','logical_duplicate_bytes_removed','allocated_duplicate_bytes_removed','observed_available_bytes_delta','all_original_paths_preserved','all_original_sha256_preserved']},indent=2))
