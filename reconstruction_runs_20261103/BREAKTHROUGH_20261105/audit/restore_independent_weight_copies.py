from pathlib import Path
import os, json, shutil, hashlib, argparse
p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()
r=json.loads((Path(__file__).resolve().parent.parent/"cleanup_receipt.json").read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
for g in r["groups"]:
 source=Path(g["canonical_verified_backup"])
 if sha(source)!=g["sha256"]:raise RuntimeError("Canonical hash mismatch")
 for old in g["before"][1:]:
  target=Path(old["path"])
  if sha(target)!=g["sha256"]:raise RuntimeError("Target changed; refusing overwrite")
  print("Restore independent copy:",target)
  if not a.execute:continue
  temp=target.with_name(target.name+".restore-independent.tmp")
  if temp.exists():raise RuntimeError("Temporary restore path exists")
  shutil.copy2(source,temp)
  if sha(temp)!=g["sha256"]:raise RuntimeError("Copy mismatch")
  os.chmod(temp,old["mode"]);os.utime(temp,ns=(old["mtime_ns"],old["mtime_ns"]))
  os.replace(temp,target)
print("Completed" if a.execute else "Preview only; add --execute to restore independent copies")
