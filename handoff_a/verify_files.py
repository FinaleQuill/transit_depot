"""Host stdlib SHA verification. Does not import torch or start training."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent.parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for row in manifest['files']:
    path=root/row['path']
    if not path.is_file() or path.stat().st_size!=row['bytes']:
        raise SystemExit('Missing or resized: '+row['path'])
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=row['sha256']:raise SystemExit('SHA mismatch: '+row['path'])
print(f"PASS: {len(manifest['files'])} files, all SHA256 matched")
