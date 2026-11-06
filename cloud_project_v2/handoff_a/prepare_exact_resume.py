"""Create a new writable exact-state E9 clone; never alter parent_run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):value.update(chunk)
    return value.hexdigest()


def main():
    bundle=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=bundle/'experiments/exact_A_best')
    args=parser.parse_args();output=args.output.resolve();parent=bundle/'parent_run'
    if output.exists() or output.is_relative_to(parent.resolve()):
        raise FileExistsError('Choose a new output outside the immutable parent_run')
    info=json.loads((bundle/'BUNDLE_INFO.json').read_text())
    expected=info['best_checkpoint_sha256'];source=parent/'best.pt'
    if sha(source)!=expected:raise ValueError('Original best.pt SHA256 differs')
    output.mkdir(parents=True)
    for name in ('best.pt','last.pt'):
        shutil.copyfile(source,output/name)
        if sha(output/name)!=expected:raise RuntimeError('Clone SHA256 mismatch')
    for name in ('config.json','pretrained_provenance.json','best_metrics.json'):
        shutil.copyfile(parent/name,output/name)
    rows=[json.loads(line) for line in (parent/'history.jsonl').read_text().splitlines() if line.strip()]
    (output/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch']<=info['best_epoch']))
    receipt={'operation':'byte-identical best-E9 checkpoint clone for exact-state continuation',
             'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'parent_best_sha256':expected,'inherited_epoch':info['best_epoch'],
             'next_epoch':info['best_epoch']+1,'parent_modified':False,
             'optimizer_rng_scaler_restored_when_training':True,'recipe_changed':False}
    (output/'clone_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'prepared':str(output),'next_epoch':receipt['next_epoch'],'parent_unchanged':True},indent=2))


if __name__=='__main__':main()
