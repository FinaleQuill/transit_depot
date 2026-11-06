"""Detached, single-owner training controller with explicit evidence and no retry."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from breakthrough.runtime import ROOT, command

def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()

def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)

def control(config_path, *, resume=False):
    config=json.loads(config_path.read_text()); output=Path(config['output'])
    with (output/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'STOP').exists():
            raise RuntimeError('STOP exists')
        if (output/'configuration.json').exists() and not resume:
            raise FileExistsError('Already started; explicit resume required')
        name='cxr-breakthrough-dinov2-20261105'
        arguments=['-m','breakthrough.train','--config',str(config_path),'--execute']
        if resume:
            arguments.append('--resume')
        cmd=command(arguments,config,name=name)
        receipt=dict(state='starting',started_at=now(),controller_pid=os.getpid(),
            ppid=os.getppid(),session_id=os.getsid(0),stdin=os.readlink('/proc/self/fd/0'),
            config=str(config_path),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            command=cmd,container=name,automatic_retries=0,checkpoint_dir=config['checkpoint_dir'])
        status=output/'controller_status.json';write(status,receipt)
        with (output/'training.log').open('a',buffering=1) as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            receipt.update(state='running',docker_client_pid=process.pid);write(status,receipt)
            code=process.wait()
        trainer=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        receipt.update(state=trainer.get('state','completed') if code==0 else 'failed',
            exit_code=code,trainer_state=trainer,finished_at=now())
        write(status,receipt)
        return code

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'breakthrough/plan.json')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--controller',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config_path=args.config.resolve()
    config=json.loads(config_path.read_text());output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if args.controller:
        raise SystemExit(control(config_path,resume=args.resume))
    if not args.execute:
        print(json.dumps(config,indent=2));return
    with (output/'.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'launch.json').exists() and not args.resume:
            raise FileExistsError('Launch receipt already exists')
        cmd=[sys.executable,'-u','-m','breakthrough.launch','--config',str(config_path),'--controller']
        if args.resume:
            cmd.append('--resume')
        with (output/'controller.log').open('a',buffering=1) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        receipt=dict(controller_pid=child.pid,launched_at=now(),command=cmd,detached=True)
        write(output/'launch.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    main()
