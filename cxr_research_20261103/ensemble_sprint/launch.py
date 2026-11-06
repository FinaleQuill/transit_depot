"""Detached fixed queue: context512 -> opacity512 -> one comparison -> winner CSV."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from sprint.runtime import ROOT, command
from sprint.launch import write_json


def now():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def controller(path, config):
    output=Path(config['output'])
    with (output/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'controller_status.json').exists():
            raise FileExistsError('Queue exists; no implicit repeat or budget extension')
        status={'state':'starting','started_at':now(),'controller_pid':os.getpid(),
                'controller_ppid':os.getppid(),'session_id':os.getsid(0),
                'stdin':os.readlink('/proc/self/fd/0'),'has_tty':os.isatty(0),
                'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'completed_steps':[],'automatic_retries':0,'final_validation_runs':0}
        write_json(output/'controller_status.json',status)
        def execute(name,arguments):
            if (output/'STOP').exists():
                status.update(state='paused',updated_at=now())
                write_json(output/'controller_status.json',status)
                return False
            cmd=command(arguments,config,name='cxr-ensemble-20261105-'+name)
            status.update(state='running',current_step=name,command=cmd,updated_at=now())
            write_json(output/'controller_status.json',status)
            with (output/(name+'.log')).open('a',buffering=1) as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status['docker_client_pid']=child.pid
                write_json(output/'controller_status.json',status)
                code=child.wait()
            if code:
                status.update(state='failed',exit_code=code,finished_at=now())
                write_json(output/'controller_status.json',status)
                return False
            status['completed_steps'].append(name)
            status.update(updated_at=now())
            write_json(output/'controller_status.json',status)
            return True
        for member in config['members']:
            if not execute(member['id'],['-m','ensemble_sprint.train','--config',str(path),
                                          '--member',member['id'],'--execute']):
                return
            result=json.loads((output/member['id']/'status.json').read_text())
            if result.get('state')!='completed':
                status.update(state=result.get('state','failed'),finished_at=now())
                write_json(output/'controller_status.json',status);return
        if not execute('final_compare',['-m','ensemble_sprint.evaluate','--config',str(path)]):
            return
        status['final_validation_runs']=1
        report=json.loads((output/'final_comparison/report.json').read_text())
        if report.get('candidate_won') and config['generate_winning_test_csv']:
            if not execute('winning_export',['-m','ensemble_sprint.deploy','--config',str(path),'--execute']):
                return
        status.update(state='completed',finished_at=now(),candidate_won=report.get('candidate_won'),
                      final_report=str(output/'final_comparison/report.json'),
                      generated_new_csv=bool(report.get('candidate_won') and config['generate_winning_test_csv']))
        write_json(output/'controller_status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'ensemble_sprint/plan.json')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--controller',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();path=args.config.resolve(strict=True);config=json.loads(path.read_text())
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True,exist_ok=True)
    if args.controller:
        controller(path,config);return
    if not args.execute:
        print(json.dumps({'members':[(m['id'],m['epochs']) for m in config['members']],
                          'frozen_base':config['base_weights'],'output':str(output)}));return
    with (output/'.launch.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'launch.json').exists() or (output/'controller_status.json').exists():
            raise FileExistsError('This fixed queue was already launched')
        if (output/'STOP').exists():
            raise RuntimeError('Ensemble STOP present')
        running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if running:
            raise RuntimeError('Finish the authorized earlier inference before starting training')
        argv=[sys.executable,'-u','-m','ensemble_sprint.launch','--controller','--config',str(path)]
        with (output/'controller.log').open('a',buffering=1) as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'state':'launched','launched_at':now(),'controller_pid':child.pid,'detached':True,
                 'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'log':str(output/'controller.log'),'status':str(output/'controller_status.json')}
        write_json(output/'launch.json',receipt)
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
