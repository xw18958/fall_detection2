#!/usr/bin/env python3
"""Run the five independent preprocessing jobs concurrently, retaining logs/status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preprocess_public_v2 import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root/'.runner.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    logs = args.output_root/'logs'; logs.mkdir(exist_ok=True)
    script = Path(__file__).with_name('preprocess_public_v2.py')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    limits = {'CGU_BES': 13, 'Cogent': 1, 'SFU_IMU': 12, 'UCI_SimulatedFalls': 12, 'PAMAP2': 1}
    status, jobs = {}, {}
    started = time.time()
    try:
        for dataset in args.datasets:
            output = args.output_root/dataset
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite {output}')
            cmd = [sys.executable, '-u', str(script), '--root', str(args.root),
                   '--dataset', dataset, '--output', str(output), '--workers', str(args.workers)]
            if args.mode == 'smoke':
                cmd += ['--limit', str(limits[dataset])]
            log = (logs/(dataset+'.log')).open('w')
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            jobs[dataset] = (process, log)
            status[dataset] = {'status': 'running', 'pid': process.pid, 'log': str(log.name)}
        while jobs:
            for dataset, (process, log) in list(jobs.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del jobs[dataset]
                status[dataset].update(status='complete' if code == 0 else 'failed', exit_code=code)
                if code == 0:
                    status[dataset]['summary'] = json.loads((args.output_root/dataset/'preprocessing_summary.json').read_text())
                print(json.dumps({dataset: status[dataset]}), flush=True)
            temp = args.output_root/'status.json.tmp'
            temp.write_text(json.dumps({'mode': args.mode, 'elapsed_seconds': round(time.time()-started, 1),
                                        'datasets': status}, indent=2)+'\n')
            temp.replace(args.output_root/'status.json')
            if jobs:
                time.sleep(2)
    finally:
        lock.unlink(missing_ok=True)
    raise SystemExit(0 if all(s['status'] == 'complete' for s in status.values()) else 1)


if __name__ == '__main__':
    main()
