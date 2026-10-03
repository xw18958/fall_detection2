#!/usr/bin/env python3
"""Reproducible seed42 smoke/full training launcher; no test-driven selection."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['smoke','train'],required=True)
    p.add_argument('--project-root',type=Path,default=Path('/raid1/xwan0900/fall_detection2'))
    p.add_argument('--work-root',type=Path,default=Path('/raid1/xwan0900/v5_m5_seed42_20261003'))
    a=p.parse_args();root=a.project_root;prior=root/'FallDetection30HzTraining/runs/v2_multisource_seed42_20261001'
    work=a.work_root/('smoke' if a.phase=='smoke' else 'seed42');work.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(Path(__file__).with_name('train.py')),'--experiment-v5',
        '--zip',str(root/'fd_datasets/30Hz_processed_clean_v2'),
        '--public-root',str(root/'fd_datasets/processed_v2'),
        '--m5-root',str(root/'fd_datasets/M5_combined_v5_20261003'),
        '--split-from',str(prior/'splits/group_splits.json'),
        '--init-ssl',str(prior/'checkpoints/ssl_pretrained.pt'),
        '--normalization-from',str(prior/'checkpoints/locked_model.pt'),
        '--baseline-checkpoint',str(prior/'checkpoints/locked_model.pt'),
        '--comparison-checkpoint',str(root/'FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt'),
        '--work',str(work),'--cache-root',str(a.work_root/'cache'),'--device','cuda:0',
        '--seed','42','--ssl-epochs','30','--head-epochs','3','--all-epochs','50',
        '--ssl-min-epochs','10','--ssl-patience','8','--min-full-epochs','10','--patience','10']
    if a.phase=='smoke':cmd+=['--smoke']
    elif not (a.work_root/'smoke/smoke_pass.json').is_file():raise RuntimeError('Complete GPU train/test/retrieval smoke before substantial training')
    if (work/'training_history.json').exists():raise FileExistsError('Never overwrite a started run')
    manifest=dict(command=cmd,python=sys.version,torch=torch.__version__,cuda_available=torch.cuda.is_available(),
                  git_revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,capture_output=True,text=True,check=True).stdout.strip(),
                  source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in
                       ['train.py','training_v2.py','training_v4.py','training_v5.py','prepare_m5_combined_v5.py']})
    if a.phase=='train':
        smoke=json.loads((a.work_root/'smoke/launch_manifest.json').read_text())
        if smoke['source_sha256']!=manifest['source_sha256']:
            raise RuntimeError('Training code changed after the GPU smoke; rerun smoke in a fresh work root')
    (work/'launch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(cmd,check=True)
    if a.phase=='smoke':
        subprocess.run([*cmd,'--mode','test'],check=True)
        (work/'smoke_pass.json').write_text(json.dumps(dict(passed=True,checks='GPU SSL/head/full train, grouped selection, locked test and retrieval',manifest_sha256=hashlib.sha256((work/'launch_manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')
    else:
        checkpoint=work/'checkpoints/locked_model.pt';c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if c['smoke'] or c['protocol_version']!=5:raise ValueError('Wrong trained checkpoint')
        lock=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  split_fingerprint=c['split_fingerprint'],threshold=c['threshold'],
                  selection=c['checkpoint_selection'],normalization_sha256=hashlib.sha256((work/'normalization.json').read_bytes()).hexdigest(),
                  test_evaluated=False,seed=42,protocol_version=5)
        (work/'training_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('V5_'+a.phase.upper()+'_PASS',flush=True)


if __name__=='__main__':main()
