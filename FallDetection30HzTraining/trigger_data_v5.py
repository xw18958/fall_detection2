"""Strict V5 SI adapter; uses existing splits and checkpoint normalization."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
# Frozen reader performs a lazy absolute import of its resampling dependency.
sys.path.insert(0,str(ROOT/'v5_reference'))
spec=importlib.util.spec_from_file_location('frozen_v5_data',ROOT/'v5_reference/training_v2.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(own,public,m5,checkpoint,split_path,cache):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['config']['fs']!=30 or ck['config']['win_sec']!=3:
        raise ValueError('V5 requires 30 Hz and 90 samples')
    root=Path(m5)/'processed_v2'
    verification=json.loads((root/'verification.json').read_text())
    if (verification['status'],verification['protocol_version'],verification['conversion_convention'],verification['software_clipping'])!=('passed',5,'si',False):
        raise ValueError('Wrong M5 data protocol')
    if digest(root/'manifest.json')!=verification['manifest_sha256']:
        raise ValueError('M5 manifest changed')
    rs=data.own_records(Path(own))+data.public_records(Path(public))
    manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest:
        path=root/row['output_file']
        if not path.resolve().is_relative_to(root.resolve()) or digest(path)!=row['output_sha256']:
            raise ValueError('M5 recording checksum/path mismatch')
        if row['fs_hz']!=30 or row['label'] not in ('fall','non-fall') or row['split'] not in ('train','val','test'):
            raise ValueError('M5 labels, rate or split invalid')
        rs.append(dict(path=path,key=data.M5+'/'+row['output_file'],domain=data.M5,
                       normalization_domain=data.M5,label=int(row['label']=='fall'),
                       ssl_label=int(row['label']=='fall'),group=row['split_group_id'],
                       fixed_split=row['split'],train_only=False))
    if len(manifest)!=26:raise ValueError('V5 M5 inventory changed')
    inherited=json.loads(Path(split_path).read_text());lookup={r['key']:r for r in rs}
    if len(lookup)!=len(rs):raise ValueError('Duplicate recording keys')
    # Both public/private inherited keys and M5 fixed roles must match the
    # actual locked V5 run, rather than attaching new data to a V2 split.
    splits={role:[lookup[k] for k in inherited['splits'][role]] for role in ('train','val','test')}
    keys=[r['key'] for rr in splits.values() for r in rr]
    if set(keys)!=set(lookup) or len(keys)!=len(lookup):raise ValueError('V5 split inventory mismatch')
    groups={role:{r['group'] for r in rr} for role,rr in splits.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Group leakage')
    for role,rr in splits.items():
        if any(r.get('fixed_split',role)!=role for r in rr):raise ValueError('M5 split changed')
    if inherited['fingerprint']!=ck['split_fingerprint']:raise ValueError('Checkpoint split differs')
    if set(ck['normalization'])!=set(['own',data.M5,*data.PUBLIC]):
        raise ValueError('Expected frozen seven-domain V5 normalization')
    cfg=SimpleNamespace(**dict(ck['config'],stride_sec=.25))
    arrays=data.arrays(rs,Path(cache))
    audit=dict(checkpoint_sha256=digest(checkpoint),split_sha256=digest(split_path),
               m5_manifest_sha256=digest(root/'manifest.json'),normalization='Frozen V5 checkpoint; no refit',
               m5_units='m/s^2 and rad/s; no private-count conversion or clipping',
               record_counts={k:len(v) for k,v in splits.items()},
               split_limitations=verification['split_limitations'])
    return splits,arrays,ck,cfg,audit

def evaluation(splits,arrays,normalization,cfg,role):
    ds=data.Clips(splits[role],arrays,normalization,cfg,random_crop=False)
    ds.entries=[];rows=[]
    for i,r in enumerate(ds.rs):
        for segment,(lo,hi) in enumerate(data.signal_ranges(arrays[r['key']])):
            steps=np.arange(int(np.floor((hi-lo-90)/7.5))+1)
            for step,start in zip(steps,lo+np.floor(steps*7.5).astype(np.int64)):
                ds.entries.append((i,int(start)))
                rows.append(dict(key=r['key'],domain=r['domain'],label=r['label'],start=int(start),
                                 segment=segment,segment_samples=int(hi-lo),step=int(step)))
    return ds,rows
