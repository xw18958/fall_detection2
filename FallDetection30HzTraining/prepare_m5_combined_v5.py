#!/usr/bin/env python3
"""Combine verified full-range M5 negatives and user-labelled M5 fall crops."""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_m5_falls import inspect, sha
from preprocess_public_v2 import crop, resample
from preprocess_noisy_fall_dataset import detect_event

FEATURES=['Acc_X','Acc_Y','Acc_Z','Gyro_X','Gyro_Y','Gyro_Z']


def prepare(falls, negatives, output):
    if (output/'processed_v2').exists():
        raise FileExistsError('Never overwrite an existing dataset version')
    verification=json.loads((negatives/'processed_v2/verification.json').read_text())
    if verification['status']!='passed' or verification['conversion_convention']!='si' or verification['software_clipping']:
        raise ValueError('Require the verified full-range SI negatives')
    sessions=sorted(p for p in falls.iterdir() if p.is_dir() and (p/'metadata.json').exists())
    inspected=[(p,*inspect(p)) for p in sessions]
    if len(inspected)!=19 or len({r['session_id'] for p,r,rows,x in inspected})!=19:
        raise ValueError('Require exactly the 19 inventoried new falls')
    # Whole recordings are atomic. This is an exploratory within-boot split,
    # explicitly weaker than the unchanged negative boot holdouts.
    rng=np.random.default_rng(42)
    order=rng.permutation(len(inspected)); roles={int(i):role for role,ids in
        [('val',order[:3]),('test',order[3:6]),('train',order[6:])] for i in ids}
    out=output/'processed_v2'; (out/'fall').mkdir(parents=True);(out/'non-fall').mkdir()
    manifest=[]; numeric_seen=set()
    for i,(session,r,rows,physical) in enumerate(inspected):
        t=np.asarray([(row[1]-rows[0][1])/1e6 for row in rows])
        x,grid=resample(t,np.asarray(physical),30.,30.)
        ev=detect_event(pd.DataFrame(x,columns=FEATURES),30.,1.)
        start,end,kind=crop(len(x),ev['mid_idx'],30.)
        if end-start<90:raise ValueError('Short fall would require padding; refuse')
        name='fall/'+session.name+'.csv'; path=out/name
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_ms',*FEATURES])
            for j,values in enumerate(x[start:end]):w.writerow([j*1000/30,*values])
        digest=__import__('hashlib').sha256(np.asarray(x[start:end],'<f4').tobytes()).hexdigest()
        if digest in numeric_seen:raise ValueError('Duplicate positive crop')
        numeric_seen.add(digest)
        manifest.append(dict(source_session=session.name,session_id=r['session_id'],output_file=name,
             output_sha256=sha(path),label='fall',split=roles[i],split_group_id='M5SESSION:'+r['session_id'],
             boot_id=r['boot_id'],participant='unknown',placement='unknown',fs_hz=30,
             raw_samples=len(rows),processed_samples=end-start,crop_kind=kind,crop_start_seconds=float(grid[start]),
             crop_end_seconds_exclusive=float(grid[end-1]+1/30),event=ev,raw_checksums=r['checksums'],
             label_source='user confirmed one fall per recording and requested existing 5s-crop method'))
    with (negatives/'processed_v2/manifest.csv').open() as f:
        for row in csv.DictReader(f):
            source=negatives/'processed_v2'/row['output_file'];target=out/row['output_file']
            if sha(source)!=row['output_sha256']:raise ValueError('Negative checksum mismatch')
            shutil.copy2(source,target)
            manifest.append(dict(source_session=row['source_session'],session_id=row['session_id'],
                 output_file=row['output_file'],output_sha256=row['output_sha256'],label='non-fall',
                 split=row['split'],split_group_id=row['split_group_id'],boot_id=row['boot_id'],
                 participant=row['participant'],placement=row['placement'],fs_hz=30,
                 raw_samples=int(row['raw_samples']),processed_samples=int(row['processed_samples']),
                 crop_kind='full_session',negative_manifest_sha256=sha(negatives/'processed_v2/manifest.csv'),
                 label_source='previous verified user-designated M5 hard negatives'))
    if len(manifest)!=26:raise ValueError('Expected 19 falls plus seven negatives')
    groups={}
    for r in manifest:groups.setdefault(r['split_group_id'],set()).add(r['split'])
    if any(len(x)!=1 for x in groups.values()):raise ValueError('Record/group leakage')
    summary=dict(status='passed',protocol_version=5,raw_sessions=len(manifest),
       conversion_convention='si',software_clipping=False,units={'acceleration':'m/s^2','gyroscope':'rad/s'},
       prepared_samples=sum(r['processed_samples'] for r in manifest),
       source_label_counts=dict(Counter(r['label'] for r in manifest)),
       split_label_counts=dict(Counter(r['split']+'/'+r['label'] for r in manifest)),
       crop_counts=dict(Counter(r['crop_kind'] for r in manifest)),
       preprocessing='existing paired acceleration/gyro event midpoint and 5s centered/3s boundary crop; no padding',
       crop_label_policy='all 3s views from positive containers remain positive; no phase pseudo-labels',
       negative_policy='seven original full-length prepared SI negatives copied byte-for-byte; original boot splits retained',
       split_limitations='new positives share one boot, participant and placement unknown; 13/3/3 whole-recording split is exploratory, not independent participant/boot validation',
       seed=42,test_used_for_preprocessing_tuning=False,
       positive_raw_samples=sum(len(rows) for p,r,rows,x in inspected),
       positive_quality_flags={k:sum(r['quality_flags'][k] for p,r,rows,x in inspected) for k in inspected[0][1]['quality_flags']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary['manifest_sha256']=sha(out/'manifest.json')
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def records(root):
    """V5 loader; does not weaken the V4 seven-negative-only input guard."""
    root=Path(root)/'processed_v2';v=json.loads((root/'verification.json').read_text())
    if (v['status'],v['protocol_version'],v['conversion_convention'],v['software_clipping'])!=('passed',5,'si',False):
        raise ValueError('Wrong combined M5 dataset protocol')
    if sha(root/'manifest.json')!=v['manifest_sha256']:raise ValueError('Manifest checksum mismatch')
    rr=[]
    for r in json.loads((root/'manifest.json').read_text()):
        if r['label'] not in ['fall','non-fall'] or r['split'] not in ['train','val','test'] or r['fs_hz']!=30:
            raise ValueError('Invalid combined label, split or rate')
        p=root/r['output_file']
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=r['output_sha256']:
            raise ValueError('Invalid prepared file path/checksum')
        # Stable legacy domain name keeps original seven-source graph and weights.
        d='m5_hard_negatives'; y=int(r['label']=='fall')
        rr.append(dict(path=p,key=d+'/'+r['output_file'],domain=d,normalization_domain=d,
                       label=y,ssl_label=y,group=r['split_group_id'],fixed_split=r['split'],train_only=False))
    if len(rr)!=26 or len({r['key'] for r in rr})!=26:raise ValueError('Invalid combined M5 recording inventory')
    return rr


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['falls','negatives','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.falls,a.negatives,a.output),indent=2))
