"""Frozen data adapter for the existing standalone detector input contract."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import train as base
import training_v2 as data

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(own,public,checkpoint,split_path,cache,m5_combined=None):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['config']['fs']!=30 or ck['config']['win_sec']!=3:raise ValueError('Full checkpoint is not 30 Hz / 3 seconds')
    rs=data.own_records(Path(own))+data.public_records(Path(public))
    inherited=json.loads(Path(split_path).read_text());lookup={r['key']:r for r in rs}
    sets={role:set(keys) for role,keys in inherited['splits'].items()}
    if set.union(*sets.values())!=set(lookup) or sum(map(len,sets.values()))!=len(lookup):raise ValueError('Inherited partitions do not match source recordings')
    if any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Split leakage')
    splits={role:[lookup[k] for k in inherited['splits'][role]] for role in sets}
    group_sets={role:{r['group'] for r in rr} for role,rr in splits.items()}
    if any(group_sets[a]&group_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Group leakage')
    m5_rows=[]
    if m5_combined:
        root=Path(m5_combined);manifest=json.loads((root/'processed_v2'/'manifest.json').read_text())
        for row in manifest:
            path=root/'processed_v2'/row['output_file']
            if digest(path)!=row['output_sha256']:raise ValueError('M5 input checksum mismatch')
            r={'path':path,'key':data.M5+'/'+row['output_file'],'domain':data.M5,'normalization_domain':'own',
               'label':int(row['label']=='fall'),'ssl_label':int(row['label']=='fall'),'group':row['split_group_id'],
               'train_only':False,'fixed_split':row['split'],'standalone_compatibility':True}
            splits[row['split']].append(r);rs.append(r);m5_rows.append(row)
    arrays=data.arrays(rs,Path(cache))
    # Explicit compatibility VIEW, not a new dataset export or fit: the current
    # standalone firmware maps MPU physical units to historical private counts,
    # clamps +/-int16, and uses original own normalization. V5 SI normalization
    # is intentionally NOT substituted. Original prepared CSVs stay unchanged.
    for r in rs:
        if r.get('standalone_compatibility'):
            signal=arrays[r['key']];x=signal.values.copy()
            x[:,:3]=x[:,:3]/np.float32(9.80665)*np.float32(16384.)
            x[:,3:]=x[:,3:]/np.float32(np.pi/180)*np.float32(16.4)
            arrays[r['key']]=data.Signal(np.clip(x,-32768,32767),signal.ranges)
    cfg=base.Cfg(**ck['config']);cfg.stride_sec=.25
    audit={'checkpoint_sha256':digest(checkpoint),'inherited_split_sha256':digest(split_path),
      'original_split_fingerprint':inherited['fingerprint'],'normalization':'Frozen original checkpoint; no fit',
      'm5_input_view':'SI -> original standalone inferred private count scale -> int16 clamp -> frozen own normalization',
      'm5_partitions':'existing 13/3/3 fall recording split plus existing 5/1/1 negative boot split; exploratory falls from one boot',
      'record_counts':{k:len(v) for k,v in splits.items()},'label_policy':'Every 3-second view inherits original recording/container label; no event pseudo-labels',
      'preprocessing':'Existing segmented 30 Hz read_array; no cross-gap views or padding',
      'evaluation_grid':'Production 250 ms deadlines, latest acquired sample (floor 7.5*k), no off-grid final tail; this explicitly differs from historical classification endpoint coverage',
      'm5_manifest_sha256':digest(Path(m5_combined)/'processed_v2'/'manifest.json') if m5_combined else None}
    return splits,arrays,ck,cfg,audit

def evaluation(splits,arrays,normalization,cfg,role,phase_samples=0.):
    """Production quarter-second views, using the latest acquired sample.

    A 7.5-sample stride alternates 7/8. Unlike the historical classification
    sampler, never append an off-grid tail window. Phase is explicit for future
    jitter/phase sensitivity checks; source samples, labels and splits are fixed.
    """
    if not 0 <= phase_samples < 1:raise ValueError('Phase must be within one sample')
    ds=data.Clips(splits[role],arrays,normalization,cfg,random_crop=False)
    ds.entries=[];rows=[]
    for i,r in enumerate(ds.rs):
        for segment,(lo,hi) in enumerate(data.signal_ranges(arrays[r['key']])):
            duration=(hi-lo-90)/30
            steps=np.arange(int(np.floor(duration/.25))+1)
            starts=lo+np.floor(steps*7.5+phase_samples).astype(np.int64)
            for step,start in zip(steps,starts):
                ds.entries.append((i,int(start)))
                rows.append({'key':r['key'],'domain':r['domain'],'label':r['label'],'start':int(start),'segment':segment,'segment_samples':int(hi-lo),'step':int(step)})
    return ds,rows
