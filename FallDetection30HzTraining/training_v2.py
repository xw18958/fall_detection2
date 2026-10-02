"""V2 clip supervision, grouped evaluation and seven-source target adaptation.

The network remains in train.py. This module owns the data protocol and execution;
no peaks or pseudo-event labels are computed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PUBLIC = ['CGU_BES', 'Cogent', 'SFU_IMU', 'UCI_SimulatedFalls', 'PAMAP2']
M5 = 'm5_hard_negatives'
NORMALIZATION_DOMAINS = ['own', *PUBLIC]
DOMAINS = ['own', M5, *PUBLIC]
FEATURES = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
PROTOCOL_VERSION = 3


class Signal:
    """Packed real segments, with boundaries that crops must never cross."""
    def __init__(self, values, ranges):
        self.values = np.asarray(values, np.float32)
        self.ranges = np.asarray(ranges, np.int64)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def signal_ranges(x):
    return getattr(x, 'ranges', np.array([[0, len(x)]], np.int64))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class Groups:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def root(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.root(self.parent[k])
        return self.parent[k]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def own_records(root):
    rs = []
    for folder, label in [('fall', 1), ('non-fall', 0)]:
        for p in sorted((root/folder).glob('*.csv')):
            rel = str(p.relative_to(root))
            rs.append({'path': p, 'key': 'own/'+rel, 'rel': rel, 'domain': 'own',
                       'label': label, 'ssl_label': label, 'train_only': False})
    if not rs:
        raise ValueError('Empty own dataset: '+str(root))
    by_name = {r['path'].name: r for r in rs}
    if len(by_name) != len(rs):
        raise ValueError('Ambiguous private CSV basenames')
    audit = root/'preprocessing_audit.csv'
    if not audit.exists():
        raise ValueError('V2 preprocessing_audit.csv is required to group related crops')
    uf = Groups([r['key'] for r in rs])
    for row in pd.read_csv(audit).fillna('').to_dict('records'):
        parent = by_name.get(row['source_file'])
        if parent is None:
            raise ValueError('Audit references missing fall file: '+row['source_file'])
        for field in ['generated_pre_negative', 'generated_post_negative']:
            name = row.get(field, '')
            if name:
                if name not in by_name:
                    raise ValueError('Audit references missing negative: '+name)
                uf.join(parent['key'], by_name[name]['key'])
        for name in str(row.get('duplicate_group', '')).split('|'):
            if name and name in by_name:
                uf.join(parent['key'], by_name[name]['key'])
        if float(row.get('padded_total_samples', 0) or 0) > 0:
            parent['train_only'] = True
    # Also link exact numeric duplicates not covered by the fall-only V2 audit.
    seen = {}
    for r in rs:
        values = pd.read_csv(r['path'], usecols=FEATURES)[FEATURES].to_numpy('<f8')
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if digest in seen:
            uf.join(r['key'], seen[digest])
        else:
            seen[digest] = r['key']
    forced = {uf.root(r['key']) for r in rs if r['train_only']}
    for r in rs:
        r['group'] = uf.root(r['key'])
        r['train_only'] = r['group'] in forced
    return rs


def m5_records(root):
    root = Path(root)
    summary_path = root/'processed_v2/verification.json'
    manifest_path = root/'processed_v2/manifest.csv'
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError('Verified M5 hard-negative preprocessing is required: '+str(root))
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'passed' or summary.get('raw_sessions') != 7:
        raise ValueError('M5 preprocessing verification did not pass')
    rs = []
    for row in pd.read_csv(manifest_path).fillna('').to_dict('records'):
        if row['label'] != 'non-fall' or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid M5 label or fixed split')
        p = root/'processed_v2'/row['output_file']
        if not p.is_file() or (row.get('output_sha256') and hashlib.sha256(p.read_bytes()).hexdigest() != row['output_sha256']):
            raise ValueError('M5 processed file missing or checksum mismatch: '+str(p))
        rs.append({'path': p, 'key': M5+'/'+row['output_file'], 'domain': M5,
                   'normalization_domain': 'own', 'label': 0, 'ssl_label': 0,
                   'group': str(row['split_group_id']), 'fixed_split': str(row['split']),
                   'train_only': False})
    if len(rs) != 7:
        raise ValueError(f'Expected seven verified M5 recordings, got {len(rs)}')
    return rs


def public_records(root):
    rs = []
    for domain in PUBLIC:
        directory = root/domain
        verification = json.loads((directory/'verification.json').read_text())
        if verification['status'] != 'passed':
            raise ValueError('Public preprocessing is not verified: '+domain)
        m = pd.read_csv(directory/'manifest.csv').fillna('')
        for row in m.to_dict('records'):
            label = row['label']
            if label not in {'fall', 'non-fall', 'ssl'}:
                continue
            if float(row['fs_hz']) != 30:
                raise ValueError('Expected canonical 30 Hz public data')
            if label == 'ssl' and domain != 'PAMAP2':
                raise ValueError('Unexpected SSL-only source')
            group = str(row['split_group_id'])
            if not group.startswith(domain+':'):
                raise ValueError('Missing namespaced participant group')
            p = directory/row['output_file']
            rs.append({'path': p, 'key': domain+'/'+row['output_file'], 'domain': domain,
                       'label': int(label == 'fall'), 'ssl_label': -1 if domain == 'PAMAP2' else int(label == 'fall'),
                       'group': group, 'train_only': False})
    return rs


def fingerprint(rs):
    rows = [(r['key'], r['group'], r['label'], r['train_only'], r.get('fixed_split'),
             r['path'].stat().st_size, r['path'].stat().st_mtime_ns) for r in rs]
    return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()


def partition(rs, seed):
    """70/15/15 by group, with at least two held-out public groups when possible."""
    grouped = defaultdict(list)
    for r in rs:
        grouped[r['group']].append(r)
    forced = {g for g, rr in grouped.items() if any(r['train_only'] for r in rr)}
    groups = np.array(sorted(set(grouped)-forced))
    if len(groups) < 5:
        raise ValueError('At least five independent groups are required')
    minimum = 2 if len(groups) >= 8 else 1
    nv = nt = max(minimum, int(round(.15*len(groups))))
    rng = np.random.default_rng(seed)
    classes = {r['label'] for r in rs}
    signatures = {g: tuple(sorted({r['label'] for r in rr})) for g, rr in grouped.items()}
    signature_counts = Counter(signatures[g] for g in groups)
    best = None
    for _ in range(500):
        order = rng.permutation(groups)
        sets = {'val': set(order[:nv]), 'test': set(order[nv:nv+nt]),
                'train': set(order[nv+nt:]) | forced}
        out = {role: [r for r in rs if r['group'] in gg] for role, gg in sets.items()}
        if any({r['label'] for r in rr} != classes for rr in out.values()):
            continue
        # Cogent's ADL-only stairs participants must be represented in every split.
        if any(count >= 3 and any(not any(signatures[g] == sig for g in sets[role])
                                 for role in sets) for sig, count in signature_counts.items()):
            continue
        totals = Counter(r['label'] for r in rs)
        score = sum(abs(sum(r['label'] == y for r in out[role])/totals[y]-fraction)
                    for role, fraction in [('train', .7), ('val', .15), ('test', .15)] for y in classes)
        if best is None or score < best[0]:
            best = (score, out)
    if best is None:
        raise ValueError('Cannot construct class-complete grouped partitions')
    return best[1]


def split_all(rs, path, seed):
    fp = fingerprint(rs)
    lookup = {r['key']: r for r in rs}
    if len(lookup) != len(rs):
        raise ValueError('Duplicate recording keys')
    if path.exists():
        data = json.loads(path.read_text())
        if data.get('fingerprint') != fp or data.get('seed') != seed or data.get('protocol_version') != PROTOCOL_VERSION:
            raise ValueError('Stored split does not match inputs/configuration; use a new work directory')
        splits = {role: [lookup[k] for k in keys] for role, keys in data['splits'].items()}
    else:
        splits = {role: [] for role in ['train', 'val', 'test']}
        for index, domain in enumerate(DOMAINS):
            source = [r for r in rs if r['domain'] == domain]
            if source and any(r.get('fixed_split') for r in source):
                if not all(r.get('fixed_split') in splits for r in source):
                    raise ValueError(f'Incomplete fixed split for {domain}')
                by_group = defaultdict(set)
                for r in source:
                    by_group[r['group']].add(r['fixed_split'])
                if any(len(roles) != 1 for roles in by_group.values()):
                    raise ValueError(f'Group leakage in fixed {domain} split')
                part = {role: [r for r in source if r['fixed_split'] == role] for role in splits}
                if any(not part[role] for role in splits):
                    raise ValueError(f'Fixed {domain} split must include train/val/test')
            else:
                part = partition(source, seed+index*1009)
            for role in splits:
                splits[role].extend(part[role])
        data = {'seed': seed, 'fingerprint': fp, 'protocol_version': PROTOCOL_VERSION,
                'atomic_unit': 'private source recording or public participant; duplicate groups linked; M5 boot groups kept intact',
                'splits': {role: sorted(r['key'] for r in rr) for role, rr in splits.items()}}
        write_json(path, data)
    keys = [set(r['key'] for r in splits[role]) for role in ['train', 'val', 'test']]
    groups = [set(r['group'] for r in splits[role]) for role in ['train', 'val', 'test']]
    if set.union(*keys) != set(lookup) or sum(map(len, keys)) != len(rs):
        raise ValueError('Partitions do not cover input recordings exactly once')
    for i in range(3):
        for j in range(i):
            if keys[i] & keys[j] or groups[i] & groups[j]:
                raise ValueError('Recording/group leakage')
    if any(r['train_only'] for role in ['val', 'test'] for r in splits[role]):
        raise ValueError('Synthetic padded private files must remain training-only')
    return splits, data


def read_array(r):
    d = pd.read_csv(r['path'], usecols=['time_ms', *FEATURES])
    x = d[FEATURES].to_numpy(np.float32)
    t = d.time_ms.to_numpy(np.float64)
    good = np.isfinite(x).all(1) & np.isfinite(t)
    t, x = t[good], x[good]
    if len(x) >= 90 and np.allclose(np.diff(t), 1000/30, atol=.002, rtol=1e-5):
        return Signal(x, [[0, len(x)]])
    # Original V2 negatives can have duplicate timestamps, jitter and pauses.
    # Resample each real segment separately; never fabricate signal across pauses.
    from preprocess_public_v2 import resample
    cuts = np.flatnonzero((np.diff(t) < 0) | (np.diff(t) > 100))+1
    parts, ranges, offset = [], [], 0
    for ii in np.split(np.arange(len(t)), cuts):
        if len(ii) < 2:
            continue
        tt, xx = t[ii], x[ii]
        if np.any(np.diff(tt) == 0):
            unique, inv, counts = np.unique(tt, return_inverse=True, return_counts=True)
            sums = np.zeros((len(unique), 6), np.float64)
            np.add.at(sums, inv, xx)
            tt, xx = unique, sums/counts[:, None]
        if len(tt) < 2:
            continue
        native_fs = 1000/np.median(np.diff(tt))
        if native_fs <= 31.5:
            seconds = (tt-tt[0])/1000.
            grid = np.arange(int(np.floor(seconds[-1]*30+1e-6))+1)/30
            yy = np.stack([np.interp(grid, seconds, xx[:, j]) for j in range(6)], axis=1)
        else:
            yy, _ = resample((tt-tt[0])/1000., xx, native_fs, 30.)
        if len(yy) >= 90:
            parts.append(yy.astype(np.float32)); ranges.append([offset, offset+len(yy)])
            offset += len(yy)
    if not parts:
        raise ValueError('No continuous 3-second real segment: '+r['key'])
    return Signal(np.concatenate(parts), ranges)


def arrays(rs, cache):
    fp = hashlib.sha256((fingerprint(rs)+'segmented_signals_v1').encode()).hexdigest()
    file = cache/(fp+'.npz')
    if file.exists():
        with np.load(file, allow_pickle=False) as z:
            return {str(k): Signal(z[f'x{i}'].copy(), z[f'b{i}'].copy()) for i, k in enumerate(z['keys'])}
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(read_array, rs))
    result = {r['key']: x for r, x in zip(rs, values)}
    temp = file.with_suffix('.tmp.npz')
    payload = {}
    for i, x in enumerate(result.values()):
        payload[f'x{i}'] = x.values; payload[f'b{i}'] = x.ranges
    np.savez(temp, keys=np.array(list(result)), **payload)
    temp.replace(file)
    return result


def fit_normalization(rs, arr):
    stats = {}
    for domain in NORMALIZATION_DOMAINS:
        parts = []
        for r in rs:
            if r.get('normalization_domain', r['domain']) == domain:
                x = arr[r['key']]
                parts.append(x[np.linspace(0, len(x)-1, min(len(x), 3000)).astype(int)])
        z = np.concatenate(parts).astype(np.float64)
        stats[domain] = {'mean': z.mean(0).tolist(), 'std': np.maximum(z.std(0), 1e-6).tolist(),
                         'fit_partition': 'train', 'fit_recordings': len(parts)}
    return stats


def record_weights(rs, target_fraction=None, m5_fraction=0.25):
    """Balance source, class and participant/source-recording, never duration."""
    present = sorted({r['domain'] for r in rs})
    masses = {d: 1/len(present) for d in present}
    if target_fraction is not None and 'own' in present and len(present) > 1:
        if M5 in present:
            if target_fraction <= 0 or m5_fraction <= 0 or target_fraction+m5_fraction >= 1:
                raise ValueError('Private/M5 shares must be positive and leave a public-data share')
            public = [d for d in present if d not in {'own', M5}]
            if not public:
                raise ValueError('At least one public source is required')
            masses = {'own': target_fraction, M5: m5_fraction}
            masses.update({d: (1-target_fraction-m5_fraction)/len(public) for d in public})
        else:
            masses = {d: (target_fraction if d == 'own' else (1-target_fraction)/(len(present)-1)) for d in present}
    domain_classes = {d: {r['label'] for r in rs if r['domain'] == d} for d in present}
    class_groups = defaultdict(set)
    sizes = Counter()
    for r in rs:
        key = (r['domain'], r['label'])
        class_groups[key].add(r['group'])
        sizes[(*key, r['group'])] += 1
    return np.array([masses[r['domain']]/len(domain_classes[r['domain']])/
                     len(class_groups[(r['domain'], r['label'])])/
                     sizes[(r['domain'], r['label'], r['group'])] for r in rs], np.float64)


def class_loss_weights(rs, source_weights):
    """Normalize the expected class contribution implied by the source sampler."""
    proportions = [sum(float(w) for r, w in zip(rs, source_weights) if r['label'] == y)
                   for y in (0, 1)]
    if min(proportions) <= 0:
        raise ValueError('Fine-tuning requires fall and non-fall examples')
    return np.asarray([1/(2*p) for p in proportions], np.float32)


def small_rotation():
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)+1e-12
    angle = np.random.uniform(-np.pi/9, np.pi/9)
    a, b, c = axis
    skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def evaluation_starts(lo, hi, window, fs, stride_sec):
    """Round the cumulative time grid, retaining the final complete window."""
    if not math.isfinite(stride_sec) or stride_sec <= 0:
        raise ValueError('Evaluation stride must be finite and positive')
    last = hi-window
    if last < lo:
        return np.empty(0, np.int64)
    offsets = np.rint(np.arange(0, last-lo+1e-9, max(1., stride_sec*fs))).astype(np.int64)
    return np.unique(np.r_[np.clip(lo+offsets, lo, last), last])


class Clips(Dataset):
    def __init__(self, rs, arr, stats, cfg, random_crop=False, two=False, smoke=False):
        self.rs, self.arr, self.cfg = rs, arr, cfg
        self.random_crop, self.two = random_crop, two
        self.stats = {d: (np.array(s['mean'], np.float32), np.array(s['std'], np.float32)) for d, s in stats.items()}
        self.w = int(round(cfg.win_sec*cfg.fs))
        self.entries = []
        if random_crop:
            self.entries = [(i, -1) for i in range(len(rs))]
        else:
            for i, r in enumerate(rs):
                ss = np.concatenate([evaluation_starts(lo, hi, self.w, cfg.fs, cfg.stride_sec)
                                     for lo, hi in signal_ranges(arr[r['key']])])
                # Production validation/test cover every start; only smoke tests subsample.
                if smoke and len(ss) > 4:
                    ss = ss[np.linspace(0, len(ss)-1, 4).round().astype(int)]
                self.entries.extend((i, int(s)) for s in ss)

    def __len__(self):
        return len(self.entries)

    def view(self, r, start, augment):
        raw = self.arr[r['key']]
        if start < 0:
            ranges = signal_ranges(raw)
            counts = ranges[:, 1]-ranges[:, 0]-self.w+1
            index = int(np.random.randint(0, int(counts.sum())))
            cumulative = np.cumsum(counts)
            segment = int(np.searchsorted(cumulative, index, side='right'))
            previous = int(cumulative[segment-1]) if segment else 0
            start = int(ranges[segment, 0]+index-previous)
        x = raw[start:start+self.w].copy()
        if augment:
            if np.random.rand() < .5:
                rotation = small_rotation()
                x[:, :3] = x[:, :3]@rotation.T
                x[:, 3:] = x[:, 3:]@rotation.T
            x[:, :3] *= np.random.uniform(.9, 1.1)
            x[:, 3:] *= np.random.uniform(.9, 1.1)
        mean, std = self.stats[r.get('normalization_domain', r['domain'])]
        x = ((x-mean)/std).astype(np.float32)
        if augment:
            x += np.random.normal(0, .015, x.shape).astype(np.float32)
        # Fixed temporal thirds: attention learns event position, without detecting it.
        n = self.w//3
        return [torch.from_numpy(a.copy()) for a in [x, x[:n], x[n:2*n], x[2*n:3*n]]], start

    def __getitem__(self, index):
        i, start = self.entries[index]
        r = self.rs[i]
        xx, sampled = self.view(r, start, self.random_crop or self.two)
        out = dict(zip(['full', 'pre', 'impact', 'post'], xx))
        out.update(label=r['ssl_label'] if self.two else r['label'], file_label=r['label'],
                   phase=-1, key=r['key'], domain=r['domain'], start=sampled)
        if self.two:
            second, _ = self.view(r, sampled, True)
            out.update(dict(zip(['full2', 'pre2', 'impact2', 'post2'], second)))
        return out


def train_loader(ds, cfg, ssl=False):
    sampler = WeightedRandomSampler(record_weights(ds.rs, None if ssl else cfg.target_fraction,
                                                   getattr(cfg, 'm5_fraction', 0.25)),
                                    cfg.batch*cfg.steps_per_epoch, replacement=True)
    return DataLoader(ds, batch_size=cfg.batch, sampler=sampler, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def evaluation_records(rs, smoke):
    if not smoke:
        return rs
    out = []
    for d in DOMAINS:
        for y in [0, 1]:
            out.extend([r for r in rs if r['domain'] == d and r['label'] == y][:4])
    return out


def validation(model, ds, cfg, dev, base):
    ld = DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=0,
                    pin_memory=dev.type == 'cuda')
    rows = base.predict(model, ld, dev)
    result, details = {}, {}
    for domain in DOMAINS:
        rr = [r for r in rows if r['key'].startswith(domain+'/')]
        if not rr:
            continue
        keys, y, p = base.aggregate(rr)
        ap = float(average_precision_score(y, p)) if len(set(y)) == 2 else None
        result[domain] = {'ap': ap, 'recordings': len(keys)}
        details[domain] = (rr, keys, y, p)
    return result, details


def selection_score(metrics, threshold, minimum_private_recall, public_recall_floor):
    """Prefer fewer device false-positive windows subject to validation recall gates."""
    own = metrics['details']['own']
    private = metrics['details'][M5]
    own_rows, own_y, own_p = own[0], own[2], own[3]
    m5_rows = private[0]
    own_recall = float(np.sum((np.asarray(own_p) >= threshold) & (np.asarray(own_y) == 1)) /
                       max(1, int(np.sum(np.asarray(own_y) == 1))))
    pub_recall = []
    for domain in PUBLIC:
        if domain == 'PAMAP2':
            continue
        _, _, y, p = metrics['details'][domain]
        positives = np.asarray(y) == 1
        pub_recall.append(float(np.mean(np.asarray(p)[positives] >= threshold)) if positives.any() else 1.)
    own_neg = [r for r in own_rows if r['label'] == 0]
    m5_neg = [r for r in m5_rows if r['label'] == 0]
    if not own_neg or not m5_neg:
        raise ValueError('Validation needs Private V2 and M5 negative windows')
    own_fpr = sum(r['prob'] >= threshold for r in own_neg)/len(own_neg)
    m5_fpr = sum(r['prob'] >= threshold for r in m5_neg)/len(m5_neg)
    valid = own_recall + 1e-12 >= minimum_private_recall and (
        not pub_recall or min(pub_recall) + 1e-12 >= public_recall_floor)
    public_ap = [v['ap'] for d, v in metrics['domains'].items()
                 if d in PUBLIC and v['ap'] is not None]
    return (int(valid), -(own_fpr+m5_fpr)/2, metrics['domains']['own']['ap'],
            float(np.mean(public_ap)) if public_ap else 0.), {
                'private_val_recall': own_recall, 'private_negative_window_fpr': own_fpr,
                'm5_negative_window_fpr': m5_fpr,
                'public_val_macro_recall': float(np.mean(pub_recall)) if pub_recall else None,
                'valid_under_recall_gates': valid}


def tune_validation_threshold(metrics, minimum_private_recall, public_recall_floor):
    """Balance Private V2/M5 negative FPR while preserving validation recall."""
    for value in (minimum_private_recall, public_recall_floor):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Validation recall floors must be finite and between zero and one')

    def probabilities(values):
        p = np.asarray(values, np.float64)
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError('Validation probabilities must be finite and between zero and one')
        return np.sort(p)

    details = metrics['details']
    own_rows, _, own_y, own_p = details['own']
    own_positive = probabilities(np.asarray(own_p)[np.asarray(own_y) == 1])
    own_negative = probabilities([r['prob'] for r in own_rows if r['label'] == 0])
    m5_negative = probabilities([r['prob'] for r in details[M5][0] if r['label'] == 0])
    if not len(own_positive) or not len(own_negative) or not len(m5_negative):
        raise ValueError('Threshold tuning needs Private V2 falls and Private V2/M5 negative windows')

    public_positive = []
    for domain in PUBLIC:
        if domain != 'PAMAP2':
            _, _, y, p = details[domain]
            public_positive.append(probabilities(np.asarray(p)[np.asarray(y) == 1]))

    # Exact decision boundaries avoid a coarse probability grid. Searchsorted
    # computes all rates without repeatedly scanning thousands of windows.
    candidates = np.unique(np.clip(np.concatenate([
        np.array([0., 1.]), own_positive, own_negative, m5_negative,
        np.nextafter(own_negative, np.inf), np.nextafter(m5_negative, np.inf),
        *public_positive]), 0., 1.))

    def positive_rate(p):
        return (len(p)-np.searchsorted(p, candidates, side='left'))/len(p)

    own_recall = positive_rate(own_positive)
    public_recall = np.stack([positive_rate(p) if len(p) else np.ones(len(candidates))
                              for p in public_positive])
    own_fpr, m5_fpr = positive_rate(own_negative), positive_rate(m5_negative)
    eligible = np.flatnonzero((own_recall+1e-12 >= minimum_private_recall) &
                             (public_recall.min(axis=0)+1e-12 >= public_recall_floor))
    if not len(eligible):
        raise ValueError('No threshold satisfies the validation recall constraints')
    mean_fpr = .5*own_fpr+.5*m5_fpr
    # On equal negative rejection, preserve more falls before preferring the
    # higher threshold. Test predictions never enter this calculation.
    order = np.lexsort((candidates[eligible], public_recall.mean(axis=0)[eligible],
                        own_recall[eligible], -mean_fpr[eligible]))
    threshold = float(candidates[eligible[order[-1]]])
    score, gates = selection_score(metrics, threshold, minimum_private_recall, public_recall_floor)
    gates.update(threshold_policy='joint_private_m5_validation_v1',
                 private_negative_weight=.5, m5_negative_weight=.5,
                 minimum_private_recall=minimum_private_recall,
                 public_recall_floor=public_recall_floor,
                 threshold_candidates=len(candidates))
    return threshold, score, gates


def amp_context(dev):
    return torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def progress(work, stage, epoch, epochs, started, total_done, total_epochs, values):
    elapsed = time.monotonic()-started
    eta = elapsed/max(total_done, 1)*(total_epochs-total_done)
    result = {'status': 'training', 'stage': stage, 'epoch': epoch, 'stage_epochs': epochs,
              'elapsed_seconds': round(elapsed, 1), 'estimated_remaining_seconds': round(eta, 1), **values}
    write_json(work/'progress.json', result)
    print(json.dumps(result), flush=True)


def ssl_train(model, ds, valds, cfg, dev, base, work, history, started, total_epochs):
    ld = train_loader(ds, cfg, ssl=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=cfg.wd)
    best, best_loss = None, float('inf')
    # Stable validation pairs without influencing the training RNG stream.
    for epoch in range(cfg.ssl_epochs):
        model.train(); losses = []
        for batch in ld:
            y = batch['label'].to(dev)
            with amp_context(dev):
                a = model(*base.batch_x(batch, dev))
                b = model(*base.batch_x(batch, dev, '2'))
            loss = cfg.ssl_inst_w*base.ntxent(a['proj'].float(), b['proj'].float(), cfg.temp)
            loss += cfg.ssl_supcon_w*base.supcon(torch.cat([a['proj'].float(), b['proj'].float()]), torch.cat([y, y]), cfg.temp)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SSL loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        rng = np.random.get_state(); np.random.seed(cfg.seed+10000)
        model.eval(); vals = []
        with torch.no_grad():
            for batch in DataLoader(valds, batch_size=cfg.batch, shuffle=False):
                with amp_context(dev):
                    a = model(*base.batch_x(batch, dev)); b = model(*base.batch_x(batch, dev, '2'))
                vals.append(float(base.ntxent(a['proj'].float(), b['proj'].float(), cfg.temp)))
        np.random.set_state(rng)
        value = float(np.mean(vals))
        if value < best_loss:
            best_loss, best = value, cpu_state(model)
        row = {'stage': 'ssl', 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'val_ntxent': value}
        history.append(row)
        progress(work, 'ssl', epoch+1, cfg.ssl_epochs, started, epoch+1, total_epochs, row)
    if best is not None:
        model.load_state_dict(best)
    torch.save({'model': cpu_state(model), 'config': asdict(cfg), 'label_mode': 'clip'}, work/'checkpoints/ssl_pretrained.pt')


def supervised_train(model, ds, valds, cfg, dev, base, work, history, started, total_epochs):
    ld = train_loader(ds, cfg)
    baseline = torch.load(ds.baseline_checkpoint, map_location='cpu', weights_only=False)
    baseline_model = base.Net(cfg).to(dev)
    baseline_model.load_state_dict(baseline['model'])
    baseline_model.eval()
    minimum_private_recall = 1.0
    initial_threshold, baseline_domains, baseline_details, _, _ = validation_candidate(
        baseline_model, valds, cfg, dev, base, baseline['threshold'])
    _, _, own_y, own_p = baseline_details['own']
    minimum_private_recall = float(base.metrics(own_y, own_p, initial_threshold)['recall'])
    public_recalls = []
    for domain in PUBLIC:
        if domain == 'PAMAP2' or domain not in baseline_details:
            continue
        _, _, y, p = baseline_details[domain]
        positive = np.asarray(y) == 1
        if positive.any():
            public_recalls.append(float(np.mean(np.asarray(p)[positive] >= initial_threshold)))
    public_recall_floor = max(0., min(public_recalls)-.02) if public_recalls else 0.
    initial_threshold, baseline_score, baseline_gates = tune_validation_threshold(
        {'domains': baseline_domains, 'details': baseline_details},
        minimum_private_recall, public_recall_floor)
    best, best_score, best_threshold = baseline['model'], baseline_score, initial_threshold
    best_metrics = {'domains': baseline_domains, 'selection': baseline_gates,
                    'baseline_fallback': True, 'threshold': initial_threshold}
    baseline_model.to('cpu')
    del baseline_model
    bad = 0; done = cfg.ssl_epochs
    source_weights = record_weights(ds.rs, cfg.target_fraction, cfg.m5_fraction)
    class_weights = torch.as_tensor(class_loss_weights(ds.rs, source_weights), dtype=torch.float32, device=dev)
    for stage, epochs, lr, encoder in [('head', cfg.head_epochs, cfg.head_lr, False), ('all', cfg.all_epochs, cfg.all_lr, True)]:
        for parameter in list(model.acc.parameters())+list(model.gyr.parameters()):
            parameter.requires_grad = encoder
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.wd)
        for epoch in range(epochs):
            model.train()
            if not encoder:
                model.acc.eval(); model.gyr.eval()  # frozen BN statistics too
            losses = []
            for batch in ld:
                y = batch['label'].to(dev)
                with amp_context(dev):
                    o = model(*base.batch_x(batch, dev))
                    ce = F.cross_entropy(o['logits'], y, weight=class_weights, reduction='none').mean()
                loss = ce.float()+cfg.supcon_w*base.supcon(o['proj'].float(), y, cfg.temp)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite classification loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); losses.append(float(loss.detach()))
            th, domains, _, score, selection = validation_candidate(
                model, valds, cfg, dev, base,
                minimum_private_recall=minimum_private_recall,
                public_recall_floor=public_recall_floor)
            score = tuple(score)
            if score > tuple(best_score):
                best_score, best, best_threshold, best_metrics, bad = score, cpu_state(model), th, {
                    'domains': domains, 'selection': selection, 'baseline_fallback': False,
                    'threshold': th}, 0
            else:
                bad += 1
            row = {'stage': stage, 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'selection_score': list(score),
                   'validation': domains, 'selection': selection, 'threshold': th}
            history.append(row); done += 1
            progress(work, stage, epoch+1, epochs, started, done, total_epochs, row)
            if stage == 'all' and cfg.patience and bad >= cfg.patience:
                print('Early stopping on validation only', flush=True)
                break
        bad = 0
    if best is None:
        raise ValueError('At least one supervised epoch is required')
    model.load_state_dict(best)
    torch.save({'model': best, 'config': asdict(cfg), 'threshold': best_threshold,
                'checkpoint_selection': best_metrics}, work/'checkpoints/best_model.pt')
    return best_metrics


def retrieval(model, train_ds, query_ds, cfg, dev, base):
    """Class retrieval on held-out private clips, with a fixed training-only gallery."""
    def encode(ds):
        embeddings, labels, keys = [], [], []
        model.eval()
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=cfg.batch):
                embeddings.append(model(*base.batch_x(b, dev))['proj'].cpu().numpy())
                labels.extend(np.asarray(b['file_label']).tolist()); keys.extend(b['key'])
        z = np.concatenate(embeddings)
        unique = sorted(set(keys))
        vectors = np.stack([z[np.array(keys) == k].mean(0) for k in unique])
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        y = np.array([labels[keys.index(k)] for k in unique])
        return vectors, y
    gallery, gy = encode(train_ds); queries, qy = encode(query_ds)
    similarity = queries@gallery.T
    order = np.argsort(-similarity, axis=1)
    relevant = gy[order] == qy[:, None]
    ranks = np.arange(1, len(gy)+1)
    precision = np.cumsum(relevant, axis=1)/ranks
    aps = (precision*relevant).sum(1)/np.maximum(relevant.sum(1), 1)
    return {'task': 'binary-class retrieval, not event-instance retrieval', 'gallery_recordings': len(gy),
            'query_recordings': len(qy), 'precision_at_1': float(relevant[:, 0].mean()),
            'hit_rate_at_5': float(relevant[:, :min(5, len(gy))].any(1).mean()), 'mAP': float(aps.mean())}


def input_paths(args, work, base):
    own = Path(args.zip)
    if not own.exists() and own.with_suffix('.zip').exists():
        own = own.with_suffix('.zip')
    root = base.extract(own, work)
    public = args.public_root or Path(base.__file__).resolve().parent.parent/'fd_datasets/processed_v2'
    m5 = args.m5_root or Path(base.__file__).resolve().parent.parent/'fd_datasets/M5_hard_negatives_20261002'
    return root, Path(public), Path(m5)


def validation_candidate(model, valds, cfg, dev, base, threshold=None,
                         minimum_private_recall=0., public_recall_floor=0.):
    domains, details = validation(model, valds, cfg, dev, base)
    if 'own' not in details or M5 not in details:
        raise ValueError('Validation requires Private V2 and M5 records')
    metrics = {'domains': domains, 'details': details}
    if threshold is None:
        threshold, score, gates = tune_validation_threshold(
            metrics, minimum_private_recall, public_recall_floor)
    else:
        score, gates = selection_score(metrics, threshold,
                                       minimum_private_recall, public_recall_floor)
    return float(threshold), domains, details, score, gates


def run(args, cfg, base):
    if cfg.fs != 30 or cfg.win_sec != 3 or cfg.label_mode != 'clip':
        raise ValueError('V2 protocol requires 30 Hz / 3-second / clip labels')
    if not math.isfinite(cfg.stride_sec) or cfg.stride_sec <= 0:
        raise ValueError('Evaluation stride must be finite and positive')
    if (cfg.target_fraction <= 0 or cfg.m5_fraction <= 0 or
            cfg.target_fraction+cfg.m5_fraction >= 1 or cfg.steps_per_epoch < 1 or cfg.batch < 2):
        raise ValueError('Invalid sampler configuration')
    base.seed_all(cfg.seed)
    torch.set_num_threads(4)
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    for folder in ['checkpoints', 'results', 'splits', 'cache']:
        (work/folder).mkdir(exist_ok=True)
    cache_root = Path(args.cache_root).resolve() if getattr(args, 'cache_root', None) else work/'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    own, public, m5 = input_paths(args, work, base)
    rs = own_records(own)+m5_records(m5)+public_records(public)
    splits, splitmeta = split_all(rs, work/'splits/group_splits.json', cfg.seed)
    audit = {'protocol_version': PROTOCOL_VERSION, 'seed': cfg.seed,
             'label_policy': 'whole fall container positive; every 3-second view inherits its clip label',
             'sampler_policy': 'SSL equal seven sources; fine-tune 25% Private V2, 25% M5 hard negatives, 10% per public source; class/group balanced',
             'target_fraction': cfg.target_fraction, 'm5_fraction': cfg.m5_fraction,
             'split': {}, 'group_leakage': False,
             'test_used_for_training_or_selection': False, 'private_eval': 'same-person unseen-source-recording evaluation',
             'pamap_policy': 'unlabelled SSL; known activities are negatives in supervised stages'}
    for domain in DOMAINS:
        audit['split'][domain] = {role: {'groups': len({r['group'] for r in rr if r['domain'] == domain}),
                                        'recordings': sum(r['domain'] == domain for r in rr),
                                        'positives': sum(r['domain'] == domain and r['label'] == 1 for r in rr)}
                                  for role, rr in splits.items()}
    write_json(work/'data_audit.json', audit)
    print('GROUPED DATA AUDIT', json.dumps(audit['split']), flush=True)
    if args.mode == 'prepare':
        arr = arrays(splits['train']+splits['val'], cache_root)
        write_json(work/'normalization.json', fit_normalization(splits['train'], arr))
        print('PREPARATION AND LEAKAGE CHECKS PASSED', flush=True)
        return
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.mode == 'test':
        checkpoint = torch.load(work/'checkpoints/locked_model.pt', map_location='cpu', weights_only=False)
        cfg = base.Cfg(**checkpoint['config'])
        if checkpoint['split_fingerprint'] != splitmeta['fingerprint']:
            raise ValueError('Locked model/input fingerprint mismatch')
        stats = checkpoint['normalization']
        if any(s['fit_partition'] != 'train' for s in stats.values()):
            raise ValueError('Invalid normalization provenance')
        model = base.Net(cfg).to(dev); model.load_state_dict(checkpoint['model'])
        arr = arrays(splits['test'], cache_root)
        selected = evaluation_records(splits['test'], args.smoke)
        testds = Clips(selected, arr, stats, cfg, smoke=args.smoke)
        _, detail = validation(model, testds, cfg, dev, base)
        result = {'threshold': checkpoint['threshold'], 'evaluation': 'locked fine-tune checkpoint; no test tuning',
                  'private_test_scope': audit['private_eval'], 'domains': {}}
        table = []
        for domain, (rows, keys, y, p) in detail.items():
            metric = base.metrics(y, p, checkpoint['threshold'])
            metric['ap'] = float(average_precision_score(y, p)) if len(set(y)) == 2 else None
            metric['window'] = base.window_metrics(rows, checkpoint['threshold'])
            if len(set(y)) < 2:
                metric['recall'] = None; metric['f2'] = None
            result['domains'][domain] = metric
            table.append({'dataset': domain, **{k: v for k, v in metric.items() if k != 'window'}})
            pd.DataFrame(rows).to_csv(work/f'results/{domain}_test_windows.csv', index=False)
            pd.DataFrame({'key': keys, 'label': y, 'probability': p}).to_csv(work/f'results/{domain}_test_recordings.csv', index=False)
        gallery_records = evaluation_records([r for r in splits['train'] if r['domain'] == 'own'], args.smoke)
        gallery_arr = arrays(gallery_records, cache_root)
        gallery = Clips(gallery_records, gallery_arr, stats, cfg, smoke=args.smoke)
        query = Clips([r for r in selected if r['domain'] == 'own'], arr, stats, cfg, smoke=args.smoke)
        result['own_class_retrieval'] = retrieval(model, gallery, query, cfg, dev, base)
        write_json(work/'results/final_test_metrics.json', result)
        pd.DataFrame(table).to_csv(work/'results/test_summary.csv', index=False)
        write_json(work/'progress.json', {'status': 'complete', 'mode': 'test', 'results': str(work/'results/final_test_metrics.json')})
        print('FINAL TEST', json.dumps(result), flush=True)
        return
    if (work/'checkpoints/locked_model.pt').exists():
        raise FileExistsError('A locked model exists; use a new work directory for another training run')
    arr = arrays(splits['train']+splits['val'], cache_root)
    if not args.normalization_from or not args.init_ssl or not args.baseline_checkpoint:
        raise ValueError('Adaptation requires --normalization-from, --init-ssl, and --baseline-checkpoint')
    normalization_checkpoint = torch.load(args.normalization_from, map_location='cpu', weights_only=False)
    stats = normalization_checkpoint['normalization']
    if any(stats.get(d, {}).get('fit_partition') != 'train' for d in NORMALIZATION_DOMAINS):
        raise ValueError('Warm-start normalization must be train-only for all six model input domains')
    init_checkpoint = torch.load(args.init_ssl, map_location='cpu', weights_only=False)
    write_json(work/'normalization.json', stats); write_json(work/'run_config.json', asdict(cfg))
    trainrs = splits['train']
    valrs = evaluation_records(splits['val'], args.smoke)
    if not any(r['domain'] == 'own' and r['label'] == 1 for r in trainrs) or not any(r['domain'] == M5 for r in trainrs):
        raise ValueError('Training needs Private V2 falls and M5 hard negatives')
    train_ds = Clips(trainrs, arr, stats, cfg, random_crop=True)
    train_ds.baseline_checkpoint = args.baseline_checkpoint
    ssl_ds = Clips(trainrs, arr, stats, cfg, random_crop=True, two=True)
    valds = Clips(valrs, arr, stats, cfg, smoke=args.smoke)
    # A small fixed set of held-out clips is sufficient for SSL validation loss.
    ssl_val = evaluation_records(splits['val'], True)
    ssl_val_ds = Clips(ssl_val, arr, stats, cfg, two=True, smoke=True)
    model = base.Net(cfg).to(dev)
    init_config = init_checkpoint.get('config', {})
    if init_config.get('channels') != cfg.channels or init_config.get('blocks') != cfg.blocks:
        raise ValueError('SSL initialization model shape differs from the requested model')
    model.load_state_dict(init_checkpoint['model'])
    print('INITIALIZATION', str(args.init_ssl), 'normalization', str(args.normalization_from), flush=True)
    class_w = class_loss_weights(trainrs, record_weights(trainrs, cfg.target_fraction, cfg.m5_fraction))
    audit['expected_class_share'] = {
        'non_fall': float(sum(w for r, w in zip(trainrs, record_weights(trainrs, cfg.target_fraction, cfg.m5_fraction)) if r['label'] == 0)),
        'fall': float(sum(w for r, w in zip(trainrs, record_weights(trainrs, cfg.target_fraction, cfg.m5_fraction)) if r['label'] == 1)),
        'loss_weights_non_fall_fall': class_w.tolist()}
    write_json(work/'data_audit.json', audit)
    print('MODEL', dev, 'parameters', sum(p.numel() for p in model.parameters()), 'batch', cfg.batch, flush=True)
    history = []; started = time.monotonic()
    epochs = cfg.ssl_epochs+cfg.head_epochs+cfg.all_epochs
    ssl_train(model, ssl_ds, ssl_val_ds, cfg, dev, base, work, history, started, epochs)
    chosen = supervised_train(model, train_ds, valds, cfg, dev, base, work, history, started, epochs)
    _, details = validation(model, valds, cfg, dev, base)
    _, _, y, p = details['own']
    threshold = float(chosen['threshold'])
    vm = base.metrics(y, p, threshold)
    selected = chosen['selection']
    checkpoint = {'model': cpu_state(model), 'config': asdict(cfg), 'threshold': threshold,
                  'normalization': stats, 'split_fingerprint': splitmeta['fingerprint'],
                  'checkpoint_selection': chosen, 'label_policy': audit['label_policy'],
                  'initialization': str(args.init_ssl), 'normalization_source': str(args.normalization_from),
                  'selection_threshold_gates': selected,
                  'feature_context': 'fixed temporal thirds; no peak finding or pseudo-phase labels',
                  'smoke': args.smoke}
    torch.save(checkpoint, work/'checkpoints/locked_model.pt')
    torch.save(checkpoint, work/'checkpoints/best_model.pt')
    write_json(work/'training_history.json', history)
    write_json(work/'results/validation_metrics.json', {'threshold': threshold, 'own': vm, 'domains': chosen})
    for domain, (rows, keys, y, p) in details.items():
        pd.DataFrame(rows).to_csv(work/f'results/{domain}_val_windows.csv', index=False)
    write_json(work/'progress.json', {'status': 'trained', 'elapsed_seconds': round(time.monotonic()-started, 1),
                                     'test_evaluated': False, 'threshold': threshold})
    print('TRAINING COMPLETE; CHECKPOINT/THRESHOLD LOCKED; TEST NOT EVALUATED', flush=True)
