"""V2 clip supervision, grouped evaluation and six-source target adaptation.

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
DOMAINS = ['own', *PUBLIC]
FEATURES = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
PROTOCOL_VERSION = 2


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
    rows = [(r['key'], r['group'], r['label'], r['train_only'],
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
            part = partition([r for r in rs if r['domain'] == domain], seed+index*1009)
            for role in splits:
                splits[role].extend(part[role])
        data = {'seed': seed, 'fingerprint': fp, 'protocol_version': PROTOCOL_VERSION,
                'atomic_unit': 'private source recording or public participant; duplicate groups linked',
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
    for domain in DOMAINS:
        parts = []
        for r in rs:
            if r['domain'] == domain:
                x = arr[r['key']]
                parts.append(x[np.linspace(0, len(x)-1, min(len(x), 3000)).astype(int)])
        z = np.concatenate(parts).astype(np.float64)
        stats[domain] = {'mean': z.mean(0).tolist(), 'std': np.maximum(z.std(0), 1e-6).tolist(),
                         'fit_partition': 'train', 'fit_recordings': len(parts)}
    return stats


def record_weights(rs, target_fraction=None):
    """Balance source, class and participant/source-recording, never duration."""
    present = sorted({r['domain'] for r in rs})
    masses = {d: 1/len(present) for d in present}
    if target_fraction is not None and 'own' in present and len(present) > 1:
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


def small_rotation():
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)+1e-12
    angle = np.random.uniform(-np.pi/9, np.pi/9)
    a, b, c = axis
    skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


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
                ss = np.concatenate([np.unique(np.r_[np.arange(lo, hi-self.w+1, max(1, round(cfg.stride_sec*cfg.fs))), hi-self.w])
                                     for lo, hi in signal_ranges(arr[r['key']])])
                # Fixed validation sampling; test runs use the full window grid.
                limit = 4 if smoke else 32
                if len(ss) > limit:
                    ss = ss[np.linspace(0, len(ss)-1, limit).round().astype(int)]
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
        mean, std = self.stats[r['domain']]
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
    sampler = WeightedRandomSampler(record_weights(ds.rs, None if ssl else cfg.target_fraction),
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


def selection_score(metrics):
    own = metrics['own']['ap']
    public = [v['ap'] for d, v in metrics.items() if d != 'own' and v['ap'] is not None]
    # Target first; public macro-AP is a small regularizer/tie breaker.
    return .8*own+.2*float(np.mean(public)) if public else own


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
    best, best_score, best_metrics = None, -float('inf'), None
    bad = 0; done = cfg.ssl_epochs
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
                    ce = F.cross_entropy(o['logits'], y)
                loss = ce.float()+cfg.supcon_w*base.supcon(o['proj'].float(), y, cfg.temp)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite classification loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); losses.append(float(loss.detach()))
            met, _ = validation(model, valds, cfg, dev, base)
            score = selection_score(met)
            if score > best_score+1e-5:
                best_score, best, best_metrics, bad = score, cpu_state(model), met, 0
            else:
                bad += 1
            row = {'stage': stage, 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'selection_score': score,
                   'validation': met}
            history.append(row); done += 1
            progress(work, stage, epoch+1, epochs, started, done, total_epochs, row)
            if stage == 'all' and cfg.patience and bad >= cfg.patience:
                print('Early stopping on validation only', flush=True)
                break
        bad = 0
    if best is None:
        raise ValueError('At least one supervised epoch is required')
    model.load_state_dict(best)
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
    return root, Path(public)


def run(args, cfg, base):
    if cfg.fs != 30 or cfg.win_sec != 3 or cfg.label_mode != 'clip':
        raise ValueError('V2 protocol requires 30 Hz / 3-second / clip labels')
    if not 0 < cfg.target_fraction < 1 or cfg.steps_per_epoch < 1 or cfg.batch < 2:
        raise ValueError('Invalid sampler configuration')
    base.seed_all(cfg.seed)
    torch.set_num_threads(4)
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    for folder in ['checkpoints', 'results', 'splits', 'cache']:
        (work/folder).mkdir(exist_ok=True)
    own, public = input_paths(args, work, base)
    rs = own_records(own)+public_records(public)
    splits, splitmeta = split_all(rs, work/'splits/group_splits.json', cfg.seed)
    audit = {'protocol_version': PROTOCOL_VERSION, 'seed': cfg.seed,
             'label_policy': 'whole fall container positive; every 3-second view inherits its clip label',
             'sampler_policy': 'SSL equal six sources; fine-tune target-weighted source/class/group balancing',
             'target_fraction': cfg.target_fraction, 'split': {}, 'group_leakage': False,
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
        arr = arrays(splits['train']+splits['val'], work/'cache')
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
        arr = arrays(splits['test'], work/'cache')
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
        gallery_arr = arrays(gallery_records, work/'cache')
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
    arr = arrays(splits['train']+splits['val'], work/'cache')
    stats = fit_normalization(splits['train'], arr)
    write_json(work/'normalization.json', stats); write_json(work/'run_config.json', asdict(cfg))
    trainrs = splits['train']
    valrs = evaluation_records(splits['val'], args.smoke)
    train_ds = Clips(trainrs, arr, stats, cfg, random_crop=True)
    ssl_ds = Clips(trainrs, arr, stats, cfg, random_crop=True, two=True)
    valds = Clips(valrs, arr, stats, cfg, smoke=args.smoke)
    # A small fixed set of held-out clips is sufficient for SSL validation loss.
    ssl_val = evaluation_records(splits['val'], True)
    ssl_val_ds = Clips(ssl_val, arr, stats, cfg, two=True, smoke=True)
    model = base.Net(cfg).to(dev)
    print('MODEL', dev, 'parameters', sum(p.numel() for p in model.parameters()), 'batch', cfg.batch, flush=True)
    history = []; started = time.monotonic()
    epochs = cfg.ssl_epochs+cfg.head_epochs+cfg.all_epochs
    ssl_train(model, ssl_ds, ssl_val_ds, cfg, dev, base, work, history, started, epochs)
    chosen = supervised_train(model, train_ds, valds, cfg, dev, base, work, history, started, epochs)
    _, details = validation(model, valds, cfg, dev, base)
    _, _, y, p = details['own']
    threshold, vm = base.tune(y, p, cfg.min_val_recall)
    checkpoint = {'model': cpu_state(model), 'config': asdict(cfg), 'threshold': threshold,
                  'normalization': stats, 'split_fingerprint': splitmeta['fingerprint'],
                  'checkpoint_selection': chosen, 'label_policy': audit['label_policy'],
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
