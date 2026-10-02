"""Reproducible seven-source adaptation with inherited holdouts and audited weights."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Sampler

import training_v2 as v2
from preprocess_public_v2 import UNITS


def source_masses(cfg):
    if not (cfg.target_fraction > 0 and cfg.m5_fraction > 0 and
            cfg.target_fraction + cfg.m5_fraction < 1):
        raise ValueError('Private/M5 weights must leave a positive public share')
    return {'own': cfg.target_fraction, v2.M5: cfg.m5_fraction,
            **{d: (1-cfg.target_fraction-cfg.m5_fraction)/5 for d in v2.PUBLIC}}


def inherited_splits(records, original, destination):
    source = json.loads(Path(original).read_text())
    lookup = {r['key']: r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Duplicate recording keys')
    result = {}
    inherited = []
    for role in ('train', 'val', 'test'):
        keys = source['splits'][role]
        if any(k not in lookup or lookup[k]['domain'] == v2.M5 for k in keys):
            raise ValueError('Original manifest does not match the six original sources')
        inherited.extend(keys)
        result[role] = [lookup[k] for k in keys] + [r for r in records
                         if r['domain'] == v2.M5 and r['fixed_split'] == role]
    if len(inherited) != len(set(inherited)) or set(inherited) != {
            r['key'] for r in records if r['domain'] != v2.M5}:
        raise ValueError('Original manifest must partition all original recordings exactly once')
    memberships = defaultdict(set)
    for role, rr in result.items():
        for r in rr:
            memberships[r['group']].add(role)
            if r['train_only'] and role != 'train':
                raise ValueError('Padded descendants escaped training')
    if any(len(roles) != 1 for roles in memberships.values()):
        raise ValueError('Inherited participant, boot or duplicate-group leakage')
    payload = {'protocol_version': 4, 'original_manifest_sha256': file_hash(original),
               'original_seed': source['seed'], 'original_fingerprint': source['fingerprint'],
               'splits': {k: sorted(r['key'] for r in rr) for k, rr in result.items()},
               'record_groups': {r['key']: r['group'] for r in records},
               'input_fingerprint': v2.fingerprint(records)}
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    destination = Path(destination)
    if destination.exists() and json.loads(destination.read_text()) != payload:
        raise ValueError('Saved v4 split differs from this input; use a new run directory')
    v2.write_json(destination, payload)
    return result, payload


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ancestry_audit(args, splits, original, baseline, initialization):
    expected = json.loads(Path(original).read_text())
    if baseline.get('split_fingerprint') != expected['fingerprint']:
        raise ValueError('Baseline checkpoint does not belong to the inherited manifest')
    for path in (args.init_ssl, args.normalization_from, args.baseline_checkpoint):
        manifest = Path(path).resolve().parent.parent/'splits/group_splits.json'
        if not manifest.is_file() or file_hash(manifest) != file_hash(original):
            raise ValueError('Warm-start ancestor has different holdouts: '+str(path))
    if initialization.get('config', {}).get('fs') != 30:
        raise ValueError('Initialization must use the 30 Hz representation')
    train = splits['train']
    for domain in v2.NORMALIZATION_DOMAINS:
        s = baseline['normalization'][domain]
        if s['fit_partition'] != 'train' or s['fit_recordings'] != sum(r['domain'] == domain for r in train):
            raise ValueError('Normalization provenance mismatch: '+domain)
    return {'status': 'passed', 'original_manifest_sha256': file_hash(original),
            'initialization_sha256': file_hash(args.init_ssl),
            'baseline_sha256': file_hash(args.baseline_checkpoint),
            'original_partitions_preserved': True, 'm5_is_new': True}


class CoverageSampler(Sampler):
    """Cycle source/class/group queues; derive epoch length from complete coverage."""
    def __init__(self, records, batch, masses, seed, smoke=False):
        self.records, self.batch, self.masses, self.seed = records, batch, masses, seed
        self.epoch, self.report = 0, {}
        self.buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, r in enumerate(records):
            self.buckets[r['domain']][r['label']][r['group']].append(i)
        if set(self.buckets) != set(masses):
            raise ValueError('Training requires every source')
        self.domains = list(masses)
        floor = {d: int(math.floor(batch*masses[d])) for d in self.domains}
        if min(floor.values()) < 1:
            raise ValueError('Batch too small to represent all sources')
        required = {d: len(classes)*max(len(groups)*max(map(len, groups.values()))
                    for groups in classes.values()) for d, classes in self.buckets.items()}
        self.steps = 2 if smoke else max(math.ceil(required[d]/floor[d]) for d in self.domains)
        self.smoke = smoke

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch*100003)
        queues, positions, classes, class_position = {}, {}, {}, Counter()
        for d, bucket in self.buckets.items():
            classes[d] = list(rng.permutation(sorted(bucket)))
            for y, groups in bucket.items():
                queues[d, y] = list(rng.permutation(sorted(groups)))
                positions[d, y] = 0
                for g, indices in groups.items():
                    queues[d, y, g] = list(rng.permutation(indices))
                    positions[d, y, g] = 0
        def draw(d):
            y = classes[d][class_position[d] % len(classes[d])]
            class_position[d] += 1
            group_key = (d, y)
            gs = queues[group_key]
            pos = positions[group_key]
            if pos and pos % len(gs) == 0:
                rng.shuffle(gs)
            g = gs[pos % len(gs)]; positions[group_key] += 1
            key = (d, y, g); indices = queues[key]; pos = positions[key]
            if pos and pos % len(indices) == 0:
                rng.shuffle(indices)
            positions[key] += 1
            return indices[pos % len(indices)]
        source_counts, seen, batches, group_counts = Counter(), set(), [], Counter()
        for step in range(self.steps):
            quota = {d: int(math.floor(self.batch*self.masses[d])) for d in self.domains}
            for _ in range(self.batch-sum(quota.values())):
                d = max(self.domains, key=lambda d: (step+1)*self.batch*self.masses[d]-source_counts[d]-quota[d])
                quota[d] += 1
            indices = []
            for d, n in quota.items():
                for _ in range(n):
                    i = draw(d); indices.append(i); seen.add(i)
                    group_counts[self.records[i]['group']] += 1
                source_counts[d] += n
            rng.shuffle(indices); batches.append(indices)
        if not self.smoke and len(seen) != len(self.records):
            raise RuntimeError('Coverage sampler omitted eligible recordings')
        self.report = {'epoch': self.epoch+1, 'steps': self.steps,
                       'crops': self.steps*self.batch, 'unique_recordings': len(seen),
                       'eligible_recordings': len(self.records), 'source_crops': dict(source_counts),
                       'group_crops': dict(group_counts), 'complete_coverage': len(seen) == len(self.records)}
        self.epoch += 1
        return iter(batches)


def source_loss(values, domains, masses):
    result = values.sum()*0
    for d, mass in masses.items():
        mask = torch.tensor([x == d for x in domains], device=values.device)
        if not mask.any():
            raise ValueError('Source absent from weighted batch: '+d)
        result = result+mass*values[mask].mean()
    return result


def instance_losses(a, b, temperature):
    z = torch.cat([a, b]); n = len(a)
    similarity = z@z.T/temperature
    similarity = similarity.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -1e9)
    target = (torch.arange(2*n, device=z.device)+n) % (2*n)
    losses = F.cross_entropy(similarity, target, reduction='none')
    return (losses[:n]+losses[n:])/2


def rate_curve(probabilities, weights, candidates):
    p, w = np.asarray(probabilities), np.asarray(weights)
    if not len(p) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Missing or invalid selection probabilities')
    order = np.argsort(p); p, w = p[order], w[order]
    prefix = np.r_[0., np.cumsum(w)]
    return np.clip((prefix[-1]-prefix[np.searchsorted(p, candidates, side='left')])/prefix[-1], 0., 1.)


def selection_curves(details, records, candidates):
    lookup = {r['key']: r for r in records}
    result = {}
    for d in v2.DOMAINS:
        rows, keys, labels, probs = details[d]
        classes = {}
        for y in (0, 1):
            keys_y = [k for k, yy in zip(keys, labels) if yy == y]
            group_counts = Counter(lookup[k]['group'] for k in keys_y)
            weights = [1/(len(group_counts)*group_counts[lookup[k]['group']]) for k in keys_y]
            by_key = dict(zip(keys, probs))
            if keys_y:
                classes[y] = rate_curve([by_key[k] for k in keys_y], weights, candidates)
        negative = [r for r in rows if r['file_label'] == 0]
        counts = Counter(r['key'] for r in negative)
        groups = Counter(lookup[k]['group'] for k in counts)
        weights = [1/(len(groups)*groups[lookup[r['key']]['group']]*counts[r['key']]) for r in negative]
        fpr = rate_curve([r['prob'] for r in negative], weights, candidates)
        recall = classes.get(1)
        result[d] = {'recall': recall, 'specificity': 1-classes[0], 'negative_window_fpr': fpr,
                     'error': fpr if recall is None else .5*(1-recall)+.5*fpr}
    return result


def select_threshold(details, records, masses, constraints=None, fixed=None):
    if fixed is None:
        values = np.asarray([r['prob'] for value in details.values() for r in value[0]])
        candidates = np.unique(np.clip(np.r_[0., 1., values, np.nextafter(values, np.inf)], 0., 1.))
    else:
        candidates = np.asarray([fixed], float)
    curves = selection_curves(details, records, candidates)
    error = sum(masses[d]*curves[d]['error'] for d in masses)
    valid = np.ones(len(candidates), bool)
    for d, gates in (constraints or {}).items():
        for metric, floor in gates.items():
            values = curves[d][metric]
            if metric == 'negative_window_fpr':
                valid &= values <= floor+1e-12
            else:
                valid &= values+1e-12 >= floor
    eligible = np.flatnonzero(valid)
    accepted = bool(len(eligible))
    if not accepted:
        eligible = np.arange(len(candidates))
    weighted_recall = sum(masses[d]*curves[d]['recall'] for d in masses if curves[d]['recall'] is not None)
    order = np.lexsort((candidates[eligible], weighted_recall[eligible],
                        curves['own']['recall'][eligible], -error[eligible]))
    index = int(eligible[order[-1]])
    result = {'threshold': float(candidates[index]), 'weighted_error': float(error[index]),
              'valid_under_constraints': accepted, 'threshold_candidates': len(candidates),
              'threshold_policy': 'seven_source_group_macro_v4', 'source_weights': masses,
              'domains': {d: {k: None if value is None else float(value[index])
                         for k, value in curves[d].items()} for d in masses}}
    return result


def prediction(model, ds, cfg, dev, base):
    return v2.validation(model, ds, cfg, dev, base)


def test_report(model, records, arr, stats, cfg, dev, base, threshold, output, train_records, cache, smoke):
    ds = v2.Clips(v2.evaluation_records(records, smoke), arr, stats, cfg, smoke=smoke)
    _, details = prediction(model, ds, cfg, dev, base)
    table, domains = [], {}
    for d, (rows, keys, y, p) in details.items():
        metric = base.metrics(y, p, threshold)
        metric['f1'] = 2*metric['tp']/max(1, 2*metric['tp']+metric['fp']+metric['fn'])
        metric['ap'] = float(average_precision_score(y, p)) if len(set(y)) == 2 else None
        metric['window'] = base.window_metrics(rows, threshold)
        if len(set(y)) < 2:
            metric.update(recall=None, precision=None, f1=None, f2=None, mcc=None)
            metric['window'].update(recall=None, precision=None, f2=None, mcc=None)
        domains[d] = metric
        table.append({'dataset': d, **{k: v for k, v in metric.items() if k != 'window'},
                      'negative_window_fpr': sum(r['prob'] >= threshold for r in rows if r['label'] == 0)/
                                             max(1, sum(r['label'] == 0 for r in rows))})
        pd.DataFrame(rows).to_csv(output/f'{d}_test_windows.csv', index=False)
        pd.DataFrame({'key': keys, 'label': y, 'probability': p}).to_csv(output/f'{d}_test_recordings.csv', index=False)
    boots = []
    for r in records:
        if r['domain'] != v2.M5:
            continue
        rows = sorted((row for row in details[v2.M5][0] if row['key'] == r['key']), key=lambda row: row['start'])
        episodes, previous, last = 0, False, None
        for row in rows:
            positive = row['prob'] >= threshold
            if positive and (not previous or last is None or row['start']-last > 8):
                episodes += 1
            previous, last = positive, row['start']
        duration = sum(hi-lo-1 for lo, hi in v2.signal_ranges(arr[r['key']]))/cfg.fs
        boots.append({'key': r['key'], 'boot': r['group'], 'duration_seconds': duration,
                      'negative_windows': len(rows), 'false_positive_windows': sum(row['prob'] >= threshold for row in rows),
                      'false_alarm_episodes': episodes, 'offline_episodes_per_hour': episodes*3600/duration,
                      'episode_policy': 'consecutive positive 0.25 s starts merged; gaps break episodes'})
    gallery_records = v2.evaluation_records([r for r in train_records if r['domain'] == 'own'], smoke)
    gallery_arr = v2.arrays(gallery_records, cache)
    retrieval = v2.retrieval(model, v2.Clips(gallery_records, gallery_arr, stats, cfg, smoke=smoke),
                            v2.Clips([r for r in ds.rs if r['domain'] == 'own'], arr, stats, cfg, smoke=smoke),
                            cfg, dev, base)
    result = {'threshold': threshold, 'domains': domains, 'm5_boots': boots,
              'own_class_retrieval': retrieval, 'zero_shot': None,
              'zero_shot_reason': 'All seven sources contribute training data; no unseen-source test is claimed',
              'evaluation': 'locked validation-selected model and threshold; exhaustive 0.25 s grid unless smoke'}
    v2.write_json(output/'final_test_metrics.json', result)
    pd.DataFrame(table).to_csv(output/'test_summary.csv', index=False)
    return result


def run(args, cfg, base):
    if cfg.fs != 30 or cfg.win_sec != 3 or cfg.stride_sec != .25 or cfg.label_mode != 'clip':
        raise ValueError('v4 requires 30 Hz, 3 s inputs, 0.25 s evaluation and clip labels')
    if not all((args.init_ssl, args.normalization_from, args.baseline_checkpoint)):
        raise ValueError('v4 requires original SSL initialization and baseline normalization checkpoints')
    base.seed_all(cfg.seed); torch.set_num_threads(4)
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoints', 'results', 'splits', 'cache', 'sampler'):
        (work/name).mkdir(exist_ok=True)
    cache = args.cache_root or work/'cache'
    own, public, m5root = v2.input_paths(args, work, base)
    verification = json.loads((m5root/'processed_v2/verification.json').read_text())
    if verification.get('conversion_convention') != 'si' or verification.get('software_clipping') is not False:
        raise ValueError('v4 requires verified full-range M5 SI preprocessing')
    m5 = v2.m5_records(m5root)
    for r in m5:
        r['normalization_domain'] = v2.M5
    records = v2.own_records(own)+m5+v2.public_records(public)
    splits, splitmeta = inherited_splits(records, args.split_from, work/'splits/group_splits.json')
    masses = source_masses(cfg)
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.mode == 'test':
        ckpt = torch.load(work/'checkpoints/locked_model.pt', map_location='cpu', weights_only=False)
        if ckpt['split_fingerprint'] != splitmeta['fingerprint']:
            raise ValueError('Locked checkpoint/input fingerprint mismatch')
        cfg = base.Cfg(**ckpt['config']); stats = ckpt['normalization']
        model = base.Net(cfg).to(dev); model.load_state_dict(ckpt['model'])
        arr = v2.arrays(splits['test'], cache)
        result = test_report(model, splits['test'], arr, stats, cfg, dev, base,
                             ckpt['threshold'], work/'results', splits['train'], cache, args.smoke)
        result['acceptable_improvement'] = ckpt['acceptable_improvement']
        v2.write_json(work/'results/final_test_metrics.json', result)
        v2.write_json(work/'progress.json', {'status': 'complete', 'results': str(work/'results/final_test_metrics.json')})
        print('FINAL TEST', json.dumps(result), flush=True)
        return
    if (work/'checkpoints/locked_model.pt').exists():
        raise FileExistsError('Use a new experiment directory')
    baseline = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=False)
    normalization = torch.load(args.normalization_from, map_location='cpu', weights_only=False)
    initialization = torch.load(args.init_ssl, map_location='cpu', weights_only=False)
    ancestry = ancestry_audit(args, splits, args.split_from, baseline, initialization)
    if normalization['normalization'] != baseline['normalization']:
        raise ValueError('Normalization must belong to the verified original baseline')
    stats = dict(normalization['normalization'])
    arr = v2.arrays(splits['train']+splits['val'], cache)
    m5train = [r for r in splits['train'] if r['domain'] == v2.M5]
    z = np.concatenate([arr[r['key']].values for r in m5train]).astype(np.float64)
    stats[v2.M5] = {'mean': z.mean(0).tolist(), 'std': np.maximum(z.std(0), 1e-6).tolist(),
                    'fit_partition': 'train', 'fit_recordings': len(m5train),
                    'fit_keys': sorted(r['key'] for r in m5train), 'units': verification['units']}
    v2.write_json(work/'normalization.json', stats)
    v2.write_json(work/'warm_start_audit.json', ancestry)
    v2.write_json(work/'run_config.json', {**asdict(cfg), 'protocol_version': 4})
    audit = {'status': 'passed', 'source_weights': masses, 'public_export_units': UNITS,
             'classification_loss': 'sum source weight * mean source CE; no global class reweighting',
             'split': {d: {role: {'recordings': sum(r['domain'] == d for r in rr),
                                  'groups': len({r['group'] for r in rr if r['domain'] == d}),
                                  'falls': sum(r['domain'] == d and r['label'] == 1 for r in rr)}
                            for role, rr in splits.items()} for d in v2.DOMAINS},
             'expected_fall_draw_share': .5*sum(masses[d] for d in masses if d not in (v2.M5, 'PAMAP2')),
             'm5_preprocessing': verification, 'test_used_for_fit_or_selection': False,
             'label_policy': 'whole 5 s fall container positive; random 3 s view inherits label'}
    v2.write_json(work/'data_audit.json', audit)
    print('DATA AUDIT', json.dumps(audit['split']), flush=True)
    if args.mode == 'prepare':
        return
    valrs = v2.evaluation_records(splits['val'], args.smoke)
    valds = v2.Clips(valrs, arr, stats, cfg, smoke=args.smoke)
    legacy_m5 = v2.m5_records(args.legacy_m5_root)
    legacy_lookup = {r['key']: r for r in legacy_m5}
    legacy_val = [legacy_lookup[r['key']] if r['domain'] == v2.M5 else r for r in valrs]
    legacy_arr = dict(arr); legacy_arr.update(v2.arrays([r for r in legacy_val if r['domain'] == v2.M5], cache))
    legacy_ds = v2.Clips(legacy_val, legacy_arr, baseline['normalization'], cfg, smoke=args.smoke)
    baseline_model = base.Net(cfg).to(dev); baseline_model.load_state_dict(baseline['model'])
    _, baseline_details = prediction(baseline_model, legacy_ds, cfg, dev, base)
    original_selection = select_threshold(baseline_details, valrs, masses, fixed=baseline['threshold'])
    initial_gates = {'own': {'recall': original_selection['domains']['own']['recall']}}
    for d in v2.PUBLIC:
        b = original_selection['domains'][d]
        if b['recall'] is not None:
            initial_gates[d] = {'recall': max(0., b['recall']-.02), 'specificity': max(0., b['specificity']-.02)}
    baseline_selection = select_threshold(baseline_details, valrs, masses, initial_gates)
    gates = {'own': {'recall': baseline_selection['domains']['own']['recall']}}
    for d in v2.PUBLIC:
        b = baseline_selection['domains'][d]
        gates[d] = ({'negative_window_fpr': b['negative_window_fpr']} if b['recall'] is None else
                    {'recall': max(0., b['recall']-.02), 'specificity': max(0., b['specificity']-.02)})
    gates[v2.M5] = {'negative_window_fpr': baseline_selection['domains'][v2.M5]['negative_window_fpr']}
    v2.write_json(work/'baseline_validation.json', {'original_threshold': original_selection,
                  'joint_threshold': baseline_selection, 'constraints': gates, 'comparison': 'complete old and new input pipelines'})
    if not args.smoke:
        base_out = work/'baseline_results'; base_out.mkdir(exist_ok=True)
        # Baseline test is deliberately deferred until the candidate has been locked.
    baseline_model.to('cpu'); del baseline_model
    model = base.Net(cfg).to(dev); model.load_state_dict(initialization['model'])
    history, started, best_ssl, ssl_loss, ssl_bad = [], time.monotonic(), None, float('inf'), 0
    ssl_ds = v2.Clips(splits['train'], arr, stats, cfg, random_crop=True, two=True)
    sampler = CoverageSampler(splits['train'], cfg.batch, masses, cfg.seed, args.smoke)
    ssl_loader = DataLoader(ssl_ds, batch_sampler=sampler, num_workers=0, pin_memory=dev.type == 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=cfg.wd)
    total = cfg.ssl_epochs+cfg.head_epochs+cfg.all_epochs
    for epoch in range(cfg.ssl_epochs):
        model.train(); losses = []
        for batch in ssl_loader:
            with v2.amp_context(dev):
                a = model(*base.batch_x(batch, dev)); b = model(*base.batch_x(batch, dev, '2'))
            inst = source_loss(instance_losses(a['proj'].float(), b['proj'].float(), cfg.temp), batch['domain'], masses)
            y = batch['label'].to(dev)
            loss = cfg.ssl_inst_w*inst+cfg.ssl_supcon_w*base.supcon(torch.cat([a['proj'].float(), b['proj'].float()]), torch.cat([y, y]), cfg.temp)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SSL loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); losses.append(float(loss.detach()))
        # Four fixed spatially spread windows from every validation recording,
        # with deterministic augmented pairs. Fine-tune validation remains exhaustive.
        rng = np.random.get_state(); np.random.seed(10042); model.eval(); val_loss = 0.; val_sources = {}
        with torch.no_grad():
            for d, mass in masses.items():
                ssl_val = v2.Clips([r for r in valrs if r['domain'] == d], arr, stats, cfg, two=True, smoke=True)
                numerator, denominator = 0., 0
                for batch in DataLoader(ssl_val, batch_size=cfg.batch):
                    if len(batch['label']) < 2:
                        continue
                    with v2.amp_context(dev):
                        a = model(*base.batch_x(batch, dev)); b = model(*base.batch_x(batch, dev, '2'))
                    value = float(base.ntxent(a['proj'].float(), b['proj'].float(), cfg.temp))
                    numerator += value*len(batch['label']); denominator += len(batch['label'])
                if not denominator:
                    raise ValueError('SSL validation lacks pairs: '+d)
                val_sources[d] = numerator/denominator; val_loss += mass*val_sources[d]
        np.random.set_state(rng)
        if val_loss < ssl_loss-1e-5:
            ssl_loss, best_ssl, ssl_bad = val_loss, v2.cpu_state(model), 0
        else:
            ssl_bad += 1
        row = {'stage': 'ssl', 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'val_ntxent': val_loss,
               'validation_sources': val_sources, 'sampling': {k: v for k, v in sampler.report.items() if k != 'group_crops'}}
        history.append(row); v2.write_json(work/f'sampler/ssl_{epoch+1:03d}.json', sampler.report)
        v2.write_json(work/'training_history.json', history)
        v2.progress(work, 'ssl', epoch+1, cfg.ssl_epochs, started, len(history), total, row)
        if not args.smoke and epoch+1 >= cfg.ssl_min_epochs and ssl_bad >= cfg.ssl_patience:
            break
    model.load_state_dict(best_ssl)
    torch.save({'model': v2.cpu_state(model), 'config': asdict(cfg), 'normalization': stats,
                'split_fingerprint': splitmeta['fingerprint'], 'ancestry': ancestry}, work/'checkpoints/ssl_pretrained.pt')
    train_ds = v2.Clips(splits['train'], arr, stats, cfg, random_crop=True)
    sampler = CoverageSampler(splits['train'], cfg.batch, masses, cfg.seed+20000, args.smoke)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=dev.type == 'cuda')
    best, best_rank, chosen = None, None, None
    for stage, epochs, lr, encoder in [('head', cfg.head_epochs, cfg.head_lr, False), ('all', cfg.all_epochs, cfg.all_lr, True)]:
        for parameter in list(model.acc.parameters())+list(model.gyr.parameters()):
            parameter.requires_grad = encoder
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=cfg.lr_patience, min_lr=1e-6)
        bad, stage_best = 0, None
        for epoch in range(epochs):
            model.train()
            if not encoder:
                model.acc.eval(); model.gyr.eval()
            losses = []
            for batch in loader:
                y = batch['label'].to(dev)
                with v2.amp_context(dev):
                    o = model(*base.batch_x(batch, dev))
                    ce = source_loss(F.cross_entropy(o['logits'], y, reduction='none'), batch['domain'], masses)
                loss = ce.float()+cfg.supcon_w*base.supcon(o['proj'].float(), y, cfg.temp)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite classification loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); losses.append(float(loss.detach()))
            _, details = prediction(model, valds, cfg, dev, base)
            selection = select_threshold(details, valrs, masses, gates)
            rank = (int(selection['valid_under_constraints']), -selection['weighted_error'], selection['domains']['own']['recall'])
            if best_rank is None or rank > best_rank:
                best_rank, best = rank, v2.cpu_state(model)
                chosen = {'stage': stage, 'epoch': epoch+1, **selection}
            if stage_best is None or rank > stage_best:
                stage_best, bad = rank, 0
            else:
                bad += 1
            scheduler.step(selection['weighted_error'])
            row = {'stage': stage, 'epoch': epoch+1, 'loss': float(np.mean(losses)),
                   'learning_rate': optimizer.param_groups[0]['lr'], 'selection': selection,
                   'sampling': {k: v for k, v in sampler.report.items() if k != 'group_crops'}}
            history.append(row); v2.write_json(work/f'sampler/{stage}_{epoch+1:03d}.json', sampler.report)
            v2.write_json(work/'training_history.json', history)
            v2.progress(work, stage, epoch+1, epochs, started, len(history), total, row)
            if stage == 'all' and not args.smoke and epoch+1 >= cfg.min_full_epochs and bad >= cfg.patience:
                break
    if best is None:
        raise ValueError('No supervised checkpoint produced')
    model.load_state_dict(best)
    accepted = chosen['valid_under_constraints'] and chosen['weighted_error'] < baseline_selection['weighted_error']-1e-9
    checkpoint = {'model': best, 'config': asdict(cfg), 'threshold': chosen['threshold'],
                  'normalization': stats, 'split_fingerprint': splitmeta['fingerprint'],
                  'checkpoint_selection': chosen, 'acceptable_improvement': accepted,
                  'ancestry': ancestry, 'source_weights': masses, 'smoke': args.smoke,
                  'preprocessing': verification, 'baseline_validation': baseline_selection}
    torch.save(checkpoint, work/'checkpoints/locked_model.pt')
    torch.save(checkpoint, work/'checkpoints/best_model.pt')
    _, details = prediction(model, valds, cfg, dev, base)
    v2.write_json(work/'results/validation_metrics.json', {'selected': chosen, 'acceptable_improvement': accepted,
                                                        'baseline': baseline_selection, 'constraints': gates})
    for d, value in details.items():
        pd.DataFrame(value[0]).to_csv(work/f'results/{d}_val_windows.csv', index=False)
    v2.write_json(work/'progress.json', {'status': 'trained', 'elapsed_seconds': time.monotonic()-started,
                  'acceptable_improvement': accepted, 'threshold': chosen['threshold'], 'test_evaluated': False})
    print('TRAINED AND LOCKED', json.dumps({'selection': chosen, 'acceptable_improvement': accepted}), flush=True)
