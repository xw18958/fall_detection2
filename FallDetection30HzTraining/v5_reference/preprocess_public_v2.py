#!/usr/bin/env python3
"""Archive-to-CSV public IMU preprocessing using the V2 30 Hz / 5 s / 3 s policy.

No videos or raw archives are extracted. Source units are preserved: undocumented
units are explicitly marked, never guessed. All descendants share a split group.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.util
import hashlib
import io
import json
import math
import os
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

FEATURES = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
DATASETS = ['CGU_BES', 'Cogent', 'SFU_IMU', 'UCI_SimulatedFalls', 'PAMAP2']
ARCHIVES = {
    'CGU_BES': 'CGU_BES/CGU-BES_Dataset.rar',
    'Cogent': 'Cogent/fall_adl_data.zip',
    'SFU_IMU': 'SFU_IMU/SFU-IMU_Dataset.zip',
    'UCI_SimulatedFalls': 'UCI_SimulatedFalls/Tests.rar',
    'PAMAP2': 'PAMAP2/PAMAP2_Dataset.zip',
}
UNITS = {
    'CGU_BES': {'acceleration': 'g', 'gyroscope': 'rad/s',
                'status': 'Wang et al. 2018 Figure 1; original decimal exports corroborated',
                'source': 'https://doi.org/10.1088/1361-6579/aae0eb'},
    'Cogent': {'acceleration': 'g', 'gyroscope': 'deg/s',
               'status': 'Ojetola 2013 Appendix A.1; original CSV excerpt corroborated',
               'source': 'https://pure.coventry.ac.uk/ws/portalfiles/portal/40391885/Ojetola_2013.pdf'},
    'SFU_IMU': {'acceleration': 'm/s^2', 'gyroscope': 'rad/s', 'status': 'README'},
    'UCI_SimulatedFalls': {'acceleration': 'm/s^2', 'gyroscope': 'rad/s',
                          'status': 'Xsens MTw calibrated export'},
    'PAMAP2': {'acceleration': 'm/s^2', 'gyroscope': 'rad/s', 'status': 'README'},
}
SENSORS = {'CGU_BES': 'chest', 'Cogent': 'chest', 'SFU_IMU': 'sternum',
           'UCI_SimulatedFalls': 'chest:340527', 'PAMAP2': 'chest:16g_accelerometer'}


class RarReader:
    """Minimal streaming libarchive reader, including nested RARs in memory."""
    def __init__(self, source):
        libname = ctypes.util.find_library('archive')
        if not libname:
            raise RuntimeError('libarchive is required for the two RAR datasets')
        self.lib = ct.CDLL(libname)
        signatures = {
            'archive_read_new': (ct.c_void_p, []),
            'archive_read_support_filter_all': (ct.c_int, [ct.c_void_p]),
            'archive_read_support_format_all': (ct.c_int, [ct.c_void_p]),
            'archive_read_open_filename': (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_size_t]),
            'archive_read_open_memory': (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_size_t]),
            'archive_read_next_header': (ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_void_p)]),
            'archive_entry_pathname': (ct.c_char_p, [ct.c_void_p]),
            'archive_entry_size': (ct.c_longlong, [ct.c_void_p]),
            'archive_read_data': (ct.c_ssize_t, [ct.c_void_p, ct.c_void_p, ct.c_size_t]),
            'archive_read_data_skip': (ct.c_int, [ct.c_void_p]),
            'archive_error_string': (ct.c_char_p, [ct.c_void_p]),
            'archive_read_free': (ct.c_int, [ct.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            f = getattr(self.lib, name)
            f.restype, f.argtypes = restype, argtypes
        self.handle = self.lib.archive_read_new()
        self.buffer = None
        self.lib.archive_read_support_filter_all(self.handle)
        self.lib.archive_read_support_format_all(self.handle)
        if isinstance(source, bytes):
            self.buffer = ct.create_string_buffer(source)
            status = self.lib.archive_read_open_memory(self.handle, self.buffer, len(source))
        else:
            status = self.lib.archive_read_open_filename(self.handle, os.fsencode(source), 1024*1024)
        self.check(status)

    def check(self, status):
        if status < 0:
            error = self.lib.archive_error_string(self.handle)
            raise RuntimeError(error.decode(errors='replace') if error else f'libarchive status {status}')

    def __iter__(self):
        entry = ct.c_void_p()
        while True:
            status = self.lib.archive_read_next_header(self.handle, ct.byref(entry))
            if status == 1:
                return
            self.check(status)
            name = self.lib.archive_entry_pathname(entry).decode(errors='replace').replace('\\', '/')
            yield name, int(self.lib.archive_entry_size(entry))
            self.check(self.lib.archive_read_data_skip(self.handle))

    def read(self):
        pieces = []
        buffer = ct.create_string_buffer(1024*1024)
        while True:
            n = self.lib.archive_read_data(self.handle, buffer, len(buffer))
            self.check(n)
            if n == 0:
                return b''.join(pieces)
            pieces.append(buffer.raw[:n])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lib.archive_read_free(self.handle)


@dataclass
class Record:
    source: str
    subject: str
    activity: str
    label: str
    fs: float
    t: np.ndarray
    x: np.ndarray
    annotations: np.ndarray | None = None
    time_origin: str = ''


def cgu_records(path, workers, limit=0):
    with RarReader(path) as ar:
        for name, _ in ar:
            if not name.lower().endswith('.txt'):
                continue
            lines = ar.read().decode('utf-8-sig').splitlines()
            subject = re.search(r'Subject\d+', name).group()
            label = lines[1].strip().upper()
            if label not in {'FALL', 'ADL'}:
                raise ValueError(f'{name}: unknown label {label}')
            x = np.loadtxt(io.StringIO('\n'.join(lines[4:])), delimiter=',', ndmin=2)
            if x.shape[1] != 6:
                raise ValueError(f'{name}: expected six columns')
            yield Record(name, subject, lines[2].strip(), 'fall' if label == 'FALL' else 'non-fall',
                         200., np.arange(len(x))/200., x)


def cogent_records(path, workers, limit=0):
    cols = ['ch_accel_x', 'ch_accel_y', 'ch_accel_z', 'ch_gyro_x', 'ch_gyro_y',
            'ch_gyro_z', 'annotation_1', 'annotation_2']
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            m = re.search(r'/(falls|walking on stairs)/subject_?(\d+)$', name)
            if not m:
                continue
            d = pd.read_csv(z.open(name), usecols=cols, dtype=np.float64)[cols].to_numpy()
            yield Record(name, 'subject'+m[2], m[1], 'annotated', 100.,
                         np.arange(len(d))/100., d[:, :6], d[:, 6:])


def excel_col(cell):
    value = 0
    for c in re.match(r'[A-Z]+', cell).group():
        value = value*26 + ord(c)-64
    return value-1


def sfu_read(job):
    path, name = job
    with zipfile.ZipFile(path) as outer:
        raw = outer.read(name)
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = sheet.find(ns+'sheetData')
        headers, data = {}, []
        for row_no, row in enumerate(rows):
            if row_no == 0:
                for c in row:
                    v = c.find(ns+'v')
                    if c.attrib.get('t') == 's':
                        value = strings[int(v.text)]
                    elif c.attrib.get('t') == 'inlineStr':
                        value = ''.join(c.find(ns+'is').itertext())
                    else:
                        value = v.text if v is not None else ''
                    headers[value] = excel_col(c.attrib['r'])
                requested = ['Time'] + [f'sternum Acceleration {a} (m/s^2)' for a in 'XYZ'] + [
                    f'sternum Angular Velocity {a} (rad/s)' for a in 'XYZ']
                indices = {headers[h]: i for i, h in enumerate(requested)}
            else:
                values = [np.nan]*7
                for c in row:
                    index = excel_col(c.attrib['r'])
                    if index in indices:
                        v = c.find(ns+'v')
                        if v is not None and v.text is not None:
                            values[indices[index]] = float(v.text)
                data.append(values)
    a = np.asarray(data, dtype=np.float64)
    valid_time = np.isfinite(a[:, 0])
    if not valid_time.any():
        raise ValueError(f'{name}: no timestamps')
    origin = int(a[valid_time, 0][0])
    t = (a[:, 0]-origin)/1e6  # source integers <2**53 are exactly representable
    m = re.search(r'/(sub\d+)/(ADLs|Falls|Near_Falls)/', name)
    label = {'ADLs': 'non-fall', 'Falls': 'fall', 'Near_Falls': 'near-fall'}[m[2]]
    return Record(name, m[1], Path(name).stem, label, 128., t, a[:, 1:], time_origin=str(origin))


def sfu_records(path, workers, limit=0):
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.xlsx'))
    if limit:
        categories = [[n for n in names if '/'+c+'/' in n] for c in ['ADLs', 'Falls', 'Near_Falls']]
        names = [categories[i % 3][i // 3] for i in range(min(limit, len(names)))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(sfu_read, [(str(path), n) for n in names], chunksize=4)


def uci_read(job):
    outer_name, raw, limit = job
    result = []
    labels_seen = Counter()
    with RarReader(raw) as ar:
        for name, _ in ar:
            if not name.endswith('/340527.txt'):
                continue
            code = re.search(r'/([89]\d{2})-', name)
            category = 'fall' if code and code[1].startswith('9') else 'non-fall'
            if limit and labels_seen[category] >= math.ceil(limit/2):
                continue
            content = ar.read()
            lines = content.decode(errors='replace').splitlines()
            m = re.search(r'(\d{3})/Testler Export/(\d{3})-([^/]+)/([^/]+)/340527.txt$', name)
            if not m:
                raise ValueError(f'Unrecognized UCI trial path: {name}')
            fs_match = re.search(r'Update Rate:\s*([\d.]+)Hz', '\n'.join(lines[:4]))
            if not fs_match:
                raise ValueError(f'{name}: missing sampling rate')
            fs = float(fs_match[1])
            cols = ['Counter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
            d = pd.read_csv(io.StringIO('\n'.join(lines[4:])), sep=r'\s+', usecols=cols)[cols].to_numpy(np.float64)
            counter = d[:, 0].copy()
            finite = np.flatnonzero(np.isfinite(counter))
            cv = counter[finite]
            if len(finite):
                wraps = np.cumsum(np.r_[0, np.diff(cv) < -32768])*65536
                counter[finite] = cv + wraps
                origin = counter[finite[0]]
            else:
                origin = 0.
            yield_label = 'fall' if 901 <= int(m[2]) <= 920 else 'non-fall' if 801 <= int(m[2]) <= 816 else None
            if yield_label is None:
                raise ValueError(f'{name}: unknown activity code')
            result.append(Record(outer_name+'!'+name, m[1], m[2]+'_'+m[3]+'_'+m[4],
                                 yield_label, fs, (counter-origin)/fs, d[:, 1:], time_origin=str(origin)))
            labels_seen[category] += 1
    return result


def uci_records(path, workers, limit=0):
    with RarReader(path) as ar, ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for name, _ in ar:
            if not name.lower().endswith('.rar'):
                continue
            pending.append(pool.submit(uci_read, (name, ar.read(), limit)))
            if limit:
                yield from pending.pop().result()
                return
            if len(pending) >= workers:
                yield from pending.pop(0).result()
        for future in pending:
            yield from future.result()


def pamap_records(path, workers, limit=0):
    # Zero-based: chest temperature=20, 16g acceleration=21:24, gyro=27:30.
    cols = [0, 1, 21, 22, 23, 27, 28, 29]
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            m = re.search(r'/(Protocol|Optional)/subject(\d+)\.dat$', name)
            if not m:
                continue
            d = pd.read_csv(z.open(name), sep=r'\s+', header=None, usecols=cols,
                            dtype=np.float64).to_numpy()
            origin = d[0, 0]
            yield Record(name, m[2], m[1], 'ssl', 100., d[:, 0]-origin, d[:, 2:],
                         d[:, 1:2], time_origin=str(origin))


ADAPTERS = {'CGU_BES': cgu_records, 'Cogent': cogent_records, 'SFU_IMU': sfu_records,
            'UCI_SimulatedFalls': uci_records, 'PAMAP2': pamap_records}


def contiguous(mask):
    mask = np.asarray(mask, bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def finite_blocks(t, x, fs, annotations=None):
    """Drop invalid samples and split discontinuities; never sort across a reset."""
    good = np.isfinite(t) & np.isfinite(x).all(axis=1)
    if annotations is not None:
        good &= np.isfinite(annotations).all(axis=1)
    indices = np.flatnonzero(good)
    if not len(indices):
        return []
    valid_t = t[indices]
    breaks = np.flatnonzero((np.diff(valid_t) < 0) | (np.diff(valid_t) > 2.5/fs))+1
    blocks = []
    for ii in np.split(indices, breaks):
        if len(ii) < 2:
            continue
        tt, xx = t[ii], x[ii]
        aa = annotations[ii] if annotations is not None else None
        if np.any(np.diff(tt) == 0):
            unique, inv, counts = np.unique(tt, return_inverse=True, return_counts=True)
            sums = np.zeros((len(unique), 6))
            np.add.at(sums, inv, xx)
            if aa is not None:
                # Conflicting labels at one timestamp cannot be averaged into a label.
                grouped = np.zeros((len(unique), aa.shape[1]))
                for j in range(aa.shape[1]):
                    lo = np.full(len(unique), np.inf)
                    hi = np.full(len(unique), -np.inf)
                    np.minimum.at(lo, inv, aa[:, j]); np.maximum.at(hi, inv, aa[:, j])
                    grouped[:, j] = np.where(lo == hi, lo, -1)
                aa = grouped
            tt, xx = unique, sums/counts[:, None]
        if len(tt) >= 2:
            blocks.append((tt, xx, aa))
    return blocks


def resample(t, x, native_fs, target_fs):
    duration = t[-1]-t[0]
    n = int(np.floor(duration*target_fs+1e-6))+1
    if n < 2:
        return np.empty((0, 6)), np.empty(0)
    grid = t[0]+np.arange(n)/target_fs
    if native_fs > target_fs:
        uniform = np.arange(int(np.floor(duration*native_fs+1e-6))+1)/native_fs
        values = np.stack([np.interp(uniform, t-t[0], x[:, j]) for j in range(6)], axis=1)
        ratio = Fraction(target_fs/native_fs).limit_denominator(1000)
        filtered = resample_poly(values, ratio.numerator, ratio.denominator, axis=0, padtype='line')
        ft = t[0]+np.arange(len(filtered))/target_fs
        values = np.stack([np.interp(grid, ft, filtered[:, j]) for j in range(6)], axis=1)
    else:
        values = np.stack([np.interp(grid, t, x[:, j]) for j in range(6)], axis=1)
    return values, grid


def robust_z(v):
    median = np.median(v)
    scale = 1.4826*np.median(np.abs(v-median))
    return (v-median)/(scale if scale > 1e-9 else np.std(v)+1e-9)


def event(x, fs, search=None):
    """Same V2 paired magnitude peaks, constrained by labels where available."""
    a = median_filter(np.linalg.norm(x[:, :3], axis=1), size=3, mode='nearest')
    g = median_filter(np.linalg.norm(x[:, 3:], axis=1), size=3, mode='nearest')
    za, zg = robust_z(a), robust_z(g)
    lo, hi = search if search is not None else (0, len(x))
    if hi <= lo:
        raise ValueError('Empty fall search interval')
    def peaks(v, z):
        pp = np.flatnonzero((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))+1
        pp = pp[(pp >= lo) & (pp < hi)]
        if not len(pp):
            pp = np.array([lo+int(np.argmax(v[lo:hi]))])
        return pp[np.argsort(z[pp])[-30:]]
    pa, pg = peaks(a, za), peaks(g, zg)
    scores = np.maximum(za[pa, None], 0)+np.maximum(zg[pg][None, :], 0)
    gaps = np.abs(pa[:, None]-pg[None, :])/fs
    scores -= 2*gaps
    scores[gaps > 1.] = -np.inf
    if np.isfinite(scores).any():
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        ia, ig, score = int(pa[i]), int(pg[j]), float(scores[i, j])
        method = 'paired_acc_gyro'
    else:
        ia = lo+int(np.argmax(a[lo:hi]))
        ig = int(pg[np.argmin(np.abs(pg-ia))])
        score = float(max(0, za[ia])+max(0, zg[ig]))
        method = 'acc_fallback'
    center = (ia+ig)/2 if abs(ia-ig)/fs <= 1 else float(ia)
    return {'mid_idx': center, 'acc_idx': ia, 'gyro_idx': ig, 'score': score,
            'peak_gap_s': abs(ia-ig)/fs, 'event_method': method,
            'review_flag': 'low_event_score' if score < 3 else ''}


def crop(n, midpoint, fs):
    full, model = int(round(5*fs)), int(round(3*fs))
    start = int(round(midpoint-(full-1)/2))
    if 0 <= start and start+full <= n:
        return start, start+full, '5s_centered'
    if n >= model:
        start = max(0, min(int(round(midpoint-(model-1)/2)), n-model))
        return start, start+model, '3s_boundary'
    return 0, n, 'short_requires_review'


class Writer:
    def __init__(self, out, dataset, fs):
        self.out, self.dataset, self.fs = out, dataset, fs
        self.manifest, self.audit = [], []
        self.seen, self.parent = {}, {}
        self.records, self.failures = 0, []
        for folder in ['fall', 'non-fall', 'ssl', 'quarantine']:
            (out/folder).mkdir(parents=True, exist_ok=True)

    def root(self, subject):
        self.parent.setdefault(subject, subject)
        if self.parent[subject] != subject:
            self.parent[subject] = self.root(self.parent[subject])
        return self.parent[subject]

    def merge(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def write(self, record, block, x, grid, start, end, label, suffix, extra=None):
        if end-start < 2:
            return
        xx = np.asarray(x[start:end], np.float32)
        if not np.isfinite(xx).all():
            raise ValueError(f'{record.source}: nonfinite output')
        identity = hashlib.sha256(record.source.encode()).hexdigest()[:12]
        stem = re.sub(r'[^A-Za-z0-9_-]', '_', record.subject+'_'+record.activity)[:100]
        relative = f'{label}/{stem}_{identity}_b{block}_{suffix}.csv'
        t = np.arange(len(xx), dtype=np.float64)*1000/self.fs
        d = pd.DataFrame(xx, columns=FEATURES)
        d.insert(0, 'time_ms', t)
        d.insert(0, 'sample_idx', np.arange(len(xx), dtype=np.int64))
        d.to_csv(self.out/relative, index=False, float_format='%.9g')
        item = {'output_file': relative, 'source_file': record.source,
                'original_subject_id': record.subject, 'activity': record.activity,
                'source_label': record.label, 'label': label, 'block': block,
                'source_time_origin': record.time_origin,
                'start_s': float(grid[start]), 'end_s_exclusive': float(grid[end-1]+1/self.fs),
                'samples': len(xx), 'fs_hz': self.fs,
                'sensor': SENSORS[self.dataset], 'padded_samples': 0,
                'numeric_sha256': hashlib.sha256(xx.tobytes()).hexdigest()}
        item.update(extra or {})
        self.manifest.append(item)

    def process(self, r):
        self.records += 1
        self.root(r.subject)
        if r.x.ndim != 2 or r.x.shape[1] != 6 or len(r.t) != len(r.x):
            raise ValueError(f'{r.source}: invalid shape')
        valid = int((np.isfinite(r.t) & np.isfinite(r.x).all(axis=1)).sum())
        if valid < 2:
            self.audit.append({'source_file': r.source, 'subject_id': r.subject,
                               'status': 'unusable_source_record', 'native_samples': len(r.x),
                               'source_label': r.label,
                               'reason': 'fewer_than_two_valid_sensor_and_time_rows'})
            return
        digest = hashlib.sha256(np.asarray(r.x, '<f8').tobytes()).hexdigest()
        if digest in self.seen:
            original = self.seen[digest]
            self.merge(r.subject, original.subject)
            conflict = (r.label == 'fall') != (original.label == 'fall')
            if conflict:
                for item in self.manifest:
                    if item['source_file'] == original.source and item['label'] != 'quarantine':
                        old = item['output_file']
                        new = 'quarantine/'+Path(old).name
                        (self.out/old).rename(self.out/new)
                        item.update(output_file=new, label='quarantine', reason='conflicting_duplicate_labels')
            self.audit.append({'source_file': r.source, 'subject_id': r.subject,
                               'status': 'exact_duplicate_removed', 'duplicate_of': original.source,
                               'conflicting_labels': conflict})
            return
        self.seen[digest] = Record(r.source, r.subject, r.activity, r.label, r.fs,
                                   np.empty(0), np.empty((0, 6)))
        blocks = finite_blocks(r.t, r.x, r.fs, r.annotations)
        self.audit.append({'source_file': r.source, 'subject_id': r.subject, 'status': 'parsed',
                           'native_samples': len(r.x), 'invalid_sensor_or_time_samples': len(r.x)-valid,
                           'blocks': len(blocks), 'source_label': r.label, 'numeric_sha256': digest})
        for b, (tt, xx, aa) in enumerate(blocks):
            if r.label == 'ssl':
                change = np.flatnonzero(np.any(np.diff(aa, axis=0) != 0, axis=1))+1
                edges = np.r_[0, change, len(tt)]
                for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
                    activity = int(aa[s, 0])
                    if e-s < 2:
                        continue
                    xr, gr = resample(tt[s:e], xx[s:e], r.fs, self.fs)
                    folder = 'ssl' if activity > 0 and len(xr) >= round(3*self.fs) else 'quarantine'
                    self.write(r, b, xr, gr, 0, len(xr), folder, f'activity{activity}_{i}',
                               {'activity_id': activity, 'supervised_y': -1,
                                'reason': '' if folder == 'ssl' else 'transition_or_short'})
                continue
            x, grid = resample(tt, xx, r.fs, self.fs)
            if len(x) < 2:
                continue
            if r.label == 'annotated':
                self.annotated(r, b, x, grid, tt, aa)
            elif r.label in {'non-fall', 'near-fall'}:
                folder = 'non-fall' if len(x) >= round(3*self.fs) else 'quarantine'
                self.write(r, b, x, grid, 0, len(x), folder, 'original', {'negative_type': r.label})
            else:
                if len(blocks) > 1:
                    self.write(r, b, x, grid, 0, len(x), 'quarantine', 'disconnected',
                               {'reason': 'disconnected_fall_recording'})
                else:
                    self.trial_fall(r, b, x, grid)

    def trial_fall(self, r, b, x, grid):
        ev = event(x, self.fs)
        start, end, kind = crop(len(x), ev['mid_idx'], self.fs)
        extra = dict(ev, output_type=kind,
                     event_time_s=float(grid[0]+ev['mid_idx']/self.fs),
                     event_index_in_output=float(ev['mid_idx']-start))
        if kind == 'short_requires_review':
            # No synthetic held-out data or noise learned from unassigned subjects.
            self.write(r, b, x, grid, 0, len(x), 'quarantine', 'short', extra)
            return
        self.write(r, b, x, grid, start, end, 'fall', 'event', extra)
        # Unannotated surroundings are pseudo-negatives only when quiet and separated
        # from impact/recovery by two additional seconds. Keep all other samples for review.
        guard = int(round(2*self.fs))
        a = np.linalg.norm(x[:, :3], axis=1)
        g = np.linalg.norm(x[:, 3:], axis=1)
        motion = np.linalg.norm(np.diff(x, axis=0, prepend=x[:1]), axis=1)
        quiet = (a <= np.quantile(a, .60)) & (g <= np.quantile(g, .60)) & (motion <= np.quantile(motion, .60))
        occupied = np.zeros(len(x), bool); occupied[start:end] = True
        for name, lo, hi in [('PRE', 0, max(0, start-guard)), ('POST', min(len(x), end+guard), len(x))]:
            for j, (s, e) in enumerate(contiguous(quiet[lo:hi])):
                s, e = s+lo, e+lo
                if e-s >= round(3*self.fs):
                    self.write(r, b, x, grid, s, e, 'non-fall', f'{name}{j}',
                               {'negative_type': 'quiet_guarded_pseudo_negative',
                                'review_flag': 'trial_label_only', 'guard_s': 2.})
                    occupied[s:e] = True
        for j, (s, e) in enumerate(contiguous(~occupied)):
            self.write(r, b, x, grid, s, e, 'quarantine', f'context{j}',
                       {'reason': 'unannotated_fall_context_or_recovery'})

    def annotated(self, r, b, x, grid, tt, aa):
        # Nearest native labels, never interpolate categorical annotations.
        jj = np.searchsorted(tt, grid, side='left').clip(0, len(tt)-1)
        left = (jj-1).clip(0)
        jj = np.where(np.abs(tt[left]-grid) <= np.abs(tt[jj]-grid), left, jj)
        labels = aa[jj].astype(np.int64)
        a1, a2 = labels[:, 0], labels[:, 1]
        fall = (a1 == 1) & np.isin(a2, [2, 6, 10, 11, 12, 13])
        near = (a1 == 2) & (a2 == 7)
        adl = (a1 == 0) & np.isin(a2, [1, 3, 4, 5, 8, 9, 15])
        occupied = np.zeros(len(x), bool)
        intervals = contiguous(fall)
        for i, (lo, hi) in enumerate(intervals):
            # Crop cannot include another annotated fall.
            safe_lo = intervals[i-1][1] if i else 0
            safe_hi = intervals[i+1][0] if i+1 < len(intervals) else len(x)
            ev = event(x, self.fs, (lo, hi))
            start, end, kind = crop(safe_hi-safe_lo, ev['mid_idx']-safe_lo, self.fs)
            start, end = start+safe_lo, end+safe_lo
            if kind == 'short_requires_review':
                self.write(r, b, x, grid, start, end, 'quarantine', f'shortfall{i}', {'reason': kind})
                occupied[start:end] = True
                continue
            self.write(r, b, x, grid, start, end, 'fall', f'event{i}',
                       dict(ev, output_type=kind, event_time_s=float(grid[0]+ev['mid_idx']/self.fs),
                            event_index_in_output=float(ev['mid_idx']-start),
                            annotated_fall_start_s=float(grid[lo]),
                            annotated_fall_end_s=float(grid[hi-1]+1/self.fs),
                            annotation_exceeds_crop=bool(lo < start or hi > end)))
            occupied[start:end] = True
        unsafe = fall.copy()
        guard = int(round(2*self.fs))
        for lo, hi in intervals:
            unsafe[max(0, lo-guard):min(len(x), hi+guard)] = True
        # Preserve short near-falls with annotation-safe surrounding ADL context.
        for i, (lo, hi) in enumerate(contiguous(near)):
            s, e, kind = crop(len(x), (lo+hi-1)/2, self.fs)
            if (e-s >= round(3*self.fs) and not unsafe[s:e].any() and
                    (near[s:e] | adl[s:e]).all() and not occupied[s:e].any()):
                self.write(r, b, x, grid, s, e, 'non-fall', f'nearfall{i}', {'negative_type': 'annotated_near_fall'})
                occupied[s:e] = True
        for negative_type, mask in [('annotated_adl', adl), ('annotated_near_fall', near)]:
            for i, (s, e) in enumerate(contiguous(mask & ~unsafe & ~occupied)):
                if e-s >= round(3*self.fs):
                    self.write(r, b, x, grid, s, e, 'non-fall', f'{negative_type}{i}', {'negative_type': negative_type})
                    occupied[s:e] = True
        for i, (s, e) in enumerate(contiguous(~occupied)):
            self.write(r, b, x, grid, s, e, 'quarantine', f'unknown{i}', {'reason': 'transition_short_or_fall_recovery'})

    def finish(self, elapsed):
        components = defaultdict(list)
        for s in self.parent:
            components[self.root(s)].append(s)
        groups = {s: self.dataset+':'+'+'.join(sorted(components[self.root(s)])) for s in self.parent}
        for row in self.manifest:
            row['subject_id'] = groups[row['original_subject_id']]
            row['split_group_id'] = groups[row['original_subject_id']]
        manifest = pd.DataFrame(self.manifest)
        manifest.to_csv(self.out/'manifest.csv', index=False)
        pd.DataFrame(self.audit).to_csv(self.out/'preprocessing_audit.csv', index=False)
        counts = Counter(row['label'] for row in self.manifest)
        summary = {'dataset': self.dataset, 'target_sampling_rate_hz': self.fs,
                   'source_records': self.records, 'source_label_counts': dict(Counter(
                       row['source_label'] for row in self.audit if row.get('status') == 'parsed')),
                   'output_counts': dict(counts), 'output_samples': dict(Counter({
                       label: sum(r['samples'] for r in self.manifest if r['label'] == label) for label in counts})),
                   'duplicates_removed': sum(r.get('status') == 'exact_duplicate_removed' for r in self.audit),
                   'unusable_source_records': sum(r.get('status') == 'unusable_source_record' for r in self.audit),
                   'original_subjects': len(self.parent), 'split_groups': len(components),
                   'cross_subject_duplicate_groups': [ss for ss in components.values() if len(ss) > 1],
                   'units': UNITS[self.dataset], 'sensor': SENSORS[self.dataset],
                   'padded_files': 0, 'short_policy': 'quarantine; no synthetic validation data',
                   'negative_guard_seconds': 2, 'processing_seconds': round(elapsed, 2),
                   'errors': self.failures, 'version': 1}
        (self.out/'preprocessing_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary


def verify(out):
    manifest = pd.read_csv(out/'manifest.csv').fillna('')
    if len(manifest) == 0:
        raise ValueError('Empty output manifest')
    files = set()
    counts = Counter()
    for r in manifest.to_dict('records'):
        relative = r['output_file']
        if relative in files:
            raise ValueError('Repeated output filename: '+relative)
        files.add(relative)
        d = pd.read_csv(out/relative)
        if list(d.columns) != ['sample_idx', 'time_ms', *FEATURES]:
            raise ValueError('Incorrect schema: '+relative)
        x = d[FEATURES].to_numpy(np.float32)
        if len(d) != int(r['samples']) or not np.isfinite(x).all():
            raise ValueError('Incorrect length or nonfinite values: '+relative)
        if not np.array_equal(d.sample_idx.to_numpy(), np.arange(len(d))):
            raise ValueError('Incorrect sample indices: '+relative)
        expected_t = np.arange(len(d))*1000/float(r['fs_hz'])
        if not np.allclose(d.time_ms, expected_t, rtol=1e-8, atol=1e-5):
            raise ValueError('Incorrect timestamps: '+relative)
        if hashlib.sha256(x.tobytes()).hexdigest() != r['numeric_sha256']:
            raise ValueError('Numeric checksum mismatch: '+relative)
        if r['label'] == 'fall':
            if len(d) not in [round(5*r['fs_hz']), round(3*r['fs_hz'])]:
                raise ValueError('Incorrect fall crop length: '+relative)
            if not 0 <= float(r['event_index_in_output']) < len(d):
                raise ValueError('Event outside crop: '+relative)
        elif r['label'] in {'non-fall', 'ssl'} and len(d) < round(3*r['fs_hz']):
            raise ValueError('Short training segment: '+relative)
        counts[r['label']] += 1
    actual = {str(p.relative_to(out)) for folder in ['fall', 'non-fall', 'ssl', 'quarantine']
              for p in (out/folder).glob('*.csv')}
    if actual != files:
        raise ValueError('Manifest/files disagree')
    result = {'verified_files': len(files), 'counts': dict(counts), 'status': 'passed'}
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Smoke test: first N source recordings')
    p.add_argument('--fs', type=float, default=30.)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output)), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output.with_name(args.output.name+f'.inprogress.{os.getpid()}')
    if stage.exists():
        raise FileExistsError(stage)
    started = time.monotonic()
    writer = Writer(stage, args.dataset, args.fs)
    records = ADAPTERS[args.dataset](args.root/ARCHIVES[args.dataset], args.workers, args.limit)
    try:
        for i, record in enumerate(records):
            writer.process(record)
            if (i+1) % 25 == 0:
                print(json.dumps({'dataset': args.dataset, 'records': i+1,
                                  'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
            if args.limit and i+1 >= args.limit:
                break
    finally:
        records.close()
    summary = writer.finish(time.monotonic()-started)
    verification = verify(stage)
    stage.rename(args.output)  # publish only a verified dataset
    print(json.dumps({'summary': summary, 'verification': verification}), flush=True)


if __name__ == '__main__':
    main()
