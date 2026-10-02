#!/usr/bin/env python3
"""Verify and convert complete M5 collector sessions into full-length V2 negatives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from preprocess_public_v2 import resample

SESSION_SPLITS = {
    '2026-10-02_17-30-52_unspecified_cc3c8a5e': 'train',
    '2026-10-02_17-46-00_unspecified_3267c7cd': 'val',
    '2026-10-02_17-55-47_unspecified_24e5413e': 'test',
    '2026-10-02_18-08-11_unspecified_9d3a85ef': 'train',
    '2026-10-02_18-13-20_unspecified_70f3e2b3': 'train',
    '2026-10-02_18-15-36_unspecified_99b24bf2': 'train',
    '2026-10-02_19-04-39_unspecified_d2e3811c': 'train',
}
RAW = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FEATURES = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
HEADER = ['seq', 'device_timestamp_us', *RAW]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_and_convert(session: Path, out: Path) -> dict:
    meta_path = session/'metadata.json'
    sample_path = session/'samples.csv'
    journal_path = session/'.recovery/journal.jsonl'
    if not all(p.is_file() for p in (meta_path, sample_path, journal_path)):
        raise ValueError(f'{session.name}: missing CSV, metadata, or recovery journal')
    metadata = json.loads(meta_path.read_text())
    quality, device = metadata['quality'], metadata['device']
    if not metadata.get('completion', {}).get('complete'):
        raise ValueError(f'{session.name}: transfer is incomplete')
    if int(device.get('rate_hz', 0)) != 30:
        raise ValueError(f'{session.name}: expected 30 Hz collector')
    accel = float(device['accel_g_per_lsb'])*16384.0
    gyro = float(device['gyro_dps_per_lsb'])*16.4

    raw = []
    flags = []
    with sample_path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError(f'{session.name}: unexpected CSV header {reader.fieldnames}')
        for row in reader:
            raw.append([int(row[k]) for k in HEADER])
    if len(raw) != int(quality['saved_samples']) or len(raw) != int(metadata['completion']['saved_samples']):
        raise ValueError(f'{session.name}: CSV, quality, and transfer row counts disagree')

    with journal_path.open() as f:
        journal = [json.loads(line)['sample'] for line in f if line.strip()]
    if len(journal) != len(raw):
        raise ValueError(f'{session.name}: recovery journal row count mismatch')
    for i, (row, sample) in enumerate(zip(raw, journal)):
        if row != [int(sample['seq']), int(sample['device_timestamp_us']), *map(int, sample['raw'])]:
            raise ValueError(f'{session.name}: CSV/journal mismatch at row {i}')
        flags.append(int(sample['flags']))
    flag_counts = Counter()
    for value in flags:
        for bit, name in ((1, 'read_error'), (2, 'timing_gap'), (4, 'saturated')):
            flag_counts[name] += bool(value & bit)
    for name in ('read_error', 'timing_gap', 'saturated'):
        meta_name = {'read_error':'read_errors', 'timing_gap':'timing_gap_flags',
                     'saturated':'saturated_samples'}[name]
        if flag_counts[name] != int(quality[meta_name]):
            raise ValueError(f'{session.name}: {name} journal/metadata counts disagree')
    if flag_counts['read_error'] or flag_counts['timing_gap']:
        raise ValueError(f'{session.name}: flagged invalid samples need explicit segmentation before training')

    timestamps = np.asarray([r[1] for r in raw], np.int64)
    x = np.asarray([r[2:] for r in raw], np.float64)
    t = (timestamps-timestamps[0]).astype(np.float64)/1e6
    dt = np.diff(t)
    if np.any(dt <= 0) or np.any(dt > 0.05):
        raise ValueError(f'{session.name}: nonmonotonic timestamps or a >50 ms gap')
    observed_gaps = int(np.sum(dt > .05))
    if observed_gaps != int(quality.get('observed_timestamp_gaps', observed_gaps)):
        raise ValueError(f'{session.name}: observed timestamp gap count disagrees with metadata')
    x[:, :3] *= accel
    x[:, 3:] *= gyro
    clipped_acceleration = (x[:, :3] < -32768.0) | (x[:, :3] > 32767.0)
    np.clip(x, -32768, 32767, out=x)
    sampled, grid = resample(t, x, 30.0, 30.0)
    if len(sampled) < 90:
        raise ValueError(f'{session.name}: fewer than one complete 3-second window')

    out.mkdir(parents=True, exist_ok=True)
    csv_name = session.name+'.csv'
    csv_path = out/'non-fall'/csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ms', *FEATURES])
        for tm, values in zip(grid*1000, sampled):
            writer.writerow([f'{tm:.6f}', *(f'{v:.7f}' for v in values)])

    return {
        'session_id': metadata['session_id'], 'source_session': session.name,
        'source_csv_sha256': sha256(sample_path), 'output_file': 'non-fall/'+csv_name,
        'output_sha256': sha256(csv_path), 'label': 'non-fall', 'split': SESSION_SPLITS[session.name],
        'split_group_id': 'M5BLE:'+str(device['boot_id']),
        'raw_samples': len(raw), 'processed_samples': len(sampled),
        'duration_seconds': round(float(t[-1]), 6), 'fs_hz': 30,
        'read_errors': flag_counts['read_error'], 'timing_gap_flags': flag_counts['timing_gap'],
        'saturated_samples': flag_counts['saturated'],
        'accel_scale_to_training_counts': accel, 'gyro_scale_to_training_counts': gyro,
        'acceleration_rows_clipped': int(np.any(clipped_acceleration, axis=1).sum()),
        'participant': metadata.get('profile', {}).get('participant', 'unspecified'),
        'placement': metadata.get('profile', {}).get('placement', 'unspecified'),
        'boot_id': device['boot_id'], 'session_sha256': {
            'metadata.json': sha256(meta_path), 'samples.csv': sha256(sample_path),
            '.recovery/journal.jsonl': sha256(journal_path)},
    }


def process(raw_root: Path, output_root: Path) -> dict:
    sessions = sorted(p for p in raw_root.iterdir() if p.is_dir() and (p/'metadata.json').is_file())
    if len(sessions) != len(SESSION_SPLITS) or {p.name for p in sessions} != set(SESSION_SPLITS):
        raise ValueError('Expected exactly the seven inventoried M5 sessions')
    processed_root = output_root/'processed_v2'
    manifest = [verify_and_convert(p, processed_root) for p in sessions]
    groups = {}
    for row in manifest:
        groups.setdefault(row['split_group_id'], set()).add(row['split'])
    if any(len(roles) != 1 for roles in groups.values()):
        raise ValueError('One device boot was assigned to multiple splits')
    split_counts = Counter(row['split'] for row in manifest)
    if split_counts != {'train': 5, 'val': 1, 'test': 1}:
        raise ValueError(f'Unexpected split allocation: {dict(split_counts)}')
    processed_root.mkdir(parents=True, exist_ok=True)
    with (processed_root/'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    checksums = []
    for session in sessions:
        for path in (session/'metadata.json', session/'samples.csv', session/'.recovery/journal.jsonl'):
            checksums.append(f'{sha256(path)}  {path.relative_to(raw_root)}')
    (output_root/'raw_checksums.sha256').write_text('\n'.join(checksums)+'\n')
    summary = {
        'status': 'passed', 'protocol': 'full-length M5 hard negatives; no event crop or peak relabeling',
        'source_root': str(raw_root), 'raw_sessions': len(manifest),
        'raw_samples': sum(r['raw_samples'] for r in manifest),
        'processed_samples': sum(r['processed_samples'] for r in manifest),
        'duration_seconds': sum(r['duration_seconds'] for r in manifest),
        'split_sessions': dict(split_counts), 'split_groups': len(groups),
        'quality_totals': {k: sum(r[k] for r in manifest) for k in
                           ('read_errors', 'timing_gap_flags', 'saturated_samples', 'acceleration_rows_clipped')},
        'label_source': 'user explicitly designated all seven sessions non-fall',
    }
    (processed_root/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='M5_hard_negatives_20261002 directory containing raw/')
    args = parser.parse_args()
    print(json.dumps(process(args.root/'raw', args.root), indent=2))


if __name__ == '__main__':
    main()
