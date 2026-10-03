#!/usr/bin/env python3
"""Audit complete raw M5 fall recordings without inventing window labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HEADER = ['seq', 'device_timestamp_us', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(session):
    meta = json.loads((session/'metadata.json').read_text())
    device, quality, completion = meta['device'], meta['quality'], meta['completion']
    if meta['schema_version'] != 3 or not completion['complete'] or completion.get('device_buffer_overflowed'):
        raise ValueError(f'{session.name}: incomplete, overflowed, or unsupported schema')
    if device['rate_hz'] != 30 or device['accel_g_per_lsb'] != 8/32768 or device['gyro_dps_per_lsb'] != 2000/32768:
        raise ValueError(f'{session.name}: unexpected sensor rate or range')
    with (session/'samples.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError(f'{session.name}: unexpected CSV header')
        rows = [[int(row[k]) for k in HEADER] for row in reader]
    journal = [json.loads(line) for line in (session/'.recovery/journal.jsonl').read_text().splitlines() if line.strip()]
    n = len(rows)
    if n < 90 or n != len(journal) or n != quality['saved_samples'] or n != completion['saved_samples']:
        raise ValueError(f'{session.name}: sample counts disagree or fewer than 90 samples')
    flags = Counter()
    for i, (row, entry) in enumerate(zip(rows, journal)):
        sample = entry['sample']
        if row != [sample['seq'], sample['device_timestamp_us'], *sample['raw']]:
            raise ValueError(f'{session.name}: CSV/journal mismatch at row {i}')
        if row[0] != i or any(v < -32768 or v > 32767 for v in row[2:]):
            raise ValueError(f'{session.name}: sequence gap or invalid signed count')
        for bit, name in [(1, 'read_errors'), (2, 'timing_gap_flags'), (4, 'saturated_samples')]:
            flags[name] += bool(sample['flags'] & bit)
    if any(flags[k] != quality[k] for k in flags) or flags['read_errors'] or flags['timing_gap_flags']:
        raise ValueError(f'{session.name}: invalid samples or inconsistent flags')
    intervals = [b[1]-a[1] for a, b in zip(rows, rows[1:])]
    if min(intervals) <= 0 or max(intervals) > 50000 or quality['sequence_gaps'] or quality['observed_timestamp_gaps']:
        raise ValueError(f'{session.name}: timing/sequence gaps')
    duration = (rows[-1][1]-rows[0][1])/1e6
    if abs(duration-quality['elapsed_seconds']) > 1e-5:
        raise ValueError(f'{session.name}: elapsed time mismatch')
    acceleration_g = [math.sqrt(sum(v*v for v in row[2:5]))*8/32768 for row in rows]
    gyro_dps = [math.sqrt(sum(v*v for v in row[5:8]))*2000/32768 for row in rows]
    peak = max(range(n), key=acceleration_g.__getitem__)
    physical = [[*(v*(8/32768)*9.80665 for v in row[2:5]),
                 *(v*(2000/32768)*(math.pi/180) for v in row[5:8])] for row in rows]
    paths = ['metadata.json', 'samples.csv', '.recovery/journal.jsonl']
    result = dict(folder=session.name, session_id=meta['session_id'], boot_id=device['boot_id'],
                  device_id=device['device_id'], firmware=device['firmware'], raw_samples=n,
                  duration_seconds=duration, measured_hz=(n-1)/duration,
                  participant=meta.get('profile', {}).get('participant'),
                  placement=meta.get('profile', {}).get('placement'),
                  annotation=meta.get('annotation'), quality_flags=dict(flags),
                  first_device_timestamp_us=rows[0][1], last_device_timestamp_us=rows[-1][1],
                  peak_acceleration_g=max(acceleration_g), peak_gyro_dps=max(gyro_dps),
                  acceleration_peak_seconds=(rows[peak][1]-rows[0][1])/1e6,
                  checksums={p: sha(session/p) for p in paths})
    return result, rows, physical


def audit(root, output, plots=False):
    if output.exists():
        raise FileExistsError('Use a fresh output directory; preserve previous audits')
    sessions = sorted(p for p in root.iterdir() if p.is_dir() and (p/'metadata.json').is_file())
    if not sessions:
        raise ValueError('No M5 recording folders found')
    inspected = [(p, *inspect(p)) for p in sessions]
    records = [r for _, r, _, _ in inspected]
    for key in ['session_id']:
        if len({r[key] for r in records}) != len(records):
            raise ValueError('Duplicate session identity')
    if len({r['checksums']['samples.csv'] for r in records}) != len(records):
        raise ValueError('Duplicate raw recording')
    for boot in {r['boot_id'] for r in records}:
        rr = sorted((r for r in records if r['boot_id'] == boot), key=lambda r:r['first_device_timestamp_us'])
        if any(a['last_device_timestamp_us'] >= b['first_device_timestamp_us'] for a,b in zip(rr,rr[1:])):
            raise ValueError('Overlapping acquisition intervals within one boot')
    output.mkdir(parents=True)
    summary = dict(status='quality_passed', recordings=len(records),
                   raw_samples=sum(r['raw_samples'] for r in records),
                   duration_seconds=sum(r['duration_seconds'] for r in records),
                   boots=len({r['boot_id'] for r in records}),
                   quality_flags={k:sum(r['quality_flags'][k] for r in records) for k in records[0]['quality_flags']},
                   fall_recording_label_source='user designated the folder as fall recordings',
                   training_ready='requires a separate label/crop and split protocol; this tool audits raw integrity only',
                   limitations=['participant and placement may be unknown', 'fall intervals are not annotated',
                                'a single positive boot cannot provide independent boot holdouts'],
                   records=records)
    (output/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    checksums=[f"{digest}  {r['folder']}/{name}" for r in records for name,digest in r['checksums'].items()]
    (output/'raw_checksums.sha256').write_text('\n'.join(checksums)+'\n')
    review=[]
    for _,r,rows,physical in inspected:
        peak=r['acceleration_peak_seconds']; duration=r['duration_seconds']
        # A plot guide only: not an inferred ground-truth fall label or crop.
        start=min(max(0.,peak-2.5),max(0.,duration-5.))
        review.append(dict(folder=r['folder'],session_id=r['session_id'],participant='',placement='',
                           fall_start_seconds='',fall_end_seconds='',confirmed=False,
                           acceleration_peak_seconds=peak,plot_guide_start_seconds=start,
                           plot_guide_end_seconds=min(duration,start+5.)))
        with (output/(r['folder']+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_ms','Acc_X','Acc_Y','Acc_Z','Gyro_X','Gyro_Y','Gyro_Z'])
            for row,values in zip(rows,physical):writer.writerow([(row[1]-rows[0][1])/1000,*values])
    (output/'annotation_review.json').write_text(json.dumps(review,indent=2)+'\n')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output/'motion_review.pdf') as pdf:
            for index, (_,r,rows,physical) in enumerate(inspected):
                t=[(row[1]-rows[0][1])/1e6 for row in rows]
                fig, axes=plt.subplots(3,1,figsize=(11,8.5),sharex=True)
                axes[0].plot(t,[math.sqrt(sum(v*v for v in row[:3]))/9.80665 for row in physical],color='#124f88')
                axes[0].set_ylabel('Acceleration magnitude (g)')
                axes[1].plot(t,[math.sqrt(sum(v*v for v in row[3:]))*180/math.pi for row in physical],color='#b45309')
                axes[1].set_ylabel('Gyroscope magnitude (deg/s)')
                for c,color in enumerate(['#124f88','#16805d','#b45309']):
                    axes[2].plot(t,[row[c]/9.80665 for row in physical],label='XYZ'[c],color=color)
                axes[2].set_ylabel('Acceleration axes (g)');axes[2].set_xlabel('Seconds from acquisition start');axes[2].legend(loc='upper right')
                for ax in axes:
                    ax.axvspan(review[index]['plot_guide_start_seconds'],review[index]['plot_guide_end_seconds'],color='#8b5cf6',alpha=.1)
                    ax.axvline(r['acceleration_peak_seconds'],color='#666',linestyle='--',linewidth=.8)
                    ax.grid(alpha=.18)
                fig.suptitle(f"{index+1:02d}: {r['folder']}\nShading is a 5-second guide around maximum acceleration; UNCONFIRMED, not a training label",fontsize=11)
                fig.tight_layout();pdf.savefig(fig)
                if index == 0:fig.savefig(output/'motion_example.png',dpi=130)
                plt.close(fig)
    return {k:v for k,v in summary.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.output,args.plots),indent=2))
