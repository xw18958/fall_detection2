"""Summarize cumulative firmware telemetry and optional externally measured current.

CSV columns: timestamp_s,current_ma (battery-side current, signed convention
positive discharge). Supply usable battery capacity explicitly for lifetime.
No estimate is manufactured from USB voltage, free memory or host replay speed.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

def summarize(path,current=None,capacity=None):
    records=[]
    for line in Path(path).read_text(errors='replace').splitlines():
        if 'POWER_JSON ' in line:records.append(json.loads(line.split('POWER_JSON ',1)[1]))
    if len(records)<2:raise ValueError('At least two telemetry records required')
    a,b=records[0],records[-1];dt=(b['uptime_us']-a['uptime_us'])/1e6
    if dt<=0 or any(b[k]<a[k] for k in ('samples','full_n','trigger_n')):raise ValueError('Select one uninterrupted boot')
    counts={k:b[k]-a[k] for k in ('samples','gaps','trigger_n','wake_n','full_n','queue_drops','trigger_deadline_misses','full_deadline_misses')}
    rates={k:(b[k]-a[k])/dt/1e6 for k in ('sensor_us','trigger_us','full_us','active_us','display_us','wifi_us')}
    latency={name:((b[name+'_us']-a[name+'_us'])/counts[name+'_n'] if counts[name+'_n'] else None) for name in ('trigger','full')}
    result=dict(seconds=dt,counts=counts,time_fractions=rates,mean_latency_us=latency,latest_memory={k:b[k] for k in ('full_arena_bytes','trigger_arena_bytes','free_internal','free_psram')},
      false_wakeups_per_hour=None,fall_recall=None,reason='Runtime counters need independently labeled activity/events to assign false wakes or recall',energy=None)
    if current:
        with Path(current).open() as stream:rows=list(csv.DictReader(stream))
        t=np.array([float(r['timestamp_s']) for r in rows]);i=np.array([float(r['current_ma']) for r in rows])
        if len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(i)) or np.any(np.diff(t)<=0):raise ValueError('Current timestamps must strictly increase and be finite')
        mah=float(np.trapezoid(i,t)/3600);average=mah*3600/(t[-1]-t[0])
        result['energy']=dict(measured_mah=mah,average_ma=average,measurement_seconds=float(t[-1]-t[0]),battery_hours=capacity/average if capacity and average>0 else None,
          scope='External measurement interval; align its timestamps and operating conditions with the telemetry interval')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--current-csv',type=Path);p.add_argument('--usable-capacity-mah',type=float);p.add_argument('--out',type=Path);a=p.parse_args()
    result=summarize(a.log,a.current_csv,a.usable_capacity_mah);text=json.dumps(result,indent=2);print(text)
    if a.out:a.out.write_text(text+'\n')
