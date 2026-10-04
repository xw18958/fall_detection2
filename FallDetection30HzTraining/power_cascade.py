"""Offline counterpart of firmware power_policy.h; no inferred event labels."""
from dataclasses import dataclass
import numpy as np

STRIDE_US=250_000
QUIET_US=1_000_000
BURST_US=8_000_000

@dataclass
class Policy:
    active:bool=False
    started:int=0
    last:int=0
    def step(self, now, suspicious):
        capped=self.active and now-self.started>=BURST_US
        if capped:self.active=False
        woke=False
        if suspicious:
            if not self.active:self.active=True;self.started=now;woke=True
            self.last=now
        if self.active and now-self.last>=QUIET_US:self.active=False
        return self.active,woke,capped

def binary_metrics(y,p,threshold):
    y=np.asarray(y,bool);pred=np.asarray(p)>=threshold
    tp=int((y&pred).sum());fn=int((y&~pred).sum());fp=int((~y&pred).sum());tn=int((~y&~pred).sum())
    div=lambda a,b:float(a/b) if b else None
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,recall=div(tp,tp+fn),specificity=div(tn,tn+fp),precision=div(tp,tp+fp),F1=div(2*tp,2*tp+fn+fp))

def simulate(rows, trigger, full, threshold, full_threshold):
    """Rows are sorted per real segment. Gaps reset policy, never join history.

    Negative exposure includes only real >=30-second continuous segments and
    excludes the initial 3-second warmup. Crop-label classification stays intact.
    """
    active=np.zeros(len(rows),bool);wake=np.zeros(len(rows),bool);caps=0
    exposure=0.;negative_wakes=0;previous=None;policy=Policy();records={}
    for i,row in enumerate(rows):
        key=(row['key'],row['segment'])
        if key!=previous:
            policy=Policy();previous=key
            if row['label']==0 and row['segment_samples']>=900:
                exposure+=(row['segment_samples']-90)/30
        # Exact cumulative quarter-second deadlines, independent of rounded
        # 7/8-sample crop indices. Arrays themselves are the existing 30 Hz views.
        active[i],wake[i],capped=policy.step(int(row['step'])*STRIDE_US,trigger[i]>=threshold)
        caps+=int(capped)
        if row['label']==0 and row['segment_samples']>=900:negative_wakes+=int(wake[i])
        record=records.setdefault(row['key'],{'label':row['label'],'trigger':False,'full':False,'baseline':False})
        record['trigger']|=bool(trigger[i]>=threshold)
        record['baseline']|=bool(full[i]>=full_threshold)
        record['full']|=bool(active[i] and full[i]>=full_threshold)
    labels=np.asarray([r['label'] for r in rows]);domains=sorted({r['domain'] for r in rows})
    result={'trigger':binary_metrics(labels,trigger,threshold),'baseline_full':binary_metrics(labels,full,full_threshold),
            'cascade_full':binary_metrics(labels,np.where(active,full,-1),full_threshold),'wakeups':int(wake.sum()),
            'full_invocations':int(active.sum()),'full_invocation_fraction':float(active.mean()),'burst_caps':caps,
            'continuous_negative_hours':exposure/3600,'false_wakeups':negative_wakes,
            'false_wakeups_per_hour':negative_wakes/(exposure/3600) if exposure else None,
            'trigger_recording_recall':float(np.mean([r['trigger'] for r in records.values() if r['label']])) if any(r['label'] for r in records.values()) else None,
            'lost_baseline_fall_recordings':sum(r['label'] and r['baseline'] and not r['full'] for r in records.values()),'per_source':{}}
    for domain in domains:
        idx=np.asarray([r['domain']==domain for r in rows]);keys={r['key'] for r in rows if r['domain']==domain}
        rr=[records[k] for k in keys if records[k]['label']]
        result['per_source'][domain]={'trigger':binary_metrics(labels[idx],np.asarray(trigger)[idx],threshold),
          'baseline_full':binary_metrics(labels[idx],np.asarray(full)[idx],full_threshold),
          'cascade_full':binary_metrics(labels[idx],np.where(active[idx],np.asarray(full)[idx],-1),full_threshold),
          'trigger_recording_recall':float(np.mean([r['trigger'] for r in rr])) if rr else None}
    return result,active

def select_threshold(rows, trigger, full, full_threshold, min_window_recall=.995):
    """Validation only: high per-source recall, preserve every baseline true positive.

    Choose the highest feasible threshold; reducing it only increases active
    windows and wake duration under the monotone quiet policy. No invalid fallback.
    """
    trigger=np.asarray(trigger);full=np.asarray(full);upper=[]
    for domain in sorted({r['domain'] for r in rows}):
        pos=np.asarray([r['domain']==domain and r['label']==1 for r in rows])
        if pos.any():
            values=np.sort(trigger[pos]);allowed=int(np.floor((1-min_window_recall)*len(values)+1e-9))
            upper.append(float(values[allowed]))
            detected=pos&(full>=full_threshold)
            if detected.any():upper.append(float(trigger[detected].min()))
    if not upper:raise ValueError('No positive validation windows')
    threshold=float(np.float32(min(upper)))
    report,active=simulate(rows,trigger,full,threshold,full_threshold)
    protected=(np.asarray([r['label'] for r in rows])==1)&(full>=full_threshold)
    qualified=bool(np.all(active[protected]) and report['lost_baseline_fall_recordings']==0 and
       all(v['trigger']['recall'] is None or v['trigger']['recall']>=min_window_recall for v in report['per_source'].values()) and
       report['full_invocation_fraction']<1/3)
    # 4 Hz gated TCN must execute less than one third of frames to improve on
    # the existing 4/3 Hz baseline before accounting for trigger/radio/display.
    return {'threshold':threshold,'qualified':qualified,'report':report,
            'protocol':{'min_per_source_window_recall':min_window_recall,'max_added_baseline_true_positive_misses':0,'max_full_invocation_fraction':1/3,'selection_partition':'val'}}

def energy_estimate(seconds, full_calls, trigger_calls, full_latency_us, trigger_latency_us, profile):
    """Requires measured board profiles; no fabricated current/power defaults."""
    if seconds<=0:raise ValueError('Positive exposure required')
    full_s=full_calls*full_latency_us/1e6;trigger_s=trigger_calls*trigger_latency_us/1e6
    if full_s+trigger_s>seconds:return {'schedulable':False,'reason':'compute time exceeds exposure'}
    idle_s=seconds-full_s-trigger_s
    mah=(full_s*profile['full_board_ma']+trigger_s*profile['trigger_board_ma']+idle_s*profile['idle_sensing_board_ma'])/3600
    return {'schedulable':True,'estimated_mah':mah,'average_ma':mah*3600/seconds,
      'estimated_battery_hours':profile['usable_capacity_mah']/(mah*3600/seconds) if mah else None,
      'assumptions':'Measured board currents include IMU/regulator; display and Wi-Fi off. Host latency cannot substitute for M5 latency.'}
