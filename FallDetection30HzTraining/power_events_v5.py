"""V5 cascade evaluation: timely baseline events and genuine state transitions.

Recording labels remain unchanged. A baseline-positive run is a diagnostic
reference, not a clinically annotated fall time. Qualification here does not
establish commercial safety or independent participant generalization.
"""
import numpy as np
from power_cascade import binary_metrics

def quantized_threshold_margin(threshold, output_scale, output_lsb_tolerance=1):
    """Lower a locked threshold by the declared two-logit replay tolerance.

    Both output logits can differ in opposite directions. This is an explicit
    calibration budget, not proof of hardware parity on unseen inputs. Validate
    the resulting wake cost; do not tune the budget on held-out test outcomes.
    """
    if not 0 < threshold < 1 or not np.isfinite(output_scale) or output_scale <= 0:
        raise ValueError('Finite interior probability and positive scale required')
    if not isinstance(output_lsb_tolerance, int) or output_lsb_tolerance < 0:
        raise ValueError('Nonnegative integer output tolerance required')
    logit = np.log(threshold) - np.log1p(-threshold)
    logit -= 2 * output_lsb_tolerance * output_scale
    return float(np.exp(-np.logaddexp(0., -logit)))


def simulate(rows, trigger, full, threshold, full_threshold,
             quiet_steps=4, burst_steps=32, max_delay_steps=1):
    trigger=np.asarray(trigger); full=np.asarray(full)
    if len(rows)!=len(trigger) or len(rows)!=len(full):
        raise ValueError('One score per window required')
    if quiet_steps<1 or burst_steps<1 or max_delay_steps<0:
        raise ValueError('Invalid wake/timing policy')
    active=np.zeros(len(rows),bool)
    wakes=cap_rearms=negative_wakes=negative_active=negative_windows=0
    exposure=0.; previous=None; state=False; last=-quiet_steps; opened=0
    segments={}
    for i,r in enumerate(rows):
        key=(r['key'],r['segment'])
        if key!=previous:
            if key in segments:raise ValueError('Segment rows must be contiguous')
            state=False;last=-quiet_steps;opened=0;previous=key
            segments[key]=[]
            if r['label']==0 and r['segment_samples']>=900:
                exposure+=(r['segment_samples']-90)/30
        elif r['step']!=rows[i-1]['step']+1:
            raise ValueError('Evaluation deadlines must be consecutive')
        step=r['step']; suspicious=bool(trigger[i]>=threshold)
        if state and step-last>=quiet_steps and not suspicious:state=False
        if state and step-opened>=burst_steps:
            # A persistent suspicious stream extends the same episode. Count
            # maintenance rearms separately, with no artificial blind cooldown.
            if suspicious:cap_rearms+=1;opened=step
            else:state=False
        if suspicious:
            if not state:
                wakes+=1;opened=step
                if r['label']==0 and r['segment_samples']>=900:negative_wakes+=1
            state=True;last=step
        active[i]=state;segments[key].append(i)
        if r['label']==0 and r['segment_samples']>=900:
            negative_windows+=1;negative_active+=int(state)
    events=[];recordings={}
    for i,r in enumerate(rows):
        record=recordings.setdefault(r['key'],dict(label=r['label'],domain=r['domain'],trigger=False,baseline=False,cascade=False))
        record['trigger']|=bool(trigger[i]>=threshold)
        record['baseline']|=bool(full[i]>=full_threshold)
        record['cascade']|=bool(active[i] and full[i]>=full_threshold)
    for key,indices in segments.items():
        rr=[rows[i] for i in indices]
        if not rr[0]['label']:continue
        flags=full[indices]>=full_threshold
        starts=np.flatnonzero(flags & ~np.r_[False,flags[:-1]])
        for start in starts:
            candidates=[j for j in range(int(start),min(len(indices),int(start)+max_delay_steps+1))
                        if active[indices[j]] and flags[j]]
            delay=(candidates[0]-int(start))*.25 if candidates else None
            events.append(dict(key=key[0],segment=key[1],domain=rr[0]['domain'],
                               reference_step=rr[int(start)]['step'],added_delay_seconds=delay,
                               timely=bool(candidates)))
    labels=np.asarray([r['label'] for r in rows])
    domains=sorted({r['domain'] for r in rows})
    result=dict(trigger_windows=binary_metrics(labels,trigger,threshold),
                baseline_windows=binary_metrics(labels,full,full_threshold),
                cascade_windows=binary_metrics(labels,np.where(active,full,-1),full_threshold),
                normal_to_suspicious_wakes=wakes,burst_cap_rearms=cap_rearms,
                continuous_negative_hours=exposure/3600,false_wakes=negative_wakes,
                false_wakes_per_hour=negative_wakes/(exposure/3600) if exposure else None,
                negative_active_fraction=negative_active/negative_windows if negative_windows else None,
                full_invocations=int(active.sum()),full_invocation_fraction=float(active.mean()) if len(active) else None,
                baseline_reference_events=len(events),lost_or_late_reference_events=sum(not e['timely'] for e in events),
                per_source_reference_recall={d: float(np.mean([e['timely'] for e in events if e['domain']==d]))
                    if any(e['domain']==d for e in events) else None for d in domains},
                per_source_trigger_recording_recall={d:float(np.mean([r['trigger'] for r in recordings.values() if r['domain']==d and r['label']]))
                    if any(r['domain']==d and r['label'] for r in recordings.values()) else None for d in domains},
                recording_metrics={name:binary_metrics([r['label'] for r in recordings.values()],
                    [float(r[name]) for r in recordings.values()],.5) for name in ('trigger','baseline','cascade')},
                reference='First positive window of each frozen V5 positive run; not ground-truth fall onset',
                events=events)
    return result,active


def select_threshold(rows,trigger,full,full_threshold,min_recall=.995):
    """Validation-only search; prioritize timely per-source event retention."""
    candidates=np.unique(np.asarray(trigger,np.float32))[::-1]
    selected=None;low=0;high=len(candidates)-1
    # Retention is monotone: lowering the threshold can only enlarge awake
    # intervals. Binary search avoids quadratic full-stream replay.
    while low<=high:
        middle=(low+high)//2;threshold=candidates[middle]
        report,active=simulate(rows,trigger,full,float(threshold),full_threshold)
        recalls=[v for v in report['per_source_reference_recall'].values() if v is not None]
        recording_recalls=[v for v in report['per_source_trigger_recording_recall'].values() if v is not None]
        if recalls and recording_recalls and min(recalls)>=min_recall and min(recording_recalls)>=min_recall and report['lost_or_late_reference_events']==0:
            selected=dict(threshold=float(threshold),report=report,
                          qualified=report['negative_active_fraction'] is not None
                          and report['negative_active_fraction']<1/3
                          and report['lost_or_late_reference_events']==0)
            high=middle-1
        else:
            low=middle+1
    if selected is None:
        return dict(threshold=None,qualified=False,reason='No timely baseline-event-preserving threshold')
    selected['hardware_deadline_qualified']=False
    selected['commercial_qualified']=False
    return selected
