"""Compact per-source classification report; excludes raw/replay data and IDs."""
import argparse,json
from pathlib import Path
import numpy as np
from trigger_data_v5 import load,evaluation,digest
from review_trigger_margin_v5 import scores
from power_events_v5 import simulate

def main():
    p=argparse.ArgumentParser()
    for name in ('own-root','public-root','m5-root','checkpoint','split-from','experiment','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();lock=json.loads((a.experiment/'selection_lock.json').read_text());winner=lock['winner']
    if not winner:raise ValueError('Validation did not select a candidate')
    splits,arrays,ck,cfg,audit=load(a.own_root,a.public_root,a.m5_root,a.checkpoint,a.split_from,a.experiment/'cache')
    if audit['checkpoint_sha256']!=lock['checkpoint_sha256'] or audit['split_sha256']!=lock['split_sha256']:
        raise ValueError('Frozen experiment contract differs')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    from train_power_trigger import lite_metadata
    fm=lite_metadata(tf,a.experiment.parent/'collector/firmware/main/model.tflite')
    candidate=lock['candidates'][winner]
    if digest(a.experiment/winner/'trigger_int8.tflite')!=candidate['metadata']['sha256']:
        raise ValueError('Candidate identity differs')
    report=dict(candidate=winner,full_model_sha256=lock['full_model_sha256'],
        checkpoint_sha256=lock['checkpoint_sha256'],split_sha256=lock['split_sha256'],
        previous_test_seen=True,test_used_for_selection=False,deployment_eligible=False,
        commercial_qualified=False,zero_shot=None,retrieval=None,
        caveat='Recording labels and frozen V5 reference timing; no clinical onset annotations or independent participant qualification')
    keep=('trigger_windows','baseline_windows','cascade_windows','recording_metrics',
          'lost_or_late_reference_events','baseline_reference_events','negative_active_fraction',
          'false_wakes_per_hour','continuous_negative_hours','per_source_trigger_recording_recall')
    for role,prefix in (('val','validation'),('test','test')):
        _,rows=evaluation(splits,arrays,ck['normalization'],cfg,role)
        trigger=scores(a.experiment/winner/(prefix+'_micro/output.bin'),candidate['metadata']['output_scale'])
        full=scores(a.experiment/(prefix+'_full/output.bin'),fm['output_scale'])
        result=simulate(rows,trigger,full,candidate['selection']['threshold'],lock['full_threshold'])[0]
        report[role]=dict(overall={k:result[k] for k in keep},per_source={})
        for domain in sorted({r['domain'] for r in rows}):
            mask=np.asarray([r['domain']==domain for r in rows]);subset=[r for r in rows if r['domain']==domain]
            result=simulate(subset,trigger[mask],full[mask],candidate['selection']['threshold'],lock['full_threshold'])[0]
            report[role]['per_source'][domain]={k:result[k] for k in keep}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('COMPACT_PER_SOURCE_REPORT_READY',winner)

if __name__=='__main__':main()
