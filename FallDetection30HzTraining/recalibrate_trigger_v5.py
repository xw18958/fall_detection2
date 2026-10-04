"""Validation-only strict zero-reference-loss calibration of frozen candidates.

Corrects the earlier 99.5% event retention search to the actual zero-loss gate.
Locks the winner before auditing its previously examined test fold. No retrain,
test-based winner switching, preprocessing, label or split changes.
"""
import argparse,json
from pathlib import Path
from trigger_data_v5 import load,evaluation,digest
from power_events_v5 import select_threshold,simulate,quantized_threshold_margin
from review_trigger_margin_v5 import scores

def main():
    p=argparse.ArgumentParser()
    for name in ('own-root','public-root','m5-root','checkpoint','split-from','experiment','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();old=json.loads((a.experiment/'selection_lock.json').read_text())
    splits,arrays,ck,cfg,audit=load(a.own_root,a.public_root,a.m5_root,a.checkpoint,a.split_from,a.experiment/'cache')
    if audit['checkpoint_sha256']!=old['checkpoint_sha256'] or audit['split_sha256']!=old['split_sha256']:
        raise ValueError('Frozen V5 contract differs')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    from train_power_trigger import lite_metadata
    fm=lite_metadata(tf,a.experiment.parent/'collector/firmware/main/model.tflite')
    _,rows=evaluation(splits,arrays,ck['normalization'],cfg,'val')
    full=scores(a.experiment/'validation_full/output.bin',fm['output_scale'])
    candidates={}
    for name,c in old['candidates'].items():
        folder=a.experiment/name
        if digest(folder/'trigger_int8.tflite')!=c['metadata']['sha256']:raise ValueError('Candidate changed')
        scale=c['metadata']['output_scale'];pred=scores(folder/'validation_micro/output.bin',scale)
        selection=select_threshold(rows,pred,full,old['full_threshold'])
        if selection['threshold'] is not None:
            original=selection['threshold'];selection['threshold']=quantized_threshold_margin(original,scale,1)
            selection['report']=simulate(rows,pred,full,selection['threshold'],old['full_threshold'])[0]
            selection['qualified']=selection['report']['lost_or_late_reference_events']==0 and selection['report']['negative_active_fraction']<1/3
            selection['unadjusted_threshold']=original
        candidates[name]=dict(c,selection=selection)
    valid=[n for n,c in candidates.items() if c['selection']['qualified']]
    winner=min(valid,key=lambda n:(candidates[n]['selection']['report']['negative_active_fraction'],candidates[n]['metadata']['model_bytes'])) if valid else None
    result=dict(old,candidates=candidates,winner=winner,previous_test_seen=True,
        test_used_for_selection=False,deployment_eligible=False,commercial_qualified=False,
        protocol='Strict zero validation reference losses; predeclared +/-1 output LSB margin per logit; validation active fraction then model bytes',
        original_selection_sha256=digest(a.experiment/'selection_lock.json'),
        calibration_source_sha256=digest(Path(__file__)),event_policy_sha256=digest(Path(__file__).with_name('power_events_v5.py')))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'selection_lock.json').write_text(json.dumps(result,indent=2)+'\n')
    if winner:
        _,rows=evaluation(splits,arrays,ck['normalization'],cfg,'test')
        pred=scores(a.experiment/winner/'test_micro/output.bin',candidates[winner]['metadata']['output_scale'])
        full=scores(a.experiment/'test_full/output.bin',fm['output_scale'])
        report=simulate(rows,pred,full,candidates[winner]['selection']['threshold'],old['full_threshold'])[0]
        recalls=[v for v in report['per_source_trigger_recording_recall'].values() if v is not None]
        report['data_gate_passed']=bool(report['lost_or_late_reference_events']==0 and recalls and min(recalls)>=.995 and report['negative_active_fraction']<1/3)
        result['test']=report
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(winner=winner,test_gate=result.get('test',{}).get('data_gate_passed'),
        lost=result.get('test',{}).get('lost_or_late_reference_events'),
        normal_active_fraction=result.get('test',{}).get('negative_active_fraction')),indent=2))

if __name__=='__main__':main()
