#!/usr/bin/env python3
"""Fair PTQ/QAT validation, pre-test selection lock, exhaustive Micro reporting."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import export_v4_tflite as e
import qat_v4 as q
from prepare_m5_combined_v5 import records


def load(args):
    lock=json.loads((args.work/'training_lock.json').read_text())
    if lock['protocol_version']!=5 or lock['seed']!=42:
        raise ValueError('Require a completed, locked V5 seed42 run')
    c,cfg,parts,model=e.load(args,expected_checkpoint=lock['checkpoint_sha256'],m5_loader=records)
    if c.get('protocol_version')!=5 or c['split_fingerprint']!=lock['split_fingerprint']:
        raise ValueError('Wrong V5 checkpoint or split lock')
    if e.sha(args.work/'normalization.json')!=lock['normalization_sha256']:
        raise ValueError('Wrong normalization lock')
    return c,cfg,parts,model


def float_logits(model,x):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            z=torch.from_numpy(x[start:start+128]).to(device)
            result.append(model(z,z[:,:30],z[:,30:60],z[:,60:90])['logits'].cpu().numpy())
    return np.concatenate(result)


def summarize(directory,datasets,threshold,model=None,float_threshold=None):
    reports={};all_y=[];all_p=[];all_float=[]
    for d,(x,y,keys,starts) in datasets.items():
        z=np.load(directory/f'{d}.predictions.npz')
        for k,v in [('labels',y),('keys',keys),('starts',starts)]:
            if not np.array_equal(z[k],v):raise ValueError('Prediction order mismatch')
        p=e.probabilities(z['logits']);report={'int8':e.metrics(y,p,threshold)}
        if model is not None:
            logits=float_logits(model,x);fp=e.probabilities(logits)
            report['float']=e.metrics(y,fp,float_threshold)
            report['probability_abs_mean']=float(np.abs(p-fp).mean())
            report['probability_abs_max']=float(np.abs(p-fp).max())
            np.savez_compressed(directory/f'{d}.float_predictions.npz',logits=logits,labels=y,keys=keys,starts=starts)
            all_float.extend(fp)
        reports[d]=report;all_y.extend(y);all_p.extend(p)
    reports['pooled']={'int8':e.metrics(all_y,np.asarray(all_p),threshold)}
    if model is not None:reports['pooled']['float']=e.metrics(all_y,np.asarray(all_float),float_threshold)
    return reports


def export(args,c,cfg,parts,model):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'export_lock.json').exists():raise FileExistsError('Never overwrite an export')
    assert e.quantize(np.array([-.5,.5,-1.5,1.5],np.float32),1,0).tolist()==[-1,1,-2,2]
    # Fast source/class/shape/parity checks precede full representative processing.
    for d in c['source_weights']:
        for y in sorted({r['label'] for r in parts['train'] if r['domain']==d}):
            r=next(r for r in parts['train'] if r['domain']==d and r['label']==y)
            ds=e.v2.Clips([r],e.v2.arrays([r],args.output/'cache'),c['normalization'],cfg,smoke=True)
            if not np.isfinite(e.torch_logits(model,np.asarray([ds[0]['full'].numpy()]))).all():
                raise ValueError('Nonfinite export smoke output')
    e.v2.write_json(args.output/'smoke.json',{'passed':True,'checkpoint_sha256':e.sha(args.checkpoint),'checks':'all source/classes, shape, finite inference, half-away rounding'})
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(4)
    e.export(args,c,cfg,parts['train'],model,tf)


def validation(args,c,cfg,parts,model):
    lock=json.loads((args.output/'export_lock.json').read_text());inv=lock['inventory']
    if lock['checkpoint_sha256']!=e.sha(args.checkpoint) or lock['model_sha256']!=e.sha(args.output/'model_int8.tflite'):
        raise ValueError('Export identity mismatch')
    protocol=json.loads((args.work/'selection_protocol.json').read_text())
    data=q.make_datasets(parts['val'],c,cfg,args.output/'cache')
    if sum(len(v[0]) for v in data.values())!=c['validation_windows']:raise ValueError('Changed exhaustive validation count')
    # Actual Micro smoke for all seven sources and both eligible classes.
    smoke=q.make_datasets(parts['val'],c,cfg,args.output/'cache',smoke=True)
    q.micro_validate(args.output/'model_int8.tflite',inv,smoke,parts['val'],c,protocol['constraints'],args.runner,args.output/'micro_smoke')
    result=q.micro_validate(args.output/'model_int8.tflite',inv,data,parts['val'],c,protocol['constraints'],args.runner,
                            args.output/'validation',protocol['window_recall_floors'])
    threshold=result['selection']['threshold']
    reports=summarize(args.output/'validation',data,threshold,model,c['threshold'])
    e.v2.write_json(args.output/'validation_report.json',dict(domains=reports,selection=result['selection'],
        unchanged_threshold_selection=result['original_threshold_selection'],constraints=protocol['constraints'],
        runtime=result['runtime'],test_used=False,threshold_selection='same rules as QAT; float32 thresholds'))
    e.v2.write_json(args.output/'candidate.json',dict(method='PTQ',model_path=str(args.output/'model_int8.tflite'),
        model_sha256=e.sha(args.output/'model_int8.tflite'),inventory=inv,selection=result['selection'],
        selection_protocol_sha256=e.sha(args.work/'selection_protocol.json')))


def lock(args,c):
    if (args.output/'selection_lock.json').exists():raise FileExistsError('Candidate choice already locked')
    ptq=json.loads((args.ptq_export/'candidate.json').read_text())
    selected=json.loads((args.qat_run/'selected.json').read_text())
    protocol=json.loads((args.qat_run/'protocol.json').read_text())
    expected=e.sha(args.work/'selection_protocol.json')
    if ptq['selection_protocol_sha256']!=expected or protocol['selection_protocol_sha256']!=expected:
        raise ValueError('PTQ/QAT were selected using different validation gates')
    if protocol['checkpoint_sha256']!=e.sha(args.checkpoint) or protocol['split_fingerprint']!=c['split_fingerprint']:
        raise ValueError('QAT used a different float checkpoint or split')
    folder=args.qat_run/f"epoch_{selected['epoch']:02d}"
    qat=dict(method='QAT',model_path=str(folder/'model_int8.tflite'),model_sha256=selected['model_sha256'],
             inventory=selected['inventory'],selection=selected['selection'],selected_epoch=selected['epoch'])
    for z in [ptq,qat]:
        if e.sha(z['model_path'])!=z['model_sha256']:raise ValueError('Candidate model checksum mismatch')
    eligible=[z for z in [ptq,qat] if z['selection']['valid_under_constraints']]
    result=dict(candidates=[ptq,qat],test_used_for_selection=False,checkpoint_sha256=e.sha(args.checkpoint),
                split_fingerprint=c['split_fingerprint'],rule='valid gates first; lowest validation seven-source weighted error; PTQ wins exact ties',
                eligible=bool(eligible))
    if eligible:
        winner=min(eligible,key=lambda z:(z['selection']['weighted_error'],z['method']!='PTQ'))
        result['winner']=winner;shutil.copy2(winner['model_path'],args.output/'model_int8.tflite')
        if winner['method']=='QAT':shutil.copy2(folder/'qat_checkpoint.pt',args.output/'qat_checkpoint.pt')
    e.v2.write_json(args.output/'selection_lock.json',result)
    print('PRE_TEST_SELECTION_LOCK',json.dumps(result),flush=True)
    if not eligible:raise ValueError('No INT8 candidate passed the predeclared validation gates; refuse deployment')


def test(args,c,cfg,parts,model):
    lockdata=json.loads((args.output/'selection_lock.json').read_text())
    if (args.output/'test_report.json').exists():raise FileExistsError('Held-out test already evaluated')
    if not lockdata['eligible'] or lockdata['checkpoint_sha256']!=e.sha(args.checkpoint):raise ValueError('Wrong or invalid pre-test choice')
    data=q.make_datasets(parts['test'],c,cfg,args.output/'cache'); reports={}
    for candidate in lockdata['candidates']:
        name=candidate['method'];directory=args.output/name.lower()/'test'
        if e.sha(candidate['model_path'])!=candidate['model_sha256']:raise ValueError('Candidate changed after lock')
        # Fixed thresholds only: these tests cannot change the winner.
        result=q.micro_validate(Path(candidate['model_path']),candidate['inventory'],data,parts['test'],c,{},args.runner,directory,
                                fixed_threshold=candidate['selection']['threshold'])
        reports[name]=summarize(directory,data,candidate['selection']['threshold'],model if name=='PTQ' else None,c['threshold'])
        episodes=[]
        z=np.load(directory/(e.v2.M5+'.predictions.npz'));prob=e.probabilities(z['logits'])
        for r in parts['test']:
            if r['domain']!=e.v2.M5:continue
            mask=z['keys']==r['key'];p=prob[mask];starts=z['starts'][mask]
            positive=p>=candidate['selection']['threshold'];n=sum(bool(v) and (i==0 or not positive[i-1] or starts[i]-starts[i-1]>8) for i,v in enumerate(positive))
            signal=e.v2.read_array(r);duration=sum(hi-lo-1 for lo,hi in signal.ranges)/cfg.fs
            episodes.append(dict(key=r['key'],label=r['label'],window_count=len(p),positive_windows=int(positive.sum()),
                 event_detected=bool(positive.any()) if r['label']==1 else None,
                 negative_false_alarm_episodes=int(n) if r['label']==0 else None,
                 negative_episodes_per_hour=n*3600/duration if r['label']==0 else None,duration_seconds=duration))
        e.v2.write_json(directory/'m5_events.json',{'recordings':episodes,'policy':'consecutive positive 0.25s starts merged; not firmware alarm/debounce replay'})
    # Honest previous-float comparison, with its original M5 normalization.
    old=torch.load(args.comparison_checkpoint,map_location='cpu',weights_only=False)
    if e.sha(args.comparison_checkpoint)!='3c92a68f6e1440a83880665d6634b01e7cb5724374a7e3f7a3eb76892e210b2e':raise ValueError('Wrong prior float checkpoint')
    oldmodel=e.base.Net(cfg);oldmodel.load_state_dict(old['model'])
    olddata=q.make_datasets(parts['test'],old,cfg,args.output/'cache')
    reports['previous_V4_float']={d:e.metrics(y,e.probabilities(float_logits(oldmodel,x)),old['threshold']) for d,(x,y,keys,starts) in olddata.items()}
    result=dict(winner=lockdata['winner']['method'],winner_sha256=lockdata['winner']['model_sha256'],
                candidates=reports,test_used_for_selection=False,
                units='complete 3-second windows; 0.25-second cumulative grid plus final; overlapping windows correlated',
                split_limitations=c['preprocessing']['split_limitations'],
                benchmark_limitations='original public/private and M5 negative holdouts reused from earlier experiments; three new fall recording holdouts share the positive acquisition boot',
                physical_device_verified=False,commercial_readiness_established=False)
    e.v2.write_json(args.output/'test_report.json',result)
    gallery=[r for r in parts['train'] if r['domain']=='own'];queries=[r for r in parts['test'] if r['domain']=='own']
    cache=args.output/'cache';device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    retrieval=e.v2.retrieval(model.to(device),e.v2.Clips(gallery,e.v2.arrays(gallery,cache),c['normalization'],cfg),
        e.v2.Clips(queries,e.v2.arrays(queries,cache),c['normalization'],cfg),cfg,device,e.base)
    e.v2.write_json(args.output/'retrieval_report.json',dict(fine_tune_float_class_retrieval=retrieval,
                zero_shot=None,zero_shot_reason='all seven sources participate in training'))
    package(args,c,data,lockdata)


def package(args,c,data,lockdata):
    winner=lockdata['winner'];inv=winner['inventory'];scale,zero=inv['input_scale_zero'];oscale,ozero=inv['output_scale_zero']
    threshold=winner['selection']['threshold'];model=args.output/'model_int8.tflite'
    if e.sha(model)!=winner['model_sha256']:raise ValueError('Package model identity mismatch')
    mean,std=(np.asarray(c['normalization'][e.v2.M5][k],np.float32) for k in ['mean','std'])
    def norm(counts):
        z=counts.astype(np.float32).copy();z[:,:3]=z[:,:3]*np.float32(8/32768)*np.float32(9.80665)
        z[:,3:]=z[:,3:]*np.float32(2000/32768)*np.float32(np.pi/180)
        return (z-mean)/std
    calibration=np.load(args.ptq_export/'calibration.npy')
    arrays=[calibration[i] for i in np.linspace(0,len(calibration)-1,4).round().astype(int)]
    synthetic=np.zeros((3,90,6),np.int16);synthetic[1]=[-32768,32767,-32768,32767,-32768,32767]
    synthetic[2]=np.where(np.arange(90)[:,None]%2,32767,-32768)
    arrays.extend(norm(z) for z in synthetic)
    # Validation-only near-threshold replay, never selected from held-out tests.
    valdir=(args.ptq_export/'validation' if winner['method']=='PTQ' else
            args.qat_run/f"epoch_{winner['selected_epoch']:02d}"/'validation')
    valdata=q.make_datasets(load(args)[2]['val'],c,e.base.Cfg(**c['config']),args.output/'cache')
    near=[]
    for d,(x,y,keys,starts) in valdata.items():
        z=np.load(valdir/f'{d}.predictions.npz');prob=e.probabilities(z['logits'])
        for i in np.argsort(abs(prob-threshold))[:4]:near.append((abs(float(prob[i])-threshold),d,int(i),x[i]))
    arrays.extend(z[3] for z in sorted(near,key=lambda z:z[0])[:4]);arrays=np.asarray(arrays,np.float32)
    inputs=e.quantize(arrays,scale,zero);inp=args.output/'replay_inputs.bin';out=args.output/'replay_outputs.bin';inp.write_bytes(inputs.tobytes())
    import subprocess
    r=subprocess.run([str(args.runner),str(model),str(inp),str(out)],capture_output=True,text=True,check=True)
    if f'TFLM_WINDOWS_PASS={len(inputs)}' not in r.stdout:raise ValueError('Incomplete package replay')
    outputs=np.fromfile(out,np.int8).reshape(-1,2)
    gold=np.asarray([[0]*6,[-32768,32767,-32768,32767,-32768,32767],[32767,-32768,32767,-32768,32767,-32768],[1,-1,1,-1,1,-1],[12345,-12345,29999,-29999,10000,-10000]],np.int16)
    expected=e.quantize(norm(gold),scale,zero)
    header='#pragma once\n#include <cstdint>\nnamespace fall_v2 {\n'+f'constexpr int kReplayCount={len(inputs)};\n'
    for typ,name,z in [('int8_t','kReplayInputs',inputs.reshape(-1,540)),('int8_t','kReplayOutputs',outputs),('int16_t','kPreprocessCounts',gold),('int8_t','kPreprocessExpected',expected)]:
        header+=f'constexpr {typ} {name}[][{z.shape[1]}]={{\n'+',\n'.join('{'+','.join(map(str,row))+'}' for row in z)+'\n};\n'
    header+=f'constexpr int kPreprocessCount={len(gold)};\n}}\n';(args.output/'model_v2_replay.h').write_text(header)
    config='#pragma once\n// V5 exploratory M5 adaptation; validation lock precedes test.\nnamespace fall_v2 {\nconstexpr int kTimesteps=90,kChannels=6,kSampleHz=30;\nconstexpr long long kInferencePeriodUs=750000;\n'
    for name,value in [('kThreshold',threshold),('kGravity',9.80665),('kDegreesToRadians',np.pi/180),('kInputScale',scale),('kOutputScale',oscale)]:config+=f'constexpr float {name}={float(np.float32(value)):.12e}f;\n'
    config+=f'constexpr int kInputZero={zero},kOutputZero={ozero};\n'
    for name,values in [('kMean',mean),('kSigma',std)]:config+=f'constexpr float {name}[6]={{'+','.join(f'{float(v):.12e}f' for v in values)+'};\n'
    ck=args.output/'qat_checkpoint.pt' if winner['method']=='QAT' else args.checkpoint
    config+=f'constexpr char kCheckpointSha256[]="{e.sha(ck)}";\n}}\n';(args.output/'model_v2_config.h').write_text(config)
    shutil.copy2(model,args.output/'model.tflite')
    e.v2.write_json(args.output/'package_manifest.json',dict(method=winner['method'],model_sha256=e.sha(model),
        checkpoint_sha256=e.sha(ck),replay_cases=len(inputs),native_replay_passed=True,
        config_sha256=e.sha(args.output/'model_v2_config.h'),replay_sha256=e.sha(args.output/'model_v2_replay.h'),
        collector_firmware_built=False,physical_device_verified=False,commercial_readiness_established=False,
        float_internal_arithmetic='LayerNormV4 uses float32 sqrt internally' if winner['method']=='QAT' else None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','work','own-root','public-root','m5-root','output','runner','ptq-export','qat-run','comparison-checkpoint']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['export','validation','lock','test'],required=True)
    p.add_argument('--calibration',type=int,default=2048);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    c,cfg,parts,model=load(a)
    if a.phase=='export':export(a,c,cfg,parts,model)
    elif a.phase=='validation':validation(a,c,cfg,parts,model)
    elif a.phase=='lock':lock(a,c)
    else:test(a,c,cfg,parts,model)
