"""Locked seed42 V4 PTQ: training-only calibration, joint validation, exhaustive windows.
Run --phase smoke, export, validation, test in order, in a separate export env.
Private windows/replays are written under --output, never into Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch

import train as base
import training_v2 as v2
import training_v4 as v4
from collections import Counter
import time

EXPECTED_CHECKPOINT = '3c92a68f6e1440a83880665d6634b01e7cb5724374a7e3f7a3eb76892e210b2e'
SUPPORTED_OPS = set('TRANSPOSE CONV_2D RESHAPE CONCATENATION PACK STRIDED_SLICE ADD SOFTMAX MUL SUM MEAN SUB SQRT DIV NEG SQUARED_DIFFERENCE RSQRT FULLY_CONNECTED QUANTIZE DEQUANTIZE GELU'.split())


def build_tf(tf, state):
    """An exact inference-only TensorFlow translation with frozen BatchNorm."""
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}

    def conv(x, key, bn=None, dilation=1):
        w = weights[key+'.weight'].transpose(2, 1, 0)
        b = weights.get(key+'.bias', np.zeros(w.shape[-1], np.float32))
        if bn:
            scale = weights[bn+'.weight']/np.sqrt(weights[bn+'.running_var']+1e-5)
            w = w*scale[None, None, :]
            b = (b-weights[bn+'.running_mean'])*scale+weights[bn+'.bias']
        y = tf.nn.conv2d(tf.expand_dims(x, 1), tf.constant(w[None]),
                         strides=[1, 1, 1, 1], padding='SAME',
                         dilations=[1, 1, dilation, 1])
        return tf.squeeze(tf.nn.bias_add(y, tf.constant(b)), 1)

    def dense(x, key):
        return tf.linalg.matmul(x, tf.constant(weights[key+'.weight'].T))+tf.constant(weights[key+'.bias'])

    def layer_norm(x, key):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.math.squared_difference(x, mean), axis=-1, keepdims=True)
        y = (x-mean)*tf.math.rsqrt(variance+1e-5)
        return y*tf.constant(weights[key+'.weight'])+tf.constant(weights[key+'.bias'])

    def encoder(x, key):
        x = tf.nn.gelu(conv(x, key+'.stem.0', key+'.stem.1'), approximate=False)
        blocks = sorted({int(k.split('.')[2]) for k in weights if k.startswith(key+'.tcn.')})
        for i in blocks:
            p = key+f'.tcn.{i}'
            y = tf.nn.gelu(conv(x, p+'.c1', p+'.b1', 2**i), approximate=False)
            y = conv(y, p+'.c2', p+'.b2', 2**i)
            x = tf.nn.gelu(x+y, approximate=False)
        attention = conv(tf.nn.gelu(conv(x, key+'.score.0'), approximate=False), key+'.score.2')
        attention = tf.nn.softmax(attention, axis=1)
        return tf.reduce_sum(x*attention, axis=1)

    def segment(x):
        h = tf.concat([encoder(x[:, :, :3], 'acc'), encoder(x[:, :, 3:], 'gyr')], axis=-1)
        return tf.nn.gelu(dense(layer_norm(h, 'fuse.0'), 'fuse.1'), approximate=False)

    class Classifier(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 90, 6], tf.float32, name='normalized_imu')])
        def __call__(self, x):
            tokens = tf.stack([segment(z) for z in [x, x[:, :30], x[:, 30:60], x[:, 60:90]]], axis=1)
            score = dense(tf.nn.gelu(dense(layer_norm(tokens, 'token.0'), 'token.1'), approximate=False), 'token.3')
            attention = tf.nn.softmax(score, axis=1)
            hidden = tf.reduce_sum(tokens*attention, axis=1)
            return dense(hidden, 'cls')

    return Classifier()


def torch_logits(model, x):
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            z = torch.from_numpy(x[start:start+128])
            out.append(model(z, z[:, :30], z[:, 30:60], z[:, 60:90])['logits'].numpy())
    return np.concatenate(out)


def probabilities(logits):
    z = logits-logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p[:, 1]/p.sum(axis=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize(x, scale, zero):
    # float division then C++ lround: halfway away from zero, zero point afterward.
    z = np.asarray(x, np.float32)/np.float32(scale)
    rounded = np.copysign(np.floor(np.abs(z).astype(np.float64)+.5), z)
    return np.clip(rounded+zero, -128, 127).astype(np.int8)


def interpreter(tf, data):
    it = tf.lite.Interpreter(model_content=data, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def lite(it, x):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    result, raw = [], []
    for z in x:
        q = quantize(z, *inp['quantization']) if inp['dtype'] == np.int8 else z
        it.set_tensor(inp['index'], q[None]); it.invoke()
        y = it.get_tensor(out['index'])[0]; raw.append(y.copy())
        result.append((y.astype(np.float32)-out['quantization'][1])*out['quantization'][0]
                      if out['dtype'] == np.int8 else y)
    return np.asarray(result), np.asarray(raw)


def inventory(it):
    tt = it.get_tensor_details(); ops = Counter(o['op_name'] for o in it._get_ops_details())
    floats = [t['name'] for t in tt if np.issubdtype(t['dtype'], np.floating)]
    unsupported = set(ops)-SUPPORTED_OPS
    if floats or unsupported:
        raise ValueError(f'Unsupported integer Micro graph: float tensors={floats}, ops={unsupported}')
    convs = [o for o in it._get_ops_details() if o['op_name'] == 'CONV_2D']
    tensors = {t['index']: t for t in tt}
    channels = [len(tensors[o['inputs'][1]]['quantization_parameters']['scales']) for o in convs]
    if not convs or any(n < 1 for n in channels):
        raise ValueError('Convolution weight quantization absent')
    return {'operators': dict(ops), 'tensor_types': dict(Counter(t['dtype'].__name__ for t in tt)),
            'float_tensors': floats, 'per_channel_conv_scale_counts': channels,
            'input_scale_zero': list(it.get_input_details()[0]['quantization']),
            'output_scale_zero': list(it.get_output_details()[0]['quantization'])}


def load(args):
    if sha(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Expected the exact V4 locked checkpoint')
    c = torch.load(args.checkpoint, weights_only=False, map_location='cpu'); cfg = base.Cfg(**c['config'])
    if c.get('smoke', True) or (cfg.seed,cfg.fs,cfg.win_sec,cfg.stride_sec,cfg.channels,cfg.blocks) != (42,30,3,.25,24,3):
        raise ValueError('Wrong architecture or smoke checkpoint')
    split = json.loads((args.work/'splits/group_splits.json').read_text())
    payload = {k:v for k,v in split.items() if k != 'fingerprint'}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != c['split_fingerprint'] or split['fingerprint'] != c['split_fingerprint']:
        raise ValueError('Split fingerprint mismatch')
    if json.loads((args.work/'normalization.json').read_text()) != c['normalization']:
        raise ValueError('Normalization file/checkpoint mismatch')
    m5 = v2.m5_records(args.m5_root)
    for r in m5: r['normalization_domain'] = v2.M5
    records = v2.own_records(args.own_root)+m5+v2.public_records(args.public_root)
    if v2.fingerprint(records) != split['input_fingerprint']:
        raise ValueError('Input fingerprint mismatch')
    lookup = {r['key']:r for r in records}
    partitions = {role:[lookup[k] for k in keys] for role,keys in split['splits'].items()}
    for d,s in c['normalization'].items():
        rr = [r for r in partitions['train'] if r['domain'] == d]
        if s['fit_partition'] != 'train' or s['fit_recordings'] != len(rr) or np.any(np.asarray(s['std']) <= 0):
            raise ValueError('Normalization provenance mismatch: '+d)
        if d == v2.M5 and s['fit_keys'] != sorted(r['key'] for r in rr):
            raise ValueError('M5 normalization training keys mismatch')
    if c['preprocessing']['software_clipping'] or c['preprocessing']['conversion_convention'] != 'si':
        raise ValueError('M5 must use full-range SI inputs')
    model = base.Net(cfg).eval(); model.load_state_dict(c['model'])
    return c,cfg,partitions,model


def calibrate(args,c,rs):
    rng = np.random.default_rng(42); windows, manifest = [], []
    arr = {}; source_counts = Counter()
    for domain,mass in c['source_weights'].items():
        pool = [r for r in rs if r['domain'] == domain]
        classes = sorted({r['label'] for r in pool})
        n = int(round(args.calibration*mass))
        selections = []
        if domain == v2.M5:
            # Training-only difficult motion: every session, each channel's absolute peak.
            for r in pool:
                x = arr.setdefault(r['key'], v2.read_array(r))
                for ch in range(6):
                    peak = int(np.argmax(np.abs(x.values[:,ch])))
                    lo,hi = next((lo,hi) for lo,hi in x.ranges if lo <= peak < hi)
                    selections.append((r,int(np.clip(peak-45,lo,hi-90)),'channel_peak'))
        for i in range(n-len(selections)):
            y = classes[i%len(classes)]; candidates = [r for r in pool if r['label'] == y]
            groups = sorted({r['group'] for r in candidates})
            group = groups[int(rng.integers(len(groups)))]; candidates = [r for r in candidates if r['group'] == group]
            r = candidates[int(rng.integers(len(candidates)))]
            if r['key'] not in arr: arr[r['key']] = v2.read_array(r)
            x = arr[r['key']]; counts = x.ranges[:,1]-x.ranges[:,0]-89
            offset = int(rng.integers(int(counts.sum()))); cumulative = np.cumsum(counts)
            j = int(np.searchsorted(cumulative,offset,side='right'))
            start = int(x.ranges[j,0]+offset-(cumulative[j-1] if j else 0))
            selections.append((r,start,'random_group_class_balanced'))
        mean,std = (np.asarray(c['normalization'][domain][k],np.float32) for k in ('mean','std'))
        for r,start,reason in selections:
            z = (arr[r['key']][start:start+90]-mean)/std
            assert z.shape == (90,6) and np.isfinite(z).all()
            windows.append(z); source_counts[domain] += 1
            manifest.append({'key':r['key'],'domain':domain,'label':r['label'],'start':start,'reason':reason,'partition':'train'})
    x = np.asarray(windows,np.float32)
    np.save(args.output/'calibration.npy',x)
    v2.write_json(args.output/'calibration_manifest.json',manifest)
    return x, {'seed':42,'source_windows':dict(source_counts),'source_label_windows':dict(Counter(f"{r['domain']}/{r['label']}" for r in manifest)),
               'unique_recordings':len({r['key'] for r in manifest}),'manifest_sha256':sha(args.output/'calibration_manifest.json'),
               'min_by_channel':x.min((0,1)).tolist(),'max_by_channel':x.max((0,1)).tolist(),'test_or_validation_used':False}


def metrics(y,p,threshold):
    y=np.asarray(y); positive=p>=threshold
    tp=int(np.sum(positive & (y==1))); fn=int(np.sum(~positive & (y==1)))
    fp=int(np.sum(positive & (y==0))); tn=int(np.sum(~positive & (y==0)))
    ratio=lambda a,b: a/b if b else None
    return {'windows':len(y),'tp':tp,'fn':fn,'fp':fp,'tn':tn,'precision':ratio(tp,tp+fp) if tp+fn else None,
            'recall':ratio(tp,tp+fn),'f1':ratio(2*tp,2*tp+fn+fp) if tp+fn else None,
            'specificity':ratio(tn,tn+fp),'negative_window_fpr':ratio(fp,fp+tn), 'accuracy':ratio(tp+tn,len(y))}


def error_stats(a,b):
    z=np.abs(a-b)
    return {'mean_absolute':float(z.mean()),'p99_absolute':float(np.quantile(z,.99)),'max_absolute':float(z.max())}


def evaluate(args,c,cfg,rs,model,tf,role):
    lock=json.loads((args.output/'export_lock.json').read_text())
    assert sha(args.output/'model_int8.tflite') == lock['model_sha256']
    it=interpreter(tf,(args.output/'model_int8.tflite').read_bytes())
    scale,zero=it.get_input_details()[0]['quantization']; details={}; reports={}; all_y=[]; all_f=[]; all_q=[]
    for d in c['source_weights']:
        rr=[r for r in rs if r['domain']==d]; arr=v2.arrays(rr,args.output/'cache')
        ds=v2.Clips(rr,arr,c['normalization'],cfg)
        x=np.asarray([ds[i]['full'].numpy() for i in range(len(ds))],np.float32)
        y=np.asarray([rr[i]['label'] for i,start in ds.entries]); keys=[rr[i]['key'] for i,start in ds.entries]
        start=time.monotonic(); f=torch_logits(model,x); q,raw=lite(it,x); pf,pq=probabilities(f),probabilities(q)
        np.save(args.output/f'{role}_{d}_inputs.npy',quantize(x,scale,zero))
        np.savez_compressed(args.output/f'{role}_{d}_outputs.npz',float_logits=f,int8_logits=q,int8_raw=raw,labels=y,keys=np.asarray(keys),starts=np.asarray([s for i,s in ds.entries]))
        rows=[{'key':k,'file_label':int(yy),'prob':float(p),'start':s} for k,yy,p,(_,s) in zip(keys,y,pq,ds.entries)]
        kk,yy,pp=base.aggregate(rows); details[d]=(rows,kk,yy,pp)
        unbounded=np.copysign(np.floor(np.abs(x/np.float32(scale)).astype(np.float64)+.5),x)+zero
        sat=(unbounded < -128)|(unbounded > 127)
        reports[d]={'float':metrics(y,pf,c['threshold']),'int8_unchanged_threshold':metrics(y,pq,c['threshold']),
                    'int8_locked_threshold':metrics(y,pq,lock['threshold']), 'logit_error':error_stats(f,q),
                    'probability_error':error_stats(pf,pq),'input_saturated_values':int(sat.sum()),'input_values':int(x.size),
                    'windows_with_saturation':int(sat.any((1,2)).sum()),'elapsed_seconds':time.monotonic()-start}
        all_y.extend(y); all_f.extend(pf); all_q.extend(pq)
        print(role,d,json.dumps(reports[d]),flush=True)
    reports['pooled']={'float':metrics(all_y,np.asarray(all_f),c['threshold']),
                       'int8_locked_threshold':metrics(all_y,np.asarray(all_q),lock['threshold'])}
    result={'unit':'complete 3-second windows; exhaustive cumulative 0.25-second grid plus final window; overlaps correlated',
            'threshold':lock['threshold'],'float_threshold':c['threshold'],'domains':reports}
    if role=='validation':
        gates=json.loads((args.work/'baseline_validation.json').read_text())['constraints']
        result['constraints']=gates
        result['unchanged_threshold_selection']=v4.select_threshold(details,rs,c['source_weights'],gates,fixed=c['threshold'])
        # Report validation alternative, but preserve original threshold for the first PTQ package.
        result['validation_only_optimum']=v4.select_threshold(details,rs,c['source_weights'],gates)
        result['threshold_decision']='Retain float threshold for first PTQ comparison; alternative not deployed or tested'
    if role=='test' and reports['pooled']['float']['windows'] != 66539:
        raise ValueError('Held-out window count differs from locked experiment')
    v2.write_json(args.output/f'{role}_report.json',result)
    return result


def export(args,c,cfg,rs,model,tf):
    x,coverage=calibrate(args,c,rs)
    module=build_tf(tf,c['model']); concrete=module.__call__.get_concrete_function()
    # Training-only parity: all sources, both classes, M5 extremes.
    ii=np.linspace(0,len(x)-1,min(128,len(x))).round().astype(int); verify=x[ii]
    a=torch_logits(model,verify); b=np.concatenate([module(z[None]).numpy() for z in verify])
    print('PYTORCH_TF_PARITY',error_stats(a,b),flush=True)
    if not np.allclose(a,b,atol=2e-4,rtol=2e-4): raise ValueError('TF parity failed')
    converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],module)
    float_data=converter.convert(); f,_=lite(interpreter(tf,float_data),verify)
    if not np.allclose(a,f,atol=3e-4,rtol=3e-4): raise ValueError('Float Lite parity failed')
    (args.output/'model_float32.tflite').write_bytes(float_data)
    converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],module)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=lambda: ([z[None]] for z in x)
    converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    data=converter.convert() # No silent float fallback.
    it=interpreter(tf,data); inv=inventory(it)
    (args.output/'model_int8.tflite').write_bytes(data)
    q,raw=lite(it,verify)
    report={'checkpoint_sha256':EXPECTED_CHECKPOINT,'model_sha256':sha(args.output/'model_int8.tflite'),
            'split_fingerprint':c['split_fingerprint'],'normalization_sha256':sha(args.work/'normalization.json'),
            'conversion_mode':'integer-only','tensorflow_version':tf.__version__,'torch_version':torch.__version__,
            'model_bytes':len(data),'float_bytes':len(float_data),'calibration':coverage,'inventory':inv,
            'pytorch_tf_logit_error':error_stats(a,b),'pytorch_float_tflite_logit_error':error_stats(a,f),
            'calibration_int8_logit_error':error_stats(a,q),'threshold':c['threshold'],'threshold_policy':'unchanged_float_threshold',
            'locked_before_test':True}
    v2.write_json(args.output/'export_lock.json',report)
    print('EXPORT_LOCK',json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','work','own-root','public-root','m5-root','output'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--phase',choices=['smoke','export','validation','test'],required=True)
    ap.add_argument('--calibration',type=int,default=2048)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); c,cfg,parts,model=load(args)
    if args.phase=='smoke':
        assert quantize(np.array([-.5,.5,-1.5,1.5],np.float32),1,0).tolist()==[-1,1,-2,2]
        for domain in c['source_weights']:
            r=next(r for r in parts['train'] if r['domain']==domain); arr={r['key']:v2.read_array(r)}
            ds=v2.Clips([r],arr,c['normalization'],cfg,smoke=True)
            x=np.asarray([ds[i]['full'].numpy() for i in range(len(ds))]); assert np.isfinite(torch_logits(model,x)).all()
        v2.write_json(args.output/'smoke.json',{'passed':True,'checkpoint_sha256':EXPECTED_CHECKPOINT,'sources':list(c['source_weights'])})
        print('SMOKE_PASS',flush=True); return
    if not (args.output/'smoke.json').exists(): raise ValueError('Run fast smoke phase first')
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2); tf.config.threading.set_intra_op_parallelism_threads(4)
    if args.phase=='export':
        if (args.output/'export_lock.json').exists(): raise FileExistsError('Export already locked')
        export(args,c,cfg,parts['train'],model,tf)
    else:
        if args.phase=='test' and not (args.output/'validation_report.json').exists(): raise ValueError('Validate before final test')
        if args.phase=='test' and (args.output/'test_report.json').exists(): raise FileExistsError('Final test already evaluated; use saved predictions for analysis')
        evaluate(args,c,cfg,parts['val' if args.phase=='validation' else 'test'],model,tf,args.phase)

if __name__=='__main__': main()
