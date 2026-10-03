"""Seed42 V4 QAT with a shared Torch/TF inference graph and Micro selection.

All observers and gradient updates use training data. Every epoch is exported
and selected using exhaustive validation predictions from the real Micro kernels.
The original checkpoint, normalization and partitions are immutable inputs.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import export_v4_tflite as e
import training_v2 as v2
import training_v4 as v4


def qparams(lo, hi):
    lo, hi = min(float(lo), 0.), max(float(hi), 0.)
    scale = max((hi-lo)/255., 1e-8)
    zero = int(np.clip(np.floor(-128-lo/scale+.5), -128, 127))
    return scale, zero, (-128-zero)*scale, (127-zero)*scale


class Graph:
    """One named operation topology for gradient training and TF export.

    Fake quantization models tensor rounding/clipping; it does not claim to
    emulate all integer accumulators, GELU LUTs or Micro requantization kernels.
    Selection therefore always invokes the exported model with Micro.
    """
    def __init__(self, state, ranges=None, tf=None, observe=False, quant=True, fused_norm=False):
        self.state, self.ranges, self.tf = state, ranges or {}, tf
        self.observe, self.quant = observe, quant
        self.fused_norm = fused_norm

    def act(self, x, key, softmax=False):
        if self.observe:
            lo, hi = float(x.detach().min()), float(x.detach().max())
            old = self.ranges.get(key, [0., 0.])
            self.ranges[key] = [min(old[0], lo), max(old[1], hi)]
            return x
        if not self.quant:
            return x
        lo, hi = [0., 255/256] if softmax else self.ranges[key]
        scale, zero, lo, hi = qparams(lo, hi)
        if self.tf:
            return self.tf.quantization.fake_quant_with_min_max_vars(x, lo, hi)
        return torch.fake_quantize_per_tensor_affine(x, scale, zero, -128, 127)

    def weight(self, w, axis):
        if not self.quant or self.observe:
            return w
        # The deployed Dense kernel uses per-tensor weights. Per-axis MatMul
        # fake quantization cannot be reliably lowered through its transpose.
        if axis is None:
            if self.tf:
                extent = self.tf.maximum(self.tf.reduce_max(self.tf.abs(w)), 1e-7)
                return self.tf.quantization.fake_quant_with_min_max_vars(w, -extent, extent, narrow_range=True)
            scale = float(w.detach().abs().max().clamp_min(1e-7))/127
            return torch.fake_quantize_per_tensor_affine(w, scale, 0, -127, 127)
        if self.tf:
            tf = self.tf
            axes = list(range(len(w.shape)-1))
            extent = tf.maximum(tf.reduce_max(tf.abs(w), axis=axes), 1e-7)
            return tf.quantization.fake_quant_with_min_max_vars_per_channel(
                w, -extent, extent, narrow_range=True)
        dims = tuple(i for i in range(w.ndim) if i != axis)
        scales = w.detach().abs().amax(dims).clamp_min(1e-7)/127
        return torch.fake_quantize_per_channel_affine(
            w, scales, torch.zeros(len(scales), dtype=torch.int32, device=w.device),
            axis, -127, 127)

    def conv(self, x, key, name, bn=None, dilation=1):
        w = self.state[key+'.weight']
        b = self.state.get(key+'.bias')
        if self.tf:
            tf = self.tf
            if bn:
                scale = self.state[bn+'.weight']*tf.math.rsqrt(self.state[bn+'.running_var']+1e-5)
                w = w*scale[:, None, None]
                b = (tf.zeros_like(scale) if b is None else b)
                b = (b-self.state[bn+'.running_mean'])*scale+self.state[bn+'.bias']
            w = self.weight(tf.transpose(w, [2, 1, 0])[None], -1)
            y = tf.nn.conv2d(tf.expand_dims(x,1), w, [1, 1, 1, 1], 'SAME', dilations=[1, 1, dilation, 1])
            if b is not None:
                y = tf.nn.bias_add(y, b)
            y = tf.squeeze(y,1)
        else:
            if bn:
                scale = self.state[bn+'.weight']*torch.rsqrt(self.state[bn+'.running_var']+1e-5)
                w = w*scale[:, None, None]
                b = torch.zeros_like(scale) if b is None else b
                b = (b-self.state[bn+'.running_mean'])*scale+self.state[bn+'.bias']
            w = self.weight(w, 0)
            y = F.conv1d(x.transpose(1, 2), w, b, padding=dilation*(w.shape[-1]//2), dilation=dilation).transpose(1, 2)
        return self.act(y, name)

    def dense(self, x, key, name):
        w, b = self.state[key+'.weight'], self.state[key+'.bias']
        if self.tf:
            w = self.weight(self.tf.transpose(w), None)
            y = self.tf.linalg.matmul(x, w)+b
        else:
            y = F.linear(x, self.weight(w, None), b)
        return self.act(y, name)

    def gelu(self, x, name):
        y = self.tf.nn.gelu(x, approximate=False) if self.tf else F.gelu(x)
        return self.act(y, name)

    def norm(self, x, key, name):
        tf = self.tf
        mean = tf.reduce_mean(x, -1, keepdims=True) if tf else x.mean(-1, keepdim=True)
        if self.fused_norm:
            variance = tf.reduce_mean(tf.math.squared_difference(x,mean), -1, keepdims=True) if tf else (x-mean).square().mean(-1, keepdim=True)
            inverse = tf.math.rsqrt(variance+1e-5) if tf else torch.rsqrt(variance+1e-5)
            z = self.act((x-mean)*inverse, name+'/standardized')
            z = self.act(z*self.state[key+'.weight'], name+'/scaled')
            return self.act(z+self.state[key+'.bias'], name+'/output')
        mean = self.act(mean, name+'/mean')
        variance = tf.math.squared_difference(x, mean) if tf else (x-mean).square()
        variance = self.act(variance, name+'/squared_difference')
        variance = tf.reduce_mean(variance, -1, keepdims=True) if tf else variance.mean(-1, keepdim=True)
        variance = self.act(variance, name+'/variance')
        variance = self.act(variance+1e-5, name+'/epsilon')
        inverse = tf.math.rsqrt(variance) if tf else torch.rsqrt(variance.clamp_min(1e-12))
        inverse = self.act(inverse, name+'/rsqrt')
        centered = self.act(x-mean, name+'/centered')
        z = self.act(centered*inverse, name+'/standardized')
        z = self.act(z*self.state[key+'.weight'], name+'/scaled')
        return self.act(z+self.state[key+'.bias'], name+'/output')

    def softmax(self, x, name):
        y = self.tf.nn.softmax(x, axis=1) if self.tf else torch.softmax(x, 1)
        return self.act(y, name, softmax=True)

    def pool(self, x, name):
        y = self.tf.reduce_sum(x, axis=1) if self.tf else x.sum(1)
        return self.act(y, name)

    def encoder(self, x, key, name):
        x = self.gelu(self.conv(x, key+'.stem.0', name+'/stem', key+'.stem.1'), name+'/stem_gelu')
        for i in range(3):
            p, n = key+f'.tcn.{i}', name+f'/block{i}'
            y = self.gelu(self.conv(x, p+'.c1', n+'/c1', p+'.b1', 2**i), n+'/gelu1')
            y = self.conv(y, p+'.c2', n+'/c2', p+'.b2', 2**i)
            x = self.gelu(self.act(x+y, n+'/add'), n+'/gelu2')
        score = self.gelu(self.conv(x, key+'.score.0', name+'/score0'), name+'/score_gelu')
        score = self.conv(score, key+'.score.2', name+'/score2')
        attention = self.softmax(score, name+'/softmax')
        return self.pool(self.act(x*attention, name+'/attention_product'), name+'/pool')

    def segment(self, x, name):
        values = [self.encoder(x[:, :, :3], 'acc', name+'/acc'), self.encoder(x[:, :, 3:], 'gyr', name+'/gyr')]
        h = self.tf.concat(values, -1) if self.tf else torch.cat(values, -1)
        h = self.act(h, name+'/concat')
        h = self.norm(h, 'fuse.0', name+'/norm')
        return self.gelu(self.dense(h, 'fuse.1', name+'/dense'), name+'/gelu')

    def __call__(self, x):
        x = self.act(x, 'input')
        values = [self.segment(z, f'segment{i}') for i, z in enumerate([x, x[:, :30], x[:, 30:60], x[:, 60:90]])]
        tokens = self.tf.stack(values, 1) if self.tf else torch.stack(values, 1)
        tokens = self.act(tokens, 'tokens')
        score = self.norm(tokens, 'token.0', 'token/norm')
        score = self.gelu(self.dense(score, 'token.1', 'token/dense1'), 'token/gelu')
        score = self.dense(score, 'token.3', 'token/dense3')
        attention = self.softmax(score, 'token/softmax')
        hidden = self.pool(self.act(tokens*attention, 'token/product'), 'token/pool')
        return self.dense(hidden, 'cls', 'logits')


def tf_export(state, ranges, path, representative, fused_norm=False):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tensors = {k: tf.constant(v.detach().cpu().numpy().astype(np.float32)) for k, v in state.items()}
    graph = Graph(tensors, ranges, tf=tf, fused_norm=fused_norm)
    class Module(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 90, 6], tf.float32)])
        def __call__(self, x):
            return graph(x)
    module = Module()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.__call__.get_concrete_function()], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([z[None]] for z in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type, converter.inference_output_type = tf.int8, tf.int8
    data = converter.convert()
    inv = e.inventory(e.interpreter(tf, data))
    if fused_norm:
        from fuse_layer_norm_v4 import fuse, inventory
        data = fuse(data)
        inv = inventory(data)
    path.write_bytes(data)
    return inv, graph


def micro_validate(model_path, inv, datasets, records, c, gates, runner, directory, window_floors=None, fixed_threshold=None):
    directory.mkdir(parents=True, exist_ok=True)
    details, reports = {}, {}
    output_scale, output_zero = inv['output_scale_zero']
    jobs = []
    for domain, (x, labels, keys, starts) in datasets.items():
        inputs = e.quantize(x, *inv['input_scale_zero'])
        for part in range(min(4, len(x))):
            lo, hi = len(x)*part//min(4,len(x)), len(x)*(part+1)//min(4,len(x))
            inp, out = directory/f'{domain}.{part}.inputs.bin', directory/f'{domain}.{part}.outputs.bin'
            inp.write_bytes(inputs[lo:hi].tobytes())
            jobs.append((domain, part, inp, out, hi-lo))
    def run(job):
        domain, part, inp, out, count = job
        result = subprocess.run([str(runner.resolve()), str(model_path.resolve()), str(inp.resolve()), str(out.resolve())], check=True, capture_output=True, text=True)
        if f'TFLM_WINDOWS_PASS={count}' not in result.stdout:
            raise ValueError('Micro did not evaluate every validation window')
        return domain, part, out
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run, jobs))
    for domain, (x, labels, keys, starts) in datasets.items():
        raw = np.concatenate([np.fromfile(out,np.int8).reshape(-1,2)
                              for d,part,out in sorted(results) if d==domain])
        if len(raw) != len(x):
            raise ValueError('Micro output count differs')
        logits = (raw.astype(np.float32)-output_zero)*output_scale
        probabilities = e.probabilities(logits)
        rows = [dict(key=k, file_label=int(y), prob=float(p), start=int(s)) for k,y,p,s in zip(keys,labels,probabilities,starts)]
        kk, yy, pp = e.base.aggregate(rows)
        details[domain] = (rows, kk, yy, pp)
        reports[domain] = e.metrics(labels, probabilities, c['threshold'])
        np.savez_compressed(directory/f'{domain}.predictions.npz', raw=raw, logits=logits, labels=labels, keys=keys, starts=starts)
    selection = v4.select_threshold(details, records, c['source_weights'], gates,
                                    window_recall_floors=window_floors, deployment_thresholds=True, fixed=fixed_threshold)
    fixed = v4.select_threshold(details, records, c['source_weights'], gates, fixed=c['threshold'])
    result = {'selection':selection, 'original_threshold_selection':fixed, 'original_threshold_window_metrics':reports,
              'windows':sum(len(z[0]) for z in datasets.values()), 'runtime':'native TFLite Micro / ESP-NN; not physical ESP32', 'test_used':False}
    v2.write_json(directory/'validation.json', result)
    return result


def make_datasets(records, c, cfg, cache, smoke=False):
    data = {}
    for d in c['source_weights']:
        rr = [r for r in records if r['domain'] == d]
        if smoke:
            rr = [next(r for r in rr if r['label']==label)
                  for label in sorted({r['label'] for r in rr})]
        arrays = v2.arrays(rr, cache)
        ds = v2.Clips(rr, arrays, c['normalization'], cfg, smoke=smoke)
        x = np.asarray([ds[i]['full'].numpy() for i in range(len(ds))], np.float32)
        data[d] = (x, np.asarray([rr[i]['label'] for i,s in ds.entries]),
                   np.asarray([rr[i]['key'] for i,s in ds.entries]), np.asarray([s for i,s in ds.entries]))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','work','own-root','public-root','m5-root','output','ptq-export','runner']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--phase', choices=['smoke','train'], required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--input-percentile', type=float, default=99.9)
    p.add_argument('--distillation-weight',type=float,default=.25)
    p.add_argument('--fused-norm', action='store_true')
    p.add_argument('--protocol-v5',action='store_true')
    args = p.parse_args()
    if not 0 < args.input_percentile <= 100 or args.epochs < 1:
        raise ValueError('Invalid bounded training protocol')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    if args.protocol_v5:
        from quantize_v5 import load
        c,cfg,parts,model=load(args)
    else:
        c, cfg, parts, model = e.load(args)
    cfg.batch = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.phase == 'train' and device.type != 'cuda':
        raise RuntimeError('Substantial QAT requires the available GPU')
    model = model.to(device).eval()
    teacher = copy.deepcopy(model).eval()
    for name, param in model.named_parameters():
        if name.startswith(('phase.', 'proj.')):
            param.requires_grad_(False)
    state = model.state_dict(keep_vars=True)
    calibration = np.load(args.ptq_export/'calibration.npy')
    manifest = json.loads((args.ptq_export/'calibration_manifest.json').read_text())
    if any(r['key'] not in {z['key'] for z in parts['train']} for r in manifest):
        raise ValueError('Calibration includes nontraining data')
    observer = Graph(state, observe=True, fused_norm=args.fused_norm)
    with torch.no_grad():
        for start in range(0, len(calibration), 64):
            observer(torch.from_numpy(calibration[start:start+64]).to(device))
    ranges = observer.ranges
    tail = (100-args.input_percentile)/2
    lo, hi = np.percentile(calibration, [tail, 100-tail])
    ranges['input'] = [float(lo), float(hi)]
    v2.write_json(args.output/'ranges.json', ranges)
    v2.write_json(args.output/'protocol.json', {'seed':42,'checkpoint_sha256':e.sha(args.checkpoint),
        'selection_protocol_sha256':e.sha(args.work/'selection_protocol.json') if args.protocol_v5 else None,
        'split_fingerprint':c['split_fingerprint'],'normalization':c['normalization'],
        'training_only_observers':True,'calibration_sha256':e.sha(args.ptq_export/'calibration.npy'),
        'fused_layer_normalization':args.fused_norm,'input_percentile':args.input_percentile,'epochs_limit':args.epochs,'patience':args.patience,
        'learning_rate':args.learning_rate,'batch':args.batch,'source_weights':c['source_weights'],
        'loss':f'source-weighted CE + {args.distillation_weight} * source-weighted KL to frozen original float teacher (T=2)',
        'BN':'frozen running statistics; train affine parameters and folded convolution weights',
        'fake_quant_limitations':'tensor/weight rounding and clipping; Micro integer kernels select each export',
        'selection':'constraints first, seven-source group-macro weighted error, own recall; exhaustive native Micro validation',
        'test_used_for_selection':False,'device':str(device)})
    graph = Graph(state, ranges, fused_norm=args.fused_norm)
    optimizer = torch.optim.AdamW([z for z in model.parameters() if z.requires_grad], lr=args.learning_rate, weight_decay=0.)
    if args.phase == 'smoke':
        x = torch.from_numpy(calibration[:8]).to(device)
        with torch.no_grad():
            original = teacher(x, x[:,:30], x[:,30:60], x[:,60:90])['logits']
            translated = Graph(state, quant=False)(x)
        if not torch.allclose(original, translated, atol=3e-4, rtol=3e-4):
            raise ValueError('Float graph parity failed')
        loss = graph(x).square().mean(); loss.backward()
        gradients = [z.grad for z in model.parameters() if z.requires_grad]
        if any(z is None or not torch.isfinite(z).all() for z in gradients):
            raise ValueError('Missing/nonfinite QAT gradient')
        optimizer.step()
        inv, tf_graph = tf_export(model.state_dict(), ranges, args.output/'smoke.tflite', calibration[:32], args.fused_norm)
        tf_logits = tf_graph(calibration[:1]).numpy()
        torch_logits = graph(torch.from_numpy(calibration[:1]).to(device)).detach().cpu().numpy()
        if not np.allclose(tf_logits, torch_logits, atol=.02, rtol=.02):
            raise ValueError(f'Torch/TF fake quant parity failed: {tf_logits} / {torch_logits}')
        datasets = make_datasets(parts['val'], c, cfg, args.output/'cache', smoke=True)
        gates = json.loads((args.work/'baseline_validation.json').read_text())['constraints']
        result = micro_validate(args.output/'smoke.tflite', inv, datasets, parts['val'], c, gates, args.runner, args.output/'smoke_micro')
        v2.write_json(args.output/'smoke.json', {'passed':True,'float_parity_max_error':float((original-translated).abs().max()),'gradient_parameters':len(gradients),'inventory':inv,'micro_windows':result['windows']})
        print('QAT_SMOKE_PASS', flush=True)
        return
    if not (args.output/'smoke.json').exists():
        raise RuntimeError('Run comprehensive smoke tests before training')
    if (args.output/'training_history.json').exists():
        raise FileExistsError('Training already started; never overwrite a run')
    # Smoke's update belongs to a separate invocation and is never a warm start.
    datasets = make_datasets(parts['val'], c, cfg, args.output/'cache')
    expected_windows=c['validation_windows'] if args.protocol_v5 else 61686
    if sum(len(z[0]) for z in datasets.values()) != expected_windows:
        raise ValueError('Exhaustive validation count changed')
    gates = json.loads((args.work/'baseline_validation.json').read_text())['constraints']
    reference=json.loads((args.ptq_export/'validation_report.json').read_text())
    window_floors={d:z['float']['recall'] for d,z in reference['domains'].items()
                   if d!='pooled' and z['float']['recall'] is not None}
    if args.protocol_v5:
        window_floors=json.loads((args.work/'selection_protocol.json').read_text())['window_recall_floors']
    train_arrays = v2.arrays(parts['train'], args.output/'cache')
    train_ds = v2.Clips(parts['train'], train_arrays, c['normalization'], cfg, random_crop=True)
    sampler = v4.CoverageSampler(parts['train'], args.batch, c['source_weights'], 42)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    history, best_rank, bad, started = [], None, 0, time.monotonic()
    for epoch in range(args.epochs+1):
        losses = []
        if epoch:
            for batch in loader:
                x = batch['full'].to(device); labels = batch['label'].to(device)
                logits = graph(x)
                with torch.no_grad():
                    target = teacher(x, x[:,:30], x[:,30:60], x[:,60:90])['logits']
                ce = v4.source_loss(F.cross_entropy(logits, labels, reduction='none'), batch['domain'], c['source_weights'])
                kd = F.kl_div(F.log_softmax(logits/2, -1), F.softmax(target/2, -1), reduction='none').sum(-1)*4
                loss = ce+args.distillation_weight*v4.source_loss(kd, batch['domain'], c['source_weights'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite QAT loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                losses.append(float(loss.detach()))
        directory = args.output/f'epoch_{epoch:02d}'; directory.mkdir()
        inv, _ = tf_export(model.state_dict(), ranges, directory/'model_int8.tflite', calibration, args.fused_norm)
        val = micro_validate(directory/'model_int8.tflite', inv, datasets, parts['val'], c, gates, args.runner, directory/'validation',window_floors)
        selection = val['selection']
        rank = (int(selection['valid_under_constraints']), -selection['weighted_error'], selection['domains']['own']['recall'])
        row = {'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,'selection':selection,
               'model_sha256':e.sha(directory/'model_int8.tflite'),'inventory':inv,
               'elapsed_seconds':time.monotonic()-started,'sampling':{k:v for k,v in sampler.report.items() if k!='group_crops'}}
        torch.save({'model':v2.cpu_state(model),'config':c['config'],'normalization':c['normalization'],
                    'split_fingerprint':c['split_fingerprint'],'source_weights':c['source_weights'],
                    'threshold':selection['threshold'],'original_checkpoint_sha256':e.sha(args.checkpoint),
                    'qat_ranges':ranges,'fused_layer_normalization':args.fused_norm,'smoke':False},directory/'qat_checkpoint.pt')
        if best_rank is None or rank > best_rank:
            best_rank, bad = rank, 0
            v2.write_json(args.output/'selected.json', row)
        elif epoch:
            bad += 1
        history.append(row); v2.write_json(args.output/'training_history.json', history)
        v2.write_json(args.output/'progress.json', {'epoch':epoch,'limit':args.epochs,'elapsed_seconds':time.monotonic()-started,'early_stop_bad_epochs':bad,'selected_epoch':json.loads((args.output/'selected.json').read_text())['epoch'],'status':'running'})
        print('QAT_EPOCH', json.dumps(row), flush=True)
        if epoch >= 4 and bad >= args.patience:
            break
    v2.write_json(args.output/'completed.json', {'epochs_completed':epoch,'selected':json.loads((args.output/'selected.json').read_text()),'elapsed_seconds':time.monotonic()-started,'test_evaluated':False,'physical_device_verified':False})
    v2.write_json(args.output/'progress.json', {'status':'complete','epochs_completed':epoch,
        'elapsed_seconds':time.monotonic()-started,'selected_epoch':json.loads((args.output/'selected.json').read_text())['epoch']})
    print('QAT_TRAIN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
