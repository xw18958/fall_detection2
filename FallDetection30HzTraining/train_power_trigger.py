"""Seed-42 trigger comparison with frozen full-model/data contracts.

Requires a separate --smoke pass before real training. Test is evaluated only
after native INT8 validation locks the candidate and threshold. RF is a measured
host baseline, never substituted for a trained INT8 CNN in the firmware.
"""
import argparse,copy,hashlib,json,os,re,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import train as base
import training_v2 as data
from trigger_data import load,evaluation,digest
from trigger_models import TinyTrigger,TinyMLP,keras_equivalent
from power_cascade import select_threshold,simulate

def write(path,value):data.write_json(path,value)
def probabilities(logits):
    z=np.asarray(logits,np.float32);return 1/(1+np.exp(np.clip(z[:,0]-z[:,1],-80,80)))
def quantize(x,scale,zero):
    z=np.asarray(x,np.float32)/np.float32(scale)
    return np.clip(np.copysign(np.floor(np.abs(z).astype(np.float64)+.5),z)+zero,-128,127).astype(np.int8)

def lite_metadata(tf,path):
    it=tf.lite.Interpreter(model_path=str(path));it.allocate_tensors()
    i,o=it.get_input_details()[0],it.get_output_details()[0]
    if list(i['shape'])!=[1,90,6] or list(o['shape'])!=[1,2] or i['dtype']!=np.int8 or o['dtype']!=np.int8:raise ValueError('Expected exact INT8 90x6 -> 2 model')
    return dict(input_scale=float(i['quantization'][0]),input_zero=int(i['quantization'][1]),output_scale=float(o['quantization'][0]),output_zero=int(o['quantization'][1]),
      tensor_types=sorted({str(d['dtype']) for d in it.get_tensor_details()}),operators=sorted({d['op_name'] for d in it._get_ops_details()}))

def micro_predict(runner,model,x,metadata,out,jobs=8):
    out.mkdir(parents=True,exist_ok=True);q=quantize(x,metadata['input_scale'],metadata['input_zero'])
    def worker(part):
        idx=np.array_split(np.arange(len(q)),min(jobs,len(q)))[part];folder=out/str(part);folder.mkdir(exist_ok=True)
        inp=folder/'inputs.bin';raw=folder/'outputs.bin';q[idx].tofile(inp)
        started=time.monotonic();r=subprocess.run([str(runner),str(model),str(inp),str(raw)],check=True,capture_output=True,text=True)
        y=np.fromfile(raw,np.int8).reshape(-1,2)
        if len(y)!=len(idx):raise ValueError('Micro replay count mismatch')
        return y,r.stdout,time.monotonic()-started
    with ThreadPoolExecutor(max_workers=jobs) as pool:results=list(pool.map(worker,range(min(jobs,len(q)))))
    raw=np.concatenate([r[0] for r in results]);logits=(raw.astype(np.float32)-metadata['output_zero'])*metadata['output_scale']
    np.savez_compressed(out/'predictions.npz',raw=raw,probabilities=probabilities(logits))
    write(out/'micro_manifest.json',{'runner_sha256':digest(runner),'windows':len(q),'model_sha256':digest(model),'parts':[{'stdout':r[1],'host_wall_seconds':r[2]} for r in results],
      'device_latency_measured':False,'warning':'Parallel host replay throughput is not M5 latency or power'})
    return probabilities(logits),raw

def tensor_windows(ds):
    xs=[]
    for batch in DataLoader(ds,batch_size=256,num_workers=0):xs.append(batch['full'].numpy())
    return np.concatenate(xs)

def torch_predict(model,x,device):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),512):out.append(model(torch.from_numpy(x[start:start+512]).to(device)).cpu().numpy())
    return probabilities(np.concatenate(out))

def export(model,cal,tf,path):
    cpu=copy.deepcopy(model).cpu().eval();keras=keras_equivalent(cpu,tf)
    with torch.no_grad():expected=cpu(torch.from_numpy(cal[:32])).numpy()
    actual=keras(cal[:32],training=False).numpy();error=float(np.abs(expected-actual).max())
    if error>1e-5:raise ValueError('Torch/TF graph mismatch: '+str(error))
    converter=tf.lite.TFLiteConverter.from_keras_model(keras);converter.optimizations=[tf.lite.Optimize.DEFAULT]
    def representative():
        for x in cal:yield [x[None].astype(np.float32)]
    converter.representative_dataset=representative;converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    path.write_bytes(converter.convert());meta=lite_metadata(tf,path)
    if any('float' in dtype for dtype in meta['tensor_types']):raise ValueError('Float tensors in trigger')
    return dict(meta,float_graph_max_error=error,model_bytes=path.stat().st_size,sha256=digest(path))

def sampler(records,count):
    domains=sorted({r['domain'] for r in records});mass={d:1/len(domains) for d in domains}
    if data.M5 in domains:mass={d:(.25 if d in ('own',data.M5) else .10) for d in domains}
    weights=[]
    for r in records:
        rr=[v for v in records if v['domain']==r['domain']];classes={v['label'] for v in rr}
        groups={v['group'] for v in rr if v['label']==r['label']}
        n=sum(v['label']==r['label'] and v['group']==r['group'] for v in rr)
        weights.append(mass[r['domain']]/len(classes)/len(groups)/n)
    return WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),count,replacement=True,generator=torch.Generator().manual_seed(42))

def source_hashes():
    root=Path(__file__).parent
    return {name:digest(root/name) for name in ['train_power_trigger.py','trigger_data.py','trigger_models.py','power_cascade.py','train.py','training_v2.py']}

def make_bundle(out,name,selection,metadata,model,x,raw,contract):
    indices=np.linspace(0,len(x)-1,min(4,len(x))).round().astype(int)
    q=quantize(x[indices],metadata['input_scale'],metadata['input_zero'])
    values=model.read_bytes();fmt=lambda a:','.join(map(str,np.asarray(a).ravel().tolist()))
    h='#pragma once\n#include <cstdint>\nnamespace trigger_bundle {\n'
    h+=f'constexpr int kSampleHz=30,kSamples=90,kReplayCount={len(indices)};\n'
    h+=f'constexpr bool kQualified={str(selection["qualified"]).lower()};\nconstexpr float kThreshold={selection["threshold"]:.12e}f;\n'
    h+=f'constexpr char kFullModelSha256[]="{contract["full_model_sha256"]}";\n'
    h+='alignas(16) constexpr unsigned char kModel[]={'+fmt(np.frombuffer(values,np.uint8))+'};\n'
    for label,shape,arr in [('kReplayInputs',540,q.reshape(-1,540)),('kReplayOutputs',2,raw[indices])]:
        h+=f'constexpr int8_t {label}[][{shape}]={{\n'+',\n'.join('{'+fmt(row)+'}' for row in arr)+'\n};\n'
    h+='}\n';(out/'trigger_bundle.h').write_text(h)
    write(out/'trigger_manifest.json',dict(contract,candidate=name,model_sha256=digest(model),bundle_sha256=digest(out/'trigger_bundle.h'),
      qualified=selection['qualified'],threshold=selection['threshold'],input_shape=[1,90,6],sample_hz=30,validation=selection,inventory=metadata))

def main():
    ap=argparse.ArgumentParser()
    for n in ['own_root','public_root','checkpoint','split_from','full_model','firmware_config','runner','work']:ap.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    ap.add_argument('--m5-combined-root',type=Path)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--preflight',type=Path)
    ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--draws',type=int,default=12800)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True);base.seed_all(42)
    if not torch.cuda.is_available():raise RuntimeError('Use remote GPU for training')
    device=torch.device('cuda');torch.backends.cudnn.benchmark=False
    if not a.smoke:
        pre=json.loads(a.preflight.read_text()) if a.preflight else {}
        if not pre.get('passed') or pre.get('source_hashes')!=source_hashes():raise ValueError('Successful matching smoke preflight required')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    splits,arrays,ck,cfg,audit=load(a.own_root,a.public_root,a.checkpoint,a.split_from,a.work/'cache',a.m5_combined_root)
    config=a.firmware_config.read_text();full_threshold=float(re.search(r'kThreshold\s*=\s*([\d.eE+-]+)',config)[1])
    if ck.get('split_fingerprint')!=audit['original_split_fingerprint']:raise ValueError('Checkpoint and inherited split differ')
    if ck['normalization'].keys()!=set(['own',*data.PUBLIC]):raise ValueError('Unexpected frozen normalization domains')
    for name,stat in [('kMean','mean'),('kSigma','std')]:
        numbers=np.asarray([float(v.strip().rstrip('f')) for v in re.search(name+r'\[6\]\s*=\s*\{([^}]+)',config)[1].split(',')],np.float32)
        if not np.array_equal(numbers,np.asarray(ck['normalization']['own'][stat],np.float32)):raise ValueError('Firmware normalization differs from checkpoint')
    checkpoint_sha=re.search(r'kCheckpointSha256\[\]\s*=\s*"([a-f0-9]+)"',config)[1]
    if checkpoint_sha!=digest(a.checkpoint):raise ValueError('Firmware checkpoint identity mismatch')
    contract=dict(audit,full_model_sha256=digest(a.full_model),firmware_config_sha256=digest(a.firmware_config),full_threshold=full_threshold,source_hashes=source_hashes())
    write(a.work/'protocol.json',dict(contract,seed=42,candidates=['depthwise_cnn8','mlp16','rf16_depth4'],epochs=a.epochs,train_draws=a.draws,
      full_model_unchanged=True,partition_selection='validation only; test once after locks',trigger_priority='per-source window recall >=99.5%, zero loss of baseline true-positive windows, then minimize false wakes',
      qualification='Must also reduce full invocations below 1/3 of the 4 Hz stream; firmware separately checks 250 ms compute budget',
      hardware_current_measured=False))
    train=data.Clips(splits['train'],arrays,ck['normalization'],cfg,random_crop=True)
    val,rows=evaluation(splits,arrays,ck['normalization'],cfg,'val')
    if a.smoke:
        val.entries=val.entries[:8];rows=rows[:8]
        # Keep data-loader checks distinct from synthetic conversion/gradient tests.
        x=tensor_windows(val);cal=np.random.default_rng(42).normal(0,1,(32,90,6)).astype(np.float32)
        for model in (TinyTrigger(),TinyMLP()):
            model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
            for _ in range(2):
                opt.zero_grad();loss=F.cross_entropy(model(torch.from_numpy(cal[:8]).to(device)),torch.arange(8,device=device)%2);loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
                opt.step()
            file=a.work/(type(model).__name__+'.tflite');meta=export(model,cal,tf,file);micro_predict(a.runner,file,cal,meta,a.work/(type(model).__name__+'_micro'),1)
        # Full model shape and kernels checked with real preserved-normalization inputs.
        meta=lite_metadata(tf,a.full_model);micro_predict(a.runner,a.full_model,x,meta,a.work/'full_micro',1)
        report={'passed':True,'source_hashes':source_hashes(),'contract':contract,'checks':['split/group disjointness','frozen normalization and checkpoint identity','segmented real 90x6 windows','GPU finite gradients CNN/MLP','Torch/TF parity','INT8-only conversion','native Micro CNN/MLP/full model']}
        write(a.work/'preflight.json',report);print('POWER_TRIGGER_SMOKE_PASS',flush=True);return
    xval=tensor_windows(val);full_meta=lite_metadata(tf,a.full_model)
    full_val,_=micro_predict(a.runner,a.full_model,xval,full_meta,a.work/'full_validation')
    train_loader=DataLoader(train,batch_size=256,sampler=sampler(train.rs,a.draws),num_workers=0)
    # Training-only calibration + RF inputs use actual training labels, never val/test.
    rf_x=[];rf_y=[];cal_source=[]
    for batch in train_loader:
        rf_x.append(batch['full'].numpy());rf_y.append(batch['label'].numpy());cal_source.extend(batch['domain'])
    xtrain=np.concatenate(rf_x);ytrain=np.concatenate(rf_y)
    indices=np.random.default_rng(42).choice(len(xtrain),min(2048,len(xtrain)),replace=False);cal=xtrain[indices]
    write(a.work/'calibration_manifest.json',{'partition':'train','windows':len(cal),'source_counts':{d:int(sum(cal_source[i]==d for i in indices)) for d in sorted(set(cal_source))},'class_counts':{str(y):int((ytrain[indices]==y).sum()) for y in (0,1)},'data_sha256':hashlib.sha256(cal.tobytes()).hexdigest()})
    candidates={};models={}
    for name,model in [('depthwise_cnn8',TinyTrigger()),('mlp16',TinyMLP())]:
        base.seed_all(42);model=model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        best=None;history=[];stale=0
        for epoch in range(1,a.epochs+1):
            model.train();losses=[];start=time.monotonic()
            for batch in train_loader:
                opt.zero_grad();logits=model(batch['full'].to(device));loss=F.cross_entropy(logits,batch['label'].to(device),weight=torch.tensor([1.,3.],device=device));loss.backward();opt.step();losses.append(float(loss.detach()))
            pv=torch_predict(model,xval,device);chosen=select_threshold(rows,pv,full_val,full_threshold)
            rank=(chosen['qualified'],-chosen['report']['full_invocation_fraction'],-chosen['report']['false_wakeups'])
            history.append({'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.monotonic()-start,'validation':chosen})
            if best is None or rank>best[0]:best=(rank,copy.deepcopy(model.state_dict()),epoch);stale=0
            else:stale+=1
            write(a.work/(name+'_history.json'),history);print(name,epoch,'active_fraction',chosen['report']['full_invocation_fraction'],'qualified',chosen['qualified'],flush=True)
            if stale>=5:break
        model.load_state_dict(best[1]);models[name]=model.cpu();folder=a.work/name;folder.mkdir(exist_ok=True)
        torch.save({'model':model.state_dict(),'selected_epoch':best[2],'contract':contract},folder/'trigger.pt')
        file=folder/'trigger_int8.tflite';meta=export(model,cal,tf,file);pv,raw=micro_predict(a.runner,file,xval,meta,folder/'validation_micro')
        selection=select_threshold(rows,pv,full_val,full_threshold)
        candidates[name]={'selection':selection,'metadata':meta,'parameters':sum(p.numel() for p in model.parameters()),'epoch':best[2]}
        make_bundle(folder,name,selection,meta,file,xval,raw,contract);write(folder/'validation.json',candidates[name])
    rf=RandomForestClassifier(n_estimators=16,max_depth=4,class_weight={0:1,1:3},random_state=42,n_jobs=8).fit(xtrain.reshape(len(xtrain),-1),ytrain)
    pv=rf.predict_proba(xval.reshape(len(xval),-1))[:,1];selection=select_threshold(rows,pv,full_val,full_threshold)
    candidates['rf16_depth4']={'selection':selection,'nodes':sum(t.tree_.node_count for t in rf.estimators_),'host_only':True}
    qualified=[name for name,v in candidates.items() if v['selection']['qualified'] and not v.get('host_only')]
    winner=min(qualified,key=lambda n:(candidates[n]['selection']['report']['false_wakeups'],candidates[n]['selection']['report']['full_invocation_fraction'],candidates[n]['metadata']['model_bytes'])) if qualified else None
    # Lock candidate, threshold and provenance before any production test inputs.
    write(a.work/'selection_lock.json',dict(contract,candidates=candidates,winner=winner,status='qualified' if winner else 'no_qualified_trigger',test_used_for_selection=False))
    test,testr=evaluation(splits,arrays,ck['normalization'],cfg,'test');xtest=tensor_windows(test)
    full_test,_=micro_predict(a.runner,a.full_model,xtest,full_meta,a.work/'full_test')
    tests={}
    for name,v in candidates.items():
        if v.get('host_only'):pt=rf.predict_proba(xtest.reshape(len(xtest),-1))[:,1]
        else:pt,_=micro_predict(a.runner,a.work/name/'trigger_int8.tflite',xtest,v['metadata'],a.work/name/'test_micro')
        report,_=simulate(testr,pt,full_test,v['selection']['threshold'],full_threshold);tests[name]=report
    # Tests are descriptive for locked candidates. Failure never changes a threshold
    # or switches to a different model based on test rankings.
    deployment_ok=bool(winner and tests[winner]['lost_baseline_fall_recordings']==0 and all(
        v['trigger']['recall'] is None or v['trigger']['recall']>=.995 for v in tests[winner]['per_source'].values()))
    result={'winner':winner,'deployment_eligible':deployment_ok,'validation':candidates,'test':tests,'contract':contract,
      'zero_shot':None,'retrieval':None,'hardware_latency':None,'actual_current':None,'scope':'Trained trigger / fixed full TCN; no physical device installation'}
    for name,v in candidates.items():
        if v.get('host_only'):continue
        folder=a.work/name;header=folder/'trigger_bundle.h'
        eligible=bool(deployment_ok and name==winner)
        text=header.read_text();text=re.sub(r'kQualified=(true|false)', 'kQualified='+str(eligible).lower(),text);header.write_text(text)
        write(folder/'trigger_bundle.json',dict(full_model_sha256=contract['full_model_sha256'],
          firmware_config_sha256=contract['firmware_config_sha256'],header_sha256=digest(header),
          sample_hz=30,samples=90,deployment_eligible=eligible,model_sha256=v['metadata']['sha256'],
          selection_lock_sha256=digest(a.work/'selection_lock.json'),test_result=tests[name]))
    write(a.work/'results.json',result);print('POWER_TRIGGER_COMPLETE',winner,'deployment_eligible',deployment_ok,flush=True)

if __name__=='__main__':main()
