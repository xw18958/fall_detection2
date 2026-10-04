"""V5 weak-event trigger experiment, separated from the legacy V2 protocol.

Run --smoke first. Bag targets retain recording labels; positive bags encourage
one timely wake rather than marking preparation/recovery windows as falls.
Validation uses exact V5 Micro predictions. No device qualification is inferred
from host latency, and this script never installs firmware.
"""
import argparse,copy,json,re,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import train as base
from trigger_models import TinyTrigger,TinyMLP
from train_power_trigger import export,lite_metadata,quantize,probabilities,tensor_windows,torch_predict
from trigger_data_v5 import load,evaluation,digest,data
from power_events_v5 import select_threshold,simulate,quantized_threshold_margin

def write(path,obj):data.write_json(path,obj)
def sources():
    root=Path(__file__).parent
    names=['train_power_trigger_v5.py','trigger_data_v5.py','power_events_v5.py','trigger_models.py','train.py',
           'train_power_trigger.py','v5_reference/training_v2.py','v5_reference/preprocess_public_v2.py']
    return {n:digest(root/n) for n in names}

def model_candidates():
    return [(f'cnn{width}_{pool}',TinyTrigger(width,pool))
            for pool in ('mean','mean_max') for width in (8,16,24)]+[('mlp16',TinyMLP())]

def micro(runner,model,x,meta,folder,trigger=False):
    import subprocess
    folder.mkdir(parents=True,exist_ok=True)
    inp=folder/'input.bin';out=folder/'output.bin'
    quantize(x,meta['input_scale'],meta['input_zero']).tofile(inp)
    command=[str(runner),str(model),str(inp),str(out)]
    if trigger:command.append('--trigger')
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    raw=np.fromfile(out,np.int8).reshape(-1,2)
    if len(raw)!=len(x):raise ValueError('Native replay count differs')
    write(folder/'manifest.json',dict(stdout=result.stdout,runner_sha256=digest(runner),
                                    model_sha256=digest(model),device_latency_measured=False))
    return probabilities((raw.astype(np.float32)-meta['output_zero'])*meta['output_scale'])

def teacher(model,x,device):
    out=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),256):
            z=torch.from_numpy(x[start:start+256]).to(device)
            logits=model(z,z[:,:30],z[:,30:60],z[:,60:])['logits']
            out.append(torch.softmax(logits,1)[:,1].cpu().numpy())
    return np.concatenate(out)

def bags(rows,teacher_scores,full_threshold):
    grouped={}
    for i,r in enumerate(rows):grouped.setdefault((r['key'],r['segment']),[]).append(i)
    result=[]
    for (key,segment),indices in grouped.items():
        r=rows[indices[0]];positive=r['label']==1
        available=np.asarray(indices)
        if positive:
            # Teacher reference restricts auxiliary bag eligibility only.
            # It does not relabel preparation/recovery or alter the full TCN.
            scored=teacher_scores[available]>=full_threshold
            starts=np.flatnonzero(scored & ~np.r_[False,scored[:-1]])
            for start in starts:
                eligible=available[start:min(len(available),start+2)]
                eligible=eligible[teacher_scores[eligible]>=full_threshold]
                # Every reference run has its own timely positive bag. A
                # recording-level max could ignore its other reference runs.
                result.append(dict(key=key,segment=segment,indices=eligible,label=1,domain=r['domain'],weak_recording=False))
            if len(starts):continue
        result.append(dict(key=key,segment=segment,indices=available,label=r['label'],domain=r['domain'],weak_recording=positive))
    return result

def prepare_bag_index(records):
    index={}
    for r in records:index.setdefault(r['domain'],{}).setdefault(r['label'],{}).setdefault(r['key'],[]).append(r)
    return index

def draw_bags(index,x,rng,count=64,bag_size=8):
    domains=sorted(index);chosen=[]
    mass=np.asarray([.25 if d in ('own',data.M5) else .10 for d in domains]);mass/=mass.sum()
    for _ in range(count):
        domain=rng.choice(domains,p=mass);labels=sorted(index[domain]);label=int(rng.choice(labels))
        keys=sorted(index[domain][label]);key=keys[int(rng.integers(len(keys)))]
        rr=index[domain][label][key];r=rr[int(rng.integers(len(rr)))]
        chosen.append(r)
    xx=np.stack([x[rng.choice(r['indices'],size=bag_size,replace=len(r['indices'])<bag_size)] for r in chosen])
    return torch.from_numpy(xx),torch.tensor([r['label'] for r in chosen]),torch.tensor([r['weak_recording'] for r in chosen])

def weak_loss(logits,labels,weak_recording=None,robust=False):
    log_probability=torch.log_softmax(logits,-1)
    # Positive: at least one timely eligible view. Negative: every view normal.
    positive=-log_probability[...,1].max(1).values
    if robust:
        if weak_recording is None:weak_recording=torch.zeros_like(labels,dtype=torch.bool)
        positive=torch.where(weak_recording,positive,-log_probability[...,1].min(1).values)
    negative=-log_probability[...,0].min(1).values
    return torch.where(labels.bool(),positive*3,negative).mean()

def main():
    ap=argparse.ArgumentParser()
    for name in ['own-root','public-root','m5-root','checkpoint','split-from','full-model','firmware-config','runner','work']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--preflight',type=Path)
    ap.add_argument('--robust-events',action='store_true',help='Train every eligible timely reference view; teacher misses remain weak recording bags')
    ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--steps',type=int,default=80)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True);base.seed_all(42)
    if not torch.cuda.is_available():raise ValueError('Remote GPU required')
    device=torch.device('cuda')
    if not a.smoke:
        pre=json.loads(a.preflight.read_text()) if a.preflight else {}
        if not pre.get('passed') or pre.get('source_hashes')!=sources() or pre.get('robust_events',False)!=a.robust_events:raise ValueError('Matching successful smoke required')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    splits,arrays,ck,cfg,audit=load(a.own_root,a.public_root,a.m5_root,a.checkpoint,a.split_from,a.work/'cache')
    config=a.firmware_config.read_text()
    if digest(a.checkpoint)!=re.search(r'kCheckpointSha256\[\]="([a-f0-9]+)"',config)[1]:raise ValueError('Wrong V5 checkpoint')
    threshold=float(re.search(r'kThreshold=([\d.eE+-]+)',config)[1])
    for name,stat in [('kMean','mean'),('kSigma','std')]:
        values=[float(v.rstrip('f')) for v in re.search(name+r'\[6\]=\{([^}]+)',config)[1].split(',')]
        if not np.array_equal(np.asarray(values,np.float32),np.asarray(ck['normalization'][data.M5][stat],np.float32)):
            raise ValueError('V5 firmware normalization differs')
    contract=dict(audit,full_model_sha256=digest(a.full_model),firmware_config_sha256=digest(a.firmware_config),
                  source_hashes=sources(),seed=42,full_threshold=threshold,commercial_qualified=False)
    datasets={};rows={};xx={}
    for role in ('train','val'):
        datasets[role],rows[role]=evaluation(splits,arrays,ck['normalization'],cfg,role)
        if a.smoke:datasets[role].entries=datasets[role].entries[:8];rows[role]=rows[role][:8]
        xx[role]=tensor_windows(datasets[role])
    fullmeta=lite_metadata(tf,a.full_model)
    if a.smoke:
        if any(x.shape[1:]!=(90,6) or not np.isfinite(x).all() for x in xx.values()):raise ValueError('Invalid real inputs')
        rng=np.random.default_rng(42);cal=rng.normal(0,1,(32,90,6)).astype(np.float32)
        results={}
        for name,model in model_candidates():
            model=model.to(device);z=model(torch.from_numpy(cal[:16]).to(device)).reshape(2,8,2)
            loss=weak_loss(z,torch.tensor([0,1],device=device),robust=a.robust_events);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite weak-event gradients')
            folder=a.work/name;folder.mkdir(exist_ok=True);model=model.cpu().eval()
            meta=export(model,cal,tf,folder/'trigger.tflite')
            score=micro(a.runner,folder/'trigger.tflite',cal[:8],meta,folder/'micro',True)
            results[name]=dict(metadata=meta,finite_scores=bool(np.isfinite(score).all()),loss=float(loss.detach()))
        micro(a.runner,a.full_model,xx['val'],fullmeta,a.work/'full_replay')
        write(a.work/'preflight.json',dict(passed=True,robust_events=a.robust_events,source_hashes=sources(),contract=contract,models=results,
                                         real_input_windows={k:len(v) for k,v in xx.items()}))
        print('V5_SMOKE_PASS',flush=True);return
    write(a.work/'protocol.json',dict(contract,epochs=a.epochs,steps=a.steps,robust_events=a.robust_events,positive_supervision=('Every eligible timely teacher-reference view must wake; teacher misses retain weak recording max bags' if a.robust_events else 'One auxiliary positive bag per teacher reference run: first positive window plus next positive deadline; teacher misses retain weak recording bags'),
        negative_supervision='Worst sampled negative view penalized; original negative labels unchanged',
        sampling='Frozen V5 source masses: own 25%, M5 25%, public 10% each; balanced classes and recordings, then uniform reference run',
        split_selection='Validation only; test locked after selection',previous_test_seen=True,commercial_qualified=False))
    teacher_model=base.Net(cfg).to(device);teacher_model.load_state_dict(ck['model'])
    train_scores=teacher(teacher_model,xx['train'],device);del teacher_model;torch.cuda.empty_cache()
    records=prepare_bag_index(bags(rows['train'],train_scores,threshold))
    val_full=micro(a.runner,a.full_model,xx['val'],fullmeta,a.work/'validation_full')
    rng=np.random.default_rng(42);cal_indices=rng.choice(len(xx['train']),size=min(512,len(xx['train'])),replace=False)
    cal=xx['train'][cal_indices];candidates={}
    for name,model in model_candidates():
        base.seed_all(42);rng=np.random.default_rng(42);model=model.to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);best=None;history=[]
        for epoch in range(1,a.epochs+1):
            start=time.monotonic();model.train();losses=[]
            for _ in range(a.steps):
                x,y,weak=draw_bags(records,xx['train'],rng);x=x.to(device);y=y.to(device);weak=weak.to(device)
                opt.zero_grad();loss=weak_loss(model(x.reshape(-1,90,6)).reshape(len(y),8,2),y,weak,robust=a.robust_events)
                loss.backward();opt.step();losses.append(float(loss.detach()))
            scores=torch_predict(model,xx['val'],device);selection=select_threshold(rows['val'],scores,val_full,threshold)
            report=selection.get('report',{});fraction=report.get('negative_active_fraction',1.)
            rank=(selection.get('qualified',False),-report.get('lost_or_late_reference_events',1000000),-fraction)
            if best is None or rank>best[0]:best=(rank,copy.deepcopy(model.state_dict()),epoch)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-start,selection=selection))
            write(a.work/(name+'_history.json'),history)
            print(name,epoch,'negative_active_fraction',fraction,'qualified',selection.get('qualified'),flush=True)
        model.load_state_dict(best[1]);model=model.cpu().eval();folder=a.work/name;folder.mkdir(exist_ok=True)
        torch.save(dict(model=model.state_dict(),selected_epoch=best[2],contract=contract),folder/'trigger.pt')
        meta=export(model,cal,tf,folder/'trigger_int8.tflite')
        scores=micro(a.runner,folder/'trigger_int8.tflite',xx['val'],meta,folder/'validation_micro',True)
        selection=select_threshold(rows['val'],scores,val_full,threshold)
        if a.robust_events and selection['threshold'] is not None:
            selection['unadjusted_threshold']=selection['threshold']
            selection['threshold']=quantized_threshold_margin(selection['threshold'],meta['output_scale'],1)
            selection['report']=simulate(rows['val'],scores,val_full,selection['threshold'],threshold)[0]
            selection['qualified']=selection['report']['lost_or_late_reference_events']==0 and selection['report']['negative_active_fraction']<1/3
        candidates[name]=dict(metadata=meta,selection=selection,parameters=sum(p.numel() for p in model.parameters()),epoch=best[2])
        write(folder/'validation.json',candidates[name])
    valid=[n for n,c in candidates.items() if c['selection']['qualified']]
    winner=min(valid,key=lambda n:(candidates[n]['selection']['report']['negative_active_fraction'],candidates[n]['metadata']['model_bytes'])) if valid else None
    write(a.work/'selection_lock.json',dict(contract,winner=winner,candidates=candidates,test_used_for_selection=False))
    test,testr=evaluation(splits,arrays,ck['normalization'],cfg,'test');xtest=tensor_windows(test)
    fulltest=micro(a.runner,a.full_model,xtest,fullmeta,a.work/'test_full');tests={}
    for name,c in candidates.items():
        threshold_t=c['selection']['threshold']
        if threshold_t is None:continue
        scores=micro(a.runner,a.work/name/'trigger_int8.tflite',xtest,c['metadata'],a.work/name/'test_micro',True)
        tests[name]=simulate(testr,scores,fulltest,threshold_t,threshold)[0]
    write(a.work/'results.json',dict(contract,winner=winner,validation=candidates,test=tests,
                                    hardware_deadline_qualified=False,deployment_eligible=False,
                                    retrieval=None,zero_shot=None,current_measurement=None))
    print('V5_TRIGGER_EXPERIMENT_COMPLETE',winner,flush=True)

if __name__=='__main__':main()
