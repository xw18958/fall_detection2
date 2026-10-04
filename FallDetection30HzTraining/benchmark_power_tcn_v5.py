"""Controlled exploratory TCN ablations; never modifies the deployed model.

Same seed, frozen normalization, original partitions, matched five-epoch fine-tune
budget for control/width/blocks/mean attention. Validation thresholds are locked
before test. Previously inspected test sources make this exploratory, not a new
independent commercial qualification. CPU host latency is not ESP32 latency.
"""
import argparse,copy,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import train as base
from types import SimpleNamespace
from trigger_data_v5 import data,load,evaluation,digest
from train_power_trigger import sampler,tensor_windows,probabilities,write,micro_predict,lite_metadata
from power_cascade import binary_metrics

class MeanNet(base.Net):
    def __init__(self,cfg):
        super().__init__(cfg)
        for enc in (self.acc,self.gyr):
            del enc.score
            enc.forward=lambda x,e=enc:e.tcn(e.stem(x.transpose(1,2))).mean(-1)
        del self.token
    def forward(self,full,pre,impact,post):
        h=torch.stack([self.seg(x) for x in (full,pre,impact,post)],1).mean(1)
        return {'logits':self.cls(h)}

def candidate(ck,width=None,blocks=None,mean=False):
    cfg=SimpleNamespace(**ck['config']);cfg.channels=width or cfg.channels;cfg.blocks=blocks or cfg.blocks
    model=(MeanNet if mean else base.Net)(cfg);original=ck['model'];state=model.state_dict();old=ck['config']['channels']
    # Concatenated acc/gyro fuse axes need separate channel slices.
    fuse=torch.cat([torch.arange(cfg.channels),old+torch.arange(cfg.channels)])
    for key,target in state.items():
        source=original[key]
        if key in ('fuse.0.weight','fuse.0.bias'):source=source[fuse]
        elif key=='fuse.1.weight':source=source[:cfg.channels,fuse]
        else:
            source=source[tuple(slice(0,n) for n in target.shape)] if target.ndim else source
        if source.shape!=target.shape:raise ValueError(key+' shape mismatch')
        state[key]=source.clone()
    model.load_state_dict(state);return model,cfg

def predict(model,x,device):
    out=[];model.eval()
    with torch.inference_mode():
        for lo in range(0,len(x),256):
            z=torch.from_numpy(x[lo:lo+256]).to(device);out.append(model(z,z[:,:30],z[:,30:60],z[:,60:90])['logits'].cpu().numpy())
    return probabilities(np.concatenate(out))

def report(rows,p,threshold):
    result={'overall':binary_metrics([r['label'] for r in rows],p,threshold),'per_source':{}}
    for domain in sorted({r['domain'] for r in rows}):
        idx=np.array([r['domain']==domain for r in rows]);result['per_source'][domain]=binary_metrics(np.array([r['label'] for r in rows])[idx],p[idx],threshold)
    return result

def guard(rows,p,reference,threshold):
    # Fixed original decision threshold isolates architecture, no threshold tuning.
    r=report(rows,p,threshold);b=report(rows,reference,threshold)
    okay=all(all(m[k] is None or (r['per_source'][d][k] is not None and r['per_source'][d][k]>=m[k]-1e-12) for k in ('recall','specificity','F1')) for d,m in b['per_source'].items())
    return okay,r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    root=a.root;contract=json.loads((root/'train_events/protocol.json').read_text());out=root/('tcn_v5_smoke' if a.smoke else 'tcn_v5_benchmark');out.mkdir(exist_ok=True)
    trainroot=Path('/raid1/xwan0900/v5_m5_seed42_20261003/seed42')
    splits,arrays,ck,cfg,audit=load('/raid1/xwan0900/fall_detection2/fd_datasets/30Hz_processed_clean_v2','/raid1/xwan0900/fall_detection2/fd_datasets/processed_v2','/raid1/xwan0900/fall_detection2/fd_datasets/M5_combined_v5_20261003',trainroot/'checkpoints/locked_model.pt',trainroot/'splits/group_splits.json',root/'train_events/cache')
    fingerprint={n:digest(Path(__file__).parent/n) for n in ('benchmark_power_tcn_v5.py','train.py','trigger_data_v5.py','train_power_trigger.py','v5_reference/training_v2.py')}
    if not a.smoke:
        pre=json.loads((root/'tcn_v5_smoke/preflight.json').read_text())
        if not pre.get('passed') or pre['script_sha256']!=fingerprint:raise ValueError('Matching smoke required')
    if not torch.cuda.is_available():raise ValueError('GPU required')
    device='cuda';base.seed_all(42)
    val,rows=evaluation(splits,arrays,ck['normalization'],cfg,'val')
    if a.smoke:val.entries=val.entries[:8];rows=rows[:8]
    xv=tensor_windows(val);original,_=candidate(ck);ref=predict(original.to(device),xv,device);threshold=contract['full_threshold']
    architectures=[('matched_control',{}),('width16',{'width':16}),('blocks2',{'blocks':2}),('mean_attention',{'mean':True})]
    loader=DataLoader(data.Clips(splits['train'],arrays,ck['normalization'],cfg,random_crop=True),batch_size=128,sampler=sampler(splits['train'],6400),num_workers=0)
    results={};models={}
    for name,kwargs in architectures:
        base.seed_all(42);model,mcfg=candidate(ck,**kwargs);model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4)
        epochs=1 if a.smoke else 5;history=[];best=None
        for epoch in range(epochs):
            model.train()
            for module in model.modules():
                if isinstance(module,nn.BatchNorm1d):module.eval()
            losses=[]
            for step,batch in enumerate(loader):
                x=batch['full'].to(device);optimizer.zero_grad();logits=model(x,x[:,:30],x[:,30:60],x[:,60:90])['logits']
                loss=nn.functional.cross_entropy(logits,batch['label'].to(device));loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite training')
                optimizer.step();losses.append(float(loss.detach()))
                if a.smoke and step==1:break
            pv=predict(model,xv,device);qualified,metrics=guard(rows,pv,ref,threshold)
            rank=(qualified,metrics['overall']['F1']);history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics,'qualified':qualified})
            if best is None or rank>best[0]:best=(rank,copy.deepcopy(model.state_dict()),epoch+1)
            write(out/(name+'_history.json'),history);print(name,epoch+1,'qualified',qualified,flush=True)
        model.load_state_dict(best[1]);model.eval();pv=predict(model,xv,device);qualified,metrics=guard(rows,pv,ref,threshold)
        # CPU time and MACs through classifier path only; excludes SSL-only heads.
        model=model.cpu();torch.set_num_threads(1);z=torch.from_numpy(xv[:1]);mac=[0];handles=[]
        def count(m,args,value):
            if isinstance(m,nn.Conv1d):mac[0]+=value.numel()*m.in_channels*m.kernel_size[0]//m.groups
            elif isinstance(m,nn.Linear):mac[0]+=value.numel()*m.in_features
        for module in model.modules():
            if isinstance(module,(nn.Conv1d,nn.Linear)):handles.append(module.register_forward_hook(count))
        with torch.inference_mode():model(z,z[:,:30],z[:,30:60],z[:,60:90])
        for handle in handles:handle.remove()
        durations=[]
        with torch.inference_mode():
            for _ in range(22):
                start=time.perf_counter_ns();model(z,z[:,:30],z[:,30:60],z[:,60:90]);durations.append((time.perf_counter_ns()-start)/1000)
        results[name]={'parameters':sum(p.numel() for p in model.parameters()),'classifier_macs':mac[0],'host_cpu_median_us':float(np.median(durations[2:])),
          'validation':metrics,'qualified_float':qualified,'epoch':best[2],'decision_threshold':threshold,'int8_qualified':False,'hardware_latency':None}
        torch.save({'model':model.state_dict(),'config':vars(mcfg),'normalization':ck['normalization'],'mean_attention':kwargs.get('mean',False)},out/(name+'.pt'));models[name]=model
    if a.smoke:
        write(out/'preflight.json',{'passed':True,'script_sha256':fingerprint,'checks':['all warm-start shapes and fuse channel mapping','finite GPU gradients each architecture','classification/resource forward paths']});return
    write(out/'selection_lock.json',{'architectures':results,'threshold':threshold,'test_used_for_selection':False})
    test,tr=evaluation(splits,arrays,ck['normalization'],cfg,'test');xt=tensor_windows(test);reference=predict(original.to(device),xt,device)
    for name,model in models.items():
        p=predict(model.to(device),xt,device);results[name]['test']=report(tr,p,threshold);results[name]['test_guard_pass']=guard(tr,p,reference,threshold)[0]
    write(out/'results.json',{'candidates':results,'original_test':report(tr,reference,threshold),'contract':contract,'previous_test_seen':True,'commercial_qualified':False,'budget':'5 epochs x 6400 draws, original BN statistics frozen, all candidates and matched control same sampler/seed42',
      'selection':'Fixed original threshold; per-source recall, specificity and F1 must not decline; validation before test',
      'deployment_changed':False,'scope':'Exploratory structural benchmark; original INT8 deployment retained; no candidate deployable without INT8 native and M5 timing qualification'})
    print('POWER_TCN_BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
