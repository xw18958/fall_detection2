from __future__ import annotations
import argparse, copy, json, math, random
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import train_exp2_converged as b

PUBLIC_FALL=['CGU_BES','Cogent','SFU_IMU','UCI_SimulatedFalls']
PUBLIC_SSL=[*PUBLIC_FALL,'PAMAP2']

def cfg_v2():
    c=b.Cfg(seed=42,batch=256); c.fs=20.; c.ssl_lr=3e-4; c.head_lr=3e-4; c.all_lr=8e-5
    c.head_epochs=10; c.all_epochs=50; c.patience=10
    return c

def nw(n,c): return len(b.starts(n,int(round(c.win_sec*c.fs)),c.fs,c.stride_sec))

def own_split(rs,arr,c,path):
    if path.exists(): s=json.loads(path.read_text())
    else:
        rng=np.random.default_rng(c.seed); ratios=np.array([.70,.15,.15]); best=None
        total=sum(nw(len(arr[r['rel']]),c) for r in rs); pos=sum(nw(len(arr[r['rel']]),c) for r in rs if r['label'])
        for _ in range(20000):
            out=[[],[],[]]; w=np.zeros(3)
            for ii in rng.permutation(len(rs)):
                r=rs[int(ii)]; j=int(np.argmax(ratios-w/max(total,1))); out[j].append(r); w[j]+=nw(len(arr[r['rel']]),c)
            if any(not x for x in out): continue
            score=sum(abs(w[i]/total-ratios[i]) for i in range(3))
            if pos:
                pw=[sum(nw(len(arr[r['rel']]),c) for r in out[i] if r['label']) for i in range(3)]
                score+=.25*sum(abs(pw[i]/pos-ratios[i]) for i in range(3))
            if best is None or score<best[0]: best=(score,out)
        names=['train','val','test']; s={'seed':c.seed,'atomic_unit':'recording','target_window_ratios':ratios.tolist(),
            'splits':{names[i]:sorted(r['rel'] for r in best[1][i]) for i in range(3)}}
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(s,indent=2))
    sets={k:set(v) for k,v in s['splits'].items()}; assert not sets['train']&sets['val']; assert not sets['train']&sets['test']; assert not sets['val']&sets['test']
    m={r['rel']:r for r in rs}; sp={k:[m[x] for x in s['splits'][k]] for k in ('train','val','test')}
    return s,sp

def pre_pub(x,y,c,prefix):
    pre=[]; imp=[]; post=[]
    for w in x:
        a,bb,cc=b.local_ctx(w,c.fs,c.ctx_sec); pre.append(a); imp.append(bb); post.append(cc)
    n=len(x); return {'full':x.astype(np.float32),'pre':np.stack(pre).astype(np.float32),'impact':np.stack(imp).astype(np.float32),'post':np.stack(post).astype(np.float32),
        'label':np.asarray(y,np.int64),'phase':np.full(n,-1,np.int64),'file_label':np.asarray(y,np.int64),'key':[f'{prefix}/{i}' for i in range(n)],'start':np.zeros(n,np.int64)}

def public_data(root,c):
    rng=np.random.default_rng(c.seed); tr={}; va={}; stats={}
    for name in PUBLIC_SSL:
        p=next((q for q in [root/f'{name}_v2.npz',root/f'{name}.npz'] if q.exists()),None)
        if p is None: raise RuntimeError(f'missing {name} public cache')
        z=np.load(p,allow_pickle=True); x=z['x'].astype(np.float32) if 'x' in z else z['ssl_x'].astype(np.float32)
        if x.shape[1:]!=(60,6): raise RuntimeError(f'{p.name}: expected 20Hz 3s windows [N,60,6], got {x.shape}')
        if 'subject_id' not in z: raise RuntimeError(f'{p.name}: missing subject_id; V2 refuses random-window public validation')
        sid=np.asarray(z['subject_id']).astype(str); y=np.full(len(x),-1,np.int64)
        if name!='PAMAP2':
            ky='y' if 'y' in z else ('label' if 'label' in z else None)
            if ky is None: raise RuntimeError(f'{p.name}: missing fall/non-fall y')
            y=np.asarray(z[ky],np.int64); subjects=np.unique(sid); nv=min(len(subjects)-1,max(2,int(round(.10*len(subjects)))))
            vs=set(rng.choice(subjects,nv,replace=False).tolist()); vi=np.array([i for i,s in enumerate(sid) if s in vs]); ti=np.array([i for i,s in enumerate(sid) if s not in vs])
            assert set(sid[ti]).isdisjoint(set(sid[vi]))
        else: ti=np.arange(len(x)); vi=np.array([],int)
        st=b.fit_domain_stats_windows(x[ti]); stats[name]={'acc_scale':st['acc_scale'],'gyro_bias':st['gyro_bias'].tolist(),'gyro_scale':st['gyro_scale']}
        tr[name]=pre_pub(b.domain_norm_windows(x[ti],st),y[ti],c,f'{name}/train'); tr[name]['subject_id']=sid[ti]
        if len(vi):
            va[name]=pre_pub(b.domain_norm_windows(x[vi],st),y[vi],c,f'{name}/val'); va[name]['subject_id']=sid[vi]
    return tr,va,stats

def find_inputs(inp):
    roots=[p for p in inp.rglob('*') if p.is_dir() and (p/'fall').is_dir() and (p/'non-fall').is_dir()]
    if not roots: raise RuntimeError('Own dataset not found')
    own=sorted(roots,key=lambda p:len(str(p)))[0]
    roots=[]
    for p in inp.rglob('*.npz'):
        if any(p.name.startswith(n) for n in PUBLIC_SSL): roots.append(p.parent)
    if not roots: raise RuntimeError('participant-aware public V2 caches not found')
    return own,sorted(set(roots),key=lambda p:len(str(p)))[0]

def ap(y,p):
    y=np.asarray(y); p=np.asarray(p); m=y>=0; y=y[m]; p=p[m]
    return float(average_precision_score(y,p)) if len(np.unique(y))==2 else float('nan')

def pred_dict(model,d,dev,batch=512):
    model.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(d['label']),batch):
            xs=[torch.from_numpy(d[k][s:s+batch]).to(dev) for k in ('full','pre','impact','post')]
            out.extend(torch.softmax(model(*xs)['logits'],1)[:,1].cpu().tolist())
    return np.asarray(d['label']),np.asarray(out)

def own_record_ap(model,d,dev):
    y,p=pred_dict(model,d,dev); z={}
    for k,fl,pp in zip(d['key'],d['file_label'],p): z.setdefault(k,[int(fl),[]])[1].append(float(pp))
    yy=np.array([z[k][0] for k in sorted(z)]); pp=np.array([max(z[k][1]) for k in sorted(z)])
    return ap(yy,pp)

def eval_domains(model,ov,pv,dev):
    out={'own_ap':own_record_ap(model,ov,dev),'public_ap':{}}
    for n,d in pv.items(): y,p=pred_dict(model,d,dev); out['public_ap'][n]=ap(y,p)
    vals=[v for v in out['public_ap'].values() if np.isfinite(v)]; out['public_mean_ap']=float(np.mean(vals)) if vals else float('nan')
    return out

def choose(cands,tol=.02):
    bo=max(x['metrics']['own_ap'] for x in cands); e=[x for x in cands if x['metrics']['own_ap']>=bo-tol]
    return max(e,key=lambda x:(x['metrics']['public_mean_ap'],x['metrics']['own_ap']))

def pack(d,dev): return {k:torch.from_numpy(np.asarray(d[k])).to(dev) for k in ('full','pre','impact','post','label','phase')}
def take(p,i): return [p[k][i] for k in ('full','pre','impact','post')]
def bal(y,n):
    a=torch.where(y==1)[0]; b0=torch.where(y==0)[0]
    if not len(a) or not len(b0): return torch.randint(0,len(y),(n,),device=y.device)
    na=n//2; return torch.cat([a[torch.randint(0,len(a),(na,),device=y.device)],b0[torch.randint(0,len(b0),(n-na,),device=y.device)]])

def probe_ssl(state,c,otr,ov,pv,dev):
    m=b.Net(c).to(dev); m.load_state_dict(state)
    for p in m.parameters(): p.requires_grad=False
    for p in m.cls.parameters(): p.requires_grad=True
    m.eval(); m.cls.train(); pk=pack(otr,dev); opt=torch.optim.AdamW(m.cls.parameters(),lr=3e-3,weight_decay=1e-4)
    for _ in range(20):
        for _ in range(max(1,math.ceil(len(pk['label'])/256))):
            i=bal(pk['label'],256); o=m(*take(pk,i)); loss=F.cross_entropy(o['logits'],pk['label'][i]); opt.zero_grad(); loss.backward(); opt.step()
    return eval_domains(m,ov,pv,dev)

def ssl_stage(model,c,ossl,otr,ov,ptr,pv,dev,out):
    op=pack(ossl,dev); pubs={n:pack(ptr[n],dev) for n in PUBLIC_SSL}; opt=torch.optim.AdamW(model.parameters(),lr=c.ssl_lr,weight_decay=c.wd); cands=[]; hist=[]
    for step in range(1,5001):
        on=128; i=torch.randint(0,len(op['label']),(on,),device=dev); os=take(op,i); oy=op['label'][i]; ext=[[] for _ in range(4)]
        for n in PUBLIC_SSL:
            p=pubs[n]; j=torch.randint(0,len(p['label']),(25 if n!='CGU_BES' else 28,),device=dev); s=take(p,j)
            for k in range(4): ext[k].append(s[k])
        base=[torch.cat([os[k],*ext[k]],0) for k in range(4)]; v1=b.ssl_augment(base); v2=b.ssl_augment(base); a=model(*v1); q=model(*v2)
        li=b.ntxent(a['proj'],q['proj'],c.temp); zz=torch.cat([a['proj'][:on],q['proj'][:on]]); yy=torch.cat([oy,oy]); loss=li+c.ssl_supcon_w*b.supcon(zz,yy,c.temp)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0: hist.append({'step':step,'loss':float(loss.detach())}); print('SSL',hist[-1],flush=True)
        if step%1000==0:
            st=copy.deepcopy(model.state_dict()); met=probe_ssl(st,c,otr,ov,pv,dev); pth=out/'checkpoints'/f'ssl_{step}.pt'; torch.save({'model':st,'metrics':met},pth); cands.append({'id':f'ssl_{step}','metrics':met,'state':st}); print('SSL VAL',met,flush=True)
    ch=choose(cands); model.load_state_dict(ch['state']); return ch,hist

def public_stage(model,c,ptr,ov,pv,dev,out):
    pubs={n:pack(ptr[n],dev) for n in PUBLIC_FALL}; opt=torch.optim.AdamW(model.parameters(),lr=c.all_lr,weight_decay=c.wd); cands=[]; hist=[]
    steps=max(math.ceil(len(pubs[n]['label'])/64) for n in PUBLIC_FALL)
    for ep in range(1,31):
        model.train(); ls=[]
        for _ in range(steps):
            seg=[[] for _ in range(4)]; ys=[]
            for n in PUBLIC_FALL:
                p=pubs[n]; i=bal(p['label'],64); s=take(p,i); ys.append(p['label'][i]); [seg[k].append(s[k]) for k in range(4)]
            x=[torch.cat(v) for v in seg]; y=torch.cat(ys); o=model(*x); loss=F.cross_entropy(o['logits'],y)+c.supcon_w*b.supcon(o['proj'],y,c.temp); opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss.detach()))
        met=eval_domains(model,ov,pv,dev); st=copy.deepcopy(model.state_dict()); cands.append({'id':f'public_{ep}','metrics':met,'state':st}); hist.append({'epoch':ep,'loss':float(np.mean(ls)),**met}); print('PUBLIC',hist[-1],flush=True)
    ch=choose(cands); model.load_state_dict(ch['state']); return ch,hist

def final_stage(model,c,otr,ptr,ov,pv,dev,out):
    own=pack(otr,dev); pubs={n:pack(ptr[n],dev) for n in PUBLIC_FALL}; cands=[]; hist=[]
    for stage,epochs,lr,enc in [('partial_freeze',10,c.head_lr,False),('all',50,c.all_lr,True)]:
        for p in list(model.acc.parameters())+list(model.gyr.parameters()): p.requires_grad=enc
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=c.wd)
        for ep in range(1,epochs+1):
            model.train(); ls=[]
            for _ in range(max(1,math.ceil(len(own['label'])/218))):
                oi=bal(own['label'],218); os=take(own,oi); oy=own['label'][oi]; ph=own['phase'][oi]; seg=[[] for _ in range(4)]; ys=[]
                for n in PUBLIC_FALL:
                    p=pubs[n]; ii=bal(p['label'],10 if n!='CGU_BES' else 8); ss=take(p,ii); ys.append(p['label'][ii]); [seg[k].append(ss[k]) for k in range(4)]
                ps=[torch.cat(v) for v in seg]; x=[torch.cat([os[k],ps[k]]) for k in range(4)]; y=torch.cat([oy,*ys]); o=model(*x)
                loss=F.cross_entropy(o['logits'],y)+c.supcon_w*b.supcon(o['proj'],y,c.temp)+c.phase_w*F.cross_entropy(o['phase_logits'][:218],ph)
                opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss.detach()))
            met=eval_domains(model,ov,pv,dev); st=copy.deepcopy(model.state_dict()); cands.append({'id':f'{stage}_{ep}','metrics':met,'state':st}); hist.append({'stage':stage,'epoch':ep,'loss':float(np.mean(ls)),**met}); print('FINAL',hist[-1],flush=True)
    ch=choose(cands); model.load_state_dict(ch['state']); return ch,hist

def prepare(c,inp,out):
    own,proot=find_inputs(inp); rs=b.records(own); arr={r['rel']:b.load_resampled(r['path'],c.fs) for r in rs}; s,sp=own_split(rs,arr,c,out/'splits'/'own_seed42.json')
    total=sum(nw(len(arr[r['rel']]),c) for r in rs); summ={k:{'recordings':len(v),'windows':sum(nw(len(arr[r['rel']]),c) for r in v)} for k,v in sp.items()}; [summ[k].update({'fraction':summ[k]['windows']/total}) for k in summ]; print('OWN SPLIT',summ,flush=True)
    st=b.fit_domain_stats_from_arrays([arr[r['rel']] for r in sp['train']]); es=b.build_entries(sp['train'],arr,c,'ssl'); et=b.build_entries(sp['train'],arr,c,'train'); ev=b.build_entries(sp['val'],arr,c,'eval')
    ossl=b.precompute_entries(es,arr,st,c); otr=b.precompute_entries(et,arr,st,c); ov=b.precompute_entries(ev,arr,st,c); ptr,pv,pst=public_data(proot,c)
    audit={'own':summ,'public':{n:{'train_subjects':len(np.unique(ptr[n]['subject_id'])),'val_subjects':len(np.unique(pv[n]['subject_id'])) if n in pv else 0} for n in PUBLIC_SSL},'checks':'LEAKAGE CHECK PASS'}
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2)); (out/'normalization.json').write_text(json.dumps({'own':{'acc_scale':st['acc_scale'],'gyro_bias':st['gyro_bias'].tolist(),'gyro_scale':st['gyro_scale']},'public':pst},indent=2)); print('LEAKAGE CHECK PASS',audit,flush=True)
    return own,arr,sp,st,ossl,otr,ov,ptr,pv

def train(args):
    c=cfg_v2(); b.seed_all(c.seed); inp=Path(args.input_root); out=Path(args.output_root); (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'splits').mkdir(exist_ok=True); (out/'run_config.json').write_text(json.dumps(asdict(c),indent=2))
    _,_,_,st,ossl,otr,ov,ptr,pv=prepare(c,inp,out); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=b.Net(c).to(dev)
    sc,sh=ssl_stage(model,c,ossl,otr,ov,ptr,pv,dev,out); pc,ph=public_stage(model,c,ptr,ov,pv,dev,out); fc,fh=final_stage(model,c,otr,ptr,ov,pv,dev,out)
    ld=b.static_loader(ov,256); rows=b.predict(model,ld,dev); _,y,p=b.aggregate(rows); th,vm=b.tune(y,p,c.min_val_recall)
    sel={k:{'id':v['id'],'metrics':v['metrics']} for k,v in [('ssl',sc),('public',pc),('final',fc)]}; torch.save({'model':model.state_dict(),'config':asdict(c),'threshold':th,'target_domain_norm':{'acc_scale':st['acc_scale'],'gyro_bias':st['gyro_bias'].tolist(),'gyro_scale':st['gyro_scale']}},out/'locked_model.pt')
    (out/'checkpoint_selection.json').write_text(json.dumps(sel,indent=2)); (out/'threshold.json').write_text(json.dumps({'threshold':th,'own_val':vm},indent=2)); (out/'training_history.json').write_text(json.dumps({'ssl':sh,'public':ph,'final':fh},indent=2)); print('TRAIN COMPLETE; OWN TEST NOT TOUCHED',flush=True)

def final_test(args):
    inp=Path(args.input_root); out=Path(args.output_root); ck=torch.load(out/'locked_model.pt',map_location='cpu'); c=cfg_v2(); own,_=find_inputs(inp); rs=b.records(own); arr={r['rel']:b.load_resampled(r['path'],c.fs) for r in rs}; s=json.loads((out/'splits'/'own_seed42.json').read_text()); m={r['rel']:r for r in rs}; te=[m[x] for x in s['splits']['test']]
    st={'acc_scale':ck['target_domain_norm']['acc_scale'],'gyro_bias':np.asarray(ck['target_domain_norm']['gyro_bias'],np.float32),'gyro_scale':ck['target_domain_norm']['gyro_scale']}; ee=b.build_entries(te,arr,c,'eval'); d=b.precompute_entries(ee,arr,st,c); ld=b.static_loader(d,256); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=b.Net(c).to(dev); model.load_state_dict(ck['model']); rows=b.predict(model,ld,dev); _,y,p=b.aggregate(rows)
    res={'own_test_recording':b.metrics(y,p,ck['threshold']),'own_test_window':b.window_metrics(rows,ck['threshold']),'note':'same-person unseen-recording test, not unseen-subject'}; (out/'final_test_metrics.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['train','final_test'],required=True); p.add_argument('--input-root',default='/kaggle/input'); p.add_argument('--output-root',default='/kaggle/working/exp2_v2'); a=p.parse_args(); train(a) if a.mode=='train' else final_test(a)
if __name__=='__main__': main()
