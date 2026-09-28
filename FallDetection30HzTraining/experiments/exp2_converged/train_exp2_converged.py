from __future__ import annotations

import argparse, copy, json, math, random, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

FEATURES = ["Acc_X","Acc_Y","Acc_Z","Gyro_X","Gyro_Y","Gyro_Z"]
LABELS = {"fall":1, "non-fall":0}

@dataclass
class Cfg:
    seed:int=42; fs:float=30.; win_sec:float=3.; stride_sec:float=.75
    pos_sec:float=.75; amb_sec:float=1.5; impact_sec:float=.7; ctx_sec:float=1.
    channels:int=24; blocks:int=3; dropout:float=.2; proj_dim:int=64
    batch:int=128; ssl_epochs:int=20; head_epochs:int=3; all_epochs:int=17
    ssl_lr:float=3e-4; head_lr:float=3e-4; all_lr:float=8e-5; wd:float=1e-4
    supcon_w:float=.2; phase_w:float=.5; ssl_supcon_w:float=.5; ssl_inst_w:float=1.
    temp:float=.1; max_ssl_per_rec:int=64; max_neg_per_rec:int=24; patience:int=5
    min_val_recall:float=.70

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def extract(zip_path, work):
    zip_path = Path(zip_path)
    if zip_path.is_dir():
        return zip_path
    root = work/"data"/"30Hz_processed_clean_v1"
    if root.exists(): return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.startswith("__MACOSX/") and "/._" not in n: z.extract(n, root.parent)
    return root

def records(root):
    out=[]
    for name,y in LABELS.items():
        for p in sorted((root/name).glob("*.csv")):
            out.append({"path":p,"rel":str(p.relative_to(root)),"label":y})
    if len(out)!=229: print("WARNING recordings=",len(out))
    return out

def split_records(rs, path, seed):
    if path.exists():
        s=json.loads(path.read_text()); m={r["rel"]:r for r in rs}
        return {k:[m[x] for x in v] for k,v in s.items()}
    rel=np.array([r["rel"] for r in rs]); y=np.array([r["label"] for r in rs])
    tr,tmp,ytr,ytmp=train_test_split(rel,y,test_size=.30,random_state=seed,stratify=y)
    va,te,_,_=train_test_split(tmp,ytmp,test_size=.50,random_state=seed,stratify=ytmp)
    s={"train":sorted(tr.tolist()),"val":sorted(va.tolist()),"test":sorted(te.tolist())}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(s,indent=2))
    m={r["rel"]:r for r in rs}; return {k:[m[x] for x in v] for k,v in s.items()}

def robust_z(v):
    med=np.median(v); mad=np.median(np.abs(v-med)); return (v-med)/(1.4826*mad+1e-6)

def load_resampled(path,fs):
    d=pd.read_csv(path)
    for c in ["time_ms",*FEATURES]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["time_ms",*FEATURES]).sort_values("time_ms")
    d=d.groupby("time_ms",as_index=False)[FEATURES].mean()
    t=d.time_ms.to_numpy(np.float64)/1000.; x=d[FEATURES].to_numpy(np.float32)
    t=t-t[0]; n=max(2,int(math.floor(t[-1]*fs))+1); g=np.arange(n)/fs
    return np.stack([np.interp(g,t,x[:,i]) for i in range(6)],1).astype(np.float32)

def event_peak(x,fs):
    a=np.linalg.norm(x[:,:3],axis=1); g=np.linalg.norm(x[:,3:],axis=1)
    j=np.zeros_like(a); j[1:]=np.linalg.norm(np.diff(x[:,:3],axis=0),axis=1)*fs
    s=np.abs(robust_z(a))+.5*np.abs(robust_z(g))+.25*np.abs(robust_z(j))
    k=max(1,int(round(.2*fs)))
    if k>1: s=np.convolve(s,np.ones(k)/k,mode="same")
    return int(np.argmax(s))

def starts(n,w,fs,stride_sec):
    if n<=w: return [0]
    a=np.unique(np.rint(np.arange(0,n-w+1e-9,stride_sec*fs)).astype(int))
    a=np.clip(a,0,n-w)
    if a[-1]!=n-w: a=np.r_[a,n-w]
    return a.tolist()

def pad(x,s,n):
    z=x[max(0,s):min(len(x),s+n)]
    if len(z)<n: z=np.pad(z,((0,n-len(z)),(0,0)),mode="edge") if len(z) else np.zeros((n,6),np.float32)
    return z.astype(np.float32,copy=False)

def phase(center,peak,fs,impact_sec):
    h=int(round(impact_sec*fs/2))
    return 1 if center<peak-h else (3 if center>peak+h else 2)

def build_entries(rs,arr,cfg,mode):
    assert mode in {"ssl","train","eval"}
    w=int(round(cfg.win_sec*cfg.fs)); pr=cfg.pos_sec*cfg.fs; ar=cfg.amb_sec*cfg.fs
    rng=np.random.default_rng(cfg.seed+(1 if mode=="train" else 2)); out=[]
    for r in rs:
        x=arr[r["rel"]]; pk=event_peak(x,cfg.fs) if r["label"] else None; loc=[]
        for s in starts(len(x),w,cfg.fs,cfg.stride_sec):
            c=s+w//2
            if mode=="ssl":
                loc.append({"key":r["rel"],"start":s,"label":r["label"],"phase":0,"file_label":r["label"]}); continue
            if not r["label"]: y,ph=0,0
            else:
                d=abs(c-pk)
                if d<=pr: y,ph=1,phase(c,pk,cfg.fs,cfg.impact_sec)
                elif d<=ar:
                    if mode=="train": continue
                    y,ph=-1,phase(c,pk,cfg.fs,cfg.impact_sec)
                else: y,ph=0,0
            loc.append({"key":r["rel"],"start":s,"label":y,"phase":ph,"file_label":r["label"]})
        if mode=="ssl" and len(loc)>cfg.max_ssl_per_rec:
            ids=np.unique(np.linspace(0,len(loc)-1,cfg.max_ssl_per_rec).round().astype(int)); loc=[loc[i] for i in ids]
        if mode=="train":
            p=[e for e in loc if e["label"]==1]; n=[e for e in loc if e["label"]==0]
            if len(n)>cfg.max_neg_per_rec:
                ids=sorted(rng.choice(len(n),cfg.max_neg_per_rec,replace=False).tolist()); n=[n[i] for i in ids]
            loc=p+n
        out.extend(loc)
    return out

def fit_norm(rs,arr):
    xs=[]
    for r in rs:
        x=arr[r["rel"]]
        if len(x)>3000: x=x[np.linspace(0,len(x)-1,3000).astype(int)]
        xs.append(x)
    z=np.concatenate(xs).astype(np.float64); m=z.mean(0).astype(np.float32); s=np.maximum(z.std(0),1e-6).astype(np.float32)
    return m,s

def local_ctx(w,fs,sec):
    p=event_peak(w,fs); n=int(round(sec*fs)); L=len(w)
    i=int(np.clip(p-n//2,0,max(0,L-n))); a=max(0,i-n); b=min(max(0,L-n),i+n)
    return pad(w,a,n),pad(w,i,n),pad(w,b,n)

class IMUDS(Dataset):
    def __init__(self,entries,arr,mean,std,cfg,aug=False,two=False):
        self.e=list(entries); self.arr=arr; self.m=mean[None]; self.s=std[None]
        self.cfg=cfg; self.aug=aug; self.two=two; self.w=int(round(cfg.win_sec*cfg.fs))
    def __len__(self): return len(self.e)
    def augment(self,x):
        x=x.copy(); x[:,:3]*=np.random.normal(1,.1); x[:,3:]*=np.random.normal(1,.1)
        if np.random.rand()<.7:
            q=max(1,int(.1*len(x))); x=np.roll(x,np.random.randint(-q,q+1),axis=0)
        if np.random.rand()<.5:
            q=max(1,int(.1*len(x))); s=np.random.randint(0,max(1,len(x)-q+1)); x[s:s+q]=0
        if np.random.rand()<.2: x[:,np.random.randint(0,6)]=0
        return x
    def make(self,e,aug):
        w=(pad(self.arr[e["key"]],e["start"],self.w)-self.m)/self.s
        if aug: w=self.augment(w)
        a,b,c=local_ctx(w,self.cfg.fs,self.cfg.ctx_sec)
        return [torch.from_numpy(z.astype(np.float32)) for z in (w,a,b,c)]
    def __getitem__(self,i):
        e=self.e[i]; z=self.make(e,self.aug)
        o={"full":z[0],"pre":z[1],"impact":z[2],"post":z[3],
           "label":torch.tensor(e["label"]),"phase":torch.tensor(e["phase"]),
           "file_label":torch.tensor(e["file_label"]),"key":e["key"],"start":e["start"]}
        if self.two:
            q=self.make(e,True)
            for k,v in zip(("full2","pre2","impact2","post2"),q): o[k]=v
        return o

class Block(nn.Module):
    def __init__(self,c,d,drop):
        super().__init__(); p=2*d
        self.c1=nn.Conv1d(c,c,5,padding=p,dilation=d,bias=False); self.b1=nn.BatchNorm1d(c)
        self.c2=nn.Conv1d(c,c,5,padding=p,dilation=d,bias=False); self.b2=nn.BatchNorm1d(c); self.drop=nn.Dropout(drop)
    def forward(self,x):
        y=self.drop(F.gelu(self.b1(self.c1(x)))); y=self.drop(self.b2(self.c2(y))); return F.gelu(x+y)

class Encoder(nn.Module):
    def __init__(self,c,blocks,drop):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(3,c,5,padding=2,bias=False),nn.BatchNorm1d(c),nn.GELU())
        self.tcn=nn.Sequential(*[Block(c,2**i,drop) for i in range(blocks)])
        h=max(4,c//2); self.score=nn.Sequential(nn.Conv1d(c,h,1),nn.GELU(),nn.Conv1d(h,1,1))
    def forward(self,x):
        h=self.tcn(self.stem(x.transpose(1,2))); a=torch.softmax(self.score(h).squeeze(1),-1)
        return (h*a[:,None]).sum(-1)

class Net(nn.Module):
    def __init__(self,cfg):
        super().__init__(); c=cfg.channels
        self.acc=Encoder(c,cfg.blocks,cfg.dropout); self.gyr=Encoder(c,cfg.blocks,cfg.dropout)
        self.fuse=nn.Sequential(nn.LayerNorm(2*c),nn.Linear(2*c,c),nn.GELU(),nn.Dropout(cfg.dropout))
        h=max(8,c//2); self.token=nn.Sequential(nn.LayerNorm(c),nn.Linear(c,h),nn.GELU(),nn.Linear(h,1))
        self.cls=nn.Linear(c,2); self.phase=nn.Linear(c,4)
        self.proj=nn.Sequential(nn.Linear(c,c),nn.GELU(),nn.Linear(c,cfg.proj_dim))
    def seg(self,x): return self.fuse(torch.cat([self.acc(x[:,:,:3]),self.gyr(x[:,:,3:])],-1))
    def forward(self,full,pre,impact,post):
        t=torch.stack([self.seg(x) for x in (full,pre,impact,post)],1); a=torch.softmax(self.token(t).squeeze(-1),1)
        h=(t*a[:,:,None]).sum(1)
        return {"logits":self.cls(h),"phase_logits":self.phase(h),"proj":F.normalize(self.proj(h),dim=-1)}

def ntxent(z1,z2,t=.1):
    # Explicitly leave AMP for the similarity matrix/loss. autocast can otherwise
    # cast the FP32 matmul back to FP16 even after calling .float().
    dev=z1.device.type
    with torch.amp.autocast(dev, enabled=False):
        n=len(z1); z=torch.cat([z1,z2]).float(); s=z@z.T/float(t)
        eye=torch.eye(2*n,device=z.device,dtype=torch.bool)
        s=s.masked_fill(eye,-1e9); y=(torch.arange(2*n,device=z.device)+n)%(2*n)
        return F.cross_entropy(s,y)

def supcon(z,y,t=.1):
    dev=z.device.type
    with torch.amp.autocast(dev, enabled=False):
        zf=z.float(); s=zf@zf.T/float(t); eye=torch.eye(len(zf),device=z.device,dtype=torch.bool)
        pos=y[:,None].eq(y[None])&~eye; s=s.masked_fill(eye,-1e9)
        lp=s-torch.logsumexp(s,1,keepdim=True); d=pos.sum(1); ok=d>0
        return -(lp.mul(pos).sum(1)[ok]/d[ok]).mean() if ok.any() else zf.sum()*0

def batch_x(b,dev,suf=""):
    return [b[k+suf].to(dev,non_blocking=True) for k in ("full","pre","impact","post")]

def loader(ds,batch,train=False,balanced=False):
    sam=None; sh=train
    if balanced:
        ys=np.array([e["label"] for e in ds.e]); cnt=np.bincount(ys,minlength=2).clip(min=1)
        sam=WeightedRandomSampler(np.array([1/cnt[y] for y in ys]),len(ys),replacement=True); sh=False
    return DataLoader(ds,batch_size=batch,shuffle=sh,sampler=sam,num_workers=0,pin_memory=torch.cuda.is_available())

def pretrain(model,ld,cfg,dev,out):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.wd)
    for ep in range(cfg.ssl_epochs):
        model.train(); ls=[]
        for b in ld:
            y=b["label"].to(dev); a=model(*batch_x(b,dev)); q=model(*batch_x(b,dev,"2"))
            li=ntxent(a["proj"],q["proj"],cfg.temp); z=torch.cat([a["proj"],q["proj"]]); yy=torch.cat([y,y])
            loss=cfg.ssl_inst_w*li+cfg.ssl_supcon_w*supcon(z,yy,cfg.temp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls.append(loss.item())
        print(f"SSL {ep+1:02d}/{cfg.ssl_epochs} loss={np.mean(ls):.4f}")
    torch.save({"model":model.state_dict(),"config":asdict(cfg)},out/"ssl_pretrained.pt")

def metrics(y,p,th):
    y=np.asarray(y,int); pred=(np.asarray(p)>=th).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(th),"n":len(y),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/max(1,tn+fp)),
            "f2":float(fbeta_score(y,pred,beta=2,zero_division=0)),
            "mcc":float(matthews_corrcoef(y,pred)) if len(np.unique(y))==2 else 0.}

def tune(y,p,min_recall):
    best=None
    for th in np.unique(np.r_[.01,np.linspace(.05,.95,181),.99,p]):
        m=metrics(y,p,float(th)); score=(m["recall"]>=min_recall,m["mcc"],m["f2"],m["specificity"])
        if best is None or score>best[0]: best=(score,float(th),m)
    return best[1],best[2]

@torch.no_grad()
def predict(model,ld,dev):
    model.eval(); rows=[]
    for b in ld:
        pr=torch.softmax(model(*batch_x(b,dev))["logits"],1)[:,1].cpu().numpy()
        for i,v in enumerate(pr):
            rows.append({"key":b["key"][i],"start":int(b["start"][i]),"label":int(b["label"][i]),
                         "file_label":int(b["file_label"][i]),"prob":float(v)})
    return rows

def aggregate(rows):
    d={}
    for r in rows: d.setdefault(r["key"],{"label":r["file_label"],"p":[]})["p"].append(r["prob"])
    k=sorted(d); return k,[d[x]["label"] for x in k],[max(d[x]["p"]) for x in k]

def window_metrics(rows,th):
    r=[x for x in rows if x["label"]>=0]; return metrics([x["label"] for x in r],[x["prob"] for x in r],th)

def save_rows(rows,path): pd.DataFrame(rows).to_csv(path,index=False)

def fit_domain_stats_from_arrays(xs, max_rows_per_array=4000):
    chunks=[]
    for x in xs:
        if len(x)>max_rows_per_array:
            ids=np.linspace(0,len(x)-1,max_rows_per_array).astype(int); x=x[ids]
        chunks.append(np.asarray(x,np.float32))
    z=np.concatenate(chunks,0)
    acc_scale=float(np.median(np.linalg.norm(z[:,:3],axis=1)))
    gyro_bias=np.median(z[:,3:],axis=0).astype(np.float32)
    gyro_scale=float(np.quantile(np.linalg.norm(z[:,3:]-gyro_bias[None],axis=1),.95))
    return {"acc_scale":max(acc_scale,1e-6),"gyro_bias":gyro_bias,"gyro_scale":max(gyro_scale,1e-6)}

def fit_domain_stats_windows(x):
    z=np.asarray(x,np.float32).reshape(-1,6)
    if len(z)>500000: z=z[np.linspace(0,len(z)-1,500000).astype(int)]
    acc_scale=float(np.median(np.linalg.norm(z[:,:3],axis=1)))
    gyro_bias=np.median(z[:,3:],axis=0).astype(np.float32)
    gyro_scale=float(np.quantile(np.linalg.norm(z[:,3:]-gyro_bias[None],axis=1),.95))
    return {"acc_scale":max(acc_scale,1e-6),"gyro_bias":gyro_bias,"gyro_scale":max(gyro_scale,1e-6)}

def domain_norm(x,st):
    z=np.asarray(x,np.float32).copy()
    z[:,:3]/=st["acc_scale"]
    z[:,3:]=(z[:,3:]-st["gyro_bias"][None])/st["gyro_scale"]
    return z

def domain_norm_windows(x,st):
    z=np.asarray(x,np.float32).copy()
    z[:,:,:3]/=st["acc_scale"]
    z[:,:,3:]=(z[:,:,3:]-st["gyro_bias"][None,None])/st["gyro_scale"]
    return z

def precompute_entries(entries,arr,st,cfg):
    full=[]; pre=[]; imp=[]; post=[]; ys=[]; ph=[]; fl=[]; keys=[]; starts_=[]
    w=int(round(cfg.win_sec*cfg.fs))
    for e in entries:
        x=domain_norm(pad(arr[e["key"]],e["start"],w),st)
        a,b,c=local_ctx(x,cfg.fs,cfg.ctx_sec)
        full.append(x); pre.append(a); imp.append(b); post.append(c)
        ys.append(e["label"]); ph.append(e["phase"]); fl.append(e["file_label"])
        keys.append(e["key"]); starts_.append(e["start"])
    return {"full":np.stack(full).astype(np.float32),"pre":np.stack(pre).astype(np.float32),
            "impact":np.stack(imp).astype(np.float32),"post":np.stack(post).astype(np.float32),
            "label":np.asarray(ys,np.int64),"phase":np.asarray(ph,np.int64),
            "file_label":np.asarray(fl,np.int64),"key":keys,"start":np.asarray(starts_,np.int64)}

def precompute_public_windows(x):
    full=np.asarray(x,np.float32); pre=[]; imp=[]; post=[]
    for w in full:
        a,b,c=local_ctx(w,30.,1.); pre.append(a); imp.append(b); post.append(c)
    return {"full":full,"pre":np.stack(pre).astype(np.float32),"impact":np.stack(imp).astype(np.float32),"post":np.stack(post).astype(np.float32)}

class StaticDS(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d["label"])
    def __getitem__(self,i):
        return {"full":torch.from_numpy(self.d["full"][i]),"pre":torch.from_numpy(self.d["pre"][i]),
                "impact":torch.from_numpy(self.d["impact"][i]),"post":torch.from_numpy(self.d["post"][i]),
                "label":torch.tensor(int(self.d["label"][i])),"phase":torch.tensor(int(self.d["phase"][i])),
                "file_label":torch.tensor(int(self.d["file_label"][i])),"key":self.d["key"][i],
                "start":torch.tensor(int(self.d["start"][i]))}

def static_loader(d,batch,train=False,balanced=False):
    ds=StaticDS(d); sam=None; sh=train
    if balanced:
        y=d["label"]; cnt=np.bincount(y,minlength=2).clip(min=1)
        sam=WeightedRandomSampler(np.asarray([1/cnt[v] for v in y]),len(y),replacement=True); sh=False
    return DataLoader(ds,batch_size=batch,shuffle=sh,sampler=sam,num_workers=0,pin_memory=True)

def random_so3(n,dev,dtype):
    q=F.normalize(torch.randn(n,4,device=dev,dtype=dtype),dim=1)
    w,x,y,z=q.unbind(1)
    R=torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                   2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                   2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1)
    return R.reshape(n,3,3)

def rotate_one(t,R):
    a=torch.bmm(t[:,:,:3],R.transpose(1,2)); g=torch.bmm(t[:,:,3:],R.transpose(1,2))
    return torch.cat([a,g],2)

def ssl_augment(segs):
    n=segs[0].shape[0]; dev=segs[0].device; dtype=segs[0].dtype; R=random_so3(n,dev,dtype)
    out=[rotate_one(x,R) for x in segs]
    sa=(1.+.10*torch.randn(n,1,1,device=dev,dtype=dtype)).clamp(.7,1.3)
    sg=(1.+.10*torch.randn(n,1,1,device=dev,dtype=dtype)).clamp(.7,1.3)
    out=[torch.cat([x[:,:,:3]*sa,x[:,:,3:]*sg],2) for x in out]
    drop=(torch.rand(n,device=dev)<.20); ch=torch.randint(0,6,(n,),device=dev)
    if drop.any():
        m=torch.ones(n,1,6,device=dev,dtype=dtype); m[torch.arange(n,device=dev),0,ch]=torch.where(drop,torch.zeros(n,device=dev,dtype=dtype),torch.ones(n,device=dev,dtype=dtype))
        out=[x*m for x in out]
    return out

def to_gpu_pack(d,dev):
    return {k:torch.from_numpy(d[k]).to(dev,non_blocking=True) for k in ("full","pre","impact","post")}

def take_pack(d,idx): return [d[k][idx] for k in ("full","pre","impact","post")]

def own_ssl_pack(entries,arr,st,cfg,dev):
    d=precompute_entries(entries,arr,st,cfg)
    pack=to_gpu_pack(d,dev); y=torch.from_numpy(d["label"]).to(dev)
    return pack,y

def prepare_public(cache_root,dev):
    names=['CGU_BES','Cogent','SFU_IMU','UCI_SimulatedFalls','PAMAP2']; ssl=[]; stats={}
    for si,name in enumerate(names):
        z=np.load(cache_root/f'{name}.npz'); x=z['ssl_x'].astype(np.float32)
        st=fit_domain_stats_windows(x); x=domain_norm_windows(x,st); d=precompute_public_windows(x)
        ssl.append({"name":name,"pack":to_gpu_pack(d,dev),"n":len(x)})
        stats[name]={"acc_scale":st["acc_scale"],"gyro_bias":st["gyro_bias"].tolist(),"gyro_scale":st["gyro_scale"]}
    z=np.load(cache_root/'supervised_public_4x2000.npz'); x=z['x'].astype(np.float32); y=z['y'].astype(np.int64); src=z['source'].astype(np.int64)
    xx=np.empty_like(x)
    for si,name in enumerate(names[:4]):
        ids=np.flatnonzero(src==si); st0=stats[name]
        st={"acc_scale":st0["acc_scale"],"gyro_bias":np.asarray(st0["gyro_bias"],np.float32),"gyro_scale":st0["gyro_scale"]}
        xx[ids]=domain_norm_windows(x[ids],st)
    d=precompute_public_windows(xx); d.update({"label":y,"phase":np.zeros(len(y),np.int64),"file_label":y,
        "key":[f'public/{int(src[i])}/{i}' for i in range(len(y))],"start":np.zeros(len(y),np.int64)})
    return names,ssl,d,stats

def ssl_train_steps(model,own_pack,own_y,cfg,dev,max_steps=5000,min_steps=2000,external=None,batch=256,check_every=100):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.ssl_lr,weight_decay=cfg.wd)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=.0025,threshold_mode='rel',min_lr=3e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    losses=[]; history=[]; best=float('inf'); stale=0
    for step in range(max_steps):
        own_n=batch//2; io=torch.randint(0,len(own_y),(own_n,),device=dev); own=take_pack(own_pack,io)
        ext_parts=[[] for _ in range(4)]; remain=batch-own_n; q,r=divmod(remain,len(external))
        for j,e in enumerate(external):
            m=q+(((j-step)%len(external))<r); ie=torch.randint(0,e["n"],(m,),device=dev); seg=take_pack(e["pack"],ie)
            for k in range(4): ext_parts[k].append(seg[k])
        base=[torch.cat([own[k],*ext_parts[k]],0) for k in range(4)]; y=own_y[io]
        v1=ssl_augment(base); v2=ssl_augment(base)
        with torch.amp.autocast('cuda',enabled=(dev.type=='cuda')):
            a=model(*v1); b=model(*v2); li=ntxent(a["proj"],b["proj"],cfg.temp)
            za=torch.cat([a["proj"][:own_n],b["proj"][:own_n]],0); yy=torch.cat([y,y],0)
            loss=cfg.ssl_inst_w*li+cfg.ssl_supcon_w*supcon(za,yy,cfg.temp)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        if (step+1)%check_every==0:
            avg=float(np.mean(losses[-check_every:])); lr=opt.param_groups[0]['lr']; sched.step(avg)
            improved=avg < best*(1-.0025)
            if improved: best=avg; stale=0
            else: stale+=1
            history.append({'step':step+1,'loss':avg,'lr':lr,'stale_checks':stale})
            print(f'SSL step {step+1}/{max_steps} loss={avg:.4f} lr={lr:.2e} plateau={stale}/6',flush=True)
            if step+1>=min_steps and stale>=6:
                print(f'SSL converged at step {step+1}: sustained <0.25% improvement over {stale} checks',flush=True)
                return history
    print(f'SSL reached max_steps={max_steps}',flush=True)
    return history

def source_supervised(model,d,cfg,dev,max_epochs=30,min_epochs=10):
    ld=static_loader(d,256,True,False); opt=torch.optim.AdamW(model.parameters(),lr=cfg.all_lr,weight_decay=cfg.wd)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=.0025,threshold_mode='rel',min_lr=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    history=[]; best=float('inf'); stale=0
    for ep in range(max_epochs):
        model.train(); ls=[]
        for b in ld:
            y=b['label'].to(dev); xs=batch_x(b,dev)
            with torch.amp.autocast('cuda',enabled=(dev.type=='cuda')):
                o=model(*xs); loss=F.cross_entropy(o['logits'],y)+cfg.supcon_w*supcon(o['proj'],y,cfg.temp)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ls.append(float(loss.detach()))
        avg=float(np.mean(ls)); lr=opt.param_groups[0]['lr']; sched.step(avg)
        improved=avg < best*(1-.0025)
        if improved: best=avg; stale=0
        else: stale+=1
        history.append({'epoch':ep+1,'loss':avg,'lr':lr,'stale_epochs':stale})
        print(f'PUBLIC SUP {ep+1}/{max_epochs} loss={avg:.4f} lr={lr:.2e} plateau={stale}/6',flush=True)
        if ep+1>=min_epochs and stale>=6:
            print(f'PUBLIC SUP converged at epoch {ep+1}',flush=True); break
    return history

def finetune_converged(model,tr,va,cfg,dev,out):
    best=copy.deepcopy(model.state_dict()); best_m=-2.; best_th=.5; history=[]
    stages=[('head',10,cfg.head_lr,False),('all',50,cfg.all_lr,True)]
    for name,epochs,lr,enc in stages:
        for p in list(model.acc.parameters())+list(model.gyr.parameters()): p.requires_grad=enc
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=cfg.wd)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,threshold=1e-4,min_lr=1e-6)
        bad=0
        for ep in range(epochs):
            model.train(); losses=[]
            for b in tr:
                y=b['label'].to(dev); ph=b['phase'].to(dev); o=model(*batch_x(b,dev))
                loss=F.cross_entropy(o['logits'],y)+cfg.supcon_w*supcon(o['proj'],y,cfg.temp)+cfg.phase_w*F.cross_entropy(o['phase_logits'],ph)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
            vr=predict(model,va,dev); _,vy,vp=aggregate(vr); th,vm=tune(vy,vp,cfg.min_val_recall); avg=float(np.mean(losses)); curlr=opt.param_groups[0]['lr']; sched.step(vm['mcc'])
            history.append({'stage':name,'epoch':ep+1,'loss':avg,'val_mcc':vm['mcc'],'val_recall':vm['recall'],'threshold':th,'lr':curlr})
            print(f"{name} {ep+1:02d}/{epochs} loss={avg:.4f} val_mcc={vm['mcc']:.4f} recall={vm['recall']:.4f} th={th:.3f} lr={curlr:.2e}",flush=True)
            if vm['mcc']>best_m+1e-6:
                best_m=vm['mcc']; best=copy.deepcopy(model.state_dict()); best_th=th; bad=0
            else: bad+=1
            if name=='all' and ep+1>=10 and bad>=10:
                print('TARGET FT converged by validation plateau',flush=True); break
    model.load_state_dict(best); torch.save({'model':best,'config':asdict(cfg),'threshold':best_th},out/'best_model.pt')
    return best_th,history

def evaluate_and_save(tag,model,th,va,te,cfg,st,outdir,extra):
    vr=predict(model,va,dev_global); tr=predict(model,te,dev_global); _,vy,vp=aggregate(vr); tk,ty,tp=aggregate(tr)
    result={"experiment":tag,"threshold":th,"val_recording":metrics(vy,vp,th),"test_recording":metrics(ty,tp,th),
            "test_window":window_metrics(tr,th),"config":asdict(cfg),
            "target_domain_norm":{"acc_scale":st["acc_scale"],"gyro_bias":st["gyro_bias"].tolist(),"gyro_scale":st["gyro_scale"]},**extra}
    od=outdir/tag; od.mkdir(parents=True,exist_ok=True); (od/'metrics.json').write_text(json.dumps(result,indent=2))
    save_rows(vr,od/'val_windows.csv'); save_rows(tr,od/'test_windows.csv'); pd.DataFrame({"key":tk,"label":ty,"prob":tp}).to_csv(od/'test_recordings.csv',index=False)
    torch.save({"model":model.state_dict(),"config":asdict(cfg),"threshold":th,"target_domain_norm":result["target_domain_norm"]},od/'best_model.pt')
    print(json.dumps(result,indent=2),flush=True); return result

def run_exp2_converged():
    global dev_global
    seed_all(42); cfg=Cfg(seed=42,batch=128); cfg.ssl_epochs=0; cfg.head_epochs=10; cfg.all_epochs=50; cfg.patience=10
    inp=Path('/kaggle/input'); roots=[p for p in inp.rglob('*') if p.is_dir() and (p/'fall').is_dir() and (p/'non-fall').is_dir()]
    if not roots: raise RuntimeError('target dataset not found')
    root=sorted(roots,key=lambda p:len(str(p)))[0]
    cache_candidates=list(inp.rglob('ssl_public_5x5000.npz'))
    if not cache_candidates: raise RuntimeError('public cache not found')
    cache_root=cache_candidates[0].parent
    out=Path('/kaggle/working/exp2_converged'); out.mkdir(parents=True,exist_ok=True); (out/'splits').mkdir(exist_ok=True)
    rs=records(root); sp=split_records(rs,out/'splits'/'seed42.json',42); print('split',{k:(len(v),sum(x['label'] for x in v)) for k,v in sp.items()},flush=True)
    arr={r['rel']:load_resampled(r['path'],30.) for r in rs}
    target_st=fit_domain_stats_from_arrays([arr[r['rel']] for r in sp['train']]); print('target domain stats',target_st,flush=True)
    es=build_entries(sp['train'],arr,cfg,'ssl'); et=build_entries(sp['train'],arr,cfg,'train'); ev=build_entries(sp['val'],arr,cfg,'eval'); ee=build_entries(sp['test'],arr,cfg,'eval')
    print('windows',{'ssl':len(es),'train':len(et),'val':len(ev),'test':len(ee)},flush=True)
    trd=precompute_entries(et,arr,target_st,cfg); vad=precompute_entries(ev,arr,target_st,cfg); ted=precompute_entries(ee,arr,target_st,cfg)
    ld_tr=static_loader(trd,128,True,True); ld_va=static_loader(vad,256); ld_te=static_loader(ted,256)
    dev_global=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev_global,flush=True)
    own_pack,own_y=own_ssl_pack(es,arr,target_st,cfg,dev_global); names,public_ssl,public_sup,public_stats=prepare_public(cache_root,dev_global)
    (out/'normalization.json').write_text(json.dumps({'method':'domain_scalar_rotation_equivariant','target':{'acc_scale':target_st['acc_scale'],'gyro_bias':target_st['gyro_bias'].tolist(),'gyro_scale':target_st['gyro_scale']},'public':public_stats},indent=2))
    seed_all(42); model=Net(cfg).to(dev_global); print('EXP2 CONVERGENCE RUN params',sum(p.numel() for p in model.parameters()),flush=True)
    ssl_hist=ssl_train_steps(model,own_pack,own_y,cfg,dev_global,max_steps=5000,min_steps=2000,external=public_ssl,batch=256)
    sup_hist=source_supervised(model,public_sup,cfg,dev_global,max_epochs=30,min_epochs=10)
    d=out/'exp2_balanced_external_converged'; (d/'checkpoints').mkdir(parents=True,exist_ok=True)
    th,ft_hist=finetune_converged(model,ld_tr,ld_va,cfg,dev_global,d/'checkpoints')
    hist={'ssl':ssl_hist,'public_supervised':sup_hist,'target_finetune':ft_hist}; (out/'training_history.json').write_text(json.dumps(hist,indent=2))
    result=evaluate_and_save('exp2_balanced_external_converged',model,th,ld_va,ld_te,cfg,target_st,out,{
        'ssl_steps_completed':ssl_hist[-1]['step'],'ssl_max_steps':5000,'ssl_min_steps':2000,'ssl_sources':['own_train',*names],
        'public_ssl_windows_per_dataset':5000,'public_supervised_windows_per_fall_dataset':2000,'public_supervised_epochs_completed':sup_hist[-1]['epoch'],
        'training_policy':'convergence-oriented; SSL/public-supervised loss plateau + validation-plateau target fine-tuning'})
    (out/'result.json').write_text(json.dumps(result,indent=2)); print('DONE EXP2 CONVERGED',flush=True)

if __name__=="__main__": run_exp2_converged()
