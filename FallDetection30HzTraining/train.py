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
    supcon_w:float=.2; phase_w:float=0.; ssl_supcon_w:float=.5; ssl_inst_w:float=1.
    temp:float=.1; max_ssl_per_rec:int=64; max_neg_per_rec:int=24; patience:int=5
    min_val_recall:float=.90
    label_mode:str="clip"; target_fraction:float=.5; steps_per_epoch:int=100

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def extract(zip_path, work):
    zip_path = Path(zip_path)
    if zip_path.is_dir():
        return zip_path
    root = work/"data"/zip_path.stem
    root.mkdir(parents=True, exist_ok=True)
    existing=[p for p in [root,*root.glob("*")] if p.is_dir() and (p/"fall").is_dir() and (p/"non-fall").is_dir()]
    if len(existing)==1: return existing[0]
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.startswith("__MACOSX/") or "/._" in n: continue
            destination=(root/n).resolve()
            if not destination.is_relative_to(root.resolve()): raise ValueError("Unsafe archive member: "+n)
            z.extract(n,root)
    candidates=[p for p in [root,*root.rglob("*")] if p.is_dir() and (p/"fall").is_dir() and (p/"non-fall").is_dir()]
    if len(candidates)!=1: raise ValueError("Expected one fall/non-fall dataset root")
    return candidates[0]

def records(root):
    out=[]
    for name,y in LABELS.items():
        for p in sorted((root/name).glob("*.csv")):
            out.append({"path":p,"rel":str(p.relative_to(root)),"label":y})
    if not out: raise ValueError(f"No fall/non-fall CSVs under {root}")
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
        x=arr[r["rel"]]; pk=event_peak(x,cfg.fs) if r["label"] and cfg.label_mode!="clip" else None; loc=[]
        for s in starts(len(x),w,cfg.fs,cfg.stride_sec):
            c=s+w//2
            if mode=="ssl" or cfg.label_mode=="clip":
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
    n=len(z1); z=torch.cat([z1,z2]); s=z@z.T/t; eye=torch.eye(2*n,device=z.device,dtype=torch.bool)
    s=s.masked_fill(eye,-1e9); y=(torch.arange(2*n,device=z.device)+n)%(2*n); return F.cross_entropy(s,y)

def supcon(z,y,t=.1):
    known=y>=0; z=z[known]; y=y[known]
    if len(z)<2: return z.sum()*0
    s=z@z.T/t; eye=torch.eye(len(z),device=z.device,dtype=torch.bool); pos=y[:,None].eq(y[None])&~eye
    s=s.masked_fill(eye,-1e9); lp=s-torch.logsumexp(s,1,keepdim=True); d=pos.sum(1); ok=d>0
    return -(lp.mul(pos).sum(1)[ok]/d[ok]).mean() if ok.any() else z.sum()*0

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

def finetune(model,tr,va,cfg,dev,out):
    best=copy.deepcopy(model.state_dict()); best_m=-2.; best_th=.5; bad=0
    stages=[("head",cfg.head_epochs,cfg.head_lr,False),("all",cfg.all_epochs,cfg.all_lr,True)]
    for name,epochs,lr,enc in stages:
        for p in list(model.acc.parameters())+list(model.gyr.parameters()): p.requires_grad=enc
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=cfg.wd)
        for ep in range(epochs):
            model.train(); losses=[]
            for b in tr:
                y=b["label"].to(dev); ph=b["phase"].to(dev); o=model(*batch_x(b,dev))
                loss=F.cross_entropy(o["logits"],y)+cfg.supcon_w*supcon(o["proj"],y,cfg.temp)+cfg.phase_w*F.cross_entropy(o["phase_logits"],ph)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
            vr=predict(model,va,dev); _,vy,vp=aggregate(vr); th,vm=tune(vy,vp,cfg.min_val_recall)
            print(f"{name} {ep+1:02d}/{epochs} loss={np.mean(losses):.4f} val_mcc={vm['mcc']:.4f} recall={vm['recall']:.4f} th={th:.3f}")
            if vm["mcc"]>best_m+1e-6:
                best_m=vm["mcc"]; best=copy.deepcopy(model.state_dict()); best_th=th; bad=0
            else:
                bad+=1
                if name=="all" and cfg.patience and bad>=cfg.patience: print("early stop"); break
        if name=="all" and cfg.patience and bad>=cfg.patience: break
    model.load_state_dict(best); torch.save({"model":best,"config":asdict(cfg),"threshold":best_th},out/"best_model.pt")
    return best_th

def save_rows(rows,path): pd.DataFrame(rows).to_csv(path,index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--zip",type=Path,required=False); ap.add_argument("--work",type=Path,default=Path("run_v2"))
    ap.add_argument("--channels",type=int,default=24); ap.add_argument("--ssl-epochs",type=int,default=20)
    ap.add_argument("--head-epochs",type=int,default=3); ap.add_argument("--all-epochs",type=int,default=17)
    ap.add_argument("--batch",type=int,default=128); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--public-root",type=Path,help="Root containing the five processed_v2 dataset directories")
    ap.add_argument("--mode",choices=["train","test","prepare"],default="train")
    ap.add_argument("--target-fraction",type=float,default=.5,help="Own V2 batch share; public half is balanced across five sources")
    ap.add_argument("--steps-per-epoch",type=int,default=100)
    ap.add_argument("--device",default="auto",help="auto, cpu, or cuda:N")
    a=ap.parse_args()
    if a.zip is None:
        server_data=Path(__file__).resolve().parent.parent/"fd_datasets"
        server_own=server_data/"30Hz_processed_clean_v2"
        if server_own.is_dir(): a.zip=server_own
        elif server_own.with_suffix(".zip").is_file(): a.zip=server_own.with_suffix(".zip")
    if a.zip is None:
        kaggle_input=Path("/kaggle/input")
        candidates=[]
        if kaggle_input.exists():
            for p in kaggle_input.rglob("*"):
                if p.is_dir() and (p/"fall").is_dir() and (p/"non-fall").is_dir():
                    candidates.append(p)
        if not candidates:
            ap.error("--zip is required unless a Kaggle input directory containing fall/ and non-fall/ is mounted")
        a.zip=sorted(candidates,key=lambda p: len(str(p)))[0]
        a.work=Path("/kaggle/working/fall_detection_30hz")
        print("Kaggle dataset root:",a.zip,flush=True)
    cfg=Cfg(seed=a.seed,channels=a.channels,ssl_epochs=a.ssl_epochs,head_epochs=a.head_epochs,all_epochs=a.all_epochs,batch=a.batch,
            target_fraction=a.target_fraction,steps_per_epoch=a.steps_per_epoch)
    if a.smoke:
        cfg.channels=min(cfg.channels,8); cfg.ssl_epochs=1; cfg.head_epochs=1; cfg.all_epochs=1; cfg.max_ssl_per_rec=4; cfg.max_neg_per_rec=4
        cfg.steps_per_epoch=2; cfg.batch=min(cfg.batch,16)
    from training_v2 import run
    import sys
    return run(a,cfg,sys.modules[__name__])

if __name__=="__main__": main()
