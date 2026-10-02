#!/usr/bin/env python3
import json,numpy as np
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Compare every native Micro output to locked float/desktop predictions; no tuning')
parser.add_argument('--source',type=Path,required=True);parser.add_argument('--reports',type=Path,required=True);parser.add_argument('--output-suffix',default='_espnn_outputs.bin');args=parser.parse_args()
root=args.source;args.reports.mkdir(parents=True,exist_ok=True); exp=json.loads((root/'export_lock.json').read_text());scale,zero=exp['inventory']['output_scale_zero'];threshold=exp['threshold']
def probs(z):
 z=z-z.max(1,keepdims=True);p=np.exp(z);return p[:,1]/p.sum(1)
def metrics(y,p):
 pred=p>=threshold; tp=int((pred&(y==1)).sum());fn=int((~pred&(y==1)).sum());fp=int((pred&(y==0)).sum());tn=int((~pred&(y==0)).sum());rat=lambda a,b:a/b if b else None
 return {'windows':len(y),'positive_windows':tp+fn,'negative_windows':fp+tn,'tp':tp,'fn':fn,'fp':fp,'tn':tn,'precision':rat(tp,tp+fp) if tp+fn else None,'recall':rat(tp,tp+fn),'f1':rat(2*tp,2*tp+fn+fp) if tp+fn else None,'specificity':rat(tn,tn+fp),'negative_window_fpr':rat(fp,fp+tn),'accuracy':rat(tp+tn,len(y))}
for role in ['validation','test']:
 micro=np.fromfile(root/f'{role}{args.output_suffix}',np.int8).reshape(-1,2);offset=0;table={}; ys=[];fs=[];ms=[];qs=[]
 for f in sorted(root.glob(f'{role}_*_outputs.npz')):
  d=f.stem[len(role)+1:-len('_outputs')];z=np.load(f);y=z['labels'];n=len(y);raw=micro[offset:offset+n];offset+=n
  m=(raw.astype(np.float32)-zero)*scale;pf=probs(z['float_logits']);pm=probs(m);pq=probs(z['int8_logits']);error=abs(pm-pq)
  table[d]={'float':metrics(y,pf),'desktop_int8':metrics(y,pq),'micro_reference':metrics(y,pm),'desktop_micro_decision_disagreements':int(((pm>=threshold)!=(pq>=threshold)).sum()),'desktop_micro_probability_error':{'mean':float(error.mean()),'max':float(error.max()),'p99':float(np.quantile(error,.99))},'desktop_micro_max_output_lsb_error':int(abs(raw.astype(np.int32)-z['int8_raw']).max()),'float_micro_probability_error':{'mean':float(abs(pm-pf).mean()),'p99':float(np.quantile(abs(pm-pf),.99)),'max':float(abs(pm-pf).max())},'float_micro_logit_error':{'mean':float(abs(m-z['float_logits']).mean()),'p99':float(np.quantile(abs(m-z['float_logits']),.99)),'max':float(abs(m-z['float_logits']).max())}}
  ys.extend(y);fs.extend(pf);ms.extend(pm);qs.extend(pq)
 assert offset==len(micro)
 table['pooled']={'float':metrics(np.asarray(ys),np.asarray(fs)),'desktop_int8':metrics(np.asarray(ys),np.asarray(qs)),'micro_reference':metrics(np.asarray(ys),np.asarray(ms))}
 (args.reports/f'{role}_runtime_comparison.json').write_text(json.dumps({'threshold':threshold,'unit':'3-second windows, overlapping 0.25-second grid','domains':table},indent=2)+'\n')
 print(role,'MICRO_COMPARISON_PASS',len(micro))
