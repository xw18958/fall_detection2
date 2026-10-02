"""Enforce fall-window recall floors and float32 threshold margins before test."""
import argparse,json
from pathlib import Path
import numpy as np
import export_v4_tflite as e
import training_v4 as v4
p=argparse.ArgumentParser(description=__doc__)
for name in ['checkpoint','work','own-root','public-root','m5-root','output','candidates','ptq-export']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c,cfg,parts,model=e.load(a)
gates=json.loads((a.work/'baseline_validation.json').read_text())['constraints']
reference=json.loads((a.ptq_export/'validation_report.json').read_text())
floors={d:z['float']['recall'] for d,z in reference['domains'].items() if d!='pooled' and z['float']['recall'] is not None}
history=json.loads((a.candidates/'history.json').read_text());reports=[];best=None
for row in history:
 folder=a.candidates/f'epoch_{row["epoch"]:02d}'/'validation';details={}
 for d in c['source_weights']:
  z=np.load(folder/f'{d}.predictions.npz');probs=e.probabilities(z['logits'])
  rows=[dict(key=k,file_label=int(y),prob=float(p),start=int(s)) for k,y,p,s in zip(z['keys'],z['labels'],probs,z['starts'])]
  kk,yy,pp=e.base.aggregate(rows);details[d]=(rows,kk,yy,pp)
 selection=v4.select_threshold(details,parts['val'],c['source_weights'],gates,
                              window_recall_floors=floors,deployment_thresholds=True)
 row={**row,'selection':selection}
 row['window_metrics']={d:e.metrics(z['labels'],e.probabilities(z['logits']),selection['threshold'])
                        for d in c['source_weights'] for z in [np.load(folder/f'{d}.predictions.npz')]}
 rank=(int(selection['valid_under_constraints']),-selection['weighted_error'],selection['domains']['own']['recall'])
 if best is None or rank>best[0]:best=(rank,row)
 reports.append(row);print('WINDOW_GATED',row['epoch'],json.dumps(selection),flush=True)
e.v2.write_json(a.output/'history.json',reports);e.v2.write_json(a.output/'selected.json',best[1])
e.v2.write_json(a.output/'completed.json',{'test_used':False,'selected':best[1]})
