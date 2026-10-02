"""Re-export saved QAT checkpoints with corrected Conv/Bias fusion; validation only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import qat_v4 as q
import export_v4_tflite as e

p=argparse.ArgumentParser(description=__doc__)
for name in ['checkpoint','work','own-root','public-root','m5-root','output','run','ptq-export','runner']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c,cfg,parts,model=e.load(a)
datasets=q.make_datasets(parts['val'],c,cfg,a.run/'cache')
gates=json.loads((a.work/'baseline_validation.json').read_text())['constraints']
cal=np.load(a.ptq_export/'calibration.npy');ranges=json.loads((a.run/'ranges.json').read_text())
history=[];best=None
for folder in sorted(a.run.glob('epoch_*')):
 epoch=int(folder.name.split('_')[1]);out=a.output/folder.name;out.mkdir()
 ckpt=torch.load(folder/'qat_checkpoint.pt',map_location='cpu',weights_only=False)
 inv,_=q.tf_export(ckpt['model'],ranges,out/'model_int8.tflite',cal)
 result=q.micro_validate(out/'model_int8.tflite',inv,datasets,parts['val'],c,gates,a.runner,out/'validation')
 row={'epoch':epoch,'selection':result['selection'],'inventory':inv,'model_sha256':e.sha(out/'model_int8.tflite')}
 history.append(row);s=row['selection'];rank=(int(s['valid_under_constraints']),-s['weighted_error'],s['domains']['own']['recall'])
 if best is None or rank>best[0]:best=(rank,row)
 e.v2.write_json(a.output/'history.json',history);e.v2.write_json(a.output/'selected.json',best[1])
 print('CORRECTED_EXPORT',json.dumps(row),flush=True)
e.v2.write_json(a.output/'completed.json',{'selected':best[1],'test_used':False})
