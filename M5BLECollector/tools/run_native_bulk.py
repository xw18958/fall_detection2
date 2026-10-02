#!/usr/bin/env python3
"""Replay every saved validation/test INT8 input in deterministic CPU shards."""
import argparse,concurrent.futures,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
if a.jobs<1:raise ValueError('Positive job count required')
started=time.monotonic();jobs=[]
for role in ['validation','test']:
 arrays=[np.load(f) for f in sorted(a.source.glob(f'{role}_*_inputs.npy'))]
 if not arrays or any(z.dtype!=np.int8 or z.shape[1:]!=(90,6) for z in arrays):raise ValueError('Expected INT8 [N,90,6] inputs')
 data=b''.join(z.tobytes() for z in arrays);count=len(data)//540
 for i in range(a.jobs):
  lo=count*i//a.jobs;hi=count*(i+1)//a.jobs
  if hi==lo:continue
  inp=a.source/f'{role}_espnn_part{i}.bin';inp.write_bytes(data[lo*540:hi*540]);jobs.append((role,i,inp,hi-lo))
def run(job):
 role,i,inp,count=job;result=subprocess.run([str(a.runner.resolve()),str(a.model.resolve()),str(inp.resolve()),str((a.source/f'{role}_espnn_part{i}_outputs.bin').resolve())],check=True,capture_output=True,text=True)
 if f'TFLM_WINDOWS_PASS={count}' not in result.stdout:raise ValueError('Native replay count mismatch')
 return {'role':role,'part':i,'windows':count,'output':result.stdout}
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:results=list(pool.map(run,jobs))
for role in ['validation','test']:
 parts=sorted(r['part'] for r in results if r['role']==role)
 (a.source/f'{role}_espnn_outputs.bin').write_bytes(b''.join((a.source/f'{role}_espnn_part{i}_outputs.bin').read_bytes() for i in parts))
(a.source/'espnn_bulk_manifest.json').write_text(json.dumps({'host_elapsed_seconds':time.monotonic()-started,'device_timing':None,'parts':results},indent=2)+'\n')
print('ESP_NN_BULK_PASS',sum(r['windows'] for r in results))
