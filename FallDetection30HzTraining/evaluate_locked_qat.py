"""Lock a validation-selected QAT candidate, then evaluate held-out windows once."""
import argparse,json,shutil,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import qat_v4 as q
import export_v4_tflite as e
p=argparse.ArgumentParser(description=__doc__)
for name in ['checkpoint','work','own-root','public-root','m5-root','output','run','ptq-export','runner']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c,cfg,parts,model=e.load(a)
selected=json.loads((a.run/'selected.json').read_text())
if not selected['selection']['valid_under_constraints']:raise ValueError('Refuse to lock an invalid candidate')
folder=a.run/f"epoch_{selected['epoch']:02d}"
shutil.copy2(folder/'model_int8.tflite',a.output/'model_int8.tflite');shutil.copy2(folder/'qat_checkpoint.pt',a.output/'qat_checkpoint.pt')
inv=selected['inventory'];scale,zero=inv['input_scale_zero'];oscale,ozero=inv['output_scale_zero'];threshold=selected['selection']['threshold']
lock=dict(model_sha256=e.sha(a.output/'model_int8.tflite'),checkpoint_sha256=e.sha(a.output/'qat_checkpoint.pt'),original_checkpoint_sha256=e.EXPECTED_CHECKPOINT,
 threshold=threshold,original_threshold=c['threshold'],inventory=inv,selected_epoch=selected['epoch'],split_fingerprint=c['split_fingerprint'],selection=selected['selection'],test_used_for_selection=False)
e.v2.write_json(a.output/'export_lock.json',lock)
for name in ['calibration.npy','calibration_manifest.json']:shutil.copy2(a.ptq_export/name,a.output/name)

def bulk(inputs,name):
 jobs=[]
 for i in range(min(4,len(inputs))):
  lo,hi=len(inputs)*i//min(4,len(inputs)),len(inputs)*(i+1)//min(4,len(inputs));inp=a.output/f'{name}.{i}.inputs.bin';out=a.output/f'{name}.{i}.outputs.bin';inp.write_bytes(inputs[lo:hi].tobytes());jobs.append((inp,out,hi-lo))
 def run(z):
  inp,out,n=z;r=subprocess.run([str(a.runner.resolve()),str((a.output/'model_int8.tflite').resolve()),str(inp.resolve()),str(out.resolve())],check=True,capture_output=True,text=True)
  if f'TFLM_WINDOWS_PASS={n}' not in r.stdout:raise ValueError('Incomplete Micro replay')
  (out.with_suffix('.log')).write_text(r.stdout+r.stderr);return np.fromfile(out,np.int8).reshape(-1,2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:raw=np.concatenate(list(ex.map(run,jobs)))
 return raw
reports={}
for role in ['validation','test']:
 datasets=q.make_datasets(parts['val' if role=='validation' else 'test'],c,cfg,a.output/'cache')
 report={}
 for domain,(x,y,keys,starts) in datasets.items():
  reference=np.load(a.ptq_export/f'{role}_{domain}_outputs.npz')
  # Fingerprint, order and labels must match the immutable previous evaluation.
  for name,value in [('labels',y),('keys',keys),('starts',starts)]:
   if not np.array_equal(reference[name],value):raise ValueError('Reference order differs: '+domain+'/'+name)
  inputs=e.quantize(x,scale,zero)
  raw=bulk(inputs,f'{role}_{domain}');logits=(raw.astype(np.float32)-ozero)*oscale;prob=e.probabilities(logits)
  fp=e.probabilities(reference['float_logits']);report[domain]=dict(float=e.metrics(y,fp,c['threshold']),qat_original_threshold=e.metrics(y,prob,c['threshold']),qat=e.metrics(y,prob,threshold),
   logit_abs_mean=float(np.abs(logits-reference['float_logits']).mean()),logit_abs_max=float(np.abs(logits-reference['float_logits']).max()),probability_abs_mean=float(np.abs(prob-fp).mean()),probability_abs_max=float(np.abs(prob-fp).max()),
   input_clipped_values=int(np.count_nonzero((x/scale+zero < -128)|(x/scale+zero > 127))),input_values=int(x.size))
  np.savez_compressed(a.output/f'{role}_{domain}_outputs.npz',float_logits=reference['float_logits'],int8_logits=logits,int8_raw=raw,labels=y,keys=keys,starts=starts)
  np.save(a.output/f'{role}_{domain}_inputs.npy',inputs)
  print(role,domain,json.dumps(report[domain]['qat']),flush=True)
 reports[role]=report;e.v2.write_json(a.output/f'{role}_report.json',dict(domains=report,runtime='native Micro with ESP-NN FC/softmax and fused LayerNormV4',threshold=threshold,test_used_for_selection=False))
# Replay from training calibration and synthetic signed counts, plus four locked validation cases.
cal=np.load(a.output/'calibration.npy');calraw=bulk(e.quantize(cal,scale,zero),'calibration');calp=e.probabilities((calraw.astype(np.float32)-ozero)*oscale);indices=np.argsort(abs(calp-threshold))[:4]
mean,std=(np.asarray(c['normalization']['m5_hard_negatives'][k],np.float32) for k in ['mean','std'])
def norm(counts):
 z=counts.astype(np.float32).copy();z[:,:3]=(z[:,:3]*np.float32(8/32768))*np.float32(9.80665);z[:,3:]=(z[:,3:]*np.float32(2000/32768))*np.float32(np.pi/180);return (z-mean)/std
counts=np.zeros((3,90,6),np.int16);counts[1]=np.array([-32768,32767,-32768,32767,-32768,32767],np.int16);counts[2]=np.where(np.arange(90)[:,None]%2,32767,-32768)
inputs=np.concatenate([e.quantize(cal[indices],scale,zero),*[e.quantize(norm(z),scale,zero)[None] for z in counts]])
val=[]
for d in c['source_weights']:
 z=np.load(a.output/f'validation_{d}_outputs.npz');prob=e.probabilities(z['int8_logits']);inp=np.load(a.output/f'validation_{d}_inputs.npy')
 for i in np.argsort(abs(prob-threshold))[:4]:val.append((abs(float(prob[i])-threshold),inp[i],d,int(i)))
val=sorted(val,key=lambda z:z[0])[:4];inputs=np.concatenate([inputs,np.array([z[1] for z in val])]);outputs=bulk(inputs,'replay');np.save(a.output/'replay_inputs.npy',inputs);np.save(a.output/'replay_raw_counts.npy',counts)
header='#pragma once\n#include <cstdint>\nnamespace fall_v2 {\n'+f'constexpr int kReplayCount = {len(inputs)};\n'
for name,z in [('kReplayInputs',inputs.reshape(-1,540)),('kReplayOutputs',outputs)]:header+=f'constexpr int8_t {name}[][{z.shape[1]}] = {{\n'+',\n'.join('{'+','.join(map(str,row))+'}' for row in z)+'\n};\n'
gold=np.array([[0]*6,[-32768,32767,-32768,32767,-32768,32767],[32767,-32768,32767,-32768,32767,-32768],[1,-1,1,-1,1,-1],[12345,-12345,29999,-29999,10000,-10000]],np.int16);expected=e.quantize(norm(gold),scale,zero)
header+=f'constexpr int kPreprocessCount = {len(gold)};\n'
for typ,name,z in [('int16_t','kPreprocessCounts',gold),('int8_t','kPreprocessExpected',expected)]:header+=f'constexpr {typ} {name}[][6] = {{\n'+',\n'.join('{'+','.join(map(str,row))+'}' for row in z)+'\n};\n'
(a.output/'model_v2_replay.h').write_text(header+'}\n')
config='#pragma once\n// Locked QAT export; training-only normalization, joint-validation threshold.\nnamespace fall_v2 {\nconstexpr int kTimesteps=90, kChannels=6, kSampleHz=30;\nconstexpr long long kInferencePeriodUs=750000;\n'
for name,v in [('kThreshold',threshold),('kGravity',9.80665),('kDegreesToRadians',np.pi/180),('kInputScale',scale),('kOutputScale',oscale)]:config+=f'constexpr float {name}={float(np.float32(v)):.12e}f;\n'
config+=f'constexpr int kInputZero={zero}, kOutputZero={ozero};\n'
for name,z in [('kMean',mean),('kSigma',std)]:config+=f'constexpr float {name}[6]={{'+','.join(f'{float(v):.12e}f' for v in z)+'};\n'
config+=f'constexpr char kCheckpointSha256[]="{lock["checkpoint_sha256"]}";\n}}\n';(a.output/'model_v2_config.h').write_text(config)
shutil.copy2(a.output/'model_int8.tflite',a.output/'model.tflite')
e.v2.write_json(a.output/'replay_manifest.json',dict(model_sha256=lock['model_sha256'],native_reference_used=True,calibration_indices=indices.tolist(),validation_cases=[dict(domain=z[2],index=z[3]) for z in val],outputs=outputs.tolist(),synthetic_preprocessing_counts=gold.tolist(),synthetic_preprocessing_expected=expected.tolist()))
print('LOCKED_QAT_EVALUATION_COMPLETE',flush=True)
