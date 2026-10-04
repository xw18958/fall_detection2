import sys,hashlib,json
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--export',type=Path,required=True)
parser.add_argument('--native-replay-outputs',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'FallDetection30HzTraining'))
import export_v4_tflite as e
import numpy as np
import tensorflow as tf
out=args.export
assert e.sha(args.checkpoint)==e.EXPECTED_CHECKPOINT
c=e.torch.load(args.checkpoint,weights_only=False,map_location='cpu')
r=json.loads((out/'export_lock.json').read_text()); scale,zero=r['inventory']['input_scale_zero']; oscale,ozero=r['inventory']['output_scale_zero']
mean,std=(np.asarray(c['normalization']['m5_hard_negatives'][k],np.float32) for k in ('mean','std'))
def normalized(counts):
    z=counts.astype(np.float32).copy()
    z[:,:3]=(z[:,:3]*np.float32(8/32768))*np.float32(9.80665)
    z[:,3:]=(z[:,3:]*np.float32(2000/32768))*np.float32(np.pi/180)
    return (z-mean)/std
cal=np.load(out/'calibration.npy'); it=e.interpreter(tf,(out/'model_int8.tflite').read_bytes())
# Replay examples derived only from calibration and deterministic synthetic counts.
logits,_=e.lite(it,cal); p=e.probabilities(logits)
idx=np.argsort(abs(p-r['threshold']))[:4]
raw=np.zeros((3,90,6),np.int16)
raw[1,:,:]=np.array([-32768,32767,-32768,32767,-32768,32767],np.int16)
raw[2,:,:]=np.where(np.arange(90)[:,None]%2,32767,-32768)
x=np.concatenate([cal[idx],*[normalized(z)[None] for z in raw]])
q=e.quantize(x,scale,zero); logits,outputs=e.lite(it,x)
# Joint validation replay coverage is diagnostic only; never calibrates or tunes.
val_cases=[]
for f in sorted(out.glob('validation_*_outputs.npz')):
    z=np.load(f); probabilities=e.probabilities(z['int8_logits'])
    inputs=np.load(out/(f.stem.replace('_outputs','_inputs')+'.npy'))
    for i in np.argsort(abs(probabilities-r['threshold']))[:4]:
        val_cases.append((abs(float(probabilities[i])-r['threshold']),inputs[i],z['int8_raw'][i],str(f.name),int(i)))
val_cases=sorted(val_cases,key=lambda z:z[0])[:4]
if len(val_cases)!=4: raise ValueError('Joint validation must be evaluated before packaging')
q=np.concatenate([q,np.asarray([z[1] for z in val_cases])])
outputs=np.concatenate([outputs,np.asarray([z[2] for z in val_cases])])
logits=(outputs.astype(np.float32)-ozero)*oscale
np.save(out/'replay_inputs.npy',q)
np.save(out/'replay_raw_counts.npy',raw)
pp=e.probabilities(logits)
desktop_outputs=outputs.copy()
if args.native_replay_outputs:
    outputs=np.fromfile(args.native_replay_outputs,dtype=np.int8).reshape(len(q),2)
header='#pragma once\n#include <cstdint>\nnamespace fall_v2 {\n'
header+=f'constexpr int kReplayCount = {len(q)};\n'
header+='constexpr int8_t kReplayInputs[][540] = {\n'+',\n'.join('{'+','.join(map(str,z.ravel()))+'}' for z in q)+'\n};\n'
header+='constexpr int8_t kReplayOutputs[][2] = {\n'+',\n'.join('{'+','.join(map(str,z))+'}' for z in outputs)+'\n};\n'
counts=np.array([[0]*6,[-32768,32767,-32768,32767,-32768,32767],[32767,-32768,32767,-32768,32767,-32768],[1,-1,1,-1,1,-1],[12345,-12345,29999,-29999,10000,-10000]],np.int16)
expected=e.quantize(normalized(counts),scale,zero)
header+=f'constexpr int kPreprocessCount = {len(counts)};\n'
for typ,name,z in [('int16_t','kPreprocessCounts',counts),('int8_t','kPreprocessExpected',expected)]:
    header+=f'constexpr {typ} {name}[][6] = {{\n'+',\n'.join('{'+','.join(map(str,row))+'}' for row in z)+'\n};\n'
header+='}\n'; (out/'model_v2_replay.h').write_text(header)
config='#pragma once\n// Exact V4 checkpoint M5 SI normalization; unchanged float threshold.\nnamespace fall_v2 {\n'
config+='constexpr int kTimesteps=90, kChannels=6, kSampleHz=30;\n'
config+='// Retain 750 ms until physical latency and acquisition are measured.\nconstexpr long long kInferencePeriodUs=750000;\n'
for name,value in [('kThreshold',r['threshold']),('kGravity',9.80665),('kDegreesToRadians',np.pi/180),('kInputScale',scale),('kOutputScale',oscale)]:
    config+=f'constexpr float {name}={float(np.float32(value)):.12e}f;\n'
config+=f'constexpr int kInputZero={zero}, kOutputZero={ozero};\n'
for name,z in [('kMean',mean),('kSigma',std)]:
    config+=f'constexpr float {name}[6]={{'+','.join(f'{float(v):.12e}f' for v in z)+'};\n'
config+=f'constexpr char kCheckpointSha256[]="{r["checkpoint_sha256"]}";\n}}\n'
(out/'model_v2_config.h').write_text(config)
e.v2.write_json(out/'replay_manifest.json',{'case_policy':'4 calibration examples; zero, static extreme, alternating extreme signed counts; 4 joint-validation windows nearest threshold','calibration_indices':idx.tolist(),'validation_cases':[{'file':z[3],'index':z[4]} for z in val_cases],'probabilities':pp.tolist(),'outputs':outputs.tolist(),'desktop_outputs':desktop_outputs.tolist(),'native_reference_used':bool(args.native_replay_outputs),'synthetic_preprocessing_counts':counts.tolist(),'synthetic_preprocessing_expected':expected.tolist(),'model_sha256':r['model_sha256']})
print('PACKAGE_PASS',r['model_sha256'])
