#!/usr/bin/env python3
"""Lock startup replay against the exact Micro/ESP-NN path, retaining desktop deltas."""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);a=p.parse_args()
 model=a.package/'model_int8.tflite';lock=json.loads((a.package/'export_lock.json').read_text())
 if hashlib.sha256(model.read_bytes()).hexdigest()!=lock['model_sha256']: raise ValueError('Model identity mismatch')
 header=a.package/'model_v2_replay.h';text=header.read_text(); body=text.split('constexpr int8_t kReplayInputs[][540] = {\n')[1].split('\n};')[0]
 inputs=b''.join(struct.pack('540b',*map(int,row.strip('{}, ').split(','))) for row in body.splitlines())
 inp=a.package/'replay_all.bin';out=a.package/'replay_micro.bin'; inp.write_bytes(inputs)
 result=subprocess.run([str(a.runner.resolve()),str(model.resolve()),str(inp.resolve()),str(out.resolve())],check=True,capture_output=True,text=True)
 values=list(struct.iter_unpack('bb',out.read_bytes()))
 if len(values)*540!=len(inputs): raise ValueError('Replay output count mismatch')
 replacement='constexpr int8_t kReplayOutputs[][2] = {\n'+',\n'.join('{'+','.join(map(str,z))+'}' for z in values)+'\n};'
 text,n=re.subn(r'constexpr int8_t kReplayOutputs\[\]\[2\] = \{.*?\n\};',replacement,text,flags=re.S)
 if n!=1: raise ValueError('Missing replay outputs')
 header.write_text(text);manifest=json.loads((a.package/'replay_manifest.json').read_text())
 manifest['desktop_outputs']=manifest.get('desktop_outputs',manifest['outputs']); manifest['outputs']=values;manifest['native_reference_used']=True
 manifest['micro_runner_stdout']=result.stdout;manifest['micro_runner_sha256']=hashlib.sha256(a.runner.read_bytes()).hexdigest()
 manifest['desktop_micro_max_lsb_error']=max(abs(x-y) for pair,ref in zip(values,manifest['desktop_outputs']) for x,y in zip(pair,ref))
 (a.package/'replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(result.stdout.strip()); print('MICRO_REPLAY_LOCKED; desktop differences retained; firmware tolerance unchanged')
if __name__=='__main__':main()
