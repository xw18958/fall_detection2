"""Create an unqualified private bundle for the validation-locked winner."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path);a=p.parse_args()
    lock_path=(a.calibration or a.experiment)/'selection_lock.json'
    lock=json.loads(lock_path.read_text())
    review=(dict(winner=lock['winner'],adjusted_threshold=lock['candidates'][lock['winner']]['selection']['threshold'])
            if a.calibration else json.loads((a.experiment/'margin_review.json').read_text()))
    name=lock['winner']
    if name!=review['winner']:raise ValueError('Winner changed')
    folder=a.experiment/name;model=(folder/'trigger_int8.tflite').read_bytes()
    sha=lambda b:hashlib.sha256(b).hexdigest()
    if sha(model)!=lock['candidates'][name]['metadata']['sha256']:raise ValueError('Model changed')
    x=np.fromfile(folder/'validation_micro/input.bin',np.int8).reshape(-1,540)
    y=np.fromfile(folder/'validation_micro/output.bin',np.int8).reshape(-1,2)
    if len(x)!=len(y):raise ValueError('Replay inventory mismatch')
    indices=np.linspace(0,len(x)-1,4).round().astype(int)
    fmt=lambda z:','.join(map(str,np.asarray(z).ravel().tolist()))
    text='#pragma once\n#include <cstdint>\nnamespace trigger_bundle {\n'
    text+='constexpr int kSampleHz=30,kSamples=90,kReplayCount=4;\nconstexpr bool kQualified=false;\n'
    text+=f'constexpr float kThreshold={review["adjusted_threshold"]:.12e}f;\n'
    text+=f'constexpr char kFullModelSha256[]="{lock["full_model_sha256"]}";\n'
    text+='alignas(16) constexpr unsigned char kModel[]={'+fmt(np.frombuffer(model,np.uint8))+'};\n'
    for label,size,array in [('kReplayInputs',540,x[indices]),('kReplayOutputs',2,y[indices])]:
        text+=f'constexpr int8_t {label}[][{size}]={{'+','.join('{'+fmt(row)+'}' for row in array)+'};\n'
    text+='}\n';a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'trigger_bundle.h').write_text(text)
    manifest=dict(candidate=name,full_model_sha256=lock['full_model_sha256'],
        firmware_config_sha256=lock['firmware_config_sha256'],header_sha256=sha(text.encode()),
        model_sha256=sha(model),sample_hz=30,input_shape=[1,90,6],deployment_eligible=False,
        commercial_qualified=False,threshold=review['adjusted_threshold'],
        selection_lock_sha256=sha(lock_path.read_bytes()),
        purpose='Hardware shadow benchmark only; frozen full model remains decision maker',
        replay_indices=indices.tolist(),previous_test_seen=True)
    (a.output/'trigger_bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('UNQUALIFIED_SHADOW_BUNDLE_READY',name)

if __name__=='__main__':main()
