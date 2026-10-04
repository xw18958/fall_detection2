"""Verify unchanged V5 computation, physical units, collector data and partitions."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def function(text,name):
    match=re.search(r'^(?:inline )?(?:bool|void|float) '+name+r'\([^\n]*\)\s*\{',text,re.M)
    if not match:raise ValueError('Missing '+name)
    i=text.index('{',match.start())+1;depth=1
    while depth:depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[match.start():i].encode()
def verify(binary=None):
    contract=json.loads((ROOT/'power_contract.json').read_text());main=ROOT/'firmware/main';text=(main/'main.cc').read_text()
    for name,expected in contract['files'].items():
        if sha((main/name).read_bytes())!=expected:raise ValueError('Frozen V5 file changed: '+name)
    for name,expected in contract['functions'].items():
        if sha(function(text,name))!=expected:raise ValueError('Frozen V5 computation changed: '+name)
    if sha((ROOT/'firmware/partitions.csv').read_bytes())!=contract['partitions_sha256']:raise ValueError('Partition changes prohibited')
    if 'constexpr float kPrototypeFallThreshold = fall_v2::kThreshold;' not in text:raise ValueError('Decision threshold changed')
    if binary:
        content=Path(binary).read_bytes();model=(main/'model.tflite').read_bytes()
        if content.count(model)!=1 or len(content)>0x3D0000:raise ValueError('Wrong embedded model or oversized OTA image')
    print('PASS: V5 model, preprocessing, threshold, recordings and partitions preserved')
if __name__=='__main__':verify(sys.argv[1] if len(sys.argv)>1 else None)
