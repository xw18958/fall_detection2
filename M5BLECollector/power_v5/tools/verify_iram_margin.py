"""Reject an unusably small ESP32 IRAM heap tail before hardware staging."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('elf',type=Path);p.add_argument('--nm',required=True)
    p.add_argument('--minimum',type=int,default=2048);a=p.parse_args()
    output=subprocess.check_output([a.nm,'-n',str(a.elf)],text=True)
    found=re.search(r'^([0-9a-f]+) A _iram_end$',output,re.M)
    if not found:raise ValueError('Missing ESP32 IRAM boundary')
    end=int(found[1],16);remaining=0x400a0000-end
    if remaining<a.minimum:raise ValueError(f'IRAM heap tail {remaining} bytes below {a.minimum}; refuse hardware staging')
    print(json.dumps(dict(elf_sha256=hashlib.sha256(a.elf.read_bytes()).hexdigest(),iram_end=hex(end),
                          iram_heap_tail_bytes=remaining,minimum_bytes=a.minimum,passed=True)))

if __name__=='__main__':main()
