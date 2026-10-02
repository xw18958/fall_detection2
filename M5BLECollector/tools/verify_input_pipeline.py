#!/usr/bin/env python3
"""Check all signed sensor counts in each channel against independent float32 SI formulas."""
import argparse,json,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cxx',default='clang++');a=p.parse_args()
 config=(ROOT/'firmware/main/model_v2_config.h').read_text()
 import re
 def array(name):return np.asarray([float(x.rstrip('f')) for x in re.search(name+r'\[6\]=\{([^}]+)',config)[1].split(',')],np.float32)
 def value(name):return float(re.search(name+r'=([^;,]+)',config)[1].rstrip('f'))
 with tempfile.TemporaryDirectory() as d:
  path=Path(d);src=path/'verify.cc';exe=path/'verify';out=path/'bytes.bin'
  src.write_text('''#include <fstream>\n#include "input_pipeline.h"\nint main(int argc,char** argv){
 if(fall_v2::QuantizeInput(-.5f,1,0)!=-1 || fall_v2::QuantizeInput(.5f,1,0)!=1)return 2;
 std::ofstream out(argv[1],std::ios::binary);
 for(int n=-32768;n<=32767;++n)for(int c=0;c<6;++c){
  int8_t q=fall_v2::QuantizeInput(fall_v2::NormalizeM5(c,fall_v2::PhysicalValue(c,n)),fall_v2::kInputScale,fall_v2::kInputZero);
  out.write(reinterpret_cast<char*>(&q),1);
 }return 0;}''')
  subprocess.run([a.cxx,'-std=c++17','-O2','-I'+str(ROOT/'firmware/main'),str(src),'-o',str(exe)],check=True)
  subprocess.run([str(exe),str(out)],check=True)
  counts=np.repeat(np.arange(-32768,32768,dtype=np.float32)[:,None],6,axis=1)
  physical=counts.copy();physical[:,:3]=(counts[:,:3]*np.float32(8/32768))*np.float32(9.80665)
  physical[:,3:]=(counts[:,3:]*np.float32(2000/32768))*np.float32(np.pi/180)
  normalized=(physical-array('kMean'))/array('kSigma');z=normalized/np.float32(value('kInputScale'))
  expected=np.clip(np.copysign(np.floor(np.abs(z).astype(np.float64)+.5),z)+int(value('kInputZero')),-128,127).astype(np.int8)
  actual=np.fromfile(out,dtype=np.int8).reshape(65536,6)
  mismatch=int((actual!=expected).sum())
  print(json.dumps({'signed_count_channel_cases':actual.size,'byte_mismatches':mismatch,'half_ties':'away from zero','device_measurement':False}))
  if mismatch:raise AssertionError('Host/device preprocessing bytes differ')
if __name__=='__main__':main()
