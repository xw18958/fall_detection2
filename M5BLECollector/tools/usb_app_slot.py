"""Narrow USB app-only recovery: verified backup, inactive slot, readback, selector.

Never writes the bootloader, partitions, NVS, PHY, or the active application.
Generic upload/erase targets remain prohibited. Run stage and activate separately.
"""
import argparse,hashlib,json,struct,subprocess,zlib
from pathlib import Path
PARTITIONS={'nvs':(0x9000,0x4000),'otadata':(0xd000,0x2000),'phy_init':(0xf000,0x1000),'ota_0':(0x10000,0x3d0000),'ota_1':(0x3e0000,0x3d0000)}
def sha(data):return hashlib.sha256(data).hexdigest()
def inspect(data):
 if len(data)!=0x800000:raise ValueError('Need a complete 8 MB preinstallation backup')
 table={}
 for pos in range(0x8000,0x9000,32):
  magic,typ,sub,offset,size,label,flags=struct.unpack_from('<HBBII16sI',data,pos)
  if magic==0xffff:break
  if magic==0xebeb:
   if data[pos+16:pos+32]!=hashlib.md5(data[0x8000:pos]).digest():raise ValueError('Partition MD5 mismatch')
   break
  if magic!=0x50aa:raise ValueError('Partition magic mismatch')
  table[label.split(b'\0')[0].decode()]=(offset,size)
 if table!=PARTITIONS:raise ValueError('Unexpected device partition layout')
 entries=[]
 for sector in range(2):
  offset=0xd000+sector*0x1000;seq=struct.unpack_from('<I',data,offset)[0];state,crc=struct.unpack_from('<II',data,offset+24)
  valid=seq!=0xffffffff and seq!=0 and state not in (3,4) and crc==zlib.crc32(data[offset:offset+4],0xffffffff)
  entries.append(dict(sector=sector,sequence=seq,state=state,crc=crc,valid=valid))
 valid=[e for e in entries if e['valid']]
 if not valid:raise ValueError('No valid OTA selection; refuse to guess active slot')
 current=max(valid,key=lambda e:e['sequence']);active=(current['sequence']-1)%2;inactive=1-active
 seq=current['sequence']+1
 while (seq-1)%2!=inactive:seq+=1
 if seq>=0xfffffffe:raise ValueError('OTA sequence wraparound requires manual review')
 sector=1-current['sector'];selector=bytearray(b'\xff'*4096)
 struct.pack_into('<I',selector,0,seq);struct.pack_into('<II',selector,24,0,zlib.crc32(selector[:4],0xffffffff))
 return dict(flash_sha256=sha(data),partitions=table,entries=entries,active_slot=active,inactive_slot=inactive,
   inactive_offset=PARTITIONS[f'ota_{inactive}'][0],selector_offset=0xd000+sector*0x1000,new_sequence=seq),bytes(selector)
def nvs_entries(data):
 found={};names={}
 pages=sorted([data[p:p+4096] for p in range(0,len(data),4096) if struct.unpack_from('<I',data,p)[0] in (0xfffffffe,0xfffffffc)],key=lambda b:struct.unpack_from('<I',b,4)[0])
 for page in pages:
  for i in range(126):
   state=(page[32+i//4] >> ((i%4)*2)) & 3
   if state!=2:continue
   entry=page[64+i*32:96+i*32];ns,typ,span,chunk=entry[:4]
   if ns==255:continue
   if struct.unpack_from('<I',entry,4)[0] != zlib.crc32(entry[:4]+entry[8:],0xffffffff):continue
   if typ not in [1,2,4,8,0x11,0x12,0x14,0x18,0x21,0x41,0x42,0x48]:continue
   try:key=entry[8:24].split(b'\0')[0].decode('ascii')
   except UnicodeDecodeError:continue
   if not key:continue
   if ns==0 and typ==1:names[entry[24]]=key
   if typ in [0x21,0x41,0x42]:
    length=struct.unpack_from('<H',entry,24)[0];payload=page[96+i*32:96+i*32+length]
   else:payload=entry[24:32]
   found[(ns,key,typ,chunk)]=hashlib.sha256(payload).hexdigest()
 return {f'{names.get(ns,ns)}/{key}/{typ:02x}/{chunk:02x}':v for (ns,key,typ,chunk),v in found.items()}
def protected_equal(before,after):
 if len(before)!=0x10000 or len(after)!=0x10000:return False
 if before==after:return True
 if before[:0x9000]!=after[:0x9000] or before[0xd000:]!=after[0xd000:]:return False
 old=nvs_entries(before[0x9000:0xd000]);new=nvs_entries(after[0x9000:0xd000])
 # Old app startup can relocate NVS pages and regenerate PHY RF calibration.
 # Preserve every logical setting, namespace, bond and mode entry exactly.
 stable=lambda z:{k:v for k,v in z.items() if not k.startswith('phy/cal_data/')}
 return bool(old) and stable(old)==stable(new)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--backup',type=Path,required=True);p.add_argument('--firmware',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--python',required=True);p.add_argument('--esptool',required=True);p.add_argument('--port',required=True);p.add_argument('--expected-mac',required=True)
 p.add_argument('--phase',choices=['inspect','stage','verify','activate'],default='inspect');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);backup=a.backup.read_bytes();info,selector=inspect(backup);firmware=a.firmware.read_bytes()
 if not 0<len(firmware)<=0x3d0000 or firmware[0]!=0xe9:raise ValueError('Invalid ESP app image')
 info['firmware_sha256']=sha(firmware);info['firmware_bytes']=len(firmware);(a.output/'usb_plan.json').write_text(json.dumps(info,indent=2)+'\n');(a.output/'new_selector.bin').write_bytes(selector)
 if a.phase=='inspect':print(json.dumps(info,indent=2));return
 base=[a.python,a.esptool,'--chip','esp32','--port',a.port,'--baud','115200','--after','no_reset']
 def call(*args):
  r=subprocess.run(base+list(map(str,args)),capture_output=True,text=True)
  with (a.output/f'{a.phase}.log').open('a') as f:f.write(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(r.stdout+r.stderr)
  # Every reset checks the physical chip identity before a subsequent mutation.
  if 'MAC: '+a.expected_mac.lower() not in r.stdout:raise ValueError('Device identity differs')
  return r.stdout
 current=a.output/'otadata_live.bin';call('read_flash',hex(0xd000),hex(0x2000),current)
 if current.read_bytes()!=backup[0xd000:0xf000]:raise ValueError('OTA state changed since backup')
 if a.phase=='stage':call('write_flash',hex(info['inactive_offset']),a.firmware)
 readback=a.output/'application_readback.bin'
 if a.phase=='stage':
  call('read_flash',hex(info['inactive_offset']),hex(len(firmware)),readback)
 else:
  # Require the immediately preceding full stage/readback proof and inspect
  # the slot header again. No intervening flash mutation is part of this flow.
  if a.phase=='activate':
   stage=json.loads((a.output/'stage_result.json').read_text())
   if not stage['passed'] or stage['firmware_sha256']!=sha(firmware) or stage['flash_sha256']!=sha(backup):raise ValueError('Missing matching stage proof')
  header=a.output/'application_header_live.bin';call('read_flash',hex(info['inactive_offset']),hex(256),header)
  if header.read_bytes()!=firmware[:256]:raise ValueError('Staged application header changed')
 if readback.read_bytes()!=firmware:raise ValueError('Application readback mismatch; selector unchanged')
 protected=a.output/'protected_readback.bin';call('read_flash',0,hex(0x10000),protected)
 if not protected_equal(backup[:0x10000],protected.read_bytes()):raise ValueError('Protected flash/settings differ before activation')
 if a.phase=='activate':
  call('write_flash',hex(info['selector_offset']),a.output/'new_selector.bin')
  sel=a.output/'selector_readback.bin';call('read_flash',hex(info['selector_offset']),hex(0x1000),sel)
  if sel.read_bytes()!=selector:raise ValueError('Selector readback mismatch')
 result=dict(phase=a.phase,passed=True,**info);(a.output/('stage_result.json' if a.phase=='verify' else f'{a.phase}_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
