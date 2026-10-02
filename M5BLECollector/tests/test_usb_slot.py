import hashlib,importlib.util,struct,unittest,zlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('usb_slot',Path(__file__).resolve().parents[1]/'tools/usb_app_slot.py');usb=importlib.util.module_from_spec(spec);spec.loader.exec_module(usb)
class USBSlotTests(unittest.TestCase):
 def fixture(self,active):
  b=bytearray(b'\xff'*0x800000);pos=0x8000
  for name,(offset,size) in usb.PARTITIONS.items():
   struct.pack_into('<HBBII16sI',b,pos,0x50aa,1,0,offset,size,name.encode(),0);pos+=32
  b[pos:pos+16]=bytes.fromhex('ebeb'+'ff'*14);b[pos+16:pos+32]=hashlib.md5(b[0x8000:pos]).digest()
  seq=3+active;struct.pack_into('<I',b,0xd000,seq);struct.pack_into('<II',b,0xd000+24,2,zlib.crc32(b[0xd000:0xd004],0xffffffff));return b
 def test_both_active_slots_preserve_current_selector(self):
  for active in (0,1):
   b=self.fixture(active);info,selector=usb.inspect(b);self.assertEqual(info['active_slot'],active);self.assertEqual(info['inactive_slot'],1-active);self.assertEqual(info['selector_offset'],0xe000);self.assertEqual((info['new_sequence']-1)%2,1-active);self.assertEqual(struct.unpack_from('<I',selector,28)[0],zlib.crc32(selector[:4],0xffffffff))
 def test_refuses_bad_crc_layout_and_incomplete_backup(self):
  b=self.fixture(0);b[0xd000]^=1
  with self.assertRaises(ValueError):usb.inspect(b)
  b=self.fixture(0);b[0x8000+4]^=1
  with self.assertRaises(ValueError):usb.inspect(b)
  with self.assertRaises(ValueError):usb.inspect(b[:0x400000])

 def nvs(self,page_offset,mode=0):
  b=bytearray(b'\xff'*0x10000);page=page_offset
  struct.pack_into('<II',b,page,0xfffffffe,1);b[page+32]=0xfa
  for i,(ns,key,value) in enumerate([(0,'m5_mode',8),(8,'mode',mode)]):
   entry=bytearray(b'\xff'*32);entry[:4]=bytes([ns,1,1,255]);entry[8:24]=key.encode().ljust(16,b'\0');entry[24]=value
   struct.pack_into('<I',entry,4,zlib.crc32(entry[:4]+entry[8:],0xffffffff));b[page+64+i*32:page+96+i*32]=entry
  return b
 def test_nvs_page_relocation_preserves_settings(self):
  self.assertTrue(usb.protected_equal(self.nvs(0x9000),self.nvs(0xa000)))
  self.assertFalse(usb.protected_equal(self.nvs(0x9000),self.nvs(0xa000,1)))
 def test_protected_boot_change_is_rejected(self):
  a=self.nvs(0x9000);b=self.nvs(0x9000);b[0x8000]^=1
  self.assertFalse(usb.protected_equal(a,b))
