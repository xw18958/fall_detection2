"""Assemble a complete snapshot from the original read and verified app writes.

This is explicitly not a fresh full-flash read. Only use after the recorded
application-only stage: its exact image readback, fresh first/last app pages,
and fresh protected prefix cover every byte changed since the original backup.
"""
import argparse,json
from pathlib import Path
from usb_app_slot import inspect,sha,protected_equal,nvs_entries

def saved_detect_transition_equal(before, after):
    """Allow only the explicitly commanded COLLECT(1)->DETECT(0) scalar."""
    if len(before)!=0x10000 or len(after)!=0x10000: return False
    if before[:0x9000]!=after[:0x9000] or before[0xd000:]!=after[0xd000:]: return False
    old=nvs_entries(before[0x9000:0xd000]);new=nvs_entries(after[0x9000:0xd000])
    key='m5ble/mode/01/ff'
    if old.get(key)!=sha(b'\x01'+b'\xff'*7) or new.get(key)!=sha(b'\x00'+b'\xff'*7): return False
    stable=lambda entries:{k:v for k,v in entries.items() if k!=key and not k.startswith('phy/cal_data/')}
    return bool(old) and stable(old)==stable(new)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['original','stage-result','stage-log','app-readback','protected','header','last-page','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--saved-detect-mode-transition',action='store_true',
                   help='Explicitly account for confirmed saved COLLECT to DETECT command; preserve every other setting/bond')
    a=p.parse_args();original=a.original.read_bytes();inspect(original)
    stage=json.loads(a.stage_result.read_text());image=a.app_readback.read_bytes()
    if not stage['passed'] or stage['flash_sha256']!=sha(original) or stage['firmware_sha256']!=sha(image):
        raise ValueError('Missing exact original backup/stage/readback proof')
    offset=stage['inactive_offset'];last=offset+(len(image)-1)//4096*4096;end=last+4096
    expected=f'Flash will be erased from 0x{offset:08x} to 0x{end-1:08x}...'
    if expected not in a.stage_log.read_text():raise ValueError('Unknown flash erase extent')
    protected=a.protected.read_bytes();header=a.header.read_bytes();tail=a.last_page.read_bytes()
    expected_protected=bytearray(original[:0x10000]);expected_protected[0xd000:0xf000]=protected[0xd000:0xf000]
    mode_transition=False
    if not protected_equal(bytes(expected_protected),protected):
        if not a.saved_detect_mode_transition or not saved_detect_transition_equal(bytes(expected_protected),protected):
            raise ValueError('Original protected settings changed')
        mode_transition=True
    if len(header)!=4096 or header!=image[:4096] or len(tail)!=4096:
        raise ValueError('Current app header/page proof differs')
    if tail[:len(image)-(last-offset)]!=image[last-offset:]:raise ValueError('Current app final bytes differ')
    current=bytearray(original);current[:0x10000]=protected;current[offset:offset+len(image)]=image
    current[last:end]=tail;info,_=inspect(current)
    if info['active_slot']!=stage['inactive_slot']:raise ValueError('Staged image is not the current active app')
    a.output.write_bytes(current)
    manifest=dict(method='Original complete hardware read + verified app readback + fresh protected prefix and app edge pages',
                  warning='Assembled snapshot; not a fresh complete hardware read',original_sha256=sha(original),
                  snapshot_sha256=sha(current),active_slot=info['active_slot'],stage_firmware_sha256=sha(image),
                  fresh_protected_sha256=sha(protected),fresh_header_sha256=sha(header),fresh_last_page_sha256=sha(tail),
                  changed_application_extent=[offset,end],saved_collect_to_detect_transition=mode_transition,passed=True)
    a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
