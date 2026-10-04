"""Capture one hardware run; --reset requires saved/discarded recordings."""
import argparse,time
from pathlib import Path
import serial

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=150);p.add_argument('--reset',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    device=serial.Serial(port=None,baudrate=115200,timeout=.5)
    device.dtr=False;device.rts=False;device.port=a.port;device.open()
    try:
        if a.reset:
            device.rts=True;time.sleep(.1);device.rts=False
        end=time.monotonic()+a.seconds
        with a.output.open('wb') as stream:
            while time.monotonic()<end:
                line=device.readline()
                if line:
                    stream.write(line);stream.flush()
                    print(line.decode(errors='replace').rstrip(),flush=True)
    finally:device.close()

if __name__=='__main__':main()
