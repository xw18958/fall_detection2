# AirPods Pro motion collector

Minimal macOS + Python pipeline for collecting AirPods motion data and streaming it to the USyd server.

## Current behavior

The Mac collector now starts automatically:

```text
launch app
→ detect whichever single AirPod is being worn
→ collect motion data
→ save a local CSV backup
→ stream batches to the server

AirPod removed
→ collection pauses

AirPod reinserted
→ Core Motion restarts automatically
→ collection resumes

server/tunnel interrupted
→ local recording continues
→ WebSocket reconnects automatically
```

Either the left or right AirPod can be used; the other bud may remain in the case. Each sample records `sensor_location` as `left`, `right`, or `unknown`.

## Remote receiver

On `gpu1-jinman-2`:

```bash
cd /raid1/xwan0900/fall_detection2
git pull
/raid1/xwan0900/venvs/ftkp_cu128/bin/python -m pip install -r requirements-server.txt
/raid1/xwan0900/venvs/ftkp_cu128/bin/python server.py
```

The server binds to `127.0.0.1:8765`.

## SSH tunnel from the Mac

```bash
ssh -N -L 8765:127.0.0.1:8765 \
  xwan0900@gpu1-jinman-2.it.usyd.edu.au
```

Leave the tunnel running while collecting.

## Mac collector

```bash
cd ~/Documents/fall_detection2
git pull
cd MacCollector
./Scripts/run-app.sh
```

No `start` command is needed. Collection begins automatically when a usable AirPod motion stream is available.

Useful commands:

```text
status
quit
```

`Ctrl+C` also shuts down cleanly.

## Recording files

Each app launch creates one session ID such as:

```text
session_20260925_034500_a1b2c3d4
```

The complete local backup is saved under:

```text
MacCollector/recordings/<session_id>.csv
```

The server writes/appends the same session to:

```text
data/<session_id>.csv
```

If the WebSocket reconnects, the server continues appending to the same session file.

## CSV columns

```text
seq
coremotion_timestamp
host_timestamp_utc
sensor_location
user_accel_x,user_accel_y,user_accel_z
gravity_x,gravity_y,gravity_z
ax,ay,az
gx,gy,gz
```

`ax/ay/az = userAcceleration + gravity`; `gx/gy/gz = rotationRate`.

The collector preserves the native AirPods timestamps and does not resample to 30 Hz. Real gaps, such as time with the AirPod out of the ear, are preserved.
