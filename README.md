# AirPods Pro motion collector

Minimal macOS collector for headphone motion data and a small Python WebSocket
receiver. The receiver writes one CSV per WebSocket connection.

## Remote receiver

On `gpu1-jinman-2`, install the small server dependency into the supplied
environment and start the receiver from the project directory:

```bash
cd /raid1/xwan0900/fall_detection2
/raid1/xwan0900/venvs/ftkp_cu128/bin/python -m pip install -r requirements-server.txt
/raid1/xwan0900/venvs/ftkp_cu128/bin/python server.py
```

The server binds to `127.0.0.1:8765` and writes files to `data/`. Keep it
running. It creates `recording_YYYY-MM-DD_HHMMSS.csv` when a collector connects.
`MOTION_HOST`, `MOTION_PORT`, and `MOTION_DATA_DIR` can override those defaults.

## SSH tunnel from the Mac

In a separate Terminal window, forward the Mac's local port to the receiver:

```bash
ssh -N -L 8765:127.0.0.1:8765 \
  xwan0900@gpu1-jinman-2.it.usyd.edu.au
```

Leave this tunnel running while collecting. The Mac app connects to
`ws://127.0.0.1:8765`; this keeps the server off the public network.

## Mac collector

See [`MacCollector/README.md`](MacCollector/README.md) for build, permission,
and run steps.

The collector preserves the timestamp provided by Core Motion, computes
`Accel = userAcceleration + gravity` and `Gyro = rotationRate`, reports the
measured average sample rate, and sends batches every 150 ms. Sampling is not
resampled or forced to a target rate.

## CSV columns

```text
timestamp,ax,ay,az,gx,gy,gz
```

`data/` is ignored by Git so recordings stay local to the server.
