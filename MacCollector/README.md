# AirPods Motion Collector

Native macOS command-line collector using Apple's `CMHeadphoneMotionManager`.

## Requirements

- macOS 14 or later
- Xcode or Xcode command-line tools
- AirPods Pro 2 connected to the Mac
- Motion & Fitness permission enabled
- SSH tunnel and remote receiver running if server transfer is required

The collector supports **one AirPod at a time**. Either the left or right bud can be used; the other can remain in the charging case.

## Run

```bash
./Scripts/run-app.sh
```

Collection starts automatically. There is no `start` command.

Commands:

```text
status
quit
```

You can also press `Ctrl+C` to stop cleanly.

## What it does

The collector:

- detects the active AirPod side from `CMDeviceMotion.sensorLocation`
- keeps the native Core Motion timestamp
- adds a UTC host timestamp and sequence number
- records user acceleration, gravity, total acceleration, and rotation rate
- measures the actual AirPods sampling rate
- saves a complete local CSV backup under `recordings/`
- sends WebSocket batches about every 150 ms
- automatically reconnects to the server
- automatically pauses when the AirPod is removed and resumes when it reconnects
- restarts Core Motion if the stream is active but no samples arrive for about 2 seconds

The displayed status looks like:

```text
AirPod: Right | State: Connected | Rate: 25.0 Hz | Samples: 1234 | Server: Connected
```

When the AirPod is removed:

```text
AirPod: -- | State: Disconnected | Rate: -- | Samples: 1234 | Server: Connected
```

## Server endpoint

Default:

```text
ws://127.0.0.1:8765
```

Override it with:

```bash
./Scripts/run-app.sh --server ws://127.0.0.1:8765
```

For the USyd server, create the tunnel in another Terminal:

```bash
ssh -N -L 8765:127.0.0.1:8765 \
  xwan0900@gpu1-jinman-2.it.usyd.edu.au
```

## Data

Each sample contains:

```text
seq
coremotion_timestamp
host_timestamp_utc
sensor_location
user_accel_x/y/z
gravity_x/y/z
ax/ay/az
gx/gy/gz
```

`ax/ay/az` are calculated as `userAcceleration + gravity`.

The collector does **not** resample or fill missing periods. If the AirPod is out of the ear for several seconds, that real timestamp gap remains in the recording.
