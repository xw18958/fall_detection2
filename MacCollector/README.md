# AirPods Motion Collector

This native macOS command-line app reads `CMHeadphoneMotionManager` samples from
compatible AirPods, derives the six fall-detection input channels, and sends small
JSON batches to a WebSocket receiver.

## Requirements

- macOS 14 or later (`CMHeadphoneMotionManager` is available from macOS 14)
- Xcode command-line tools or Xcode
- AirPods Pro 2 connected to the Mac with headphone motion available
- An SSH tunnel and WebSocket receiver running as described below

## Run

On the Mac, open a second terminal for the tunnel:

```bash
ssh -N -L 8765:127.0.0.1:8765 xwan0900@gpu1-jinman-2.it.usyd.edu.au
```

From this directory, build, package, and start the collector:

```bash
./Scripts/run-app.sh
```

This creates `.build/AirPodsMotionCollector.app`, embeds `Info.plist` with
`NSMotionUsageDescription`, ad-hoc signs it, then runs its console interface in the
same Terminal. This is the required run path for CoreMotion privacy consent; do not
use `swift run` to collect motion data. The default server is `ws://127.0.0.1:8765`.
Use `start` to begin collection, `stop` to pause it, `status` to display the current
values, and `quit` to exit.
To use another endpoint:

```bash
./Scripts/run-app.sh --server ws://127.0.0.1:8765
```

The console shows the AirPods connection state reported by CoreMotion, a rolling
two-second sampling-rate estimate calculated from the original CoreMotion timestamps,
total collected samples, and WebSocket state. It sends a JSON message about every
150 ms while connected:

```json
{
  "samples": [
    {"timestamp": 123.040, "ax": 0.01, "ay": -0.04, "az": 1.01,
     "gx": 0.03, "gy": -0.01, "gz": 0.04}
  ]
}
```

For each `CMDeviceMotion` sample, the collector preserves `motion.timestamp` and
calculates `ax`, `ay`, and `az` as `userAcceleration + gravity`; `gx`, `gy`, and
`gz` are `rotationRate`. It does not resample or impose a target sample rate.

## Mac-side check

Wear the AirPods, run `start`, and move your head. The periodic status output shows
an increasing sample count, a non-zero measured rate, and the latest timestamp,
acceleration, and rotation values. Run `status` at any time for the same snapshot.
