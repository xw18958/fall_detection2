# Fall Detection

This repository contains the fall-detection data-collection, model, and embedded-device prototypes.

## M5StickC PLUS2 embedded fall detector

`M5StickCPlus2FallDetection/` runs the pruned/fine-tuned TCN locally on an M5Stack M5StickC PLUS2 using the MPU6886 six-axis IMU and TensorFlow Lite Micro.

Current device behavior includes:

- 20 Hz six-axis IMU sampling
- 3.0 s / 60-sample model window
- inference every 0.75 s
- on-device `NORMAL` / `FALL` display
- fall probability with one decimal place
- one short fall-alert beep
- 3.0 s beep cooldown with matching screen countdown
- current live fall threshold: `0.88`
- **wireless A/B OTA software updates with rollback**

### Wireless software update

The OTA-capable firmware and A/B partition table must first be installed once using USB-C. After that, future firmware and embedded-model updates can normally be installed wirelessly.

Build the new image:

```bash
cd M5StickCPlus2FallDetection
pio run
```

The OTA file is:

```text
.pio/build/m5stickc-plus2/firmware.bin
```

Connect a Mac or phone to the device's update Wi-Fi:

```text
SSID:     FallDetector-OTA
Password: fallupdate
```

Then open:

```text
http://192.168.4.1/
```

and upload `firmware.bin`.

The TFLite model is embedded in the firmware image, so OTA can update the **software and model together**, including a different compatible model architecture. The complete firmware image must fit one OTA slot and the new model must still satisfy the ESP32's PSRAM/RAM and inference-time limits.

See [`M5StickCPlus2FallDetection/README.md`](M5StickCPlus2FallDetection/README.md) for the complete build, first-flash, OTA, rollback, model, and troubleshooting instructions.

## AirPods Pro motion collector

Minimal macOS + Python pipeline for collecting AirPods motion data and streaming it to the USyd server.

## iPhone prototype

`iPhoneFallDetection/` contains a 25 Hz iPhone-only inference prototype. AirPods Pro 2 provides the six-axis motion stream; the iPhone handles the rolling 75 x 6 window and a dual-stream TCN-like model with separate accelerometer/gyroscope branches, residual dilated TCN blocks, attention pooling, modality gating/fusion, and a two-class output head. The prototype weights are deterministic random initialization for pipeline testing only.

Open `iPhoneFallDetection/iPhoneFallDetection.xcodeproj` in Xcode. See `iPhoneFallDetection/README.md` for setup and CSV replay instructions.

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
