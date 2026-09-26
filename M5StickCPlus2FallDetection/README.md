# M5StickC PLUS2 Fall Detection

ESP-IDF / PlatformIO firmware for running the pruned and fine-tuned TCN fall-detection model locally on an **M5Stack M5StickC PLUS2**.

Current pipeline:

```text
MPU6886
  -> 20 Hz six-axis IMU
  -> [ax, ay, az, gx, gy, gz]
  -> 60 x 6 rolling window (3.0 s)
  -> saved normalization
  -> TensorFlow Lite Micro
  -> softmax Normal / Fall probability
  -> screen + buzzer
```

The firmware now also supports **wireless A/B OTA software updates**. The TFLite model is embedded in the firmware image, so a wireless update can replace the application code and the model together.

## Current model interface

- architecture: `TCNAttnClassifier`, pruned C16 deployment version
- input shape: `[1, 60, 6]`
- channel order: `[ax, ay, az, gx, gy, gz]`
- sample rate: `20 Hz`
- window length: `60` samples = `3.0 s`
- inference stride: `15` samples = `0.75 s`
- classes: `2` (`0 = normal`, `1 = fall`)
- current live fall threshold: `0.88`

The current fine-tuned C16 TFLite model is about 62 KB. `main/model.tflite` is intentionally gitignored and remains a local deployment artifact.

## Current device behavior

The screen shows:

- large `NORMAL` / `FALL` state
- fall probability with one decimal place, e.g. `FALL 88.4%`
- inference latency
- probability bar
- a visible cooldown countdown after an alarm

A fall alarm is triggered when:

```text
p(fall) >= 0.88
```

The buzzer produces one short beep. After that beep finishes, a **3.0 s cooldown** begins. The screen countdown and the buzzer use the same cooldown timer.

## Prepare the model

Place the deployment model at:

```text
M5StickCPlus2FallDetection/main/model.tflite
```

For the current fine-tuned C16 deployment, this is the model previously exported as:

```text
model_finetuned_c16_seed42.tflite
```

Example:

```bash
cd ~/Documents/fall_detection2
cp ~/Downloads/model_finetuned_c16_seed42.tflite \
  M5StickCPlus2FallDetection/main/model.tflite
```

Do not commit the model binary; it is ignored by Git.

## PlatformIO configuration

The project uses:

```text
platform = espressif32@6.13.0
framework = espidf
board = m5stick-c
```

The PLUS2 uses an ESP32-PICO-V3-02 with:

- 8 MB flash
- 2 MB PSRAM

`m5stick-c` is used as the closest PlatformIO base board, with project-specific flash/PSRAM settings.

## A/B OTA partition layout

The 8 MB flash now uses two application slots:

```text
nvs
otadata
phy_init
ota_0   ~3.8 MB
ota_1   ~3.8 MB
```

The running application stays in one slot while a new firmware image is written to the inactive slot. After the upload, the ESP32 reboots into the new image.

Rollback support is enabled. A newly installed OTA image is marked valid only after the firmware successfully initializes the important runtime components, including the IMU, model, and wireless OTA service. If that startup validation fails, the device can return to the previous known-good OTA image.

## First installation: USB-C is required once

Because the partition table changed from a single application partition to A/B OTA, install this OTA-capable firmware once over USB-C.

From the project folder:

```bash
cd ~/Documents/fall_detection2/M5StickCPlus2FallDetection
source ~/.venvs/platformio39/bin/activate
export PYTHONEXEPATH="$HOME/.venvs/platformio39/bin/python"

pio run
pio run -t upload --upload-port /dev/cu.usbserial-5B1E0454241
```

If the serial port is different:

```bash
pio device list
```

and substitute the correct `/dev/cu.usbserial-...` device.

After this one USB-C installation, normal future software/model updates can be performed wirelessly.

## Wireless software update

The M5StickC PLUS2 creates its own local Wi-Fi access point:

```text
SSID:     FallDetector-OTA
Password: fallupdate
```

To update:

1. Build the new firmware on the Mac:

```bash
cd ~/Documents/fall_detection2/M5StickCPlus2FallDetection
pio run
```

2. The file to upload is:

```text
.pio/build/m5stickc-plus2/firmware.bin
```

3. On the Mac or phone, connect Wi-Fi to:

```text
FallDetector-OTA
```

using password:

```text
fallupdate
```

4. Open a browser and go to:

```text
http://192.168.4.1/
```

5. Select `firmware.bin` and press **Upload and reboot**.

6. Do not power off the M5StickC PLUS2 during the upload. When the image has been fully written and validated, the device reboots into the new OTA slot.

Because `model.tflite` is embedded into `firmware.bin`, changing the local model and rebuilding the project also updates the model wirelessly. This allows OTA deployment of a different model architecture as long as the resulting firmware fits in an OTA slot and the model still fits the device's runtime RAM/PSRAM and latency constraints.

## Important OTA limitation

Each OTA application slot is about **3.8 MB**, so the complete compiled firmware image, including the embedded TFLite model, must fit inside one slot.

A larger model can therefore be delivered wirelessly, but OTA does not remove the hardware constraints:

- firmware/model image must fit the flash slot
- TensorFlow Lite tensor arena must fit available RAM/PSRAM
- required TFLite operators must be compiled into the firmware
- inference should remain fast enough for the `0.75 s` inference stride

## OTA security status

The current OTA service is intended for **local prototype development**. It uses a WPA2-protected local access point and a browser upload page, but the OTA image itself is not yet production-signed and the web page uses local HTTP.

Before commercial deployment, add at minimum:

- unique per-device credentials instead of the shared prototype password
- signed firmware verification / secure boot as appropriate
- encrypted production update transport
- version policy and downgrade protection
- staged rollout / fleet management if devices are deployed remotely

## Monitor serial logs

```bash
pio device monitor --port /dev/cu.usbserial-5B1E0454241 -b 115200
```

Useful startup lines include:

```text
MPU6886 WHO_AM_I = 0x..
Embedded TFLite model: ... bytes
PSRAM initialized=1
input shape=[1,60,6] type=float32
output shape=[1,2] type=float32
Wireless software update ready
Wi-Fi SSID: FallDetector-OTA
Wi-Fi password: fallupdate
Update page: http://192.168.4.1/
Ready. Collecting MPU6886 at 20 Hz...
```

Inference logs look like:

```text
infer=... us | logits=[... ...] | normal=... fall=... | sample=...
```

## Live preprocessing

The MPU6886 is configured to:

- accelerometer: ±8 g
- gyroscope: ±2000 deg/s

The firmware converts:

- acceleration: `g -> m/s^2`
- angular velocity: `deg/s -> rad/s`

and then applies:

```text
x_norm = (x - mean) / (sigma + 1e-6)
```

with the normalization values stored in `main.cc`.

## GELU / TensorFlow Lite Micro support

The TCN uses GELU. The project includes the required TensorFlow Lite Micro GELU kernel and custom op resolver rather than changing the trained architecture to a different activation.

If a future model architecture introduces new TensorFlow Lite operators, update the resolver and firmware together. The resulting firmware can then be delivered through the same OTA mechanism.

## Troubleshooting

Build failure:

```bash
pio run
```

Runtime / OTA debugging:

```bash
pio device monitor --port /dev/cu.usbserial-5B1E0454241 -b 115200
```

If wireless updating is unavailable because the currently installed firmware predates the OTA implementation or the partition table has not yet been migrated, use USB-C once to install the current firmware and partition table.
