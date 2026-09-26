# M5StickC PLUS2 fall detection with the supplied TCN model

This folder runs the **actual trained fall-detection model from `model_convert(1).zip`** on an M5Stack M5StickC PLUS2.

The current pipeline is:

```text
MPU6886 -> 20 Hz -> [ax, ay, az, gx, gy, gz] -> 60 x 6 window
        -> saved normalization -> TensorFlow Lite Micro -> Normal/Fall logits
```

The goal of this stage is to prove that the trained model can execute locally on the device. Model OTA will be added only after the USB-flashed inference path is working.

## Model interface

The uploaded conversion manifest is treated as the source of truth:

- architecture: `TCNAttnClassifier`
- input shape: `[1, 60, 6]`
- channel order: `[ax, ay, az, gx, gy, gz]`
- sample rate: `20 Hz`
- window length: `60` samples = `3.0 s`
- stride: `15` samples = `0.75 s`
- classes: `2`

Although the checkpoint filename contains `30hz`, this exported model interface uses `20 Hz`.

## Model artifact used

The firmware uses:

```text
tf_saved_model/model_float32_integer_quant.tflite
```

This keeps float input/output while quantizing internal model data/operations and is about 705 KB.

The nominal full-INT8 I/O model is not used for the first hardware test because the supplied conversion report shows saturated output on its included smoke-test sample.

## Prepare the model once

`main/model.tflite` is intentionally gitignored. Create it from your original model package:

```bash
cd ~/Documents/fall_detection2/M5StickCPlus2FallDetection
python3 tools/prepare_model.py /path/to/model_convert\(1\).zip
```

The helper verifies both the expected size and SHA256 before writing the model.

## PlatformIO configuration

This project is now configured for PlatformIO with the ESP-IDF framework.

`platformio.ini` pins:

```text
platform = espressif32@6.13.0
framework = espidf
board = m5stick-c
```

PlatformIO does not expose a separate M5StickC PLUS2 ESP-IDF board definition. M5Stack's own PLUS2 PlatformIO setup also uses `m5stick-c` as the base board, so this project overrides the PLUS2-specific resources:

- 8 MB flash
- 2 MB PSRAM through `sdkconfig.defaults`
- a large single-app partition for the current embedded-model prototype

## Build

From the project folder:

```bash
pio run
```

The first run can take a while because PlatformIO downloads the pinned ESP32 platform, ESP-IDF toolchain, and Espressif TensorFlow Lite Micro component.

## Flash

Connect the M5StickC PLUS2 with USB-C and check the detected serial device:

```bash
pio device list
```

Then flash:

```bash
pio run -t upload
```

If automatic port selection fails:

```bash
pio run -t upload --upload-port /dev/cu.usbserial-XXXX
```

## Monitor

```bash
pio device monitor
```

Or explicitly:

```bash
pio device monitor --port /dev/cu.usbserial-XXXX --baud 115200
```

Exit with `Ctrl+C`.

## Expected startup

Look for output similar to:

```text
MPU6886 WHO_AM_I = 0x..
Embedded TFLite model: 704896 bytes
PSRAM initialized=1
Tensor arena: ... bytes in PSRAM
input shape=[1,60,6] type=float32
output shape=[1,2] type=float32
Ready. Collecting MPU6886 at 20 Hz...
```

After the first 3 seconds, inference should run every 0.75 seconds:

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

Then it applies the saved model normalization:

```text
x_norm = (x - mean) / (sigma + 1e-6)
```

This is the best current alignment to the training input. It does not yet prove cross-device accuracy.

## GELU support

The converted TCN contains GELU operations. The project includes a TensorFlow Lite Micro GELU kernel and a custom op resolver so the trained architecture can be attempted without replacing GELU with ReLU or another activation.

## Current prototype trigger

Class index 1 is treated as `fall`. The firmware applies softmax to the two logits and currently beeps when:

```text
p(fall) >= 0.9806883345
```

That threshold is provisional for this hardware smoke test.

## If it fails

Send the full output of:

```bash
pio run
```

or, if the build succeeds but the device fails at runtime:

```bash
pio device monitor
```

The most useful runtime message is the first TensorFlow Lite Micro error immediately before `AllocateTensors failed` or `Invoke() failed`.
