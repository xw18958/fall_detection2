# M5StickC PLUS2 fall detection with the supplied TCN model

This folder is an ESP-IDF firmware prototype for running the **actual trained fall-detection model supplied in `model_convert(1).zip`** on an M5Stack M5StickC PLUS2.

It does not retrain or replace the model. The firmware reads the built-in MPU6886, builds the model input window, runs TensorFlow Lite Micro locally on the ESP32, and reports Normal/Fall output probabilities over Serial. A buzzer trigger is included as a provisional demo action.

## Model interface

The uploaded conversion manifest is treated as the source of truth:

- architecture: `TCNAttnClassifier`
- input shape: `[1, 60, 6]`
- channel order: `[ax, ay, az, gx, gy, gz]`
- sample rate: `20 Hz`
- window length: `60` samples = `3.0 s`
- stride: `15` samples = `0.75 s`
- classes: `2`

Although the checkpoint filename contains `30hz`, the exported model interface uses `20 Hz`.

## Model artifact used

The firmware uses the supplied:

`tf_saved_model/model_float32_integer_quant.tflite`

This keeps float input/output with quantized internal weights/operations and is about 705 KB.

The nominal full-INT8 I/O artifact is not used for this first hardware smoke test because the supplied conversion report shows its output saturating to `[-128, -128]` on the included smoke-test sample, whereas the PyTorch reference logits are different.

## Prepare the model file

The repository keeps the firmware source here. Before the first build, extract the supplied model from your original `model_convert(1).zip` with:

```bash
cd M5StickCPlus2FallDetection
python3 tools/prepare_model.py /path/to/model_convert\(1\).zip
```

This creates:

```text
main/model.tflite
```

The script verifies the exact expected model size and SHA256 before writing it.

## GELU support

The converted TCN contains GELU operations. This project includes a small TensorFlow Lite Micro GELU implementation for float32/int8 and a resolver wrapper, so the supplied trained architecture can be attempted without replacing GELU with another activation.

## Live sensor preprocessing

The MPU6886 is configured to:

- accelerometer: ±8 g
- gyroscope: ±2000 deg/s

The firmware converts:

- acceleration: `g -> m/s²`
- angular velocity: `deg/s -> rad/s`

Then it applies the exact saved normalization:

`x_norm = (x - mean) / (sigma + 1e-6)`

This is sufficient for the current goal: proving the model can execute from live M5StickC PLUS2 IMU data. It does not yet prove cross-device/domain accuracy.

## Build

Use ESP-IDF 5.1 or newer.

From the repository root:

```bash
cd M5StickCPlus2FallDetection
idf.py set-target esp32
idf.py build
```

## Flash

Connect the M5StickC PLUS2 by USB-C and find the serial port:

```bash
ls /dev/cu.*
```

Then flash and monitor, for example:

```bash
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

Exit the monitor with `Ctrl+]`.

## Expected startup

You should see messages similar to:

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

## Current prototype trigger

Class index 1 is treated as `fall`. The firmware applies softmax to the two logits and currently beeps if:

`p(fall) >= 0.9806883345`

That threshold came from the training checkpoint's file-level scoring and is only provisional for this hardware smoke test.

## If it fails

Send the complete serial output, especially the first TensorFlow Lite Micro error immediately before `AllocateTensors failed` or `Invoke() failed`.

The goal of this stage is simply: **live MPU6886 -> preprocessing -> the supplied trained TCN -> on-device inference**.
