# M5StickC PLUS2 Fall Detection

ESP-IDF / PlatformIO firmware for running the pruned and fine-tuned TCN fall-detection model locally on an **M5Stack M5StickC PLUS2**.

Current inference pipeline:

```text
MPU6886
  -> 20 Hz six-axis IMU
  -> [ax, ay, az, gx, gy, gz]
  -> 60 x 6 rolling window (3.0 s)
  -> saved normalization
  -> TensorFlow Lite Micro
  -> Normal / Fall probability
  -> screen + buzzer
```

The TFLite model is embedded into the application firmware. Updating the application therefore updates both the firmware and model together.

## Current model interface

- architecture: `TCNAttnClassifier`, pruned C16 deployment version
- input shape: `[1, 60, 6]`
- channel order: `[ax, ay, az, gx, gy, gz]`
- sample rate: `20 Hz`
- window: `60` samples = `3.0 s`
- inference stride: `15` samples = `0.75 s`
- classes: `0 = normal`, `1 = fall`
- current prototype fall threshold: `0.88`

The current fine-tuned C16 TFLite model is about 62 KB. `main/model.tflite` is deliberately gitignored and must remain a local deployment artifact.

## Runtime behavior

The screen displays the current `NORMAL` / `FALL` state, fall probability, inference latency, probability bar, and alarm cooldown. A fall alarm triggers when `p(fall) >= 0.88`; the buzzer emits one short beep followed by a 3-second cooldown.

Live sensor preprocessing uses MPU6886 ±8 g accelerometer and ±2000 deg/s gyroscope settings. Firmware converts acceleration from `g` to `m/s^2`, angular velocity from `deg/s` to `rad/s`, then applies the saved normalization:

```text
x_norm = (x - mean) / (sigma + 1e-6)
```

## PlatformIO / ESP-IDF

The project uses:

```text
platform = espressif32@6.13.0
framework = espidf
board = m5stick-c
```

The PLUS2 has an ESP32-PICO-V3-02, 8 MB flash and 2 MB PSRAM. The project uses the PlatformIO `m5stick-c` definition as the closest base board and overrides the PLUS2 flash/PSRAM configuration.

## Local files that must never be committed

The following are ignored by Git:

```text
main/model.tflite
main/wifi_secrets.h
main/ota_private.pem
main/ota_private_blob.S
security/ota_public.pem
dist/
```

Prepare Wi-Fi credentials once:

```bash
cp main/wifi_secrets.example.h main/wifi_secrets.h
```

then edit `main/wifi_secrets.h` with the development Wi-Fi SSID and password.

Place the deployment model at:

```text
main/model.tflite
```

## A/B OTA and rollback

The 8 MB flash uses two application slots:

```text
nvs
otadata
phy_init
ota_0   ~3.8 MB
ota_1   ~3.8 MB
```

An OTA update is written to the inactive slot. After reboot, a newly installed image must successfully initialize the required detector components before it is marked valid. If startup validation fails while the image is pending verification, rollback can return to the previous known-good slot.

The complete firmware image, including the embedded TFLite model, must fit in one OTA slot. Runtime tensor memory must also fit the available RAM/PSRAM.

# GitHub Internet OTA (development stage)

The repository can currently act as the update host without publishing the plaintext model-containing `firmware.bin`.

The flow is:

```text
local model.tflite + source
        ↓
pio run
        ↓
plaintext firmware.bin       (local only)
        ↓
RSA-3072/AES-GCM packaging
        ↓
firmware.enc                 (public GitHub release asset)
        ↓
GitHub Release vX.Y.Z
        ↓ HTTPS
M5StickC PLUS2
        ↓
decrypt -> inactive OTA slot -> reboot -> startup validation
```

The Internet updater checks this manifest once at boot:

```text
M5StickCPlus2FallDetection/ota/stable.json
```

The tracked manifest is intentionally disabled until a real release has been uploaded and physically tested:

```json
{
  "enabled": false,
  "version": "1.0.0",
  "sha256": ""
}
```

When enabled, the device downloads the exact versioned asset:

```text
https://github.com/xw18958/fall_detection2/releases/download/v<VERSION>/firmware.enc
```

It does **not** use GitHub's `/releases/latest/` alias. The encrypted asset SHA-256 must match the hash in `stable.json` before the OTA image is activated. HTTPS uses the ESP-IDF certificate bundle.

## One-time development OTA key setup

Generate the RSA-3072 development key pair locally:

```bash
bash tools/generate_ota_keys.sh
```

This creates:

```text
main/ota_private.pem          # RSA private key; gitignored
main/ota_private_blob.S       # generated build source containing that key; gitignored
security/ota_public.pem       # encrypts firmware.enc locally; gitignored
```

The generated assembly file is only a PlatformIO/ESP-IDF build workaround. It contains the same secret key material as `main/ota_private.pem` and must never be committed or shared.

The script refuses to overwrite existing key material. Back up `main/ota_private.pem` before deploying it to any device you care about. A device built with one private key cannot decrypt a release encrypted for another key.

If the private key/blob is absent, the project still builds, but GitHub Internet OTA is compiled as disabled.

## Build

```bash
cd ~/Documents/fall_detection2/M5StickCPlus2FallDetection
source ~/.venvs/platformio39/bin/activate
pio run
```

The plaintext build output is:

```text
.pio/build/m5stickc-plus2/firmware.bin
```

**Do not upload this plaintext file to the public GitHub repository or a public GitHub Release.** It contains the embedded model.

## Prepare an encrypted GitHub release

The firmware version has one source of truth:

```text
main/version.h
```

For example:

```cpp
#define FALL_FIRMWARE_VERSION "1.1.0"
```

After changing the version and building:

```bash
pio run
python tools/prepare_release.py
```

`prepare_release.py`:

1. encrypts `firmware.bin` with the local OTA public key;
2. decrypts the result again locally and verifies it exactly matches the original firmware;
3. calculates SHA-256 of the encrypted artifact;
4. writes:

```text
dist/v1.1.0/firmware.enc
dist/v1.1.0/stable.json
```

## Publish an update

Publish in this order so a device can never see a manifest pointing to a missing asset:

1. Create a GitHub Release tagged exactly `v1.1.0`.
2. Upload **only** `dist/v1.1.0/firmware.enc` as an asset named `firmware.enc`.
3. Copy `dist/v1.1.0/stable.json` to `ota/stable.json`.
4. Review the manifest and push it to `main`.

Example generated manifest:

```json
{
  "enabled": true,
  "version": "1.1.0",
  "sha256": "<sha256-of-firmware.enc>"
}
```

On the next boot with Internet access, an older device checks the manifest, downloads the exact newer encrypted release, verifies the encrypted download hash, decrypts it while OTA is running, validates the resulting ESP32 application image, switches the A/B boot slot and reboots.

Failures in Wi-Fi, SNTP, HTTPS, manifest parsing, hashing, decryption or OTA leave the installed detector firmware in use.

## First physical installation

Because the project uses a custom A/B partition table, install this OTA-capable firmware by USB-C at least once:

```bash
pio device list
pio run -t upload --upload-port /dev/cu.usbserial-XXXXXXXX
```

Then monitor startup:

```bash
pio device monitor --port /dev/cu.usbserial-XXXXXXXX -b 115200
```

Useful log lines include model size, PSRAM allocation, model input/output shapes, MPU6886 identification, recovery OTA startup, Internet OTA version status and inference output.

# Local recovery OTA

For development/recovery, the current firmware also keeps the existing local access point:

```text
SSID:     FallDetector-OTA
Password: fallupdate
URL:      http://192.168.4.1/
```

Build with `pio run`, connect a Mac/phone to that AP, open the URL, select the local plaintext `.pio/build/m5stickc-plus2/firmware.bin`, then choose **Upload and reboot**.

This local browser updater is intentionally retained for the current testing stage and writes to the inactive A/B slot. Do not power the device off during an upload.

# Current security boundary

This implementation is a **development-stage protection**, not the final commercial security architecture.

What it currently protects:

- `model.tflite` is not in Git.
- plaintext `firmware.bin`, which contains the model, does not need to be published.
- public GitHub Releases contain only `firmware.enc`.
- GitHub downloads use HTTPS certificate verification.
- encrypted release corruption/tampering is rejected by AES-GCM and the manifest SHA-256 check.
- A/B OTA and rollback reduce the risk of an unusable remote update.

What it does **not** yet protect:

- a skilled attacker with physical access can potentially dump the ESP32 because Flash Encryption is not enabled yet;
- the OTA private key is embedded in the development firmware;
- Secure Boot is not enabled yet;
- the public manifest is not independently signed;
- the recovery AP uses one shared prototype password and local HTTP;
- there is no per-device identity, private fleet authorization, staged rollout or server-side health reporting.

Before handing production units to customers, the next security stage should add ESP32 Flash Encryption + Secure Boot, production key provisioning, stronger recovery access control and a real authenticated backend. Those changes are deliberately postponed while the device and model are still being developed.

# Software-side verification

The GitHub Actions build/package test has been exercised successfully without physical hardware. It confirmed that:

- the generated private-key assembly source compiles and links into the ESP-IDF firmware;
- the firmware image fits comfortably in the current OTA slot using the CI dummy model;
- `firmware.bin` can be pre-encrypted with the generated public key;
- the resulting `firmware.enc` can be decrypted with the matching private key;
- the decrypted bytes exactly match the original `firmware.bin`.

The workflow uses a dummy model, test Wi-Fi values, and throwaway OTA keys, so this is a build/release-path verification only. It does not test the real model or real device behavior.

# CI checks

`.github/workflows/m5stick-build.yml` is intentionally **manual-only** (`workflow_dispatch`) during development, so ordinary pushes do not generate repeated build notifications. When manually run, it builds with a dummy local-only model, test Wi-Fi values and throwaway OTA keys, then runs `tools/prepare_release.py` as a strict round-trip packaging check. No real model, Wi-Fi credential or persistent OTA private key is stored by the workflow.

# Hardware validation still required

The source/build/release path can be checked without a device, but the following must be tested when an M5StickC PLUS2 is available:

1. USB flash and normal boot;
2. display, IMU, buzzer and real model inference;
3. Internet update from one version to a newer version;
4. local recovery OTA;
5. rollback after a deliberately bad startup;
6. loss of Wi-Fi / Internet during update;
7. power interruption during OTA.

Until those tests pass, Internet OTA should remain disabled in `ota/stable.json`.
