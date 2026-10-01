# M5 BLE body-motion collector

Collect six-axis MPU6886 motion at 30 Hz over Bluetooth Low Energy and save it locally on a Mac. No Wi-Fi network, Internet connection or university server is needed. The Mac must remain awake, including when its lid is closed in a backpack.

This folder contains an isolated firmware project and a native Swift/CoreBluetooth Mac collector. The existing `M5StickCPlus2FallDetection/`, `MacCollector/`, training code and device firmware are not modified by building or running these tests.

## Detector preservation

- The new firmware embeds the **same locked V2 C24 INT8 model**, SHA-256 `1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530`.
- The detector retains the 30 Hz / 90×6 window, 750 ms inference cadence, 0.415 threshold, normalization, sensor configuration, GELU and optimized convolution kernels. `detector_baseline.json` records the local source snapshot and locked function hashes.
- **DETECT and COLLECT run on separate boots of the same application.** Holding B switches mode by a brief reboot, without reflashing. BLE and the recording buffer are initialized only in COLLECT. After the model replay test, collection releases the interpreter/arena RAM; the model bytes remain in flash. Returning to DETECT reinitializes the model and builds a fresh three-second window.
- The first boot defaults to DETECT. Subsequent boots restore the selected mode from the application's `m5ble` NVS namespace.
- No Internet firmware-update check runs. Recovery Wi-Fi OTA starts only in DETECT. Shared NVS is never erased to recover an initialization error.
- USB upload/erase targets are blocked. Building produces an application image; it never installs it.

The baseline is the local V2 working tree, not a claim that the physical device currently contains that exact image. Before installation, identify and back up the device's running image, partition table and OTA state. If its deployed model differs, stop and establish that actual baseline first.

## What requires one firmware installation

The current detector cannot acquire new BLE functionality without a firmware update. After validation and device backup, install the new **application-only** `firmware.bin` into the inactive OTA slot using the existing recovery page. Keep the existing partition table and bootloader. The previous active application remains in the other slot after this first update; a later OTA update may reuse that slot, so keep the external backup.

Do not run a generic USB upload, erase flash, or upload `model.tflite`. Do not publish this development build through the Internet OTA release channel. See the installation record below for the development device's actual status.

The development M5 was installed on 1 October 2026 using a targeted USB write to the inactive `ota_1` application slot at `0x3E0000`, after backing up and validating the complete active detector application and boot/settings regions. Only the unused second OTA selector sector was updated to activate the new application with rollback enabled. The original `ota_0` detector, first valid OTA selector, bootloader, partition table and NVS were preserved. The deployed model matched the locked model byte for byte. Private backups and device logs are kept under the ignored `artifacts/` directory. See `VALIDATION.md` for hardware checks.

## Prepare private model inputs

Model binaries and deterministic private replay inputs are intentionally absent from Git. On the existing Mac checkout:

```bash
cd M5BLECollector
python3 tools/prepare_model.py
```

The tool reads the existing detector's model, replay header and configuration, verifies their locked hashes, then copies only the model/replay into this project's ignored local inputs. It performs no conversion, training, detector edits or device access.

For a fresh checkout, supply a trusted directory containing the matching `model.tflite`, `model_v2_replay.h` and `model_v2_config.h`:

```bash
python3 tools/prepare_model.py --source /path/to/private/model-inputs
```

## Build firmware without installing it

Use the same PlatformIO environment/toolchain as the existing detector:

```bash
python3 -m platformio run -d firmware
python3 tools/verify_preservation.py \
  --original ../M5StickCPlus2FallDetection \
  --firmware-binary firmware/.pio/build/m5stickc-plus2-ble/firmware.bin
```

On the original development Mac, PlatformIO is installed at `~/.venvs/platformio39/bin/python`. A fresh checkout can omit `--original`: that option additionally checks that the existing local detector snapshot has not changed.

The build guard verifies the locked model/configuration, kernels, inference functions and partition table. The verification command also confirms that the finished application embeds the exact model once and fits the existing 0x3D0000-byte OTA slot.

## Run the Mac collector

Requirements: macOS 14+, Apple's Command Line Tools or Xcode, Bluetooth enabled, and a device running the reviewed BLE firmware in COLLECT mode.

Double-click `StartCollector.command` to open the collector in Terminal. Without flags, participant, activity and placement are recorded as `unspecified`; use the command below to describe a recording, or `activity LABEL` between sessions.

```bash
cd M5BLECollector/macos
./Scripts/run-app.sh \
  --participant P001 \
  --activity walking \
  --placement waist
```

The script builds a distinct app bundle, signs it locally and runs it. Approve its Bluetooth access/pairing request. It does not use the existing AirPods collector, ports or server.

By default it saves to `M5BLECollector/macos/recordings/`. Override with `--recordings /absolute/path`. `--device DEVICE_ID_OR_PERIPHERAL_UUID` selects a particular M5. The app remembers the previously connected peripheral; the device bonds to the first Mac used in collection mode and rejects other identities.

Terminal commands:

```text
status
start
mark stairs_start
stop
activity sitting
detect
quit
```

`activity` changes the label for the next session. `detect` requires all samples saved and reboots the M5 into detection. `quit` stops an active recording and waits for transfer completion. `quit force` detaches immediately; the Mac's committed journal remains safe, but unsent device samples remain only in RAM. A second interrupt also detaches.

## Buttons and field workflow

| Action | Behavior |
|---|---|
| Hold B for 2 seconds | Switch DETECT ↔ COLLECT by reboot; rejected during recording or pending transfer |
| Press A in COLLECT | Start/stop recording |
| Press B during recording | Insert marker ID 1 at the next acquired sample |
| Hold A+B for 5 seconds in idle COLLECT | Clear this collector's BLE bond/owner and reboot; does not erase shared NVS or alter the model |

The Mac must connect, subscribe and complete its readiness handshake before a new recording can start. Once recording, temporary disconnects do not stop sampling. Stop continues working on the device without the Mac; reconnect to finish saving.

Before walking outside:

1. Charge the M5; fasten it consistently and record placement/orientation in the session profile (`--placement`, e.g. `waist_front_axes_up`).
2. Connect the collector and check `MAC READY` on the device.
3. Close the Mac in the actual backpack setup and verify reception remains continuous for several minutes. The collector prevents **idle sleep**; it does not override lid-triggered sleep. Your separate closed-lid awake configuration must be working on battery.
4. Start with A. Confirm `RECORDING`; distinct short beeps indicate state changes. The screen turns off after 20 seconds and wakes on button activity.
5. Stop with A, wait for `SAVED`, then return to detection by holding B if desired.

The screen reports acquired and pending samples. `DISCONNECTED` means the RAM buffer is being used. `SAVING` means acquisition has stopped but transfer is incomplete. `BUFFER FULL` means recording stopped to preserve existing unsaved samples. Detection and fall alarms are paused in COLLECT.

## Data and reliability

Each session directory contains:

```text
metadata.json       Device/model identity, sensor units, profile, marker labels
journal.jsonl       Authoritative, synchronized sample journal
samples.csv         Raw counts and physical-unit export
events.csv          Marker timestamps and labels
quality_report.json Timing, measured rate, gaps, sensor errors and saturation
completion.json     Created only after device completion and saved-count agreement
```

Journal writes are synchronized before acknowledgements are sent. Retransmitted samples are deduplicated and checked against committed contents. On collector restart the journal is recovered, including truncation of an incomplete final line. CSV snapshots are regenerated on startup, approximately every minute, at completion and on orderly exit. Use `completion.json` to distinguish a finished recording from a partial session.

The device has an 8,192-record / 256 KiB PSRAM buffer: approximately **273 seconds at 30 Hz** before overhead/timing effects. A full buffer stops acquisition instead of overwriting unsaved data. Power loss or reboot destroys pending RAM samples. Extended standalone recording is not supported in this version.

Sampling timestamps use the device's monotonic microsecond clock. Mac reception UTC is recorded separately; it is not acquisition UTC and cannot be used as the sample interval during replay. Store raw counts before inferred training-scale mapping, normalization or INT8 quantization. Acceleration includes gravity. Failed reads are explicitly flagged; timestamp gaps remain visible and are not filled or resampled.

Flags: `1=read error`, `2=sampling timing gap`, `4=sensor rail/saturation`, `8=marker`. Button markers refer to the next acquisition (normally within one 30 Hz interval). Use labels and session metadata for ground truth; model predictions are not labels.

BLE uses encrypted Just Works bonding and one active central. Initial pairing has no passkey/MITM authentication; pair with your intended Mac in a controlled setting. If pairing fails after removing a Mac bond, reset the device bond using A+B while all data is saved, then pair again.

### Optional unpaired transport test

The default build requires encrypted bonding. A local diagnostic override can
test an unpaired link when macOS reports an encrypted-packet MIC failure:
create `firmware/main/ble_transport_config.local.h` containing
`#define M5BLE_UNPAIRED_TRANSPORT 1`, then rebuild. Remove the local override to
return to the encrypted build. The override file is ignored by Git.

Install this test only with the device owner's approval: motion data is sent
without encryption or paired-Mac access control. A nearby central could connect
to it. It uses a separate static BLE address to avoid restoring the encrypted
bond, while retaining the same physical device ID, detector model, sensor data
and save protocol. The Mac prints the unpaired transport and stores
`device.transport = "ble_unpaired"` in session metadata. This test does not erase
existing pairing records. Stop and save all data before changing firmware.

## Tests

Fast tests use Command Line Tools and do not access hardware:

```bash
./tools/smoke-tests.sh
```

They cover overflow, ring wraparound, stale/future ACKs, mode-exit guards, actual C++→Swift wire compatibility, small-MTU fragmentation, reconnect replay, deduplication, journal recovery, sequence/timestamp rejection and marker CSV escaping. XCTest is not required.

After a firmware build downloads its pinned Micro component, the native model smoke test can also be run:

```bash
cmake -S tests/native_model -B /tmp/m5ble-native
cmake --build /tmp/m5ble-native -j 4
/tmp/m5ble-native/replay firmware/main/model.tflite
/tmp/m5ble-native/replay --kernel-test
```

Physical acceptance still requires model startup replay on the ESP32, sustained 30 Hz sampling, the actual body/backpack radio path, reconnect tests, closed-lid Mac reception, battery runtime and returning to detection. See `VALIDATION.md` for measured software checks and remaining hardware checks.
