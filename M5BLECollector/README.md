# M5 BLE body-motion collector

Collect six-axis MPU6886 motion at 30 Hz over Bluetooth Low Energy and save it locally on a Mac. No Wi-Fi network, Internet connection or university server is needed. The Mac must remain awake, including when its lid is closed in a backpack.

This folder contains an isolated firmware project and a native Swift/CoreBluetooth Mac collector. The existing `M5StickCPlus2FallDetection/`, `MacCollector/`, training code and device firmware are not modified by building or running these tests.

## Detector preservation

- The new firmware embeds the **same locked V2 C24 INT8 model**, SHA-256 `1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530`.
- The detector retains the 30 Hz / 90×6 window, 750 ms inference cadence, 0.415 threshold, normalization, sensor configuration, GELU and optimized convolution kernels. `detector_baseline.json` records the local source snapshot and locked function hashes.
- **DETECT and COLLECT run on separate boots of the same application.** The existing detector-side B hold still enters COLLECT. In COLLECT, C returns to DETECT only when no trial is recording, awaiting review, or transferring. BLE and the recording buffer are initialized only in COLLECT. After the model replay test, collection releases the interpreter/arena RAM; the model bytes remain in flash. Returning to DETECT reinitializes the model and builds a fresh three-second window.
- The first boot defaults to DETECT. Subsequent boots restore the selected mode from the application's `m5ble` NVS namespace.
- No Internet firmware-update check runs. Recovery Wi-Fi OTA starts only in DETECT. Shared NVS is never erased to recover an initialization error.
- USB upload/erase targets are blocked. Building produces an application image; it never installs it.

The baseline is the local V2 working tree, not a claim that the physical device currently contains that exact image. Before installation, identify and back up the device's running image, partition table and OTA state. If its deployed model differs, stop and establish that actual baseline first.

## What requires one firmware installation

The current detector cannot acquire new BLE functionality without a firmware update. After validation and device backup, install the new **application-only** `firmware.bin` into the inactive OTA slot using the existing recovery page. Keep the existing partition table and bootloader. The previous active application remains in the other slot after this first update; a later OTA update may reuse that slot, so keep the external backup.

Do not run a generic USB upload, erase flash, or upload `model.tflite`. Do not publish this development build through the Internet OTA release channel. No physical installation or radio validation is performed by this source change.

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
  \
  --placement waist
```

The script builds a distinct app bundle, signs it locally and runs it. Approve its Bluetooth access/pairing request. It does not use the existing AirPods collector, ports or server.

By default it saves to `M5BLECollector/macos/recordings/`. Override with `--recordings /absolute/path`. `--device DEVICE_ID_OR_PERIPHERAL_UUID` selects a particular M5. The app remembers the previously connected peripheral; the device bonds to the first Mac used in collection mode and rejects other identities.

Terminal commands:

```text
status
start
stop
keep
discard
detect
quit
```

`keep` and `discard` mirror the REVIEW decision for terminal/debug use. `detect` requires a clean idle state and reboots the M5 into detection. `quit` never silently keeps a trial: if a recording is active it stops it and asks for an explicit KEEP/DISCARD decision. `quit force` detaches immediately; unsent device samples remain only in RAM.

## Buttons and field workflow

| State / action | Behavior |
|---|---|
| Screen asleep + A/B/C | Wake the display only; the wake press is consumed |
| READY + A | Start a locally buffered trial |
| RECORDING + A | Stop and enter REVIEW |
| RECORDING + B/C | Ignored |
| REVIEW + A | KEEP: enable BLE transfer to the Mac |
| REVIEW + B | DISCARD: clear the trial locally; no sample data reaches the Mac |
| Idle COLLECT + C | Return to DETECT by reboot |
| Hold A+B for 5 seconds in idle COLLECT | Clear this collector's BLE bond/owner and reboot; retained as a guarded recovery action |

Collection is local-first. A new trial can be recorded into M5 PSRAM even if the Mac is temporarily unavailable. No sample records are offered over BLE while RECORDING or REVIEW. Only KEEP enters the transfer state. Temporary disconnects therefore do not stop acquisition; after KEEP the M5 waits for or reconnects to the Mac and resumes the acknowledged transfer.

Before walking outside:

1. Charge the M5; fasten it consistently and record placement/orientation in the session profile (`--placement`, e.g. `chest_front_axes_up`).
2. Start with A. The screen stays awake in RECORDING and shows the elapsed trial time.
3. Perform the activity, then press A to stop. The device enters REVIEW without transferring the sample data.
4. Press A to KEEP or B to DISCARD. DISCARD clears the local trial. KEEP transfers it; if the Mac is unavailable the screen shows `WAITING FOR MAC` until transfer can resume.
5. Wait for `SAVED` after KEEP before intentionally rebooting or returning to DETECT with C. In READY the screen may sleep after 60 seconds; the first button press then wakes it without executing an action.

The normal screen emphasizes state, recording duration, BLE readiness, and the available action. `BUFFER FULL` is treated like REVIEW so the retained partial trial can still be explicitly kept or discarded. Detection and fall alarms are paused in COLLECT.

## Data and reliability

Each session directory contains:

```text
metadata.json       Device/model identity, sensor configuration and session profile
journal.jsonl       Authoritative transport/recovery journal
samples.csv         Clean acquisition data: seq,time,ax,ay,az,gx,gy,gz
quality_report.json Timing, measured rate, gaps, sensor errors and saturation
completion.json     Created only after device completion and saved-count agreement
```

Journal writes are synchronized before acknowledgements are sent. Retransmitted samples are deduplicated and checked against committed contents. On collector restart the journal is recovered, including truncation of an incomplete final line. CSV snapshots are regenerated on startup, approximately every minute, at completion and on orderly exit. Use `completion.json` to distinguish a finished recording from a partial session.

The device has an 8,192-record / 256 KiB PSRAM buffer: approximately **273 seconds at 30 Hz** before overhead/timing effects. A full buffer stops acquisition instead of overwriting unsaved data. Power loss or reboot destroys pending RAM samples. Extended standalone recording is not supported in this version.

Sampling timestamps use the device's monotonic microsecond clock. The M5 collection path stores the six signed 16-bit MPU6886 register counts untouched and performs no g/dps conversion, normalization, training-count mapping or model preprocessing. `samples.csv` contains only `seq,device_timestamp_us,ax,ay,az,gx,gy,gz`. Mac reception UTC remains only in the durable journal for transport debugging and is not exported as a training column.

Per-sample quality bits (`read error`, `timing gap`, `sensor saturation`) remain internal to the wire/journal so `quality_report.json` can report collection problems without polluting the training CSV. There is no marker workflow and no `events.csv`. Timestamp gaps remain visible and are not filled or resampled.

BLE uses encrypted Just Works bonding and one active central. Initial pairing has no passkey/MITM authentication; pair with your intended Mac in a controlled setting. If pairing fails after removing a Mac bond, reset the device bond using A+B while all data is saved, then pair again.

## Tests

Fast tests use Command Line Tools and do not access hardware:

```bash
./tools/smoke-tests.sh
```

They cover local REVIEW/KEEP/DISCARD gating, overflow, stale/future ACKs, mode-exit guards, actual C++→Swift wire compatibility, small-MTU fragmentation, reconnect replay, deduplication, journal recovery, sequence/timestamp rejection, raw-only CSV export and quality reporting. XCTest is not required.

After a firmware build downloads its pinned Micro component, the native model smoke test can also be run:

```bash
cmake -S tests/native_model -B /tmp/m5ble-native
cmake --build /tmp/m5ble-native -j 4
/tmp/m5ble-native/replay firmware/main/model.tflite
/tmp/m5ble-native/replay --kernel-test
```

Physical acceptance still requires model startup replay on the ESP32, sustained 30 Hz sampling, the actual body/backpack radio path, reconnect tests, closed-lid Mac reception, battery runtime and returning to detection. See `VALIDATION.md` for measured software checks and remaining hardware checks.
