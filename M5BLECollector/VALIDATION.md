# Software validation — 1 October 2026

## Latest-main BLE regression recovery — 2 October 2026

The recovery branch is based on `e288bdd`, retaining the local RECORDING →
REVIEW → KEEP/DISCARD workflow and eight-column raw sensor CSV export.

- All six Python guards, sanitized C++ buffer tests and C++ → Swift protocol/
  journal smoke scenarios passed. The signed Swift release receiver built.
- A live second receiver was rejected by the recordings-folder lock before
  changing the participant profile or recording files.
- ESP-IDF 5.5.3 firmware built with NimBLE host flow control disabled; the
  generated configuration was checked. Image size: 1,444,144 bytes; SHA-256:
  `0dca200be65e0c94ef06e35efe1b42ec0ee242e227497cad09d4853167dbcb7e`.
- Installed only `ota_1` at `0x3E0000`; esptool verified the written image hash.
  The locked detector model, inference functions, kernels, configuration and
  partition hashes passed preservation checks. Post-update serial output
  confirmed model replay passed with an 80,192-byte arena and COLLECT startup.
- macOS first reported `Peer removed pairing information`. The M5-specific
  Mac bond removal was confirmed in Bluetooth logs. The next attempt was
  blocked by `LeConnectionDenyList`; macOS ignored the connection request.
  A user-approved Bluetooth off/on refresh cleared that block. The receiver
  then completed service/characteristic discovery, and macOS logged the Just
  Works pairing prompt. The Mac was locked, so approval could not complete;
  physical KEEP/DISCARD validation is pending pairing approval.

The measurements below describe the original installation, not the recovery
branch's pending radio acceptance tests.

The implementation is isolated in `M5BLECollector/`. The existing detector source files were not edited. The connected M5 was backed up and the collector was installed into its inactive OTA slot on 1 October 2026.

| Check | Result |
|---|---|
| ESP-IDF 5.5.3 / PlatformIO release firmware build | PASS |
| Application image size | 1,442,848 bytes; existing OTA slot is 3,997,696 bytes |
| Static internal RAM | 52,768 bytes; runtime heap requirements still need physical validation |
| Exact embedded INT8 model | PASS; model is embedded once, 147,976 bytes |
| Model SHA-256 | `1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530` |
| Original detector source snapshot unchanged | PASS |
| Inference function/configuration/kernel/partition hashes | PASS |
| Native Micro replay, both deterministic cases | PASS; output differences within existing tolerance of 4 INT8 counts |
| Native Micro arena used | 101,376 bytes |
| Optimized convolution versus reference implementation | All 594 cases bit-exact |
| Buffer tests with address/undefined-behavior sanitizers | PASS; overflow, wraparound, stale/future ACKs, replay and mode-exit guards |
| Mac protocol/journal smoke scenarios | PASS; fragmentation, replay, crash recovery, deduplication, invalid data rejection and markers |
| Actual C++ buffer → Swift decoder fixture | PASS, including default ATT MTU 23 fragmentation |
| Python build/input/preservation guards | All 3 tests PASS |
| Native Swift Mac app release build and signed launcher `--help` | PASS |

Hardware checks completed during installation:

- Saved bootloader, partition table, NVS and OTA settings, plus the complete original active application. The active application passed ESP image checksum and SHA-256 checks (1,232,304 bytes).
- Confirmed the deployed model exactly matches the locked 147,976-byte model above.
- Installed the 1,442,848-byte collector into inactive `ota_1`; serial tool verified the written image hash. Activated it through the second OTA selector as a NEW image, preserving the original VALID application/selector and all other flash regions.
- Confirmed startup from `0x3E0000`, successful PSRAM/IMU initialization, model replay pass, recovery OTA initialization and live detector inference. ESP32 arena usage was 80,192 bytes; sampled inference durations were approximately 0.458–0.473 seconds.
- Private backup checksums and startup logs are stored in ignored `artifacts/device-20261001/`.

Not yet physically verified:

- BLE connection/bonding and saving a physical recording on the actual M5.
- Sustained collection timing and runtime memory under BLE load.
- Outdoor body-to-backpack radio reliability and reconnect recovery.
- Continuous reception with the Mac closed and awake on battery.
- Battery runtime, button interactions, OTA fallback and return to detection.
- Sensor mounting/orientation and the existing detector's provisional training-count scale.

No claim of validated real-device accuracy or battery duration is made. Pending device samples are buffered in PSRAM and are lost on power loss/reboot; the Mac's synchronized journal is the persistent record.
