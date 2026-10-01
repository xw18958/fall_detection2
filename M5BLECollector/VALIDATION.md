# Software validation — 1 October 2026

The implementation is isolated in `M5BLECollector/`. The existing detector source files were not edited. The connected M5 was backed up and the collector was installed into its inactive OTA slot on 1 October 2026.

| Check | Result |
|---|---|
| ESP-IDF 5.5.3 / PlatformIO release firmware build | PASS |
| Application image size | 1,443,696 bytes after BLE diagnostics/stability changes; existing OTA slot is 3,997,696 bytes |
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

BLE troubleshooting update: encrypted model identification and the READY handshake
have succeeded on the actual M5. Two short setup attempts durably saved 11 and
40 samples respectively, but neither recording completed. Repeated supervision
timeouts and GATT discovery timeouts remain unresolved. These interrupted tests
must not be treated as complete training recordings. The Mac now serializes its
initial encrypted reads before notification subscription, avoids duplicate
pending reads, and reports readiness only after the device confirms it. The
collector radio keeps modem sleep disabled and requests a 30–45 ms interval,
zero slave latency and a six-second supervision timeout; a physical test showed
the request accepted (45 ms). These changes have not established sustained BLE
reliability. Opening the USB diagnostic port was observed to reboot the board;
do not open serial tools during recording or while samples remain pending.

The Mac's Bluetooth daemon additionally recorded an explicit MIC failure during
sample transfer. Restarting Mac Bluetooth did not recover the link. A candidate
with a 64-byte fragment payload and legacy-sized encrypted link packets also
disconnected while idle, before recording. The optional unpaired transport test
is isolated behind an ignored local build override; it is not the default build
and must not be installed without approval for unencrypted motion transfer.
The optional transport field is backward compatible, and a smoke test confirms
that its value survives creation of saved session metadata.

No claim of validated real-device accuracy or battery duration is made. Pending device samples are buffered in PSRAM and are lost on power loss/reboot; the Mac's synchronized journal is the persistent record.

The approved unpaired candidate (1,443,088 bytes) also lost its connection while
idle and repeatedly timed out during GATT discovery. Testing without USB did
not establish reliability. The Mac receiver now uses its main Foundation run
loop and explicitly prevents App Nap during reception.

The pinned SDK is affected by Espressif's documented ESP-IDF 5.5.3 NimBLE
host-flow-control defect (issue 18323). Its flow control was enabled in the
failing builds. The project defaults and local generated configuration now
disable it using the vendor's workaround. Build/installation and a completed
physical recording with this workaround still require verification.
