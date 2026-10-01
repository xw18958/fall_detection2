# Software validation — 1 October 2026

The implementation is isolated in `M5BLECollector/`. The existing detector source files were not edited. The connected M5 was backed up and the collector was installed into its inactive OTA slot on 1 October 2026.

| Check | Result |
|---|---|
| ESP-IDF 5.5.3 / PlatformIO release firmware build | PASS |
| Application image size | 1,442,352 bytes with the host-flow-control workaround; existing OTA slot is 3,997,696 bytes |
| Static internal RAM | 52,760 bytes; long-duration runtime memory still needs physical validation |
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

- Sustained encrypted collection/bonding with the flow-control workaround (current physical tests use the approved unpaired build).
- Long-duration collection timing and runtime memory under BLE load.
- Outdoor body-to-backpack radio reliability and reconnect recovery.
- Continuous reception with the Mac closed and awake on battery.
- Battery runtime, extended button/mode interactions, OTA fallback and return to detection.
- Sensor mounting/orientation and the existing detector's provisional training-count scale.

BLE troubleshooting update: encrypted model identification and the READY handshake
have succeeded on the actual M5. Two short setup attempts durably saved 11 and
40 samples respectively, but neither recording completed. Repeated supervision
timeouts and GATT discovery timeouts occurred before the SDK workaround below. These interrupted tests
must not be treated as complete training recordings. The Mac now serializes its
initial encrypted reads before notification subscription, avoids duplicate
pending reads, and reports readiness only after the device confirms it. The
collector radio keeps modem sleep disabled and requests a 30–45 ms interval,
zero slave latency and a six-second supervision timeout; a physical test showed
the request accepted (45 ms). Those changes alone did not establish sustained BLE
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
disable it using the vendor's workaround. The final image built successfully,
passed detector/model preservation verification, and was installed only at
`0x3E0000`; esptool verified the written image hash. Image SHA-256:
`b55d30874cc81e72171d50879a56a2cfaae227d2846c041631ac4f843c9a142c`.

With that workaround, five physical recordings completed and were saved on
the Mac: 759, 194, 108, 203 and 217 samples (1,481 total). All five quality
reports show no sequence/timestamp gaps, read errors, saturation or timing-gap
flags. The first recording spans 25.266667 seconds
at 29.9999996 Hz, with no sequence gaps, timestamp gaps, timing-gap flags,
sensor read errors or saturation. Its JSONL contains 759 records and its CSV
759 data rows; `completion.json` confirms 759 saved samples, no overflow, and
device status confirms zero pending samples. The device buttons stopped and
started sessions, and a button marker was retained. The first three are setup
recordings (`activity=setup_test`), not activity
ground truth for training. Further sessions
default to `activity=unspecified` until the operator supplies a truthful label.
The user unplugged USB while the M5 was idle. Its boot ID changed and the Mac
automatically reconnected. A subsequent battery-powered BLE recording saved all
217 samples (7.2 seconds at 30 Hz), with zero pending samples and a matching CSV
and completion file. Unplug USB before starting acquisition; a power-transition
reboot during acquisition would lose unsaved RAM samples. Long-duration outdoor
body-to-backpack and closed-lid reception still need physical checks.

Later out-of-range test: an interrupted session saved a contiguous 516-sample
prefix on the Mac before disconnection. The M5 subsequently reported SAVING /
DISCONNECTED beside the awake Mac. Receiver restart, cached direct connection,
broader identity-filtered scanning and one approved Mac Bluetooth refresh did
not restore the transfer. This session has no completion file and must not be
treated as a fully saved recording. The M5 has not been rebooted or reflashed;
its pending RAM samples remain subject to power loss.

A recovery candidate now retries inactive BLE advertising every two seconds on
the host event queue, reports advertising setup failures, and clears stale link
flags on a host reset without clearing the recording buffer. The Mac can also
discover the known M5 using a name-only advertisement. The firmware build,
embedded-model/source preservation check and existing replay/buffer/journal
smoke tests pass. This candidate is not installed or physically verified yet;
installation would reboot the M5 and lose any unsent RAM samples. The observed
failure's exact controller/host cause remains unconfirmed without device logs.
