# V5 whole-system power development

This directory targets the actual V5 detector/collector installed on the M5.
The older standalone project uses a different model and must not be flashed as
a replacement merely by changing its sample-rate constants.

The V5 model remains SHA256
`4eb70382f3306a4c8580e5d08342032edae43d3bd4e07cb4a3a9090e85dfa367`:
30 Hz, 90 × 6, SI units, frozen normalization, and threshold 0.6313204765319824.
`power_contract.json` and the build guard bind those files, collector behavior,
and the partition layout. Generic upload/erase is blocked; the USB tool stages
and verifies the inactive application slot before changing the OTA selector.

## System changes

- Rational 30 Hz sensing deadlines and continuous history, with gap detection.
- Normal active CPU clock 80 MHz; temporary 240 MHz full-model execution;
  compatible tickless idle/light sleep between work.
- DETECT display expires after 15 seconds. Alerts and buttons wake it.
  COLLECT retains recording/review/KEEP controls, with an idle display timeout.
- Normal Wi-Fi off. DETECT button A held for two seconds enables the existing
  HTTP OTA path for a bounded session. OTA completion/timeout shuts Wi-Fi down.
  Startup health validation does not require a network connection.
- Production logging summarizes timing and counters rather than every sample.
- Optional trained-trigger task uses a copied, immutable triggering window and
  queues that same W0 immediately. Awake windows continue every 250 ms, with a
  one-second quiet timeout. Persistent suspicion has bounded eight-second
  maintenance bursts without blind cooldowns.
- Trigger faults, queue overflow or missed cascade deadlines disable gating and
  explicitly retain the original 750 ms full-model fallback.

## On-demand buzzer

Firmware `1.4.4-v5-power-buzzer` restores the existing fall-alert and collector
feedback beeps. The dedicated 2 kHz PWM uses 8-bit resolution on the 1 MHz
REF_TICK clock; the earlier 10-bit setting required an impossible divider.
GPIO2 is driven low and the PWM timer is paused between tones. A sleep lock is
held only while sounding; tone calls are serialized and failures are reported.
The existing 100 ms fall beep and three-second alarm cooldown are retained.
No periodic or startup demonstration beep was added.

The original 250 ms full-model budget remains a future architecture target;
the user has relaxed it as a requirement for the current optimization stage.
The trigger remains disabled because its data qualification still fails.
Live OTA exercises and prolonged reliability tests are excluded at the user's
request. Current-meter and generalization limitations remain documented.

For a private rebuild, run `tools/prepare_model.py --source /path/to/verified/V5/package`
with the pinned `model.tflite`, `model_v2_config.h` and `model_v2_replay.h`.
The helper generates the current firmware identity from `power_contract.json`
and checks all hashes before copying. Then build the
`m5stickc-plus2-internal-benchmark` profile. Generated models and replay vectors
are not part of the public source tree.

## Memory benchmark

The `m5stickc-plus2-internal-benchmark` profile uses the supported split
MicroAllocator: 64 KiB internal activation memory and 128 KiB PSRAM metadata.
The ordinary profile retains the original PSRAM arena. Model weights and
arithmetic are identical. The split allocator passed byte-identical native
replay on 61,155 validation windows and 594 kernel cases. Hardware performance
must be measured separately; host timing cannot establish the 250 ms budget.

The pre-buzzer measurement build `1.4.3-v5-power-dev` passed device replay and startup health.
In DETECT, 128 measured inferences had median 401,864 µs, p95 402,880 µs,
and worst 403,179 µs; median input-to-probability time was 402,397 µs.
The prior median was 478,318 µs, giving about 16% latency reduction.
Across the two-minute device capture, sampling gaps remained zero, Wi-Fi time
remained zero, and display time stopped increasing after about 15 seconds.
The full model still exceeds the 250 ms budget. Cascade remains disabled and
the detector uses its explicit 750 ms fallback cadence.

The exact dual-core convolution benchmark also passed all 61,155 validation
outputs. On the M5 it measured 272,716 µs median, but still exceeded 250 ms and
had higher aggregate CPU duty than the single-core profile. It remains an
optional benchmark and is disabled in the final image; lower latency is not
proof of lower battery consumption.

## Qualification status

Trigger training and evaluation live in
`FallDetection30HzTraining/train_power_trigger_v5.py`. Splits and normalization
remain frozen. Evaluation distinguishes recording recall, timely frozen-V5
reference retention, continuous-normal false wake-ups/hour, and active fraction.
V5 reference events are diagnostic model runs, not annotated clinical fall times.

Current candidate experiments have not passed every held-out timely-reference
gate. Production trigger gating remains disabled. A small model, high recording
recall, or low active fraction alone does not qualify a trigger. Existing test
folds have been examined; independent validation remains necessary.

The additional robust-event training experiment selected `cnn24_mean_max`
on validation before auditing test. It woke on all 391 fall-labelled test
recordings, including all three M5 test falls, with 3.93% normal active fraction
and 76.66 false wake-ups/hour. It lost two timely frozen-V5 references and
reduced Cogent/own window recall, so it was not enabled in the final firmware.
Recording recall does not establish timely clinical fall detection.

Strict zero-loss validation recalibration selected `cnn24_mean_max` before
auditing test. It achieved 100% positive-recording trigger recall across the
tested sources and 3.35% continuous-normal active fraction, but missed one
frozen-V5 timely reference. It remains unqualified for gating. Earlier reports
and selection locks are retained; no test-based winner switching was performed.

The first `cnn16_mean` hardware shadow benchmark passed startup replay with
14,419 µs worst trigger latency at 80 MHz and used 3,500 bytes of its reserved
24 KiB arena. It ran over 470 trigger evaluations with zero sensing gaps.
That trial identified a wake-time-jitter bug in fallback scheduling. The fixed
deadline scheduler passes the hour-long cadence test; the original shadow
trial's CPU duty must not be presented as an energy improvement.

The isolated BLE KEEP smoke test saved and acknowledged 662 samples without
changing existing recordings. COLLECT idle measurements showed zero sampling
gaps, Wi-Fi off, and the display timing out. These are functional observations,
not measured current consumption or a battery-life claim. No external power
meter is available in this session.

Private binaries, device backups, calibration/replay inputs, and collector smoke
recordings are excluded from publication. Build and USB evidence is under
`deployment_artifacts/power_v5_20261004`.
