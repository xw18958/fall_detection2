# V5 on-demand buzzer restoration — 4 October 2026

Firmware: `1.4.4-v5-power-buzzer`. USB inactive-slot staging and full image
readback passed; activation changes only the selector after protected-setting
verification. Startup verification is recorded in `BUZZER_DEPLOYMENT.json`. The device accepted
the buzzer settings, passed unchanged-model replay and startup health, and
reported zero sampling gaps and Wi-Fi on-time of zero in the short check.
Audible user confirmation remains pending.

## Cause and fix

The power build requested 2 kHz with 10-bit PWM on the 1 MHz REF_TICK clock.
That requires a divider below one; the driver rejects it and the old helper
returned silently. The corrected setting is 8-bit with duty 128 (50%).
[Espressif's LEDC documentation](https://docs.espressif.com/projects/esp-idf/en/v5.0/esp32/api-reference/peripherals/ledc.html)
documents REF_TICK's clock and the frequency/resolution relationship; the
installed ESP-IDF driver also checks that divider directly.

A dedicated driver initializes with output low and a paused timer, serializes
calls, holds the no-light-sleep lock only during a tone, and stops the output,
pauses the timer and releases the lock on completion and playback errors.
Initialization and playback failures are reported instead of silently ignored.
The existing 100 ms fall-alert beep, three-second alarm cooldown, and collector
feedback sounds are retained. No periodic demonstration tone was added.

## Preserved detector and collector

The model SHA remains
`4eb70382f3306a4c8580e5d08342032edae43d3bd4e07cb4a3a9090e85dfa367`.
30 Hz / 90 x 6, SI units, normalization, threshold and 750 ms full-model fallback
remain unchanged. Trigger gating and parallel convolution remain disabled.
The model/collector/partition guards and 20 host smoke tests passed, including
fault-injected actual buzzer code and Mac protocol/journal/trust checks.
IRAM margin passed at 12,292 bytes. The private-model preparation helper now
reproduces the current power firmware identity without reverting to the older
baseline firmware version.

## Publication

The publication branch starts at the latest GitHub main, `5f2efb8`, and adds the
current V5 power implementation, power/trigger training experiments and compact
reports. The Mac receiver now accepts the exact V5 model hash while retaining the three
previous hashes and rejecting arbitrary identities. Its UI/backend pins remain
protected, with the single authorized trust-file update recorded explicitly.
Newer V5 training and collector files on main are retained; obsolete
local copies are not used to revert them. Device backups, raw recordings,
models/replay inputs, generated firmware and private credentials remain local.

The user relaxed 250 ms as an immediate acceptance requirement. It remains a
future scheduling target; trigger data qualification is still unmet. Live OTA
exercises and prolonged reliability tests will not be performed, as requested.
No current/battery-life measurement or commercial qualification is claimed.
