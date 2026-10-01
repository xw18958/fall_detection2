# M5 BLE body-motion collector

Collect six-axis MPU6886 motion at 30 Hz over Bluetooth Low Energy and save accepted trials locally on a Mac. Collection is local-first: samples stay in M5 PSRAM until the user stops the trial and explicitly chooses **KEEP**. **DISCARD** clears the local trial and sends no sample data to the Mac.

## Detector preservation

- The firmware embeds the same locked detector model and keeps DETECT and COLLECT on separate boots.
- Detector inference, normalization, threshold, kernels, sampling cadence and OTA behavior are not part of the collection workflow redesign.
- Collection mode records untouched signed 16-bit MPU6886 register counts. It does not perform physical-unit conversion, normalization or model preprocessing for saved datasets.

## Normal collection workflow

1. In `READY`, press **A** to start.
2. During `RECORDING`, perform the activity. The screen shows a recording timer. **B** and **C** are ignored.
3. Press **A** to stop.
4. In `REVIEW`, press **A = KEEP** or **B = DISCARD**.
5. KEEP enables BLE transfer. If the Mac is unavailable, the M5 waits with the complete trial buffered in PSRAM and resumes when the connection is ready.
6. DISCARD clears the trial locally. No sample data from that trial is transferred or saved on the Mac.

When the display is asleep, the first A/B/C press only wakes the display; that press performs no collection action.

The existing BLE pairing reset remains a maintenance/recovery operation and is still guarded so it cannot clear/reboot while recording or unsaved trial data exist.

## Saved dataset

Accepted trials produce `samples.csv` with only acquisition sequence, device acquisition timestamp and the six untouched sensor counts:

```text
seq,device_timestamp_us,ax,ay,az,gx,gy,gz
```

`ax, ay, az, gx, gy, gz` are the signed 16-bit values read from the MPU6886 registers. Conversion to g, degrees/second, normalization and all training preprocessing belong in downstream Mac/server code.

`host_received_timestamp_utc` is not exported to `samples.csv`; reception timing is transport metadata and remains only in the internal durable journal for debugging/recovery. Marker/event columns and `events.csv` are not part of the new collection dataset.

`quality_report.json` retains timing-gap, sensor-read, saturation and measured-rate diagnostics so quality metadata does not pollute the training CSV.

## Reliability model

The M5 has an 8,192-record PSRAM buffer (about 273 seconds at 30 Hz). A trial never streams sample data before KEEP. After KEEP, the existing acknowledged BLE transfer protocol remains responsible for reconnect replay and durable Mac writes. Unacknowledged samples remain on the M5 until the Mac has durably stored and acknowledged them.

Power loss/reboot still destroys samples that exist only in PSRAM, so wait for `SAVED` after KEEP before intentionally rebooting or switching back to DETECT.
