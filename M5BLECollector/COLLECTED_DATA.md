# Understanding and using M5 motion recordings

This guide describes **M5BLECollector schema 3**, checked against repository
`main` at `ec6c3798d00a51f64148d04cd612b8a0e89e1e42` on 2 October 2026.
It explains the acquisition files, their limitations and downstream training
requirements. It does not change recordings, firmware or preprocessing.

A recording contains six MPU6886 motion channels read by the M5 application
at a target rate of 30 Hz. The Mac saves the original signed integer readings.
A completed transfer does not establish that the movement was a fall, that every
sensor read was valid, or that the file is already in the trainer's input format.

## 1. Keep the recording folder together

Each session normally exposes two files, with a hidden recovery journal:

```text
session-folder/
  samples.csv
  metadata.json
  .recovery/
    journal.jsonl
```

`samples.csv` is the readable acquisition table. `metadata.json` describes the
sensor, session, user-supplied profile, quality summary and transfer completion.
The hidden journal preserves individual quality flags and supports recovery.
Migrated older sessions can also contain hidden backups of legacy sidecars.
At the recordings root, `.collector.lock` coordinates writers; it is not a
recording or a training feature.

Back up the **whole session folder**, including hidden files. The CSV and metadata
are sufficient for many uses when the quality summary reports no errors, but
only the journal identifies exactly which rows have flagged acquisition errors.
Do not move an unfinished session. A completed folder can be moved without
blocking a subsequent trial. Deleting that folder removes its saved data; an
already acknowledged recording is not automatically recoverable from the M5.

## 2. The exact CSV columns

The header is always:

```csv
seq,device_timestamp_us,ax,ay,az,gx,gy,gz
```

| Column | Meaning | Representation |
| --- | --- | --- |
| `seq` | Row sequence within this recording; first stored row is 0 | Integer; normally 0 through N−1 without gaps |
| `device_timestamp_us` | M5 monotonic time captured just before the IMU register read | Integer microseconds since device timer initialization |
| `ax`, `ay`, `az` | Accelerometer X/Y/Z register outputs | Signed 16-bit counts, −32768 through 32767 |
| `gx`, `gy`, `gz` | Gyroscope X/Y/Z register outputs | Signed 16-bit counts, −32768 through 32767 |

The axes are the MPU6886 register axes. The collector does not rotate them into
body, world or gravity-aligned coordinates. It does not estimate posture,
subtract gravity, remove bias, or normalize the six channels. No software
unit conversion, smoothing or resampling is applied in the collection path.
The sensor's configured internal digital filtering still applies: “raw” here
means untouched register values, not an unfiltered analog signal.

An accelerometer at rest ordinarily measures approximately one g in magnitude,
distributed across its axes according to orientation. Its readings are not
velocity or displacement. A gyroscope reports angular rate, not an angle.
Body placement and mounting orientation matter; the saved `placement` string
is supplied by the user and is not measured or verified by the firmware.

There are no activity, marker, quality-flag, physical-unit, host-time or label
columns in this CSV. A fall label cannot be inferred from the CSV filename,
KEEP decision, or embedded detector model.

## 3. Sensor settings and physical units

The current collection firmware configures the accelerometer for **±8 g** and
the gyroscope for **±2000 degrees/second**. The Mac validates these settings and
the firmware-declared conversion factors when connecting. Use the factors
stored in that recording's `metadata.json`, rather than assuming settings for
an unrelated device or an older dataset.

To reproduce the collector's declared nominal physical-unit convention:

```text
acceleration_g = accelerometer_count × device.accel_g_per_lsb
angular_rate_dps = gyroscope_count × device.gyro_dps_per_lsb
```

| Metadata field | Current value | Meaning |
| --- | --- | --- |
| `device.accel_range_g` | 8 | Nominal ±8 g range |
| `device.accel_g_per_lsb` | 0.000244140625 = 8 / 32768 | g per count; 4096 counts corresponds to 1 g |
| `device.gyro_range_dps` | 2000 | Nominal ±2000 degrees/second range |
| `device.gyro_dps_per_lsb` | 0.06103515625 = 2000 / 32768 | Degrees/second per count under the firmware convention |

These formulas match the [official M5Stack MPU6886 implementation](https://github.com/m5stack/M5Stack/blob/master/src/utility/MPU6886.cpp)
(`updateAres` and `updateGres`). They are nominal scale conversions, not a
per-device calibration. The [MPU6886 datasheet](https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/docs/datasheet/core/MPU-6886-000193%2Bv1.1_GHIC_en.pdf)
lists 4096 LSB/g for ±8 g and 16.4 LSB/(degrees/second) for ±2000 degrees/second.
Its rounded gyro sensitivity gives `count / 16.4`, slightly different from the
metadata factor `count × 2000 / 32768` (about 0.098% difference). Record which
convention you use; reproducing the existing detector requires its firmware
convention. Neither conversion removes sensor bias or establishes calibrated
accuracy.

For SI units, a derived dataset can multiply g by 9.80665 to obtain m/s² and
multiply degrees/second by π/180 to obtain rad/s. Preserve the source CSV and
write converted values to separate derived files with their units documented.
Raw counts can also be valid model inputs if training and deployment use the
same sensor settings, axis order and preprocessing; conversion is not mandatory
for every model.

## 4. Time and sampling rate

`device_timestamp_us` is based on [`esp_timer_get_time()`](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/api-reference/system/esp_timer.html).
It is not Unix time, Mac reception time, or time since pressing START. Multiple
sessions during one boot share the same device clock; a reboot starts a new
clock. `device.boot_id` distinguishes boots. The timestamp is taken immediately
before the read request; it is not an independently measured time of the
sensor's internal conversion.

For elapsed time relative to the first row:

```text
t_seconds[i] = (device_timestamp_us[i] − device_timestamp_us[0]) / 1,000,000
time_ms[i]   = (device_timestamp_us[i] − device_timestamp_us[0]) / 1,000
```

Use integer subtraction before floating-point conversion. Do not replace real
timestamps with `seq / 30` when inspecting timing quality. The target interval
is approximately 33,333.333 microseconds; integer scheduling, I2C reads and task
scheduling can introduce variation. The target 30 Hz is the application's
register-read rate, not a claim that the sensor internally operates at 30 Hz
or that all recorded intervals are perfectly uniform.

For N ≥ 2, the exported quality calculation is:

```text
elapsed_seconds = (last_timestamp − first_timestamp) / 1,000,000
measured_hz     = (N − 1) / elapsed_seconds
```

For zero or one stored row, the recorder reports zero elapsed time and measured
rate. For example, 90 perfectly spaced samples at 30 Hz span 89 intervals,
about 2.9667 seconds, although they form the detector's nominal three-second
input window. The screen's count-based duration `N / 30` can therefore differ
from the first-to-last timestamp span. Neither measures button-to-first-read
latency.

The journal's `host_received_timestamp_utc` is Mac wall-clock reception time
in seconds since the Unix epoch. Delayed transfer after KEEP and BLE retries
make it unsuitable as a substitute acquisition timestamp. Several rows from
one received batch can share that time. The current collector does not
synchronize M5 acquisition time to UTC; do not reconstruct absolute event time
from the session folder name or metadata creation time.

## 5. Metadata fields

| Field | Meaning and limitation |
| --- | --- |
| `schema_version` | Current visible-file schema is 3; check before interpreting older folders |
| `session_id` | Random nonzero 64-bit recording identifier serialized as 16 hexadecimal characters; not a person ID or label |
| `created_timestamp_utc` | Mac time when this session recorder was first created; often after KEEP, not acquisition START |
| `sample_columns` | Ordered CSV column names |
| `sample_value_note`, `timestamp_note` | Saved descriptions of raw values and device time |
| `profile.participant` | User-supplied participant identifier; defaults to `unspecified` |
| `profile.placement` | User-supplied body placement/orientation description; defaults to `unspecified` |
| `annotation.fall_label` | Initially JSON `null`; the collector does not infer a label |
| `device` | Device information snapshot described below |
| `quality` | Statistics derived from the saved journal rows |
| `completion` | Device-confirmed transfer accounting, described below |

The folder's date/time prefix is derived from Mac local time when a session
directory is created. It is a naming convenience, not an event timestamp.
A resumed session retains its earlier saved profile. Changing receiver options
does not automatically relabel an existing session.

The `device` object contains:

- `protocol`: BLE protocol version, currently 1.
- `device_id`: device Bluetooth MAC identity, encoded as 12 hexadecimal characters.
- `boot_id`: random nonzero 64-bit boot identifier, encoded as 16 hexadecimal
  characters; distinct from `session_id`.
- `firmware`: advertised firmware version string. It is not a firmware binary
  hash; historical builds can share a version string.
- `model_sha256`: hash of the embedded detector model, not of the recording
  or the entire firmware. It does not label motion or prove detector inference
  ran during acquisition; inference is paused in COLLECT.
- `rate_hz`: target acquisition rate, currently 30; compare with measured quality.
- `accel_range_g`, `gyro_range_dps`, `accel_g_per_lsb`, `gyro_dps_per_lsb`:
  the settings and nominal factors in section 3.
- `buffer_records`: capacity, currently 8192 records.
- `time_us`: device timer value at an information read; it is a snapshot and
  can predate the recording. It is not the first sample's timestamp.
- `last_error`: collector command-error status at that information read;
  it is not a per-sample IMU error count. Use `quality.read_errors` for that.

Suggested verified annotation values are `fall` and `non_fall`. The current
collector does not validate an annotation vocabulary or provide an event-marker
workflow. The trainer's directory name is `non-fall`, with a hyphen; the names
must be mapped explicitly by a future importer. Unknown labels must remain
unknown. KEEP means retain the trial, not “fall”; DISCARD means remove the trial,
not “non-fall”.

## 6. Acquisition quality: completion is not a quality verdict

Per-sample flags are retained as `sample.flags` in each JSON line of the hidden
journal; `sample.raw` contains the same six counts as the CSV. The flags are a
bit mask, so multiple conditions can occur on one row:

| Bit value | Flag | Current firmware condition |
| --- | --- | --- |
| 1 | Read error | IMU I2C read failed; the row retains its sequence/time and all six counts are zero placeholders |
| 2 | Timing gap | Sampling task was more than one nominal sample period late relative to its scheduled target |
| 4 | Saturated | At least one successfully read channel equals −32768 or 32767 |

Never interpret a flagged read-error row as six valid zero measurements. The
CSV alone cannot reliably identify such a row; use the journal, or exclude an
error-containing session when its journal is unavailable. A saturation flag
only detects exact signed endpoints. No flag is a guarantee that all impacts
were within range or that there was no near-limit clipping, bias or other error.

| `quality` field | Exact interpretation |
| --- | --- |
| `saved_samples` | Number of rows in the saved journal/CSV snapshot |
| `read_errors` | Count of rows with flag bit 1 |
| `saturated_samples` | Count of rows with flag bit 4, not the number of affected axes |
| `timing_gap_flags` | Count of rows with flag bit 2 |
| `observed_timestamp_gaps` | Number of adjacent intervals strictly greater than 1.5 / `device.rate_hz` seconds; currently greater than 50 ms |
| `elapsed_seconds` | Sum of adjacent timestamp intervals; equivalent to first-to-last span |
| `measured_hz` | (N−1) / elapsed seconds for N ≥ 2 and positive elapsed time |
| `interval_stddev_ms` | Population standard deviation of adjacent acquisition intervals, in milliseconds |
| `max_interval_ms` | Largest adjacent acquisition interval, in milliseconds; zero if none |
| `sequence_gaps` | Always written as 0 by this recorder, which rejects noncontiguous rows instead of saving them |

`timing_gap_flags` and `observed_timestamp_gaps` have different definitions and
need not match. Contiguous sequence numbers establish stored-row order; they
do not prove that the sampling task captured every intended time interval.
Flags can overlap, so do not sum their counts to obtain unique bad rows.
A mean rate close to 30 Hz can hide occasional pauses; also inspect the maximum
interval, interval variation and individual timestamps.

A conservative supervised-training pilot should use completed sessions with
no read errors, flagged endpoint saturation or unexplained timing gaps, enough
continuous real samples, known labels and known placement. This is a proposed
selection policy, not an automatic filter implemented by the collector. Preserve
excluded sessions and reasons for review. Advanced processing may retain valid
segments from flagged sessions by using the journal; do not silently fill long
pauses or failed-read placeholders and call them measured data.

## 7. What a saved recording guarantees

The workflow is RECORDING → REVIEW → KEEP → SAVING → COMPLETE. No sample records
are sent while RECORDING or REVIEW. DISCARD clears the local trial. KEEP enables
transfer; disconnects during recording do not stop local acquisition. After
KEEP, interrupted transfers can resume while the M5 remains powered.

Before acknowledging a batch, the Mac validates increasing timestamps and
contiguous sequences, writes and synchronizes the recovery journal. Identical
retransmissions do not add duplicate journal rows; conflicting duplicates or
gaps are rejected. CSV snapshots are regenerated on attachment/startup, about
every minute while unfinished, at completion and on orderly receiver exit.
An intermediate CSV may lag the journal.

`completion.complete: true` is written only after the Mac observes device
COMPLETE and checks its recovered/saved row count against the device's final
produced count. The `completion` object also contains `saved_samples`,
`device_buffer_overflowed` and `finished_timestamp_utc` for completed sessions.
An unfinished session starts with `complete: false` and `saved_samples`; the
other completion fields may be absent. The finished UTC timestamp is Mac
confirmation time, not the time of a fall or the last acquisition sample.

Check that CSV row count, `quality.saved_samples` and `completion.saved_samples`
agree for a completed original folder. COMPLETE means all rows the device
stored for that kept trial were accounted for; it does not guarantee that the
trial lasted as long as intended, had valid sensor readings, or was a fall.
`device_buffer_overflowed: true` means recording reached the full-buffer
condition and stopped. Its retained rows can still transfer completely.
A transferred empty recording is not a usable model input.

The 8192-record buffer occupies 256 KiB of PSRAM and holds roughly 273 seconds
at the target rate. Full-buffer handling stops recording rather than overwriting
older unsaved rows. PSRAM is volatile: reboot, power-off or battery loss destroys
pending samples. “SAMPLES SAFE” on the display means currently retained in RAM,
not saved persistently on the device. Mac ACKs precede the final CSV/completion
update, so after an interrupted receiver, preserve the journal and let the
receiver recover before judging a stale CSV. If the device loses power before
completion confirmation, already saved journal rows may remain usable partial
data, but the recording must not be declared complete without that confirmation.

## 8. Preparing data for fall-detection training

The existing trainer does **not** directly import M5 session folders. It expects
`time_ms,Acc_X,Acc_Y,Acc_Z,Gyro_X,Gyro_Y,Gyro_Z`, labelled dataset directories
and appropriate grouping/audit files. See the [V2 training protocol](../FallDetection30HzTraining/README_V2_TRAINING.md)
and [public preprocessing documentation](../FallDetection30HzTraining/README_PUBLIC_V2.md).
Renaming columns or placing `samples.csv` under `fall/` is not a complete or
validated import process. There is no finished M5-specific importer in this
version of the repo.

A downstream preparation pipeline should:

1. Preserve original folders. Record the source session/device/boot, conversion
   convention, processing version and every exclusion in a derived-data manifest.
2. Validate completion, row counts, sequences, timestamps, counts, configuration
   and quality. Use journal flags to identify individual failed/saturated reads.
3. Supply independently verified labels. For a long trial containing a fall,
   identify the event interval and prepare appropriate labelled clips; do not
   label every walking/standing window positive because the trial contains one
   fall. The current V2 trainer treats prepared fall containers as positive
   clips, so the importer must define their scope before training.
4. Specify body placement, orientation and participant identity. Defaults of
   `unspecified` are not known values. Do not guess them from sensor signals.
5. Choose one consistent representation: sensor counts, physical units, or the
   existing deployed detector's mapped input scale. Retain X/Y/Z channel order;
   any coordinate transform needs an explicit definition used at deployment too.
6. Convert device timestamps to relative time. If uniform 30 Hz data are needed,
   resample within verified valid segments, with a documented gap policy. Never
   bridge long missing intervals or failed reads without explicitly recording
   imputation. The collector itself does not perform this step.
7. Split by participant where independent-person evaluation is intended, and
   keep related sessions/crops/duplicates together. Split before generating
   overlapping windows. The present private V2 trainer groups source recordings
   and duplicates; it does not automatically consume M5 participant metadata.
   A future importer/trainer integration must preserve the desired grouping.
8. Form 90×6 windows for the current 30 Hz / nominal three-second detector.
   Record the handling of shorter clips; do not silently pad evaluation data or
   relabel missing observations as measurements. The historical `exp2_converged`
   pipeline uses 20 Hz / 60×6 and needs a different, explicit resampling path.
9. Fit any new normalization on training data only. Keep validation/test data
   out of that fit and export identical preprocessing for the deployed model.

### Reproducing the existing locked detector

M5 register counts are **not** already the locked detector's normalized input.
Its current path is implemented in `ReadImu`, `PushSample`, `Normalize` and
`FillModelInput` in [firmware/main/main.cc](firmware/main/main.cc):

```text
accelerometer mapped count = raw count × (8 / 32768) × 16384
                           = raw count × 4

gyroscope mapped count      = raw count × (2000 / 32768) × 16.4

mapped count is then clamped to [−32768, 32767]
normalized[channel] = (mapped count − mean[channel]) / (sigma[channel] + 1e−6)
```

The deployed INT8 model then uses its input scale/zero point, rounds and clamps
to int8. Exact numerical reproduction must follow the firmware's floating-point
and quantization operations, not just these explanatory real-number formulas.
The fixed mapping and normalization parameters are in [model_v2_config.h](firmware/main/model_v2_config.h).
That file explicitly marks the historical training-count scale as **inferred**;
it is not a calibrated physical-unit definition for every fall dataset. This
guide documents the current implementation, not proof that the historical unit
assumption is scientifically correct. Do not apply its stored means/stds directly
to M5 register counts, to arbitrary public SI signals, or twice to already
normalized data. Retraining on M5 data may instead use a new documented
representation and newly fitted training-only statistics.

## 9. Read-only inspection example

This standard-library Python snippet checks the visible CSV and reports the
saved metadata. Replace the argument with a session folder. It does not alter
files, read journal flags, convert units, assign labels, resample or train.

```bash
python3 - /path/to/session-folder <<'PY'
import csv, json, sys
from pathlib import Path

folder = Path(sys.argv[1])
metadata = json.loads((folder / "metadata.json").read_text())
assert metadata["schema_version"] == 3
with (folder / "samples.csv").open(newline="") as source:
    reader = csv.DictReader(source)
    assert reader.fieldnames == ["seq", "device_timestamp_us", "ax", "ay", "az", "gx", "gy", "gz"]
    rows = list(reader)
timestamps = []
for expected, row in enumerate(rows):
    assert int(row["seq"]) == expected
    timestamps.append(int(row["device_timestamp_us"]))
    assert timestamps[-1] >= 0
    assert all(-32768 <= int(row[c]) <= 32767 for c in ["ax", "ay", "az", "gx", "gy", "gz"])
assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
if metadata["completion"].get("complete") is True:
    assert len(rows) == metadata["completion"]["saved_samples"]
    assert len(rows) == metadata["quality"]["saved_samples"]
elapsed = (timestamps[-1] - timestamps[0]) / 1e6 if len(rows) > 1 else 0
print("Rows:", len(rows), "elapsed seconds:", elapsed)
print("Measured Hz:", (len(rows) - 1) / elapsed if elapsed > 0 else 0)
print("Profile:", metadata.get("profile"))
print("Label:", metadata.get("annotation", {}).get("fall_label"))
print("Completion:", metadata["completion"])
print("Quality:", metadata.get("quality"))
print("Inspection finished; this is not a training-readiness certificate.")
PY
```

## 10. Implementation references and scope

The authoritative code for the claims above is:

- [Firmware acquisition and detector preprocessing](firmware/main/main.cc):
  `InitMpu6886`, `ReadImuCounts`, `SensorTask`, `ReadImu`, `PushSample`.
- [Recording buffer](firmware/main/recording_buffer.h): state transitions,
  sequence assignment, flags, capacity, overflow and acknowledgment accounting.
- [BLE information and command handling](firmware/main/ble_collector.cc):
  exported device fields and transfer gating.
- [Mac wire decoder](macos/Sources/M5BLECollectorCore/Protocol.swift): supported
  configuration, signed count decoding and packet validation.
- [Mac recorder](macos/Sources/M5BLECollectorCore/Recorder.swift): journal,
  timestamp checks, metadata, quality calculations and completion verification.
- [Mac receiver](macos/Sources/M5BLECollector/main.swift): READY handshake,
  durable-write ACKs and device COMPLETE processing.
- [Current trainer](../FallDetection30HzTraining/training_v2.py): expected columns,
  grouping, resampling and training-only normalization.

The firmware collection behavior and output schema have not been changed for
this explanation. Transport checks, nominal scale factors and host tests do
not replace physical sampling, calibration, placement or label validation.
