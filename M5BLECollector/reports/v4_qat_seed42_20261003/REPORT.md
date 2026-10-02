# V4 QAT development deployment

Commercial readiness has not been established. This is a development build. The M5 held-out set contains no falls, so it cannot establish M5 fall sensitivity. Its QAT negative-window false-positive rate is 8.04% (87/1,082), versus 6.19% (67/1,082) for float. Overlapping window fractions are not alarm rates per hour.

## Locked model and method

- Model SHA-256: `7a649721e02ab88e85594dd67efe415a0d5a777d9215df03d903dd0de748cf57`
- QAT checkpoint SHA-256: `d12ba0f0915859ae6ae4056b21d8ab2730142dfc5b3299cbaaabe71c98529866`
- Original immutable checkpoint SHA-256: `3c92a68f6e1440a83880665d6634b01e7cb5724374a7e3f7a3eb76892e210b2e`
- Threshold: `0.27264320850372314` (float baseline `0.2602156102657318`).
- Seed 42; same 90×6 input, 30 Hz, three-second window and all four temporal branches. Frozen BN statistics; train inference weights/affine parameters; source-weighted CE plus 2.0× teacher KL (T=2), AdamW 1e-5, batch 128. Training never uses validation/test samples.
- Training: 4,628 recordings, 49,536 random crops per epoch; all eligible recordings visited. Calibration: 2,049 training-only windows from 970 recordings across seven sources. Input observer range uses the 99.9 percentile; all other activation ranges use training minima/maxima.
- Selected epoch 5 of 9 completed epochs in the fused run; early-stopping patience 4, maximum 12. Ordinary QAT and full-range variants failed validation recall guards. Selection uses exhaustive actual Micro predictions, inherited per-source group guards and added original-float positive-window recall floors. Fused selected validation weighted error 0.0014317353521103677 versus float 0.0013422465811370773 (+6.67%). Test was evaluated only after the model/threshold lock.
- QAT is not automatically superior to PTQ. It improved private test F1 over the earlier PTQ (91.55% versus 88.70%), with the same 157/162 fall detections, but worsened M5 false-positive windows (87 versus 69). Selection was based on validation, never these test outcomes.
- Model graph: 145,880 bytes; 413 INT8 and 104 INT32 tensors. Per-channel convolution weights. Five LayerNormV4 operators retain the trained normalization formula and learned affine weights, using exact int64 sums and variance numerators with a float32 square root internally. Tensor interfaces are INT8; this is not an exclusively integer-arithmetic implementation. No float tensor fallback.
- All 128,225 validation/test windows evaluated by native TFLite Micro with ESP32 portable ESP-NN FC/softmax, custom convolution/GELU and fused normalization. Native hardware timer stubs are not device latency measurements.

## Three-second window metrics

Each row counts overlapping complete 3-second windows on the inherited cumulative 0.25-second grid. TP/FN denominators are positive windows; FP/TN denominators are negative windows. Percentages below are window metrics; negative-only recall/precision/F1 are undefined.

### Validation

|Source|Model|Windows|TP|FN|FP|TN|Precision %|Recall %|F1 %|Specificity %|Negative FPR %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|float|2862|128|10|3|2721|97.710|92.754|95.167|99.890|0.110|
|own|qat|2862|128|10|3|2721|97.710|92.754|95.167|99.890|0.110|
|m5_hard_negatives|float|1082|0|0|0|1082|—|—|—|100.000|0.000|
|m5_hard_negatives|qat|1082|0|0|0|1082|—|—|—|100.000|0.000|
|CGU_BES|float|955|72|0|0|883|100.000|100.000|100.000|100.000|0.000|
|CGU_BES|qat|955|72|0|0|883|100.000|100.000|100.000|100.000|0.000|
|Cogent|float|12171|629|1|1|11540|99.841|99.841|99.841|99.991|0.009|
|Cogent|qat|12171|629|1|2|11539|99.683|99.841|99.762|99.983|0.017|
|SFU_IMU|float|4509|362|7|40|4100|90.050|98.103|93.904|99.034|0.966|
|SFU_IMU|qat|4509|362|7|43|4097|89.383|98.103|93.540|98.961|1.039|
|UCI_SimulatedFalls|float|17714|2510|1|4|15199|99.841|99.960|99.900|99.974|0.026|
|UCI_SimulatedFalls|qat|17714|2511|0|5|15198|99.801|100.000|99.901|99.967|0.033|
|PAMAP2|float|22393|0|0|156|22237|—|—|—|99.303|0.697|
|PAMAP2|qat|22393|0|0|167|22226|—|—|—|99.254|0.746|

### Held-out test

|Source|Model|Windows|TP|FN|FP|TN|Precision %|Recall %|F1 %|Specificity %|Negative FPR %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|float|15616|158|4|19|15435|89.266|97.531|93.215|99.877|0.123|
|own|qat|15616|157|5|24|15430|86.740|96.914|91.545|99.845|0.155|
|m5_hard_negatives|float|1082|0|0|67|1015|—|—|—|93.808|6.192|
|m5_hard_negatives|qat|1082|0|0|87|995|—|—|—|91.959|8.041|
|CGU_BES|float|967|72|0|25|870|74.227|100.000|85.207|97.207|2.793|
|CGU_BES|qat|967|72|0|23|872|75.789|100.000|86.228|97.430|2.570|
|Cogent|float|10942|620|10|7|10305|98.884|98.413|98.648|99.932|0.068|
|Cogent|qat|10942|623|7|5|10307|99.204|98.889|99.046|99.952|0.048|
|SFU_IMU|float|4509|349|20|21|4119|94.324|94.580|94.452|99.493|0.507|
|SFU_IMU|qat|4509|350|19|18|4122|95.109|94.851|94.980|99.565|0.435|
|UCI_SimulatedFalls|float|16588|2177|10|0|14401|100.000|99.543|99.771|100.000|0.000|
|UCI_SimulatedFalls|qat|16588|2177|10|0|14401|100.000|99.543|99.771|100.000|0.000|
|PAMAP2|float|16835|0|0|111|16724|—|—|—|99.341|0.659|
|PAMAP2|qat|16835|0|0|121|16714|—|—|—|99.281|0.719|

## Input, replay and collector protection

- M5 DETECT input uses signed MPU6886 counts at ±8 g/±2000°/s converted to m/s²/rad/s and the original M5 training-only mean/std, without another epsilon. COLLECT still stores original counts.
- All 393,216 signed-count/channel host preprocessing cases agree byte-for-byte, including half-away-from-zero ties. Convolution tests: 594 bit-exact reference cases. Fused normalization: 30,060 rows versus an independent double-precision formula; zero differing elements in that test. Eleven native startup replay cases pass with exact host outputs; original four-LSB device startup tolerance is retained.
- Fourteen collector/preservation/USB tests, ASan/UBSan buffer/transport and UI tests, and Swift protocol/journal/recovery/identity tests passed. All collector acquisition, transport, journal, controls, UI/battery and mode-switch pins remain unchanged. Only the exact new model identity is added alongside both earlier trusted identities.
- Combined application retains startup hash/scale/preprocessing/replay validation in DETECT and COLLECT. Model/interpreter allocations are freed before COLLECT initializes BLE and its PSRAM recording buffer.
- Wireless deployment was cancelled by the user. The active firmware does not add a GitHub updater. Existing local recovery code is retained but not used to install this build.

## USB deployment and physical checks

See device verification report for final install state, measured latency, memory, recording checks and any required physical actions. Full 8 MB preinstall backup is private, SHA-256 `758e95c8636f5b7580921bee64b5151daa07b2885bb074c4fe4d5012bc244391`. Verified active slot ota_0 (sequence 3); target ota_1 at 0x3e0000, new selector sequence 4 in the older OTA metadata sector. Never erase_flash or generic upload. Full app readback must match before activating. Bootloader, partitions, NVS, PHY, prior valid selector and old app are preserved.

## Limits for commercial use

These seven-source retrospective splits and short M5 negative recordings do not establish target-user, target-placement, device-to-device or real-world fall performance. Private sensor calibration remains unresolved. New independently collected real-device fall and continuous ADL data, predeclared event-level sensitivity and false alarms per hour, calibration/placement checks and commercial dataset/software license checks remain necessary. No commercial release claim is made.

## Artifact locations

- Server immutable evaluated package: `/raid1/xwan0900/v4_qat_locked_seed42_20261003`.
- Server QAT run: `/raid1/xwan0900/v4_qat_fused_seed42_20261003`; rejected full-range comparison: `/raid1/xwan0900/v4_qat_fused_full_seed42_20261003`.
- Private local package/backup/build logs: `/Users/henrywang/Documents/fall_detection2/deployment_artifacts/v4_qat_seed42_20261003`.
- Original checkpoint and original data/experiments unchanged. Public commits contain source and compact aggregate reports only; no recordings, raw/calibration/replay inputs, model/firmware binaries, flash backups or keys.

## Measured device deployment

USB installation succeeded. The bootloader loaded ota_1 at 0x3e0000. Model/checkpoint identity, preprocessing goldens and replay passed, and the application marked the image VALID. Firmware SHA-256: `6884a0feb3c18c9faafc0cc5865dec63d7e835e77235aed931349176e0ea6fd4`. All 25 original recording/journal files are byte-identical. The deployed Mac receiver accepts the exact QAT hash plus both prior identities; the old application bundle is backed up privately.

128 observed inferences: median 451.296 ms, p95 459.942 ms, p99 466.286 ms, worst 469.297 ms. Input-to-probability: median 452.011 ms, p95 461.978 ms, p99 469.908 ms, worst 497.993 ms. Arena used 78,704 bytes in a 262,144-byte PSRAM allocation; preferred 112 KB internal allocation was unavailable. Free internal memory 135,615 bytes, free PSRAM 1,832,272 bytes at the timing report. The below-300-ms target was not achieved. Cadence remains 750 ms; 250 ms live updates are not enabled. Full path including first profiling/logging/UI overhead reached about 513.9 ms in the observed log.

NVS bytes changed between the full backup and preactivation readback due to old-app startup page compaction and PHY calibration refresh. CRC-verified logical Wi-Fi/BLE/mode settings match the backup; USB did not write any NVS sector. Bootloader, partition table, PHY partition and old OTA selector were byte-identical before activation. The previous ota_0 application remains as recovery. Only one older OTA selector sector was changed to sequence 4.

Live COLLECT checks passed after the user entered COLLECT with held B. A 420-sample trial reached REVIEW and DISCARD returned READY with no saved samples. A second 758-sample trial survived receiver disconnect/reconnect in REVIEW without resetting the device, then KEEP produced COMPLETE with device=758, journal/CSV=758 and pending=0. The schema-3 CSV has exactly sequences 0–757. Measured acquisition 29.9999992 Hz; no sequence gaps, timestamp gaps, read errors, timing-gap flags, saturation or overflow. Existing recording files remain unchanged. Mode exit was refused by the host while RECORDING; DETECT was requested only after COMPLETE/pending=0. The physical A/buttons/battery/wake flows are protected by unchanged pins and host UI tests; not every physical button action was manually repeated.

Final device mode: DETECT. Return-to-DETECT boot again verified the exact model/checkpoint and passed startup replay. No samples remain pending.
