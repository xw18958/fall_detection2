# V4 seed42 INT8 collector development candidate

Status: export, exhaustive evaluation, native Micro/ESP-NN checks and combined firmware/Mac builds complete. **Not installed; performance acceptance is incomplete.** No USB serial device was present. Original checkpoint, experiment, dirty Mac checkout and recordings were not modified. No retraining or additional seeds were run.

The same locked model and float threshold were fixed before final testing. No calibration/model/threshold candidate was selected using the held-out test. INT8 has measurable degradation; it is not a performance-preserving acceptance result.

## Held-out classification: 3-second windows

All 66,539 complete windows use the inherited holdouts, cumulative 0.25-second grid plus final complete window, source-specific stored normalization and no gap crossing or caps. Windows overlap and are correlated. Fall is class 1; predicted fall means two-class softmax probability >= 0.2602156102657318.

Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), specificity=TN/(TN+FP), negative-window FPR=FP/(FP+TN). Percentages below use those denominators; negative-only fall metrics are N/A. Per-source rows precede pooled results because pooled accuracy is dominated by negative windows.

| Source | Runtime | Windows | TP | FN | FP | TN | Precision % | Recall % | F1 % | Specificity % | Negative-window FPR % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| own | PyTorch float | 15616 | 158 | 4 | 19 | 15435 | 89.26554 | 97.53086 | 93.21534 | 99.87705 | 0.12295 |
| own | Desktop INT8 | 15616 | 157 | 5 | 33 | 15421 | 82.63158 | 96.91358 | 89.20455 | 99.78646 | 0.21354 |
| own | Micro / ESP-NN INT8 | 15616 | 157 | 5 | 35 | 15419 | 81.77083 | 96.91358 | 88.70056 | 99.77352 | 0.22648 |
| m5_hard_negatives | PyTorch float | 1082 | 0 | 0 | 67 | 1015 | N/A | N/A | N/A | 93.80776 | 6.19224 |
| m5_hard_negatives | Desktop INT8 | 1082 | 0 | 0 | 67 | 1015 | N/A | N/A | N/A | 93.80776 | 6.19224 |
| m5_hard_negatives | Micro / ESP-NN INT8 | 1082 | 0 | 0 | 69 | 1013 | N/A | N/A | N/A | 93.62292 | 6.37708 |
| CGU_BES | PyTorch float | 967 | 72 | 0 | 25 | 870 | 74.22680 | 100.00000 | 85.20710 | 97.20670 | 2.79330 |
| CGU_BES | Desktop INT8 | 967 | 72 | 0 | 34 | 861 | 67.92453 | 100.00000 | 80.89888 | 96.20112 | 3.79888 |
| CGU_BES | Micro / ESP-NN INT8 | 967 | 72 | 0 | 35 | 860 | 67.28972 | 100.00000 | 80.44693 | 96.08939 | 3.91061 |
| Cogent | PyTorch float | 10942 | 620 | 10 | 7 | 10305 | 98.88357 | 98.41270 | 98.64757 | 99.93212 | 0.06788 |
| Cogent | Desktop INT8 | 10942 | 622 | 8 | 6 | 10306 | 99.04459 | 98.73016 | 98.88712 | 99.94182 | 0.05818 |
| Cogent | Micro / ESP-NN INT8 | 10942 | 621 | 9 | 4 | 10308 | 99.36000 | 98.57143 | 98.96414 | 99.96121 | 0.03879 |
| SFU_IMU | PyTorch float | 4509 | 349 | 20 | 21 | 4119 | 94.32432 | 94.57995 | 94.45196 | 99.49275 | 0.50725 |
| SFU_IMU | Desktop INT8 | 4509 | 347 | 22 | 28 | 4112 | 92.53333 | 94.03794 | 93.27957 | 99.32367 | 0.67633 |
| SFU_IMU | Micro / ESP-NN INT8 | 4509 | 346 | 23 | 25 | 4115 | 93.26146 | 93.76694 | 93.51351 | 99.39614 | 0.60386 |
| UCI_SimulatedFalls | PyTorch float | 16588 | 2177 | 10 | 0 | 14401 | 100.00000 | 99.54275 | 99.77085 | 100.00000 | 0.00000 |
| UCI_SimulatedFalls | Desktop INT8 | 16588 | 2177 | 10 | 1 | 14400 | 99.95409 | 99.54275 | 99.74800 | 99.99306 | 0.00694 |
| UCI_SimulatedFalls | Micro / ESP-NN INT8 | 16588 | 2177 | 10 | 0 | 14401 | 100.00000 | 99.54275 | 99.77085 | 100.00000 | 0.00000 |
| PAMAP2 | PyTorch float | 16835 | 0 | 0 | 111 | 16724 | N/A | N/A | N/A | 99.34066 | 0.65934 |
| PAMAP2 | Desktop INT8 | 16835 | 0 | 0 | 103 | 16732 | N/A | N/A | N/A | 99.38818 | 0.61182 |
| PAMAP2 | Micro / ESP-NN INT8 | 16835 | 0 | 0 | 109 | 16726 | N/A | N/A | N/A | 99.35254 | 0.64746 |
| pooled | PyTorch float | 66539 | 3376 | 44 | 250 | 62869 | 93.10535 | 98.71345 | 95.82742 | 99.60392 | 0.39608 |
| pooled | Desktop INT8 | 66539 | 3375 | 45 | 272 | 62847 | 92.54182 | 98.68421 | 95.51436 | 99.56907 | 0.43093 |
| pooled | Micro / ESP-NN INT8 | 66539 | 3373 | 47 | 277 | 62842 | 92.41096 | 98.62573 | 95.41726 | 99.56115 | 0.43885 |

Micro relative to new float: Private TP158→157, FN4→5, FP19→35, TN15435→15419; recall 97.53086%→96.91358% (−0.61728 percentage points), F1 93.21534%→88.70056% (−4.51477 pp), negative-window FPR 0.12295%→0.22648%. M5 FP67→69 of 1,082 negative windows (6.19224%→6.37708%). Pooled TP3376→3373, FN44→47, FP250→277, TN62869→62842. Desktop is not a substitute for these Micro results.

## Export and calibration

Checkpoint SHA-256: `3c92a68f6e1440a83880665d6634b01e7cb5724374a7e3f7a3eb76892e210b2e`. Model SHA-256: `ddac71cffd5e10f85d9e9122fe5334a3dbabf3717da41b598231ef4a69a720bc`. Model bytes: 148,176. Split fingerprint: `b800373b861e340976fd77c441acb12061a641c8e4d6af969f16e46e66177673`. Normalization SHA-256: `f5f6561dc6dd5a659ce6bef0c51b5c151323cece762840656e2ec103d3332937`.

PyTorch→TF maximum logit error 2.14576721e-06; PyTorch→float TFLite 2.86102295e-06. Frozen BatchNorm folded exactly; full 90-sample and all three 30-sample branches retained. Encoder 24 channels, three blocks; no architectural change.

Calibration uses 2049 training-only windows from 970 recordings, seed 42. Rounding the source quotas yields 2,049 windows from nominal 2,048. Own and M5 each 512; each public source 205. Fall-capable sources balance classes and groups. M5 includes each training session and per-channel absolute-peak windows, with full SI range and no count clipping. Calibration manifest stays outside Git; its hash and source/class coverage are in export_lock.json.

INT8 input scale/zero point: [0.34698188304901123, 9]; logits output: [0.07270710915327072, -5]. Per-channel convolution weights (72 convolution nodes, including single-channel score heads). Tensor inventory: {'int8': 484, 'int32': 162}; no float tensors or fallback. Operator counts: `{"ADD": 29, "CONCATENATION": 4, "CONV_2D": 72, "FULLY_CONNECTED": 7, "GELU": 69, "MEAN": 10, "MUL": 14, "PACK": 1, "RESHAPE": 154, "RSQRT": 5, "SOFTMAX": 9, "SQUARED_DIFFERENCE": 5, "STRIDED_SLICE": 11, "SUB": 5, "SUM": 9}`.

No validation/test input values saturated the INT8 input range. Per-source logit/probability mean, p99 and worst errors and saturation counts are in test_report.json. Calibration ranges and p99 magnitude by source/channel are in calibration_coverage.json. The common activation input step is approximately 0.347 normalized units; training-source extrema widen that range. Integer attention/layer-normalization and accumulated kernel rounding remain precision-sensitive. These observations diagnose limitations; no new calibration variant was chosen on test results.

## Validation and runtime diagnostics

61,686 validation windows were evaluated. M5 remained FP0/1082 in float, desktop and Micro. The unchanged-threshold desktop seven-source group-macro weighted error is 0.0017077905360709201 versus float 0.0013422465811370773. These are not pooled window error rates.

Inherited guard failures for desktop INT8: SFU negative-recording specificity 0.8589743589743593 below 0.8646153846153848; UCI 0.9749520521672421 below 0.98. The earlier unconstrained validation diagnostic optimum 0.46371036767959595 was never adopted or tested. The deployed-candidate threshold remains unchanged; no post-test adjustment was made.

The optimized convolution matches Micro reference bit-for-bit on 594 randomized shape/dilation/stride/bias/activation cases. Tensor tracing first found a desktop-versus-Micro difference at operator 9 (CONV_2D): two of 720 values differ by one LSB; differences propagate through later operations. A single-rounding diagnostic did not eliminate the discrepancy; firmware/custom kernels remain unchanged. This is not evidence of complete desktop/Micro numerical parity.

Portable Micro and the ESP32 firmware’s generic ESP-NN FC/softmax paths produced identical output bytes on every validation and test window (128,225 windows). Native ESP-NN arena use is 101,600 bytes; portable reference 101,376. Neither is a measured ESP32 arena/free-heap result. Native bulk elapsed time is host timing, never a device-speed claim.

Eleven startup replay cases cover training calibration examples, zero counts, signed range extremes, alternating extremes and four joint-validation windows closest to threshold. Expectations are pinned to the verified Micro/ESP-NN path; desktop differences are retained in replay_manifest.json. The prior 4-LSB startup tolerance remains unchanged. Five count-vector goldens and all 393,216 signed-count/channel combinations verify raw counts→float32 SI→stored M5 normalization without a second epsilon→half-away rounding→saturated INT8 bytes. Native replay verifies logits; runtime_comparison.json verifies probabilities and decisions.

## Collector preservation and hardware status

All ten Python guard tests, sanitizer-enabled C++ recording-buffer/wire and battery/display tests, Swift protocol/journal/recovery/deduplication/completed-count tests, old/new trusted-identity tests, and the receiver build passed. Collector SensorTask, ReadImuCounts, range/filter initialization, BLE backend, recording buffer, Recorder/Protocol, button/wake/sleep regions, battery UI and recovery OTA source remain pinned to their existing hashes. Only documented detector/preprocessing/model-package, receiver identity and additive test pins changed. Generic upload/erase guards remain enabled.

Both boot modes retain startup model initialization/replay; COLLECT frees interpreter/arena before BLE and PSRAM recording-buffer allocation. No inference or SI conversion was added to COLLECT. Saved CSV schema 3, original signed counts, KEEP/DISCARD, durable journal ACK, retransmission/deduplication and count-agreement completion are unchanged. Existing old-model trust and recovery model/boot/session identity checks remain strict.

Generated sdkconfig verified: ESP32, 240 MHz, performance optimization, 8 MB flash, PSRAM enabled, CONFIG_BT_NIMBLE_HS_FLOW_CTRL disabled. Native arena fits the nominal 112 KiB internal-arena size, but actual device internal allocation and PSRAM fallback need physical confirmation.

The previous IDF embedding ALIGN argument was ignored, leaving the model unaligned. Explicit .balign 16 assembly now embeds the exact model once, without a second RAM copy. ELF alignment/extent and application size checks pass; config, replay, model identities and actual input/output scales are verified at startup, with rollback on startup failure.

Application-only firmware bytes: 1,470,272, below 0x3d0000. Firmware SHA-256: `25ab46031432dd36673707277034f2a0defe9d85853b55fb65b8648390929f38`. Embedded model start `0x3f41d760`; extent 148,176 bytes. Bootloader/partitions/NVS were not written.

Pending physical acceptance: identify installed firmware/transport and active OTA slot; finish/save any acquisition and establish pending=0; back up installed image/boot state/settings; application-only OTA to the inactive slot; measure median/p95/p99/worst invoke, input-to-probability and complete-path times plus arena/free-memory; verify 30 Hz raw capture, KEEP/transfer/COMPLETE count agreement, DISCARD/reconnect/deduplication, mode exits/buttons/wake/battery and rollback. The firmware retains 750 ms cadence until hardware measurements justify a change. Neither <300 ms inference nor a 250 ms complete path is claimed.

## Retrieval and zero-shot scope

| Check | Result |
|---|---:|
| Float binary-class retrieval P@1 (51 test query recordings, 245 gallery recordings) | 1.0 |
| Float hit rate@5 | 1.0 |
| Float mAP | 0.9999746819042967 |
| INT8 retrieval | N/A: classifier graph exports logits, not retrieval embeddings |
| Zero-shot | N/A: all seven sources contribute training data |

## Reproduction and locations

See REPRODUCE.md. Exact private deployment package, replays, firmware ELF/bin and build logs are delivered outside Git. Compact reports/source are on codex/v4-int8-collector. Server export: `/raid1/xwan0900/v4_int8_seed42_20261003`; server source: `/raid1/xwan0900/fall_detection2_v4_quantization`; Mac isolated checkout: `/private/tmp/fall_detection2-v4-quantization`; durable delivery: `/Users/henrywang/Documents/fall_detection2/deployment_artifacts/v4_seed42_20261003`. Raw recording folders and hidden journals were untouched.
