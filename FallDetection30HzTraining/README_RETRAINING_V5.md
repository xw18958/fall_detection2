# V5: combined M5 fall and hard-negative adaptation

The new input contains 19 M5 recordings, each confirmed by the user to contain
one fall. Use the existing `preprocess_noisy_fall_dataset.detect_event` paired
acceleration/gyroscope midpoint and the existing 5-second centered crop / real
3-second boundary crop. Do not pad these recordings or create timestep/phase
pseudo-labels. Every 3-second view inherits its fall-container label. Preserve
the full original recordings, metadata and recovery journals privately.

The existing seven full-range SI M5 negative sessions are copied byte-for-byte,
with their original boot splits. Historical artificially clipped conversion is
excluded. Both M5 classes use one source (the legacy internal name
`m5_hard_negatives` is retained for compatibility) and one train-only M5
normalization. Counts convert using documented 8 g and 2000 deg/s ranges into
m/s² and rad/s. COLLECT continues to save raw counts.

All new positives share a single device boot; participant and placement are
unknown. Seed42 assigns 13 whole recordings to train, 3 to validation and 3 to
test. Descendants stay with their source recording. This is an exploratory
within-boot holdout, **not** independent participant or boot validation.
Original six-source holdouts and the negative boot holdouts are unchanged;
these existing benchmarks have been inspected in earlier experiments.

Source masses remain Private 25%, M5 25%, and each of five public sources 10%.
The existing source/class/group coverage sampler visits every eligible
recording every epoch. With both M5 classes, expected fall draws rise from
32.5% to 45%; repeated crops are not independent examples.

Start from the original six-source SSL checkpoint, preserve original source
normalization, refit M5 normalization from combined training recordings only,
and repeat the V4 SSL/head/full schedule: up to 30/3/50 epochs, early stopping
and seed42. An end-to-end GPU train/test/retrieval smoke must pass first.
The immutable prior V4 float checkpoint supplies the validation comparison
using its original normalization.

Before training, declare the existing private/public validation guards, an M5
recording recall floor of 90%, a positive-window recall floor of 90%, and a
negative-window FPR ceiling of 1%. These exploratory floors do not establish
commercial acceptability. Keep failed checkpoints and report their failure;
do not relax the gates using test results.

PTQ calibrates using 2,049 reproducible training-only windows across all seven
sources, covering both M5 classes and hard motion. Verify PyTorch → TensorFlow
→ float TFLite parity, no float tensor fallback, per-channel convolution
weights, signed-count conversion and half-away rounding. Evaluate INT8 using
the actual native TFLite Micro resolver/custom kernels.

Compare PTQ and the established fused-normalization QAT recipe under identical
validation guards and float32 threshold selection. QAT uses seed42, frozen BN
statistics, AdamW 1e-5, batch128, at most12 epochs / patience4, weighted CE plus
2× teacher KL at temperature2 and training-only observers. Its INT8 graph uses
`LayerNormV4`, whose kernel computes int64 statistics and a float32 square root.
Report this internal floating arithmetic explicitly.

Select the eligible candidate with the lowest validation source-weighted
error, preferring PTQ on exact ties. Save the checksum, threshold and selection
lock **before** final tests. Held-out evaluation uses fixed thresholds for
float/PTQ/QAT comparisons and cannot change that choice. If no candidate
qualifies, refuse deployment and defer the production candidates' full final-test
evaluation. The separate pipeline smoke checkpoint exercises a small fixed
test subset solely to verify execution, not to select a production model.

Report all complete 3-second windows on the cumulative 0.25-second grid plus
the last window, without crossing gaps. Include each source's TP/FN/FP/TN,
precision, recall, F1, specificity and negative-window FPR. Separately report
fall-recording detections and merged offline false-positive episodes per hour;
these are not firmware alarm/debounce replay. Include private class retrieval;
no zero-shot task is claimed because all seven sources contribute training.

Example server commands, from `FallDetection30HzTraining`:

```sh
/raid1/xwan0900/venvs/ftkp_cu128/bin/python run_v5_training.py --phase smoke
/raid1/xwan0900/venvs/ftkp_cu128/bin/python run_v5_training.py --phase train
CUDA_VISIBLE_DEVICES=0 /raid1/xwan0900/venvs/fall_v2_export/bin/python run_v5_quantization.py
```

Data: `/raid1/xwan0900/fall_detection2/fd_datasets/M5_combined_v5_20261003`.
Private outputs: `/raid1/xwan0900/v5_m5_seed42_20261003`.
Publish source and compact aggregate reports only. Keep raw data, calibration,
replay inputs, weights and firmware binaries outside Git. No device reset or
installation is part of these training/quantization commands.
