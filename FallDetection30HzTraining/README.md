# 30 Hz Fall Detection Training

Training pipeline for the processed 30 Hz six-axis IMU fall-detection dataset.

## Pipeline

- Recording-level 70/15/15 stratified split before window generation.
- 30 Hz end-to-end processing.
- 3 s windows (90 samples) with 0.75 s stride.
- Train-only normalization.
- SSL pretraining with instance contrastive and supervised contrastive losses.
- Dual-stream accelerometer/gyroscope TCN encoder.
- Fine-tuning with classification, supervised contrastive, and phase losses.
- Validation-only threshold selection; final reporting on the held-out test split.

## Dataset

Expected ZIP layout:

```text
30Hz_processed_clean_v1/
  fall/*.csv
  non-fall/*.csv
```

The dataset is intentionally not committed to this repository.

## Run

```bash
python train.py --zip /path/to/30Hz_processed_clean_v1.zip --work run
```

For a short end-to-end sanity check:

```bash
python train.py --zip /path/to/30Hz_processed_clean_v1.zip --work smoke_run --smoke
```

Default model width is 24 channels and is designed to remain small enough for later pruning/quantization and embedded deployment experiments.
