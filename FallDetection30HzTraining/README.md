# 30 Hz Fall Detection Training

Compact six-axis IMU fall-detection training pipeline plus a reusable preprocessing pipeline for labelled fall recordings that contain surrounding non-fall activity.

## Current six-source V2 training

`train.py` now trains from private V2 fall containers and five processed public
datasets: CGU_BES, Cogent, SFU_IMU, UCI_SimulatedFalls and PAMAP2. Random
3-second crops inherit the complete container label, including non-fall motion;
training does not estimate fall timing again. SSL balances all six sources.
Fine-tuning assigns 50% to private V2 and 10% to each public source.

See [the training protocol](README_V2_TRAINING.md),
[public preprocessing](README_PUBLIC_V2.md), and
[the completed seed-42 report](runs/v2_multisource_seed42_20261001/REPORT.md).
The held-out private test detected all 26 falls, with 3 false positives among
25 negative recordings. Its scope is unseen recordings from the existing person.
Public tests hold out participant groups; no zero-shot claim is made.

## Repository structure

```text
FallDetection30HzTraining/
  preprocess_noisy_fall_dataset.py   # universal noisy-fall preprocessing
  train.py                            # 3 s / 90-sample detector training
  requirements.txt
  splits/
  results/
  experiments/
```

## V2 noisy-fall preprocessing

The raw dataset uses recording-level fall labels: a file under `fall/` means that the recording contains a fall, not that every timestep is a fall. Long fall recordings can therefore contain substantial normal activity before or after the true fall event.

`preprocess_noisy_fall_dataset.py` converts those noisy recording-level labels into cleaner event-level fall files while recovering the surrounding activity as hard negatives.

### Preprocessing rule

1. Resample every fall recording from `time_ms` to true 30 Hz.
2. Compute accelerometer and gyroscope magnitudes and find strong peaks that occur close together.
3. Define the fall-event center as the midpoint between the selected accelerometer and gyroscope peaks.
4. Normal case: extract a 5 s / 150-sample fall file with that midpoint at the center.
5. Boundary case: if the recording does not contain enough real data for a centered 5 s window, extract a real 3 s / 90-sample fall window instead.
6. Short case: if the full recording is shorter than 3 s after 30 Hz resampling, preserve every real sample and pad only the missing samples to 3 s with small IMU-like noise estimated from quiet non-fall recordings.
7. Every remaining real PRE/POST chunk of at least 3 s becomes a hard-negative file under `non-fall/`.
8. No guard band around the selected fall window is discarded.
9. Existing non-fall files are preserved unchanged.
10. `preprocessing_audit.csv` records the detected peaks, midpoint, output type, crop positions, padding, generated negatives, and duplicate groups.

The 5 s fall file is an offline clean event container. The commercial model still receives 3 s / 90-sample inputs; during later training, different 3 s views can be cropped from the 5 s container so the fall does not always occur at the same position inside the model input.

### Run preprocessing

```bash
python preprocess_noisy_fall_dataset.py \
  --input /path/to/30Hz_processed_clean_v1.zip \
  --output /path/to/30Hz_processed_clean_v2 \
  --zip-output /path/to/30Hz_processed_clean_v2.zip
```

Important defaults:

```text
sampling rate       = 30 Hz
clean fall window   = 5 s
model window        = 3 s
Acc/Gyro pair gap   = <= 1 s
padding seed        = 42
```

All defaults are configurable through command-line arguments so the preprocessing logic can be reused for other labelled IMU fall datasets with the same six-axis columns and `time_ms` timestamps.

### Current V1 -> V2 processing result

For `30Hz_processed_clean_v1.zip`:

| Output | Count |
| --- | ---: |
| Original fall recordings | 174 |
| Centered 5 s fall files | 100 |
| Boundary 3 s fall files | 65 |
| Padded 3 s fall files | 9 |
| Existing non-fall files preserved | 55 |
| New hard-negative files recovered | 120 |
| Final non-fall files | 175 |
| Duplicate fall files flagged in audit | 4 |

The input files are first resampled to true 30 Hz because the original timestamps are not uniformly 30 Hz. This matches the resampling policy already used by `train.py`.

## Detector training method

- Grouped approximately 70/15/15 splits before window generation: public
  participant groups and linked private source recordings remain together.
- True 30 Hz processing with random 3 s training crops (`90 x 6`); deterministic
  validation/test views use 0.75 s stride and at most 32 views per recording.
- Normalization fitted on training recordings only.
- SSL pretraining with instance contrastive + supervised contrastive objectives.
- Dual-stream accelerometer/gyroscope TCN with attention pooling.
- Fine-tuning with classification + supervised contrastive objectives;
  pseudo-phase supervision is disabled.
- Balanced supervised sampler and validation-only threshold selection.
- Compact C24 model: 40,561 parameters.

### Legacy V1 supervision note

The historical trainer and seed-42 results below were produced with the original V1 dataset. For V1, fall location is estimated inside each recording using an IMU impact score based on acceleration magnitude, gyroscope magnitude, and acceleration jerk. Windows close to that estimated event are positive and an ambiguity band is excluded. This is pseudo-event supervision.

The current trainer uses cleaned V2 containers directly. The V1 results below are historical and should not be presented as V2 results.

## Dataset layout

Input to the preprocessing script:

```text
30Hz_processed_clean_v1/
  fall/*.csv
  non-fall/*.csv
```

V2 output:

```text
30Hz_processed_clean_v2/
  fall/*.csv
  non-fall/*.csv
  preprocessing_audit.csv
  preprocessing_summary.json
```

The dataset itself is not committed to this repository.

## Newly collected M5 recordings

M5 collection CSVs contain untouched MPU6886 register counts and device
microsecond timestamps. They are not directly accepted by this trainer. See
[the M5 collected-data guide](../M5BLECollector/COLLECTED_DATA.md) for units,
quality/completion checks, labels, placement, grouping and the missing importer.
The collector's raw counts must not be assumed to have the historical private
dataset's count scale or normalization.

## Local training run

```bash
pip install -r requirements.txt
python -m unittest -v test_training_v2 test_preprocess_public_v2
python train.py --zip /path/to/30Hz_processed_clean_v2 \
  --public-root /path/to/processed_v2 --work runs/v2_multisource
python train.py --mode test --zip /path/to/30Hz_processed_clean_v2 \
  --public-root /path/to/processed_v2 --work runs/v2_multisource
```

Use `--smoke` with a separate work directory for a short end-to-end check.
The current trainer rejects V1 input without a V2 audit.

## Kaggle

The training script auto-detects a mounted V2 dataset containing `fall/` and `non-fall/`, writes outputs under `/kaggle/working/fall_detection_30hz`, and runs the full configuration by default. `kaggle/kernel-metadata.json` records the private kernel configuration used for the final V1 run.

Final V1 configuration: seed 42, C24, 20 SSL epochs, 3 head epochs, up to 17 full fine-tuning epochs with patience 5. Fine-tuning stopped early after epoch 13.

## Final seed-42 V1 results

Threshold was selected using validation recordings only (`0.83`).

| Level | Accuracy | Precision | Recall | Specificity | F2 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation recording | 0.9412 | 1.0000 | 0.9231 | 1.0000 | 0.9375 | 0.8593 |
| Test recording | 0.9143 | 1.0000 | 0.8846 | 1.0000 | 0.9055 | 0.8145 |
| Test window | 0.9940 | 0.8750 | 0.5600 | 0.9991 | 0.6034 | 0.6974 |

Test recording confusion matrix: TN=9, FP=0, FN=3, TP=23.

The exact split, normalization statistics, and full V1 metrics are committed under `splits/`, `normalization_seed42.json`, and `results/` for reproducibility. Model checkpoints are intentionally not committed; they remain available in the Kaggle run output.
