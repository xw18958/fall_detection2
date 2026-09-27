# 30 Hz Fall Detection Training

Compact six-axis IMU fall-detection training pipeline for the processed 30 Hz dataset.

## Method

- Recording-level 70/15/15 stratified split before any window generation.
- True 30 Hz processing with 3 s windows (`90 x 6`) and 0.75 s stride.
- Normalization fitted on training recordings only.
- SSL pretraining with instance contrastive + supervised contrastive objectives.
- Dual-stream accelerometer/gyroscope TCN with attention pooling.
- Fine-tuning with classification + supervised contrastive + phase objectives.
- Balanced supervised sampler and validation-only threshold selection.
- Compact C24 model: 40,561 parameters.

The dataset provides recording-level labels only. For supervised window training, fall location is therefore estimated inside each fall recording using an IMU impact score (accelerometer magnitude, gyroscope magnitude, and acceleration jerk). Windows close to that estimated event are positive; an ambiguity band is excluded from training. This is pseudo-event supervision, not manually annotated fall timing.

## Dataset

Expected layout:

```text
30Hz_processed_clean_v1/
  fall/*.csv
  non-fall/*.csv
```

The dataset is not committed to this repository. The experiment used 229 recordings: 174 fall and 55 non-fall.

## Local run

```bash
pip install -r requirements.txt
python train.py --zip /path/to/30Hz_processed_clean_v1.zip --work run
```

Short smoke test:

```bash
python train.py --zip /path/to/30Hz_processed_clean_v1.zip --work smoke_run --smoke
```

## Kaggle

The script auto-detects a mounted Kaggle dataset containing `fall/` and `non-fall/`, writes outputs under `/kaggle/working/fall_detection_30hz`, and runs the full configuration by default. `kaggle/kernel-metadata.json` records the private kernel configuration used for the final run.

Final configuration: seed 42, C24, 20 SSL epochs, 3 head epochs, up to 17 full fine-tuning epochs with patience 5. Fine-tuning stopped early after epoch 13.

## Final seed-42 results

Threshold was selected using validation recordings only (`0.83`).

| Level | Accuracy | Precision | Recall | Specificity | F2 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation recording | 0.9412 | 1.0000 | 0.9231 | 1.0000 | 0.9375 | 0.8593 |
| Test recording | 0.9143 | 1.0000 | 0.8846 | 1.0000 | 0.9055 | 0.8145 |
| Test window | 0.9940 | 0.8750 | 0.5600 | 0.9991 | 0.6034 | 0.6974 |

Test recording confusion matrix: TN=9, FP=0, FN=3, TP=23.

The exact split, normalization statistics, and full metrics are committed under `splits/`, `normalization_seed42.json`, and `results/` for reproducibility. Model checkpoints are intentionally not committed; they remain available in the Kaggle run output.
