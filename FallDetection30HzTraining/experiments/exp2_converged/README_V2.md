# Exp2 V2: commercial-device generalization protocol

This V2 keeps the existing Exp2 model/loss family but changes the experimental protocol around the six datasets.

## Fixed assumptions

- Actual model sampling rate: **20 Hz** (the historical folder/repo name still says `30Hz`).
- Input: 6-axis IMU `[Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z]`.
- Window: 3 s = 60 samples at 20 Hz.
- Stride: 0.75 s = 15 samples.
- Own dataset is from one person and is the closest deployment domain because it uses the real commercial device.

## Own split

Subject-disjoint splitting is impossible because Own contains one person. V2 therefore uses **whole-recording separation** and optimizes assignment by resulting window count:

- TRAIN ~70%
- VAL ~15%
- TEST ~15%

Windows are generated only after recordings are assigned. A recording can never appear in multiple splits.

Own TEST is explicitly a **same-person unseen-recording/device-domain test**, not an unseen-subject test.

## Public datasets

- CGU_BES
- Cogent
- SFU_IMU
- UCI_SimulatedFalls
- PAMAP2

The four labelled fall datasets use participant-disjoint TRAIN/VAL splits. Approximately 10% of participants (minimum 2 when possible) are held out for public validation. PAMAP2 is used for SSL training and not binary fall validation.

### Required participant-aware public cache schema

For each public dataset, attach `<dataset>_v2.npz` (preferred) or `<dataset>.npz` containing:

- `x`: `[N, 60, 6]` float windows at 20 Hz
- `subject_id`: `[N]`
- for CGU/Cogent/SFU/UCI: `y` or `label`, binary fall/non-fall labels

V2 deliberately fails if `subject_id` is missing. It will not silently create random-window public validation because that would not measure unseen-person generalization.

## Training stages

1. **Hybrid contrastive pretraining**
   - 50% Own TRAIN windows
   - 50% public TRAIN windows combined across all five public datasets
   - checkpoints every 1,000 steps up to 5,000 steps
   - checkpoint quality assessed by a frozen-representation/head probe, not SSL loss alone

2. **Public supervised stage**
   - CGU + Cogent + SFU + UCI TRAIN
   - source-balanced batches so the largest public dataset cannot dominate

3. **Final deployment-oriented fine-tuning**
   - ~85% Own TRAIN
   - ~15% replay from the four labelled public fall datasets
   - preserves commercial-device specialization while reducing catastrophic loss of cross-domain robustness

## Checkpoint selection

V2 does not give all domains equal importance.

For every candidate checkpoint:

- Own VAL is evaluated at recording level using max window probability.
- Public validation is evaluated separately per dataset.
- First find the best Own VAL AP.
- Keep checkpoints within 0.02 AP of the best Own result.
- Among those, choose the checkpoint with the best mean public-validation AP.

Thus public data can break ties between similarly strong deployment-domain models, but cannot select a checkpoint that materially sacrifices Own performance.

## Threshold calibration

After the final checkpoint is selected, tune the operating threshold **once on Own VAL only**, using the existing minimum-recall / MCC / F2 priority.

Threshold calibration is intentionally separate from checkpoint selection.

## Final test isolation

Training mode never evaluates Own TEST.

Run training:

```bash
python train_exp2_v2.py --mode train
```

Only after the model and threshold are locked, run:

```bash
python train_exp2_v2.py --mode final_test
```

The final-test command writes `final_test_metrics.json`.

## Expected outputs

- `run_config.json`
- `splits/own_seed42.json`
- `data_audit.json`
- `normalization.json`
- `checkpoint_selection.json`
- `threshold.json`
- `training_history.json`
- `locked_model.pt`
- `final_test_metrics.json` (only after final-test mode)

## Historical Exp2

`train_exp2_converged.py` and its existing results are preserved unchanged for reproducibility. V2 is a new experimental protocol and should not overwrite the historical result.