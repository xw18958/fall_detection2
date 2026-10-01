# Public dataset V2 preprocessing

`preprocess_public_v2.py` reads the original archives directly and produces six-axis,
30 Hz CSVs using the V2 5-second fall-container / 3-second boundary-window policy.
It selects chest/sternum sensors consistently across the five datasets. Raw archives
and the existing private V2 dataset are preserved.

Install Python dependencies with `pip install -r requirements.txt`. The two RAR
datasets also require system `libarchive` with RAR support.

Run in the existing server environment:

```bash
/raid1/xwan0900/venvs/ftkp_cu128/bin/python -m unittest -v test_preprocess_public_v2

/raid1/xwan0900/venvs/ftkp_cu128/bin/python run_public_preprocessing.py \
  --root /raid1/xwan0900/fall_detection2/fd_datasets \
  --output-root /raid1/xwan0900/fall_detection2/fd_datasets/processed_v2 \
  --mode full --workers 6
```

Use an unused output directory for another run. The script refuses to overwrite
existing results and publishes each dataset only after reading and verifying every
output CSV. `--mode smoke` runs a small representative archive pilot instead.

Each dataset contains:

```text
fall/                        150-sample or boundary 90-sample event containers
non-fall/                    original ADL, near-falls, and screened context
ssl/                         PAMAP2 labelled-activity segments, supervised y=-1
quarantine/                  transitions, uncertain context, and unusable short crops
manifest.csv                 source, participant/group, timing, event and checksum
preprocessing_audit.csv      source-level parsing, missing data and duplicate audit
preprocessing_summary.json  counts, units, sensor, policies and runtime
verification.json            exhaustive output-verification result
```

All CSVs have `sample_idx,time_ms,Acc_X,Acc_Y,Acc_Z,Gyro_X,Gyro_Y,Gyro_Z`.
CSV timestamps restart at zero; the manifest retains the source interval and
original timestamp origin. Downsampling uses an anti-alias filter. Invalid selected
channels and large timestamp gaps split the data; interpolation never bridges a
large gap or clock reset. PAMAP2 missing heart-rate values do not remove valid IMU
samples, and its 16g chest accelerometer is selected explicitly.

## Dataset-specific cleaning

| Dataset | Policy |
| --- | --- |
| CGU_BES | 200 Hz six-column text; explicit trial labels; paired Acc/Gyro peaks; remove exact numeric duplicates. |
| Cogent | 100 Hz chest channels; each annotated fall processed separately; use both annotations to identify valid falls, near-falls and ADLs; transitions stay in quarantine. |
| SFU_IMU | Microsecond timestamps from spreadsheets; sternum six-axis channels; trial labels distinguish falls, near-falls and ADLs. |
| UCI_SimulatedFalls | Nested RARs; chest sensor 340527; 8xx activities are ADL, 9xx activities are falls; read the exported rate and sample counter, including counter wrap. Empty sensor exports are audited and excluded. |
| PAMAP2 | 100 Hz chest channels; split by activity and sensor gaps; activity 0 stays in quarantine; all retained data is for SSL, with no invented fall labels. |

For fall trials without time annotations, context becomes a pseudo-negative only
when it is quiet and at least two additional seconds outside the selected crop.
Other context is preserved in quarantine. Cogent negatives must also be outside a
two-second guard around every annotated fall. A disconnected trial-level fall
recording stays in quarantine because its individual blocks cannot all be assumed
to contain a fall. Short segments remain available for review rather than padding
validation data with synthetic noise.

## Using the results for training

- Split using the manifest's **subject_id / split_group_id**, before generating
  windows. These combine participants connected by exact duplicate recordings.
  Keep all crops, negatives and sessions from the same group in one partition.
- A 5-second fall CSV is a positive clip, including surrounding non-fall motion.
  The V2 trainer samples random 3-second views that inherit this positive label.
  Event metadata remains available for auditing; it is not used to relabel views.
- Fit normalization on training participants separately for each source domain.
  Values retain their exported units. SFU/PAMAP2/Xsens use m/s² and rad/s;
  CGU and Cogent export units are not documented in their bundled readmes and
  are explicitly marked `source_exported`. No guessed unit conversion is applied.
  Do not apply the private dataset's raw-count normalization to these signals.
- Exclude `quarantine/` from supervised training. Near-falls are binary negatives.
  PAMAP2 is exported with unknown SSL labels; the six-source V2 trainer also uses
  its known labelled activities as non-fall examples during fine-tuning.
- These cleaned files are **30 Hz**. The existing `exp2_converged` public model
  cache expects **20 Hz, 60×6** windows; that requires a separate cache conversion.
  The canonical CSVs are consumed directly by the six-source V2 trainer; see
  [README_V2_TRAINING.md](README_V2_TRAINING.md) for the completed training protocol.

Logs and durable job status are under the output root. A failed unpublished staging
directory remains available for diagnosis; it is never used as a completed dataset.
