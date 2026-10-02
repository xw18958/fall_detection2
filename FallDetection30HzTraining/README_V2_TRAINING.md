# Seven-source V2 M5 adaptation

`train.py` and `training_v2.py` keep the current 30 Hz, 3-second, 90×6 model and
adapt its existing SSL initialization to Private V2, M5 hard negatives and five
public sources.

- Each fall container remains positive, including its surrounding non-fall motion.
  Training takes fresh random three-second crops. M5 sessions remain full-length
  non-fall recordings; neither source is relabelled using peak detection.
- The model receives the complete crop and its three temporal thirds.
- SSL samples seven sources equally. PAMAP2 activities are negatives in
  fine-tuning and use unknown labels for supervised contrastive SSL.
- Fine-tuning draws 25% Private V2, 25% M5 and 10% per public source. Within each
  source, available classes and participant or recording groups are balanced.
  Global class weights give expected weighted CE coefficient shares of 28.49%
  Private V2 and 18.52% M5; equal sampling does not imply equal loss influence.
- Private splits keep recovered negatives, duplicates and padded variants grouped.
  Public participants remain grouped. M5 sessions from one device boot stay in one
  partition within a run. Warm-start independence additionally requires matching
  inherited checkpoint partitions; the completed October 2 run failed the
  public-source ancestry check described in its report.
- Reuse the existing train-only normalization: M5 and Private V2 both use the
  Private V2 statistics, matching the device path. Public data retains its saved
  source normalization.
- Select checkpoints by preserving Private V2 validation recall and keeping each
  public fall dataset above a common floor: the worst baseline source's recall
  minus two percentage points. This is not a per-source relative bound. Among
  candidates passing those gates, minimize the equally weighted Private V2/M5
  negative-window false-positive rates, then use private and public AP as tie-breaks.
  Current code jointly chooses the threshold to minimize
  `0.5 * Private V2 negative-window FPR + 0.5 * M5 negative-window FPR`
  subject to the same recall gates. Equal FPR ties preserve more validation falls,
  then prefer the higher threshold. The baseline receives the same validation-only
  calibration before comparison. Test predictions never enter threshold selection.
  The historical October 2 run used Private V2 alone to tune its threshold;
  its locked checkpoint and reported metrics have not been changed.
  Public specificity is not protected by the current selection code.
- New validation and test runs evaluate every three-second start on the cumulative
  0.25-second grid, including the final complete window in each valid segment.
  Training continues to use random crops. Smoke tests alone cap evaluation at four
  views. Recording scores are the maximum probability across their windows.

The completed October 2 candidate is experimental and is not recommended for
deployment. Read [the detailed results and ancestry audit](runs/v2_m5_hardneg_seed42_20261002/REPORT.md)
before interpreting public scores or starting another adaptation. Inserting M5
changed domain-index-based public split seeds while reusing prior SSL weights
and normalization. The commands below document the archived run's arguments;
current code now uses joint threshold selection instead of that run's private-only
policy. A clean
follow-up must preserve the original six-source partitions and append the M5 split
before restarting from the original SSL checkpoint.

The previous six-source run and its historic scores remain in their original run
directory. The new experiment uses a fresh directory and records its own split,
sampler, normalization provenance, threshold and results. Public test participants
are disjoint within this run but exposed in the inherited checkpoint's
training/validation history. Private test data is
from the same person; the seven M5 sessions provide only about 4.5 minutes in the
final test split, so neither is a large independent deployment study.

Run on the authorized server with the configured environment:

```bash
cd /raid1/xwan0900/fall_detection2/FallDetection30HzTraining
PYTHON=/raid1/xwan0900/venvs/ftkp_cu128/bin/python
BASE=../fd_datasets
OLD=runs/v2_multisource_seed42_20261001/checkpoints
NEW=runs/v2_m5_hardneg_seed42_20261002

$PYTHON preprocess_m5_hard_negatives.py --root "$BASE/M5_hard_negatives_20261002"
$PYTHON -m unittest -v test_training_v2 test_preprocess_m5_hard_negatives test_preprocess_public_v2
$PYTHON train.py --smoke --device cuda:0 --work "$NEW-smoke" \
  --init-ssl "$OLD/ssl_pretrained.pt" --normalization-from "$OLD/locked_model.pt" \
  --baseline-checkpoint "$OLD/locked_model.pt"
$PYTHON train.py --mode test --smoke --device cuda:0 --work "$NEW-smoke"
$PYTHON train.py --device cuda:0 --work "$NEW" --ssl-epochs 5 --head-epochs 3 --all-epochs 20 \
  --stride-sec 0.25 --target-fraction 0.25 --m5-fraction 0.25 \
  --init-ssl "$OLD/ssl_pretrained.pt" --normalization-from "$OLD/locked_model.pt" \
  --baseline-checkpoint "$OLD/locked_model.pt"
$PYTHON train.py --mode test --device cuda:0 --work "$NEW"
```

The M5 importer verifies each CSV against its hidden sample journal and writes a
full-length processed file, a source checksum manifest and a preprocessing audit.
Use `--m5-root`, `--zip` and `--public-root` only to override their default paths.
A V1 private dataset without the V2 audit is rejected. `--mode prepare` checks
grouped splits and signal caches without training. A new run directory is required
when the input fingerprint changes.

The run produces grouped split manifests, preprocessing/data audits, frozen
normalization provenance, sampled source counts, expected class weights, progress
and ETA, SSL and locked checkpoints, per-source prediction CSVs, classification
metrics and private binary-class retrieval against a training-only gallery.

## M5 hard-negative adaptation

The seven complete M5BLECollector recordings are labelled non-fall by the user.
Their raw counts, metadata and hidden recovery journals stay under
`fd_datasets/M5_hard_negatives_20261002/raw`. `preprocess_m5_hard_negatives.py`
verifies row-for-row CSV/journal agreement and metadata quality counts, then writes
one full-length 30 Hz file per session. It keeps difficult and saturated motion;
it never runs fall detection or event cropping. M5's three recordings from one
device boot share one split group. The fixed split uses five sessions for training,
one for validation and one extreme-motion session for final testing.

The importer applies the declared ±8 g/±2000°/s metadata factors and the current
firmware's 16,384 training acceleration counts/g, 16.4 gyroscope counts/(°/s),
and signed-16-bit input bounds. It records the number of acceleration rows clipped
in the manifest and keeps the original raw readings for audit. M5 records share
Private V2 normalization, so the model does not need a source-specific device
normalizer at inference.

Fine-tuning draws expected source shares of 25% Private V2, 25% M5 and 10% from
each public dataset. SSL gives each of the seven sources equal mass. Both stages
balance the available classes and groups inside a source. Fine-tuning also weights
cross-entropy from the resulting expected 32.5% fall / 67.5% non-fall draw.
`comparison.json` records expected coefficients; realized sampler shares and
gradient contributions were not measured in this run.

The adaptation starts from the existing SSL checkpoint and reuses the existing
train-only normalization. The existing locked model sets the validation gates:
Private V2 validation recall must be preserved, and each public fall dataset must
meet the common floor based on the worst baseline source minus two percentage
points. Among models that pass,
selection minimizes the mean Private V2 and M5 negative-window false-positive
rates; Private V2 AP and public macro-AP break ties. Current threshold selection
also minimizes this equally weighted device-negative FPR on validation, subject
to recall gates. The threshold policy and weights are saved with selection metadata.
The M5 test boot is used only after model and threshold selection.

The importer preserves the historical firmware input convention: 16,384
acceleration counts/g followed by signed-16-bit clipping, effectively about ±2 g
per axis. This is a compatibility rule, not a requirement of the neural network or
INT8 quantization. For example, 3 g converts to 49,152 counts and is clamped to
32,767. Changing the convention requires matching data processing, model export
and eventual firmware input conversion; original raw readings remain preserved.

New validation and test runs evaluate all 3-second starts on a cumulative 0.25 s
grid, including the final complete window in each valid segment. Training continues
to use random 3-second crops. Existing six-source checkpoints and their historic
scores are kept as they were; the M5 experiment uses a fresh run directory and its
own locked checkpoint.
