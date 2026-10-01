# Six-source V2 training

`train.py` now uses `training_v2.py` for training, validation and locked testing.
The 30 Hz / 3-second / 90×6 compact dual-stream model is retained.

- Fall containers are positive clips, including surrounding non-fall motion.
  Every training visit samples a fresh random 3-second crop. No event is detected
  again, no windows are relabelled from peaks, and no pseudo-phase loss is used.
- The four model streams are the full crop and its three temporal thirds. The
  attention module learns which time region matters without moving the event.
- SSL samples the six sources equally. PAMAP2 labels are unknown for supervised
  contrastive loss, but its known activities are negatives during fine-tuning.
- Fine-tuning uses 50% private V2 and 10% each public dataset by default. Sampling
  balances class and participant/source recording inside each domain rather than
  weighting long files more heavily. `--target-fraction 0.1666666667` makes the six
  sources approximately equal during fine-tuning too.
- Private splits group each fall crop with its recovered negatives and duplicates.
  Padded private recordings and their related clips stay in training only.
  Public splits use the manifest's linked participant groups. Approximately
  70/15/15 is used, with at least two held-out groups for small public datasets.
- Normalization is learned separately per source from training recordings only;
  the locked model carries the private normalization needed for deployment.
- Checkpoint selection uses 80% private recording AP and 20% mean AP across the
  four public fall datasets. PAMAP2 cannot supply fall AP. The threshold is chosen
  only after checkpoint selection, using private validation and a 90% minimum
  recall target. No test scores influence either choice.
- Validation and testing use fixed views and max probability per clip, with a
  deterministic cap of 32 views on long negative clips. Full-length continuous
  false-alarm monitoring is a separate deployment evaluation.

Run on the server with the configured environment:

```bash
cd /raid1/xwan0900/fall_detection2/FallDetection30HzTraining
PYTHON=/raid1/xwan0900/venvs/ftkp_cu128/bin/python

$PYTHON -m unittest -v test_training_v2 test_preprocess_public_v2

$PYTHON train.py --smoke --device cuda:0 --work runs/smoke_v2
$PYTHON train.py --mode test --smoke --device cuda:0 --work runs/smoke_v2

$PYTHON train.py --device cuda:0 --work runs/v2_multisource_seed42
$PYTHON train.py --mode test --device cuda:0 --work runs/v2_multisource_seed42
```

Defaults locate the private V2 directory or ZIP and `fd_datasets/processed_v2`.
Use `--zip` and `--public-root` to override them. A V1 dataset without the V2 audit
is rejected. `--mode prepare` validates groups and builds train/validation caches
without launching training. Reusing a work directory with different inputs is
rejected; a locked model cannot be overwritten by another training run.

Outputs include grouped split manifests, data audits, training-only normalization,
progress/ETA, training history, SSL and locked checkpoints, per-source prediction
CSVs and classification metrics. Testing also reports private binary-class
retrieval against a training-only gallery; this is not event-instance retrieval.
No zero-shot claim is made: the model trains on all six source domains, and public
test participants are unseen participants in those domains. Private testing is
unseen-source-recording evaluation from the existing person, not an independent
new-person deployment test.
