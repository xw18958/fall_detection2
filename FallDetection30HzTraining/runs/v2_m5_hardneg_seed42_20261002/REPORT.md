# M5 hard-negative adaptation: results and takeaways

Run: `v2_m5_hardneg_seed42_20261002`; seed 42; completed 2 October 2026.

**Status: experimental candidate, not recommended for deployment.** The candidate reduced M5 false-positive windows by 33.87% relative to the previous checkpoint, but still flagged over half of the held-out session's windows. Private V2 recording results did not improve. A subsequent ancestry audit found public partition contamination, so the public scores below are diagnostic and cannot establish unseen-person generalization.

## Data actually used

These are processed recordings/containers, not independent people or training windows.

| Source | Train | Validation | Test | Fine-tuning draw share |
|---|---:|---:|---:|---:|
| Private V2 | 245 | 53 | 51 | 25% |
| M5 hard negatives | 5 | 1 | 1 | 25% |
| CGU_BES | 132 | 26 | 25 | 10% |
| Cogent | 1225 | 267 | 266 | 10% |
| SFU_IMU | 367 | 119 | 119 | 10% |
| UCI_SimulatedFalls | 2386 | 445 | 525 | 10% |
| PAMAP2 | 266 | 56 | 60 | 10% |
| **Total** | **4,626** | **967** | **1,047** | **100%** |

All seven M5 sessions participated: five in training, one in validation and one in testing. They span 20.19 minutes: 11.09 training minutes and 4.55 minutes each for validation/test. There are five boot groups, assigned 3/1/1. Sessions sharing a boot stay together; different boots do not prove different people. Public participants and linked Private V2 recordings are grouped within this run.

The importer retained full-length M5 negatives without event cropping or peak relabelling. CSV/recovery-journal verification passed. There were zero read errors or timing-gap flags. All 159 sensor-saturated samples were retained. The uniform 30 Hz grid has 36,345 samples versus 36,351 originals; six endpoint samples differ because of grid placement, not removal of difficult motion.

The firmware-compatible conversion uses 16,384 acceleration counts/g and signed-16-bit bounds, effectively about ±2 g per axis despite ±8 g acquisition. It clipped acceleration in 1,527 raw rows. Validation had 71 clipped rows and 2 saturated samples, whereas test had 884 clipped rows and 70 saturated samples. The test session has substantially stronger clipping/saturation than validation. This could contribute to the generalization gap, but causality has not been established. Raw readings remain preserved on the server.

## What training actually did

The model retained 30 Hz, 3-second 90×6 inputs, 24 channels, three encoder blocks and 40,561 parameters. It encodes the complete window and its three temporal thirds. No architecture or firmware changes were part of this run, and device latency was not measured.

Initialization came from the original six-source **SSL** checkpoint. Existing train-fitted normalization was reused; M5 shares Private V2 normalization. Whole five-second fall containers remain positive; fresh random three-second training crops inherit the parent label, including surrounding non-fall motion. M5 is negative. PAMAP2 supplies supervised negatives and has unknown labels for the supervised contrastive portion of SSL.

SSL draws seven sources equally. Fine-tuning uses the shares above and balances available classes/groups within sources. Batch size is 128; each epoch has 100 updates. Sampling is with replacement: all training recordings are eligible, but not every possible crop is guaranteed to be presented. Fourteen executed epochs mean 179,200 crop presentations, including 115,200 supervised presentations; these are not unique examples.

| Stage | Configured | Executed | Outcome |
|---|---:|---:|---|
| SSL adaptation | 5 epochs | 5 | Warm-start continuation |
| Head training | 3 epochs | 3 | Encoder frozen |
| Full-model fine-tuning | At most 20 epochs | 6 | Epoch 1 selected; next five did not improve the selection score |

Training took approximately 73 seconds, excluding transfer, preprocessing, smoke checks and later evaluation. This was a small warm-started model with fixed updates per epoch. The budgets were practical choices, **not proven optimal numbers**. Early stopping used validation with patience five. The selected checkpoint contains new weights, not the baseline fallback.

## Comparable evaluation settings

Validation/test use every complete three-second start on the cumulative 0.25-second grid, plus the final complete window of each valid segment, with no 32-view cap. At 30 Hz, starts alternate approximately seven/eight samples. Recording probability is the maximum across all its windows.

The previous checkpoint was reevaluated on exactly the same current split/grid. Its locked threshold is 0.425; the candidate's validation-selected threshold is 0.345. The comparison therefore reflects both weights and operating thresholds. The historical Private V2 result of 94.12% accuracy/three false-positive recordings used sparser/capped views. Under this exhaustive evaluation, that same old checkpoint gives 88.24%/six false-positive recordings; the historical score is not the proper baseline here.

## Recording-level classification

Public rows are diagnostic owing to the ancestry issue below. Candidate accuracy/precision are shown; FP/FN count false-positive/missed-fall recordings. Negative-only sources have no meaningful fall recall or precision.

| Source | Test N | Candidate accuracy | Candidate precision | Recall: old → new | Specificity: old → new | FP: old → new | FN: old → new |
|---|---:|---:|---:|---:|---:|---:|---:|
| Private V2 | 51 | 88.24% | 81.25% | 100.00% → 100.00% | 76.00% → 76.00% | 6 → 6 | 0 → 0 |
| M5 hard negatives | 1 | 0.00% | — | — → — | 0.00% → 0.00% | 1 → 1 | 0 → 0 |
| CGU_BES | 25 | 84.00% | 66.67% | 100.00% → 100.00% | 82.35% → 76.47% | 3 → 4 | 0 → 0 |
| Cogent | 266 | 99.25% | 97.22% | 100.00% → 100.00% | 100.00% → 98.98% | 0 → 2 | 0 → 0 |
| SFU_IMU | 119 | 78.15% | 61.19% | 100.00% → 100.00% | 84.62% → 66.67% | 12 → 26 | 0 → 0 |
| UCI_SimulatedFalls | 525 | 97.90% | 97.53% | 99.64% → 98.57% | 99.59% → 97.14% | 1 → 7 | 1 → 4 |
| PAMAP2 | 60 | 75.00% | — | — → — | 93.33% → 75.00% | 4 → 15 | 0 → 0 |

Private V2 detected 26/26 falls and correctly rejected 19/25 negatives. F2 is 95.59%, MCC 0.7858 and recording AP 100%. AP 100% means positives ranked above negatives in this small test; it does not mean the locked threshold gave perfect classification. Calibration is a plausible issue, but choosing a better threshold from this test would invalidate an independent improvement claim.

The M5 test contains one full negative session, and both models flagged it. Across the diagnostic public tests, false-positive recordings increased from 20 to 54, and UCI missed falls increased from 1 to 4. The run cannot be called a general improvement.

## Window-level target results

| Measure | Previous checkpoint | Candidate |
|---|---:|---:|
| Private V2 negative-window FP | 44 / 15,454 | 40 / 15,454 |
| Private V2 negative-window FPR | 0.2847% | 0.2588% |
| Private V2 positive-window recall | 161/162 = 99.38% | 157/162 = 96.91% |
| M5 negative-window FP | 862 / 1,082 | 570 / 1,082 |
| M5 negative-window FPR | 79.67% | 52.68% |
| M5 negative-window specificity | 20.33% | 47.32% |

M5 FPR fell by **26.99 percentage points**, a **33.87% relative reduction**. This is real progress on this held-out session, but inadequate rejection. Private V2 window false positives improved slightly while positive-window recall worsened. Those labels inherit noisy parent-container labels; a missed positive window is not necessarily a missed fall event. All 26 fall recordings remained detected.

Private V2 validation had 26/26 falls and 27/27 negatives correct. M5 validation had 35/1,082 false-positive windows (3.23%), versus 570/1,082 (52.68%) on test. One validation session did not represent the test session's difficulty. Session-specific fitting and distribution differences are plausible; a single test boot cannot establish a precise population rate.

Windows overlap by 2.75 seconds, so 570 flagged windows are not 570 separate alerts or independent observations. False-alert events/hour and detection latency were not measured: event timing, alert merging and cooldown require a separate deployment evaluation.

## Retrieval and zero-shot results

Gallery: 245 training recordings. Queries: 51 Private V2 test recordings. This is binary-class retrieval, not retrieval of the same event or operational device accuracy.

| Metric | Previous checkpoint | Candidate |
|---|---:|---:|
| Precision@1 | 94.12% | 94.12% |
| Hit rate@5 | 94.12% | 96.08% |
| Retrieval mAP | 95.94% | 95.66% |

There is no consistent retrieval gain. Zero-shot dataset results are unavailable because all seven source domains participate in fitting. Private V2 covers unseen recordings from the existing person, not unseen people or future devices.

## The ancestry audit changes the public interpretation

`split_all` derives each source seed from its index in `DOMAINS`. Inserting M5 before the public domains changed their seeds, while the reused SSL checkpoint and normalization retained the original training history. Within-run group checks passed, but did not audit inherited exposure.

| Current public test source | Previous training | Previous validation | Previous untouched test |
|---|---:|---:|---:|
| CGU_BES | 12 / 25 | 13 / 25 | 0 |
| Cogent | 160 / 266 | 106 / 266 | 0 |
| SFU_IMU | 0 / 119 | 119 / 119 | 0 |
| UCI_SimulatedFalls | 180 / 525 | 345 / 525 | 0 |
| PAMAP2 | 47 / 60 | 13 / 60 | 0 |
| **Total** | **399 / 995** | **596 / 995** | **0** |

Thus all 995 current public test recordings had been used in earlier training or validation. The adaptation also moved 500 original public test recordings into training. This candidate cannot be used to restart a clean evaluation on those original tests. Private V2 membership is unchanged, and M5 was new: their test partitions were not exposed to fitting through the inherited checkpoint. Private V2 test outcomes were already known from earlier work, so genuinely new device data would strengthen final claims.

`data_audit.json` records current-run disjointness, not full checkpoint ancestry. The newly added `warm_start_audit.json` supplies that missing audit; original metrics and run files are preserved.

## Weighting and selection limitations

Equal 25% draw shares did **not** give equal classification-loss coefficient mass. Negative-only M5/PAMAP2 make the expected draw 67.5% negative and 32.5% positive. Global CE weights are 0.7407 for negatives and 1.5385 for positives.

| Source | Sampling share | Expected weighted CE coefficient share |
|---|---:|---:|
| Private V2 | 25% | 28.49% |
| M5 hard negatives | 25% | 18.52% |
| Each fall-containing public source | 10% | 11.40% |
| PAMAP2 | 10% | 7.41% |

These are expected coefficients, not measured gradients: individual errors and contrastive loss also affect training. Realized sampler shares were not logged. The distinction matters for the requested equal importance of Private V2 and M5.

Thresholds were selected using Private V2 validation MCC/F2/specificity under a recall constraint. M5 influenced checkpoint ranking through its negative-window FPR, but did not directly optimize the threshold. Public recall had one common floor based on the worst baseline source minus two percentage points, **not** a separate bound relative to each source's baseline. Public specificity had no protective gate; negative-only PAMAP2 has no recall gate. These choices permitted the observed negative-rejection regressions. They are documented as executed, not silently changed after seeing test results.

## What should happen next

1. Restart from the original SSL checkpoint and preserve the original six-source split, appending the M5 boot split. Audit weights, normalization and prior selection exposure before fitting. Do not start from this contaminated candidate.
2. Keep the requested sampling shares while explicitly defining equal Private V2/M5 loss importance with source-normalized classification losses. Log realized shares as well as expected coefficients.
3. Collect/use additional independent M5 boots for validation across varied hard motion, reserving genuinely new recordings for final testing. A 25% sampler cannot create missing motion diversity from 11 training minutes.
4. Jointly select the threshold on validation Private V2 and M5, preserving Private V2 recall and applying source-specific public recall/specificity checks. Whole noisy fall containers remain positive.
5. Audit the ±2 g compatibility clamp before changing it. Any input-map change must match training and firmware and be evaluated separately.
6. Compare a small number of seeds after these corrections; measure merged false alerts/hour and event timing before quantization/deployment. The current history does not support more epochs alone as the solution.

These are recommendations, not additional experiments already performed. Device work remains paused.

## Artifacts

- `checkpoints/locked_model.pt`: selected experimental candidate, configuration, normalization and locked threshold, approximately 199 KB.
- `checkpoints/ssl_pretrained.pt`: SSL adaptation endpoint.
- `results/final_test_metrics.json`, `test_summary.csv`: original full scores.
- `results/*_test_recordings.csv`: recording predictions; M5 validation/test window CSVs preserve target score sequences. Other window CSVs remain on the server.
- `training_history.json`, `train.log`, `results/validation_metrics.json`: original fitting and selection evidence.
- `comparison.json`, `comparison.csv`, `run_analysis.json`: comparisons and execution analysis.
- `warm_start_audit.json`, `code_hashes.json`, `m5_preprocessing_verification.json`: ancestry, provenance and processing evidence.
- `splits/group_splits.json`, `run_config.json`, `normalization.json`, `data_audit.json`: original configuration and split records.

The sibling `v2_m5_baseline_comparison_seed42_20261002` preserves the baseline evaluation. Raw sensor readings and caches remain on the server. Unit tests and GPU smoke checks passed before training, but did not catch the inherited-partition problem discovered in this review.
