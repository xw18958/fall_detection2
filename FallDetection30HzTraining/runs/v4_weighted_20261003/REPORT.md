# Single-seed seven-source V2 retraining

Only seed 42 was completed and tested. Seeds 43 and 44 were cancelled at the user's request; partial seed 43 output is excluded.

## Setup and timing

Training: 405.81 seconds; test and baseline evaluation: 23.44 seconds. Preprocessing, preflight, and orchestration delays are excluded.
Completed stages: SSL 24, head 3, full-model 18. Selected full-model epoch 8; threshold 0.260215610.
SSL stopped after 8 validation-loss epochs without improvement. Full-model training stopped after 10 epochs without a better validation rank. Epoch budgets were limits, not established optimal counts.
Source weights: Private V2 25%, M5 25%, and each of CGU, Cogent, SFU, UCI and PAMAP2 10%. Every one of the 4,628 training recordings was sampled in every epoch; source/group/class queues were balanced.
Original participant/recording splits were restored. All 963 test recordings were evaluated using the uncapped 0.25-second grid. No test result changed checkpoint or threshold selection.
M5 uses metadata-derived full-range SI values and its own training-only normalization. Private V2 keeps its prior numeric representation; its physical calibration is still unresolved.

## Fine-tuned test classification

Recording classification uses the maximum window probability: any positive window makes the recording positive. Negative-window FPR uses all negative windows. The units below are fractions.

| dataset | n | tp | fn | fp | tn | accuracy | precision | recall | specificity | f1 | f2 | mcc | ap | negative_window_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| own | 51 | 26 | 0 | 5 | 20 | 0.9020 | 0.8387 | 1.0000 | 0.8000 | 0.9123 | 0.9630 | 0.8191 | 1.0000 | 0.0012 |
| m5_hard_negatives | 1 | 0 | 0 | 1 | 0 | 0.0000 | N/A | N/A | 0.0000 | N/A | N/A | N/A | N/A | 0.0619 |
| CGU_BES | 25 | 8 | 0 | 2 | 15 | 0.9200 | 0.8000 | 1.0000 | 0.8824 | 0.8889 | 0.9524 | 0.8402 | 1.0000 | 0.0279 |
| Cogent | 266 | 70 | 0 | 4 | 192 | 0.9850 | 0.9459 | 1.0000 | 0.9796 | 0.9722 | 0.9887 | 0.9626 | 1.0000 | 0.0007 |
| SFU_IMU | 119 | 40 | 1 | 4 | 74 | 0.9580 | 0.9091 | 0.9756 | 0.9487 | 0.9412 | 0.9615 | 0.9099 | 0.9978 | 0.0051 |
| UCI_SimulatedFalls | 445 | 243 | 0 | 0 | 202 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| PAMAP2 | 56 | 0 | 0 | 4 | 52 | 0.9286 | N/A | N/A | 0.9286 | N/A | N/A | N/A | N/A | 0.0066 |

## Comparison with the historical pipeline

Both pipelines use the same test partitions, the exhaustive grid, and their respective validation-selected thresholds. The baseline retains its historical M5 conversion. This comparison combines model and preprocessing changes.

| dataset | accuracy_baseline | accuracy_new | recall_baseline | recall_new | fp_baseline | fp_new | negative_window_fpr_baseline | negative_window_fpr_new |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| own | 0.8824 | 0.9020 | 1.0000 | 1.0000 | 6 | 5 | 0.0023 | 0.0012 |
| m5_hard_negatives | 0.0000 | 0.0000 | N/A | N/A | 1 | 1 | 0.7153 | 0.0619 |
| CGU_BES | 0.9200 | 0.9200 | 1.0000 | 1.0000 | 2 | 2 | 0.0190 | 0.0279 |
| Cogent | 0.9850 | 0.9850 | 1.0000 | 1.0000 | 4 | 4 | 0.0010 | 0.0007 |
| SFU_IMU | 0.9580 | 0.9580 | 1.0000 | 0.9756 | 5 | 4 | 0.0123 | 0.0051 |
| UCI_SimulatedFalls | 0.9798 | 1.0000 | 0.9671 | 1.0000 | 1 | 0 | 0.0002 | 0.0000 |
| PAMAP2 | 0.9107 | 0.9286 | N/A | N/A | 5 | 4 | 0.0062 | 0.0066 |

## Private V2 retrieval

Binary-class retrieval with 245 training gallery recordings and 51 held-out queries: P@1 1.0000; hit@5 1.0000; mAP 0.999975. This does not measure unseen-event retrieval.

## M5 false alarms

| key | boot | duration_seconds | negative_windows | false_positive_windows | false_alarm_episodes | offline_episodes_per_hour | episode_policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m5_hard_negatives/non-fall/2026-10-02_17-55-47_unspecified_24e5413e.csv | M5BLE:84ad620b2a937134 | 273.0000 | 1082 | 67 | 14 | 184.6154 | consecutive positive 0.25 s starts merged; gaps break episodes |

Offline episodes merge consecutive positive window starts and break at gaps. These are not measured firmware alarm events. One 273-second held-out M5 boot limits the reliability of an hourly estimate.

## Takeaways

- Private V2 detected 26/26 falls with five false-positive recordings (baseline six). Accuracy is 90.20%; false positives remain the main weakness.
- M5 false-positive windows fell from 71.53% to 6.19%, but 67/1,082 windows were still positive and formed 14 offline episodes in 4.55 minutes. This is insufficient for deployment.
- UCI improved to 243/243 falls detected. SFU missed one of 41 falls; CGU and PAMAP negative-window FPR worsened. Improvement is not uniform across sources.
- Validation passed the predeclared guards; this flag does not guarantee test or device performance. Keep this checkpoint as an experiment rather than silently replacing deployed firmware.
- Next: collect more independent M5 non-fall sessions and confirmed M5 falls, review error windows, and validate a device alert rule on held-out sessions. Keep the current test threshold fixed when reporting this run.
- There is no zero-shot evaluation: all seven sources contributed training. Single-seed results do not estimate seed variance.

## Checkpoint

Canonical server: /raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt
The checkpoint, compact reports, source splits, and sampler audits are versioned. Raw data, caches, window CSVs, partial extra seeds, and model logs are excluded from Git.

