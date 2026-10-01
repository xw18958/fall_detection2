# Completed six-source V2 training

Run: `v2_multisource_seed42_20261001`. Seed 42; GPU training and locked held-out testing completed successfully in 167.5 seconds. Epochs completed: {'ssl': 20, 'head': 3, 'all': 14}.

## Implemented settings

- Whole 5-second fall containers remain positive; fresh random 3-second training crops inherit the parent label, including noisy non-fall motion. No V1 peak relabelling or pseudo-phase supervision.
- SSL samples all six sources equally. Fine-tuning samples private V2 at 50% and each public source at 10%, balancing class and participant/recording within each source.
- PAMAP2 supplies unlabeled SSL data and known-activity supervised negatives.
- Public participants and linked private source recordings stay within one split. Padded private examples stay in training. Normalization uses training data only.
- Checkpoint selection uses validation only. The threshold is locked from private validation at 0.425 before testing.
- Timestamp duplicates and pauses in original private negatives are handled without cropping across missing time.

## Locked fine-tune test results

Metrics are recording-level, using the maximum probability across fixed 3-second views (at most 32 views per long recording). Percentages below.

| Dataset | Test recordings | Accuracy | Precision | Recall | Specificity | AP | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|---|
| own | 51 | 94.12% | 89.66% | 100.00% | 88.00% | 99.86% | 26 / 3 / 0 / 22 |
| CGU_BES | 25 | 84.00% | 66.67% | 100.00% | 76.47% | 100.00% | 8 / 4 / 0 / 13 |
| Cogent | 266 | 98.87% | 95.89% | 100.00% | 98.47% | 100.00% | 70 / 3 / 0 / 193 |
| SFU_IMU | 119 | 94.96% | 87.23% | 100.00% | 92.31% | 99.27% | 41 / 6 / 0 / 72 |
| UCI_SimulatedFalls | 445 | 98.20% | 99.16% | 97.53% | 99.01% | 99.95% | 237 / 2 / 6 / 200 |
| PAMAP2 | 56 | 94.64% | — | — | 94.64% | — | 0 / 3 / 0 / 53 |

## Private binary-class retrieval

Training gallery: 245 recordings. Held-out queries: 51 recordings.

| Precision@1 | Hit rate@5 | mAP |
|---:|---:|---:|
| 94.12% | 94.12% | 95.94% |

## Evaluation scope and validation

- 22 pipeline/preprocessing tests passed; a small GPU run completed SSL, head training, unfrozen fine-tuning and locked testing before the full run.
- The private test evaluates unseen source recordings from the existing person. Public tests evaluate unseen participant groups from trained source domains.
- Zero-shot results are unavailable because all six datasets participate in training. These results do not establish performance on future unseen people or devices.
- Recording-level specificity is not a continuous false-alarm rate; long negative recordings use capped evaluation views.
- Retrieval measures binary class agreement, not recovery of the same fall event.
- No test score was used for model selection or threshold tuning. Local source SHA256 hashes match the completed remote run.

## Artifacts

- `checkpoints/locked_model.pt`: selected model, configuration, normalization and threshold.
- `results/final_test_metrics.json`: full recording and window metrics plus retrieval.
- `splits/group_splits.json`: reproducible split membership.
- `normalization.json`, `data_audit.json`, `code_hashes.json`, `training_history.json`: reproducibility evidence.

Raw datasets, prediction rows and checkpoint binaries are retained on the training server and excluded from this Git commit.
