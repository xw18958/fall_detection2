# V5 combined M5 training and quantization — seed42

Training and quantization completed. **No candidate passed all declared validation gates. No firmware was installed; full final-test evaluation of the production candidates was not run.**

## Data and protocol

Nineteen user-confirmed M5 falls were combined with seven verified full-range M5 negatives. The existing paired-peak crop method yielded eighteen 5-second containers and one real 3-second boundary container. Every 3-second view inherits its positive container label. No padding or new phase labels were introduced.

Positive recordings split 13/3/3 with seed42. They share one boot; participant and placement are unknown. These are exploratory whole-recording holdouts. Negative boot splits and original six-source splits were preserved. Normalization uses combined M5 training inputs only; source masses remain 25% Private /25% M5 /10% each public.

Training completed SSL/head/full epochs {'ssl': 30, 'head': 3, 'all': 14}; selected full checkpoint epoch 4. QAT completed 12 epochs; selected epoch 10.

## Validation classification

All 61,713 complete 3-second windows were evaluated on the 0.25-second cumulative grid plus final window. Overlapping windows are correlated. INT8 predictions use native TFLite Micro and the firmware custom kernels. These are validation results, not final-test results.

| Model | Source | TP | FN | FP | TN | Precision | Recall | F1 | Specificity | Negative-window FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Float | own | 130 | 8 | 0 | 2724 | 100.00% | 94.20% | 97.01% | 100.00% | 0.00% |
| Float | m5_hard_negatives | 19 | 8 | 0 | 1082 | 100.00% | 70.37% | 82.61% | 100.00% | 0.00% |
| Float | CGU_BES | 72 | 0 | 0 | 883 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Float | Cogent | 627 | 3 | 0 | 11541 | 100.00% | 99.52% | 99.76% | 100.00% | 0.00% |
| Float | SFU_IMU | 349 | 20 | 16 | 4124 | 95.62% | 94.58% | 95.10% | 99.61% | 0.39% |
| Float | UCI_SimulatedFalls | 2471 | 40 | 1 | 15202 | 99.96% | 98.41% | 99.18% | 99.99% | 0.01% |
| Float | PAMAP2 | 0 | 0 | 40 | 22353 | N/A | N/A | N/A | 99.82% | 0.18% |
| Float | pooled | 3668 | 79 | 57 | 57909 | 98.47% | 97.89% | 98.18% | 99.90% | 0.10% |
| PTQ | own | 129 | 9 | 0 | 2724 | 100.00% | 93.48% | 96.63% | 100.00% | 0.00% |
| PTQ | m5_hard_negatives | 16 | 11 | 0 | 1082 | 100.00% | 59.26% | 74.42% | 100.00% | 0.00% |
| PTQ | CGU_BES | 72 | 0 | 0 | 883 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| PTQ | Cogent | 627 | 3 | 0 | 11541 | 100.00% | 99.52% | 99.76% | 100.00% | 0.00% |
| PTQ | SFU_IMU | 348 | 21 | 16 | 4124 | 95.60% | 94.31% | 94.95% | 99.61% | 0.39% |
| PTQ | UCI_SimulatedFalls | 2482 | 29 | 1 | 15202 | 99.96% | 98.85% | 99.40% | 99.99% | 0.01% |
| PTQ | PAMAP2 | 0 | 0 | 27 | 22366 | N/A | N/A | N/A | 99.88% | 0.12% |
| PTQ | pooled | 3674 | 73 | 44 | 57922 | 98.82% | 98.05% | 98.43% | 99.92% | 0.08% |
| QAT | own | 129 | 9 | 0 | 2724 | 100.00% | 93.48% | 96.63% | 100.00% | 0.00% |
| QAT | m5_hard_negatives | 17 | 10 | 0 | 1082 | 100.00% | 62.96% | 77.27% | 100.00% | 0.00% |
| QAT | CGU_BES | 72 | 0 | 0 | 883 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| QAT | Cogent | 627 | 3 | 0 | 11541 | 100.00% | 99.52% | 99.76% | 100.00% | 0.00% |
| QAT | SFU_IMU | 350 | 19 | 16 | 4124 | 95.63% | 94.85% | 95.24% | 99.61% | 0.39% |
| QAT | UCI_SimulatedFalls | 2477 | 34 | 1 | 15202 | 99.96% | 98.65% | 99.30% | 99.99% | 0.01% |
| QAT | PAMAP2 | 0 | 0 | 39 | 22354 | N/A | N/A | N/A | 99.83% | 0.17% |
| QAT | pooled | 3672 | 75 | 56 | 57910 | 98.50% | 98.00% | 98.25% | 99.90% | 0.10% |

## Selection and threshold diagnostics

PTQ and QAT used identical checkpoint/split provenance, validation gates and float32 threshold selection. The M5 gates require ≥90% positive-window recall, ≥90% fall-recording recall and ≤1% negative-window FPR, alongside inherited Private/public protection. The default weighted error includes recording-level fall detection; a small value does not establish high positive-window recall.

| Model | Selected threshold | Qualified | M5 positive windows detected | M5 negative FP windows |
|---|---:|---|---:|---:|
| Float | 0.892929316 | False | 19/27 | 0/1082 |
| PTQ | 0.949923158 | False | 16/27 | 0/1082 |
| QAT | 0.909689844 | False | 17/27 | 0/1082 |

Validation-only M5 sensitivity diagnostics below are **not adopted or deployed**. Gates were not relaxed.

- Float: threshold 0.343523502, M5 window recall 92.59%, M5 negative-window FPR 0.00%; violated full gates: [{"domain": "CGU_BES", "metric": "specificity", "value": 0.888888888888889, "bound": 0.98}, {"domain": "SFU_IMU", "metric": "specificity", "value": 0.8589743589743593, "bound": 0.8646153846153848}].
- PTQ: threshold 0.631320477, M5 window recall 92.59%, M5 negative-window FPR 0.09%; violated full gates: [{"domain": "CGU_BES", "metric": "specificity", "value": 0.888888888888889, "bound": 0.98}].
- QAT: threshold 0.909689844, M5 window recall 62.96%, M5 negative-window FPR 0.00%; violated full gates: [{"domain": "m5_hard_negatives", "metric": "positive_window_recall", "value": 0.6296296296296297, "bound": 0.9}].

## Retrieval and quantization checks

| Evaluation | P@1 | Hit@5 | mAP | Gallery / queries |
|---|---:|---:|---:|---|
| Float fine-tune validation, binary-class retrieval | 0.981132 | 0.981132 | 0.986930 | 245 / 53 |

Fine-tune final-test results: pending validation qualification. A small fixed subset was exercised by the pipeline smoke test, using its separate smoke checkpoint, and did not select a production candidate. Zero-shot: not available; all sources participate in training. Retrieval is binary-class retrieval, not event-instance retrieval.

Thirty-six unit tests, GPU SSL/head/full/test/retrieval smoke, exact float graph parity, source/class calibration checks, QAT finite-gradient smoke and native Micro validation passed. PTQ uses per-channel convolution weights and has INT8/INT32 tensors only. QAT uses five LayerNormV4 operators with INT8 interfaces, int64 statistics and an internal float32 sqrt. These are host runtime checks; new hardware latency, alarms and commercial readiness are not established.

## Reproduction and private artifacts

See `README_RETRAINING_V5.md`, `run_v5_training.py`, and `run_v5_quantization.py`. Source and compact aggregates may be public. Raw data, calibration/replay inputs and model binaries remain private.

- Combined dataset: `/raid1/xwan0900/fall_detection2/fd_datasets/M5_combined_v5_20261003`
- Experiment and exported candidates: `/raid1/xwan0900/v5_m5_seed42_20261003`
- Float checkpoint SHA256: `69df2aed93411a7ec78afb6b726c558196182870595e4fb98028d5de533dfe55`
- PTQ model SHA256: `4eb70382f3306a4c8580e5d08342032edae43d3bd4e07cb4a3a9090e85dfa367`
- Selected QAT model SHA256: `636dca88ede1cd7e65bf0204225f833f88b329b5226310a67249af57c450e15c`
