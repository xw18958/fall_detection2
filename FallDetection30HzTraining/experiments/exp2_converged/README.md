# Exp2 convergence run

Kaggle kernel: `xw18958/fall-detection-exp2-converged`

This folder contains the exact Exp2-only training script and the final result JSON from the completed Kaggle run.

Setup:
- Own split: 160 train / 34 val / 35 test
- 30 Hz, 3 s windows, 6-axis IMU
- Public SSL: 5,000 windows each from CGU-BES, Cogent, SFU, UCI, PAMAP2
- Public supervised: 2,000 windows each from CGU-BES, Cogent, SFU, UCI
- SSL: up to 5,000 steps
- Public supervised: up to 30 epochs
- Target fine-tuning: head up to 10 epochs, all layers up to 50 epochs
- Target fine-tuning checkpoint/threshold selected on own validation set only

Important caveat: the SSL and public-supervised stages in this run were selected by their own training-loss convergence rules, not by the own validation set. Therefore this run should not be treated as a correctly selected final pipeline checkpoint.

Final recording-level result of this run:
- Validation MCC: 0.8593
- Test MCC: 0.7282
- Test accuracy: 0.8857
- Test recall: 0.8846
- Test precision: 0.9583

The complete checkpoint and full outputs remain saved in the Kaggle kernel outputs.