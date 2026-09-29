# Exp2 V2 results

This directory contains the completed 20 Hz Exp2 V2 run. It is kept separate from the historical Exp2 `result.json` so the earlier experiment remains untouched.

## Run
- Kaggle kernel: `xw18958/fall-detection-exp2-v2-greedy`
- Protocol: sequential greedy tuning, then one full selected run
- Two T4 GPUs were used as independent workers during tuning
- Own TEST was not used during tuning
- Public validation was subject-disjoint
- Leakage audit: PASS

## Selected hyperparameters
- SSL learning rate: `1e-4`
- Fine-tuning learning rate: `3e-5`
- Threshold selected on Own VAL only: `0.6`

## Selected checkpoints
- SSL: `ssl_1000`
- Public supervised: `public_7`
- Final: `partial_freeze_4`

## Final Own TEST (recording level)
- n = 28
- TP/TN/FP/FN = 18/6/3/1
- Accuracy = 0.8571
- Precision = 0.8571
- Recall = 0.9474
- Specificity = 0.6667
- F2 = 0.9278
- MCC = 0.6623

## Notes
The Own TEST is a same-person unseen-recording test, not unseen-subject generalization. The UCI cache retained all 17 subjects but skipped 34 unreadable source records from damaged/incomplete archive contents.
