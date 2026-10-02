# Reproduce the locked V4 QAT development package

Use the immutable original checkpoint SHA in export_lock.json and the same inherited split/normalization/data fingerprints. All private data and calibration inputs stay on the GPU server. One seed, 42; never evaluate candidates on the test partition.

Server checkout: /raid1/xwan0900/fall_detection2_v4_qat. Python: /raid1/xwan0900/venvs/fall_v2_export/bin/python. Build the native replay runner with the matching custom normalization source before QAT. Run qat_v4.py --phase smoke, then --phase train with --fused-norm --input-percentile 99.9 --distillation-weight 2.0. Use the original checkpoint/work/data/ptq-export/runner paths documented in REPORT.md and the preceding V4 PTQ REPRODUCE.md. The actual launch script is /raid1/xwan0900/run_v4_fused.sh. The exhaustive validation-qualified run selected epoch 5. Rejected variants are audit artifacts, not deployment alternatives selected on test.

Evaluate once with evaluate_locked_qat.py against the selected run. It writes export_lock.json before reading any test data, verifies old evaluation window order and labels, reports original/adjusted-threshold metrics, and produces actual Micro replay expectations. The actual invocation is /raid1/xwan0900/eval_v4_locked.sh. Never overwrite the existing locked package; use a new output directory for reproduction.

Prepare the private local model with tools/prepare_model.py --source PACKAGE. Run tools/smoke-tests.sh, tools/verify_input_pipeline.py, native replay --kernel-test and native replay MODEL. Build with platformio run, without upload targets. Verify the exact embedded model once, 16-byte alignment and slot bounds with tools/verify_preservation.py --firmware-binary BIN --firmware-elf ELF --nm NM. Original upload/erase guards remain in force.

USB recovery uses tools/usb_app_slot.py in inspect, stage and activate phases, with a complete private 8 MB backup, exact firmware, port and expected MAC. Stage writes only the derived inactive slot and checks the full readback and protected flash. Activate requires that proof and changes only the older 4 KB OTA selector sector. Save recordings and confirm DETECT/no pending samples before any reset. Never erase flash, replace the bootloader/partition table/NVS, or use generic upload.

The Mac receiver retains strict old/PTQ/QAT model identities and all journal checks. Use the built signed M5BLECollector.app. Deployment testing uses a separate test_recordings directory. Existing real recordings remain at the original Mac path.

Exact smoke-first training invocation used (on the GPU server):

```bash
#!/bin/bash
set -euo pipefail
cd /raid1/xwan0900/fall_detection2_v4_qat
common=(--checkpoint /raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt --work /raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42 --own-root /raid1/xwan0900/fall_detection2/fd_datasets/30Hz_processed_clean_v2 --public-root /raid1/xwan0900/fall_detection2/fd_datasets/processed_v2 --m5-root /raid1/xwan0900/fall_detection2/fd_datasets/M5_hard_negatives_full_range_v4_20261003 --ptq-export /raid1/xwan0900/v4_int8_seed42_20261003 --runner M5BLECollector/artifacts/native-linux/replay --distillation-weight 2.0 --fused-norm --input-percentile 99.9 --output /raid1/xwan0900/v4_qat_fused_seed42_20261003)
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 /raid1/xwan0900/venvs/fall_v2_export/bin/python FallDetection30HzTraining/qat_v4.py --phase smoke "${common[@]}" > /raid1/xwan0900/v4_qat_fused_smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 /raid1/xwan0900/venvs/fall_v2_export/bin/python FallDetection30HzTraining/qat_v4.py --phase train "${common[@]}" > /raid1/xwan0900/v4_qat_fused_train.log 2>&1
```

Exact locked evaluation invocation used:

```bash
#!/bin/bash
set -euo pipefail
cd /raid1/xwan0900/fall_detection2_v4_qat
common=(--checkpoint /raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt --work /raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42 --own-root /raid1/xwan0900/fall_detection2/fd_datasets/30Hz_processed_clean_v2 --public-root /raid1/xwan0900/fall_detection2/fd_datasets/processed_v2 --m5-root /raid1/xwan0900/fall_detection2/fd_datasets/M5_hard_negatives_full_range_v4_20261003 --ptq-export /raid1/xwan0900/v4_int8_seed42_20261003 --runner M5BLECollector/artifacts/native-linux/replay --run /raid1/xwan0900/v4_qat_fused_seed42_20261003 --output /raid1/xwan0900/v4_qat_locked_seed42_20261003)
OPENBLAS_NUM_THREADS=4 /raid1/xwan0900/venvs/fall_v2_export/bin/python FallDetection30HzTraining/evaluate_locked_qat.py "${common[@]}" > /raid1/xwan0900/v4_qat_locked_evaluation.log 2>&1
```
