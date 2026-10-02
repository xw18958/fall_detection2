# Reproduce V4 PTQ and collector package

Use the `codex/v4-int8-collector` source in an isolated checkout. The exact original checkpoint and dataset splits remain read-only; use a new export output directory. Keep TensorFlow dependencies separate from the training environment. No training is needed.

On the GPU server:

```bash
cd /raid1/xwan0900/fall_detection2_v4_quantization
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2
V4_EXPORT_DIR=/raid1/xwan0900/v4_int8_seed42_REPRODUCE
V4_CHECKPOINT=/raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt
V4_RUN=/raid1/xwan0900/fall_detection2/FallDetection30HzTraining/runs/v4_weighted_20261003/seed42
V4_PYTHON=/raid1/xwan0900/venvs/fall_v2_export/bin/python
V4_ARGS=(--checkpoint "$V4_CHECKPOINT" --work "$V4_RUN" --own-root /raid1/xwan0900/fall_detection2/fd_datasets/30Hz_processed_clean_v2 --public-root /raid1/xwan0900/fall_detection2/fd_datasets/processed_v2 --m5-root /raid1/xwan0900/fall_detection2/fd_datasets/M5_hard_negatives_full_range_v4_20261003 --output "$V4_EXPORT_DIR")
"$V4_PYTHON" FallDetection30HzTraining/export_v4_tflite.py --phase smoke "${V4_ARGS[@]}"
"$V4_PYTHON" FallDetection30HzTraining/export_v4_tflite.py --phase export "${V4_ARGS[@]}"
"$V4_PYTHON" FallDetection30HzTraining/export_v4_tflite.py --phase validation "${V4_ARGS[@]}"
# Fixed first PTQ threshold: do not select another model/calibration/threshold on test.
"$V4_PYTHON" FallDetection30HzTraining/export_v4_tflite.py --phase test "${V4_ARGS[@]}"
"$V4_PYTHON" M5BLECollector/tools/prepare_v4_package.py --checkpoint "$V4_CHECKPOINT" --export "$V4_EXPORT_DIR"
```

`environment.json` records observed dependencies. The export environment's SciPy warns about NumPy 2.5.3 being beyond its declared supported range; completed smoke, parity and evaluation succeeded. No dependencies were installed into the training environment. A byte-identical re-export also depends on the recorded TensorFlow converter version and platform; validate its hash before using it with these package pins.

On the Mac, copy only the exact model/config/replay package into an isolated checkout. The original run uses `M5BLECollector/artifacts/v4_seed42` for private artifacts. The source pins intentionally reject a different export hash. The provided delivery package contains the final Micro goldens, so `prepare_model.py` can install them directly. If reproducing Micro goldens from the desktop-generated replay header, place that header and the exact config/model into the isolated firmware directory, build the native runner, run `lock_micro_replay.py`, and verify the resulting header SHA against detector_baseline.json before `prepare_model.py`.

```bash
V4_CMAKE=/Users/henrywang/.platformio/packages/tool-cmake/bin/cmake
V4_PIO=/Users/henrywang/.venvs/platformio39/bin/python
V4_NUMPY_PY=/Users/henrywang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
python3 M5BLECollector/tools/prepare_model.py --source /Users/henrywang/Documents/fall_detection2/deployment_artifacts/v4_seed42_20261003/package
"$V4_CMAKE" -S M5BLECollector/tests/native_model -B M5BLECollector/artifacts/native-espnn
"$V4_CMAKE" --build M5BLECollector/artifacts/native-espnn -j 8
M5BLECollector/artifacts/native-espnn/replay --kernel-test
M5BLECollector/artifacts/native-espnn/replay M5BLECollector/firmware/main/model.tflite
"$V4_NUMPY_PY" M5BLECollector/tools/verify_input_pipeline.py
M5BLECollector/tools/smoke-tests.sh
swift build --package-path M5BLECollector/macos --product M5BLECollector
"$V4_PIO" -m platformio run -d M5BLECollector/firmware
python3 M5BLECollector/tools/verify_preservation.py --firmware-binary M5BLECollector/firmware/.pio/build/m5stickc-plus2-ble/firmware.bin --firmware-elf M5BLECollector/firmware/.pio/build/m5stickc-plus2-ble/firmware.elf --nm /Users/henrywang/.platformio/packages/toolchain-xtensa-esp-elf/bin/xtensa-esp32-elf-nm
```

Native bulk replay accepts `MODEL INPUT_BYTES OUTPUT_BYTES`: each input window is 540 signed INT8 bytes and output is two signed INT8 logits. `export_v4_tflite.py` saves source-separated `*_inputs.npy` and `*_outputs.npz`; concatenate input arrays in sorted source filename order and replay the exact bytes. `compare_micro_results.py` joins outputs back to saved labels/probabilities and checks exact window counts. `run_native_bulk.py` executes deterministic shards and records counts. The ESP-NN native path uses the ESP32 generic fully-connected and optimized portable softmax kernels; native timer stubs affect diagnostic counters only. The original portable reference path is available with `-DFALL_ESP_NN_PORTABLE=OFF`; every validation/test output byte matched the ESP-NN path.

`lock_micro_replay.py --package PACKAGE --runner MICRO_RUNNER` stores verified Micro replay outputs and retains desktop deltas. Re-locking package bytes never authorizes regenerating every baseline: only the model/configuration/replay and listed detector/host identity updates are authorized here. The current pinned package is already locked and passes the original four-LSB replay tolerance.

No upload, erase, flash-slot write, Internet release or board reset is included in these commands. For physical installation, first finish/save acquisition and confirm pending=0, identify installed firmware/transport, back up active app/boot/settings and inspect active OTA slot. Use only the local recovery OTA application-image path into the inactive slot. Device verification and performance acceptance remain pending.

To use the new receiver with the preserved recording directory after stopping the old receiver:

```bash
/Users/henrywang/Documents/fall_detection2/deployment_artifacts/v4_seed42_20261003/M5BLECollector --recordings /Users/henrywang/Documents/fall_detection2/M5BLECollector/macos/recordings
```

Existing journal recovery still requires the original model/boot/session identities. Never resume an incomplete old-boot journal under the new model/boot. The recording directory's lock prevents two receivers from opening it together.
