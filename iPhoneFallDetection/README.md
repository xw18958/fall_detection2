# iPhone Fall Detection Prototype

This folder is an end-to-end iPhone prototype inside the existing `fall_detection2` repository.

## Goal of this stage

Validate the complete local path:

`AirPods Pro 2 -> iPhone Core Motion -> 25 Hz / 75 x 6 rolling window -> dual-stream TCN-like model -> Normal/Fall probabilities -> SwiftUI`

Model accuracy is intentionally out of scope. The prototype model has deterministic randomly initialized weights, but its architecture is deliberately similar to the trained project model rather than an arbitrary toy classifier:

- separate 3-axis accelerometer and 3-axis gyroscope streams;
- Conv1D stem per stream;
- four residual dilated TCN blocks with dilations 1, 2, 4, 8;
- attention pooling per stream;
- learned modality gate and fused representation;
- two-class `Normal/Fall` output head.

The input is always `[ax, ay, az, gx, gy, gz]`, where `a = userAcceleration + gravity`, matching the existing Mac collector.

## Sampling/window

- Input rate: approximately 25 Hz from `CMHeadphoneMotionManager`.
- No resampling in this prototype.
- Window: 3 seconds = 75 samples x 6 channels.
- Inference stride: 13 new samples, approximately 0.52 seconds.

## Two input modes

### Live AirPods

Uses `CMHeadphoneMotionManager` on the iPhone. One AirPods Pro 2 bud can provide the motion stream. The app shows connection state, active left/right sensor, measured sampling rate, live 6-axis values, buffer fill, and model output.

### CSV Replay

One earlier AirPods recording is bundled in `Resources/SampleData/`. It uses the same CSV schema as `MacCollector` and replays at 25 Hz through exactly the same rolling-window and inference path as live input. This is useful for debugging the app independently of the AirPods connection.

## Run on iPhone

1. Pull the latest `fall_detection2` repository on the Mac.
2. Open `iPhoneFallDetection/iPhoneFallDetection.xcodeproj` in Xcode.
3. Select the `iPhoneFallDetection` target.
4. Under **Signing & Capabilities**, select your Apple development team. The repository intentionally does not hard-code a personal team ID.
5. Connect the iPhone 15 Plus, select it as the run destination, and run the app.
6. Allow Motion & Fitness access when prompted.
7. Test **CSV Replay** first, then switch to **Live AirPods**.

## Important limitation

The displayed Normal/Fall probabilities have no clinical or detection meaning because the model weights are not trained. The purpose is to prove that on-device acquisition, windowing, neural-network inference, and UI all work before replacing the prototype model with the trained deployment model.

The existing `MacCollector` and remote-server code are unchanged.
