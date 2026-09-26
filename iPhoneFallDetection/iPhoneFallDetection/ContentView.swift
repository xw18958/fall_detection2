import Foundation
import SwiftUI

struct ContentView: View {
    @StateObject private var controller = FallDetectionController()

    var body: some View {
        NavigationStack {
            Form {
                Section("Input") {
                    Picker("Source", selection: $controller.sourceMode) {
                        ForEach(MotionSourceMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .onChange(of: controller.sourceMode) { _ in
                        controller.sourceModeChanged()
                    }

                    if controller.sourceMode == .csvReplay {
                        Picker("CSV", selection: $controller.selectedCSV) {
                            ForEach(FallDetectionController.sampleCSVs, id: \.self) { name in
                                Text(name).tag(name)
                            }
                        }
                    }

                    Button(controller.isRunning ? "Stop" : "Start") {
                        controller.isRunning ? controller.stop() : controller.start()
                    }
                    .buttonStyle(.borderedProminent)
                }

                Section("Motion stream") {
                    row("Connection", controller.isConnected ? "Connected" : "Disconnected")
                    row("Sensor", controller.sensorLocation.capitalized)
                    row("Measured rate", controller.measuredRateHz > 0 ? String(format: "%.1f Hz", controller.measuredRateHz) : "--")
                    row("Model rate", "25 Hz")
                    row("Window", "\(controller.bufferCount) / \(MotionWindowBuffer.capacity) samples")
                }

                if let sample = controller.latestSample {
                    Section("Latest 6-axis sample") {
                        vectorRow("Accel", sample.ax, sample.ay, sample.az, unit: "g")
                        vectorRow("Gyro", sample.gx, sample.gy, sample.gz, unit: "rad/s")
                    }
                }

                Section("Prototype dual-stream TCN") {
                    Text("Separate accelerometer and gyroscope TCN streams, 4 dilated residual blocks, attention pooling, gated fusion, and a 2-class head. Weights are deterministic random initialization for pipeline testing only.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    row("Normal", percent(controller.normalProbability))
                    row("Fall", percent(controller.fallProbability))
                    row("Inference", controller.inferenceCount > 0 ? String(format: "%.2f ms", controller.inferenceMilliseconds) : "waiting for 75 samples")
                    row("Inference count", "\(controller.inferenceCount)")
                }

                if let error = controller.errorMessage {
                    Section("Error") {
                        Text(error)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("AirPods Fall Prototype")
        }
    }

    @ViewBuilder
    private func row(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }

    @ViewBuilder
    private func vectorRow(_ title: String, _ x: Float, _ y: Float, _ z: Float, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            Text(String(format: "X %.4f   Y %.4f   Z %.4f %@", x, y, z, unit))
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    private func percent(_ value: Float) -> String {
        String(format: "%.1f%%", value * 100)
    }
}
