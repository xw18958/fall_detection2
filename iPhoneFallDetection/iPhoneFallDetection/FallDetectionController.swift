import Combine
import Foundation
import QuartzCore

@MainActor
final class FallDetectionController: ObservableObject {
    static let sampleCSVs = [
        "session_20260925_065155_20ce9a51",
    ]

    @Published var sourceMode: MotionSourceMode = .liveAirPods
    @Published var selectedCSV = sampleCSVs[0]
    @Published private(set) var isRunning = false
    @Published private(set) var isConnected = false
    @Published private(set) var sensorLocation = "--"
    @Published private(set) var measuredRateHz: Double = 0
    @Published private(set) var bufferCount = 0
    @Published private(set) var latestSample: MotionSample?
    @Published private(set) var normalProbability: Float = 0.5
    @Published private(set) var fallProbability: Float = 0.5
    @Published private(set) var inferenceMilliseconds: Double = 0
    @Published private(set) var inferenceCount = 0
    @Published var errorMessage: String?

    private let airPods = AirPodsMotionManager()
    private let csvReplay = CSVReplaySource()
    private let model = PrototypeDualStreamTCN()
    private var window = MotionWindowBuffer()
    private var recentTimestamps: [TimeInterval] = []
    private var samplesSinceInference = 0
    private let inferenceStrideSamples = 13 // about 0.52 s at 25 Hz

    init() {
        airPods.onSample = { [weak self] sample in
            self?.handle(sample)
        }
        airPods.onConnectionChanged = { [weak self] connected, location in
            guard let self else { return }
            self.isConnected = connected
            self.sensorLocation = connected ? location : "--"
        }
        airPods.onError = { [weak self] message in
            self?.errorMessage = message
        }

        csvReplay.onSample = { [weak self] sample in
            guard let self else { return }
            self.isConnected = true
            self.sensorLocation = "CSV"
            self.handle(sample)
        }
        csvReplay.onFinished = { [weak self] in
            guard let self else { return }
            self.isRunning = false
            self.isConnected = false
            self.sensorLocation = "--"
        }
        csvReplay.onError = { [weak self] message in
            self?.errorMessage = message
        }
    }

    func start() {
        guard !isRunning else { return }
        resetStreamState()
        isRunning = true
        errorMessage = nil

        switch sourceMode {
        case .liveAirPods:
            airPods.start()
        case .csvReplay:
            csvReplay.start(resourceName: selectedCSV)
        }
    }

    func stop() {
        airPods.stop()
        csvReplay.stop()
        isRunning = false
        isConnected = false
        sensorLocation = "--"
    }

    func sourceModeChanged() {
        if isRunning {
            stop()
        }
        resetStreamState()
    }

    private func resetStreamState() {
        window.reset()
        recentTimestamps.removeAll(keepingCapacity: true)
        samplesSinceInference = 0
        measuredRateHz = 0
        bufferCount = 0
        latestSample = nil
        normalProbability = 0.5
        fallProbability = 0.5
        inferenceMilliseconds = 0
        inferenceCount = 0
    }

    private func handle(_ sample: MotionSample) {
        latestSample = sample
        updateMeasuredRate(timestamp: sample.timestamp)
        window.append(sample)
        bufferCount = window.count
        samplesSinceInference += 1

        guard window.isReady, samplesSinceInference >= inferenceStrideSamples,
              let input = window.modelInput() else { return }
        samplesSinceInference = 0

        let startTime = CACurrentMediaTime()
        let prediction = model.predict(window: input)
        inferenceMilliseconds = (CACurrentMediaTime() - startTime) * 1000.0
        normalProbability = prediction.normal
        fallProbability = prediction.fall
        inferenceCount += 1
    }

    private func updateMeasuredRate(timestamp: TimeInterval) {
        recentTimestamps.append(timestamp)
        while let first = recentTimestamps.first, timestamp - first > 2.0 {
            recentTimestamps.removeFirst()
        }
        guard let first = recentTimestamps.first,
              let last = recentTimestamps.last,
              recentTimestamps.count > 1,
              last > first else {
            measuredRateHz = 0
            return
        }
        measuredRateHz = Double(recentTimestamps.count - 1) / (last - first)
    }
}
