import Foundation

final class CSVReplaySource {
    var onSample: ((MotionSample) -> Void)?
    var onFinished: (() -> Void)?
    var onError: ((String) -> Void)?

    private var replayTask: Task<Void, Never>?
    private(set) var isRunning = false

    func start(resourceName: String) {
        stop()
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "csv", subdirectory: "SampleData")
                ?? Bundle.main.url(forResource: resourceName, withExtension: "csv") else {
            onError?("Could not find bundled CSV: \(resourceName).csv")
            return
        }

        do {
            let samples = try Self.load(url: url)
            guard !samples.isEmpty else {
                onError?("CSV contains no usable 6-axis samples")
                return
            }
            isRunning = true
            replayTask = Task { @MainActor [weak self] in
                guard let self else { return }
                for sample in samples {
                    guard !Task.isCancelled else { break }
                    self.onSample?(sample)
                    try? await Task.sleep(nanoseconds: 40_000_000) // 25 Hz
                }
                let completed = !Task.isCancelled
                self.isRunning = false
                if completed { self.onFinished?() }
            }
        } catch {
            onError?(error.localizedDescription)
        }
    }

    func stop() {
        replayTask?.cancel()
        replayTask = nil
        isRunning = false
    }

    private static func load(url: URL) throws -> [MotionSample] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(whereSeparator: \.isNewline)
        guard let headerLine = lines.first else { return [] }
        let header = headerLine.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        let column = Dictionary(uniqueKeysWithValues: header.enumerated().map { ($1, $0) })
        let required = ["coremotion_timestamp", "sensor_location", "ax", "ay", "az", "gx", "gy", "gz"]
        guard required.allSatisfy({ column[$0] != nil }) else {
            throw CSVReplayError.missingColumns
        }

        func value(_ fields: [Substring], _ name: String) -> String? {
            guard let index = column[name], index < fields.count else { return nil }
            return String(fields[index])
        }

        var samples: [MotionSample] = []
        samples.reserveCapacity(max(0, lines.count - 1))
        for line in lines.dropFirst() {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false)
            guard
                let timestampText = value(fields, "coremotion_timestamp"), let timestamp = Double(timestampText),
                let axText = value(fields, "ax"), let ax = Float(axText),
                let ayText = value(fields, "ay"), let ay = Float(ayText),
                let azText = value(fields, "az"), let az = Float(azText),
                let gxText = value(fields, "gx"), let gx = Float(gxText),
                let gyText = value(fields, "gy"), let gy = Float(gyText),
                let gzText = value(fields, "gz"), let gz = Float(gzText)
            else { continue }

            samples.append(
                MotionSample(
                    timestamp: timestamp,
                    sensorLocation: value(fields, "sensor_location") ?? "csv",
                    ax: ax, ay: ay, az: az,
                    gx: gx, gy: gy, gz: gz
                )
            )
        }
        return samples
    }
}

private enum CSVReplayError: LocalizedError {
    case missingColumns

    var errorDescription: String? {
        switch self {
        case .missingColumns:
            return "CSV must contain coremotion_timestamp, sensor_location, ax, ay, az, gx, gy, gz"
        }
    }
}
