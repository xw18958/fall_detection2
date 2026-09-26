import Foundation

struct MotionSample: Equatable, Sendable {
    let timestamp: TimeInterval
    let sensorLocation: String
    let ax: Float
    let ay: Float
    let az: Float
    let gx: Float
    let gy: Float
    let gz: Float

    var vector: [Float] { [ax, ay, az, gx, gy, gz] }
}

struct ModelOutput: Sendable {
    let normalProbability: Float
    let fallProbability: Float
    let inferenceMilliseconds: Double
}

enum MotionSourceMode: String, CaseIterable, Identifiable {
    case liveAirPods = "Live AirPods"
    case csvReplay = "CSV Replay"

    var id: String { rawValue }
}
