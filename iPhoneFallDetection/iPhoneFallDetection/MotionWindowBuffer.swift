import Foundation

struct MotionWindowBuffer {
    static let sampleRateHz = 25
    static let windowSeconds = 3
    static let capacity = sampleRateHz * windowSeconds // 75

    private(set) var samples: [MotionSample] = []

    var count: Int { samples.count }
    var isReady: Bool { samples.count == Self.capacity }

    mutating func append(_ sample: MotionSample) {
        samples.append(sample)
        if samples.count > Self.capacity {
            samples.removeFirst(samples.count - Self.capacity)
        }
    }

    mutating func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    func modelInput() -> [[Float]]? {
        guard isReady else { return nil }
        return samples.map(\.vector)
    }
}
