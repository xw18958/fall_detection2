import Foundation

/// Prototype on-device neural network used only to validate the full iPhone pipeline.
/// Architecture intentionally mirrors the trained project family:
/// separate accelerometer/gyroscope streams -> residual dilated TCN blocks ->
/// attention pooling -> modality gate -> fused classifier.
/// Weights are deterministic random initialization (fixed seed), not trained.
struct PrototypeDualStreamTCN {
    private let hidden = 16
    private let kernelSize = 5
    private let dilations = [1, 2, 4, 8]

    private let accelStem: Conv1DLayer
    private let gyroStem: Conv1DLayer
    private let accelBlocks: [ResidualTCNBlock]
    private let gyroBlocks: [ResidualTCNBlock]
    private let accelAttention: AttentionPool
    private let gyroAttention: AttentionPool
    private let gateLayer: LinearLayer
    private let fusionLayer: LinearLayer
    private let classifier: LinearLayer

    init(seed: UInt64 = 0xF411_D37E_C710_25AA) {
        var rng = SplitMix64(state: seed)
        let h = 16
        let k = 5
        let dils = [1, 2, 4, 8]
        accelStem = Conv1DLayer(inChannels: 3, outChannels: h, kernelSize: k, dilation: 1, rng: &rng)
        gyroStem = Conv1DLayer(inChannels: 3, outChannels: h, kernelSize: k, dilation: 1, rng: &rng)
        var aBlocks: [ResidualTCNBlock] = []
        var gBlocks: [ResidualTCNBlock] = []
        for d in dils {
            aBlocks.append(ResidualTCNBlock(channels: h, kernelSize: k, dilation: d, rng: &rng))
        }
        for d in dils {
            gBlocks.append(ResidualTCNBlock(channels: h, kernelSize: k, dilation: d, rng: &rng))
        }
        accelBlocks = aBlocks
        gyroBlocks = gBlocks
        accelAttention = AttentionPool(channels: h, rng: &rng)
        gyroAttention = AttentionPool(channels: h, rng: &rng)
        gateLayer = LinearLayer(inputSize: h * 2, outputSize: h, rng: &rng)
        fusionLayer = LinearLayer(inputSize: h * 3, outputSize: h * 2, rng: &rng)
        classifier = LinearLayer(inputSize: h * 2, outputSize: 2, rng: &rng)
    }

    func predict(window: [[Float]]) -> (normal: Float, fall: Float) {
        precondition(window.count == MotionWindowBuffer.capacity, "Expected exactly 75 samples")
        precondition(window.allSatisfy { $0.count == 6 }, "Expected six IMU channels")

        let accel = window.map { Array($0[0..<3]) }
        let gyro = window.map { Array($0[3..<6]) }

        let accFeature = encode(accel, stem: accelStem, blocks: accelBlocks, attention: accelAttention)
        let gyrFeature = encode(gyro, stem: gyroStem, blocks: gyroBlocks, attention: gyroAttention)

        let gate = gateLayer.forward(accFeature + gyrFeature).map(sigmoid)
        var mixed = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden {
            mixed[i] = gate[i] * accFeature[i] + (1 - gate[i]) * gyrFeature[i]
        }

        let fused = fusionLayer.forward(accFeature + gyrFeature + mixed).map(gelu)
        let logits = classifier.forward(fused)
        let probs = softmax(logits)
        return (normal: probs[0], fall: probs[1])
    }

    private func encode(
        _ input: [[Float]],
        stem: Conv1DLayer,
        blocks: [ResidualTCNBlock],
        attention: AttentionPool
    ) -> [Float] {
        var x = stem.forward(input).map { $0.map(gelu) }
        for block in blocks {
            x = block.forward(x)
        }
        return attention.forward(x)
    }
}

private struct Conv1DLayer {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let dilation: Int
    let weights: [Float]
    let bias: [Float]

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int, rng: inout SplitMix64) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        let fanIn = Float(max(1, inChannels * kernelSize))
        let scale = sqrt(2.0 / fanIn)
        self.weights = (0..<(inChannels * outChannels * kernelSize)).map { _ in rng.normal() * scale }
        self.bias = [Float](repeating: 0, count: outChannels)
    }

    func forward(_ x: [[Float]]) -> [[Float]] {
        let time = x.count
        guard time > 0 else { return [] }
        let padding = ((kernelSize - 1) * dilation) / 2
        var output = Array(repeating: [Float](repeating: 0, count: outChannels), count: time)

        for t in 0..<time {
            for oc in 0..<outChannels {
                var sum = bias[oc]
                for k in 0..<kernelSize {
                    let sourceT = t + (k * dilation) - padding
                    guard sourceT >= 0, sourceT < time else { continue }
                    for ic in 0..<inChannels {
                        let wi = ((oc * inChannels + ic) * kernelSize) + k
                        sum += x[sourceT][ic] * weights[wi]
                    }
                }
                output[t][oc] = sum
            }
        }
        return output
    }
}

private struct ResidualTCNBlock {
    let conv1: Conv1DLayer
    let conv2: Conv1DLayer

    init(channels: Int, kernelSize: Int, dilation: Int, rng: inout SplitMix64) {
        conv1 = Conv1DLayer(inChannels: channels, outChannels: channels, kernelSize: kernelSize, dilation: dilation, rng: &rng)
        conv2 = Conv1DLayer(inChannels: channels, outChannels: channels, kernelSize: kernelSize, dilation: dilation, rng: &rng)
    }

    func forward(_ x: [[Float]]) -> [[Float]] {
        let h1 = conv1.forward(x).map { $0.map(gelu) }
        let h2 = conv2.forward(h1)
        var output = x
        for t in 0..<output.count {
            for c in 0..<output[t].count {
                output[t][c] = gelu(output[t][c] + h2[t][c])
            }
        }
        return output
    }
}

private struct AttentionPool {
    let scoreWeights: [Float]
    let scoreBias: Float

    init(channels: Int, rng: inout SplitMix64) {
        let scale = 1.0 / sqrt(Float(max(channels, 1)))
        scoreWeights = (0..<channels).map { _ in rng.normal() * scale }
        scoreBias = 0
    }

    func forward(_ x: [[Float]]) -> [Float] {
        guard let first = x.first else { return [] }
        let channels = first.count
        let scores = x.map { row -> Float in
            zip(row, scoreWeights).reduce(scoreBias) { $0 + ($1.0 * $1.1) }
        }
        let weights = softmax(scores)
        var pooled = [Float](repeating: 0, count: channels)
        for t in 0..<x.count {
            for c in 0..<channels {
                pooled[c] += weights[t] * x[t][c]
            }
        }
        return pooled
    }
}

private struct LinearLayer {
    let inputSize: Int
    let outputSize: Int
    let weights: [Float]
    let bias: [Float]

    init(inputSize: Int, outputSize: Int, rng: inout SplitMix64) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        let scale = sqrt(2.0 / Float(max(inputSize, 1)))
        self.weights = (0..<(inputSize * outputSize)).map { _ in rng.normal() * scale }
        self.bias = [Float](repeating: 0, count: outputSize)
    }

    func forward(_ x: [Float]) -> [Float] {
        precondition(x.count == inputSize)
        var y = bias
        for o in 0..<outputSize {
            var sum = y[o]
            let offset = o * inputSize
            for i in 0..<inputSize {
                sum += x[i] * weights[offset + i]
            }
            y[o] = sum
        }
        return y
    }
}

private func gelu(_ x: Float) -> Float {
    let c: Float = 0.7978845608
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x * x * x)))
}

private func sigmoid(_ x: Float) -> Float {
    1 / (1 + exp(-x))
}

private func softmax(_ x: [Float]) -> [Float] {
    guard let maxValue = x.max() else { return [] }
    let exps = x.map { exp($0 - maxValue) }
    let denom = max(exps.reduce(0, +), Float.leastNonzeroMagnitude)
    return exps.map { $0 / denom }
}

private struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func uniform() -> Float {
        let value = next() >> 11
        return Float(Double(value) / Double(1 << 53))
    }

    mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7)
        let u2 = uniform()
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }
}
