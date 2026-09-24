import CoreMotion
import Foundation

private struct MotionSample: Encodable {
    let timestamp: TimeInterval
    let ax: Double
    let ay: Double
    let az: Double
    let gx: Double
    let gy: Double
    let gz: Double
}

private struct MotionBatch: Encodable {
    let samples: [MotionSample]
}

private enum ServerState: String {
    case disconnected = "Disconnected"
    case connecting = "Connecting"
    case connected = "Connected"
}

private enum AirPodsState: String {
    case checking = "Checking"
    case connected = "Connected"
    case notConnected = "Not Connected"
}

private final class MotionCollector: NSObject, URLSessionWebSocketDelegate, CMHeadphoneMotionManagerDelegate {
    private let serverURL: URL
    private let motionManager = CMHeadphoneMotionManager()
    private let motionQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "AirPodsMotionCollector.motion"
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let stateQueue = DispatchQueue(label: "AirPodsMotionCollector.state")
    private var session: URLSession!
    private let encoder = JSONEncoder()

    private var socketTask: URLSessionWebSocketTask?
    private var batchTimer: DispatchSourceTimer?
    private var statusTimer: DispatchSourceTimer?
    private var pendingSamples: [MotionSample] = []
    private var latestSample: MotionSample?
    private var samplingTimestamps: [TimeInterval] = []
    private var sampleCount = 0
    private var samplingRate = 0.0
    private var isCollecting = false
    private var airPodsState: AirPodsState = .checking
    private var serverState: ServerState = .disconnected

    init(serverURL: URL) {
        self.serverURL = serverURL
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = false
        super.init()
        self.session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        self.motionManager.delegate = self
    }

    func run() {
        stateQueue.async { [weak self] in
            self?.motionManager.startConnectionStatusUpdates()
            self?.startTimers()
            self?.printStatus()
        }
    }

    func start() {
        let canStart = stateQueue.sync { () -> Bool in
            guard !isCollecting else {
                print("Collection is already running.")
                return false
            }
            // This establishes whether the hardware can provide motion samples. The
            // delegate and the first received sample establish the displayed connection state.
            guard motionManager.isDeviceMotionAvailable else {
                airPodsState = .notConnected
                print("AirPods: Not Connected (headphone motion is unavailable).")
                return false
            }
            airPodsState = .checking
            isCollecting = true
            if socketTask == nil {
                connectWebSocket()
            }
            return true
        }

        guard canStart else { return }
        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                self.stateQueue.async {
                    print("CoreMotion error: \(error.localizedDescription)")
                    self.airPodsState = .notConnected
                }
                return
            }
            guard let motion else { return }

            // CMDeviceMotion.timestamp is the original CoreMotion/AirPods timestamp.
            let sample = MotionSample(
                timestamp: motion.timestamp,
                ax: motion.userAcceleration.x + motion.gravity.x,
                ay: motion.userAcceleration.y + motion.gravity.y,
                az: motion.userAcceleration.z + motion.gravity.z,
                gx: motion.rotationRate.x,
                gy: motion.rotationRate.y,
                gz: motion.rotationRate.z
            )
            self.stateQueue.async {
                self.record(sample)
            }
        }
        print("Collection started.")
    }

    func stop() {
        let wasCollecting = stateQueue.sync { () -> Bool in
            guard isCollecting else { return false }
            isCollecting = false
            return true
        }
        guard wasCollecting else {
            print("Collection is already stopped.")
            return
        }

        motionManager.stopDeviceMotionUpdates()
        stateQueue.async { [weak self] in
            self?.flushBatch()
        }
        print("Collection stopped.")
    }

    func shutdown() {
        stop()
        stateQueue.sync {
            motionManager.stopConnectionStatusUpdates()
            batchTimer?.cancel()
            statusTimer?.cancel()
            batchTimer = nil
            statusTimer = nil
            socketTask?.cancel(with: .goingAway, reason: nil)
            socketTask = nil
            serverState = .disconnected
        }
        session.invalidateAndCancel()
    }

    func printHelp() {
        print("Commands: start, stop, status, quit")
    }

    func printCurrentStatus() {
        stateQueue.async { [weak self] in
            self?.printStatus()
        }
    }

    private func startTimers() {
        let batchTimer = DispatchSource.makeTimerSource(queue: stateQueue)
        batchTimer.schedule(deadline: .now() + .milliseconds(150), repeating: .milliseconds(150))
        batchTimer.setEventHandler { [weak self] in
            self?.flushBatch()
        }
        batchTimer.resume()
        self.batchTimer = batchTimer

        let statusTimer = DispatchSource.makeTimerSource(queue: stateQueue)
        statusTimer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
        statusTimer.setEventHandler { [weak self] in
            self?.printStatus()
        }
        statusTimer.resume()
        self.statusTimer = statusTimer
    }

    private func connectWebSocket() {
        serverState = .connecting
        let task = session.webSocketTask(with: serverURL)
        socketTask = task
        task.resume()
        receiveServerMessages(on: task)
    }

    private func receiveServerMessages(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                // The receiver does not need to send messages. Keep a receive pending so
                // URLSession reports a remote close promptly.
                self.receiveServerMessages(on: task)
            case .failure(let error):
                self.stateQueue.async {
                    guard self.socketTask === task else { return }
                    self.serverState = .disconnected
                    print("WebSocket disconnected: \(error.localizedDescription)")
                }
            }
        }
    }

    private func record(_ sample: MotionSample) {
        guard isCollecting else { return }
        airPodsState = .connected
        pendingSamples.append(sample)
        latestSample = sample
        sampleCount += 1

        samplingTimestamps.append(sample.timestamp)
        while let oldest = samplingTimestamps.first,
              sample.timestamp - oldest > 2.0 {
            samplingTimestamps.removeFirst()
        }
        if let first = samplingTimestamps.first,
           let last = samplingTimestamps.last,
           last > first {
            samplingRate = Double(samplingTimestamps.count - 1) / (last - first)
        }
    }

    private func flushBatch() {
        guard serverState == .connected,
              !pendingSamples.isEmpty,
              let socketTask else { return }

        let batch = MotionBatch(samples: pendingSamples)
        pendingSamples.removeAll(keepingCapacity: true)
        do {
            let data = try encoder.encode(batch)
            socketTask.send(.data(data)) { [weak self, weak socketTask] error in
                guard let error else { return }
                self?.stateQueue.async {
                    guard self?.socketTask === socketTask else { return }
                    self?.serverState = .disconnected
                    print("WebSocket send failed: \(error.localizedDescription)")
                }
            }
        } catch {
            print("Could not encode motion batch: \(error.localizedDescription)")
        }
    }

    private func printStatus() {
        print(
            "AirPods: \(airPodsState.rawValue) | Sampling rate: \(String(format: "%.1f", samplingRate)) Hz | " +
            "Samples collected: \(sampleCount) | Server: \(serverState.rawValue)"
        )
        if let sample = latestSample {
            print(
                "Latest: timestamp=\(String(format: "%.6f", sample.timestamp)) | " +
                "Accel=(\(String(format: "%.4f", sample.ax)), \(String(format: "%.4f", sample.ay)), \(String(format: "%.4f", sample.az))) | " +
                "Gyro=(\(String(format: "%.4f", sample.gx)), \(String(format: "%.4f", sample.gy)), \(String(format: "%.4f", sample.gz)))"
            )
        }
    }

    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        stateQueue.async { [weak self] in
            self?.airPodsState = .connected
        }
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        stateQueue.async { [weak self] in
            self?.airPodsState = .notConnected
        }
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        stateQueue.async { [weak self] in
            guard self?.socketTask === webSocketTask else { return }
            self?.serverState = .connected
            print("WebSocket connected to \(self?.serverURL.absoluteString ?? "server").")
        }
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        stateQueue.async { [weak self] in
            guard self?.socketTask === webSocketTask else { return }
            self?.serverState = .disconnected
        }
    }
}

private func serverURL(from arguments: [String]) -> URL? {
    guard let index = arguments.firstIndex(of: "--server"),
          arguments.indices.contains(index + 1) else {
        return URL(string: "ws://127.0.0.1:8765")
    }
    return URL(string: arguments[index + 1])
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.contains("--help") {
    print("Usage: AirPodsMotionCollector [--server ws://127.0.0.1:8765]")
    print("Commands: start, stop, status, quit")
    exit(EXIT_SUCCESS)
}

guard let url = serverURL(from: arguments),
      url.scheme == "ws" || url.scheme == "wss" else {
    fputs("--server must be a valid ws:// or wss:// URL.\n", stderr)
    exit(EXIT_FAILURE)
}

private let collector = MotionCollector(serverURL: url)
collector.run()
collector.printHelp()

DispatchQueue.global(qos: .userInitiated).async {
    while let command = readLine(strippingNewline: true)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        switch command {
        case "start":
            collector.start()
        case "stop":
            collector.stop()
        case "status":
            collector.printCurrentStatus()
        case "quit", "exit":
            collector.shutdown()
            exit(EXIT_SUCCESS)
        case "help", "":
            collector.printHelp()
        default:
            print("Unknown command: \(command)")
            collector.printHelp()
        }
    }
    collector.shutdown()
    exit(EXIT_SUCCESS)
}

RunLoop.main.run()
