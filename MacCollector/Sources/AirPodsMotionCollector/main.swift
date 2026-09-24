import CoreMotion
import Darwin
import Dispatch
import Foundation

private struct MotionValues {
    let coreMotionTimestamp: TimeInterval
    let sensorLocation: String
    let userAccelX: Double
    let userAccelY: Double
    let userAccelZ: Double
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double
    let ax: Double
    let ay: Double
    let az: Double
    let gx: Double
    let gy: Double
    let gz: Double
}

private struct MotionSample: Encodable {
    let seq: Int64
    let coreMotionTimestamp: TimeInterval
    let hostTimestampUTC: TimeInterval
    let sensorLocation: String
    let userAccelX: Double
    let userAccelY: Double
    let userAccelZ: Double
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double
    let ax: Double
    let ay: Double
    let az: Double
    let gx: Double
    let gy: Double
    let gz: Double

    enum CodingKeys: String, CodingKey {
        case seq
        case coreMotionTimestamp = "coremotion_timestamp"
        case hostTimestampUTC = "host_timestamp_utc"
        case sensorLocation = "sensor_location"
        case userAccelX = "user_accel_x"
        case userAccelY = "user_accel_y"
        case userAccelZ = "user_accel_z"
        case gravityX = "gravity_x"
        case gravityY = "gravity_y"
        case gravityZ = "gravity_z"
        case ax, ay, az, gx, gy, gz
    }
}

private struct MotionBatch: Encodable {
    let schemaVersion = 2
    let sessionID: String
    let samples: [MotionSample]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sessionID = "session_id"
        case samples
    }
}

private enum ServerState: String {
    case disconnected = "Disconnected"
    case connecting = "Connecting"
    case connected = "Connected"
}

private enum AirPodsState: String {
    case checking = "Checking"
    case connected = "Connected"
    case notConnected = "Disconnected"
}

private final class LocalCSVRecorder {
    let fileURL: URL
    private let handle: FileHandle

    init(sessionID: String) throws {
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("recordings", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        fileURL = directory.appendingPathComponent("\(sessionID).csv")
        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        handle = try FileHandle(forWritingTo: fileURL)
        let header = "seq,coremotion_timestamp,host_timestamp_utc,sensor_location,user_accel_x,user_accel_y,user_accel_z,gravity_x,gravity_y,gravity_z,ax,ay,az,gx,gy,gz\n"
        try handle.write(contentsOf: Data(header.utf8))
    }

    func append(_ sample: MotionSample) throws {
        let row = [
            String(sample.seq),
            format(sample.coreMotionTimestamp),
            format(sample.hostTimestampUTC),
            sample.sensorLocation,
            format(sample.userAccelX),
            format(sample.userAccelY),
            format(sample.userAccelZ),
            format(sample.gravityX),
            format(sample.gravityY),
            format(sample.gravityZ),
            format(sample.ax),
            format(sample.ay),
            format(sample.az),
            format(sample.gx),
            format(sample.gy),
            format(sample.gz),
        ].joined(separator: ",") + "\n"
        try handle.write(contentsOf: Data(row.utf8))
    }

    func close() {
        try? handle.synchronize()
        try? handle.close()
    }

    private func format(_ value: Double) -> String {
        String(format: "%.9f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

private final class MotionCollector: NSObject, URLSessionWebSocketDelegate, CMHeadphoneMotionManagerDelegate {
    private let serverURL: URL
    private let sessionID: String
    private let localRecorder: LocalCSVRecorder
    private let motionManager = CMHeadphoneMotionManager()
    private let motionQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "AirPodsMotionCollector.motion"
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let stateQueue = DispatchQueue(label: "AirPodsMotionCollector.state")
    private let encoder = JSONEncoder()
    private var urlSession: URLSession!

    private var socketTask: URLSessionWebSocketTask?
    private var batchTimer: DispatchSourceTimer?
    private var statusTimer: DispatchSourceTimer?
    private var watchdogTimer: DispatchSourceTimer?
    private var reconnectWorkItem: DispatchWorkItem?
    private var motionRestartWorkItem: DispatchWorkItem?

    private var pendingSamples: [MotionSample] = []
    private let maxPendingSamples = 15_000
    private let maxSamplesPerBatch = 250
    private var sendInFlight = false
    private var droppedNetworkSamples = 0

    private var latestSample: MotionSample?
    private var samplingTimestamps: [TimeInterval] = []
    private var sampleCount: Int64 = 0
    private var nextSequence: Int64 = 0
    private var samplingRate = 0.0
    private var activeSensorLocation = "unknown"
    private var airPodsState: AirPodsState = .checking
    private var serverState: ServerState = .disconnected

    private var userWantsCollection = true
    private var motionStreamActive = false
    private var streamStartedUptime: TimeInterval?
    private var lastSampleUptime: TimeInterval?
    private var serverReconnectDelay = 1.0
    private var isShuttingDown = false

    init(serverURL: URL) throws {
        self.serverURL = serverURL
        self.sessionID = Self.makeSessionID()
        self.localRecorder = try LocalCSVRecorder(sessionID: sessionID)
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = false
        super.init()
        self.urlSession = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        self.motionManager.delegate = self
    }

    func run() {
        stateQueue.async { [weak self] in
            guard let self else { return }
            self.motionManager.startConnectionStatusUpdates()
            self.startTimers()
            self.connectWebSocketIfNeeded()
            print("AirPods motion collector started.")
            print("Session: \(self.sessionID)")
            print("Local backup: \(self.localRecorder.fileURL.path)")
            if self.motionManager.isDeviceMotionAvailable {
                self.startMotionStreamIfNeeded(reason: "startup")
            } else {
                print("Waiting for a motion-capable AirPod...")
            }
            self.printStatus()
        }
    }

    func shutdown() {
        var finalSamples: [MotionSample] = []
        var finalTask: URLSessionWebSocketTask?

        stateQueue.sync {
            guard !isShuttingDown else { return }
            isShuttingDown = true
            userWantsCollection = false
            cancelReconnectWork()
            motionRestartWorkItem?.cancel()
            motionRestartWorkItem = nil
            batchTimer?.cancel()
            statusTimer?.cancel()
            watchdogTimer?.cancel()
            batchTimer = nil
            statusTimer = nil
            watchdogTimer = nil

            if motionStreamActive || motionManager.isDeviceMotionActive {
                motionManager.stopDeviceMotionUpdates()
            }
            motionStreamActive = false
            motionManager.stopConnectionStatusUpdates()

            finalSamples = pendingSamples
            pendingSamples.removeAll(keepingCapacity: false)
            if serverState == .connected {
                finalTask = socketTask
            }
            localRecorder.close()
        }

        bestEffortSendFinal(finalSamples, using: finalTask)
        finalTask?.cancel(with: .normalClosure, reason: nil)
        urlSession.invalidateAndCancel()

        print("Local recording saved: \(localRecorder.fileURL.path)")
        if !finalSamples.isEmpty && finalTask == nil {
            print("Server was disconnected; \(finalSamples.count) final sample(s) remain only in the local backup.")
        }
    }

    func printCurrentStatus() {
        stateQueue.async { [weak self] in
            self?.printStatus()
        }
    }

    func printHelp() {
        print("Commands: status, quit")
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

        let watchdogTimer = DispatchSource.makeTimerSource(queue: stateQueue)
        watchdogTimer.schedule(deadline: .now() + .seconds(1), repeating: .milliseconds(500))
        watchdogTimer.setEventHandler { [weak self] in
            self?.checkMotionWatchdog()
        }
        watchdogTimer.resume()
        self.watchdogTimer = watchdogTimer
    }

    private func startMotionStreamIfNeeded(reason: String) {
        guard userWantsCollection,
              !isShuttingDown,
              !motionStreamActive,
              motionManager.isDeviceMotionAvailable else { return }

        motionRestartWorkItem?.cancel()
        motionRestartWorkItem = nil
        motionStreamActive = true
        streamStartedUptime = Self.uptime()
        lastSampleUptime = nil
        samplingTimestamps.removeAll(keepingCapacity: true)
        samplingRate = 0

        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                self.stateQueue.async {
                    guard !self.isShuttingDown else { return }
                    print("Core Motion error: \(error.localizedDescription)")
                    self.restartMotionStream(reason: "Core Motion error")
                }
                return
            }
            guard let motion else { return }

            let hostTimestampUTC = Date().timeIntervalSince1970
            let sensorLocation = Self.sensorLocationText(motion.sensorLocation)
            let values = MotionValues(
                coreMotionTimestamp: motion.timestamp,
                sensorLocation: sensorLocation,
                userAccelX: motion.userAcceleration.x,
                userAccelY: motion.userAcceleration.y,
                userAccelZ: motion.userAcceleration.z,
                gravityX: motion.gravity.x,
                gravityY: motion.gravity.y,
                gravityZ: motion.gravity.z,
                ax: motion.userAcceleration.x + motion.gravity.x,
                ay: motion.userAcceleration.y + motion.gravity.y,
                az: motion.userAcceleration.z + motion.gravity.z,
                gx: motion.rotationRate.x,
                gy: motion.rotationRate.y,
                gz: motion.rotationRate.z
            )
            self.stateQueue.async {
                self.record(values, hostTimestampUTC: hostTimestampUTC)
            }
        }

        if reason != "startup" {
            print("Starting AirPods motion stream (\(reason))...")
        }
    }

    private func restartMotionStream(reason: String) {
        guard userWantsCollection, !isShuttingDown else { return }
        motionRestartWorkItem?.cancel()
        motionRestartWorkItem = nil

        if motionStreamActive || motionManager.isDeviceMotionActive {
            motionManager.stopDeviceMotionUpdates()
        }
        motionStreamActive = false
        streamStartedUptime = nil
        lastSampleUptime = nil
        samplingTimestamps.removeAll(keepingCapacity: true)
        samplingRate = 0

        guard airPodsState != .notConnected else { return }

        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.motionRestartWorkItem = nil
            self.startMotionStreamIfNeeded(reason: reason)
        }
        motionRestartWorkItem = workItem
        stateQueue.asyncAfter(deadline: .now() + .milliseconds(500), execute: workItem)
    }

    private func checkMotionWatchdog() {
        guard userWantsCollection,
              motionStreamActive,
              airPodsState != .notConnected,
              !isShuttingDown else { return }

        let now = Self.uptime()
        let reference = lastSampleUptime ?? streamStartedUptime
        guard let reference, now - reference >= 2.0 else { return }

        print("No AirPods motion samples for 2 seconds — restarting Core Motion...")
        restartMotionStream(reason: "no-sample watchdog")
    }

    private func record(_ values: MotionValues, hostTimestampUTC: TimeInterval) {
        guard userWantsCollection, motionStreamActive, !isShuttingDown else { return }

        let sample = MotionSample(
            seq: nextSequence,
            coreMotionTimestamp: values.coreMotionTimestamp,
            hostTimestampUTC: hostTimestampUTC,
            sensorLocation: values.sensorLocation,
            userAccelX: values.userAccelX,
            userAccelY: values.userAccelY,
            userAccelZ: values.userAccelZ,
            gravityX: values.gravityX,
            gravityY: values.gravityY,
            gravityZ: values.gravityZ,
            ax: values.ax,
            ay: values.ay,
            az: values.az,
            gx: values.gx,
            gy: values.gy,
            gz: values.gz
        )
        nextSequence += 1

        do {
            try localRecorder.append(sample)
        } catch {
            print("Local recording error: \(error.localizedDescription)")
        }

        if pendingSamples.count >= maxPendingSamples {
            pendingSamples.removeFirst()
            droppedNetworkSamples += 1
        }
        pendingSamples.append(sample)

        airPodsState = .connected
        activeSensorLocation = sample.sensorLocation
        latestSample = sample
        sampleCount += 1
        lastSampleUptime = Self.uptime()

        samplingTimestamps.append(sample.coreMotionTimestamp)
        while let oldest = samplingTimestamps.first,
              sample.coreMotionTimestamp - oldest > 2.0 {
            samplingTimestamps.removeFirst()
        }
        if let first = samplingTimestamps.first,
           let last = samplingTimestamps.last,
           last > first {
            samplingRate = Double(samplingTimestamps.count - 1) / (last - first)
        }
    }

    private func connectWebSocketIfNeeded() {
        guard !isShuttingDown,
              socketTask == nil,
              serverState != .connecting,
              serverState != .connected else { return }

        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        serverState = .connecting
        let task = urlSession.webSocketTask(with: serverURL)
        socketTask = task
        task.resume()
        receiveServerMessages(on: task)
    }

    private func receiveServerMessages(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.receiveServerMessages(on: task)
            case .failure(let error):
                self.stateQueue.async {
                    self.handleSocketFailure(task, message: error.localizedDescription)
                }
            }
        }
    }

    private func flushBatch() {
        guard serverState == .connected,
              !sendInFlight,
              !pendingSamples.isEmpty,
              let socketTask else { return }

        let count = min(maxSamplesPerBatch, pendingSamples.count)
        let samples = Array(pendingSamples.prefix(count))
        pendingSamples.removeFirst(count)
        let batch = MotionBatch(sessionID: sessionID, samples: samples)

        do {
            let data = try encoder.encode(batch)
            sendInFlight = true
            socketTask.send(.data(data)) { [weak self, weak socketTask] error in
                guard let self else { return }
                self.stateQueue.async {
                    self.sendInFlight = false
                    if let error {
                        self.pendingSamples.insert(contentsOf: samples, at: 0)
                        self.trimNetworkBufferIfNeeded()
                        if let socketTask {
                            self.handleSocketFailure(socketTask, message: error.localizedDescription)
                        }
                    } else {
                        self.flushBatch()
                    }
                }
            }
        } catch {
            pendingSamples.insert(contentsOf: samples, at: 0)
            trimNetworkBufferIfNeeded()
            print("Could not encode motion batch: \(error.localizedDescription)")
        }
    }

    private func trimNetworkBufferIfNeeded() {
        if pendingSamples.count > maxPendingSamples {
            let excess = pendingSamples.count - maxPendingSamples
            pendingSamples.removeFirst(excess)
            droppedNetworkSamples += excess
        }
    }

    private func handleSocketFailure(_ task: URLSessionWebSocketTask, message: String) {
        guard socketTask === task, !isShuttingDown else { return }
        socketTask = nil
        serverState = .disconnected
        sendInFlight = false
        task.cancel(with: .goingAway, reason: nil)
        print("Server disconnected: \(message)")
        scheduleServerReconnect()
    }

    private func scheduleServerReconnect() {
        guard !isShuttingDown, reconnectWorkItem == nil else { return }
        let delay = serverReconnectDelay
        serverReconnectDelay = min(serverReconnectDelay * 2.0, 10.0)
        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.reconnectWorkItem = nil
            self.connectWebSocketIfNeeded()
        }
        reconnectWorkItem = workItem
        stateQueue.asyncAfter(deadline: .now() + delay, execute: workItem)
    }

    private func cancelReconnectWork() {
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
    }

    private func bestEffortSendFinal(_ samples: [MotionSample], using task: URLSessionWebSocketTask?) {
        guard let task, !samples.isEmpty else { return }
        var offset = 0
        while offset < samples.count {
            let end = min(offset + maxSamplesPerBatch, samples.count)
            let batch = MotionBatch(sessionID: sessionID, samples: Array(samples[offset..<end]))
            guard let data = try? encoder.encode(batch) else { break }
            let semaphore = DispatchSemaphore(value: 0)
            var sendFailed = false
            task.send(.data(data)) { error in
                sendFailed = error != nil
                semaphore.signal()
            }
            if semaphore.wait(timeout: .now() + .seconds(1)) == .timedOut || sendFailed {
                print("Final server flush was incomplete; the complete data remains in the local backup.")
                return
            }
            offset = end
        }
    }

    private func printStatus() {
        let rateText: String
        if airPodsState == .connected, let lastSampleUptime, Self.uptime() - lastSampleUptime < 2.0 {
            rateText = String(format: "%.1f Hz", samplingRate)
        } else {
            rateText = "--"
        }
        let locationText = airPodsState == .connected ? activeSensorLocation.capitalized : "--"
        var line = "AirPod: \(locationText) | State: \(airPodsState.rawValue) | Rate: \(rateText) | Samples: \(sampleCount) | Server: \(serverState.rawValue)"
        if droppedNetworkSamples > 0 {
            line += " | Network-buffer drops: \(droppedNetworkSamples)"
        }
        print(line)
    }

    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        stateQueue.async { [weak self] in
            guard let self, !self.isShuttingDown else { return }
            self.airPodsState = .connected
            print("AirPod connected — starting motion stream...")
            self.motionRestartWorkItem?.cancel()
            let workItem = DispatchWorkItem { [weak self] in
                guard let self else { return }
                self.motionRestartWorkItem = nil
                if self.motionStreamActive || self.motionManager.isDeviceMotionActive {
                    self.motionManager.stopDeviceMotionUpdates()
                    self.motionStreamActive = false
                }
                self.startMotionStreamIfNeeded(reason: "AirPod reconnect")
            }
            self.motionRestartWorkItem = workItem
            self.stateQueue.asyncAfter(deadline: .now() + .milliseconds(750), execute: workItem)
        }
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        stateQueue.async { [weak self] in
            guard let self, !self.isShuttingDown else { return }
            print("AirPod removed/disconnected — collection paused until it returns.")
            self.airPodsState = .notConnected
            self.activeSensorLocation = "unknown"
            self.motionRestartWorkItem?.cancel()
            self.motionRestartWorkItem = nil
            if self.motionStreamActive || self.motionManager.isDeviceMotionActive {
                self.motionManager.stopDeviceMotionUpdates()
            }
            self.motionStreamActive = false
            self.streamStartedUptime = nil
            self.lastSampleUptime = nil
            self.samplingTimestamps.removeAll(keepingCapacity: true)
            self.samplingRate = 0
        }
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        stateQueue.async { [weak self] in
            guard let self,
                  self.socketTask === webSocketTask,
                  !self.isShuttingDown else { return }
            self.serverState = .connected
            self.serverReconnectDelay = 1.0
            self.cancelReconnectWork()
            print("Server connected.")
            self.flushBatch()
        }
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            self.handleSocketFailure(webSocketTask, message: "WebSocket closed (code \(closeCode.rawValue))")
        }
    }

    private static func sensorLocationText(_ location: CMDeviceMotion.SensorLocation) -> String {
        switch location {
        case .headphoneLeft:
            return "left"
        case .headphoneRight:
            return "right"
        case .default:
            return "unknown"
        @unknown default:
            return "unknown"
        }
    }

    private static func makeSessionID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = formatter.string(from: Date())
        let suffix = UUID().uuidString.prefix(8).lowercased()
        return "session_\(stamp)_\(suffix)"
    }

    private static func uptime() -> TimeInterval {
        ProcessInfo.processInfo.systemUptime
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
    print("Collection starts automatically. Commands: status, quit")
    exit(EXIT_SUCCESS)
}

guard let url = serverURL(from: arguments),
      url.scheme == "ws" || url.scheme == "wss" else {
    fputs("--server must be a valid ws:// or wss:// URL.\n", stderr)
    exit(EXIT_FAILURE)
}

let collector: MotionCollector
do {
    collector = try MotionCollector(serverURL: url)
} catch {
    fputs("Could not create local recording: \(error.localizedDescription)\n", stderr)
    exit(EXIT_FAILURE)
}

collector.run()
collector.printHelp()

signal(SIGINT, SIG_IGN)
let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
signalSource.setEventHandler {
    print("\nStopping collector...")
    collector.shutdown()
    exit(EXIT_SUCCESS)
}
signalSource.resume()

DispatchQueue.global(qos: .userInitiated).async {
    while let command = readLine(strippingNewline: true)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        switch command {
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
