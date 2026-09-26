import CoreMotion
import Foundation

final class AirPodsMotionManager: NSObject, CMHeadphoneMotionManagerDelegate {
    var onSample: ((MotionSample) -> Void)?
    var onConnectionChanged: ((Bool, String) -> Void)?
    var onError: ((String) -> Void)?

    private let manager = CMHeadphoneMotionManager()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "iPhoneFallDetection.airpods-motion"
        queue.maxConcurrentOperationCount = 1
        return queue
    }()

    private(set) var isRunning = false

    override init() {
        super.init()
        manager.delegate = self
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        manager.startConnectionStatusUpdates()
        startMotionIfAvailable()
    }

    func stop() {
        isRunning = false
        if manager.isDeviceMotionActive {
            manager.stopDeviceMotionUpdates()
        }
        manager.stopConnectionStatusUpdates()
        DispatchQueue.main.async { [weak self] in
            self?.onConnectionChanged?(false, "unknown")
        }
    }

    private func startMotionIfAvailable() {
        guard isRunning else { return }
        guard manager.isDeviceMotionAvailable else {
            DispatchQueue.main.async { [weak self] in
                self?.onConnectionChanged?(false, "unknown")
            }
            return
        }

        if manager.isDeviceMotionActive {
            manager.stopDeviceMotionUpdates()
        }

        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, error in
            guard let self, self.isRunning else { return }
            if let error {
                DispatchQueue.main.async {
                    self.onError?(error.localizedDescription)
                }
                return
            }
            guard let motion else { return }

            let location = Self.locationText(motion.sensorLocation)
            let sample = MotionSample(
                timestamp: motion.timestamp,
                sensorLocation: location,
                ax: Float(motion.userAcceleration.x + motion.gravity.x),
                ay: Float(motion.userAcceleration.y + motion.gravity.y),
                az: Float(motion.userAcceleration.z + motion.gravity.z),
                gx: Float(motion.rotationRate.x),
                gy: Float(motion.rotationRate.y),
                gz: Float(motion.rotationRate.z)
            )
            DispatchQueue.main.async {
                self.onConnectionChanged?(true, location)
                self.onSample?(sample)
            }
        }
    }

    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isRunning else { return }
            self.onConnectionChanged?(true, "unknown")
            self.startMotionIfAvailable()
        }
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onConnectionChanged?(false, "unknown")
        }
    }

    private static func locationText(_ location: CMDeviceMotion.SensorLocation) -> String {
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
}
