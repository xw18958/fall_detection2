import Foundation
import Combine
import AppKit
import CoreBluetooth
import IOKit.pwr_mgt
import Darwin
import M5BLECollectorCore

final class CollectorViewModel: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published var participant: String {
        didSet { UserDefaults.standard.set(participant, forKey: "M5CollectorParticipant") }
    }
    @Published var placement: String {
        didSet { UserDefaults.standard.set(placement, forKey: "M5CollectorPlacement") }
    }
    @Published private(set) var connectionText = "Starting…"
    @Published private(set) var deviceName = "—"
    @Published private(set) var stateText = "DISCONNECTED"
    @Published private(set) var samplesSaved = 0
    @Published private(set) var trialsSaved = 0
    @Published private(set) var message = "Turn on the M5 and switch it to COLLECT mode."
    @Published private(set) var lastError: String?
    @Published private(set) var profileLocked = false

    let recordingsURL: URL
    var displayDataPath: String {
        recordingsURL.path.replacingOccurrences(of: FileManager.default.homeDirectoryForCurrentUser.path, with: "~")
    }

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var characteristics: [String: CBCharacteristic] = [:]
    private var info: DeviceInfo?
    private var status: DeviceStatus?
    private var recorder: Recorder?
    private var assembler = FrameAssembler()
    private var timer: DispatchSourceTimer?
    private var subscribed = false
    private var readySent = false
    private var statusReading = false
    private var infoReading = false
    private var subscriptionsRequested = false
    private var deviceReadyReported = false
    private var recordingLock: Int32 = -1
    private var activity: NSObjectProtocol?
    private var writes: [(Data, String)] = []
    private var inFlight: (Data, String)?
    private var writing = false
    private var fatalStorage = false
    private var lastExport = Date.distantPast
    private var finishedSession: UInt64 = 0
    private var assertion: IOPMAssertionID = 0
    private var selected: UUID?
    private var started = false

    override init() {
        participant = UserDefaults.standard.string(forKey: "M5CollectorParticipant") ?? ""
        placement = UserDefaults.standard.string(forKey: "M5CollectorPlacement") ?? ""
        recordingsURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("M5CollectorData", isDirectory: true)
        if let cached = UserDefaults.standard.string(forKey: "M5BLEPeripheral") {
            selected = UUID(uuidString: cached)
        }
        super.init()
    }

    func start() {
        guard !started else { return }
        started = true
        do {
            try FileManager.default.createDirectory(at: recordingsURL, withIntermediateDirectories: true)
            recordingLock = open(recordingsURL.appendingPathComponent(".collector.lock").path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
            guard recordingLock >= 0, flock(recordingLock, LOCK_EX | LOCK_NB) == 0 else {
                throw ProtocolError.invalid("Another M5 collector is already using the data folder.")
            }
            trialsSaved = countCompletedTrials()
            activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical], reason: "Receive and durably save M5 motion samples")
            let rc = IOPMAssertionCreateWithName(
                kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
                IOPMAssertionLevel(kIOPMAssertionLevelOn),
                "M5 BLE motion recording" as CFString,
                &assertion
            )
            guard rc == kIOReturnSuccess else {
                throw ProtocolError.invalid("Could not keep the Mac awake during collection.")
            }
            connectionText = "Searching…"
            central = CBCentralManager(delegate: self, queue: .main)
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 1, repeating: 1)
            timer.setEventHandler { [weak self] in self?.poll() }
            timer.resume()
            self.timer = timer
        } catch {
            fail("Collector could not start: \(error)")
        }
    }

    func shutdown() {
        timer?.cancel()
        timer = nil
        if let recorder, finishedSession != recorder.session { try? recorder.exportCSV() }
        if let peripheral, peripheral.state == .connected { central?.cancelPeripheralConnection(peripheral) }
        if assertion != 0 {
            IOPMAssertionRelease(assertion)
            assertion = 0
        }
        if let activity {
            ProcessInfo.processInfo.endActivity(activity)
            self.activity = nil
        }
        if recordingLock >= 0 {
            flock(recordingLock, LOCK_UN)
            Darwin.close(recordingLock)
            recordingLock = -1
        }
    }

    func openDataFolder() {
        do {
            try FileManager.default.createDirectory(at: recordingsURL, withIntermediateDirectories: true)
            NSWorkspace.shared.open(recordingsURL)
        } catch {
            lastError = "Could not open data folder: \(error.localizedDescription)"
        }
    }

    deinit { shutdown() }

    private var currentProfile: [String: String] {
        let p = participant.trimmingCharacters(in: .whitespacesAndNewlines)
        let place = placement.trimmingCharacters(in: .whitespacesAndNewlines)
        return [
            "participant": p.isEmpty ? "unspecified" : p,
            "placement": place.isEmpty ? "unspecified" : place
        ]
    }

    private func scan() {
        guard let central, central.state == .poweredOn, !fatalStorage else { return }
        connectionText = "Searching…"
        if let selected,
           let known = central.retrievePeripherals(withIdentifiers: [selected]).first {
            peripheral = known
            known.delegate = self
            connectionText = "Connecting…"
            central.connect(known)
        } else {
            central.scanForPeripherals(withServices: [CBUUID(string: Wire.service)])
        }
    }

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            lastError = nil
            scan()
        case .unauthorized:
            connectionText = "Bluetooth permission needed"
            stateText = "DISCONNECTED"
            message = "Allow Bluetooth for M5 Data Collector in System Settings > Privacy & Security > Bluetooth."
        case .poweredOff:
            connectionText = "Bluetooth off"
            stateText = "DISCONNECTED"
            message = "Turn on Bluetooth on this Mac."
        default:
            connectionText = "Bluetooth unavailable"
            stateText = "DISCONNECTED"
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover candidate: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
        guard peripheral == nil else { return }
        if let selected, candidate.identifier != selected { return }
        peripheral = candidate
        candidate.delegate = self
        deviceName = candidate.name ?? "M5"
        connectionText = "Connecting…"
        central.stopScan()
        central.connect(candidate)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectionText = "Connected"
        deviceName = peripheral.name ?? "M5"
        characteristics.removeAll()
        info = nil
        status = nil
        subscribed = false
        readySent = false
        statusReading = false
        writing = false
        inFlight = nil
        writes.removeAll()
        assembler.reset()
        infoReading = false
        subscriptionsRequested = false
        deviceReadyReported = false
        peripheral.discoverServices([CBUUID(string: Wire.service)])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        reconnect(error)
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        reconnect(error)
    }

    private func reconnect(_ error: Error?) {
        self.peripheral = nil
        info = nil
        status = nil
        characteristics.removeAll()
        subscribed = false
        readySent = false
        assembler.reset()
        writes.removeAll()
        writing = false
        inFlight = nil
        statusReading = false
        infoReading = false
        subscriptionsRequested = false
        deviceReadyReported = false
        connectionText = fatalStorage ? "Stopped" : "Reconnecting…"
        stateText = "DISCONNECTED"
        profileLocked = false
        if let error, !fatalStorage { lastError = "Bluetooth disconnected: \(error.localizedDescription)" }
        if !fatalStorage {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.scan() }
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil,
              let service = peripheral.services?.first(where: { $0.uuid == CBUUID(string: Wire.service) }) else {
            fail("M5 collection service could not be discovered.")
            return
        }
        peripheral.discoverCharacteristics(
            [Wire.info, Wire.control, Wire.samples, Wire.status].map(CBUUID.init(string:)),
            for: service
        )
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard error == nil else {
            fail("M5 BLE characteristics could not be discovered.")
            return
        }
        for characteristic in service.characteristics ?? [] {
            characteristics[characteristic.uuid.uuidString.uppercased()] = characteristic
        }
        guard let information = characteristic(Wire.info),
              characteristic(Wire.samples) != nil,
              characteristic(Wire.status) != nil,
              characteristic(Wire.control) != nil else {
            fail("The connected M5 is missing a required collection characteristic.")
            return
        }
        message = "Connected. Approve Bluetooth pairing if macOS asks."
        infoReading = true
        peripheral.readValue(for: information)
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil else {
            fail("Could not subscribe to M5 data notifications: \(error!.localizedDescription)")
            return
        }
        if characteristic.uuid == CBUUID(string: Wire.samples) {
            subscribed = characteristic.isNotifying
            handshake()
        }
    }

    private func characteristic(_ uuid: String) -> CBCharacteristic? {
        characteristics[uuid.uppercased()]
    }

    private func handshake() {
        guard info != nil, status != nil, subscribed, !readySent else { return }
        readySent = true
        send(Wire.command(5), label: "ready")
    }

    private func attach(_ session: UInt64) throws {
        guard session != 0, let info else {
            throw ProtocolError.invalid("Session arrived before device information")
        }
        if let current = recorder, current.session == session { return }
        if let old = recorder, finishedSession != old.session { try old.exportCSV() }
        recorder = try Recorder(root: recordingsURL, session: session, info: info, profile: currentProfile)
        finishedSession = 0
        samplesSaved = Int(recorder?.exclusive ?? 0)
    }

    private func hasJournal(_ session: UInt64) -> Bool {
        let hex = String(format: "%016llx", session)
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: recordingsURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries.contains { entry in
            guard let data = try? Data(contentsOf: entry.appendingPathComponent("metadata.json")),
                  let metadata = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  metadata["session_id"] as? String == hex else { return false }
            return Recorder.hasJournal(in: entry)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if characteristic.uuid == CBUUID(string: Wire.info) { infoReading = false }
        if characteristic.uuid == CBUUID(string: Wire.status) { statusReading = false }
        guard error == nil, let data = characteristic.value else {
            lastError = "Bluetooth read failed: \(error?.localizedDescription ?? "empty value")"
            return
        }
        do {
            switch characteristic.uuid {
            case CBUUID(string: Wire.info):
                let incoming = try JSONDecoder().decode(DeviceInfo.self, from: data)
                try incoming.validate()
                try TrustedModels.validate(incoming.model_sha256)
                info = incoming
                selected = peripheral.identifier
                UserDefaults.standard.set(peripheral.identifier.uuidString, forKey: "M5BLEPeripheral")
                deviceName = "M5MOTION-" + incoming.device_id.suffix(4).uppercased()
                if let state = self.characteristic(Wire.status) {
                    statusReading = true
                    peripheral.readValue(for: state)
                }
                handshake()

            case CBUUID(string: Wire.status):
                status = try DeviceStatus(data)
                if let old = recorder, finishedSession == old.session, status!.session != old.session {
                    recorder = nil
                    finishedSession = 0
                    samplesSaved = 0
                }
                if !subscriptionsRequested, info != nil,
                   let stream = self.characteristic(Wire.samples),
                   let state = self.characteristic(Wire.status) {
                    subscriptionsRequested = true
                    peripheral.setNotifyValue(true, for: stream)
                    peripheral.setNotifyValue(true, for: state)
                }
                if status!.flags & 2 != 0, !deviceReadyReported {
                    deviceReadyReported = true
                    connectionText = "Connected & ready"
                    lastError = nil
                }
                if let state = status, state.session != 0, info != nil, state.saving || state.complete {
                    if state.complete, recorder?.session != state.session, !hasJournal(state.session) {
                        finishedSession = state.session
                    } else {
                        try attach(state.session)
                        if state.complete, finishedSession != state.session, let recorder {
                            try recorder.finish(produced: state.produced, overflowed: state.flags & 4 != 0)
                            finishedSession = state.session
                            samplesSaved = Int(recorder.exclusive)
                            trialsSaved = countCompletedTrials()
                        }
                    }
                }
                publishStatus()
                handshake()

            case CBUUID(string: Wire.samples):
                if let batch = try assembler.accept(data) {
                    try attach(batch.session)
                    guard let recorder else { return }
                    let exclusive = try recorder.append(batch.samples)
                    samplesSaved = Int(exclusive)
                    send(Wire.command(3, session: batch.session, exclusive: exclusive), label: "ack")
                }

            default:
                break
            }
        } catch {
            fail("Recording stopped on this Mac to preserve data integrity: \(error)")
        }
    }

    private func send(_ data: Data, label: String) {
        guard peripheral?.state == .connected, characteristic(Wire.control) != nil else { return }
        if label == "ack" { writes.removeAll { $0.1 == "ack" } }
        writes.append((data, label))
        writeNext()
    }

    private func writeNext() {
        guard !writing,
              let first = writes.first,
              let peripheral,
              let control = characteristic(Wire.control) else { return }
        writes.removeFirst()
        inFlight = first
        writing = true
        peripheral.writeValue(first.0, for: control, type: .withResponse)
    }

    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == CBUUID(string: Wire.control) else { return }
        writing = false
        let sent = inFlight
        inFlight = nil
        if let error {
            lastError = "M5 control write failed: \(error.localizedDescription)"
            if sent?.1 == "ready" { readySent = false }
        }
        writeNext()
    }

    private func poll() {
        guard !fatalStorage else { return }
        if let peripheral, peripheral.state == .connected {
            if info == nil, !infoReading, let information = characteristic(Wire.info) {
                infoReading = true
                peripheral.readValue(for: information)
            }
            if info != nil, !statusReading, let state = characteristic(Wire.status) {
                statusReading = true
                peripheral.readValue(for: state)
            }
            handshake()
        }
        if Date().timeIntervalSince(lastExport) > 60,
           let recorder,
           finishedSession != recorder.session {
            do {
                try recorder.exportCSV()
                lastExport = Date()
            } catch {
                fail("CSV export failed: \(error)")
            }
        }
    }

    private func publishStatus() {
        guard let s = status else {
            stateText = "DISCONNECTED"
            profileLocked = false
            return
        }
        let states = ["READY", "RECORDING", "REVIEW", "SAVING", "COMPLETE", "BUFFER FULL"]
        stateText = states[Int(s.state)]
        profileLocked = !(s.state == 0 || s.state == 4)
        if let recorder, recorder.session == s.session { samplesSaved = Int(recorder.exclusive) }

        switch s.state {
        case 0:
            message = "Ready. Use A on the M5 to start a trial."
        case 1:
            message = "Recording on the M5. Press A on the M5 when the movement is finished."
        case 2:
            message = "Review on the M5: A = KEEP, B = DISCARD."
        case 3:
            message = "Saving the kept trial to this Mac. Keep the M5 nearby."
        case 4:
            message = "Trial saved. The M5 is ready for another trial."
        case 5:
            message = "M5 buffer is full. Choose KEEP or DISCARD on the M5."
        default:
            message = "Connected to the M5."
        }
    }

    private func countCompletedTrials() -> Int {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: recordingsURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries.reduce(0) { count, entry in
            guard let data = try? Data(contentsOf: entry.appendingPathComponent("metadata.json")),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let completion = json["completion"] as? [String: Any],
                  completion["complete"] as? Bool == true else { return count }
            return count + 1
        }
    }

    private func fail(_ message: String) {
        lastError = message
        fatalStorage = true
        connectionText = "Stopped"
        stateText = "ERROR"
        self.message = "Collection stopped to protect the recording. Close and reopen the app after resolving the error."
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
    }
}
