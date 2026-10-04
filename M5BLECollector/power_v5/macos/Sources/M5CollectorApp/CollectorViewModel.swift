import Foundation
import Combine
import SwiftUI
import AppKit
import CoreBluetooth
import IOKit.pwr_mgt
import Darwin
import M5BLECollectorCore

final class CollectorViewModel: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published private(set) var connectionText = "Waiting for save folder"
    @Published private(set) var stateText = "WAITING"
    @Published private(set) var message = "Choose a save folder to begin."
    @Published private(set) var lastError: String?
    @Published private(set) var recordingsURL: URL?
    @Published private(set) var canChangeSaveFolder = true
    @Published private(set) var recordingSeconds: Double = 0
    @Published private(set) var saveProgress: Double?
    @Published private(set) var lastSaveText = "No recording saved yet."
    @Published private(set) var dataQualityText = "—"
    @Published private(set) var dataQualityWarning = false
    @Published private(set) var hasSavedRecording = false

    var hasSaveFolder: Bool { recordingsURL != nil }
    var displayDataPath: String {
        guard let recordingsURL else { return "No folder selected" }
        return recordingsURL.path.replacingOccurrences(
            of: FileManager.default.homeDirectoryForCurrentUser.path,
            with: "~"
        )
    }
    var showRecordingTimer: Bool {
        ["RECORDING", "REVIEW", "SAVING", "BUFFER FULL"].contains(stateText)
    }
    var recordingTimeText: String {
        let totalTenths = max(0, Int((recordingSeconds * 10).rounded()))
        let minutes = totalTenths / 600
        let seconds = (totalTenths % 600) / 10
        let tenths = totalTenths % 10
        return String(format: "%02d:%02d.%d", minutes, seconds, tenths)
    }
    var connectionIndicatorColor: Color {
        switch connectionText {
        case "Connected": return .green
        case "Searching…", "Connecting…", "Reconnecting…": return .orange
        case "Waiting for save folder": return .gray
        default: return .red
        }
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
    private var transportStarted = false

    override init() {
        if let savedPath = UserDefaults.standard.string(forKey: "M5CollectorSaveFolder"), !savedPath.isEmpty {
            recordingsURL = URL(fileURLWithPath: savedPath, isDirectory: true).standardizedFileURL
        }
        if let cached = UserDefaults.standard.string(forKey: "M5BLEPeripheral") {
            selected = UUID(uuidString: cached)
        }
        super.init()
    }

    func start() {
        guard !started else { return }
        started = true
        guard let saved = recordingsURL else {
            connectionText = "Waiting for save folder"
            stateText = "WAITING"
            message = "Choose where recordings should be saved."
            return
        }
        do {
            try switchStorage(to: saved)
            try beginTransport()
        } catch {
            releaseStorageLock()
            recordingsURL = nil
            UserDefaults.standard.removeObject(forKey: "M5CollectorSaveFolder")
            connectionText = "Waiting for save folder"
            stateText = "WAITING"
            message = "The previous save folder is unavailable. Choose another folder."
            lastError = "Could not use the saved folder: \(error.localizedDescription)"
        }
    }

    func chooseDataFolder() {
        guard canChangeSaveFolder else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose where M5 recordings will be saved"
        panel.prompt = "Choose Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if let recordingsURL { panel.directoryURL = recordingsURL }
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            do {
                try self.switchStorage(to: url)
                UserDefaults.standard.set(self.recordingsURL?.path, forKey: "M5CollectorSaveFolder")
                self.lastError = nil
                if !self.transportStarted {
                    try self.beginTransport()
                } else if self.status?.idle == true {
                    self.message = "Save folder changed. Ready for the next recording."
                }
            } catch {
                self.lastError = "Could not use that folder: \(error.localizedDescription)"
            }
        }
    }

    func openDataFolder() {
        guard let recordingsURL else { return }
        NSWorkspace.shared.open(recordingsURL)
    }

    private func switchStorage(to url: URL) throws {
        if let recorder, finishedSession != recorder.session {
            throw ProtocolError.invalid("Finish the current recording before changing the save folder")
        }
        let newURL = url.standardizedFileURL
        if recordingsURL == newURL, recordingLock >= 0 { return }

        try FileManager.default.createDirectory(at: newURL, withIntermediateDirectories: true)
        let newLock = open(
            newURL.appendingPathComponent(".collector.lock").path,
            O_CREAT | O_RDWR,
            S_IRUSR | S_IWUSR
        )
        guard newLock >= 0 else {
            throw ProtocolError.invalid("The selected folder cannot be opened for writing")
        }
        guard flock(newLock, LOCK_EX | LOCK_NB) == 0 else {
            Darwin.close(newLock)
            throw ProtocolError.invalid("Another M5 collector is already using that folder")
        }

        releaseStorageLock()
        recordingLock = newLock
        recordingsURL = newURL
        recorder = nil
        finishedSession = status?.complete == true ? status!.session : 0
        lastExport = .distantPast
        UserDefaults.standard.set(newURL.path, forKey: "M5CollectorSaveFolder")
    }

    private func beginTransport() throws {
        guard !transportStarted else { return }
        guard recordingLock >= 0 else {
            throw ProtocolError.invalid("Choose a writable save folder first")
        }
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .latencyCritical],
            reason: "Receive and durably save M5 motion samples"
        )
        let rc = IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "M5 BLE motion recording" as CFString,
            &assertion
        )
        guard rc == kIOReturnSuccess else {
            if let activity {
                ProcessInfo.processInfo.endActivity(activity)
                self.activity = nil
            }
            throw ProtocolError.invalid("Could not keep the Mac awake during collection")
        }
        transportStarted = true
        connectionText = "Searching…"
        stateText = "DISCONNECTED"
        message = "Turn on the M5 and switch it to COLLECT mode."
        central = CBCentralManager(delegate: self, queue: .main)
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in self?.poll() }
        timer.resume()
        self.timer = timer
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
        releaseStorageLock()
    }

    private func releaseStorageLock() {
        if recordingLock >= 0 {
            flock(recordingLock, LOCK_UN)
            Darwin.close(recordingLock)
            recordingLock = -1
        }
    }

    deinit { shutdown() }

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
        connectionText = "Connecting…"
        central.stopScan()
        central.connect(candidate)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectionText = "Connected"
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
        let wasBusy = status.map { !$0.idle } ?? false
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
        saveProgress = nil
        canChangeSaveFolder = !wasBusy && (recorder == nil || finishedSession == recorder?.session)
        if !fatalStorage {
            message = "M5 disconnected. Keep it on and nearby; reconnecting automatically."
            if let error { lastError = "Bluetooth disconnected: \(error.localizedDescription)" }
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
        guard session != 0, let info, let recordingsURL else {
            throw ProtocolError.invalid("Recording arrived before storage or device information was ready")
        }
        if let current = recorder, current.session == session { return }
        if let old = recorder, finishedSession != old.session { try old.exportCSV() }
        recorder = try Recorder(root: recordingsURL, session: session, info: info, profile: [:])
        finishedSession = 0
    }

    private func hasJournal(_ session: UInt64) -> Bool {
        guard let recordingsURL else { return false }
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
                    connectionText = "Connected"
                    lastError = nil
                }
                if let state = status, state.session != 0, info != nil, state.saving || state.complete {
                    if state.complete, recorder?.session != state.session, !hasJournal(state.session) {
                        finishedSession = state.session
                    } else {
                        try attach(state.session)
                        if state.complete, finishedSession != state.session, let recorder {
                            let overflowed = state.flags & 4 != 0
                            try recorder.finish(produced: state.produced, overflowed: overflowed)
                            finishedSession = state.session
                            updateLastSave(recorder: recorder, overflowed: overflowed)
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
                    send(Wire.command(3, session: batch.session, exclusive: exclusive), label: "ack")
                }

            default:
                break
            }
        } catch {
            fail("Recording stopped on this Mac to preserve data integrity: \(error)")
        }
    }

    private func updateLastSave(recorder: Recorder, overflowed: Bool) {
        hasSavedRecording = true
        lastSaveText = "Saved successfully • \(formatDuration(recorder.elapsedSeconds))"
        do {
            let url = recorder.directory.appendingPathComponent("metadata.json")
            let data = try Data(contentsOf: url)
            guard let metadata = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let quality = metadata["quality"] as? [String: Any] else {
                throw ProtocolError.invalid("quality report missing")
            }
            func intValue(_ key: String) -> Int {
                (quality[key] as? NSNumber)?.intValue ?? 0
            }
            let readErrors = intValue("read_errors")
            let timingGaps = max(intValue("timing_gap_flags"), intValue("observed_timestamp_gaps"))
            let saturated = intValue("saturated_samples")
            let sequenceGaps = intValue("sequence_gaps")
            let hz = (quality["measured_hz"] as? NSNumber)?.doubleValue ?? 0
            var warnings: [String] = []
            if readErrors > 0 { warnings.append("\(readErrors) sensor read error\(readErrors == 1 ? "" : "s")") }
            if timingGaps > 0 { warnings.append("\(timingGaps) timing gap\(timingGaps == 1 ? "" : "s")") }
            if saturated > 0 { warnings.append("\(saturated) saturated sample\(saturated == 1 ? "" : "s")") }
            if sequenceGaps > 0 { warnings.append("\(sequenceGaps) sequence gap\(sequenceGaps == 1 ? "" : "s")") }
            if overflowed { warnings.append("device buffer filled") }
            dataQualityWarning = !warnings.isEmpty
            if warnings.isEmpty {
                dataQualityText = hz > 0 ? String(format: "OK • %.1f Hz", hz) : "OK"
            } else {
                let rate = hz > 0 ? String(format: " • %.1f Hz", hz) : ""
                dataQualityText = "Warning: " + warnings.joined(separator: ", ") + rate
            }
        } catch {
            dataQualityWarning = true
            dataQualityText = "Warning: quality report unavailable"
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        let minutes = Int(seconds) / 60
        let remaining = seconds - Double(minutes * 60)
        return String(format: "%d:%04.1f", minutes, remaining)
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
            saveProgress = nil
            return
        }
        let states = ["READY", "RECORDING", "REVIEW", "SAVING", "COMPLETE", "BUFFER FULL"]
        stateText = states[Int(s.state)]
        recordingSeconds = s.state == 0 ? 0 : Double(s.produced) / 30.0
        canChangeSaveFolder = s.idle && (recorder == nil || finishedSession == recorder?.session)

        if s.state == 3, s.produced > 0 {
            let saved = max(0, Int(s.produced) - Int(s.pending))
            saveProgress = min(1, max(0, Double(saved) / Double(s.produced)))
        } else {
            saveProgress = nil
        }

        switch s.state {
        case 0:
            message = "Ready for the next recording. Press A on the M5 to start."
        case 1:
            message = "Recording. Press A on the M5 when the movement is finished."
        case 2:
            message = "Review on the M5: A = KEEP, B = DISCARD."
        case 3:
            message = "Saving to this Mac. Keep the M5 on and nearby until saving finishes."
        case 4:
            message = hasSavedRecording ? "✓ SAVED — READY FOR NEXT RECORDING" : "M5 is ready for the next recording."
        case 5:
            message = "M5 buffer is full. Choose KEEP or DISCARD on the M5."
        default:
            message = "Connected to the M5."
        }
    }

    private func fail(_ message: String) {
        lastError = message
        fatalStorage = true
        connectionText = "Stopped"
        stateText = "ERROR"
        saveProgress = nil
        canChangeSaveFolder = false
        self.message = "Collection stopped to protect the recording. Close and reopen the app after resolving the error."
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
    }
}
