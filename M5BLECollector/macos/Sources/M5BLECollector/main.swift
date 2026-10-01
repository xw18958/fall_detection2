import Foundation
import CoreBluetooth
import IOKit.pwr_mgt
import Darwin
import M5BLECollectorCore

struct Options {
    var recordings = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("recordings")
    var profile = ["participant": "unspecified", "activity": "unspecified", "placement": "unspecified"]
    var device: String?
    init() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        var index = 0
        while index < args.count {
            let key = args[index]
            if key == "--help" {
                print("M5BLECollector [--recordings PATH] [--participant ID] [--activity NAME] [--placement NAME] [--device DEVICE_ID_OR_UUID]")
                print("Commands: status, start, stop, mark LABEL, activity LABEL, detect, quit, quit force")
                exit(0)
            }
            guard index+1 < args.count else { throw ProtocolError.invalid("Missing value for \(key)") }
            let value = args[index+1]
            switch key {
            case "--recordings": recordings = URL(fileURLWithPath: value).standardizedFileURL
            case "--participant", "--activity", "--placement": profile[String(key.dropFirst(2))] = value
            case "--device":
                guard UUID(uuidString: value) != nil || (value.count == 12 && value.allSatisfy({ $0.isHexDigit })) else {
                    throw ProtocolError.invalid("--device must be a 12-digit device ID or peripheral UUID")
                }
                device = value.lowercased()
            default: throw ProtocolError.invalid("Unknown argument \(key)")
            }
            index += 2
        }
    }
}

final class Collector: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    private let options: Options
    private var profile: [String: String]
    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var characteristics: [String: CBCharacteristic] = [:]
    private var info: DeviceInfo?, status: DeviceStatus?
    private var recorder: Recorder?
    private var assembler = FrameAssembler()
    private var timer: DispatchSourceTimer?
    private var signals: [DispatchSourceSignal] = []
    private var subscribed = false, readySent = false, statusReading = false
    private var infoReading = false, subscriptionsRequested = false, deviceReadyReported = false
    private var writes: [(Data, String)] = []
    private var inFlight: (Data, String)?
    private var writing = false, quitting = false, detecting = false, fatalStorage = false
    private var lastExport = Date.distantPast, lastStatusPrint = Date.distantPast
    private var finishedSession: UInt64 = 0
    private var assertion: IOPMAssertionID = 0
    private var activity: NSObjectProtocol?
    private var selected: UUID?
    private let lockedHash = "1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530"

    init(options: Options) {
        self.options = options; profile = options.profile
        if let cached = UserDefaults.standard.string(forKey: "M5BLEPeripheral") { selected = UUID(uuidString: cached) }
        super.init()
    }
    func run() throws {
        try FileManager.default.createDirectory(at: options.recordings, withIntermediateDirectories: true)
        try saveProfile()
        let rc = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn), "M5 BLE motion recording" as CFString, &assertion)
        guard rc == kIOReturnSuccess else { throw ProtocolError.invalid("Cannot create idle-sleep prevention assertion") }
        activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical], reason: "Receive and durably save M5 motion samples")
        print("Bluetooth collector; Wi-Fi and Internet are unnecessary. Recordings: \(options.recordings.path)")
        print("Idle sleep prevented. Verify your separate closed-lid awake configuration before walking outside.")
        print("Switch M5 into COLLECT by holding B for 2 seconds. Press A to start/stop; B adds a marker.")
        central = CBCentralManager(delegate: self, queue: .main)
        timer = DispatchSource.makeTimerSource(queue: .main)
        timer?.schedule(deadline: .now()+1, repeating: 1)
        timer?.setEventHandler { [weak self] in self?.poll() }; timer?.resume()
        for number in [SIGINT, SIGTERM] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in
                guard let self = self else { return }
                if self.quitting { self.close() } else { self.command("quit") }
            }
            source.resume(); signals.append(source)
        }
        DispatchQueue.global().async { [weak self] in
            while let line = readLine() { DispatchQueue.main.async { self?.command(line) } }
        }
    }
    private func saveProfile() throws {
        let data = try JSONSerialization.data(withJSONObject: profile, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: options.recordings.appendingPathComponent("collector_profile.json"), options: .atomic)
    }
    private func scan() {
        guard central.state == .poweredOn, !detecting, !fatalStorage else { return }
        if options.device == nil, let id = selected, let known = central.retrievePeripherals(withIdentifiers: [id]).first {
            peripheral = known; known.delegate = self; central.connect(known)
        } else {
            // Recovery scans also accept a name-only advertisement. The known
            // identity filter below and subsequent service/model verification
            // still select the collector before any recording commands are sent.
            central.scanForPeripherals(withServices: nil)
        }
    }
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn: scan()
        case .unauthorized: print("Bluetooth permission denied. Enable M5 BLE Collector in System Settings > Privacy & Security > Bluetooth.")
        case .poweredOff: print("Bluetooth is off; enable it on the Mac.")
        default: print("Bluetooth state: \(central.state.rawValue)")
        }
    }
    func centralManager(_ central: CBCentralManager, didDiscover candidate: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
        guard peripheral == nil else { return }
        // A --device UUID can be filtered before connecting; device MAC identity is checked after discovery.
        if let wanted = options.device, let uuid = UUID(uuidString: wanted), candidate.identifier != uuid { return }
        if let wanted = options.device, UUID(uuidString: wanted) == nil {
            let advertised = (advertisementData[CBAdvertisementDataLocalNameKey] as? String) ?? candidate.name ?? ""
            guard advertised.uppercased() == "M5MOTION-" + wanted.suffix(4).uppercased() else { return }
        }
        if options.device == nil, let selected = selected, candidate.identifier != selected { return }
        print("Found \(candidate.name ?? "M5") RSSI=\(RSSI)")
        peripheral = candidate; candidate.delegate = self; central.stopScan(); central.connect(candidate)
    }
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("BLE link connected; discovering collector service.")
        characteristics.removeAll(); info = nil; status = nil; subscribed = false; readySent = false
        infoReading = false; subscriptionsRequested = false; deviceReadyReported = false
        statusReading = false; writing = false; inFlight = nil; writes.removeAll(); assembler.reset()
        peripheral.discoverServices([CBUUID(string: Wire.service)])
    }
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        reconnect(error)
    }
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        reconnect(error)
    }
    private func reconnect(_ error: Error?) {
        print("Disconnected\(error.map { ": \($0.localizedDescription)" } ?? ""); device retains unacknowledged data in RAM.")
        peripheral = nil; info = nil; status = nil; characteristics.removeAll(); subscribed = false; readySent = false
        infoReading = false; subscriptionsRequested = false; deviceReadyReported = false
        assembler.reset(); writes.removeAll(); writing = false; inFlight = nil; statusReading = false
        if detecting { close(); return }
        if !fatalStorage { DispatchQueue.main.asyncAfter(deadline: .now()+2) { [weak self] in self?.scan() } }
    }
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil, let service = peripheral.services?.first(where: { $0.uuid == CBUUID(string: Wire.service) }) else { fail("Service discovery failed: \(String(describing: error))"); return }
        peripheral.discoverCharacteristics([Wire.info, Wire.control, Wire.samples, Wire.status].map(CBUUID.init(string:)), for: service)
    }
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard error == nil else { fail("Characteristic discovery failed"); return }
        for characteristic in service.characteristics ?? [] { characteristics[characteristic.uuid.uuidString.uppercased()] = characteristic }
        guard let information = characteristic(Wire.info), characteristic(Wire.samples) != nil, characteristic(Wire.status) != nil, characteristic(Wire.control) != nil else { fail("Missing BLE characteristic"); return }
        print("Collector service found; reading device information.")
        infoReading = true; peripheral.readValue(for: information)
    }
    private func characteristic(_ uuid: String) -> CBCharacteristic? { characteristics[uuid.uppercased()] }
    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil else { fail("Notification subscription failed: \(error!.localizedDescription)"); return }
        if characteristic.uuid == CBUUID(string: Wire.samples) { subscribed = characteristic.isNotifying; handshake() }
    }
    private func handshake() {
        guard info != nil, status != nil, subscribed, !readySent else { return }
        readySent = true; send(Wire.command(5), label: "ready")
        print("Collector handshake sent; waiting for device readiness.")
    }
    private func attach(_ session: UInt64) throws {
        guard session != 0, let info = info else { throw ProtocolError.invalid("Session arrived before device information") }
        if let current = recorder, current.session == session { return }
        if let old = recorder { try old.exportCSV() }
        recorder = try Recorder(root: options.recordings, session: session, info: info, profile: profile)
        finishedSession = 0
        print("Session: \(recorder!.directory.path)")
    }
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if characteristic.uuid == CBUUID(string: Wire.status) { statusReading = false }
        if characteristic.uuid == CBUUID(string: Wire.info) { infoReading = false }
        guard error == nil, let data = characteristic.value else {
            print("BLE read failed: \(error?.localizedDescription ?? "empty value"). Approve pairing if requested.")
            return
        }
        do {
            switch characteristic.uuid {
            case CBUUID(string: Wire.info):
                let incoming = try JSONDecoder().decode(DeviceInfo.self, from: data); try incoming.validate()
                guard incoming.model_sha256 == lockedHash else { throw ProtocolError.invalid("Device model differs from the locked detector baseline") }
                if let wanted = options.device, UUID(uuidString: wanted) == nil, wanted != incoming.device_id.lowercased() {
                    throw ProtocolError.invalid("Connected device does not match --device")
                }
                info = incoming; selected = peripheral.identifier
                UserDefaults.standard.set(peripheral.identifier.uuidString, forKey: "M5BLEPeripheral")
                print("Verified detector model \(incoming.model_sha256.prefix(12))…; firmware \(incoming.firmware)")
                if incoming.transport == "ble_unpaired" { print("Transport: unpaired BLE; motion data is not encrypted.") }
                if let state = self.characteristic(Wire.status) { statusReading = true; peripheral.readValue(for: state) }
                handshake()
            case CBUUID(string: Wire.status):
                status = try DeviceStatus(data)
                if !subscriptionsRequested, info != nil,
                   let stream = self.characteristic(Wire.samples), let state = self.characteristic(Wire.status) {
                    subscriptionsRequested = true
                    peripheral.setNotifyValue(true, for: stream); peripheral.setNotifyValue(true, for: state)
                }
                if status!.flags & 2 != 0, !deviceReadyReported {
                    deviceReadyReported = true
                    print("Connected and ready. Recording starts with A or the start command.")
                }
                if let state = status, state.session != 0, info != nil {
                    try attach(state.session)
                    if state.complete, finishedSession != state.session, let recorder = recorder {
                        try recorder.finish(produced: state.produced, overflowed: state.flags & 4 != 0)
                        finishedSession = state.session; print("Saved \(state.produced) samples: \(recorder.directory.path)")
                        if quitting { close(); return }
                    }
                }
                handshake()
            case CBUUID(string: Wire.samples):
                if let batch = try assembler.accept(data) {
                    try attach(batch.session)
                    guard let recorder = recorder else { return }
                    let exclusive = try recorder.append(batch.samples)
                    send(Wire.command(3, session: batch.session, exclusive: exclusive), label: "ack")
                }
            default: break
            }
        } catch {
            fail("Recording stopped on Mac to preserve integrity: \(error)")
        }
    }
    private func send(_ data: Data, label: String) {
        guard peripheral?.state == .connected, characteristic(Wire.control) != nil else { print("Cannot send \(label): disconnected"); return }
        if label == "ack" { writes.removeAll { $0.1 == "ack" } }
        writes.append((data, label)); writeNext()
    }
    private func writeNext() {
        guard !writing, let first = writes.first, let peripheral = peripheral, let control = characteristic(Wire.control) else { return }
        writes.removeFirst(); inFlight = first; writing = true
        peripheral.writeValue(first.0, for: control, type: .withResponse)
    }
    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == CBUUID(string: Wire.control) else { return }
        writing = false
        let sent = inFlight; inFlight = nil
        if let error = error { print("Control write failed: \(error.localizedDescription)"); if sent?.1 == "ready" { readySent = false } }
        if sent?.1 == "detect" {
            if error == nil { detecting = true } else { detecting = false }
        }
        writeNext()
    }
    private func poll() {
        guard !fatalStorage else { return }
        if let peripheral = peripheral, peripheral.state == .connected {
            if info == nil, !infoReading, let information = characteristic(Wire.info) {
                infoReading = true; peripheral.readValue(for: information)
            }
            if info != nil, !statusReading, let state = characteristic(Wire.status) { statusReading = true; peripheral.readValue(for: state) }
            handshake()
        }
        if Date().timeIntervalSince(lastExport) > 60, let recorder = recorder {
            do { try recorder.exportCSV(); lastExport = Date() } catch { fail("CSV export failed: \(error)") }
        }
        if Date().timeIntervalSince(lastStatusPrint) >= 10 { showStatus(); lastStatusPrint = Date() }
    }
    private func showStatus() {
        let states = ["READY", "RECORDING", "SAVING", "COMPLETE", "BUFFER FULL"]
        if let s = status {
            let timing = String(format: "%.1f Hz | %.1f s", recorder?.measuredHz ?? 0, recorder?.elapsedSeconds ?? 0)
            print("\(states[Int(s.state)]) | \(timing) | acquired=\(s.produced) saved=\(recorder?.exclusive ?? 0) pending=\(s.pending)\(s.flags & 8 != 0 ? " | command rejected" : "")")
        } else { print("Disconnected or connecting | saved=\(recorder?.exclusive ?? 0)") }
    }
    func command(_ line: String) {
        let tokens = line.split(maxSplits: 1, whereSeparator: { $0.isWhitespace }).map(String.init)
        guard let name = tokens.first else { return }
        let value = tokens.count > 1 ? tokens[1] : ""
        switch name {
        case "status": showStatus()
        case "start":
            guard readySent, let s = status, s.flags & 2 != 0, !s.recording, s.pending == 0 else { print("Wait for a ready connection and finish saving the preceding session."); return }
            send(Wire.command(1), label: "start")
        case "stop":
            guard let s = status, s.recording else { print("No active recording."); return }
            send(Wire.command(2, session: s.session), label: "stop")
        case "mark":
            guard !value.isEmpty, let s = status, s.recording, let recorder = recorder else { print("Use mark LABEL during recording."); return }
            do { let id = try recorder.nextMarker(); try recorder.setMarker(id, label: value); send(Wire.command(4, session: s.session, marker: id), label: "marker") }
            catch { fail("Cannot save marker: \(error)") }
        case "activity":
            guard !value.isEmpty, status?.recording != true, (status?.pending ?? 0) == 0 else { print("Set activity LABEL between sessions."); return }
            profile["activity"] = value
            do { try saveProfile(); print("Next session activity: \(value)") } catch { fail("Cannot save profile: \(error)") }
        case "detect":
            guard let s = status, !s.recording, s.pending == 0 else { print("Stop recording and wait for all data to be saved first."); return }
            send(Wire.command(6), label: "detect")
        case "quit":
            if value == "force" { close(); return }
            quitting = true
            if let s = status, peripheral?.state == .connected {
                if s.recording { send(Wire.command(2, session: s.session), label: "stop"); print("Stopping; waiting for saved completion.") }
                else if s.pending == 0 { close() }
                else { print("Waiting for pending samples. Use quit force only to detach without finishing transfer.") }
            } else if recorder == nil { close() }
            else { print("Disconnected: reconnect to finish saving. Use quit force or interrupt again to detach.") }
        default: print("Commands: status, start, stop, mark LABEL, activity LABEL, detect, quit, quit force")
        }
    }
    private func fail(_ message: String) {
        print("ERROR: \(message). No further acknowledgements will be sent.")
        fatalStorage = true
        if let peripheral = peripheral { central.cancelPeripheralConnection(peripheral) }
    }
    private func close() {
        do { try recorder?.exportCSV() } catch { print("Export failed; journal retained: \(error)") }
        if assertion != 0 { IOPMAssertionRelease(assertion) }
        if let activity = activity { ProcessInfo.processInfo.endActivity(activity) }
        exit(fatalStorage ? 1 : 0)
    }
}

do {
    let collector = Collector(options: try Options())
    try collector.run()
    withExtendedLifetime(collector) { RunLoop.main.run() }
} catch {
    fputs("\(error)\n", stderr); exit(1)
}
