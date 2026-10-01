import Foundation
import Darwin

public struct JournalRow: Codable {
    public let sample: Sample
    public let host_received_timestamp_utc: Double
}

public final class Recorder {
    public let directory: URL, session: UInt64
    public private(set) var exclusive: UInt32 = 0
    public var elapsedSeconds: Double {
        guard let first = rows.first, let last = rows.last else { return 0 }
        return Double(last.sample.device_timestamp_us - first.sample.device_timestamp_us) / 1e6
    }
    public var measuredHz: Double { elapsedSeconds > 0 ? Double(max(0, rows.count - 1)) / elapsedSeconds : 0 }
    private let handle: FileHandle
    private var rows: [JournalRow] = []
    private var encoder = JSONEncoder()
    private let info: DeviceInfo
    private var profile: [String: String]
    private var markerLabels: [String: String]
    private let createdUTC: Double

    public init(root: URL, session: UInt64, info: DeviceInfo, profile: [String: String]) throws {
        try info.validate()
        guard session != 0 else { throw ProtocolError.invalid("Cannot record a zero session") }
        self.session = session; self.info = info; self.profile = profile
        directory = root.appendingPathComponent("\(info.device_id)_\(info.boot_id)_\(String(format: "%016llx", session))")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let metadataURL = directory.appendingPathComponent("metadata.json")
        if FileManager.default.fileExists(atPath: metadataURL.path) {
            let old = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
            let oldInfo = old?["device"] as? [String: Any]
            guard oldInfo?["model_sha256"] as? String == info.model_sha256,
                  oldInfo?["boot_id"] as? String == info.boot_id,
                  old?["session_id"] as? String == String(format: "%016llx", session) else {
                throw ProtocolError.invalid("Existing session metadata does not match this device")
            }
            if let savedProfile = old?["profile"] as? [String: String] { self.profile = savedProfile }
            markerLabels = old?["marker_labels"] as? [String: String] ?? ["1": "device_button_marker"]
            createdUTC = old?["created_timestamp_utc"] as? Double ?? Date().timeIntervalSince1970
        } else {
            markerLabels = ["1": "device_button_marker"]; createdUTC = Date().timeIntervalSince1970
        }
        let journal = directory.appendingPathComponent("journal.jsonl")
        if FileManager.default.fileExists(atPath: journal.path) {
            let data = try Data(contentsOf: journal)
            // A crash can leave the final line incomplete; committed complete lines remain authoritative.
            let end = data.lastIndex(of: 10).map { $0 + 1 } ?? 0
            let committed = data.prefix(end)
            for line in committed.split(separator: 10) {
                let row = try JSONDecoder().decode(JournalRow.self, from: Data(line))
                try Self.validate(row.sample)
                guard row.sample.seq == exclusive else { throw ProtocolError.invalid("Journal contains a sequence gap or duplicate") }
                if let last = rows.last, row.sample.device_timestamp_us <= last.sample.device_timestamp_us {
                    throw ProtocolError.invalid("Journal timestamp is not increasing")
                }
                rows.append(row); exclusive += 1
            }
            handle = try FileHandle(forUpdating: journal)
            if end != data.count { try handle.truncate(atOffset: UInt64(end)); try handle.synchronize() }
        } else {
            guard FileManager.default.createFile(atPath: journal.path, contents: nil) else { throw ProtocolError.invalid("Cannot create journal") }
            handle = try FileHandle(forUpdating: journal)
        }
        try handle.seekToEnd()
        // Recovered complete lines must also be synchronized before a replay ACK.
        try handle.synchronize()
        try saveMetadata()
        try exportCSV()
    }
    deinit { try? handle.close() }

    private static func validate(_ sample: Sample) throws {
        guard sample.raw.count == 6, sample.seq < UInt32.max, sample.flags & ~UInt16(15) == 0,
              (sample.marker == 0) == (sample.flags & 8 == 0) else { throw ProtocolError.invalid("Invalid sample fields") }
    }

    // ACK is permitted only after this method returns: synchronize() happens before exclusive advances.
    public func append(_ samples: [Sample], receivedUTC: Double = Date().timeIntervalSince1970) throws -> UInt32 {
        var pending: [JournalRow] = []
        var next = exclusive
        var previousTime = rows.last?.sample.device_timestamp_us
        for sample in samples {
            try Self.validate(sample)
            if sample.seq < exclusive {
                guard rows[Int(sample.seq)].sample == sample else { throw ProtocolError.invalid("Duplicate sample differs from committed data") }
                continue
            }
            guard sample.seq == next else { throw ProtocolError.invalid("Sequence gap; refusing to ACK unsaved samples") }
            if let time = previousTime, sample.device_timestamp_us <= time { throw ProtocolError.invalid("Device clock went backwards") }
            pending.append(JournalRow(sample: sample, host_received_timestamp_utc: receivedUTC)); previousTime = sample.device_timestamp_us; next += 1
        }
        if !pending.isEmpty {
            var data = Data()
            for row in pending { data.append(try encoder.encode(row)); data.append(10) }
            try handle.write(contentsOf: data); try handle.synchronize()
            rows.append(contentsOf: pending); exclusive = next
        }
        return exclusive
    }

    public func setMarker(_ id: UInt16, label: String) throws {
        markerLabels[String(id)] = label; try saveMetadata()
    }
    public func nextMarker() throws -> UInt16 {
        let next = max(2, (markerLabels.keys.compactMap { UInt32($0) }.max() ?? 1) + 1)
        guard next <= UInt16.max else { throw ProtocolError.invalid("Marker IDs exhausted") }
        return UInt16(next)
    }
    private func saveMetadata() throws {
        let device = try JSONSerialization.jsonObject(with: encoder.encode(info))
        let metadata: [String: Any] = ["schema_version": 1, "session_id": String(format: "%016llx", session),
            "created_timestamp_utc": createdUTC, "profile": profile, "device": device, "marker_labels": markerLabels,
            "timestamp_note": "device_timestamp_us is monotonic acquisition time; host timestamp is reception time, not acquisition UTC"]
        let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
        try durableReplace(data, at: directory.appendingPathComponent("metadata.json"))
    }
    private func durableReplace(_ data: Data, at url: URL) throws {
        let temp = url.appendingPathExtension("tmp")
        try data.write(to: temp)
        let file = try FileHandle(forWritingTo: temp)
        do { try file.synchronize(); try file.close() } catch { try? file.close(); throw error }
        let result = temp.path.withCString { source in url.path.withCString { target in Darwin.rename(source, target) } }
        guard result == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let fd = Darwin.open(directory.path, O_RDONLY)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(fd) }
        guard Darwin.fsync(fd) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
    private func csv(_ text: String) -> String { "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
    public func exportCSV() throws {
        var samples = "seq,device_timestamp_us,host_received_timestamp_utc,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,flags,marker\n"
        var events = "seq,device_timestamp_us,marker,label\n"
        var readErrors = 0, saturated = 0, timingFlags = 0, gaps = 0
        var deltas: [Double] = []
        for (index, row) in rows.enumerated() {
            let s = row.sample
            let units = s.raw.enumerated().map { Double($0.element) * ($0.offset < 3 ? info.accel_g_per_lsb : info.gyro_dps_per_lsb) }
            var columns = [String(s.seq), String(s.device_timestamp_us), String(row.host_received_timestamp_utc)]
            columns.append(contentsOf: s.raw.map { String($0) })
            columns.append(contentsOf: units.map { String($0) })
            columns.append(contentsOf: [String(s.flags), String(s.marker)])
            samples += columns.joined(separator: ",") + "\n"
            if s.marker != 0 { events += "\(s.seq),\(s.device_timestamp_us),\(s.marker),\(csv(markerLabels[String(s.marker)] ?? "unknown"))\n" }
            if s.flags & 1 != 0 { readErrors += 1 }; if s.flags & 2 != 0 { timingFlags += 1 }; if s.flags & 4 != 0 { saturated += 1 }
            if index > 0 { let delta = Double(s.device_timestamp_us - rows[index-1].sample.device_timestamp_us)/1e6; deltas.append(delta); if delta > 1.5/Double(info.rate_hz) { gaps += 1 } }
        }
        try durableReplace(Data(samples.utf8), at: directory.appendingPathComponent("samples.csv"))
        try durableReplace(Data(events.utf8), at: directory.appendingPathComponent("events.csv"))
        let elapsed = deltas.reduce(0, +), mean = deltas.isEmpty ? 0 : elapsed/Double(deltas.count)
        let variance = deltas.isEmpty ? 0 : deltas.reduce(0) { $0 + ($1-mean)*($1-mean) }/Double(deltas.count)
        let report: [String: Any] = ["saved_samples": rows.count, "read_errors": readErrors, "saturated_samples": saturated,
            "timing_gap_flags": timingFlags, "observed_timestamp_gaps": gaps, "elapsed_seconds": elapsed,
            "measured_hz": elapsed > 0 ? Double(deltas.count)/elapsed : 0, "interval_stddev_ms": sqrt(variance)*1000,
            "max_interval_ms": (deltas.max() ?? 0)*1000, "sequence_gaps": 0]
        try durableReplace(try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]), at: directory.appendingPathComponent("quality_report.json"))
    }
    public func finish(produced: UInt32, overflowed: Bool) throws {
        guard exclusive == produced else { throw ProtocolError.invalid("Device completion disagrees with saved sample count") }
        try exportCSV()
        let report: [String: Any] = ["complete": true, "saved_samples": exclusive, "device_buffer_overflowed": overflowed,
                                   "finished_timestamp_utc": Date().timeIntervalSince1970]
        try durableReplace(try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted]), at: directory.appendingPathComponent("completion.json"))
    }
}
