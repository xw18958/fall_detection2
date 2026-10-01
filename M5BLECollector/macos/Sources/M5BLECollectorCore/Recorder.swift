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
    private let createdUTC: Double
    private var metadata: [String: Any] = [:]
    private var completion: [String: Any] = ["complete": false, "saved_samples": 0]

    public static func journalURL(in directory: URL) -> URL {
        directory.appendingPathComponent(".recovery/journal.jsonl")
    }
    public static func hasJournal(in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: journalURL(in: directory).path) ||
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("journal.jsonl").path)
    }

    // Caller must hold the recordings-folder lock. Unfinished sessions migrate
    // when attached, so this offline pass never invents a completion decision.
    public static func migrateCompletedRecordings(root: URL) throws -> Int {
        let entries = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        var migrated = 0
        for entry in entries {
            guard ["journal.jsonl", "quality_report.json", "completion.json"].contains(where: {
                FileManager.default.fileExists(atPath: entry.appendingPathComponent($0).path)
            }) else { continue }
            let data = try Data(contentsOf: entry.appendingPathComponent("metadata.json"))
            guard let old = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let hex = old["session_id"] as? String, let session = UInt64(hex, radix: 16),
                  let device = old["device"] else { throw ProtocolError.invalid("Invalid legacy recording metadata") }
            let completionURL = entry.appendingPathComponent("completion.json")
            var completion = old["completion"] as? [String: Any]
            if completion == nil, FileManager.default.fileExists(atPath: completionURL.path) {
                completion = try JSONSerialization.jsonObject(with: Data(contentsOf: completionURL)) as? [String: Any]
            }
            guard completion?["complete"] as? Bool == true else { continue }
            let info = try JSONDecoder().decode(DeviceInfo.self, from: JSONSerialization.data(withJSONObject: device))
            _ = try Recorder(root: root, session: session, info: info, profile: old["profile"] as? [String: String] ?? [:])
            migrated += 1
        }
        return migrated
    }

    private static func safeName(_ value: String, fallback: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let mapped = value.unicodeScalars.map { allowed.contains($0) ? Character(String($0)) : Character("_") }
        let text = String(mapped).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return text.isEmpty ? fallback : String(text.prefix(40))
    }

    private static func existingDirectory(root: URL, sessionHex: String) -> URL? {
        guard let entries = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { return nil }
        for entry in entries {
            let metadata = entry.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: metadata),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  json["session_id"] as? String == sessionHex else { continue }
            return entry
        }
        return nil
    }

    private static func newDirectory(root: URL, sessionHex: String, profile: [String: String], created: Date) -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let participant = safeName(profile["participant"] ?? "", fallback: "participant")
        let shortSession = String(sessionHex.suffix(8))
        return root.appendingPathComponent("\(formatter.string(from: created))_\(participant)_\(shortSession)")
    }

    public init(root: URL, session: UInt64, info: DeviceInfo, profile: [String: String]) throws {
        try info.validate()
        guard session != 0 else { throw ProtocolError.invalid("Cannot record a zero session") }
        self.session = session; self.info = info; self.profile = profile.filter { $0.key != "activity" }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sessionHex = String(format: "%016llx", session)
        directory = Self.existingDirectory(root: root, sessionHex: sessionHex) ?? Self.newDirectory(root: root, sessionHex: sessionHex, profile: profile, created: Date())
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let metadataURL = directory.appendingPathComponent("metadata.json")
        if FileManager.default.fileExists(atPath: metadataURL.path) {
            let old = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
            let oldInfo = old?["device"] as? [String: Any]
            guard oldInfo?["model_sha256"] as? String == info.model_sha256,
                  oldInfo?["boot_id"] as? String == info.boot_id,
                  old?["session_id"] as? String == sessionHex else {
                throw ProtocolError.invalid("Existing session metadata does not match this device")
            }
            if let savedProfile = old?["profile"] as? [String: String] { self.profile = savedProfile.filter { $0.key != "activity" } }
            metadata = old ?? [:]
            createdUTC = old?["created_timestamp_utc"] as? Double ?? Date().timeIntervalSince1970
        } else {
            createdUTC = Date().timeIntervalSince1970
        }
        if let saved = metadata["completion"] as? [String: Any] {
            completion = saved
        } else {
            let legacy = directory.appendingPathComponent("completion.json")
            if FileManager.default.fileExists(atPath: legacy.path) {
                guard let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: legacy)) as? [String: Any] else {
                    throw ProtocolError.invalid("Invalid legacy completion")
                }
                completion = saved
            }
        }
        let hiddenJournal = Self.journalURL(in: directory)
        let legacyJournal = directory.appendingPathComponent("journal.jsonl")
        guard !(FileManager.default.fileExists(atPath: hiddenJournal.path) && FileManager.default.fileExists(atPath: legacyJournal.path)) else {
            throw ProtocolError.invalid("Both legacy and hidden journals exist; refusing ambiguous recovery")
        }
        let journal = FileManager.default.fileExists(atPath: legacyJournal.path) ? legacyJournal : hiddenJournal
        try FileManager.default.createDirectory(at: hiddenJournal.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: journal.path) {
            let data = try Data(contentsOf: journal)
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
            if completion["complete"] as? Bool == true {
                guard completion["saved_samples"] as? UInt32 == exclusive else {
                    throw ProtocolError.invalid("Completed recording disagrees with recovered journal count")
                }
            }
            handle = try FileHandle(forUpdating: journal)
            if end != data.count { try handle.truncate(atOffset: UInt64(end)); try handle.synchronize() }
        } else {
            guard completion["complete"] as? Bool != true else { throw ProtocolError.invalid("Completed recording has no recovery journal") }
            guard FileManager.default.createFile(atPath: journal.path, contents: nil) else { throw ProtocolError.invalid("Cannot create journal") }
            handle = try FileHandle(forUpdating: journal)
        }
        try handle.seekToEnd()
        try handle.synchronize()
        try exportCSV()
        // Metadata and CSV are durable before old visible files are relocated.
        // Keep the original sidecars as hidden backups rather than deleting them.
        for name in ["journal.jsonl", "quality_report.json", "completion.json"] {
            let source = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            let target = directory.appendingPathComponent(".recovery/\(name)")
            guard !FileManager.default.fileExists(atPath: target.path) else { throw ProtocolError.invalid("Recovery backup already exists: \(name)") }
            try FileManager.default.moveItem(at: source, to: target)
        }
        try syncDirectory(hiddenJournal.deletingLastPathComponent())
        try syncDirectory(directory)
    }
    deinit { try? handle.close() }

    private static func validate(_ sample: Sample) throws {
        guard sample.raw.count == 6, sample.seq < UInt32.max, sample.flags & ~UInt16(7) == 0 else {
            throw ProtocolError.invalid("Invalid sample fields")
        }
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
            guard completion["complete"] as? Bool != true else { throw ProtocolError.invalid("Cannot append new samples to a completed recording") }
            var data = Data()
            for row in pending { data.append(try encoder.encode(row)); data.append(10) }
            try handle.write(contentsOf: data); try handle.synchronize()
            rows.append(contentsOf: pending); exclusive = next
        }
        return exclusive
    }

    private func saveMetadata() throws {
        let device = try JSONSerialization.jsonObject(with: encoder.encode(info))
        let fields: [String: Any] = [
            "schema_version": 3,
            "session_id": String(format: "%016llx", session),
            "created_timestamp_utc": createdUTC,
            "profile": profile,
            "device": device,
            "sample_columns": ["seq", "device_timestamp_us", "ax", "ay", "az", "gx", "gy", "gz"],
            "sample_value_note": "ax,ay,az,gx,gy,gz are untouched signed 16-bit MPU6886 register counts; conversions are intentionally left to downstream processing",
            "timestamp_note": "device_timestamp_us is monotonic acquisition time; host reception time is retained only in .recovery/journal.jsonl for transport debugging",
            "completion": completion
        ]
        metadata.merge(fields) { _, new in new }
        if metadata["annotation"] == nil { metadata["annotation"] = ["fall_label": NSNull()] }
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
        try syncDirectory(url.deletingLastPathComponent())
    }
    private func syncDirectory(_ url: URL) throws {
        let fd = Darwin.open(url.path, O_RDONLY)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(fd) }
        guard Darwin.fsync(fd) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
    public func exportCSV() throws {
        var samples = "seq,device_timestamp_us,ax,ay,az,gx,gy,gz\n"
        var readErrors = 0, saturated = 0, timingFlags = 0, gaps = 0
        var deltas: [Double] = []
        for (index, row) in rows.enumerated() {
            let s = row.sample
            var columns = [String(s.seq), String(s.device_timestamp_us)]
            columns.append(contentsOf: s.raw.map { String($0) })
            samples += columns.joined(separator: ",") + "\n"
            if s.flags & 1 != 0 { readErrors += 1 }
            if s.flags & 2 != 0 { timingFlags += 1 }
            if s.flags & 4 != 0 { saturated += 1 }
            if index > 0 {
                let delta = Double(s.device_timestamp_us - rows[index-1].sample.device_timestamp_us)/1e6
                deltas.append(delta)
                if delta > 1.5/Double(info.rate_hz) { gaps += 1 }
            }
        }
        try durableReplace(Data(samples.utf8), at: directory.appendingPathComponent("samples.csv"))
        let elapsed = deltas.reduce(0, +), mean = deltas.isEmpty ? 0 : elapsed/Double(deltas.count)
        let variance = deltas.isEmpty ? 0 : deltas.reduce(0) { $0 + ($1-mean)*($1-mean) }/Double(deltas.count)
        let report: [String: Any] = [
            "saved_samples": rows.count,
            "read_errors": readErrors,
            "saturated_samples": saturated,
            "timing_gap_flags": timingFlags,
            "observed_timestamp_gaps": gaps,
            "elapsed_seconds": elapsed,
            "measured_hz": elapsed > 0 ? Double(deltas.count)/elapsed : 0,
            "interval_stddev_ms": sqrt(variance)*1000,
            "max_interval_ms": (deltas.max() ?? 0)*1000,
            "sequence_gaps": 0
        ]
        metadata["quality"] = report
        if completion["complete"] as? Bool != true { completion["saved_samples"] = exclusive }
        try saveMetadata()
    }
    public func finish(produced: UInt32, overflowed: Bool) throws {
        guard exclusive == produced else { throw ProtocolError.invalid("Device completion disagrees with saved sample count") }
        try exportCSV()
        if completion["complete"] as? Bool == true {
            guard completion["device_buffer_overflowed"] as? Bool == overflowed else {
                throw ProtocolError.invalid("Device overflow state disagrees with saved completion")
            }
            return
        }
        completion = [
            "complete": true,
            "saved_samples": exclusive,
            "device_buffer_overflowed": overflowed,
            "finished_timestamp_utc": Date().timeIntervalSince1970
        ]
        try saveMetadata()
    }
}
