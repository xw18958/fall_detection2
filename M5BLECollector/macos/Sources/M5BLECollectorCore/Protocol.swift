import Foundation

public enum ProtocolError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String { if case .invalid(let reason) = self { return reason }; return "Invalid data" }
}

public enum Wire {
    public static let service = "6F4D0001-8F2B-4E3D-9A11-189580000001"
    public static let info = "6F4D0001-8F2B-4E3D-9A11-189580000002"
    public static let control = "6F4D0001-8F2B-4E3D-9A11-189580000003"
    public static let samples = "6F4D0001-8F2B-4E3D-9A11-189580000004"
    public static let status = "6F4D0001-8F2B-4E3D-9A11-189580000005"
    public static func integer(_ bytes: [UInt8], at offset: Int, count: Int) -> UInt64 {
        (0..<count).reduce(UInt64(0)) { $0 | UInt64(bytes[offset + $1]) << (8 * $1) }
    }
    public static func command(_ op: UInt8, session: UInt64 = 0, exclusive: UInt32 = 0) -> Data {
        var b: [UInt8] = [1, op]
        for i in 0..<8 { b.append(UInt8(truncatingIfNeeded: session >> (8*i))) }
        for i in 0..<4 { b.append(UInt8(truncatingIfNeeded: exclusive >> (8*i))) }
        b += [0, 0]
        return Data(b)
    }
}

public struct DeviceInfo: Codable {
    public let `protocol`: Int
    public let device_id, boot_id, firmware, model_sha256: String
    public let rate_hz, accel_range_g, gyro_range_dps, buffer_records: Int
    public let accel_g_per_lsb, gyro_dps_per_lsb: Double
    public let time_us: UInt64
    public let last_error: Int
    public func validate() throws {
        guard `protocol` == 1, rate_hz == 30, accel_range_g == 8, gyro_range_dps == 2000,
              accel_g_per_lsb == 8.0/32768, gyro_dps_per_lsb == 2000.0/32768,
              device_id.count == 12, device_id.allSatisfy({ $0.isHexDigit }),
              boot_id.count == 16, boot_id.allSatisfy({ $0.isHexDigit }),
              model_sha256.count == 64, model_sha256.allSatisfy({ $0.isHexDigit }) else {
            throw ProtocolError.invalid("Unsupported sensor configuration or device identity")
        }
    }
}

public struct DeviceStatus {
    public let state, flags: UInt8
    public let session: UInt64
    public let produced, pending: UInt32
    public var recording: Bool { state == 1 }
    public var review: Bool { state == 2 || state == 5 }
    public var saving: Bool { state == 3 }
    public var complete: Bool { state == 4 && pending == 0 }
    public var idle: Bool { (state == 0 || state == 4) && pending == 0 }
    public init(_ data: Data) throws {
        let b = Array(data)
        guard b.count == 20, b[0] == 1, b[1] == 1, b[2] <= 5 else { throw ProtocolError.invalid("Invalid status packet") }
        state = b[2]; flags = b[3]; session = Wire.integer(b, at: 4, count: 8)
        produced = UInt32(Wire.integer(b, at: 12, count: 4)); pending = UInt32(Wire.integer(b, at: 16, count: 4))
        guard pending <= produced else { throw ProtocolError.invalid("Invalid pending count") }
    }
}

public struct Sample: Codable, Equatable {
    public let seq: UInt32, device_timestamp_us: UInt64
    public let raw: [Int16]
    public let flags: UInt16
    public init(seq: UInt32, deviceTimestamp: UInt64, raw: [Int16], flags: UInt16 = 0) {
        self.seq = seq; device_timestamp_us = deviceTimestamp; self.raw = raw; self.flags = flags
    }
    public static func decode(_ data: Data) throws -> [Sample] {
        let b = Array(data)
        guard !b.isEmpty, b.count <= 192, b.count % 32 == 0 else { throw ProtocolError.invalid("Invalid sample batch length") }
        return try stride(from: 0, to: b.count, by: 32).map { start in
            guard b[(start+26)..<(start+32)].allSatisfy({ $0 == 0 }) else { throw ProtocolError.invalid("Nonzero reserved bytes") }
            let raw = (0..<6).map { Int16(bitPattern: UInt16(Wire.integer(b, at: start+12+2*$0, count: 2))) }
            return Sample(seq: UInt32(Wire.integer(b, at: start, count: 4)),
                          deviceTimestamp: Wire.integer(b, at: start+4, count: 8), raw: raw,
                          flags: UInt16(Wire.integer(b, at: start+24, count: 2)))
        }
    }
}

public struct Batch { public let session: UInt64; public let samples: [Sample] }

public final class FrameAssembler {
    private var session: UInt64 = 0, batch: UInt32 = 0
    private var parts = 0
    private var fragments: [Int: Data] = [:]
    public init() {}
    public func reset() { fragments.removeAll(); parts = 0 }
    public func accept(_ data: Data) throws -> Batch? {
        let b = Array(data)
        guard b.count > 16, b.count <= 256, b[0] == 1, b[1] == 1,
              b[3] > 0, b[3] <= 48, b[2] < b[3] else { throw ProtocolError.invalid("Malformed BLE fragment") }
        let s = Wire.integer(b, at: 4, count: 8), id = UInt32(Wire.integer(b, at: 12, count: 4))
        guard s != 0 else { throw ProtocolError.invalid("Zero session") }
        if parts == 0 || s != session || id != batch {
            reset(); session = s; batch = id; parts = Int(b[3])
        }
        guard parts == Int(b[3]) else { throw ProtocolError.invalid("Fragment count changed") }
        let payload = data.dropFirst(16), part = Int(b[2])
        if let old = fragments[part], old != payload { throw ProtocolError.invalid("Conflicting duplicate fragment") }
        fragments[part] = Data(payload)
        guard fragments.values.reduce(0, { $0 + $1.count }) <= 192 else { throw ProtocolError.invalid("Oversized batch") }
        guard fragments.count == parts else { return nil }
        var records = Data()
        for i in 0..<parts { guard let fragment = fragments[i] else { return nil }; records.append(fragment) }
        let samples = try Sample.decode(records); reset()
        return Batch(session: s, samples: samples)
    }
}
