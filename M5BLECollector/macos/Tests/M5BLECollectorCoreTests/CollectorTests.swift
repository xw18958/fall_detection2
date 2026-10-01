import Foundation
import M5BLECollectorCore

// Standalone tests run with Apple's Command Line Tools; XCTest/Xcode is not required.
func XCTAssertEqual<T: Equatable>(_ a: T, _ b: T, file: StaticString = #file, line: UInt = #line) { precondition(a == b, "Expected \(a) == \(b)", file: file, line: line) }
func XCTAssertTrue(_ value: Bool, file: StaticString = #file, line: UInt = #line) { precondition(value, "Expected true", file: file, line: line) }
func XCTAssertFalse(_ value: Bool, file: StaticString = #file, line: UInt = #line) { precondition(!value, "Expected false", file: file, line: line) }
func XCTAssertNil<T>(_ value: T?, file: StaticString = #file, line: UInt = #line) { precondition(value == nil, "Expected nil", file: file, line: line) }
func XCTAssertThrowsError<T>(_ body: @autoclosure () throws -> T, file: StaticString = #file, line: UInt = #line) {
    do { _ = try body() } catch { return }
    preconditionFailure("Expected an error", file: file, line: line)
}
final class CollectorTests {
    private var cleanup: [() -> Void] = []
    func addTeardownBlock(_ block: @escaping () -> Void) { cleanup.append(block) }
    deinit { cleanup.forEach { $0() } }
    private func info() throws -> DeviceInfo {
        let json = """
        {"protocol":1,"device_id":"112233445566","boot_id":"123456789abcdef0","firmware":"1.2.0-ble.1",
        "model_sha256":"1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530",
        "rate_hz":30,"accel_range_g":8,"gyro_range_dps":2000,"buffer_records":8192,
        "accel_g_per_lsb":0.000244140625,"gyro_dps_per_lsb":0.06103515625,"time_us":1,"last_error":0}
        """
        return try JSONDecoder().decode(DeviceInfo.self, from: Data(json.utf8))
    }
    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("m5ble-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    private func sample(_ seq: UInt32) -> Sample {
        Sample(seq: seq, deviceTimestamp: 1_000_000+UInt64(seq)*33333, raw: [-32768,32767,-123,123,0,-1])
    }
    private func put(_ value: UInt64, bytes: Int) -> [UInt8] { (0..<bytes).map { UInt8(truncatingIfNeeded: value >> (8*$0)) } }
    private func record(_ sample: Sample) -> Data {
        var b = put(UInt64(sample.seq), bytes: 4)+put(sample.device_timestamp_us, bytes: 8)
        for raw in sample.raw { b += put(UInt64(UInt16(bitPattern: raw)), bytes: 2) }
        b += put(UInt64(sample.flags), bytes: 2)+[0,0,0,0,0,0]
        return Data(b)
    }
    private func fragments(_ payload: Data, chunk: Int, session: UInt64 = 42, batch: UInt32 = 7) -> [Data] {
        let parts = (payload.count+chunk-1)/chunk
        return (0..<parts).map { index in
            let header: [UInt8] = [1,1,UInt8(index),UInt8(parts)] + put(session, bytes: 8)+put(UInt64(batch), bytes: 4)
            return Data(header)+payload.subdata(in: index*chunk..<min(payload.count,(index+1)*chunk))
        }
    }
    func testSmallMTUFragmentationAndDuplicateReplay() throws {
        let assembler = FrameAssembler(), payload = record(sample(0))+record(sample(1))+record(sample(2))
        let frames = fragments(payload, chunk: 4) // Default ATT MTU 23: 20-byte notifications.
        XCTAssertEqual(frames.count,24)
        XCTAssertNil(try assembler.accept(frames[0])); XCTAssertNil(try assembler.accept(frames[0]))
        var batch: Batch?
        for frame in frames.dropFirst() { batch = try assembler.accept(frame) }
        XCTAssertEqual(batch?.session,42); XCTAssertEqual(batch?.samples,[sample(0),sample(1),sample(2)])
        for frame in frames { batch = try assembler.accept(frame) }
        XCTAssertEqual(batch?.samples.count,3)
    }
    func testCppWireFixture(_ path: String) throws {
        let bytes = try Data(contentsOf: URL(fileURLWithPath:path))
        XCTAssertEqual(try Sample.decode(bytes),[sample(0),sample(1),sample(2)])
        let assembler = FrameAssembler()
        var batch: Batch?
        for frame in fragments(bytes,chunk:4) { batch = try assembler.accept(frame) }
        XCTAssertEqual(batch?.samples,[sample(0),sample(1),sample(2)])
    }
    func testDisconnectDropsPartialBatchAndNewBatchReassembles() throws {
        let assembler = FrameAssembler(), frames = fragments(record(sample(0)), chunk: 4)
        XCTAssertNil(try assembler.accept(frames[0])); assembler.reset()
        var batch: Batch?
        for frame in fragments(record(sample(0)),chunk:166,batch:8) { batch = try assembler.accept(frame) }
        XCTAssertEqual(batch?.samples,[sample(0)])
        XCTAssertThrowsError(try assembler.accept(Data([1,1])))
    }
    func testJournalReplayDedupAndTruncatedTailRecovery() throws {
        let root = try root(), info = try info()
        var recorder: Recorder? = try Recorder(root:root,session:42,info:info,profile:["participant":"P001", "activity":"walking"])
        XCTAssertEqual(try recorder!.append([sample(0),sample(1)]),2)
        XCTAssertEqual(try recorder!.append([sample(0),sample(1)]),2)
        let directory = recorder!.directory; recorder = nil
        let handle = try FileHandle(forWritingTo:directory.appendingPathComponent("journal.jsonl"))
        try handle.seekToEnd(); try handle.write(contentsOf:Data("{\"partial\":".utf8)); try handle.close()
        let recovered = try Recorder(root:root,session:42,info:info,profile:["participant":"P999", "activity":"wrong-new-profile"])
        XCTAssertEqual(recovered.exclusive,2)
        XCTAssertEqual(try recovered.append([sample(1),sample(2)]),3)
        try recovered.finish(produced:3,overflowed:false)
        let csv = try String(contentsOf:directory.appendingPathComponent("samples.csv"))
        XCTAssertEqual(csv.split(separator:"\n").count,4)
        XCTAssertTrue(csv.hasPrefix("seq,device_timestamp_us,ax,ay,az,gx,gy,gz\n"))
        XCTAssertTrue(csv.contains("-32768,32767,-123,123,0,-1"))
        XCTAssertFalse(csv.contains("host_received_timestamp_utc"))
        let meta = try String(contentsOf:directory.appendingPathComponent("metadata.json"))
        XCTAssertTrue(meta.contains("P001")); XCTAssertFalse(meta.contains("walking")); XCTAssertFalse(meta.contains("wrong-new-profile")); XCTAssertFalse(meta.contains("activity"))
    }
    func testInvalidSequenceCannotAdvanceAcknowledgement() throws {
        let recorder = try Recorder(root:root(),session:42,info:info(),profile:[:])
        XCTAssertThrowsError(try recorder.append([sample(1)])); XCTAssertEqual(recorder.exclusive,0)
        XCTAssertEqual(try recorder.append([sample(0)]),1)
        let altered = Sample(seq:0,deviceTimestamp:1_000_000,raw:[1,2,3,4,5,6])
        XCTAssertThrowsError(try recorder.append([altered])); XCTAssertEqual(recorder.exclusive,1)
        XCTAssertThrowsError(try recorder.finish(produced:2,overflowed:false))
        let backwards = Sample(seq:1,deviceTimestamp:5,raw:[1,2,3,4,5,6])
        XCTAssertThrowsError(try recorder.append([backwards])); XCTAssertEqual(recorder.exclusive,1)
    }
    func testQualityFlagsStayOutOfTrainingCSV() throws {
        let recorder = try Recorder(root:root(),session:42,info:info(),profile:[:])
        let flagged = Sample(seq:0,deviceTimestamp:1_000_000,raw:[0,0,0,0,0,0],flags:7)
        _ = try recorder.append([flagged]); try recorder.finish(produced:1,overflowed:true)
        let csv = try String(contentsOf:recorder.directory.appendingPathComponent("samples.csv"))
        XCTAssertEqual(csv.split(separator:"\n").count,2)
        XCTAssertFalse(csv.contains("flags"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: recorder.directory.appendingPathComponent("events.csv").path))
        let quality = try JSONSerialization.jsonObject(with:Data(contentsOf:recorder.directory.appendingPathComponent("quality_report.json"))) as! [String:Any]
        XCTAssertEqual(quality["read_errors"] as? Int,1)
        XCTAssertEqual(quality["timing_gap_flags"] as? Int,1)
        XCTAssertEqual(quality["saturated_samples"] as? Int,1)
    }
}

@main struct SmokeRunner {
    static func main() throws {
        let tests = CollectorTests()
        try tests.testSmallMTUFragmentationAndDuplicateReplay()
        try tests.testDisconnectDropsPartialBatchAndNewBatchReassembles()
        try tests.testJournalReplayDedupAndTruncatedTailRecovery()
        try tests.testInvalidSequenceCannotAdvanceAcknowledgement()
        try tests.testQualityFlagsStayOutOfTrainingCSV()
        if CommandLine.arguments.count == 2 { try tests.testCppWireFixture(CommandLine.arguments[1]) }
        print("PASS: protocol/journal smoke scenarios; no Bluetooth or device access.")
    }
}
