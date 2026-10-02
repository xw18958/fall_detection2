import M5BLECollectorCore

func testTrustedModelIdentities() throws {
    for hash in TrustedModels.hashes { try TrustedModels.validate(hash) }
    XCTAssertEqual(TrustedModels.hashes.count, 2)
    XCTAssertThrowsError(try TrustedModels.validate(String(repeating: "0", count: 64)))
    XCTAssertThrowsError(try TrustedModels.validate("ddac71cf"))
    print("PASS: exact old/new model compatibility; arbitrary hashes rejected.")
}
