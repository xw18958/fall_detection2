import M5BLECollectorCore

func testTrustedModelIdentities() throws {
    for hash in TrustedModels.hashes { try TrustedModels.validate(hash) }
    XCTAssertEqual(TrustedModels.hashes, Set([
        "1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530",
        "ddac71cffd5e10f85d9e9122fe5334a3dbabf3717da41b598231ef4a69a720bc",
        "7a649721e02ab88e85594dd67efe415a0d5a777d9215df03d903dd0de748cf57",
        "4eb70382f3306a4c8580e5d08342032edae43d3bd4e07cb4a3a9090e85dfa367"
    ]))
    XCTAssertThrowsError(try TrustedModels.validate(String(repeating: "0", count: 64)))
    XCTAssertThrowsError(try TrustedModels.validate("ddac71cf"))
    print("PASS: exact old/new model compatibility; arbitrary hashes rejected.")
}
