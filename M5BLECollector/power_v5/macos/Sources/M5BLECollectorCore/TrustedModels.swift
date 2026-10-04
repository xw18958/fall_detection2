import Foundation

public enum TrustedModels {
    // Exact previously deployed V2 and verified V4 PTQ/QAT identities.
    public static let hashes: Set<String> = [
        "1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530",
        "ddac71cffd5e10f85d9e9122fe5334a3dbabf3717da41b598231ef4a69a720bc",
        "7a649721e02ab88e85594dd67efe415a0d5a777d9215df03d903dd0de748cf57",
        // Exact installed experimental V5 identity; hash acceptance does not
        // imply commercial qualification or allow arbitrary replacement models.
        "4eb70382f3306a4c8580e5d08342032edae43d3bd4e07cb4a3a9090e85dfa367"
    ]
    public static func validate(_ hash: String) throws {
        guard hashes.contains(hash) else {
            throw ProtocolError.invalid("Device model differs from the trusted detector models")
        }
    }
}
