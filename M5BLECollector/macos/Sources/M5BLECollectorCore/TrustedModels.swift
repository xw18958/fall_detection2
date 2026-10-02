import Foundation

public enum TrustedModels {
    // Exact previously deployed V2 and verified V4 PTQ identities.
    public static let hashes: Set<String> = [
        "1553dde844bf34928d360cc5f23e06f353e1c78e7aac6b2271cbc36410920530",
        "ddac71cffd5e10f85d9e9122fe5334a3dbabf3717da41b598231ef4a69a720bc"
    ]
    public static func validate(_ hash: String) throws {
        guard hashes.contains(hash) else {
            throw ProtocolError.invalid("Device model differs from the trusted detector models")
        }
    }
}
