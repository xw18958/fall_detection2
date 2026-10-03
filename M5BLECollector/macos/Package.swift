// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "M5BLECollector",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "M5BLECollector", targets: ["M5BLECollector"]),
        .executable(name: "M5CollectorApp", targets: ["M5CollectorApp"])
    ],
    targets: [
        .target(name: "M5BLECollectorCore"),
        .executableTarget(name: "M5BLECollector", dependencies: ["M5BLECollectorCore"]),
        .executableTarget(name: "M5CollectorApp", dependencies: ["M5BLECollectorCore"]),
        .executableTarget(name: "M5BLECollectorSmokeTests", dependencies: ["M5BLECollectorCore"], path: "Tests/M5BLECollectorCoreTests")
    ]
)
