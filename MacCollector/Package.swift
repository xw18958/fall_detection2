// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AirPodsMotionCollector",
    platforms: [.macOS(.v14)],
    products: [
        .executable(
            name: "AirPodsMotionCollector",
            targets: ["AirPodsMotionCollector"]
        )
    ],
    targets: [
        .executableTarget(name: "AirPodsMotionCollector")
    ]
)
