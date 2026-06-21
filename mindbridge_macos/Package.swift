// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MindBridgeDesktop",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "MindBridgeApp", targets: ["MindBridgeApp"])
    ],
    targets: [
        .executableTarget(
            name: "MindBridgeApp",
            path: "Sources/MindBridgeApp"
        )
    ]
)
