// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DepressionEEGRealCase",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "RealCaseApp", targets: ["RealCaseApp"])
    ],
    targets: [
        .executableTarget(
            name: "RealCaseApp",
            path: "Sources/RealCaseApp"
        )
    ]
)
