// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AlifeControl",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "AlifeControl",
            path: "Sources/AlifeControl"
        )
    ]
)
