// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AlifeControl",
    platforms: [.macOS(.v14)],
    targets: [
        // Pure logic with no AppKit or Metal: levers and command lines, the director's settings, the preset
        // reader, the `alife stream` protocol decoder and encoder, the camera. Its declarations are `package`
        // access, visible to the app and the tests of this package only.
        .target(
            name: "AlifeCore",
            path: "Sources/AlifeCore"
        ),
        // The app: the run controller, the 3D viewer's stream process, Metal renderer and window.
        .executableTarget(
            name: "AlifeControl",
            dependencies: ["AlifeCore"],
            path: "Sources/AlifeControl"
        ),
        .testTarget(
            name: "AlifeCoreTests",
            dependencies: ["AlifeCore"],
            path: "Tests/AlifeCoreTests"
        ),
        // Tests of the app target (`@testable import AlifeControl`): the stream reader and controller, the viewer's
        // pure parts, and the opt-in GPU rendering tests (ALIFE_GPU_TESTS=1).
        .testTarget(
            name: "AlifeControlTests",
            dependencies: ["AlifeControl", "AlifeCore"],
            path: "Tests/AlifeControlTests"
        ),
    ]
)
