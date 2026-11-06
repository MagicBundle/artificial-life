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
        // The one place a slower test host is allowed for (`ALIFE_TEST_TIME_SCALE`, DESIGN §22.0, I91): not a test target, so both test
        // targets can import it.
        .target(
            name: "AlifeTestSupport",
            path: "Tests/AlifeTestSupport"
        ),
        .testTarget(
            name: "AlifeCoreTests",
            dependencies: ["AlifeCore", "AlifeTestSupport"],
            path: "Tests/AlifeCoreTests"
        ),
        // Tests of the app target (`@testable import AlifeControl`): the stream reader and controller, the viewer's
        // pure parts, and the opt-in GPU rendering tests (ALIFE_GPU_TESTS=1).
        .testTarget(
            name: "AlifeControlTests",
            dependencies: ["AlifeControl", "AlifeCore", "AlifeTestSupport"],
            path: "Tests/AlifeControlTests"
        ),
    ]
)
