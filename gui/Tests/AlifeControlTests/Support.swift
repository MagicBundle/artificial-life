import Foundation

@testable import AlifeControl

/// Where the repository is, found from this file's own path: `gui/Tests/AlifeControlTests/Support.swift`.
enum Repo {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static let streamData = root.appendingPathComponent("tests/data", isDirectory: true)

    static func streamFixture(_ kind: String) throws -> [UInt8] {
        [UInt8](try Data(contentsOf: streamData.appendingPathComponent("stream_tiny_\(kind).bin")))
    }

    static func streamFixtureURL(_ kind: String) -> URL {
        streamData.appendingPathComponent("stream_tiny_\(kind).bin")
    }

    /// A committed play stream (`play_tiny3d`, `play_tiny2d`, `play_tiny3d_n70`).
    static func playFixtureURL(_ kind: String) -> URL {
        streamData.appendingPathComponent("play_tiny\(kind).bin")
    }
}

/// A fresh empty directory under the system temporary directory, removed by `cleanup()`.
struct Scratch {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("alifecontrol-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    func cleanup() { try? FileManager.default.removeItem(at: url) }

    /// An executable shell script `name` with `body` (a stand-in for `alife`).
    func script(_ name: String, _ body: String) throws -> URL {
        let file = url.appendingPathComponent(name)
        try ("#!/bin/bash\n" + body + "\n").write(to: file, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        return file
    }
}

/// Polls `condition` on the main actor until it holds or `timeout` passes. True if it held.
@MainActor
func eventually(timeout: Duration = .seconds(20), _ condition: @MainActor () -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        if ContinuousClock.now > deadline { return false }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return true
}
