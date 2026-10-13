import Foundation

/// Where the repository is, found from this file's own path: `gui/Tests/AlifeCoreTests/TestSupport.swift`.
enum Repo {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static let presets = root.appendingPathComponent("src/alife/presets", isDirectory: true)
    static let streamData = root.appendingPathComponent("tests/data", isDirectory: true)

    static func preset(_ name: String) throws -> String {
        try String(contentsOf: presets.appendingPathComponent("\(name).toml"), encoding: .utf8)
    }
}

/// A fresh empty directory under the system temporary directory, removed by `cleanup()`.
struct ScratchDirectory {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("alifecore-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    func cleanup() { try? FileManager.default.removeItem(at: url) }
}
