import Foundation
import Testing

@testable import AlifeControl

/// Which repository folder the controller starts with, what it says when the folder is gone, and that "Set up environment"
/// and Start launch nothing from a missing folder. Real temporary folders; no process is
/// started in any of them.
@MainActor
@Suite struct RunControllerRepoTests {
    /// The stored path is passed in and the saved one recorded here, so the app's own preferences are never touched.
    private final class Saved {
        var paths: [String] = []
    }

    /// A folder under `scratch`, with a `.venv/bin/alife` stand-in when `engine`.
    private func folder(_ name: String, in scratch: Scratch, engine: Bool) throws -> String {
        let dir = scratch.url.appendingPathComponent(name)
        let bin = dir.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: engine ? bin : dir, withIntermediateDirectories: true)
        if engine {
            let alife = bin.appendingPathComponent("alife")
            try "#!/bin/bash\nexit 0\n".write(to: alife, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        }
        return dir.path
    }

    private func make(stored: String?, default path: String, saved: Saved = Saved()) -> RunController {
        RunController(storedPath: stored, savePath: { saved.paths.append($0) }, defaultPath: path, isApp: false)
    }

    @Test func aMissingStoredFolderFallsBackToTheDefaultAndLogsIt() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let fallback = try folder("default", in: scratch, engine: true)
        let gone = scratch.url.appendingPathComponent("gone-worktree").path
        let saved = Saved()
        let controller = make(stored: gone, default: fallback, saved: saved)
        #expect(controller.repoPath == fallback)
        #expect(controller.repoReady)
        #expect(controller.repoProblem == nil)
        #expect(controller.logLines.map(\.text) == ["Stored repository folder not found: \(gone). Using \(fallback)."])
        #expect(saved.paths.isEmpty)  // the replacement is not written until the user chooses a folder
    }

    @Test func aMissingStoredFolderWithoutAUsableDefaultSaysSoAndLaunchesNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let noEngine = try folder("default", in: scratch, engine: false)  // exists, but no .venv/bin/alife
        let gone = scratch.url.appendingPathComponent("gone-worktree").path
        for fallback in [noEngine, scratch.url.appendingPathComponent("no-default").path] {
            let controller = make(stored: gone, default: fallback)
            let message = "Folder not found: \(gone). Choose the repository folder."
            #expect(controller.repoPath == gone)
            #expect(!controller.repoReady)
            #expect(!controller.repoFolderExists)
            #expect(controller.repoProblem == message)
            #expect(controller.logLines.isEmpty)

            controller.setupEnvironment()
            #expect(controller.job == nil)
            #expect(controller.logLines.map(\.text) == [message])
            #expect(controller.logLines.allSatisfy { $0.isError })

            controller.start()
            #expect(controller.job == nil)
            #expect(controller.logLines.map(\.text) == [message, message])
        }
    }

    @Test func aStoredFolderThatExistsIsKept() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let fallback = try folder("default", in: scratch, engine: true)
        // Without the engine (the status line then blames the venv, not the folder) and with it.
        for engine in [false, true] {
            let stored = try folder("stored-\(engine)", in: scratch, engine: engine)
                let saved = Saved()
            let controller = make(stored: stored, default: fallback, saved: saved)
            #expect(controller.repoPath == stored)
            #expect(controller.repoFolderExists)
            #expect(controller.repoReady == engine)
            #expect(controller.repoProblem == nil)
            #expect(controller.logLines.isEmpty)
            #expect(saved.paths.isEmpty)
        }
    }

    @Test func aFolderChosenByTheUserIsSaved() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let chosen = try folder("chosen", in: scratch, engine: true)
        let saved = Saved()
        let controller = make(stored: nil, default: scratch.url.appendingPathComponent("none").path, saved: saved)
        controller.repoPath = chosen
        #expect(saved.paths == [chosen])
        #expect(controller.repoReady)
    }

    @Test func noStoredFolderUsesTheDefaultWithoutANote() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let fallback = try folder("default", in: scratch, engine: true)
        let controller = make(stored: nil, default: fallback)
        #expect(controller.repoPath == fallback)
        #expect(controller.repoReady)
        #expect(controller.logLines.isEmpty)
    }

    @Test func aFolderThatDisappearsLaterIsReported() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let stored = try folder("stored", in: scratch, engine: true)
        let controller = make(stored: stored, default: scratch.url.appendingPathComponent("none").path)
        #expect(controller.repoReady)
        try FileManager.default.removeItem(atPath: stored)
        controller.refreshRepo()
        #expect(!controller.repoReady)
        #expect(controller.repoProblem == "Folder not found: \(stored). Choose the repository folder.")
    }

    @Test func setUpEnvironmentRemovesNothing() {
        // Exact argument list: `--inexact` keeps what is installed; a plain `uv sync --extra ...` would uninstall the
        // dev group and the other extras.
        #expect(RunController.setupArguments == ["sync", "--inexact", "--extra", "viewer", "--extra", "video"])
    }
}
