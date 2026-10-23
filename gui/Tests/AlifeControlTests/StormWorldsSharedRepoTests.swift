import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// The Lab app and Storm Worlds agree on the repository folder, without either being set twice, and the app delegate
/// follows the launch plan. In-memory preferences domains: the owner's preferences are never touched.
@MainActor
@Suite struct StormWorldsSharedRepoTests {
    private func folder(_ name: String, in scratch: Scratch) throws -> String {
        let bin = scratch.url.appendingPathComponent(name).appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let alife = bin.appendingPathComponent("alife")
        try "#!/bin/bash\nexit 0\n".write(to: alife, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        return scratch.url.appendingPathComponent(name).path
    }

    private func controller(on store: RepoPathStore, default path: String) -> RunController {
        RunController(storedPath: store.load(), savePath: { store.save($0) }, defaultPath: path, isApp: false)
    }

    @Test func aFolderSetInTheLabIsTheGamesAndTheOtherWayRound() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let shared = MemoryDomain(), labOwn = MemoryDomain(), stormOwn = MemoryDomain()
        let first = try folder("first", in: scratch), second = try folder("second", in: scratch)
        let none = scratch.url.appendingPathComponent("none").path

        // The Lab, as installed before this change, has the folder in its own preferences.
        labOwn.store(first, forKey: RepoPathStore.key)
        let lab = controller(on: RepoPathStore(shared: shared, own: labOwn), default: none)
        #expect(lab.repoPath == first)
        // Storm Worlds, started for the first time, finds it without being told.
        let storm = controller(on: RepoPathStore(shared: shared, own: stormOwn, legacy: [labOwn]), default: none)
        #expect(storm.repoPath == first)
        #expect(storm.repoReady)

        // Choosing another folder in the game is the Lab's at its next start.
        storm.repoPath = second
        let labAgain = controller(on: RepoPathStore(shared: shared, own: labOwn), default: none)
        #expect(labAgain.repoPath == second)
        #expect(labAgain.repoReady)
    }

    @Test func stormWorldsWithNoFolderAnywhereSaysSoInsteadOfGuessing() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let store = RepoPathStore(shared: MemoryDomain(), own: MemoryDomain())
        let missing = scratch.url.appendingPathComponent("no-such-default").path
        let storm = controller(on: store, default: missing)
        #expect(storm.repoPath == missing)
        #expect(!storm.repoReady)
        #expect(storm.repoProblem == RepoFolder.missingMessage(missing))
    }

    @Test func theDelegateQuitsWhenTheLastWindowClosesInTheLabAndInStormWorlds() {
        // `LaunchMode.current()` is the Lab in a test run; the plan says the same of Storm Worlds (LaunchModeTests).
        #expect(AppDelegate().applicationShouldTerminateAfterLastWindowClosed(NSApplication.shared))
    }
}
