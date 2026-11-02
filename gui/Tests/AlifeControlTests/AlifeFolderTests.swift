import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// The alife folder, as the app finds it: the one the old apps stored in the domain they shared, or in their own, and none when nothing
/// was ever chosen. In-memory preferences domains: the owner's preferences are never touched.
@MainActor
@Suite struct AlifeFolderTests {
    private func folder(_ name: String, in scratch: Scratch) throws -> String {
        let bin = scratch.url.appendingPathComponent(name).appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let alife = bin.appendingPathComponent("alife")
        try "#!/bin/bash\nexit 0\n".write(to: alife, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        return scratch.url.appendingPathComponent(name).path
    }

    private func controller(on store: RepoPathStore, default path: String) -> RunController {
        RunController(storedPath: store.load(), savePath: { store.save($0) }, candidates: [path], isApp: false)
    }

    private func app(own: MemoryDomain, storm: MemoryDomain = MemoryDomain(), lab: MemoryDomain = MemoryDomain()) -> RepoPathStore {
        RepoPathStore.forApp(info: "alife", environment: [:], standard: own, suite: {
            $0 == RepoPathStore.stormDomain ? storm : ($0 == RepoPathStore.labDomain ? lab : nil)
        })
    }

    @Test func aFolderTheOldAppsChoseIsTheAppsWithoutBeingToldAndItsOwnChoiceIsTheirsToo() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let first = try folder("first", in: scratch), second = try folder("second", in: scratch)
        let none = scratch.url.appendingPathComponent("none").path
        let own = MemoryDomain(), lab = MemoryDomain()

        // The Lab, as installed before, has the folder in its own preferences: the app finds it on its first start.
        lab.store(first, forKey: RepoPathStore.key)
        let fresh = controller(on: app(own: own, lab: lab), default: none)
        #expect(fresh.repoPath == first && fresh.repoReady)
        #expect(own.string(forKey: RepoPathStore.key) == first, "copied into the app's domain")
        #expect(lab.string(forKey: RepoPathStore.key) == first, "and left where it was")

        // Choosing another folder is stored in the shared domain, which is the app's own: an old app reads it at its next start.
        fresh.repoPath = second
        let oldApp = controller(on: RepoPathStore(shared: own, own: lab), default: none)
        #expect(oldApp.repoPath == second && oldApp.repoReady)
    }

    @Test func withNoFolderAnywhereTheAppSaysSoInsteadOfGuessing() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let missing = scratch.url.appendingPathComponent("no-such-default").path
        let fresh = controller(on: app(own: MemoryDomain()), default: missing)
        // No folder is named that nobody chose (docs/APP.md F3): the path is empty and the status says that none is chosen.
        #expect(fresh.repoPath.isEmpty)
        #expect(!fresh.repoReady)
        #expect(fresh.repoProblem == RepoFolder.missingMessage(""))
        #expect(fresh.repoProblem?.hasPrefix("No alife folder is chosen yet.") == true)
    }

    @Test func theDelegateQuitsWhenTheLastWindowClosesUnlessARunIsKept() {
        let delegate = AppDelegate(run: .shared, play: .shared, quitPanel: .offscreen())
        #expect(delegate.applicationShouldTerminateAfterLastWindowClosed(NSApplication.shared))
    }
}
