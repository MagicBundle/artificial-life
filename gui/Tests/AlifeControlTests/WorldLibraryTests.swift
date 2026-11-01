import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The library object: the scan runs off the main thread, is asked for only when something may have changed, copes with a world that
/// vanishes and tells the page what it needs (docs/APP.md §5.3).
@MainActor
@Suite("The world library", .serialized) struct WorldLibraryTests {
    private func runs(_ scratch: Scratch) throws -> URL {
        let runs = scratch.url.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        return runs
    }

    private func library(_ runs: URL?, lockHeld: @escaping @Sendable (URL) -> Bool = { _ in false }) -> WorldLibrary {
        WorldLibrary(runsDirectory: { runs }, observesApp: false, lockHeld: lockHeld)
    }

    @Test func theFirstScanIsNotAnEmptyLibrary() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "r")
        let library = library(runs)
        #expect(!library.hasScanned && library.state.rows.isEmpty, "until the first scan finishes the page says it is looking")
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.state.rows.map(\.name) == ["r"])
    }

    @Test func theScanAndTheLockProbesNeverRunOnTheMainThread() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.played(runs, "w")
        final class Seen: @unchecked Sendable {
            private let lock = NSLock()
            private var threads: [Bool] = []
            func add(_ onMain: Bool) { lock.withLock { threads.append(onMain) } }
            var all: [Bool] { lock.withLock { threads } }
        }
        let seen = Seen()
        let library = library(runs) { _ in
            seen.add(Thread.isMainThread)
            return false
        }
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(!seen.all.isEmpty && !seen.all.contains(true))
    }

    @Test func askingWhileAScanRunsMakesOneMoreScanNotMany() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.many(runs, count: 30)
        let library = library(runs)
        for _ in 0..<10 { library.refresh() }
        #expect(await eventually { library.hasScanned && !library.isScanning && library.scans >= 1 })
        try await Task.sleep(for: .milliseconds(200))
        #expect(library.scans <= 2, "ten requests in a row are one scan and at most one more: \(library.scans)")
    }

    @Test func aNewWorldInTheFolderIsSeenWithoutAsking() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "first")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        try LibraryTestFolders.research(runs, "second")
        #expect(await eventually { library.state.rows.count == 2 }, "the watch on runs/ finds it")
    }

    @Test func theChosenWorldThatVanishesIsSaidAndUnchosen() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryTestFolders.research(runs, "goner")
        try LibraryTestFolders.research(runs, "stays")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        library.select("goner")
        #expect(library.selected == "goner" && library.stillThere("goner"))
        try FileManager.default.removeItem(at: dir)
        #expect(!library.stillThere("goner"))
        library.noteGone("goner")
        #expect(library.vanished == "goner" && library.selected == nil)
        #expect(await eventually { library.state.rows.map(\.name) == ["stays"] })
        // the same thing found by a scan alone
        library.select("stays")
        try FileManager.default.removeItem(at: runs.appendingPathComponent("stays"))
        library.refresh()
        #expect(await eventually { library.vanished == "stays" })
    }

    @Test func aWorldInUseByAnotherAppIsSaidSoAndIsNotCalledMine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.played(runs, "elsewhere")
        try LibraryTestFolders.played(runs, "mine")
        let library = library(runs) { _ in true }
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.state.row(named: "elsewhere")?.state == .inUseElsewhere)
        // the app's own world holds the lock as well: it is Live
        library.update(live: LiveFacts(openWorld: "mine", openPhase: .live))
        #expect(library.state.row(named: "mine")?.state == .live)
        #expect(library.state.row(named: "elsewhere")?.state == .inUseElsewhere)
    }

    @Test func choosingAWorldClearsTheFilterThatWouldHideIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "r")
        try LibraryTestFolders.played(runs, "p")
        try LibraryTestFolders.unfinished(runs, "u")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        library.filter = .played
        library.query = "p"
        library.select("r")
        #expect(library.filter == .all && library.query == "" && library.selected == "r")
        library.select("u")
        #expect(library.filter == .unfinished && library.selected == "u")
    }

    @Test func noFolderIsAnEmptyLibraryThatHasBeenLookedAt() async {
        let library = library(nil)
        library.refresh()
        #expect(library.hasScanned && library.state.rows.isEmpty)
    }

    @Test func linksToADriveThatIsNotConnectedAreCountedInWords() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "here")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.offlineSentence == nil)
        for name in ["a", "b"] {
            try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent(name).path, withDestinationPath: "/Volumes/Gone/runs/\(name)")
        }
        library.refresh()
        #expect(await eventually { library.offlineSentence != nil })
        #expect(library.offlineSentence == "2 worlds are on the drive \u{201C}Gone\u{201D}, which is not connected. Connect it to see them here.")
        #expect(library.state.rows.map(\.name) == ["here"])
    }

    @Test func foldersThatAreNotWorldsAreSaidOnceNotListed() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.write("{}", "metrics.jsonl", in: runs.appendingPathComponent("logs_only"))
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.othersSentence == "1 folder in runs/ has no save point, so it is not a world and is not listed.")
    }

    @Test func theRowsOfTheSidebarOpenThePageOnTheWorldOrTheList() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "r")
        try LibraryTestFolders.played(runs, "p")
        try LibraryTestFolders.unfinished(runs, "u")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        let suite = "alife-sidebar-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let play = PlayController()
        let shell = ShellModel(session: WorldSession(play: play, defaults: defaults), defaults: defaults)
        shell.select(.lab)
        library.query = "zzz"
        SidebarItem.world("p").choose(shell: shell, library: library)
        #expect(shell.page == .worlds && library.selected == "p" && library.query == "", "a search that would hide the world is cleared")
        SidebarItem.page(.worlds).choose(shell: shell, library: library)
        #expect(library.selected == nil && shell.page == .worlds)
        SidebarItem.unfinished.choose(shell: shell, library: library)
        #expect(library.filter == .unfinished && shell.page == .worlds && library.state.unfinishedRows(query: library.query).map(\.name) == ["u"])
        shell.select(.activity)
        SidebarItem.allWorlds.choose(shell: shell, library: library)
        #expect(library.filter == .all && shell.page == .worlds)
        SidebarItem.page(.activity).choose(shell: shell, library: library)
        #expect(shell.page == .activity)
    }

    @Test func onlyAChoiceFromTheSidebarScrollsThePage() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "r")
        let library = library(runs)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        library.select("r", scroll: false)
        #expect(library.selected == "r" && library.reveal == nil, "a click on a card in view moves nothing")
        library.select("r")
        #expect(library.reveal?.name == "r")
        let first = library.reveal
        library.select("r")
        #expect(library.reveal != first, "choosing the same world again scrolls again")
    }
}
