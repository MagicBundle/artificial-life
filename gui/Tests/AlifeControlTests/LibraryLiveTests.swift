import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The library object while the world changes under it: the watch on `runs/`, the follow-up scans, the folder that changes during a scan,
/// the menu's Reveal, the Lab's refusal and the sidebar's anchor (docs/APP.md §5.2, §5.3).
@MainActor
@Suite("The library while things change", .serialized) struct LibraryLiveTests {
    private func runs(_ scratch: Scratch, _ name: String = "runs") throws -> URL {
        let runs = scratch.url.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        return runs
    }

    /// A library that probes every 0.15 s (the app's 4 s would make these tests slow) and whose timer does not wait for the app to be active.
    private func quick(_ runs: @escaping () -> URL?, interval: Duration = .milliseconds(150), events: Bool = true,
                       lockHeld: @escaping @Sendable (URL) -> Bool = { _ in false }) -> WorldLibrary {
        let library = WorldLibrary(runsDirectory: runs, scanner: LibraryScanner(settle: 0), observesApp: false, probeInterval: interval,
                                   requiresActiveApp: false, lockHeld: lockHeld)
        library.watchEventsEnabled = events
        library.startTimer()
        return library
    }

    // MARK: M2: the watch follows runs/

    /// `runs/` moved away and a new one made: the event on the old folder re-opens the watch (the timer is a minute away, so only the event
    /// can do it), and a world made in the new folder is then seen by the watch alone.
    @Test func aRunsFolderThatIsReplacedIsWatchedAgainByTheEvent() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        var runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "before")
        let library = quick({ runs }, interval: .seconds(60))
        defer { library.stopTimer() }
        library.refresh()
        #expect(await eventually { library.state.rows.map(\.name) == ["before"] })
        try await Task.sleep(for: .seconds(1.5))  // (the scan that follows the watch being opened has been made: only the watch can see what comes)
        try FileManager.default.moveItem(at: runs, to: scratch.url.appendingPathComponent("runs.old"))
        runs = try self.runs(scratch)
        #expect(await eventually { library.state.rows.isEmpty }, "the old folder's worlds are gone from the list")
        try LibraryTestFolders.research(runs, "after")
        #expect(await eventually(timeout: .seconds(40)) { library.state.rows.map(\.name) == ["after"] }, "a world made in the new folder is seen")
    }

    /// The same when the event is missed: the probe compares the folder's identity with the one the watch was opened on.
    @Test func aRunsFolderThatIsReplacedIsFoundByTheProbeWhenTheEventIsMissed() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        var runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "before")
        let library = quick({ runs }, events: false)
        defer { library.stopTimer() }
        library.refresh()
        #expect(await eventually { library.state.rows.map(\.name) == ["before"] })
        try await Task.sleep(for: .seconds(1.5))  // (the scan that follows the watch being opened has been made: only the watch can see what comes)
        try FileManager.default.moveItem(at: runs, to: scratch.url.appendingPathComponent("runs.old"))
        runs = try self.runs(scratch)
        try LibraryTestFolders.research(runs, "after")
        #expect(await eventually(timeout: .seconds(40)) { library.state.rows.map(\.name) == ["after"] })
    }

    @Test func aRunsFolderThatIsRemovedAndComesBackIsSeen() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "first")
        let library = quick { runs }
        defer { library.stopTimer() }
        library.refresh()
        #expect(await eventually { library.state.rows.count == 1 })
        try FileManager.default.removeItem(at: runs)
        #expect(await eventually { library.state.rows.isEmpty })
        _ = try self.runs(scratch)
        try LibraryTestFolders.research(runs, "second")
        #expect(await eventually(timeout: .seconds(40)) { library.state.rows.map(\.name) == ["second"] })
    }

    @Test func aRunsFolderThatDoesNotExistYetIsWatchedWhenItAppears() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        let library = quick { runs }
        defer { library.stopTimer() }
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.state.rows.isEmpty)
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        try LibraryTestFolders.research(runs, "late")
        #expect(await eventually(timeout: .seconds(40)) { library.state.rows.map(\.name) == ["late"] })
        // and from then on a new world is seen by the watch alone (the timer only probes)
        try LibraryTestFolders.research(runs, "later")
        #expect(await eventually(timeout: .seconds(40)) { library.state.rows.count == 2 })
    }

    // MARK: M4: the end of Saving

    @Test func openingAWorldAndTheEndOfItsSavingEachScanAgain() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.played(runs, "w")
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        var scans = library.scans
        library.update(live: LiveFacts(openWorld: "w"))
        #expect(await eventually { library.scans > scans }, "a world opened")
        scans = library.scans
        library.update(live: LiveFacts(openWorld: "w", openPhase: .paused))
        try await Task.sleep(for: .milliseconds(300))
        #expect(library.scans == scans, "pausing is not a reason to look at the disk")
        library.update(live: LiveFacts(saving: ["w"]))
        #expect(await eventually { library.scans > scans }, "closed: saving")
        scans = library.scans
        library.update(live: LiveFacts())
        #expect(await eventually { library.scans > scans }, "the saving is over: its date, its step and its lock are read again")
    }

    // MARK: Follow-up scans

    final class Gate: @unchecked Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var entered = 0
        func enter() { lock.withLock { entered += 1 } }
        var count: Int { lock.withLock { entered } }
    }

    @Test func aRequestDuringAScanIsAnsweredByOneMoreScan() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.played(runs, "p")
        let gate = Gate()
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false) { _ in
            gate.enter()
            _ = gate.semaphore.wait(timeout: .now() + 2)
            return false
        }
        library.refresh()
        #expect(await eventually { gate.count == 1 }, "the first scan is inside its probe")
        try LibraryTestFolders.research(runs, "late")
        library.refresh()  // asked while it runs
        gate.semaphore.signal()
        gate.semaphore.signal()
        #expect(await eventually { library.state.rows.map(\.name).sorted() == ["late", "p"] }, "the follow-up scan saw what the first could not")
        #expect(library.scans == 2)
    }

    final class Box: @unchecked Sendable {
        var url: URL?
    }

    @Test func aFolderChosenWhileAScanRunsIsNotShownAsTheOldOnesLibrary() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let a = try runs(scratch, "runs_a")
        let b = try runs(scratch, "runs_b")
        try LibraryTestFolders.played(a, "from_a")
        try LibraryTestFolders.research(b, "from_b")
        let gate = Gate()
        let box = Box()
        box.url = a
        let library = WorldLibrary(runsDirectory: { box.url }, observesApp: false, watchesFolder: false) { url in
            if url.deletingLastPathComponent().lastPathComponent == "runs_a" {
                gate.enter()
                _ = gate.semaphore.wait(timeout: .now() + 2)
            }
            return false
        }
        library.refresh()
        #expect(await eventually { gate.count == 1 })
        box.url = b  // the alife folder was changed in Settings
        gate.semaphore.signal()
        #expect(await eventually { library.state.rows.map(\.name) == ["from_b"] }, "A's scan is dropped and B's is made")
        #expect(library.state.rows.contains { $0.name == "from_a" } == false)
    }

    // MARK: M6: Reveal

    @Test func revealFollowsThePageInViewAndAGoneWorldDoesNotRevealSomethingElse() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryTestFolders.research(runs, "r")
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        var shown: [URL] = []
        library.revealURL = { shown.append($0) }
        library.select("r", scroll: false)
        var fallbacks = 0
        await library.revealFromMenu(page: .lab) { fallbacks += 1 }
        #expect(fallbacks == 1 && shown.isEmpty, "on the Lab's page the Lab's run is the one: the world chosen on Worlds is not")
        await library.revealFromMenu(page: .worlds) { fallbacks += 1 }
        #expect(fallbacks == 1 && shown.map(\.lastPathComponent) == ["r"])
        try FileManager.default.removeItem(at: dir)
        await library.revealFromMenu(page: .worlds) { fallbacks += 1 }
        #expect(fallbacks == 1 && shown.count == 1, "gone: told, and no second Finder window for another folder")
        #expect(library.vanished == "r" && library.selected == nil)
        await library.revealFromMenu(page: .worlds) { fallbacks += 1 }
        #expect(fallbacks == 2, "nothing is chosen any more: the fallback is the alife folder")
    }

    // MARK: H1: the Lab

    @Test func theLabRefusesToContinueAWorldThatIsPlayedOrInUse() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.played(runs, "played")
        try LibraryTestFolders.research(runs, "ok")
        try LibraryTestFolders.research(runs, "busy")
        try LibraryTestFolders.research(runs, "mine")
        try LibraryTestFolders.unfinished(runs, "build")
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.reasonNotToResume("ok") == nil)
        #expect(library.reasonNotToResume("played")?.contains("is a played world") == true)
        #expect(library.reasonNotToResume("build")?.contains("unfinished build") == true)
        #expect(library.reasonNotToResume("nothing") == nil, "the Lab's own check says a missing run does not exist")
        library.update(live: LiveFacts(openWorld: "mine"))
        #expect(await eventually { library.reasonNotToResume("mine") != nil })
        #expect(library.reasonNotToResume("mine") == "'runs/mine' is open in this window.")
        library.update(live: LiveFacts(saving: ["mine"]))
        #expect(await eventually { library.reasonNotToResume("mine") == "'runs/mine' is still being saved." })
    }

    @Test func aResearchRunAnotherProcessWritesIsRefused() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryTestFolders.research(runs, "busy", status: "running")
        try LibraryTestFolders.write("{}", "metrics.jsonl", in: dir)
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.reasonNotToResume("busy") == "'runs/busy' is in use by another app or process. Wait until it stops.")
    }

    @Test func theLabsStartIsGreyWithThatReason() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = scratch.url.appendingPathComponent("folder")
        let bin = root.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let alife = bin.appendingPathComponent("alife")
        try "#!/bin/bash\nexit 0\n".write(to: alife, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        try LibraryTestFolders.played(root.appendingPathComponent("runs"), "played")
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        controller.settings.mode = .resume
        controller.settings.sourceRun = "played"
        #expect(controller.canStart, "the two old apps keep today's checks")
        controller.resumeRefusal = { _ in "refused" }
        #expect(!controller.canStart && controller.startBlockedReason == "refused")
        controller.settings.mode = .fork
        controller.settings.sourceRun = "played"
        #expect(controller.validationErrors.contains("refused") == false, "only continuing a directory writes it")
    }

    // MARK: L7, M7

    @Test func aWorldThatIsNoLongerAWorldButIsStillThereIsNotReportedGone() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryTestFolders.research(runs, "pruned")
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        library.select("pruned", scroll: false)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("checkpoints"))
        library.refresh()
        #expect(await eventually { library.selected == nil })
        #expect(library.vanished == nil, "the folder is there: it is only not a world any more")
    }

    @Test func theRowChosenLastStaysChosenWhileThePageShowsWhatItOpened() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "r")
        try LibraryTestFolders.unfinished(runs, "u")
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        let suite = "alife-anchor-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let shell = ShellModel(session: WorldSession(play: PlayController(), defaults: defaults), defaults: defaults)
        func current() -> SidebarItem { SidebarItem.current(page: shell.page, library: library) }
        SidebarItem.unfinished.choose(shell: shell, library: library)
        #expect(current() == .unfinished, "arrow keys go on from the row that was chosen, not from the Worlds row")
        SidebarItem.allWorlds.choose(shell: shell, library: library)
        #expect(current() == .allWorlds)
        library.filter = .played
        #expect(current() == .page(.worlds), "the page was changed by hand: the row no longer says what it shows")
        SidebarItem.world("r").choose(shell: shell, library: library)
        #expect(current() == .world("r"))
        shell.select(.activity)
        #expect(current() == .page(.activity))
        SidebarItem.page(.worlds).choose(shell: shell, library: library)
        #expect(library.sidebarAnchor == nil && current() == .page(.worlds))
    }

    // MARK: M5: words

    @Test func eachKindOfEntryThatIsLeftOutHasItsOwnSentence() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryTestFolders.research(runs, "here")
        func link(_ name: String, _ target: String) throws {
            try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent(name).path, withDestinationPath: target)
        }
        try link("d1", "/Volumes/Gone/runs/d1")
        try link("broken", scratch.url.appendingPathComponent("nope").path)
        try link("la", runs.appendingPathComponent("lb").path)
        try link("lb", runs.appendingPathComponent("la").path)
        let library = WorldLibrary(runsDirectory: { runs }, observesApp: false, watchesFolder: false)
        library.refresh()
        #expect(await eventually { library.hasScanned })
        #expect(library.offlineSentence == "1 world is on the drive \u{201C}Gone\u{201D}, which is not connected. Connect it to see it here.")
        let sentences = library.problemSentences
        #expect(sentences.contains("1 link in runs/ leads to a place that is not there."))
        #expect(sentences.contains("2 links in runs/ lead back to itself, so it cannot be read."))
        #expect(!sentences.contains { $0.contains("drive") }, "a broken link and a loop are not a drive that is away")
    }
}
