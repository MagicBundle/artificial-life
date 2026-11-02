import Foundation
import Testing

@testable import AlifeCore

/// The scan behind the world library: what it reads, what it remembers, what it leaves alone (docs/APP.md §5.3).
final class Asked: @unchecked Sendable {
    private let lock = NSLock()
    private var names: [String] = []
    func add(_ name: String) { lock.lock(); names.append(name); lock.unlock() }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return names.sorted() }
}

@Suite struct LibraryScannerTests {
    private func runs(_ scratch: ScratchDirectory) throws -> URL {
        let runs = scratch.url.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        return runs
    }

    @Test func aDirectoryIsReadOnceAndThenTakenFromTheCache() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.many(runs, count: 40)
        let scanner = LibraryScanner(settle: 0)
        let first = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.listed == 40 && scanner.lastStats.read == 40 && scanner.lastStats.reused == 0)
        let second = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 0 && scanner.lastStats.reused == 40, "nothing changed: nothing is read again")
        #expect(first == second)
    }

    @Test func aChangeInOneWorldRereadsThatWorldOnly() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "r1", steps: [100])
        try LibraryFixture.research(runs, "r2", steps: [100])
        let scanner = LibraryScanner(settle: 0)
        _ = scanner.scan(runsDirectory: runs)
        // a new save point (the run directory itself is not touched, its checkpoints/ is)
        try LibraryFixture.write("x", "checkpoints/step_0000000200.safetensors", in: dir)
        let after = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 1 && scanner.lastStats.reused == 1)
        #expect(after.records.first { $0.name == "r1" }?.steps == [100, 200])
        // the status changes (meta.json is replaced)
        try LibraryFixture.write(#"{"status": "extinct", "start_step": 0}"#, "meta.json", in: dir)
        let again = scanner.scan(runsDirectory: runs)
        #expect(again.records.first { $0.name == "r1" }?.status == "extinct")
    }

    @Test func aWorldThatVanishesLeavesTheListAndTheCache() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let gone = try LibraryFixture.research(runs, "gone")
        try LibraryFixture.research(runs, "stays")
        let scanner = LibraryScanner(settle: 0)
        #expect(scanner.scan(runsDirectory: runs).records.map(\.name) == ["gone", "stays"])
        try FileManager.default.removeItem(at: gone)
        let after = scanner.scan(runsDirectory: runs)
        #expect(after.records.map(\.name) == ["stays"])
        // made again under the same name, it is read afresh and not remembered as the old one
        try LibraryFixture.research(runs, "gone", steps: [5])
        #expect(scanner.scan(runsDirectory: runs).records.first { $0.name == "gone" }?.steps == [5])
    }

    @Test func aLinkToADriveThatIsNotConnectedIsCountedNotListed() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.research(runs, "here")
        try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent("archived_a").path, withDestinationPath: "/Volumes/Archive Drive/runs/archived_a")
        try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent("archived_b").path, withDestinationPath: "/Volumes/Archive Drive/runs/archived_b")
        let snapshot = LibraryScanner().scan(runsDirectory: runs)
        #expect(snapshot.records.map(\.name) == ["here"])
        #expect(snapshot.offline == ["archived_a", "archived_b"] && snapshot.offlineVolumes == ["Archive Drive"])
        #expect(LibraryState.resolve(snapshot).rows.map(\.name) == ["here"])
    }

    @Test func aLinkToAFolderThatIsThereIsAWorld() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let elsewhere = scratch.url.appendingPathComponent("archive")
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        try LibraryFixture.research(elsewhere, "far")
        try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent("far").path, withDestinationPath: elsewhere.appendingPathComponent("far").path)
        let snapshot = LibraryScanner().scan(runsDirectory: runs)
        #expect(snapshot.records.map(\.name) == ["far"] && snapshot.offline.isEmpty)
    }

    @Test func filesHiddenEntriesAndAMissingFolderAreIgnored() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.write("x", "notes.txt", in: runs)
        try LibraryFixture.write("x", ".DS_Store", in: runs)
        try FileManager.default.createDirectory(at: runs.appendingPathComponent(".hidden"), withIntermediateDirectories: true)
        let scanner = LibraryScanner(settle: 0)
        #expect(scanner.scan(runsDirectory: runs).records.isEmpty)
        #expect(scanner.scan(runsDirectory: scratch.url.appendingPathComponent("nope")).records.isEmpty)
    }

    @Test func theLastSavedDateIsTheNewestOfTheDirectoryAndItsSavePoints() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "r")
        try LibraryFixture.setModified(dir, Date(timeIntervalSince1970: 1_000_000))
        try LibraryFixture.setModified(dir.appendingPathComponent("checkpoints"), Date(timeIntervalSince1970: 2_000_000))
        let record = try #require(LibraryScanner().scan(runsDirectory: runs).records.first)
        #expect(record.directoryModified == Date(timeIntervalSince1970: 1_000_000))
        #expect(record.lastSaved == Date(timeIntervalSince1970: 2_000_000))
    }

    @Test func theLibraryAndTheLauncherSeeTheSameFolderTheSameWay() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.many(runs, count: 60)
        let old = PlayCatalogScanner.scan(runsDirectory: runs)
        let new = LibraryState.resolve(LibraryScanner().scan(runsDirectory: runs)).catalog
        #expect(old.worlds.sorted { $0.name < $1.name } == new.worlds.sorted { $0.name < $1.name })
        #expect(old.sources.sorted { $0.name < $1.name } == new.sources.sorted { $0.name < $1.name })
        #expect(old.unfinished == new.unfinished && old.other == new.other)
        #expect(!old.worlds.isEmpty && !old.sources.isEmpty && !old.unfinished.isEmpty && !old.other.isEmpty)
    }

    @Test func aHeldLockIsProbedNotCached() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.played(runs, "w")
        try LibraryFixture.research(runs, "r")
        let scanner = LibraryScanner(settle: 0)
        let scanned = scanner.scan(runsDirectory: runs)
        #expect(scanned.held.isEmpty)
        let asked = Asked()
        let held = scanner.probe(scanned, runsDirectory: runs) { url in
            asked.add(url.lastPathComponent)
            return url.lastPathComponent == "w"
        }
        #expect(held.held == ["w"] && asked.all == ["w"], "only played worlds and builds are asked; a research run has no lock")
        #expect(scanner.probe(scanned, runsDirectory: runs, lockHeld: { _ in false }).held.isEmpty)
    }

    @Test func theLockOfABuildIsProbedSoThatABuildInProgressIsNotCalledUnfinished() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.unfinished(runs, "building", steps: [])
        try LibraryFixture.unfinished(runs, "stopped", steps: [])
        let asked = Asked()
        let scanner = LibraryScanner(settle: 0)
        let probed = scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs) { url in
            asked.add(url.lastPathComponent)
            return url.lastPathComponent == "building"
        }
        #expect(asked.all == ["building", "stopped"], "a directory that looks like a stopped build is asked whether `alife play` still holds it")
        #expect(probed.held == ["building"])
        let state = LibraryState.resolve(probed)
        #expect(state.unfinished.map(\.name) == ["stopped"], "the held one is being made, not unfinished")
    }

    @Test func aResearchRunWrittenAMomentAgoByAnotherProcessIsInUse() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let busy = try LibraryFixture.research(runs, "busy", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: busy)
        let idle = try LibraryFixture.research(runs, "idle", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: idle)
        let done = try LibraryFixture.research(runs, "done", status: "done")
        try LibraryFixture.write("{}", "metrics.jsonl", in: done)
        let now = Date()
        try LibraryFixture.setAge(idle, all: 600, now: now)
        let scanner = LibraryScanner(settle: 0)
        let probed = scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs, now: now, lockHeld: { _ in false })
        #expect(probed.writtenElsewhere == ["busy"], "a finished run, or one not written for minutes, is not in use")
    }

    @Test func aBigEventsFolderIsNotListedToKnowItHasChunks() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let events = scratch.url.appendingPathComponent("events")
        try FileManager.default.createDirectory(at: events, withIntermediateDirectories: true)
        #expect(!LibraryScanner.hasChunk(in: events))
        try LibraryFixture.write("x", ".hidden", in: events.deletingLastPathComponent().appendingPathComponent("events"))
        #expect(!LibraryScanner.hasChunk(in: events))
        try LibraryFixture.write("{}", "chunk_0.json", in: events)
        #expect(LibraryScanner.hasChunk(in: events))
        #expect(!LibraryScanner.hasChunk(in: scratch.url.appendingPathComponent("missing")))
    }

    @Test func aSourceIsNamedTheWayTheEngineWritesIt() {
        #expect(SourceRef.parse("runs/allfeat_a:0") == SourceRef(name: "allfeat_a", step: 0))
        #expect(SourceRef.parse("64k3d_s0:500000") == SourceRef(name: "64k3d_s0", step: 500_000))
        #expect(SourceRef.parse("/Users/me/artificial-life/runs/x_y/") == SourceRef(name: "x_y", step: nil))
        #expect(SourceRef.parse("") == nil && SourceRef.parse(nil) == nil && SourceRef.parse("runs/") == nil)
    }
}
