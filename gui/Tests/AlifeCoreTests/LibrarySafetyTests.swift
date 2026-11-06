import AlifeTestSupport
import Foundation
import Testing

@testable import AlifeCore

/// The library never lets two writers meet, never calls a slow world finished, and is not held up by one entry (docs/APP.md §5.2 R6, §5.3).
@Suite struct LibrarySafetyTests {
    private func runs(_ scratch: ScratchDirectory) throws -> URL {
        let runs = scratch.url.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        return runs
    }

    private func world(_ name: String, held: Bool = false) -> PlayWorld {
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil)
        return PlayWorld(name: name, file: file, newestStep: 100, status: nil, modified: Date(), locked: held, geometry: "sphere")
    }

    // MARK: H1: two writers

    @Test func aWorldThisWindowIsGrowingCannotBePlayedAndIsNeverTheNextStep() {
        var catalog = PlayCatalog()
        catalog.worlds = [world("grown"), world("quiet")]
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: catalog, growing: "grown")
        #expect(WorldsGuide.reason(forPlaying: world("grown"), in: facts) == "Still growing")
        #expect(WorldsGuide.reason(forPlaying: world("quiet"), in: facts) == nil)
        #expect(WorldsGuide.nextStep(facts) == .play(world: "quiet"))
        var only = PlayCatalog()
        only.worlds = [world("grown")]
        #expect(WorldsGuide.nextStep(WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: only, growing: "grown")) == .waitForSave)
    }

    @Test func aPlayedWorldAnotherProcessResumedIsInUseNotSavedAndNotPlayable() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.played(runs, "resumed")
        try LibraryFixture.write(#"{"status": "running", "start_step": 0, "play_build": "planet"}"#, "meta.json", in: dir)
        try LibraryFixture.write("{}", "metrics.jsonl", in: dir)
        let scanner = LibraryScanner(settle: 0)
        let probed = scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs, lockHeld: { _ in false })
        #expect(probed.writtenElsewhere == ["resumed"], "the engine's resume takes no play.lock: the metrics are the only sign")
        let state = LibraryState.resolve(probed)
        #expect(state.row(named: "resumed")?.state == .inUseElsewhere)
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: state.catalog)
        #expect(WorldsGuide.reason(forPlaying: state.catalog.worlds[0], in: facts) == "In use elsewhere")
        #expect(WorldsGuide.nextStep(facts) == .waitForSave, "never picks it")
    }

    // MARK: M1: a slow world is not a finished one

    @Test func theWindowGrowsWithTheRunsCadence() {
        #expect(LibraryScanner.signWindow(metricsEvery: 100) == 75)
        #expect(LibraryScanner.signWindow(metricsEvery: 10) == 20, "never under 20 s")
        #expect(LibraryScanner.signWindow(metricsEvery: 1000) == 750)
        #expect(LibraryScanner.cutOffAfter(metricsEvery: 100) == 750 && LibraryScanner.cutOffAfter(metricsEvery: 10) == 600)
    }

    private func probe(_ dir: URL, runs: URL, status: String = "running", metricsEvery: Int? = nil, ages: [String: TimeInterval], all: TimeInterval) throws
        -> LibrarySnapshot {
        let now = Date()
        _ = status
        try LibraryFixture.setAge(dir, all: all, ages: ages, now: now)
        let scanner = LibraryScanner(settle: 0)
        return scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs, now: now, lockHeld: { _ in false })
    }

    @Test func aRowThatIsLateIsStillARunningWorldUntilItsWindowIsOver() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "slow", status: "running", metricsEvery: 1000)
        try LibraryFixture.write("{}", "metrics.jsonl", in: dir)
        // 60 s since every sign: over the default window (75 s? no: under), so use a slow cadence to show the scaling
        let slow = try probe(dir, runs: runs, ages: [:], all: 300)
        #expect(slow.writtenElsewhere == ["slow"], "a heavy-cadence run (window 750 s) quiet for 300 s is still running")
        #expect(slow.cutOff.isEmpty)
        let fast = try LibraryFixture.research(runs, "fast", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: fast)
        let late = try probe(fast, runs: runs, ages: [:], all: 100)
        #expect(!late.writtenElsewhere.contains("fast") && late.cutOff.isEmpty, "100 s on a default cadence: not written, and not 'cut off' either")
        let state = LibraryState.resolve(late)
        #expect(state.row(named: "fast")?.note == nil, "an uncertain world is not called cut off")
    }

    @Test func aCheckpointWrittenOrAResumeLoadingIsASignOfLife() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let a = try LibraryFixture.research(runs, "writing", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: a)
        let r1 = try probe(a, runs: runs, ages: ["checkpoints": 2], all: 3000)
        #expect(r1.writtenElsewhere == ["writing"], "a long checkpoint write moves checkpoints/ while the metrics wait")
        let b = try LibraryFixture.research(runs, "loading", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: b)
        let r2 = try probe(b, runs: runs, ages: ["meta.json": 5], all: 3000)
        #expect(r2.writtenElsewhere.contains("loading"), "a resume sets running in meta.json before it has loaded anything")
    }

    @Test func aWorldSilentForALongTimeIsCutOffAndOnlyThenSaidSo() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "dead", status: "running")
        try LibraryFixture.write("{}", "metrics.jsonl", in: dir)
        let snap = try probe(dir, runs: runs, ages: [:], all: 5000)
        #expect(snap.cutOff == ["dead"] && snap.writtenElsewhere.isEmpty)
        #expect(LibraryState.resolve(snap).row(named: "dead")?.note == "Cut off without a clean stop.")
    }

    // MARK: M5: one bad entry does not hold up the rest

    @Test func anEntryThatHangsIsReportedSlowKeepsItsLastRecordAndIsNotAskedAgain() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.research(runs, "good")
        try LibraryFixture.research(runs, "hangs")
        let scanner = LibraryScanner(settle: 0, timeout: 0.4)
        let first = scanner.scan(runsDirectory: runs)
        #expect(first.records.map(\.name) == ["good", "hangs"] && first.slow.isEmpty)
        let gate = DispatchSemaphore(value: 0)
        let calls = Asked()
        let hung = Asked()
        scanner.inspectHook = { name in
            calls.add(name)
            if name == "hangs", hung.all.isEmpty {
                hung.add(name)
                _ = gate.wait(timeout: .now() + TestTime.limit(seconds: 5))
            }
        }
        let started = Date()
        let second = scanner.scan(runsDirectory: runs)
        #expect(Date().timeIntervalSince(started) < TestTime.limit(seconds: 2), "the deadline, not the hang")
        #expect(second.slow == ["hangs"] && second.records.map(\.name) == ["good", "hangs"], "the last known record stays")
        // still stuck: not asked again
        let third = scanner.scan(runsDirectory: runs)
        #expect(third.slow == ["hangs"])
        #expect(calls.all.filter { $0 == "hangs" }.count == 1, "one look is out; a second is not started on top of it")
        gate.signal()
        Thread.sleep(forTimeInterval: TestTime.limit(seconds: 0.2))
        let fourth = scanner.scan(runsDirectory: runs)
        #expect(fourth.slow.isEmpty, "it answered")
        let state = LibraryState.resolve(second)
        #expect(state.slow == ["hangs"] && state.catalog.existingNames.contains("hangs"))
    }

    @Test func aLockProbeThatHangsKeepsTheLastAnswerAndHoldsUpNothing() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.played(runs, "fine")
        try LibraryFixture.played(runs, "stuck")
        let scanner = LibraryScanner(settle: 0, timeout: 0.4)
        var snapshot = scanner.scan(runsDirectory: runs)
        snapshot.held = ["stuck"]
        let started = Date()
        let probed = scanner.probe(snapshot, runsDirectory: runs) { url in
            if url.lastPathComponent == "stuck" { Thread.sleep(forTimeInterval: TestTime.limit(seconds: 2)) }
            return false
        }
        #expect(Date().timeIntervalSince(started) < TestTime.limit(seconds: 1.5))
        #expect(probed.held == ["stuck"], "no answer in time: what was known stays; the world that answered is not held")
    }

    @Test func eachKindOfEntryThatCannotBeListedIsToldInItsOwnWords() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        func link(_ name: String, to target: String) throws {
            try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent(name).path, withDestinationPath: target)
        }
        try link("on_drive", to: "/Volumes/NoSuchDriveAtAll/runs/x")
        try link("elsewhere", to: scratch.url.appendingPathComponent("nothing_here").path)
        try link("loop_a", to: runs.appendingPathComponent("loop_b").path)
        try link("loop_b", to: runs.appendingPathComponent("loop_a").path)
        let locked = scratch.url.appendingPathComponent("locked")
        try FileManager.default.createDirectory(at: locked.appendingPathComponent("inside"), withIntermediateDirectories: true)
        try link("denied", to: locked.appendingPathComponent("inside").path)
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: locked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path) }
        let snapshot = LibraryScanner(settle: 0).scan(runsDirectory: runs)
        #expect(snapshot.problems["on_drive"] == .driveNotConnected(volume: "NoSuchDriveAtAll"))
        #expect(snapshot.problems["elsewhere"] == .brokenLink, "not a drive: a link to nothing")
        #expect(snapshot.problems["loop_a"] == .loop && snapshot.problems["loop_b"] == .loop)
        #expect(snapshot.problems["denied"] == .noPermission)
        #expect(snapshot.offline == ["on_drive"], "only a missing /Volumes target is 'a drive that is not connected'")
        let state = LibraryState.resolve(snapshot)
        for name in ["on_drive", "elsewhere", "loop_a", "denied"] { #expect(state.catalog.existingNames.contains(name), "\(name) is a taken name") }
    }

    @Test func aClickTimeLookIsDoneOnTheLaneAndSaysThereGoneOrUnknown() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "w")
        let scanner = LibraryScanner(settle: 0)
        #expect(scanner.presence(of: dir) == .there)
        try FileManager.default.removeItem(at: dir)
        #expect(scanner.presence(of: dir) == .gone)
    }

    // MARK: M3: the order

    @Test func aGrowingSeedIsNotHiddenUnderAQuietParent() {
        func row(_ name: String, kind: WorldKind, state: WorldState, saved: Double, from: String? = nil) -> LibraryRow {
            LibraryRow(name: name, kind: kind, shape: .planet, state: state, note: nil, step: 1, lastSaved: Date(timeIntervalSince1970: saved),
                       madeFrom: from.map { SourceRef(name: $0, step: 1) })
        }
        var rows = (0..<12).map { row("w\($0)", kind: .played, state: .saved, saved: 1000 - Double($0)) }
        rows.append(row("old_parent", kind: .played, state: .saved, saved: 1, from: "seed"))
        rows.append(row("seed", kind: .research, state: .growing, saved: 2))
        let tree = LibraryState.tree(LibraryState.order(rows))
        #expect(tree.first?.row.name == "old_parent" && tree.first?.children.map(\.name) == ["seed"], "the node with a growing seed is first")
        #expect(tree.dropFirst().map(\.row.name) == (0..<12).map { "w\($0)" }, "the rest keep their order: newest saved first")
    }

    @Test func theOrderMovesARowOnlyWhenItsRankOrItsDateChanges() {
        func row(_ name: String, _ state: WorldState, _ saved: Double) -> LibraryRow {
            LibraryRow(name: name, kind: .research, shape: nil, state: state, note: nil, step: 1, lastSaved: Date(timeIntervalSince1970: saved), madeFrom: nil)
        }
        let base = [row("a", .saved, 30), row("b", .saved, 20), row("c", .saved, 10)]
        #expect(LibraryState.order(base).map(\.name) == ["a", "b", "c"])
        #expect(LibraryState.order(base.reversed()).map(\.name) == ["a", "b", "c"], "the input order is nothing")
        var busy = base
        busy[1] = row("b", .inUseElsewhere, 20)
        #expect(LibraryState.order(busy).map(\.name) == ["b", "a", "c"], "a state in a better class goes first")
        busy[1] = row("b", .saved, 20)
        #expect(LibraryState.order(busy).map(\.name) == ["a", "b", "c"], "and goes back when it leaves it")
        var saved = base
        saved[2] = row("c", .saved, 40)
        #expect(LibraryState.order(saved).map(\.name) == ["c", "a", "b"], "a world just saved goes to the top of its class")
    }

    // MARK: L1, L2, L3, L5

    /// A fixed, whole-second date for every file and folder of a run, so that "the same date" really is the same to the nanosecond.
    private func pinDates(_ dir: URL, _ when: Date) throws {
        for name in ["meta.json", "config.json", "checkpoints", "events"] {
            let url = dir.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { try LibraryFixture.setModified(url, when) }
        }
        try LibraryFixture.setModified(dir, when)
    }

    @Test func aFileRewrittenInTheSameInstantWithAnotherSizeIsRead() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "r", status: "done")
        let when = Date(timeIntervalSince1970: 1_700_000_000)
        try pinDates(dir, when)
        let scanner = LibraryScanner(settle: 0)
        _ = scanner.scan(runsDirectory: runs)
        // in place, longer text, every date as it was (a disk with coarse dates): only the size says so
        try Data(#"{"status": "extinct", "start_step": 0, "note": "same second"}"#.utf8).write(to: dir.appendingPathComponent("meta.json"))
        try pinDates(dir, when)
        #expect(scanner.scan(runsDirectory: runs).records.first?.status == "extinct", "the size is part of the stamp")
    }

    @Test func theMetaStampAloneNoticesAnInPlaceRewrite() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "r", status: "done")
        let when = Date(timeIntervalSince1970: 1_700_000_000)
        try pinDates(dir, when)
        let scanner = LibraryScanner(settle: 0)
        _ = scanner.scan(runsDirectory: runs)
        // the same length, in place; the directory and everything else keep their dates, `meta.json` alone gets a later one
        try Data(#"{"status": "dead", "start_step": 0}"#.utf8).write(to: dir.appendingPathComponent("meta.json"))
        try pinDates(dir, when)
        try LibraryFixture.setModified(dir.appendingPathComponent("meta.json"), when.addingTimeInterval(5))
        #expect(scanner.scan(runsDirectory: runs).records.first?.status == "dead")
    }

    @Test func aDirectoryTooFreshToTrustIsReadAgainUntilItSettles() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let dir = try LibraryFixture.research(runs, "r")
        let scanner = LibraryScanner(settle: 2)
        _ = scanner.scan(runsDirectory: runs)
        _ = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 1, "written a moment ago: read again, in case a second write shared its date")
        try LibraryFixture.setAge(dir, all: 3600)
        _ = scanner.scan(runsDirectory: runs)  // reads (stamps moved) and settles
        _ = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 0 && scanner.lastStats.reused == 1)
    }

    @Test func theLibraryAndTheLauncherAgreeOnRealDirectoriesAndDifferOnLinksAndDates() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.many(runs, count: 40)
        // a world on a drive that is connected, reached by a link: the library lists it; the launcher's scan has always skipped links
        let archive = scratch.url.appendingPathComponent("archive")
        try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: true)
        try LibraryFixture.research(archive, "far")
        try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent("far").path, withDestinationPath: archive.appendingPathComponent("far").path)
        // a world held by a process that is not this one (a second open file description of the same file)
        let held = try LibraryFixture.played(runs, "held_world")
        let fd = open(held.appendingPathComponent("play.lock").path, O_RDWR | O_CREAT, 0o644)
        #expect(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0)
        defer { close(fd) }
        let old = PlayCatalogScanner.scan(runsDirectory: runs)
        let scanner = LibraryScanner(settle: 0)
        let scanned = scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs)
        let new = LibraryState.resolve(scanned).catalog
        func names<T>(_ items: [T], _ name: (T) -> String) -> [String] { items.map(name).sorted() }
        #expect(names(old.worlds, { $0.name }) == names(new.worlds, { $0.name }))
        #expect(old.worlds.first { $0.name == "held_world" }?.locked == true && new.worlds.first { $0.name == "held_world" }?.locked == true)
        #expect((names(old.sources, { $0.name }) + ["far"]).sorted() == names(new.sources, { $0.name }), "the one difference: the link to a folder that is there")
        #expect(old.unfinished == new.unfinished && old.other == new.other)
        // `modified` is the last-saved date (the newest of the directory's and checkpoints/'s), not the directory's alone
        for source in new.sources {
            let o = old.sources.first { $0.name == source.name }
            if let o { #expect(source.modified >= o.modified.addingTimeInterval(-0.001)) }
        }
    }

    @Test func theWorldTheNextStepOffersIsTheFirstOfTheList() throws {
        // directory dates say "a" is the newer; the save points say "b" is
        let rec: (String, Double, Double) -> LibraryRecord = { name, dir, saved in
            let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
            let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil)
            return LibraryRecord(
                name: name, play: file, steps: [100], status: "done", parent: nil, dims: 3, geometry: "sphere", worldSize: nil, planetRadius: nil,
                planetShape: nil, weatherOn: false, hasEvents: false, directoryModified: Date(timeIntervalSince1970: dir),
                lastSaved: Date(timeIntervalSince1970: saved), hasContent: true, looksUnfinished: false, buildMark: nil)
        }
        var snapshot = LibrarySnapshot()
        snapshot.records = [rec("a", 1000, 1000), rec("b", 500, 2000)]
        let state = LibraryState.resolve(snapshot)
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: state.catalog)
        #expect(state.rows.first?.name == "b")
        #expect(WorldsGuide.nextStep(facts) == .play(world: "b"), "the card and the first row name the same world")
    }

    @Test func aWorldThisAppIsStartingIsNotProbed() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        try LibraryFixture.played(runs, "mine")
        try LibraryFixture.played(runs, "other")
        let asked = Asked()
        let scanner = LibraryScanner(settle: 0)
        let probed = scanner.probe(scanner.scan(runsDirectory: runs), runsDirectory: runs, skipping: ["mine"]) { url in
            asked.add(url.lastPathComponent)
            return false
        }
        #expect(asked.all == ["other"], "the lock of a world that `alife play` is about to take is not touched")
        #expect(probed.held == ["mine"], "and it counts as held: it is this app's")
    }

    // MARK: Notes, words

    @Test func aSessionNoteIsDroppedWhenTheDirectoryWasTouchedAfterIt() {
        let r = LibraryRecord(
            name: "u", play: nil, steps: [], status: "interrupted", parent: nil, dims: 3, geometry: "sphere", worldSize: nil, planetRadius: nil,
            planetShape: nil, weatherOn: false, hasEvents: false, directoryModified: Date(timeIntervalSince1970: 5000),
            lastSaved: Date(timeIntervalSince1970: 5000), hasContent: true, looksUnfinished: true, buildMark: "planet")
        var snapshot = LibrarySnapshot()
        snapshot.records = [r]
        let early = LibraryState.resolve(snapshot, notes: ["u": "You cancelled it"], noteTimes: ["u": Date(timeIntervalSince1970: 1000)])
        #expect(early.unfinished.first?.reason.hasPrefix("Cancelled before") == true, "touched long after the note: a later, different stop")
        let fresh = LibraryState.resolve(snapshot, notes: ["u": "You cancelled it"], noteTimes: ["u": Date(timeIntervalSince1970: 4999)])
        #expect(fresh.unfinished.first?.reason.hasPrefix("You cancelled it before") == true)
    }

    @Test func theFormsMessagesAreInThePagesWords() {
        #expect(WorldsGuide.plain("Choose a source run.") == "Choose a world to start from.")
        #expect(WorldsGuide.plain("'runs/planet_a' exists.") == "A world named planet_a is already here.")
        #expect(WorldsGuide.plain("The checkpoint step is not negative.") == "The save point is not negative.")
        var spec = PlanetSpec(source: "", name: "a b")
        spec.storm = true
        spec.weatherSpinup = 0
        spec.sourceStep = -1
        spec.directorEnabled = false
        spec.director = .rules
        spec.sourceDims = 2
        var all = spec.validate(existing: []).map(WorldsGuide.plain)
        spec.storm = false
        spec.weatherSpinup = 5
        spec.name = "taken"
        all += spec.validate(existing: ["taken"]).map(WorldsGuide.plain)
        all.append(WorldsGuide.plain(PlanetSpec.weatherObstacle(dims: 2) ?? ""))
        #expect(all.count >= 8)
        for text in all {
            for word in ["source", "Source", "checkpoint", "Checkpoint", "spin-up", "director.enabled", "runs/"] {
                #expect(!text.contains(word), "'\(word)' in \"\(text)\"")
            }
        }
    }
}
