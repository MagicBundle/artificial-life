import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The readings of one world's files for the Run panel and Details (docs/APP.md §7.5): by name, not through the set-up page's target or the
/// run this app grows, so a world that is not growing has its numbers and its journal.
@MainActor
@Suite("The reader of a world's files", .serialized)
struct WorldReaderTests {
    @MainActor struct Rig {
        let scratch: Scratch
        var runs: URL { scratch.url.appendingPathComponent("runs") }
        init() throws { scratch = try Scratch() }

        /// A world with a metrics file, an autopilot log, a journal, a meta and a config.
        @discardableResult
        func world(_ name: String, step: Int = 1000, alive: Double = 900, status: String = "done", director: Bool = true, extra: String = "") throws -> URL {
            let dir = runs.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try row(step: step, alive: alive, extra: extra).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
            try #"{"status": "\#(status)", "start_step": 0}"#.write(to: dir.appendingPathComponent("meta.json"), atomically: true, encoding: .utf8)
            try #"{"run": {"seed": 3, "steps": 5000}, "world": {"dims": 2, "geometry": "box", "size": 64}}"#
                .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
            if director {
                let log = #"{"kind": "header", "persona": "gardener"}"# + "\n"
                    + #"{"step_applied": 500, "source": "rule", "action": "regen_pulse", "params": {"gain": 1.5}, "reason": "the population fell"}"# + "\n"
                try log.write(to: dir.appendingPathComponent("director.jsonl"), atomically: true, encoding: .utf8)
                try (#"{"step": 500, "text": "rule fired: regen_pulse"}"# + "\n").write(to: dir.appendingPathComponent("journal.jsonl"), atomically: true, encoding: .utf8)
            }
            return dir
        }

        func row(step: Int, alive: Double, extra: String = "") -> String {
            #"{"step": \#(step), "n_alive": \#(alive), "steps_per_s": 321.5, "clades_alive": 12, "generation_max": 40, "ledger_error": 0\#(extra)}"# + "\n"
        }
    }

    private func reader(_ rig: Rig) -> WorldReader {
        WorldReader(runsDirectory: { rig.runs }, interval: .milliseconds(30))
    }

    @Test func aWorldThatIsNotGrowingHasItsNumbersItsJournalAndItsSettings() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("saved_one")
        let r = reader(rig)
        r.follow("saved_one")
        await r.readNow()
        let reading = try #require(r.reading)
        #expect(reading.world == "saved_one" && reading.exists && reading.status == "done")
        #expect(reading.metrics?.step == 1000 && reading.metrics?.nAlive == 900 && reading.metrics?.clades == 12 && reading.metrics?.generationMax == 40)
        let director = try #require(reading.director)
        #expect(director.hasLog && director.interventions.map(\.action) == ["regen_pulse"])
        if case .note(let note)? = director.journal.first { #expect(note.text == "rule fired: regen_pulse") } else { Issue.record("no journal note") }
        #expect(reading.madeWith?.seed == 3 && reading.madeWith?.steps == 5000)
    }

    @Test func aWorldWithoutAnAutopilotHasNoLogAndSaysSo() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("plain", director: false)
        let r = reader(rig)
        r.follow("plain")
        await r.readNow()
        #expect(r.reading?.director?.hasLog == false && r.reading?.metrics != nil)
    }

    @Test func aWorldThatIsNotThereIsReadAsNotThere() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        let r = reader(rig)
        r.follow("nowhere")
        await r.readNow()
        #expect(r.reading?.exists == false && r.reading?.metrics == nil)
        r.follow(nil)
        await r.readNow()
        #expect(r.reading == nil)
    }

    @Test func aFolderWithoutAMetricsFileHasNoNumbersAndStillAnswers() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try FileManager.default.createDirectory(at: rig.runs.appendingPathComponent("bare"), withIntermediateDirectories: true)
        let r = reader(rig)
        r.follow("bare")
        await r.readNow()
        #expect(r.reading?.exists == true && r.reading?.metrics == nil && r.reading?.director?.hasLog == false && r.reading?.status == nil)
    }

    @Test func theNumbersOfAnotherWorldAreNeverShownUnderThisOnesName() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 111)
        try rig.world("b", step: 222)
        let r = reader(rig)
        r.follow("a")
        await r.readNow()
        #expect(r.reading?.metrics?.step == 111)
        r.follow("b")
        #expect(r.reading == nil, "a different world clears what was read at once")
        await r.readNow()
        #expect(r.reading?.world == "b" && r.reading?.metrics?.step == 222)
    }

    @Test func aReadThatWasOnItsWayWhenTheWorldChangedIsNotApplied() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 111)
        try rig.world("b", step: 222)
        let r = reader(rig)
        var inFlight = false
        r.afterRead = { name in
            if name == "a" {
                inFlight = true
                try? await Task.sleep(for: .milliseconds(200))  // (the read of a is back, and waits)
            }
        }
        r.follow("a")
        let late = Task { await r.readNow() }
        #expect(await eventually { inFlight })
        r.follow("b")
        await late.value
        #expect(r.reading == nil || r.reading?.world == "b", "the late read of a must not land under b: \(String(describing: r.reading?.world))")
        r.afterRead = nil
        await r.readNow()
        #expect(r.reading?.world == "b" && r.reading?.metrics?.step == 222)
    }

    @Test func twoReadsThatOverlapNeverLetTheOlderOverwriteTheNewer() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        let dir = try rig.world("a", step: 1)
        let r = reader(rig)
        var first = true
        r.afterRead = { _ in
            if first {
                first = false
                try? await Task.sleep(for: .milliseconds(200))  // the older read is slow to apply
            }
        }
        r.follow("a")
        let older = Task { await r.readNow() }
        try await Task.sleep(for: .milliseconds(60))
        try rig.row(step: 2, alive: 5).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
        await r.readNow()
        #expect(r.reading?.metrics?.step == 2)
        await older.value
        #expect(r.reading?.metrics?.step == 2, "the older read finished later and must not show step 1 again")
    }

    @Test func theFeatureKeysThatOnlySomeRowsCarryKeepTheirLastValue() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        let dir = try rig.world("f", extra: #", "arms_hits": 7"#)
        let r = reader(rig)
        r.follow("f")
        await r.readNow()
        #expect(r.reading?.metrics?.features["arms_hits"] == 7)
        // the next row is a light one: no heavy key
        try rig.row(step: 2000, alive: 880).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
        await r.readNow()
        #expect(r.reading?.metrics?.step == 2000 && r.reading?.metrics?.features["arms_hits"] == 7)
        // and a new value replaces the old
        try rig.row(step: 3000, alive: 880, extra: #", "arms_hits": 9"#).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
        await r.readNow()
        #expect(r.reading?.metrics?.features["arms_hits"] == 9)
    }

    @Test func theMemoryOfOneWorldIsNotLentToTheNext() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", extra: #", "arms_hits": 7"#)
        try rig.world("b")
        let r = reader(rig)
        r.follow("a")
        await r.readNow()
        r.follow("b")
        await r.readNow()
        #expect(r.reading?.metrics?.features["arms_hits"] == nil)
    }

    @Test func theSettingsAreReadAgainWhileTheConfigIsNotThereYet() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        let dir = try rig.world("late")
        try FileManager.default.moveItem(at: dir.appendingPathComponent("config.json"), to: dir.appendingPathComponent("config.off"))
        let r = reader(rig)
        r.follow("late")
        await r.readNow()
        #expect(r.reading?.madeWith == nil)
        try FileManager.default.moveItem(at: dir.appendingPathComponent("config.off"), to: dir.appendingPathComponent("config.json"))
        await r.readNow()
        #expect(r.reading?.madeWith?.seed == 3)
    }

    // MARK: Polling

    @Test func itPollsOnlyWhileSomeoneShowsIt() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        let dir = try rig.world("w", step: 10)
        let r = reader(rig)
        r.follow("w")
        #expect(!r.isPolling && r.reading == nil, "following alone reads nothing and polls nothing")
        r.retain()
        #expect(r.isPolling)
        #expect(await eventually { r.reading?.metrics?.step == 10 })
        try rig.row(step: 20, alive: 5).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
        #expect(await eventually { r.reading?.metrics?.step == 20 }, "a growing world's numbers move by themselves")
        r.retain()
        r.release()
        #expect(r.isPolling, "two pages show it; one left")
        r.release()
        #expect(!r.isPolling)
        r.release()
        #expect(!r.isPolling, "an extra release does nothing")
        try await Task.sleep(for: .milliseconds(200))  // (a read that was already on its way when the last page left may still land)
        try rig.row(step: 30, alive: 5).write(to: dir.appendingPathComponent("metrics.jsonl"), atomically: true, encoding: .utf8)
        try await Task.sleep(for: .milliseconds(300))
        #expect(r.reading?.metrics?.step == 20, "nothing reads while nobody shows it")
    }

    @Test func changingTheWorldWhileItIsShownReadsTheNewOneAtOnce() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 1)
        try rig.world("b", step: 2)
        let r = reader(rig)
        r.retain()
        defer { r.release() }
        r.follow("a")
        #expect(await eventually { r.reading?.metrics?.step == 1 })
        r.follow("b")
        #expect(await eventually { r.reading?.metrics?.step == 2 && r.reading?.world == "b" })
    }

    @Test func followingTheSameWorldAgainKeepsWhatWasRead() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 5)
        let r = reader(rig)
        r.follow("a")
        await r.readNow()
        r.follow("a")
        #expect(r.reading?.metrics?.step == 5, "a page that appears again must not blank the numbers")
    }

    @Test func changingTheWorldWhileItIsShownReadsTheNewOneAtOnceNotAtTheNextPoll() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 1)
        try rig.world("b", step: 2)
        let r = WorldReader(runsDirectory: { rig.runs }, interval: .seconds(30))
        r.retain()
        defer { r.release() }
        r.follow("a")
        #expect(await eventually(timeout: .seconds(5)) { r.reading?.metrics?.step == 1 })
        r.follow("b")
        #expect(await eventually(timeout: .seconds(5)) { r.reading?.metrics?.step == 2 }, "the poll is thirty seconds away: only the immediate read can show it")
    }

    @Test func aFolderThatGoesAwayClearsTheReading() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 1)
        var folder: URL? = rig.runs
        let r = WorldReader(runsDirectory: { folder }, interval: .milliseconds(30))
        r.follow("a")
        await r.readNow()
        #expect(r.reading != nil)
        folder = nil
        await r.readNow()
        #expect(r.reading == nil)
    }

    @Test func aSlowVolumeCollectsOneReadNotOnePerPoll() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 1)
        let r = reader(rig)  // polls every 30 ms
        var reads = 0
        r.afterRead = { _ in
            reads += 1
            // A read that is still not back when the test looks (far longer than the look: a stalled main actor can delay the look by seconds,
            // and a read that came back meanwhile would start the next one legitimately).
            try? await Task.sleep(for: .seconds(6))
        }
        r.follow("a")
        r.retain()
        defer { r.release() }
        #expect(await eventually { reads == 1 })
        try await Task.sleep(for: .milliseconds(300))
        #expect(reads == 1, "\(reads) reads were started while the first had not come back")
    }

    @Test func aWorldThatChangesWhileAReadIsOnItsWayIsReadWhenThatReadIsBack() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        try rig.world("a", step: 1)
        try rig.world("b", step: 2)
        let r = WorldReader(runsDirectory: { rig.runs }, interval: .seconds(60))  // no poll will come to the rescue
        var slow = true
        r.afterRead = { name in if name == "a", slow { slow = false; try? await Task.sleep(for: .milliseconds(300)) } }
        r.follow("a")
        r.retain()
        defer { r.release() }
        try await Task.sleep(for: .milliseconds(100))
        r.follow("b")
        #expect(await eventually(timeout: .seconds(5)) { r.reading?.world == "b" && r.reading?.metrics?.step == 2 }, "the read for b waits for the one in flight and then follows it")
    }

    @Test func noFolderMeansNothingToRead() async {
        let r = WorldReader(runsDirectory: { nil }, interval: .milliseconds(30))
        r.follow("w")
        await r.readNow()
        #expect(r.reading == nil)
    }
}
