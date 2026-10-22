import Foundation
import Testing

@testable import AlifeCore

@Suite struct ChapterBarTests {
    /// A clock of the chapter `chapter` (absolute) with `inChapter` steps done, as G3b's server sends them (§34.2.1).
    private func clock(chapter: UInt32, inChapter: UInt32, rate: Float = 1, achieved: Float = 14.4, flags: UInt16 = 3,
                       reason: PlayClock.Reason = .running, len: UInt32 = 6000) -> PlayClock {
        PlayClock(head: RecordHead(step: 0, seq: 0, version: 1, flags: 0), rate: rate, achieved: achieved, stepMs: 15, frameMs: 3,
                  lateSteps: 0, saveStallSteps: 0, reason: reason, saving: false, flags: flags, fe: 1, chapter: chapter,
                  inChapter: inChapter, chapterLen: len)
    }

    @Test func aPresetWorldCountsChaptersFromOne() {
        let bar = ChapterBar(clock: clock(chapter: 3, inChapter: 1500), originChapter: 0, baseRate: 14.4)
        #expect(bar.number == 3 && bar.label == "Chapter 3" && bar.progress == 0.25 && bar.stepsLeft == 4500)
        #expect(bar.chapterStart == 12_000 && bar.chapterLength == 6000)
        #expect(!bar.atOrigin)
    }

    @Test func theFirstChapterOfAForkIsChapterOne() {
        // A fork made at 450,000 (chapter 75, `made_at_step / 6000`): the clock at step 450,001 says chapter 76, 1 step in.
        let first = ChapterBar(clock: clock(chapter: 76, inChapter: 1), originChapter: 75, baseRate: 14.4)
        #expect(first.number == 1 && first.label == "Chapter 1" && first.chapterStart == 450_000 && !first.atOrigin)
        #expect(first.stepsLeft == 5999)
        let third = ChapterBar(clock: clock(chapter: 78, inChapter: 3000), originChapter: 75, baseRate: 14.4)
        #expect(third.number == 3 && third.progress == 0.5 && third.chapterStart == 462_000)
    }

    @Test func theVeryFirstMomentOfAForkReadsZeroNotFull() {
        // At step 450,000 the clock names chapter 75 at 100% (the chapter the fork was made in); to the player it is the start of
        // Chapter 1, with nothing done (the stage-3 corner).
        let bar = ChapterBar(clock: clock(chapter: 75, inChapter: 6000), originChapter: 75, baseRate: 14.4)
        #expect(bar.atOrigin && bar.number == 1 && bar.label == "Chapter 1" && bar.progress == 0)
        #expect(bar.stepsLeft == 6000 && bar.chapterStart == 450_000)
    }

    @Test func aChapterEndReadsFullAndTheChapterDoesNotChangeUntilTheNextStep() {
        let end = ChapterBar(clock: clock(chapter: 3, inChapter: 6000), originChapter: 0, baseRate: 14.4)
        #expect(end.number == 3 && end.progress == 1 && end.stepsLeft == 0 && end.remaining == nil && end.nextReport == nil)
        let next = ChapterBar(clock: clock(chapter: 4, inChapter: 1), originChapter: 0, baseRate: 14.4)
        #expect(next.number == 4 && next.progress < 0.001)
    }

    @Test func theTimeToTheReportIsFromThePaceOfTheLastTwoSeconds() {
        // 3,600 steps left at 14.4 steps a second: 250 s.
        let bar = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, achieved: 14.4), originChapter: 0, baseRate: 14.4)
        #expect(bar.remaining == "4:10" && bar.nextReport == "next report in 4:10")
        // At max (3.68x) it is the achieved pace that counts, not the request.
        let fast = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: -1, achieved: 53), originChapter: 0, baseRate: 14.4)
        #expect(fast.remaining == "1:08")
        // Without a measured pace the request times the base rate stands in for it.
        let fresh = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: 2, achieved: 0), originChapter: 0, baseRate: 14.4)
        #expect(fresh.remaining == "2:05")
    }

    @Test func aSlowPaceIsTheMeasuredOneNotTheRequests() {
        // 3,600 steps left at 2 steps a second (a loaded machine, or 0.14x): 30 minutes, not the 4:10 of the request.
        let bar = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: 1, achieved: 2), originChapter: 0, baseRate: 14.4)
        #expect(bar.remaining == "30:00")
        // Under half a step a second there is no measured pace worth dividing by: the request stands in.
        let stalled = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: 1, achieved: 0.25), originChapter: 0, baseRate: 14.4)
        #expect(stalled.remaining == "4:10")
    }

    @Test func aPausedWorldHasNoNextReportTime() {
        let paused = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: 0, achieved: 0), originChapter: 0, baseRate: 14.4)
        #expect(paused.remaining == nil && paused.nextReport == nil)
        // What the window just asked wins over the clock, as in the strip.
        let asked = ChapterBar(clock: clock(chapter: 1, inChapter: 2400), originChapter: 0, baseRate: 14.4, requested: .multiplier(0))
        #expect(asked.nextReport == nil)
        let resumed = ChapterBar(clock: clock(chapter: 1, inChapter: 2400, rate: 0, achieved: 0), originChapter: 0, baseRate: 14.4,
                                 requested: .multiplier(1))
        #expect(resumed.nextReport != nil)
    }

    @Test func theLockAndTheBellAreTheClocksFlags() {
        #expect(ChapterBar(clock: clock(chapter: 1, inChapter: 1, flags: 3), originChapter: 0, baseRate: 14.4).pauseArmed)
        let only = ChapterBar(clock: clock(chapter: 1, inChapter: 1, flags: 2), originChapter: 0, baseRate: 14.4)
        #expect(!only.pauseArmed && only.alertPauseArmed)
        let none = ChapterBar(clock: clock(chapter: 1, inChapter: 1, flags: 0), originChapter: 0, baseRate: 14.4)
        #expect(!none.pauseArmed && !none.alertPauseArmed)
    }

    @Test func noClockYetIsChapterOneAtZero() {
        let bar = ChapterBar(clock: nil, originChapter: 4, baseRate: 14.4)
        #expect(bar.label == "Chapter 1" && bar.progress == 0 && bar.nextReport == nil && bar.ticks(events: [], alerts: []).isEmpty)
    }

    @Test func aChapterOfAnotherLengthScalesTheBar() {
        let bar = ChapterBar(clock: clock(chapter: 1, inChapter: 15, len: 30), originChapter: 0, baseRate: 14.4)
        #expect(bar.progress == 0.5 && bar.stepsLeft == 15 && bar.chapterLength == 30)
        // A clock that says it is further than the chapter is long is the end, not beyond it.
        #expect(ChapterBar(clock: clock(chapter: 1, inChapter: 99, len: 30), originChapter: 0, baseRate: 14.4).progress == 1)
    }

    @Test func clockTextIsMinutesAndSeconds() {
        #expect(ChapterBar.clockText(seconds: 250) == "4:10" && ChapterBar.clockText(seconds: 7) == "0:07")
        #expect(ChapterBar.clockText(seconds: 4500) == "75:00" && ChapterBar.clockText(seconds: 59.6) == "1:00" && ChapterBar.clockText(seconds: 59.2) == "0:59")
        #expect(ChapterBar.clockText(seconds: -3) == "0:00")
    }

    // MARK: The ticks

    private func event(_ step: Int, imp: Int) -> WorldEvent {
        WorldEvent(id: UInt64(step) * 1000, step: step, kind: 11, fam: "storm", a: 1, b: 0, x: nil, y: nil, z: nil, imp: imp, text: "t")
    }

    private func alert(_ step: Int, cls: String = "storm_named", surface: PlayAlert.Surface = .alert,
                       state: PlayAlert.State = .raised) -> PlayAlert {
        PlayAlert(id: UInt64(step) * 1000 + 600, version: 1, step: step, alertClass: cls, key: "k", state: state, surface: surface,
                  imp: 2, title: "a", text: "t", x: nil, y: nil, z: nil, ref: nil, pause: false, expires: nil, watched: false,
                  calibrated: false)
    }

    @Test func theBarMarksTheToastSizedEventsAndTheAlertsOfThisChapterOnly() {
        let bar = ChapterBar(clock: clock(chapter: 2, inChapter: 3000), originChapter: 0, baseRate: 14.4)  // (6000, 12000]
        let ticks = bar.ticks(
            events: [event(5000, imp: 2), event(6001, imp: 2), event(7500, imp: 1), event(9000, imp: 2), event(12_001, imp: 2)],
            alerts: [alert(7500), alert(9000, cls: "chapter"), alert(8000, surface: .log), alert(10_500, state: .cleared),
                     alert(12_000, cls: "pop_fall")])
        // events at 6,001 and 9,000; alerts at 7,500 and 12,000 (the chapter's own alert, the log row and the cleared one are not marks)
        #expect(ticks.map(\.fraction) == [1.0 / 6000, 0.25, 0.5, 1])
        #expect(ticks.map(\.alert) == [false, true, false, true])
    }

    @Test func theMarksAreBounded() {
        let bar = ChapterBar(clock: clock(chapter: 1, inChapter: 3000), originChapter: 0, baseRate: 14.4)
        let many = (1...500).map { event($0 * 10, imp: 2) }
        #expect(bar.ticks(events: many, alerts: []).count == 48)
    }

    // MARK: The origin

    @Test func theOriginChapterIsTheChapterOfTheStepThePlanetWasMadeAt() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("chapter-origin-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("play.json")
        func write(_ planet: String) throws {
            let json = """
            {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 1000,
             "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25\(planet)}
            """
            try json.write(to: file, atomically: true, encoding: .utf8)
        }
        try write(#", "planet": {"source": "s", "step": 450000, "houses": "exact", "made_at_step": 450000}"#)
        #expect(ChapterOrigin.chapter(playFile: file, chapterLength: 6000) == 75)
        try write(#", "planet": {"source": "s", "step": 450100, "houses": "exact", "made_at_step": 450100}"#)
        #expect(ChapterOrigin.chapter(playFile: file, chapterLength: 6000) == 75, "an off-grid fork: the same chapter, an arrival chapter")
        try write("")
        #expect(ChapterOrigin.chapter(playFile: file, chapterLength: 6000) == 0, "a preset or adopted run")
        #expect(ChapterOrigin.chapter(playFile: dir.appendingPathComponent("none.json"), chapterLength: 6000) == 0)
        #expect(ChapterOrigin.chapter(playFile: file, chapterLength: 0) == 0)
    }
}
