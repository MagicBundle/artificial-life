import Foundation
import Testing

@testable import AlifeCore

@Suite struct ClockStripTests {
    private func clock(rate: Float, achieved: Float = 14.4, reason: PlayClock.Reason = .running, late: UInt32 = 0,
                       saving: Bool = false, stall: UInt32 = 0, step: UInt64 = 1000) -> PlayClock {
        PlayClock(head: RecordHead(step: step, seq: 0, version: 1, flags: 0), rate: rate, achieved: achieved,
                  stepMs: 15, frameMs: 3, lateSteps: late, saveStallSteps: stall, reason: reason, saving: saving,
                  flags: 0, fe: 1, chapter: 0, inChapter: 0, chapterLen: 6000)
    }

    @Test func aRunningWorldSelectsItsRungAndSaysHowCloseItIs() {
        let strip = ClockStrip(clock: clock(rate: 1, achieved: 14.112), baseRate: 14.4)
        #expect(strip.selected == .rate(.multiplier(1)))
        #expect(!strip.paused)
        #expect(strip.readout == "0.98x of 1x")
        #expect(strip.step == 1000)
    }

    @Test func maxIsARungAndReportsTheAchievedMultiplier() {
        let strip = ClockStrip(clock: clock(rate: -1, achieved: 54), baseRate: 14.4)
        #expect(strip.selected == .rate(.max))
        #expect(strip.readout == "3.75x of max")
    }

    @Test func aRateThatIsNoRungSelectsNothingButIsNamed() {
        let strip = ClockStrip(clock: clock(rate: 4, achieved: 57.6), baseRate: 14.4)
        #expect(strip.selected == nil)
        #expect(strip.readout == "4.00x of 4x")
    }

    @Test func aPausedWorldSelectsPauseWithTheReason() {
        let strip = ClockStrip(clock: clock(rate: 0, reason: .userPause), baseRate: 14.4)
        #expect(strip.selected == .pause && strip.paused)
        #expect(strip.readout == "paused")
        #expect(ClockStrip(clock: clock(rate: 0, reason: .endOfSteps), baseRate: 14.4).readout == "end of steps")
        #expect(ClockStrip(clock: clock(rate: 1, reason: .error), baseRate: 14.4).readout == "stopped by an error")
    }

    @Test func aChapterOrAlertPauseIsPausedAndTheCardRunningIsNot() {
        // Reasons 3 and 4 with rate 0 are the world's own pauses (the chapter end, an alert): the strip shows Pause and why.
        let chapter = ClockStrip(clock: clock(rate: 0, reason: .chapter), baseRate: 14.4)
        #expect(chapter.selected == .pause && chapter.paused && chapter.readout == "end of chapter")
        #expect(chapter.stopReason == "end of chapter")
        let alert = ClockStrip(clock: clock(rate: 0, reason: .alert), baseRate: 14.4)
        #expect(alert.selected == .pause && alert.paused && alert.readout == "alert")
        // The announcement while the checkpoint of a chapter end is written, and a chapter end whose pause is off, carry
        // reason 3 with the world still running: the strip does not say paused for a second and then run again.
        let saving = ClockStrip(clock: clock(rate: 1, achieved: 14.112, reason: .chapter, saving: true), baseRate: 14.4)
        #expect(!saving.paused && saving.selected == .rate(.multiplier(1)) && saving.saving)
        #expect(saving.readout == "0.98x of 1x")
        // The window's own request still wins over the clock.
        let asked = ClockStrip(clock: clock(rate: 0, reason: .chapter), requested: .multiplier(1), baseRate: 14.4)
        #expect(!asked.paused && asked.selected == .rate(.multiplier(1)))
    }

    @Test func whatTheWindowJustAskedWinsOverTheClockUntilTheNextOne() {
        // Running at 1x; the user picked Pause: the strip shows it at once.
        let asked = ClockStrip(clock: clock(rate: 1), requested: .multiplier(0), baseRate: 14.4)
        #expect(asked.selected == .pause && asked.paused)
        // Paused; the user picked 2x.
        let resumed = ClockStrip(clock: clock(rate: 0, reason: .userPause), requested: .multiplier(2), baseRate: 14.4)
        #expect(resumed.selected == .rate(.multiplier(2)) && !resumed.paused)
    }

    @Test func beforeTheFirstClockTheStripWaits() {
        let strip = ClockStrip(clock: nil, baseRate: 14.4)
        #expect(strip.readout == "waiting for the clock" && strip.step == nil)
        #expect(!strip.saving && strip.lateSteps == 0)
    }

    @Test func lateStepsAndASaveAreShown() {
        let strip = ClockStrip(clock: clock(rate: 1, late: 3, saving: true, stall: 20), baseRate: 14.4)
        #expect(strip.lateSteps == 3 && strip.saving && strip.saveStallSteps == 20)
    }

    @Test func theSegmentsAreThePauseAndTheRungsAndSendTheirCommands() {
        #expect(ClockSegment.all.map(\.label) == ["Pause", "0.25x", "0.5x", "1x", "2x", "Max"])
        #expect(ClockSegment.all.first?.command == .pause)
        #expect(ClockSegment.rate(.max).command == .rate(.max))
        #expect(ClockSegment.rate(.multiplier(0.5)).command == .rate(.multiplier(0.5)))
    }
}

@Suite struct ClockStripDecideTests {
    private func clock(rate: Float, reason: PlayClock.Reason, flags: UInt16 = 3, saving: Bool = false) -> PlayClock {
        PlayClock(head: RecordHead(step: 12_000, seq: 0, version: 1, flags: 0), rate: rate, achieved: rate == 0 ? 0 : 14.4, stepMs: 15,
                  frameMs: 3, lateSteps: 0, saveStallSteps: 0, reason: reason, saving: saving, flags: flags, fe: 1, chapter: 2,
                  inChapter: 6000, chapterLen: 6000)
    }

    @Test func aChapterEndNamesTheChapterAndAnAlertNamesItsTitle() {
        let chapter = ClockStrip(clock: clock(rate: 0, reason: .chapter), baseRate: 14.4, chapterLabel: "Chapter 3")
        #expect(chapter.paused && chapter.readout == "Chapter 3 complete" && chapter.stopReason == "Chapter 3 complete")
        let alert = ClockStrip(clock: clock(rate: 0, reason: .alert), baseRate: 14.4, alertTitle: "Population fell")
        #expect(alert.paused && alert.readout == "Population fell")
        // Without the words the old ones stand; an empty title is none.
        #expect(ClockStrip(clock: clock(rate: 0, reason: .chapter), baseRate: 14.4).readout == "end of chapter")
        #expect(ClockStrip(clock: clock(rate: 0, reason: .alert), baseRate: 14.4, alertTitle: "").readout == "alert")
        // The reasons of a running world are not rewritten.
        let userPause = ClockStrip(clock: clock(rate: 0, reason: .userPause), baseRate: 14.4, chapterLabel: "Chapter 3", alertTitle: "x")
        #expect(userPause.readout == "paused")
    }

    @Test func theLockAndTheBellFollowTheFlagBits() {
        let both = ClockStrip(clock: clock(rate: 1, reason: .running, flags: 3), baseRate: 14.4)
        #expect(both.chapterPauseArmed && both.alertPauseArmed)
        let neither = ClockStrip(clock: clock(rate: 1, reason: .running, flags: 0), baseRate: 14.4)
        #expect(!neither.chapterPauseArmed && !neither.alertPauseArmed)
        #expect(!ClockStrip(clock: nil, baseRate: 14.4).chapterPauseArmed)
    }
}
