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
