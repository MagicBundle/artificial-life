import Foundation
import Testing

@testable import AlifeCore

/// Every number the new pages take from the person or the engine ends in words or a command without trapping: the stop timer's hours and
/// minutes, a length typed by hand, a speed of zero, a time of any size. (A trap would end the app with an engine still writing a world.)
@Suite struct NumbersCannotTrapTests {
    @Test func theStopTimerIsClampedWhateverWasTypedAndTheSumNeverOverflows() {
        var s = RunSettings()
        s.timerEnabled = true
        for hours in [Int.max, Int.min, 10_000_000_000_000_000, -5, 100, 99, 0, 1] {
            for minutes in [Int.max, Int.min, -1, 60, 59, 0, 30] {
                s.timerHours = hours
                s.timerMinutes = minutes
                let seconds = s.timerDuration
                #expect(seconds >= 0 && seconds <= TimeInterval(99 * 3600 + 59 * 60), "\(hours) h \(minutes) min -> \(seconds)")
                _ = s.validate(existingRuns: [])   // the page asks on every redraw
            }
        }
        s.timerHours = 1; s.timerMinutes = 30
        #expect(s.timerDuration == 5400)
        s.timerHours = -3; s.timerMinutes = -10
        #expect(s.timerDuration == 0 && s.validate(existingRuns: []).contains("Timer duration must be greater than zero."), "a negative time is not a time")
        s.timerHours = 1_000_000; s.timerMinutes = 0
        #expect(s.timerDuration == TimeInterval(99 * 3600))
    }

    @Test func wordsForTimeNeverTrapOnAnyDouble() {
        for seconds in [Double.infinity, -.infinity, .nan, -1, 0, 4, 9.99, 10, 59, 60, 599, 600, 3599, 3600, 359_999, 360_000, 1e12, 1e30, Double.greatestFiniteMagnitude] {
            #expect(!TimeWords.about(seconds).isEmpty)
            #expect(TimeWords.clock(seconds).contains(":"))
            #expect(TimeWords.countdown(seconds).contains(":"))
        }
        #expect(TimeWords.about(1e30) == "more than 100 h")
        #expect(TimeWords.countdown(5400) == "1:30:00" && TimeWords.clock(65) == "1:05")
        // a speed of zero, or a length of any size, says nothing instead of trapping
        #expect(GrowEstimate.line(steps: Int.max, stepsPerSecond: 0.5) != nil)
        #expect(GrowEstimate.line(steps: 1000, stepsPerSecond: 0) == nil && GrowEstimate.line(steps: 1000, stepsPerSecond: .nan) == nil)
    }

    @Test func aLengthPlusItsStartStopsAtTheLargestNumber() {
        #expect(Int.max.saturatingAdding(20_000) == Int.max)
        #expect(Int.min.saturatingAdding(-1) == Int.min)
        #expect(30_000.saturatingAdding(20_000) == 50_000)
    }

    @Test func aNameWithATrailingNewlineIsNotAName() {
        let problems = SetupPlan.nameProblems("planet-1\n", goal: .research, intent: .create, taken: [])
        #expect(problems == ["A name can use letters, digits, dots, dashes and underscores, and must start with a letter or a digit."], "\(problems)")
        #expect(SetupPlan.nameProblems("planet-1", goal: .research, intent: .create, taken: []).isEmpty)
    }

    @Test func namesCompareAsTheFileSystemDoesWithoutCapitalsAndWithoutTellingComposedFromDecomposedLetters() {
        let composed = "caf\u{00E9}", decomposed = "cafe\u{0301}"
        #expect(NameFolding.same(composed, decomposed) && NameFolding.same("Planet", "PLANET"))
        #expect(NameFolding.clash(of: composed.uppercased(), in: [decomposed]) == decomposed)
        #expect(!NameFolding.same("planet-1", "planet-2"))
    }

    @Test func theRunOfAPlanetsWorldIsNamedByThePlanetNotByItsDirectory() {
        #expect(ActivityGuide.runTitle("planet-1-grow") == "Growing the world for planet-1")
        #expect(ActivityGuide.runTitle("planet-2") == "Research run planet-2")
        #expect(ActivityGuide.runTitle("-grow") == "Research run -grow", "a directory called only that is a name, not a suffix")
    }
}
