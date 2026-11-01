import Foundation
import Testing

@testable import AlifeCore

/// The Dock icon while a run goes with the window closed (docs/APP.md D3 = C): the numbers and the words.
@Suite("The Dock icon's progress and what it says at the end") struct DockProgressTests {
    @Test func theFractionIsHowFarTheRunHasGotAndNeverMoreThanFull() {
        #expect(DockProgress.fraction(step: 15_000, target: 30_000) == 0.5)
        #expect(DockProgress.fraction(step: 0, target: 30_000) == 0)
        #expect(DockProgress.fraction(step: 40_000, target: 30_000) == 1, "a step past the target is full")
        #expect(DockProgress.fraction(step: -5, target: 30_000) == 0)
        #expect(DockProgress.fraction(step: 20_000, target: 30_000, start: 10_000) == 0.5, "from a start step")
    }

    @Test func whatCannotBeToldIsNotMadeUp() {
        #expect(DockProgress.fraction(step: nil, target: 30_000) == nil)
        #expect(DockProgress.fraction(step: 100, target: nil) == nil)
        #expect(DockProgress.fraction(step: 100, target: 0) == nil)
        #expect(DockProgress.fraction(step: 100, target: 50, start: 50) == nil, "a target at the start is no run")
    }

    @Test func thePercentageRoundsDownSoNothingIsFullBeforeItIs() {
        #expect(DockProgress.percentText(0.5) == "50%" && DockProgress.percentText(0) == "0%" && DockProgress.percentText(1) == "100%")
        #expect(DockProgress.percentText(0.999) == "99%")
    }

    @Test func theBadgeSaysDoneForARunThatEndedAtASavePointAndFailedForTheRest() {
        #expect(DockProgress.endBadge(status: 0, signalled: false) == "Done")
        #expect(DockProgress.endBadge(status: 130, signalled: false) == "Done", "a stop at a save point")
        #expect(DockProgress.endBadge(status: 1, signalled: false) == "Failed")
        #expect(DockProgress.endBadge(status: 4, signalled: false) == "Failed")
        #expect(DockProgress.endBadge(status: 0, signalled: true) == "Failed", "a signal is not a clean end whatever the number")
    }

    @Test func aRunThePersonStoppedThemselvesIsNotNews() {
        #expect(!DockProgress.endIsNews(stopRequested: true, byTimer: false))
        #expect(DockProgress.endIsNews(stopRequested: false, byTimer: false))
        #expect(DockProgress.endIsNews(stopRequested: true, byTimer: true), "the timer's stop was not theirs")
    }
}
