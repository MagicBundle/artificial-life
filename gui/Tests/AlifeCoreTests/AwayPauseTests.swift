import Foundation
import Testing

@testable import AlifeCore

/// R4 of docs/APP.md (owner decision D4 = A): leaving the Watch page pauses a played world and coming back continues it, and the app only
/// ever undoes a pause of its own. Every state of the world against leaving and returning.
@Suite("Leaving Watch pauses the world, and only the app's pause is undone (D4)") struct AwayPauseTests {
    private func running(_ world: String = "w", rate: Float = 1) -> AwayPause.Facts {
        AwayPause.Facts(world: world, live: true, rate: rate, reason: .running)
    }
    private func paused(by reason: PlayClock.Reason, _ world: String = "w") -> AwayPause.Facts {
        AwayPause.Facts(world: world, live: true, rate: 0, reason: reason)
    }

    // MARK: Leaving

    @Test func leavingWhileItRunsPausesItOnceAndHoldsThePause() {
        var pause = AwayPause()
        #expect(pause.leftWatch(running()) == .pause)
        #expect(pause.hold == .away(world: "w"))
        #expect(pause.leftWatch(running()) == nil, "a second look at the same visit does not pause again")
    }

    @Test func aWorldAtTheMaxRateIsRunningToo() {
        var pause = AwayPause()
        #expect(pause.leftWatch(running(rate: -1)) == .pause)
    }

    @Test func aWorldThePlayerPausedIsLeftAlone() {
        var pause = AwayPause()
        #expect(pause.leftWatch(paused(by: .userPause)) == nil)
        #expect(pause.hold == .none)
        #expect(pause.returned(paused(by: .userPause), pickUp: true) == nil, "and it is not run when they come back")
    }

    @Test func aPauseThatIsOnItsWayIsNotPausedAgain() {
        var pause = AwayPause()
        var f = running()
        f.requestedPause = true
        #expect(pause.leftWatch(f) == nil)
    }

    @Test func aWorldAtAChapterOrAnAlertOrItsEndIsLeftAloneAndStaysSo() {
        for reason in [PlayClock.Reason.chapter, .alert, .endOfSteps, .error] {
            var pause = AwayPause()
            #expect(pause.leftWatch(paused(by: reason)) == nil, "\(reason)")
            #expect(pause.hold == .none)
            #expect(pause.returned(paused(by: reason), pickUp: true) == nil, "\(reason): the card it shows has not been read")
        }
    }

    @Test func aWorldAboutToStopByItselfIsNotPausedUnderIt() {
        // a chapter or an alert announces its card while the save point is written: the pause that follows is the world's own
        var chapter = running()
        chapter.reason = .chapter
        chapter.chapterPauseArmed = true
        var alert = running()
        alert.reason = .alert
        alert.alertPauseArmed = true
        var pause = AwayPause()
        #expect(pause.leftWatch(chapter) == nil && pause.leftWatch(alert) == nil)
        // with that pause switched off the chapter's reason stays on a world that goes on running: that one is paused
        var unarmed = running()
        unarmed.reason = .chapter
        #expect(pause.leftWatch(unarmed) == .pause)
    }

    @Test func noClockNoWorldNoPause() {
        var pause = AwayPause()
        #expect(pause.leftWatch(AwayPause.Facts(world: "w", live: true, rate: nil, reason: nil)) == nil)
        #expect(pause.leftWatch(AwayPause.Facts(world: nil, live: false, rate: 1, reason: .running)) == nil)
        #expect(pause.leftWatch(AwayPause.Facts(world: "w", live: false, rate: 1, reason: .running)) == nil, "a world that is not live")
    }

    // MARK: Returning

    @Test func comingBackRunsTheWorldTheAppPausedWhenPickingUp() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        #expect(pause.returned(paused(by: .userPause), pickUp: true) == .resume)
        #expect(pause.hold == .none)
        #expect(pause.returned(paused(by: .userPause), pickUp: true) == nil, "once")
    }

    @Test func notPickingUpLeavesItPausedAndSaysWhy() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        #expect(pause.returned(paused(by: .userPause), pickUp: false) == nil)
        #expect(pause.hold == .left(world: "w"))
        // the player presses Run: the note goes
        #expect(pause.clockChanged(running(), watchActive: true) == nil)
        #expect(pause.hold == .none)
    }

    @Test func aChapterThatTookTheWorldFirstIsNotResumed() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        // the world reached its chapter end before it read the app's pause: its card has not been read, and `run` would acknowledge it
        #expect(pause.returned(paused(by: .chapter), pickUp: true) == nil)
        #expect(pause.hold == .none)
        var alert = AwayPause()
        _ = alert.leftWatch(running())
        #expect(alert.returned(paused(by: .alert), pickUp: true) == nil)
    }

    @Test func aPauseNotYetConfirmedByTheClockIsStillTheAppsAndIsRun() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        var f = running()
        f.requestedPause = true  // the player came back before the world answered
        #expect(pause.returned(f, pickUp: true) == .resume)
    }

    @Test func aWorldThatIsRunningAgainNeedsNothing() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        #expect(pause.returned(running(), pickUp: true) == nil, "something else ran it (the pause was refused)")
        #expect(pause.hold == .none)
    }

    @Test func aDifferentWorldIsNotTheOneThatWasPaused() {
        var pause = AwayPause()
        _ = pause.leftWatch(running("a"))
        #expect(pause.returned(paused(by: .userPause, "b"), pickUp: true) == nil)
        #expect(pause.hold == .none)
        var gone = AwayPause()
        _ = gone.leftWatch(running("a"))
        #expect(gone.returned(AwayPause.Facts(world: nil, live: false, rate: nil, reason: nil), pickUp: true) == nil)
    }

    // MARK: A clock while away

    @Test func aWorldThatStartsRunningWhileAwayIsPaused() {
        var pause = AwayPause()
        #expect(pause.clockChanged(running(), watchActive: false) == .pause, "it went live while the player was on another page")
        #expect(pause.hold == .away(world: "w"))
        #expect(pause.clockChanged(running(), watchActive: true) == nil, "on Watch nothing is paused")
        var watching = AwayPause()
        #expect(watching.clockChanged(running(), watchActive: true) == nil)
    }

    @Test func theWorldTakingOverGivesTheHoldBack() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        #expect(pause.clockChanged(paused(by: .chapter), watchActive: false) == nil)
        #expect(pause.hold == .none, "a chapter took it: the pause is not the app's any more")
        var other = AwayPause()
        _ = other.leftWatch(running("a"))
        #expect(other.clockChanged(running("b"), watchActive: true) == nil)
        #expect(other.hold == .none, "another world")
    }

    @Test func aCancelledPauseHoldsNothing() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        pause.cancelPause()
        #expect(pause.hold == .none)
        var left = AwayPause()
        _ = left.leftWatch(running())
        _ = left.returned(paused(by: .userPause), pickUp: false)
        left.cancelPause()
        #expect(left.hold == .left(world: "w"), "a note is not a pause that can be taken back")
    }

    @Test func leavingAgainRefreshesTheNote() {
        var pause = AwayPause()
        _ = pause.leftWatch(running())
        _ = pause.returned(paused(by: .userPause), pickUp: false)
        #expect(pause.hold == .left(world: "w"))
        #expect(pause.leftWatch(paused(by: .userPause)) == nil, "paused already: nothing to pause")
        #expect(pause.hold == .none)
    }

    // MARK: The words of the sidebar row

    @Test func theSidebarRowSaysWhyTheWorldIsPaused() {
        let away = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 0, looking: false, pause: .away(world: "w"))
        #expect(away.text == "w · paused: you left" && !away.needsPlayer, "the app did it: nothing waits for the player")
        let left = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 0, looking: true, pause: .left(world: "w"))
        #expect(left.text == "w · paused: you left")
        let leftInBackground = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 0, looking: false, pause: .left(world: "w"))
        #expect(leftInBackground.text == "w · paused: you left")
        // a chapter or an alert is told as that, whatever the app did
        let chapter = WatchCue.cue(phase: .live, world: "w", stop: .chapter, chapter: 3, unread: 0, looking: false, pause: .none)
        #expect(chapter.text == "w · chapter 3 is waiting" && chapter.needsPlayer)
        let live = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 0, looking: true)
        #expect(live.text == "w · live")
        let unread = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 2, looking: false, pause: .away(world: "w"))
        #expect(unread.text == "w · paused: you left · 2 unread alerts" && unread.needsPlayer)
    }

    @Test func theActivityPageSaysTheSameAndSharesTheGPUNote() {
        let item = ActivityGuide.items(ActivityGuide.Facts(worldName: "w", worldPausedAway: true)).first!
        #expect(item.detail == "Paused while you are away from Watch. It continues when you go back.")
        let stays = ActivityGuide.items(ActivityGuide.Facts(worldName: "w", worldPausedAway: true, picksUp: false)).first!
        #expect(stays.detail.contains("stays paused until you press Run in Watch"))
        let both = ActivityGuide.items(ActivityGuide.Facts(runName: "r", worldName: "w")).last!
        #expect(both.detail.hasSuffix("A run is using the GPU; the world may be slower."))
        #expect(ActivityGuide.items(ActivityGuide.Facts(worldName: "w")).first!.detail == "Open in Watch. Closing it saves first.")
    }
}
