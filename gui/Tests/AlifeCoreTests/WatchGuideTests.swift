import Foundation
import Testing

@testable import AlifeCore

@Suite struct WatchGuideTests {
    private func row(_ kind: WorldKind = .research, shape: WorldShape = .planet, state: WorldState = .saved, step: Int? = 30_000, name: String = "r") -> LibraryRow {
        LibraryRow(name: name, kind: kind, shape: shape, state: state, step: step, lastSaved: Date())
    }

    // MARK: The form

    @Test func anOpenWorldIsWhatWatchShowsWhateverElseWasChosen() {
        #expect(WatchGuide.form(worldOpen: true, worldEnded: false, observed: "r") == .played)
        #expect(WatchGuide.form(worldOpen: true, worldEnded: true, observed: nil) == .played)
        #expect(WatchGuide.form(worldOpen: false, worldEnded: false, observed: "r") == .research)
        // a played world that ended keeps its page only until a research world is chosen to watch
        #expect(WatchGuide.form(worldOpen: false, worldEnded: true, observed: nil) == .played)
        #expect(WatchGuide.form(worldOpen: false, worldEnded: true, observed: "r") == .research)
        #expect(WatchGuide.form(worldOpen: false, worldEnded: false, observed: nil) == nil)
    }

    @Test func theChipSaysWhichWatchItIs() {
        #expect(WatchForm.played.chip == "Live" && WatchForm.research.chip == "Observer")
        #expect(WatchForm.research.chipMeaning.contains("cannot be played"))
        #expect(WatchForm.played.chipMeaning.contains("powers"))
    }

    // MARK: Plain words for what a research world can and cannot do

    @Test func aResearchWorldSaysWhatItCannotDoAndHowToPlayIt() {
        #expect(WatchGuide.note(forShape: .planet) == WatchGuide.observerNote && WatchGuide.observerNote.contains("make a planet"))
        #expect(WatchGuide.note(forShape: .box) == WatchGuide.observerNote && WatchGuide.note(forShape: nil) == WatchGuide.observerNote)
        #expect(WatchGuide.note(forShape: .flat) == WatchGuide.flatObserverNote && WatchGuide.flatObserverNote.contains("cannot become a planet"))
        #expect(WatchGuide.researchCannot.joined().contains("powers") && WatchGuide.researchCannot.joined().contains("chronicle"))
        #expect(WatchGuide.researchCan.joined().contains("Follow") && WatchGuide.researchCan.joined().contains("Re-simulate"))
        #expect(WatchGuide.playedCan.joined().contains("powers"))
    }

    // MARK: Why a world cannot be watched

    @Test func aResearchWorldCanBeWatchedUnlessThereIsAReason() {
        #expect(WatchGuide.reasonToWatch(row(), openWorld: nil, runGrowing: nil) == nil)
        #expect(WatchGuide.reasonToWatch(row(), openWorld: "p", runGrowing: nil) == "Close p first: Watch shows one world at a time.")
        #expect(WatchGuide.reasonToWatch(row(step: nil), openWorld: nil, runGrowing: nil) == "No save point to look at yet.")
        // the run this app grows is watched from its first moment, before its first save point is on the disk
        #expect(WatchGuide.reasonToWatch(row(step: nil, name: "g"), openWorld: nil, runGrowing: "g") == nil)
        #expect(WatchGuide.reasonToWatch(row(step: nil, name: "g"), openWorld: nil, runGrowing: "h") != nil)
        #expect(WatchGuide.reasonToWatch(row(state: .saving), openWorld: nil, runGrowing: nil) == "Saving…")
        var placeholder = row(step: nil)
        placeholder.isPlaceholder = true
        #expect(WatchGuide.reasonToWatch(placeholder, openWorld: nil, runGrowing: nil) == "It opens in Watch when it is ready.")
    }

    @Test func aPlayedWorldIsWatchedWhilePlayingOnly() {
        #expect(WatchGuide.reasonToWatch(row(.played), openWorld: "r", runGrowing: nil) == nil)
        #expect(WatchGuide.reasonToWatch(row(.played), openWorld: nil, runGrowing: nil)?.contains("Press Play") == true)
        #expect(WatchGuide.reasonToWatch(row(.played), openWorld: "other", runGrowing: nil)?.contains("Press Play") == true)
    }

    // MARK: The observer's stream

    private func facts(_ stream: WatchGuide.ObserverFacts.Stream = .idle, save: Bool = true, growing: Bool = false, on: String? = nil, follows: Bool = false,
                       failures: Int = 0) -> WatchGuide.ObserverFacts {
        .init(world: "w", hasSavePoint: save, growing: growing, streamWorld: on, follows: follows, stream: stream, failures: failures)
    }

    @Test func theStreamStartsWhenTheWorldHasASavePointAndFollowsAGrowingRun() {
        #expect(WatchGuide.observerStep(facts()) == .start(follow: false))
        #expect(WatchGuide.observerStep(facts(growing: true)) == .start(follow: true))
        #expect(WatchGuide.observerStep(facts(save: false)) == .wait(WatchGuide.waitingText))
        #expect(WatchGuide.observerStep(facts(save: false, growing: true)) == .wait(WatchGuide.waitingText))
    }

    @Test func aStreamOfAnotherWorldIsReplaced() {
        #expect(WatchGuide.observerStep(facts(.live, growing: true, on: "other")) == .start(follow: true))
        #expect(WatchGuide.observerStep(facts(.starting, on: "other")) == .start(follow: false))
        #expect(WatchGuide.observerStep(facts(.live, save: false, on: "other")) == .wait(WatchGuide.waitingText))
        // (no stream at all is the idle case, with the stale name ignored)
        #expect(WatchGuide.observerStep(facts(.idle, on: "other")) == .start(follow: false))
    }

    @Test func aStreamThatIsStartingOrLiveIsLeftAlone() {
        #expect(WatchGuide.observerStep(facts(.starting, growing: true, on: "w", follows: true)) == .none)
        #expect(WatchGuide.observerStep(facts(.live, growing: true, on: "w", follows: true)) == .none)
        #expect(WatchGuide.observerStep(facts(.live, growing: false, on: "w", follows: false)) == .none)
        // a re-simulation the player chose while the run grows stays
        #expect(WatchGuide.observerStep(facts(.live, growing: true, on: "w", follows: false)) == .none)
    }

    @Test func aFollowedRunThatEndedIsShownByReSimulatingIt() {
        #expect(WatchGuide.observerStep(facts(.live, growing: false, on: "w", follows: true)) == .restart(follow: false))
        #expect(WatchGuide.observerStep(facts(.ended, growing: false, on: "w", follows: true)) == .start(follow: false))
        #expect(WatchGuide.observerStep(facts(.ended, growing: true, on: "w", follows: true)) == .start(follow: true))
        #expect(WatchGuide.observerStep(facts(.ended, growing: false, on: "w", follows: false)) == .none)
    }

    @Test func aStartThatFailsIsTriedAgainAFewTimesAndThenLeftToThePlayer() {
        #expect(WatchGuide.startRetries == 3)
        for failures in 0..<WatchGuide.startRetries {
            #expect(WatchGuide.observerStep(facts(.failed, growing: true, on: "w", failures: failures)) == .start(follow: true))
        }
        #expect(WatchGuide.observerStep(facts(.failed, growing: true, on: "w", failures: 3)) == .none)
        #expect(WatchGuide.observerStep(facts(.failed, growing: true, on: "w", failures: 9)) == .none)
        #expect(WatchGuide.observerStep(facts(.failed, save: false, on: "w", failures: 1)) == .wait(WatchGuide.waitingText))
    }

    @Test func thePictureSaysWhatItWaitsFor() {
        #expect(WatchGuide.streamText(.live, growing: true, failure: nil) == nil)
        #expect(WatchGuide.streamText(.starting, growing: false, failure: nil)?.contains("engine") == true)
        #expect(WatchGuide.streamText(.idle, growing: true, failure: nil) == WatchGuide.waitingText)
        #expect(WatchGuide.streamText(.idle, growing: false, failure: nil) == nil)
        #expect(WatchGuide.streamText(.failed, growing: false, failure: "boom") == "boom")
        #expect(WatchGuide.streamText(.failed, growing: false, failure: nil) == "The viewer stopped.")
        #expect(WatchGuide.streamText(.ended, growing: false, failure: nil) == "The viewer ended.")
    }

    @Test func theSubtitleSaysHowThePictureIsMade() {
        #expect(WatchGuide.observerLine(follows: nil, step: nil) == "Observer")
        #expect(WatchGuide.observerLine(follows: true, step: nil) == "Observer · following the run")
        #expect(WatchGuide.observerLine(follows: false, step: 1_234) == "Observer · re-simulating · step 1,234")
    }
}
