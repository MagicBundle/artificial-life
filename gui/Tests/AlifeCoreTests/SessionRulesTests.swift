import Foundation
import Testing

@testable import AlifeCore

/// R3 and R1 of docs/APP.md (PR-E): what closing the window does while a run is going (owner decision D3 = C), and what the window says
/// about a run and a world sharing the GPU. Pure: every case without a window.
@Suite("Closing the window with a run going (D3) and the rules around it") struct SessionRulesTests {
    private func facts(run: Bool = true, name: String? = "exp-1", stopping: Bool = false, world: String? = nil) -> CloseFacts {
        CloseFacts(runGoing: run, runName: name, runStopping: stopping, world: world)
    }

    // MARK: Whether to ask

    @Test func withNoRunTheWindowClosesAndTheAppQuitsAsBefore() {
        for preference in ClosePreference.allCases {
            #expect(SessionRules.decideClose(facts(run: false, name: nil), preference: preference) == .closeAndQuit)
            #expect(SessionRules.decideClose(facts(run: false, name: nil, world: "mini3s_a"), preference: preference) == .closeAndQuit,
                    "only a world: it is saved and nothing is lost, so nothing is asked")
        }
    }

    @Test func aRunMakesTheWindowAskUnlessTheAnswerIsRemembered() {
        guard case .ask(let question) = SessionRules.decideClose(facts(), preference: .ask) else {
            Issue.record("a run going must ask")
            return
        }
        #expect(question.keepLabel == "Keep Running" && question.stopLabel == "Stop and Quit" && question.cancelLabel == "Cancel")
        #expect(SessionRules.decideClose(facts(), preference: .keep) == .keepRunning)
        #expect(SessionRules.decideClose(facts(), preference: .stop) == .stopAndQuit)
    }

    @Test func aRunThatIsAlreadyStoppingAsksNothingAndRemembersNothingElse() {
        for preference in ClosePreference.allCases {
            #expect(SessionRules.decideClose(facts(stopping: true), preference: preference) == .closeAndQuit,
                    "the quit then waits for its save point, as PR-A's does")
        }
    }

    @Test func anUnknownStoredValueIsTheQuestion() {
        #expect(ClosePreference(stored: nil) == .ask)
        #expect(ClosePreference(stored: "") == .ask)
        #expect(ClosePreference(stored: "always") == .ask)
        #expect(ClosePreference(stored: "keep") == .keep && ClosePreference(stored: "stop") == .stop && ClosePreference(stored: "ask") == .ask)
    }

    // MARK: The words

    @Test func theQuestionSaysWhatEachButtonDoesInPlainWords() {
        let q = SessionRules.question(facts(name: "planet-1-grow", world: nil))
        #expect(q.title == "The world for \u{2018}planet-1\u{2019} is still growing.")
        #expect(q.detail.contains("Keep Running closes the window and lets it carry on in the background"))
        #expect(q.detail.contains("bounces when it ends") && q.detail.contains("brings the window back"))
        #expect(q.detail.contains("Stop and Quit asks it to write a save point first (up to 120 seconds)"))
        #expect(q.detail.contains("Keep growing"), "the way back to a stopped run is the app's own word")
        #expect(!q.detail.contains("open world"), "no world, no sentence about one")
        #expect(SessionRules.question(facts(name: "exp-2")).title == "The run \u{2018}exp-2\u{2019} is still growing.")
        #expect(SessionRules.question(facts(name: nil)).title == "The run is still growing.")
    }

    @Test func anOpenWorldIsToldItIsSavedAndClosed() {
        let q = SessionRules.question(facts(world: "mini3s_a"))
        #expect(q.detail.contains("The open world \u{2018}mini3s_a\u{2019} is saved and closed when the window closes."))
        #expect(!SessionRules.question(facts(world: "")).detail.contains("open world"))
    }

    @Test func theSafeChoiceIsTheDefaultAndCancelIsNotRememberable() {
        // the first answer is the one that loses nothing and does what closing meant; the destructive one is never a default
        #expect(SessionRules.remembered(.keepRunning, remember: true) == .keep)
        #expect(SessionRules.remembered(.stopAndQuit, remember: true) == .stop)
        #expect(SessionRules.remembered(.cancel, remember: true) == nil)
        #expect(SessionRules.remembered(.keepRunning, remember: false) == nil)
        #expect(SessionRules.remembered(.stopAndQuit, remember: false) == nil)
    }

    @Test func eachPreferenceHasATitleAndAnEffectLineThatDiffer() {
        let titles = Set(ClosePreference.allCases.map(\.title)), effects = Set(ClosePreference.allCases.map(\.effect))
        #expect(titles.count == 3 && effects.count == 3)
        #expect(ClosePreference.keep.effect.contains("carries on in the background") && ClosePreference.stop.effect.contains("Keep growing"))
    }

    // MARK: R1

    @Test func aRunAndAWorldTogetherSayTheyShareTheGPU() {
        #expect(SessionRules.gpuNote(runGoing: true, worldOpen: true) == "A run is using the GPU; the world may be slower.")
        #expect(SessionRules.gpuNote(runGoing: true, worldOpen: false) == nil)
        #expect(SessionRules.gpuNote(runGoing: false, worldOpen: true) == nil)
    }

    @Test func theWorldThatWasOpenIsToldOnTheWayBack() {
        #expect(SessionRules.closedWorldNote("w").contains("\u{2018}w\u{2019} was saved and closed"))
        #expect(SessionRules.closedWorldNote("w").contains("Press Play"))
    }

    // MARK: Only the alife window

    @Test func onlyTheAlifeWindowAsksAndTheOldAppsQuitWithTheirLastWindow() {
        #expect(LaunchMode.alife.plan.asksWhenClosingWithRun)
        #expect(!LaunchMode.lab.plan.asksWhenClosingWithRun && !LaunchMode.world.plan.asksWhenClosingWithRun)
        for mode in LaunchMode.allCases { #expect(mode.plan.quitsAfterLastWindow, "the pinned default stays; the alife app overrides it when it asks") }
    }
}
