import Foundation
import Testing

@testable import AlifeCore

/// The state table of docs/APP.md §3.2, state by state: Stop, Cancel, failure, the window closed, the app quit. The machine is pure, so every
/// row of the table is one test, and the rules it states are checked over all states at the end.
@Suite struct PipelinePlanTests {
    typealias Plan = PipelinePlan
    static let here = Plan.Surroundings(visible: true, openWorld: nil, nameTaken: false, engineReady: true)

    func grow(target: Int? = 30_000) -> Plan {
        Plan(mode: .growThenPlay, name: "planet-1", growName: "planet-1-grow", targetSteps: target, expectedPlanetSeconds: 11)
    }

    func exit(_ status: Int32 = 0, signalled: Bool = false, stop: Bool = false, problem: String? = nil) -> Plan.Event {
        .growExited(.init(status: status, signalled: signalled, stopRequested: stop, problem: problem))
    }

    // MARK: Stage 1

    @Test func aNewPipelineIsGrowingAndHasAStopButton() {
        let p = grow()
        #expect(p.state == .growing(stopping: false))
        let b = p.banner(elapsed: 0)
        #expect(b?.buttons == [.stop] && b?.tone == .working)
        #expect(b?.title == "Growing planet-1")
        #expect(b?.next?.contains("Next it becomes a planet") == true)
    }

    @Test func progressShowsTheStepTheFractionAndTheTimeLeft() {
        var p = grow()
        _ = p.apply(.progress(step: 12_300, stepsPerSecond: 300), in: Self.here)
        let b = p.banner(elapsed: 40)
        #expect(b?.detail.hasPrefix("Step 12,300 of 30,000") == true)
        #expect(b?.detail.contains("left") == true)
        #expect(abs((b?.fraction ?? 0) - 0.41) < 0.001)
        _ = p.apply(.progress(step: 100, stepsPerSecond: nil), in: Self.here)
        #expect(p.step == 12_300, "the step only grows")
    }

    @Test func stopIsOneSigintAndTheRunExitingParksThePipelineNeverContinuing() {
        var p = grow()
        #expect(p.apply(.stopPressed, in: Self.here) == [.stopGrowing])
        #expect(p.state == .growing(stopping: true))
        #expect(p.banner(elapsed: 3)?.buttons == [], "a second Stop has nothing to press")
        #expect(p.apply(.stopPressed, in: Self.here) == [], "and a second press does nothing")
        _ = p.apply(.progress(step: 9_100, stepsPerSecond: 300), in: Self.here)
        let effects = p.apply(exit(130, stop: true), in: Self.here)
        #expect(effects == [], "never starts the planet by itself after a Stop, with the window in front")
        #expect(p.state == .parked(.stopped, step: 9_100))
        let b = p.banner(elapsed: 0)
        #expect(b?.title == "Growing stopped at step 9,100")
        #expect(b?.buttons == [.makePlanetFromHere, .keepGrowing, .done])
        #expect(b?.detail.contains("The grown world is saved") == true)
    }

    @Test func aPipelineThatWasToldToStopKnowsItWhateverTheExitSays() {
        // (the exit that follows a Stop need not carry the flag: the pipeline asked, and it never goes on after asking)
        var p = grow()
        _ = p.apply(.stopPressed, in: Self.here)
        #expect(p.apply(.growExited(.init(status: 0, stopRequested: false)), in: Self.here) == [])
        #expect(p.state == .parked(.stopped, step: 0))
        var q = Plan(mode: .keepGrowing, name: "r", growName: "r", targetSteps: 9)
        _ = q.apply(.stopPressed, in: Self.here)
        _ = q.apply(.growExited(.init(status: 0)), in: Self.here)
        #expect(q.state == .finished(stopped: true))
    }

    @Test func aSigintFromOutsideLeavesASavePointAndParksToo() {
        var p = grow()
        #expect(p.apply(exit(130), in: Self.here) == [])
        #expect(p.state == .parked(.stopped, step: 0))
    }

    @Test func aFinishedRunThePersonCanSeeStartsThePlanet() {
        var p = grow()
        let effects = p.apply(exit(0), in: Self.here)
        #expect(effects == [.startMaking])
        #expect(p.state == .making(cancelling: false))
        #expect(p.step == 30_000)
    }

    @Test func aFinishedRunNobodyCanSeeParksBouncesOnceAndStartsNothing() {
        var p = grow()
        let effects = p.apply(exit(0), in: .init(visible: false))
        #expect(effects == [.bounce])
        #expect(p.state == .parked(.notVisible, step: 30_000))
        let b = p.banner(elapsed: 0)
        #expect(b?.buttons == [.makePlanet, .keepGrowing, .done])
        #expect(b?.detail.contains("not in front") == true)
        #expect(p.apply(.progress(step: 1, stepsPerSecond: 1), in: .init(visible: true)) == [], "coming back to the window starts nothing")
        #expect(p.state == .parked(.notVisible, step: 30_000))
    }

    @Test func aFinishedRunWithAnotherWorldOpenParksAndSaysWhichToClose() {
        var p = grow()
        let effects = p.apply(exit(0), in: .init(visible: true, openWorld: "mini3s_a"))
        #expect(effects == [])
        #expect(p.state == .parked(.worldOpen("mini3s_a"), step: 30_000))
        let b = p.banner(elapsed: 0)
        #expect(b?.detail.contains("Close mini3s_a") == true && b?.buttons == [.makePlanet, .done])
    }

    @Test func aNameThatAppearedWhileItGrewParksWithAWayToUseAnother() {
        var p = grow()
        _ = p.apply(exit(0), in: .init(visible: true, nameTaken: true))
        #expect(p.state == .parked(.nameTaken, step: 30_000))
        #expect(p.banner(elapsed: 0)?.buttons == [.makeAgainNewName, .done])
        // pressing Make planet again while the name is taken does not start it
        var q = grow()
        _ = q.apply(exit(130, stop: true), in: Self.here)
        #expect(q.apply(.makePlanetPressed, in: .init(visible: true, nameTaken: true)) == [])
        #expect(q.state == .parked(.nameTaken, step: 0))
    }

    @Test func aWorldThatDiedOutIsNotAFinishedGrowthAndStartsNoPlanetInAnyMode() {
        for mode in [Plan.Mode.growThenPlay, .growOnly, .keepGrowing] {
            var p = Plan(mode: mode, name: "planet-1", growName: "planet-1-grow", targetSteps: 30_000, expectedPlanetSeconds: 11)
            let effects = p.apply(.growExited(.init(status: 0, savedStep: 4_200, worldStatus: "extinct")), in: Self.here)
            #expect(effects == [], "\(mode)")
            #expect(p.state == .diedOut(step: 4_200), "\(mode): not the 30,000 it was asked to reach")
            let b = p.banner(elapsed: 0)
            #expect(b?.title == "The population died out at step 4,200" && b?.tone == .attention && p.isOver)
            #expect(b?.buttons.contains(.makePlanet) == false && b?.buttons.contains(.makePlanetFromHere) == false)
        }
        // a Stop that arrives as the world dies out is still told as what happened
        var q = grow()
        _ = q.apply(.stopPressed, in: Self.here)
        _ = q.apply(.growExited(.init(status: 0, stopRequested: true, savedStep: 4_200, worldStatus: "extinct")), in: Self.here)
        #expect(q.state == .diedOut(step: 4_200))
        // other statuses behave as before
        var r = grow()
        _ = r.apply(.growExited(.init(status: 0, worldStatus: "done")), in: Self.here)
        #expect(r.state == .making(cancelling: false))
        var done = r; _ = done.apply(.donePressed, in: Self.here); _ = done
    }

    @Test func aPlanetsEngineThatEndsByItselfIsToldWhyAndOnlyACancelIsCalledCancelled() {
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        _ = p.apply(.makeEnded("play: the world is extinct at step 4200"), in: Self.here)
        #expect(p.state == .diedOut(step: 4_200))
        var q = grow()
        _ = q.apply(exit(0), in: Self.here)
        _ = q.apply(.makeEnded("The planet's engine ended by itself before the planet was ready."), in: Self.here)
        #expect(q.state == .failed(.making("The planet's engine ended by itself before the planet was ready.")))
        #expect(q.banner(elapsed: 0)?.title == "planet-1 could not be made")
        var c = grow()
        _ = c.apply(exit(0), in: Self.here)
        _ = c.apply(.cancelPressed, in: Self.here)
        _ = c.apply(.makeEnded("anything the engine said while it was stopped"), in: Self.here)
        #expect(c.state == .cancelled, "nobody stopped it in the other case; here somebody did")
        #expect(PipelineWords.diedOutStep(in: "play: the world is extinct at step 4200") == 4_200 && PipelineWords.diedOutStep(in: "ended") == nil)
        #expect(PipelineWords.planetEnded("The world ended before it sent anything: play: the world is extinct at step 77").contains("extinct at step 77"))
        #expect(PipelineWords.planetEnded("The world ended before it sent anything").hasPrefix("The planet's engine ended by itself"))
    }

    @Test func aFailureThatLeftNoSavePointOffersNoPlanetAndNoKeepGrowing() {
        var p = grow()
        _ = p.apply(.growExited(.init(status: 4, problem: "boom", savePoint: false)), in: Self.here)
        #expect(p.banner(elapsed: 0)?.buttons == [.showLog, .tryAgainNewName, .done])
    }

    @Test func nothingOfTheDefaultPathNamesTheGrowDirectoryOrAnExceptionClass() {
        var p = grow()
        _ = p.apply(.progress(step: 12_000, stepsPerSecond: 300), in: Self.here)
        var texts: [String] = []
        func add(_ b: Plan.Banner?) { if let b { texts += [b.title, b.detail, b.next ?? ""] } }
        add(p.banner(elapsed: 1))
        _ = p.apply(.stopPressed, in: Self.here); add(p.banner(elapsed: 1))
        _ = p.apply(exit(130, stop: true), in: Self.here); add(p.banner(elapsed: 1))
        for why in [Plan.Why.notVisible, .worldOpen("x"), .nameTaken, .engineMissing] { var q = p; q.state = .parked(why, step: 5); add(q.banner(elapsed: 0)) }
        var f = grow(); _ = f.apply(.growExited(.init(status: 2, problem: "bad")), in: Self.here); add(f.banner(elapsed: 0))
        var m = grow(); m.state = .failed(.making("x")); add(m.banner(elapsed: 0))
        var c = grow(); c.state = .cancelled; add(c.banner(elapsed: 0))
        var d = grow(); d.state = .diedOut(step: 5); add(d.banner(elapsed: 0))
        for text in texts { #expect(!text.contains("-grow"), "\"\(text)\"") }
    }

    @Test func aMissingEngineParksInsteadOfFailingSilently() {
        var p = grow()
        _ = p.apply(exit(0), in: .init(visible: true, engineReady: false))
        #expect(p.state == .parked(.engineMissing, step: 30_000))
        #expect(p.banner(elapsed: 0)?.detail.contains("Settings") == true)
    }

    @Test(arguments: [Int32(1), 2, 3, 4, 9])
    func anyOtherExitIsAFailureThatKeepsWhatWasWritten(status: Int32) {
        var p = grow()
        let effects = p.apply(.growExited(.init(status: status, problem: "ValueError: no", savedStep: 12_000)), in: Self.here)
        #expect(effects == [])
        #expect(p.state == .failed(.growing("ValueError: no")))
        let b = p.banner(elapsed: 0)
        #expect(b?.tone == .failure && b?.title == "Growing failed" && b?.detail == "ValueError: no")
        #expect(b?.buttons == [.showLog, .makePlanet, .keepGrowing, .tryAgainNewName, .done], "a save point is left: the planet can still be made from it")
        #expect(b?.next == "The grown world is kept as it was written.")
        #expect(p.isOver)
    }

    @Test func aFailureWithoutAMessageStillSaysSomethingPlain() {
        var p = grow()
        _ = p.apply(exit(2), in: Self.here)
        #expect(p.state == .failed(.growing("The engine did not accept these settings.")))
        var q = grow()
        _ = q.apply(exit(15, signalled: true), in: Self.here)
        #expect(q.state == .failed(.growing("The engine was ended before it could save.")), "force stop (SIGTERM) is a failure, not a stop")
    }

    @Test func aStopThatTheEngineAnswersWithAnErrorIsAFailure() {
        var p = grow()
        _ = p.apply(.stopPressed, in: Self.here)
        _ = p.apply(exit(1, stop: true, problem: "boom"), in: Self.here)
        #expect(p.state == .failed(.growing("boom")))
    }

    @Test func aStopInTheEnginesFirstSecondsEndsOnTheSignalAndIsAStopOnlyIfAnythingWasSaved() {
        // (Python takes SIGINT before it has installed its handler as the signal itself, not as an exit code)
        var saved = grow()
        _ = saved.apply(.stopPressed, in: Self.here)
        #expect(saved.apply(.growExited(.init(status: 2, signalled: true, stopRequested: true, savePoint: true)), in: Self.here) == [])
        #expect(saved.state == .parked(.stopped, step: 0))
        var nothing = grow()
        _ = nothing.apply(.stopPressed, in: Self.here)
        #expect(nothing.apply(.growExited(.init(status: 2, signalled: true, stopRequested: true, savePoint: false)), in: Self.here) == [])
        #expect(nothing.state == .failed(.nothingSaved))
        let b = nothing.banner(elapsed: 1)
        #expect(b?.title == "Stopped before anything was saved" && b?.tone == .attention && b?.buttons == [.tryAgainNewName, .done])
        // a SIGINT that nobody asked for is a failure, as any other signal
        var unasked = grow()
        _ = unasked.apply(.growExited(.init(status: 2, signalled: true, stopRequested: false, savePoint: true)), in: Self.here)
        #expect(unasked.state == .failed(.growing("The engine was ended before it could save.")))
        // a research run that is stopped so early is simply stopped
        var research = Plan(mode: .growOnly, name: "r", growName: "r", targetSteps: 5)
        _ = research.apply(.growExited(.init(status: 2, signalled: true, stopRequested: true, savePoint: true)), in: Self.here)
        #expect(research.state == .finished(stopped: true))
    }

    // MARK: Parked

    @Test func makingThePlanetFromAParkedPipelineStartsIt() {
        var p = grow()
        _ = p.apply(exit(130, stop: true), in: Self.here)
        #expect(p.apply(.makePlanetPressed, in: Self.here) == [.startMaking])
        #expect(p.state == .making(cancelling: false))
    }

    @Test func whatBlockedItStillBlocksItWhenTheButtonIsPressed() {
        var p = grow()
        _ = p.apply(exit(0), in: .init(visible: false))
        // by hand the window is in front by definition
        #expect(p.apply(.makePlanetPressed, in: .init(visible: false)) == [.startMaking])
        var q = grow()
        _ = q.apply(exit(0), in: .init(visible: true, openWorld: "w"))
        #expect(q.apply(.makePlanetPressed, in: .init(visible: true, openWorld: "w")) == [])
        #expect(q.state == .parked(.worldOpen("w"), step: 30_000))
        #expect(q.apply(.makePlanetPressed, in: Self.here) == [.startMaking])
    }

    @Test func keepGrowingFromAParkedPipelineStartsTheRunAgain() {
        var p = grow()
        _ = p.apply(exit(130, stop: true), in: Self.here)
        #expect(p.apply(.keepGrowingPressed, in: Self.here) == [.resumeGrowing])
        #expect(p.state == .growing(stopping: false))
        // and a missing engine parks it with the reason instead
        var q = grow()
        _ = q.apply(exit(130, stop: true), in: Self.here)
        #expect(q.apply(.keepGrowingPressed, in: .init(engineReady: false)) == [])
        #expect(q.state == .parked(.engineMissing, step: 0))
    }

    @Test func doneEndsAParkedPipelineAndNothingStartsLater() {
        var p = grow()
        _ = p.apply(exit(130, stop: true), in: Self.here)
        #expect(p.apply(.donePressed, in: Self.here) == [])
        #expect(p.state == .dismissed && p.isOver && p.banner(elapsed: 0) == nil)
        #expect(p.apply(.makePlanetPressed, in: Self.here) == [], "after Done a stale button starts nothing")
    }

    // MARK: Stage 2

    @Test func theProgressOfThePlanetIsInWordsWithAFractionAndTheTimeLeft() {
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        _ = p.apply(.makeProgress(["houses: rebuilding the exact houses of runs/x at step 30000"]), in: Self.here)
        #expect(p.banner(elapsed: 1)?.detail.hasPrefix("Rebuilding the family lines") == true && p.banner(elapsed: 1)?.fraction == nil)
        _ = p.apply(.makeProgress([
            "houses: rebuilding the exact houses of runs/x at step 30000", "run runs/x: N=1024 | steps 30000 → 60000",
            "weather spin-up: 5040/10000 steps",
        ]), in: Self.here)
        let b = p.banner(elapsed: 5)
        #expect(b?.detail.hasPrefix("Warming up the weather (5,040 of 10,000 steps)") == true)
        #expect(b?.detail.contains("left") == false, "no precision that was not measured")
        #expect(abs((b?.fraction ?? 0) - 0.504) < 0.001, "only the warm-up counts its own steps")
        #expect(b?.buttons == [.cancel] && b?.next == "Then it opens in Watch.")
        #expect(b?.title == "Making planet-1")
    }

    @Test func theLastWarmUpLineIsOpening() {
        #expect(PlayProgress.parse(["weather spin-up: 10000/10000 steps"]).phase == .opening)
        #expect(PlayProgress.parse([]).phase == .starting && PlayProgress.parse([]).fraction == nil)
        #expect(PlayProgress.parse(["something else"]).phase == .starting)
        #expect(PlayProgress.parse(["weather spin-up: 5/0 steps"]).phase == .starting, "a zero total is not a fraction")
    }

    @Test func theWorldGoingLiveEndsThePipelineWithAWatchButton() {
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        _ = p.apply(.makeLive, in: Self.here)
        #expect(p.state == .live && p.isOver)
        #expect(p.banner(elapsed: 0)?.buttons == [.watch, .done])
    }

    @Test func aFailureOfThePlanetKeepsTheGrownWorldAndOffersTheNameAgain() {
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        _ = p.apply(.makeFailed("alife play found no usable checkpoint"), in: Self.here)
        let b = p.banner(elapsed: 0)
        #expect(p.state == .failed(.making("alife play found no usable checkpoint")))
        #expect(b?.title == "planet-1 could not be made")
        #expect(b?.next == "The grown world is kept, so only the planet needs making again.")
        #expect(b?.buttons == [.showLog, .makeAgainNewName, .done])
    }

    @Test func cancelIsOneStopAndTheProcessEndingCancelsThePipeline() {
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        #expect(p.apply(.cancelPressed, in: Self.here) == [.stopMaking])
        #expect(p.state == .making(cancelling: true))
        #expect(p.apply(.cancelPressed, in: Self.here) == [], "once")
        #expect(p.banner(elapsed: 2)?.buttons == [] && p.banner(elapsed: 2)?.detail == "Cancelling.")
        _ = p.apply(.makeEnded(nil), in: Self.here)
        #expect(p.state == .cancelled)
        let b = p.banner(elapsed: 0)
        #expect(b?.title == "planet-1 was not finished" && b?.detail.contains("Unfinished") == true)
        #expect(b?.next == "The grown world is kept.")
        #expect(b?.buttons == [.makeAgainNewName, .done])
    }

    @Test func theWindowClosedWhileThePlanetIsBeingMadeEndsItAsCancelled() {
        // (closing the window stops the world: PlayController.stop; the process ends without the world ever being live)
        var p = grow()
        _ = p.apply(exit(0), in: Self.here)
        _ = p.apply(.makeEnded(nil), in: .init(visible: false))
        #expect(p.state == .cancelled)
    }

    // MARK: The single-stage modes

    @Test func makePlanetAloneBeginsAtTheSecondStage() {
        var p = Plan(mode: .makePlanet, name: "planet-2", growName: nil, source: "mini3s_a", targetSteps: nil)
        #expect(p.state == .making(cancelling: false))
        #expect(p.planetSource == "mini3s_a" && p.directories == ["planet-2"])
        #expect(p.banner(elapsed: 0)?.next == "Then it opens in Watch.")
        _ = p.apply(.makeLive, in: Self.here)
        #expect(p.state == .live)
    }

    @Test func aResearchRunEndsWhereItEndsAndOffersTheNextStep() {
        var p = Plan(mode: .growOnly, name: "run-1", growName: "run-1", targetSteps: 20_000)
        _ = p.apply(.progress(step: 19_500, stepsPerSecond: 350), in: Self.here)
        #expect(p.apply(exit(0), in: Self.here) == [], "a research world never makes a planet by itself")
        #expect(p.state == .finished(stopped: false))
        #expect(p.step == 20_000 && p.banner(elapsed: 0)?.detail == "run-1 is saved at step 20,000.", "a run that ended by itself reached its target")
        #expect(p.banner(elapsed: 0)?.buttons == [.makePlanet, .keepGrowing, .done])
        p.offersPlanet = false
        #expect(p.banner(elapsed: 0)?.buttons == [.keepGrowing, .done], "a flat world has no planet to make")
    }

    @Test func stoppingAResearchRunSaysWhereItStopped() {
        var p = Plan(mode: .keepGrowing, name: "run-1", growName: "run-1", targetSteps: 50_000, startStep: 30_000)
        _ = p.apply(.progress(step: 34_000, stepsPerSecond: 300), in: Self.here)
        let b = p.banner(elapsed: 2)
        #expect(abs((b?.fraction ?? 0) - 0.2) < 0.001, "a continued run counts from where it began")
        _ = p.apply(.stopPressed, in: Self.here)
        _ = p.apply(exit(130, stop: true), in: Self.here)
        #expect(p.state == .finished(stopped: true))
        #expect(p.banner(elapsed: 0)?.title == "Stopped at step 34,000")
    }

    // MARK: Rules over every state

    /// Every state a pipeline can be in, reached by events.
    func states() -> [(String, Plan)] {
        var out: [(String, Plan)] = [("growing", grow())]
        var stopping = grow(); _ = stopping.apply(.stopPressed, in: Self.here); out.append(("stopping", stopping))
        var stopped = grow(); _ = stopped.apply(exit(130, stop: true), in: Self.here); out.append(("stopped", stopped))
        var unseen = grow(); _ = unseen.apply(exit(0), in: .init(visible: false)); out.append(("notVisible", unseen))
        var open = grow(); _ = open.apply(exit(0), in: .init(openWorld: "w")); out.append(("worldOpen", open))
        var taken = grow(); _ = taken.apply(exit(0), in: .init(nameTaken: true)); out.append(("nameTaken", taken))
        var noEngine = grow(); _ = noEngine.apply(exit(0), in: .init(engineReady: false)); out.append(("noEngine", noEngine))
        var failed = grow(); _ = failed.apply(exit(1), in: Self.here); out.append(("failedGrowing", failed))
        var making = grow(); _ = making.apply(exit(0), in: Self.here); out.append(("making", making))
        var cancelling = making; _ = cancelling.apply(.cancelPressed, in: Self.here); out.append(("cancelling", cancelling))
        var cancelled = cancelling; _ = cancelled.apply(.makeEnded(nil), in: Self.here); out.append(("cancelled", cancelled))
        var nothingSaved = grow(); _ = nothingSaved.apply(.stopPressed, in: Self.here)
        _ = nothingSaved.apply(.growExited(.init(status: 2, signalled: true, stopRequested: true, savePoint: false)), in: Self.here); out.append(("nothingSaved", nothingSaved))
        var madeFailed = making; _ = madeFailed.apply(.makeFailed("x"), in: Self.here); out.append(("failedMaking", madeFailed))
        var live = making; _ = live.apply(.makeLive, in: Self.here); out.append(("live", live))
        var finished = Plan(mode: .growOnly, name: "r", growName: "r", targetSteps: 5); _ = finished.apply(exit(0), in: Self.here); out.append(("finished", finished))
        return out
    }

    static let events: [(String, Plan.Event)] = [
        ("progress", .progress(step: 5, stepsPerSecond: 1)),
        ("exit0", .growExited(.init(status: 0))), ("exit1", .growExited(.init(status: 1))), ("exit130", .growExited(.init(status: 130, stopRequested: true))),
        ("stop", .stopPressed), ("cancel", .cancelPressed), ("makePlanet", .makePlanetPressed), ("keepGrowing", .keepGrowingPressed),
        ("done", .donePressed), ("live", .makeLive), ("failed", .makeFailed("x")), ("ended", .makeEnded(nil)), ("lines", .makeProgress(["houses: x"])),
    ]

    @Test func aWaitingPipelineAlwaysTellsThePersonWhatToDo() {
        for (name, plan) in states() where !plan.isOver {
            let b = plan.banner(elapsed: 3)
            #expect(b != nil, "\(name) has a banner")
            if case .parked = plan.state {
                #expect(b?.tone == .attention && !(b?.buttons.isEmpty ?? true) && b?.buttons.contains(.done) == true, "\(name): a parked pipeline has buttons and a way out")
                #expect(!(b?.detail.isEmpty ?? true), "\(name)")
            }
        }
    }

    @Test func everyStateHasABannerOrIsDismissedAndNeverSaysAnythingEmpty() {
        for (name, plan) in states() {
            let b = plan.banner(elapsed: 1)
            #expect(b != nil, "\(name)")
            #expect(!(b?.title.isEmpty ?? true) && !(b?.detail.isEmpty ?? true), "\(name)")
            if let f = b?.fraction { #expect((0...1).contains(f), "\(name): \(f)") }
            for button in b?.buttons ?? [] { #expect(!button.label.isEmpty) }
        }
    }

    @Test func theSecondStageStartsOnlyFromTheTwoPlacesThatMayAndNeverFromAnyOtherEvent() {
        for (name, plan) in states() {
            for (eventName, event) in Self.events {
                var p = plan
                let effects = p.apply(event, in: Self.here)
                if effects.contains(.startMaking) {
                    let fromGrowing: Bool = { if case .growing(false) = plan.state { return true } else { return false } }() || {
                        if case .growing(true) = plan.state { return true } else { return false }
                    }()
                    let fromParked: Bool = { if case .parked = plan.state { return true } else { return false } }()
                    #expect(fromGrowing || fromParked, "\(name) + \(eventName) started stage 2")
                    if fromGrowing { #expect(eventName == "exit0", "\(name) + \(eventName): only a run that finished by itself continues") }
                    if fromParked { #expect(eventName == "makePlanet") }
                }
                // an ended pipeline stays ended whatever arrives
                if plan.isOver, plan.state != .dismissed, eventName != "done" {
                    #expect(p.state == plan.state, "\(name) + \(eventName): an ended pipeline changed")
                }
            }
        }
    }

    @Test func nothingStartsInvisiblyWhateverTheEvent() {
        let dark = Plan.Surroundings(visible: false, openWorld: nil, nameTaken: false, engineReady: true)
        for (name, plan) in states() {
            for (eventName, event) in Self.events where eventName != "makePlanet" {
                var p = plan
                #expect(!p.apply(event, in: dark).contains(.startMaking), "\(name) + \(eventName) started stage 2 with nobody looking")
            }
        }
    }

    @Test func theBannersWordsAreNotTheRetiredOnes() {
        let banned = ["Resume", "Checkpoint", "Fork", "Home", "spin-up", "Source", "the Lab", "director"]
        for (name, plan) in states() {
            let b = plan.banner(elapsed: 1)
            let text = [b?.title, b?.detail, b?.next].compactMap { $0 } + (b?.buttons.map(\.label) ?? [])
            for word in banned { for t in text { #expect(!t.contains(word), "\(name): '\(word)' in \"\(t)\"") } }
        }
    }

    @Test func aPlanetThatDidNotOpenIsToldInPlainWordsWithTheEnginesLastLine() {
        #expect(PipelineWords.planetFailure("alife play found no usable checkpoint (exit code 4): alife play: refused (exit code 4)")
            == "The engine could not open the world the planet was to start from. It said: alife play: refused (exit code 4)")
        #expect(PipelineWords.planetFailure("alife play failed (exit code 4): alife play found no usable checkpoint (exit code 4)")
            == "The engine could not open the world the planet was to start from. It said: alife play found no usable save point (exit code 4)")
        #expect(PipelineWords.planetFailure("alife play failed (exit code 1)") == "The engine stopped while it was making the planet.")
        #expect(PipelineWords.planetFailure("alife play refused the request (exit code 2): weather needs 3D") == "The engine did not accept the planet's settings. It said: weather needs 3D")
        #expect(PipelineWords.planetFailure("alife play was interrupted (exit code 130)") == "The planet was interrupted.")
        #expect(PipelineWords.planetFailure("alife play exited with code 9") == "alife play exited with code 9")
        #expect(PipelineWords.planetFailure("'w' is still saving: wait until it has closed.").hasPrefix("That world is still being saved"))
        #expect(PipelineWords.planetFailure("Failed to launch /x/alife: no such file").hasPrefix("The engine could not be started."))
        for code in [1, 2, 3, 4, 130, 7] {
            let text = PipelineWords.planetFailure("alife play x (exit code \(code))")
            for word in ["checkpoint", "alife play", "Resume", "Fork"] { #expect(!text.contains(word), "\(text)") }
        }
    }

    @Test func theProblemLineIsTheEndOfTheTracebackNotItsStart() {
        let lines = ["Traceback (most recent call last):", "  File \"x.py\", line 3, in <module>", "    boom()", "ValueError: the steps must be positive"]
        #expect(PipelineWords.firstProblem(in: lines) == "the steps must be positive")
        #expect(PipelineWords.firstProblem(in: ["alife: error: unknown preset 'x'"]) == "unknown preset 'x'")
        #expect(PipelineWords.firstProblem(in: ["just some text", "more"]) == "just some text")
        #expect(PipelineWords.firstProblem(in: []) == nil && PipelineWords.firstProblem(in: ["", "  "]) == nil)
        #expect(PipelineWords.firstProblem(in: [String(repeating: "x", count: 500)])!.count == 238)
    }

    @Test func theEnginesRefusalOfAForksSettingsIsSaidInPlainWordsWithItsReason() {
        let lines = ["--init-from runs/planet-1: the state violates the new config:", "  mutation rate 0.0011 is below the floor 0.004", "  second thing"]
        let said = PipelineWords.firstProblem(in: lines)
        #expect(said == "The newest save point of planet-1 does not fit the changed settings: mutation rate 0.0011 is below the floor 0.004; second thing", "\(said ?? "-")")
        #expect(!said!.contains("--init-from"))
    }

    @Test func theStepAtTheEndIsTheNewestSavePointNotTheLastNumbersTheBarRead() {
        var p = grow(target: nil)
        _ = p.apply(.growExited(.init(status: 0, savedStep: 40_000)), in: Self.here)
        #expect(p.step == 40_000, "\(p.step)")
        var q = grow()
        _ = q.apply(.stopPressed, in: Self.here)
        _ = q.apply(.growExited(.init(status: 130, stopRequested: true, savedStep: 12_300)), in: Self.here)
        #expect(q.step == 12_300)
    }

    @Test func aRunThatLeftNoSavePointDoesNotSayItIsKept() {
        var p = grow()
        _ = p.apply(.growExited(.init(status: 4, problem: "boom", savePoint: false)), in: Self.here)
        #expect(p.wroteNothing && p.banner(elapsed: 0)?.next?.contains("kept") == false, "\(String(describing: p.banner(elapsed: 0)?.next))")
        var q = grow()
        _ = q.apply(.growExited(.init(status: 4, problem: "boom", savePoint: true)), in: Self.here)
        #expect(!q.wroteNothing && q.banner(elapsed: 0)?.next?.contains("kept") == true)
    }
}
