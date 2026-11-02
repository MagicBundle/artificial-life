import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// The pipeline of docs/APP.md §3.2 with real processes (a stand-in engine that makes the files the engine makes): for every state of the
/// table, Stop, Cancel, failure, the window closed, the app quit and the app launched again. Nothing starts invisibly, a parked pipeline says
/// what to do, two writers are impossible, one click is one world, and names never collide.
@MainActor
@Suite("The pipeline, with a stand-in engine", .serialized)
struct PipelineTests {
    // MARK: The whole path

    @Test func smallPlanetGoesFromTheButtonToALivePlanetWithTheSeedNestedUnderIt() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        let reasons = await rig.create("planet-1")
        #expect(reasons.isEmpty, "\(reasons)")
        #expect(rig.coordinator.plan?.mode == .growThenPlay && rig.coordinator.isActive)
        #expect(rig.setup.taken.isSuperset(of: ["planet-1", "planet-1-grow"]), "the names are held from the first moment, before the library has looked")
        // one click is one world: the planet is listed at once, being grown, with its seed under it
        let row = rig.library.state.row(named: "planet-1")
        #expect(row?.isPlaceholder == true && row?.state == .growing && row?.kind == .played)
        #expect(await patiently { rig.exists("planet-1-grow") })
        var nested = false
        for _ in 0..<40 where !nested {
            await rig.scan()
            nested = LibraryState.tree(rig.library.state.rows).first { $0.row.name == "planet-1" }?.children.map(\.name) == ["planet-1-grow"]
            if !nested { try? await Task.sleep(for: .milliseconds(50)) }
        }
        #expect(nested, "one click is one world: the seed is nested under the planet")
        #expect(await rig.waitForState { $0 == .live }, "state: \(String(describing: rig.coordinator.plan?.state)) \(rig.controller.logText)")
        #expect(rig.play.phase == .live && rig.play.world == "planet-1")
        #expect(rig.playerStarts == 1, "the window is told that the player's click started the world")
        await rig.scan()
        let tree = LibraryState.tree(rig.library.state.rows)
        #expect(tree.first { $0.row.name == "planet-1" }?.children.map(\.name) == ["planet-1-grow"])
        #expect(rig.library.state.rows.filter { $0.kind == .played }.map(\.name) == ["planet-1"])
        #expect(rig.coordinator.banner?.title == "planet-1 is ready")
        #expect(rig.exists("planet-1-grow") && rig.exists("planet-1"))
        // the commands that ran are the commands that were shown
        let log = rig.controller.logText
        #expect(log.contains("run --preset tiny3s --seed 1 --steps 30000 --out runs/planet-1-grow"), "\(log)")
        rig.coordinator.press(.done)
        #expect(rig.coordinator.plan == nil && rig.coordinator.banner == nil)
    }

    @Test func theStageTwoCommandIsTheShownOne() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        rig.setup.name = "planet-1"
        let shown = rig.setup.commandBlocks.map(\.command)
        #expect(shown.count == 2)
        rig.setup.start()
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.request?.arguments == rig.setup.plan.stages[1].arguments || rig.play.request != nil)
        let args = try #require(rig.play.request).arguments
        #expect(shown[1].hasSuffix(args.map(shellQuote).joined(separator: " ")), "shown: \(shown[1])")
    }

    // MARK: Stage 1: Stop

    @Test func stopInStageOneParksWithTheTwoWaysOnAndStartsNothing() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        #expect(rig.coordinator.banner?.buttons == [.stop])
        rig.coordinator.press(.stop)
        #expect(rig.coordinator.banner?.buttons == [], "stopping: nothing left to press")
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.stopped, step: rig.coordinator.plan?.step ?? 0))
        #expect(rig.coordinator.banner?.buttons == [.makePlanetFromHere, .keepGrowing, .done])
        try await Task.sleep(for: .milliseconds(600))
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"), "a Stop never continues by itself, window in front or not")
        #expect(rig.controller.job == nil)
        // the world is saved and listed honestly, with the next action
        await rig.scan()
        let seed = try #require(rig.library.state.row(named: "planet-1-grow"))
        #expect(seed.step == 12_600 && seed.unmadePlanet == "planet-1" && seed.note == "Stopped at step 12,600 of 30,000.")
        // Make planet from here
        rig.mode = "ok:ok"
        rig.coordinator.press(.makePlanetFromHere)
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.exists("planet-1"))
    }

    @Test func keepGrowingFromAParkedPipelineRunsTheRunAgainFromItsSavePoint() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        rig.coordinator.press(.stop)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        rig.mode = "ok:ok"
        rig.coordinator.press(.keepGrowing)
        #expect(rig.coordinator.plan?.state == .growing(stopping: false))
        #expect(await patiently { rig.controller.lastRunExit?.serial == 2 || rig.coordinator.plan?.state != .growing(stopping: false) })
        #expect(await rig.waitForState { $0 == .live }, "it was grown again and the planet made: \(String(describing: rig.coordinator.plan?.state)) \(rig.controller.logText)")
        #expect(rig.controller.logText.contains("resume runs/planet-1-grow"))
    }

    @Test func watchIsToldOfTheResearchRunAtItsStartAndAtEveryKeepGrowing() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(rig.researchStarts == ["planet-1-grow"], "the world Watch follows is the run's, named as the folder")
        #expect(await rig.waitForGrowing())
        rig.coordinator.press(.stop)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        rig.mode = "ok:ok"
        rig.coordinator.press(.keepGrowing)
        #expect(rig.researchStarts == ["planet-1-grow", "planet-1-grow"])
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.researchStarts.count == 2, "making the planet is not a research run: Watch is told of nothing then")
    }

    @Test func whileAWorldIsBeingMadeEveryButtonOfTheWorldsSaysSo() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        await rig.scan()
        let row = try #require(rig.library.state.row(named: "planet-1-grow"))
        let a = WorldActionSet.make(row: row, controller: rig.controller, play: rig.play, library: rig.library, setup: rig.setup)
        let busy = SetupRules.pipelineReason(.running)
        #expect(busy != nil && a.makePlanet.reason == busy && a.branch.reason == busy && a.keepGrowing.reason == busy,
                "\(String(describing: a.makePlanet.reason)) / \(String(describing: a.branch.reason))")
        rig.coordinator.press(.stop)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
    }

    @Test func aRefusedStartTellsWatchNothing() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        rig.controller.settings.name = "bad name"
        _ = await rig.create("bad name")
        #expect(rig.researchStarts.isEmpty)
    }

    @Test func cancelInStageOneDoesNothingBecauseStopIsItsWord() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        rig.coordinator.press(.cancel)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.coordinator.plan?.state == .growing(stopping: false) && rig.controller.job == .run)
    }

    @Test func stopFromTheActivityPageIsHeardToo() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        rig.controller.stop()   // the Activity page's Stop
        #expect(await patiently { rig.coordinator.plan?.state == .growing(stopping: true) || rig.coordinator.plan?.state != .growing(stopping: false) })
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.play.phase == .idle)
    }

    @Test func aStopInTheEnginesFirstSecondsIsToldAndNothingIsParkedOnAWorldThatDoesNotExist() async throws {
        let rig = try PipelineRig(mode: "silent:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        rig.coordinator.press(.stop)
        #expect(await rig.waitForState { if case .failed = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .failed(.nothingSaved), "the run ended on the signal with no save point")
        #expect(rig.coordinator.banner?.title == "Stopped before anything was saved" && rig.coordinator.banner?.buttons == [.tryAgainNewName, .done])
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"))
    }

    @Test func howARunEndedIsRecordedForThePipeline() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try rig.makeSource("src")
        await rig.scan()
        rig.setup.openKeepGrowing(world: "src")
        rig.setup.start()
        #expect(await patiently { rig.exists("src/checkpoints/step_0000012000.safetensors") }, "the stand-in is up and listening")
        rig.controller.stop()
        #expect(await patiently { rig.controller.lastRunExit != nil })
        var exit = try #require(rig.controller.lastRunExit)
        #expect(exit.run == "src" && exit.status == 130 && exit.stopRequested && !exit.forced && !exit.signalled && exit.serial == 1, "\(exit)")
        rig.coordinator.press(.done)
        rig.setup.openKeepGrowing(world: "src")
        rig.setup.start()
        #expect(await patiently { rig.controller.job == .run })
        rig.controller.forceStop()
        #expect(await patiently { rig.controller.lastRunExit?.serial == 2 })
        exit = try #require(rig.controller.lastRunExit)
        #expect(exit.forced && exit.signalled && exit.serial == 2, "a force stop is not a stop")
    }

    @Test func aRunNotStartedByThePageStillGreysTheButtonWithItsReason() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try rig.makeSource("src")
        await rig.scan()
        rig.controller.refreshRuns()
        rig.controller.settings.mode = .resume
        rig.controller.settings.sourceRun = "src"
        rig.controller.start()
        #expect(rig.controller.job == .run, "\(rig.controller.logText) \(rig.controller.validationErrors)")
        rig.setup.openCreate()
        #expect(rig.setup.reasons.contains("A run is going. Wait for it to finish, or stop it in Activity."), "\(rig.setup.reasons)")
        rig.controller.forceStop()
    }

    // MARK: Stage 1: failure

    @Test func aRefusedRunFailsWithItsLineKeepsWhatItWroteAndStartsNothing() async throws {
        let rig = try PipelineRig(mode: "refuse:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .failed = $0 { return true } else { return false } })
        let b = rig.coordinator.banner
        #expect(b?.title == "Growing failed" && b?.detail == "refused by the stand-in")
        #expect(b?.buttons == [.showLog, .tryAgainNewName, .done] && b?.next == "The grown world is kept as it was written.")
        #expect(rig.exists("planet-1-grow") && !rig.exists("planet-1") && rig.play.phase == .idle)
        #expect(!rig.coordinator.isActive, "the form comes back")
        rig.coordinator.press(.showLog)
        #expect(rig.activityShown == 1)
        rig.coordinator.press(.tryAgainNewName)
        #expect(rig.opened.last?.0 == .create)
        // and the name of the failed attempt is taken, so the next suggestion is a new one
        await rig.scan()
        #expect(rig.setup.taken.contains("planet-1-grow"))
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: rig.setup.taken) == "planet-2")
        // a failed run is a world with one save point or none: it never counts as the planet
        #expect(rig.library.state.rows.filter { $0.kind == .played }.isEmpty)
    }

    @Test func aRunThatCouldNotBeStartedAtAllIsToldInTheFormNotInvisibly() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try FileManager.default.removeItem(at: rig.repo.appendingPathComponent(".venv/bin/alife"))
        rig.controller.refreshRepo()
        await rig.scan()
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.reasons.contains("Set up the engine first."))
        #expect(rig.coordinator.plan == nil)
    }

    // MARK: 1 to 2: nothing starts invisibly

    @Test func aRunThatFinishesWithTheWindowAwayParksBouncesOnceAndWaits() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        rig.looking = false   // the window is closed, or another app is in front
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.notVisible, step: 30_000))
        #expect(rig.bounces == 1)
        try await Task.sleep(for: .milliseconds(500))
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"), "nothing started behind anyone's back")
        #expect(rig.coordinator.banner?.buttons == [.makePlanet, .keepGrowing, .done])
        // coming back starts nothing either; the button does
        rig.looking = true
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.play.phase == .idle && rig.coordinator.plan?.state == .parked(.notVisible, step: 30_000))
        #expect(rig.bounces == 1, "once")
        rig.coordinator.press(.makePlanet)
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.world == "planet-1")
    }

    @Test func aRunThatFinishesWhileTheAppIsQuittingStartsNothingEvenWithTheWindowShown() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        rig.coordinator.isQuitting = { true }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.notVisible, step: 30_000))
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.play.phase == .idle)
    }

    @Test func anotherWorldBeingOpenParksTheSecondStageInsteadOfReplacingIt() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        // a played world is open when the run finishes
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.worldOpen("mini3s_a"), step: 30_000))
        #expect(rig.play.world == "mini3s_a" && rig.play.phase == .live, "the open world was not replaced")
        #expect(rig.coordinator.banner?.detail.contains("Make planet asks to stop mini3s_a first") == true)
        // pressing the button while it is still open changes nothing
        rig.coordinator.press(.makePlanet)
        #expect(rig.play.world == "mini3s_a" && rig.coordinator.plan?.state == .parked(.worldOpen("mini3s_a"), step: 30_000))
        // closed, it works
        rig.play.stop()
        #expect(await patiently { rig.play.phase == .idle })
        rig.coordinator.press(.makePlanet)
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.world == "planet-1")
    }

    @Test func aNameThatAppearedWhileItGrewIsNeverOverwrittenNorTakenOverInCapitals() async throws {
        for taken in ["planet-1", "PLANET-1"] {
            let rig = try PipelineRig(mode: "long:ok")
            defer { rig.finish() }
            #expect(await rig.create("planet-1").isEmpty)
            #expect(await rig.waitForGrowing())
            // somebody makes the folder meanwhile (the command line, another app)
            try FileManager.default.createDirectory(at: rig.runs.appendingPathComponent(taken), withIntermediateDirectories: true)
            try "mine".write(to: rig.runs.appendingPathComponent(taken).appendingPathComponent("keep.txt"), atomically: true, encoding: .utf8)
            rig.mode = "ok:ok"
            // the stand-in's long run only ends on SIGINT; end it as a run that ran to its end by sending the same signal the engine's timer would
            rig.controller.stop()
            #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
            // a stop parks as "stopped"; press Make planet: the name is taken, so it stays parked with the reason
            rig.coordinator.press(.makePlanetFromHere)
            #expect(rig.coordinator.plan?.state == .parked(.nameTaken, step: rig.coordinator.plan?.step ?? 0), "\(taken)")
            #expect(rig.play.phase == .idle)
            #expect((try? String(contentsOf: rig.runs.appendingPathComponent(taken).appendingPathComponent("keep.txt"), encoding: .utf8)) == "mine")
            #expect(rig.coordinator.banner?.buttons == [.makeAgainNewName, .done])
        }
    }

    // MARK: Stage 2: Cancel, failure, window, quit

    @Test func cancelInStageTwoStopsAtOnceAndLeavesAnUnfinishedBuildWithTheSeedKept() async throws {
        let rig = try PipelineRig(mode: "ok:slow")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { $0 == .making(cancelling: false) })
        #expect(await patiently { rig.coordinator.plan?.making.phase != .starting })
        #expect(rig.coordinator.banner?.detail.hasPrefix("Warming up the weather") == true || rig.coordinator.banner?.detail.hasPrefix("Rebuilding") == true || rig.coordinator.banner?.detail.hasPrefix("Copying") == true)
        #expect(rig.coordinator.banner?.buttons == [.cancel])
        rig.coordinator.press(.cancel)
        #expect(await rig.waitForState { $0 == .cancelled }, "Cancel is one interrupt, now: the world was stopped at once")
        #expect(await patiently { !rig.play.hasProcess }, "and the process ended")
        #expect(rig.play.phase == .idle)
        await rig.scan()
        #expect(rig.library.state.unfinished.map(\.name) == ["planet-1"], "listed under Unfinished with its reason")
        #expect(rig.library.state.row(named: "planet-1-grow") != nil && rig.exists("planet-1-grow"), "the seed is kept")
        #expect(rig.coordinator.banner?.buttons == [.makeAgainNewName, .done])
        #expect(rig.coordinator.pendingPlanets.isEmpty, "the promise is over")
    }

    @Test func aPlanetThatFailsToOpenIsToldOnceByThePipelineAndKeepsTheSeed() async throws {
        let rig = try PipelineRig(mode: "ok:fail")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .failed = $0 { return true } else { return false } })
        let b = rig.coordinator.banner
        #expect(b?.title == "planet-1 could not be made" && b?.tone == .failure)
        #expect(b?.next == "The grown world is kept, so only the planet needs making again.")
        #expect(b?.buttons == [.showLog, .makeAgainNewName, .done])
        #expect(await patiently { rig.dismissedFailures >= 1 }, "the Worlds page does not say it a second time")
        #expect(b?.detail == "The engine could not open the world the planet was to start from. It said: alife play found no usable checkpoint (exit code 4)" || b?.detail.hasPrefix("The engine could not open") == true, "\(b?.detail ?? "")")
        #expect(rig.exists("planet-1-grow"))
        rig.coordinator.press(.makeAgainNewName)
        #expect(rig.opened.last?.0 == .makePlanet(from: "planet-1-grow"))
    }

    @Test func closingTheWindowWhileThePlanetIsMadeEndsItAsCancelled() async throws {
        let rig = try PipelineRig(mode: "ok:slow")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { $0 == .making(cancelling: false) },
                "state \(String(describing: rig.coordinator.plan?.state)) play \(rig.play.phase) job \(String(describing: rig.controller.job))")
        #expect(await patiently { rig.coordinator.plan?.making.phase != .starting }, "the stand-in is listening")
        rig.looking = false
        rig.play.stop()   // what closing the window does to the world (PlayController.stop: a world being built is interrupted)
        #expect(await rig.waitForState { $0 == .cancelled })
        #expect(rig.exists("planet-1-grow"))
    }

    @Test func quittingWhileAWorldIsBeingMadeInterruptsItAndNothingStartsAfterwards() async throws {
        let rig = try PipelineRig(mode: "ok:slow")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { $0 == .making(cancelling: false) })
        #expect(await patiently { rig.coordinator.plan?.making.phase != .starting }, "the stand-in is listening")
        rig.play.stopForQuit()   // the gate's stop: the world being built is interrupted and the app goes on to end
        #expect(await patiently { !rig.play.hasProcess }, "the interrupted build ended before the app did")
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.play.phase != .live && rig.coordinator.plan?.state != .live, "it never became a world")
        rig.mode = "ok:ok"
        #expect(rig.controller.job == nil && rig.play.hasProcess == false, "and nothing else was started")
    }

    // MARK: Quit and launch again

    @Test func quittingInStageOneStopsTheRunAtASavePointAndTheNextLaunchShowsTheNextAction() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        // the real gate, with the question answered Stop and Quit
        let delegate = AppDelegate(run: rig.controller, play: rig.play, quitPanel: .offscreen())
        delegate.replyToTerminate = {}
        delegate.terminate = {}
        delegate.confirmStopRun = { name in name == "planet-1-grow" }
        rig.coordinator.isQuitting = { delegate.terminationGate.isWaiting }
        rig.looking = true
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(await patiently { rig.controller.job == nil })
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } }, "the run ended at a save point, so the pipeline parked")
        try await Task.sleep(for: .milliseconds(500))
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"), "a quit never starts the second stage, even with the window shown")
        delegate.terminationGate.abandonWait()
        rig.finish(keepFolder: true)

        // launch again: a new app over the same folder
        let launched = try PipelineRig(mode: "ok:ok", repo: rig.repo, scratch: rig.scratch)
        defer { launched.finish() }
        #expect(launched.coordinator.plan == nil, "nothing is resumed by itself")
        await launched.scan()
        #expect(launched.controller.job == nil && launched.play.phase == .idle)
        let seed = try #require(launched.library.state.row(named: "planet-1-grow"))
        #expect(seed.state == .saved && seed.unmadePlanet == "planet-1" && seed.step == 12_600)
        #expect(seed.note == "Stopped at step 12,600 of 30,000.")
        let facts = launched.setup.ruleFacts
        #expect(SetupRules.reason(.makePlanet, on: seed, in: facts) == nil && SetupRules.reason(.keepGrowing, on: seed, in: facts) == nil)
        // the next action works from the card
        launched.setup.openMakePlanet(from: "planet-1-grow", name: "planet-1")
        #expect(launched.setup.name == "planet-1" && launched.setup.reasons.isEmpty, "\(launched.setup.reasons)")
        launched.setup.start()
        #expect(await launched.waitForState { $0 == .live })
    }

    @Test func aWorldCutOffInStageTwoShowsAsAnUnfinishedBuildAfterALaunch() async throws {
        let rig = try PipelineRig(mode: "ok:slow")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { $0 == .making(cancelling: false) })
        #expect(await patiently { rig.coordinator.plan?.making.phase != .starting }, "the stand-in is listening")
        rig.play.stopForQuit()
        #expect(await patiently { !rig.play.hasProcess }, "the interrupted build ended and wrote what it leaves")
        rig.finish(keepFolder: true)
        let launched = try PipelineRig(mode: "ok:ok", repo: rig.repo, scratch: rig.scratch)
        defer { launched.finish() }
        await launched.scan()
        #expect(launched.library.state.unfinished.map(\.name) == ["planet-1"])
        #expect(launched.library.state.row(named: "planet-1-grow")?.unmadePlanet == nil, "its planet exists, as an unfinished build")
    }

    // MARK: Two writers are impossible

    @Test func aSecondPipelineWhileOneRunsIsRefusedAndTheFormSaysWhy() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        #expect(rig.setup.reasons.contains("A world is being made. Wait for it, or press Stop in the bar."))
        let setupPlan = rig.setup.plan
        let refusal = rig.coordinator.begin(setupPlan, planet: nil, timing: .init())
        #expect(refusal == "Another world is being made. Wait for it, or stop it.")
        rig.controller.forceStop()
    }

    @Test func theDirectoryAPipelineGrowsCannotBeBranchedKeptGrowingOrMadeIntoAPlanetFromBehindTheForm() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        await rig.scan()
        let seed = try #require(rig.library.state.row(named: "planet-1-grow"))
        let facts = rig.setup.ruleFacts
        #expect(SetupRules.reason(.keepGrowing, on: seed, in: facts) == "A world is being made. Wait for it, or press Stop in the bar.")
        #expect(SetupRules.reason(.branch, on: seed, in: facts) != nil, "a run is going")
        // the guard itself, asked for the seed and for the planet that is promised
        let g = rig.coordinator.currentGuard()
        #expect(g.writeRefusal("planet-1-grow") == "planet-1-grow is growing. Wait for it to finish, or stop it.")
        #expect(g.writeRefusal("Planet-1") == "planet-1 is being made.", "the promised planet, in other capitals")
        rig.controller.forceStop()
    }

    @Test func aWorldAnotherProcessHoldsCannotBeWrittenByAPipeline() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        rig.coordinator.elsewhere = { ["planet-1-grow"] }
        let reasonsBefore = await rig.create("planet-1")
        #expect(reasonsBefore.isEmpty || reasonsBefore.contains { $0.contains("already here") })
        // the guard is asked again at start, by the coordinator, whatever the form said
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        rig.setup.name = "planet-9"
        rig.coordinator.elsewhere = { ["planet-9-grow"] }
        let refusal = rig.coordinator.begin(rig.setup.plan, planet: rig.setup.stageTwoSpec(seed: "planet-9-grow"), timing: .init())
        #expect(refusal == "planet-9-grow is in use by another app or process.")
        #expect(rig.coordinator.plan == nil && rig.controller.job == nil)
    }

    @Test func makePlanetFromTheWorldThatIsOpenIsRefusedAtClickTimeAndAtStart() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        await rig.scan()
        rig.setup.openMakePlanet(from: "mini3s_a")
        #expect(rig.setup.reasons.contains("Close it first"), "\(rig.setup.reasons)")
        let refusal = rig.coordinator.begin(rig.setup.plan, planet: rig.setup.makePlanetSpec, timing: .init())
        #expect(refusal == "Close mini3s_a first: a new planet replaces the open world.")
        #expect(rig.play.world == "mini3s_a" && rig.play.phase == .live)
    }

    @Test func aBranchFromTheWorldThatIsOpenIsRefusedAtStartWhateverTheFormSaid() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        rig.setup.openBranch(from: "mini3s_a")
        let refusal = rig.coordinator.begin(rig.setup.plan, planet: nil, timing: .init())
        #expect(refusal == "Close mini3s_a first: it is being written.")
        #expect(rig.coordinator.plan == nil && rig.controller.job == nil)
    }

    // MARK: Names

    @Test func aNameTakenByAnythingIsRefusedBeforeAnythingStarts() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "planet-1")
        try FileManager.default.createSymbolicLink(atPath: rig.runs.appendingPathComponent("planet-2").path, withDestinationPath: "/Volumes/Archive/runs/planet-2")
        try LibraryTestFolders.research(rig.runs, "planet-3-GROW")
        await rig.scan()
        for name in ["planet-1", "PLANET-1", "planet-2", "Planet-2", "planet-3"] {
            let reasons = await rig.create(name)
            #expect(!reasons.isEmpty && reasons.contains { $0.contains("already here") }, "\(name): \(reasons)")
            #expect(rig.coordinator.plan == nil && rig.controller.job == nil, "\(name)")
        }
        #expect(rig.setup.freeAlternative != nil)
        let free = SetupPlan.suggestName(base: "planet", goal: .play, taken: rig.setup.taken)
        #expect(free == "planet-4", "planet-1, the link planet-2 and the seed of planet-3 are all taken")
    }

    // MARK: Make planet alone, Branch and Keep growing

    @Test func makePlanetFromAWorldIsOneStageAndOneWorld() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "src", steps: [10_000, 20_000])
        await rig.scan()
        rig.setup.openMakePlanet(from: "src")
        rig.setup.launcher.selectCheckpoint(10_000)
        rig.setup.name = "planet-5"
        #expect(rig.setup.reasons.isEmpty, "\(rig.setup.reasons)")
        #expect(rig.setup.commandBlocks.count == 1 && rig.setup.commandBlocks[0].command.contains("--init-from runs/src:10000"))
        rig.setup.start()
        #expect(rig.coordinator.plan?.mode == .makePlanet)
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.world == "planet-5" && rig.coordinator.banner?.title == "planet-5 is ready")
    }

    @Test func aBranchWithOneChangedSettingRunsTheForkCommandAndRemembersTheEdit() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try rig.makeSource("src")
        await rig.scan()
        rig.setup.openBranch(from: "src")
        #expect(rig.setup.name == "src-branch-1" && rig.setup.intent == .branch(from: "src"))
        rig.controller.settings.levers.rateMin = "0.004"
        #expect(rig.setup.changes.map(\.key) == ["mutation.rate_min"])
        #expect(ChangeList.sentence(rig.setup.changes) == "1 setting changed: Mutation rate floor 0.001 to 0.004")
        #expect(rig.setup.reasons.isEmpty, "\(rig.setup.reasons)")
        let shown = rig.setup.commandBlocks[0].command
        #expect(shown.contains("run --init-from runs/src --out runs/src-branch-1") && shown.contains("mutation.rate_min=0.004"))
        rig.setup.start()
        #expect(rig.coordinator.plan?.targetSteps == 60_000, "a blank length is the source's own run length beyond its save point: \(String(describing: rig.coordinator.plan?.targetSteps))")
        #expect(await rig.waitForState { $0 == .finished(stopped: false) })
        #expect(rig.exists("src-branch-1") && rig.play.phase == .idle, "a branch is one research world and starts no planet")
        #expect(rig.coordinator.banner?.buttons == [.makePlanet, .keepGrowing, .done])
        // a branch's edit belongs to that branch: the next Branch starts from its source's own values, and so does a Create
        rig.setup.openBranch(from: "src")
        #expect(rig.setup.rememberedShown.isEmpty && rig.controller.settings.levers.rateMin == "0.001", "\(rig.controller.settings.levers.rateMin)")
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.rememberedShown.isEmpty, "only a Create's edits are remembered")
    }

    @Test func keepGrowingContinuesAResearchWorldWithoutANewOne() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "src", steps: [12_000], status: "interrupted")
        await rig.scan()
        rig.setup.openKeepGrowing(world: "src")
        #expect(rig.setup.reasons.isEmpty && rig.setup.commandBlocks[0].command.hasSuffix("resume runs/src"), "\(rig.setup.reasons) \(rig.setup.commandBlocks) \(rig.library.state.rows.map(\.name)) \(rig.controller.settings.sourceRun)")
        let before = Set((try? FileManager.default.contentsOfDirectory(atPath: rig.runs.path)) ?? [])
        rig.setup.start()
        #expect(await rig.waitForState { $0 == .finished(stopped: false) })
        #expect(Set((try? FileManager.default.contentsOfDirectory(atPath: rig.runs.path)) ?? []) == before, "no new directory")
    }

    @Test func aPlayedWorldCannotBeKeptGrowingAndTheFormSaysWhatToDoInstead() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.setup.openKeepGrowing(world: "mini3s_a")
        #expect(rig.setup.reasons == ["A played world is continued by playing it."], "\(rig.setup.reasons) \(rig.library.state.rows.map(\.name)) \(rig.controller.settings.sourceRun)")
    }

    // MARK: The remembered values never ride unseen

    @Test func aRememberedEditShowsOnTheNextCreateAndNeverOnADefaultFormUnseen() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.changes.isEmpty && rig.setup.rememberedShown.isEmpty, "a first form has nothing remembered")
        rig.controller.settings.levers.world3D.gravity = "0.09"
        rig.setup.name = "planet-1"
        rig.setup.goal = .research
        rig.setup.start()
        #expect(await rig.waitForState { $0 == .finished(stopped: false) })
        rig.coordinator.press(.done)
        await rig.scan()
        // the next form: the edit is back, as a chip and in the Changes list, before anything is pressed
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.rememberedShown.map(\.key) == ["physics.gravity"])
        #expect(rig.setup.changes.map(\.key) == ["physics.gravity"])
        #expect(rig.setup.plan.stages[0].arguments.contains("physics.gravity=0.09"), "and it is in the command that would run")
        // a form with no edit forgets them
        rig.setup.resetRemembered()
        #expect(rig.setup.changes.isEmpty && !rig.setup.plan.stages[0].arguments.contains("physics.gravity=0.09"))
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.rememberedShown.isEmpty)
    }

    @Test func aRememberedEditThatDoesNotFitTheWorldIsSaidNotDropped() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        rig.controller.settings.levers.world3D.gravity = "0.09"
        rig.controller.settings.levers.rateMin = "0.002"
        rig.setup.name = "planet-1"
        rig.setup.goal = .research
        rig.setup.start()
        _ = await rig.waitForState { $0 == .finished(stopped: false) }
        rig.coordinator.press(.done)
        rig.setup.openCreate(.recipe(Recipes.flatWorld.id))
        #expect(rig.setup.rememberedShown.map(\.key) == ["mutation.rate_min"])
        #expect(rig.setup.rememberedSkipped == ["physics.gravity"], "kept for a world it fits, and told")
    }

    // MARK: Review fixes (H1, M1, M6, M7)

    @Test func aWorldThatDiedOutIsNeverReportedAsFinishedAndNeverBecomesAPlanet() async throws {
        let rig = try PipelineRig(mode: "extinct:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .diedOut = $0 { return true } else { return false } })
        let b = rig.coordinator.banner
        #expect(b?.title == "The population died out at step 4,200" && b?.tone == .attention, "\(b?.title ?? "")")
        #expect(b?.detail.contains("no planet can be made") == true && b?.buttons == [.tryAgainNewName, .done])
        #expect(rig.coordinator.plan?.step == 4200, "not the 30,000 it was asked to reach")
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"), "stage 2 never started")
        #expect(rig.playerStarts == 0)
        #expect(!rig.coordinator.isActive, "the form comes back")
        // the library tells the same: the grown world is a research world that died out, offered no planet
        await rig.scan()
        let row = try #require(rig.library.state.row(named: "planet-1-grow"))
        #expect(row.diedOut && row.unmadePlanet == nil)
        #expect(WorldsGuide.reasonToMakePlanet(from: row, in: WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: rig.library.state.catalog)) == WorldsGuide.diedOutHasNoPlanet)
        #expect(SetupRules.reason(.makePlanet, on: row, in: rig.setup.ruleFacts) == WorldsGuide.diedOutHasNoPlanet)
        #expect(SetupRules.reason(.keepGrowing, on: row, in: rig.setup.ruleFacts)?.contains("died out") == true)
    }

    @Test func aResearchWorldThatDiesOutIsToldAsSuchToo() async throws {
        let rig = try PipelineRig(mode: "extinct:ok")
        defer { rig.finish() }
        #expect(await rig.create("study", goal: .research).isEmpty)
        #expect(await rig.waitForState { if case .diedOut = $0 { return true } else { return false } })
        #expect(rig.coordinator.banner?.title == "The population died out at step 4,200")
    }

    @Test func aPlanetThatEndsByItselfOnAWorldThatDiedOutIsToldTheRealReason() async throws {
        let rig = try PipelineRig(mode: "ok:extinct")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .diedOut = $0 { return true } else { return false } })
        #expect(rig.coordinator.banner?.title == "The population died out at step 4,200", "not 'stopped': nobody stopped it")
        #expect(!(rig.coordinator.banner?.detail.contains("stopped") ?? false))
    }

    @Test func aPlanetThatEndsByItselfForAnotherReasonIsNotCalledStopped() async throws {
        let rig = try PipelineRig(mode: "ok:quits")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .failed(.making) = $0 { return true } else { return false } })
        let b = rig.coordinator.banner
        #expect(b?.title == "planet-1 could not be made" && b?.detail == "The planet's engine ended by itself before the planet was ready.", "\(b?.detail ?? "")")
        #expect(!(b?.detail.contains("stopped") ?? false))
    }

    @Test func makeAgainWithANewNameOnAParkedNameTakenPlanDismissesItAndOpensTheForm() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        // the hook as the window wires it: the page's own model opens the form
        rig.coordinator.hooks.openSetup = { [unowned rig] intent, name in
            if case .makePlanet(let world) = intent { rig.setup.openMakePlanet(from: world, name: name) }
        }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        try LibraryTestFolders.research(rig.runs, "planet-1")   // a world of that name appears meanwhile
        rig.mode = "ok:ok"
        rig.controller.stop()
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        rig.coordinator.press(.makePlanetFromHere)
        #expect(rig.coordinator.plan?.state == .parked(.nameTaken, step: rig.coordinator.plan?.step ?? 0))
        #expect(rig.coordinator.isActive, "the page would show the bar and no form")
        await rig.scan()
        rig.coordinator.press(.makeAgainNewName)
        #expect(rig.coordinator.plan == nil && !rig.coordinator.isActive, "the bar went, so the form shows")
        #expect(rig.setup.intent == .makePlanet(from: "planet-1-grow"))
        #expect(rig.setup.name == "planet-2", "a free name, not the taken one: \(rig.setup.name)")
        #expect(rig.setup.reasons.isEmpty, "\(rig.setup.reasons)")
        rig.setup.start()
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.world == "planet-2")
    }

    @Test func aCardsButtonsAreGreyWhileAWorldIsBeingMadeOrWaitsForAnAnswer() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try rig.makeSource("src")
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let row = try #require(rig.library.state.row(named: "src"))
        for action in [SetupAction.keepGrowing, .branch, .makePlanet] {
            #expect(SetupRules.reason(action, on: row, in: rig.setup.ruleFacts) == "A world is being made. Wait for it, or press Stop in the bar.", "\(action)")
        }
        rig.controller.stop()
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        for action in [SetupAction.keepGrowing, .branch, .makePlanet] {
            #expect(SetupRules.reason(action, on: row, in: rig.setup.ruleFacts)?.hasPrefix("A world is waiting for your answer") == true, "\(action)")
        }
        // and a form is not opened under the bar
        rig.setup.openBranch(from: "src")
        #expect(rig.setup.intent == .create || rig.setup.intent != .branch(from: "src"))
        rig.coordinator.press(.done)
        rig.setup.openBranch(from: "src")
        #expect(rig.setup.intent == .branch(from: "src"))
    }

    @Test func anUnfinishedBranchOrMakePlanetFormSurvivesThePageAppearingAgainInEitherWindow() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try rig.makeSource("src", steps: [10_000, 20_000])
        await rig.scan()
        let m = rig.setup
        m.openBranch(from: "src")
        m.name = "my-branch"
        rig.controller.settings.levers.world3D.gravity = "0.05"
        m.pageAppeared()   // the page (or the window of its own) appears
        m.pageAppeared()   // and the other window
        #expect(m.intent == .branch(from: "src") && m.name == "my-branch" && rig.controller.settings.levers.world3D.gravity == "0.05")
        m.openMakePlanet(from: "src")
        m.name = "my-planet"
        m.pageAppeared()
        m.pageAppeared()
        #expect(m.intent == .makePlanet(from: "src") && m.name == "my-planet")
        m.openCreate()   // an explicit new world is the only thing that ends them
        #expect(m.intent == .create && m.name != "my-planet")
    }

    @Test func theCommandThatRunsIsTheCommandTheFormShowedOrNothingStarts() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        // the folder has the grown world's name, but the library has not seen it yet: the engine's naming would silently pick `-2`
        try LibraryTestFolders.research(rig.runs, "planet-1-grow")
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        rig.setup.name = "planet-1"
        let plan = rig.setup.plan
        #expect(plan.stages[0].arguments.contains("planet-1-grow") || plan.stages[0].arguments.joined(separator: " ").contains("runs/planet-1-grow"))
        let refusal = rig.coordinator.begin(plan, planet: rig.setup.stageTwoSpec(seed: "planet-1-grow"), timing: .init())
        #expect(refusal == "The command changed while it was being started. Press the button again.", "\(refusal ?? "nil")")
        #expect(rig.coordinator.plan == nil && rig.controller.job == nil)
    }

    @Test func anExitThatAlreadyCameIsNotCountedTwiceAndAnExitUnderAnotherNameFailsThePlanVisibly() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let serial = rig.controller.lastRunExit?.serial ?? 0
        // a stale exit (a serial the coordinator has already seen) changes nothing
        rig.coordinator.runEnded(RunExit(run: "planet-1-grow", status: 0, signalled: false, stopRequested: false, forced: false, problem: nil, serial: serial))
        #expect(rig.coordinator.plan?.state == .growing(stopping: false), "an old exit is not this run's")
        // an exit of the run, under another name than the one the plan holds: the plan fails in the open instead of waiting for ever
        rig.coordinator.runEnded(RunExit(run: "other", status: 0, signalled: false, stopRequested: false, forced: false, problem: nil, serial: serial + 1))
        if case .failed(.growing(let message))? = rig.coordinator.plan?.state { #expect(message.contains("other")) } else { Issue.record("the plan still waits: \(String(describing: rig.coordinator.plan?.state))") }
        #expect(!rig.coordinator.isActive)
    }

    @Test func whatWasEditedIsRememberedOnlyOnceTheStartWentThrough() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        rig.setup.name = "planet-9"
        rig.controller.settings.levers.world3D.gravity = "0.09"
        rig.coordinator.elsewhere = { ["planet-9-grow"] }   // the guard refuses at the moment of the press, whatever the form said
        rig.setup.start()
        #expect(rig.setup.refusal == "planet-9-grow is in use by another app or process." && rig.coordinator.plan == nil)
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.rememberedShown.isEmpty, "a press that was refused remembers nothing")
        rig.coordinator.elsewhere = { [] }
        rig.setup.name = "planet-9"
        rig.controller.settings.levers.world3D.gravity = "0.09"
        rig.setup.start()
        #expect(rig.coordinator.plan != nil && rig.setup.refusal == nil)
        _ = await rig.waitForState { $0 == .live }
        rig.coordinator.press(.done)
        rig.setup.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(rig.setup.rememberedShown.map(\.key) == ["physics.gravity"])
        // what a Create remembered does not ride into a Branch: it starts from its source's own values
        try rig.makeSource("src")
        await rig.scan()
        rig.setup.openBranch(from: "src")
        #expect(rig.setup.rememberedShown.isEmpty && rig.setup.rememberedSkipped.isEmpty)
        #expect(rig.controller.settings.levers.world3D.gravity == "0.03" && rig.setup.changes.isEmpty, "\(rig.controller.settings.levers.world3D.gravity)")
    }

    @Test func aLengthOfAnySizeStartsNothingThatCrashes() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try rig.makeSource("src", steps: [20_000])
        await rig.scan()
        rig.setup.openBranch(from: "src")
        for huge in [String(Int.max), "99999999999999999999999", "-5", "0", "1e9"] {
            rig.setup.setGrowSteps(huge)
            #expect(!rig.setup.reasons.isEmpty, "\(huge) is told, not started")
        }
        rig.setup.setGrowSteps("999999999")   // the largest the page accepts
        #expect(rig.setup.reasons.isEmpty, "\(rig.setup.reasons)")
        rig.setup.start()
        #expect(rig.coordinator.plan?.targetSteps == 999_999_999 + 20_000)
        _ = rig.coordinator.plan?.banner(elapsed: 5)   // the bar reads it
        rig.controller.forceStop()
    }
}
