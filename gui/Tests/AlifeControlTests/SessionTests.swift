import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The safety rules of the `alife` window (docs/APP.md PR-E): closing the window with a run going (D3), leaving the Watch page (D4), the
/// sheet that stops one world to play another (R5) and the Dock icon. Stand-in processes, no real window: the windows these tests make are
/// borderless, transparent and placed far off every display.
@MainActor
@Suite("The window, the run and the world: closing, leaving and switching", .serialized)
struct SessionTests {
    // MARK: A world that is playing, with the clock under the test's control

    @MainActor private final class WorldRig {
        let scratch: Scratch
        let play = PlayController()
        let session: WorldSession
        let shell: ShellModel
        let defaults: UserDefaults
        let suite: String
        let commands: URL
        var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }
        var model: WorldModel { session.model }

        /// `fixture` is the committed play stream the stand-in plays: `3d` ends running at max, `3d_n70` paused by the player, `3d_g3b` stopped
        /// for an alert at a chapter end.
        init(fixture: String = "3d", pickUp: Bool? = nil, page: ShellPage? = .watch) throws {
            scratch = try Scratch()
            commands = scratch.url.appendingPathComponent("commands.txt")
            script = try scratch.script("alife", "cat \"\(Repo.playFixtureURL(fixture).path)\"\nwhile IFS= read -r line; do echo \"$line\" >> \"\(commands.path)\"; done")
            suite = "alife-session-\(UUID().uuidString)"
            defaults = try #require(UserDefaults(suiteName: suite))
            if let pickUp { defaults.set(pickUp, forKey: AppShell.pickUpKey) }
            session = WorldSession(play: play, defaults: defaults)
            shell = ShellModel(session: session, defaults: defaults)
            if let page { shell.select(page) }
        }

        let script: URL

        func start() async -> Bool {
            play.start(.resume(PlayResume(run: "w")), executable: script, directory: scratch.url, arguments: [])
            return await eventually { play.phase == .live && play.store.clock != nil }
        }

        /// The world's own clock, as the host sends it: `rate`, why it is stopped, and the pauses it has armed.
        func clock(rate: Float, reason: PlayClock.Reason, chapterArmed: Bool = true, chapter: UInt32 = 1) {
            var clock = play.store.clock!
            clock.rate = rate
            clock.reason = reason
            clock.saving = false
            clock.flags = chapterArmed ? 1 : 0
            clock.chapter = chapter
            clock.head = RecordHead(step: clock.head.step + 1, seq: 0, version: 1, flags: 0)
            var delivery = StreamDelivery()
            delivery.records["CLCK"] = .clck(clock)
            play.reduce(delivery)
        }

        func lines(_ word: String) -> Int { sent.filter { $0 == word }.count }

        func settled(_ test: @MainActor () -> Bool) async -> Bool { await eventually(timeout: .seconds(10), test) }

        func finish() {
            play.shutdown()
            defaults.removePersistentDomain(forName: suite)
            scratch.cleanup()
        }
    }

    // MARK: D4: leaving Watch

    @Test func leavingWatchPausesARunningWorldAndComingBackRunsItAgain() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        #expect(rig.model.watchActive)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.lines("pause") == 1 }, "leaving Watch pauses it: \(rig.sent)")
        #expect(rig.model.awayHold == .away(world: "w"))
        #expect(rig.shell.awayHold == .away(world: "w"), "the shell's mirror, which the Worlds page and the sidebar read")
        rig.clock(rate: 0, reason: .userPause)  // the world answers
        #expect(rig.model.watchCue(startFailed: false).text == "w · paused: you left")
        rig.shell.select(.activity)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.lines("pause") == 1, "going from page to page does not pause again")
        rig.shell.select(.watch)
        #expect(await rig.settled { rig.lines("run") == 1 }, "coming back runs it: \(rig.sent)")
        #expect(rig.model.awayHold == .none)
    }

    @Test func aWorldThePlayerPausedIsNeverRunByComingBack() async throws {
        let rig = try WorldRig(fixture: "3d_n70")
        defer { rig.finish() }
        #expect(await rig.start())
        #expect(rig.play.store.clock?.rate == 0 && rig.play.store.clock?.reason == .userPause)
        rig.shell.select(.worlds)
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.sent.isEmpty, "nothing was sent either way: \(rig.sent)")
        #expect(rig.model.awayHold == .none)
    }

    @Test func aChapterOrAlertPauseIsNeverRunByComingBack() async throws {
        let rig = try WorldRig(fixture: "3d_g3b")
        defer { rig.finish() }
        #expect(await rig.start())
        #expect(rig.play.store.clock?.reason == .alert && rig.play.store.clock?.rate == 0)
        rig.shell.select(.worlds)
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.sent.isEmpty, "the card the world shows has not been read: \(rig.sent)")
        // the same for a chapter
        rig.clock(rate: 0, reason: .chapter)
        rig.shell.select(.worlds)
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(200))
        #expect(rig.sent.isEmpty)
    }

    @Test func aChapterThatTakesTheWorldWhileAwayIsNotRunOnTheWayBack() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.lines("pause") == 1 })
        // the world reached its chapter end before it read the pause
        rig.clock(rate: 0, reason: .chapter)
        #expect(rig.model.awayHold == .none, "the pause is the world's now")
        #expect(rig.model.watchCue(startFailed: false).text.contains("chapter 1 is waiting"))
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.lines("run") == 0, "run would acknowledge the chapter: \(rig.sent)")
    }

    @Test func notPickingUpLeavesItPausedAndWatchSaysWhy() async throws {
        let rig = try WorldRig(pickUp: false)
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.lines("pause") == 1 })
        rig.clock(rate: 0, reason: .userPause)
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.lines("run") == 0, "the setting says to stay paused")
        #expect(rig.model.awayHold == .left(world: "w"))
        #expect(rig.model.clockStrip.readout == "paused when you left Watch")
        #expect(rig.model.clockStrip.paused)
        // a world the player paused reads "paused"; this one says it is paused because they left
        #expect(rig.model.watchCue(startFailed: false).text == "w · paused: you left")
        // the player runs it: the note goes
        rig.play.resume()
        rig.clock(rate: 1, reason: .running)
        #expect(rig.model.awayHold == .none)
        #expect(rig.model.clockStrip.readout != "paused when you left Watch")
    }

    @Test func theSettingDefaultsToPickingUpAndReadsTheStoredChoice() throws {
        let on = try WorldRig(page: nil)
        defer { on.finish() }
        #expect(on.model.picksUpWhereLeft)
        let off = try WorldRig(pickUp: false, page: nil)
        defer { off.finish() }
        #expect(!off.model.picksUpWhereLeft)
    }

    @Test func aWorldThatGoesLiveWhileThePlayerIsOnAnotherPageIsNotLeftRunningUnseen() async throws {
        let rig = try WorldRig(page: .activity)
        defer { rig.finish() }
        #expect(!rig.model.watchActive)
        #expect(await rig.start())
        // the fixture runs at max when it starts; the player is on Activity
        #expect(await rig.settled { rig.lines("pause") == 1 }, "\(rig.sent)")
        #expect(rig.model.awayHold == .away(world: "w"))
        rig.clock(rate: 0, reason: .userPause)
        rig.shell.select(.watch)
        #expect(await rig.settled { rig.lines("run") == 1 }, "they arrive and it goes")
    }

    @Test func deactivatingTheAppIsNotLeavingAndNothingIsPaused() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.model.windowIsKey = { false }  // another app is in front: Command-Tab
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: NSApp)
        rig.clock(rate: 1, reason: .running)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.sent.isEmpty, "the world goes on: \(rig.sent)")
        #expect(rig.model.awayHold == .none)
        #expect(!rig.model.viewerIsLooking, "and the dock badge and the bounce keep their meaning: this player is not looking")
        // coming back sends nothing either
        rig.model.windowIsKey = { true }
        rig.clock(rate: 1, reason: .running)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent.isEmpty)
    }

    @Test func aNewWorldOrNoneForgetsThePause() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.model.awayHold == .away(world: "w") })
        rig.shell.closeWorld()
        #expect(rig.model.awayHold == .none && rig.shell.awayHold == .none)
        #expect(rig.play.phase == .idle)
    }

    @Test func aViewThatAttachesAgainDoesNotMakeTheAppForgetItPausedTheWorld() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.model.awayHold == .away(world: "w") })
        // a Metal view that is made again asks the model for everything, which resets the selection and the camera (found by drawing the page)
        rig.model.deliver([.header, .clock, .frame])
        #expect(rig.model.awayHold == .away(world: "w"), "or the world would stay paused for ever")
        rig.clock(rate: 0, reason: .userPause)
        rig.shell.select(.watch)
        #expect(await rig.settled { rig.lines("run") == 1 })
    }

    @Test func aPauseThatCouldNotBeWrittenIsNotHeldAndTheWorldIsNotRunOnTheWayBack() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.play.stopForQuit()  // the process is being stopped: nothing can be written to it, whatever the phase still says
        rig.shell.select(.worlds)
        try await Task.sleep(for: .milliseconds(150))
        #expect(rig.model.awayHold == .none, "no pause was made, so none is held")
        #expect(rig.lines("pause") == 0)
    }

    @Test func theClockSaysThePauseIsTheAppsWhileAwayToo() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.model.awayHold.isAway })
        rig.clock(rate: 0, reason: .userPause)
        #expect(rig.model.clockStrip.readout == "paused while you are away")
    }

    @Test func aPauseThatCouldNotBeSentIsNotHeld() {
        // no live world to write to: nothing is held that could be given back
        var model = AwayPause()
        let f = AwayPause.Facts(world: "w", live: true, rate: 1, reason: .running)
        #expect(model.leftWatch(f) == .pause)
        model.cancelPause()
        #expect(model.hold == .none)
    }

    // MARK: The state is honest everywhere the app says it

    @Test func theActivityPageAndTheSidebarSayTheSameThing() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.shell.select(.worlds)
        #expect(await rig.settled { rig.shell.awayHold.isAway })
        let facts = ActivityPage.facts(controller: RunController(storedPath: "", savePath: { _ in }, candidates: [], isApp: false), play: rig.play,
                                       stream: .shared, away: rig.shell.awayHold, picksUp: rig.model.picksUpWhereLeft)
        let item = try #require(ActivityGuide.items(facts).first { $0.kind == .world })
        #expect(item.detail.hasPrefix("Paused while you are away from Watch"))
    }

    // MARK: D3: closing the window

    private func delegate(_ run: RunController, _ play: PlayController, mode: LaunchMode = .alife) -> AppDelegate {
        let delegate = AppDelegate(run: run, play: play, quitPanel: .offscreen())
        delegate.plan = mode.plan
        delegate.replyToTerminate = {}
        delegate.terminate = {}
        delegate.stopStream = {}
        delegate.closePreference = { .ask }
        delegate.storeClosePreference = { _ in }
        return delegate
    }

    private func growing(_ rig: PipelineRig, name: String = "exp-1") async throws {
        rig.looking = true
        #expect(await rig.create(name, goal: .research).isEmpty)
        #expect(await rig.waitForGrowing(name))
    }

    @Test func withNoRunTheWindowClosesAndTheAppEndsAsBefore() throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in Issue.record("nothing to ask"); return (.cancel, false) }
        #expect(d.mainWindowShouldClose())
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "the pinned default")
    }

    @Test func aRunMakesTheWindowAskAndKeepRunningLeavesTheAppAliveWithTheRunGoing() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var asked: [CloseQuestion] = []
        d.askClose = { asked.append($0); return (.keepRunning, false) }
        var told: [String?] = []
        d.onKeptRunning = { world, _ in told.append(world) }
        #expect(d.mainWindowShouldClose(), "the window may close")
        #expect(asked.count == 1 && asked[0].title == "The run \u{2018}exp-1\u{2019} is still growing." && asked[0].keepLabel == "Keep Running")
        #expect(!d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "the run is the reason to stay")
        d.mainWindowClosed(windows: { [] })
        #expect(told.count == 1 && told[0] == nil, "no world was open")
        #expect(rig.controller.job == .run && !rig.controller.stopRequested, "the run is not touched")
        #expect(d.terminationGate.isWaiting == false)
    }

    @Test func cancelLeavesEverythingAsItWasWindowIncluded() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in (.cancel, true) }
        var stored: [ClosePreference] = []
        d.storeClosePreference = { stored.append($0) }
        #expect(!d.mainWindowShouldClose(), "the window stays")
        #expect(stored.isEmpty, "Cancel is not a choice that can be remembered")
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        #expect(rig.controller.job == .run && !rig.controller.stopRequested)
        #expect(!d.terminationGate.isWaiting)
    }

    @Test func stopAndQuitKeepsTheWindowUpAndQuitsThroughTheGateWithoutAskingTwice() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var terminated = 0
        d.terminate = { terminated += 1 }
        d.askClose = { _ in (.stopAndQuit, false) }
        var quitQuestions = 0
        d.confirmStopRun = { _ in quitQuestions += 1; return true }
        #expect(!d.mainWindowShouldClose(), "the window stays up while the quit goes on")
        #expect(await eventually { terminated == 1 }, "the quit was asked for")
        #expect(d.applicationShouldTerminate(NSApp) == .terminateLater, "through the gate and its panel")
        #expect(quitQuestions == 0, "the question was answered at the window")
        #expect(rig.controller.stopRequested)
        d.terminationGate.abandonWait()
        // the next Command-Q asks again
        let fresh = delegate(rig.controller, rig.play)
        fresh.confirmStopRun = { _ in quitQuestions += 1; return false }
        _ = fresh.applicationShouldTerminate(NSApp)
    }

    @Test func rememberingAnAnswerIsStoredAndThenNothingIsAsked() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var stored: [ClosePreference] = []
        var current = ClosePreference.ask
        d.closePreference = { current }
        d.storeClosePreference = { stored.append($0); current = $0 }
        d.askClose = { _ in (.keepRunning, true) }
        #expect(d.mainWindowShouldClose())
        #expect(stored == [.keep])
        d.askClose = { _ in Issue.record("remembered: no question"); return (.cancel, false) }
        #expect(d.mainWindowShouldClose() && !d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        // Stop and Quit is not rememberable: ticking the box and choosing it stores nothing, and an old stored "stop" asks
        stored = []
        current = .ask
        d.askClose = { _ in (.stopAndQuit, true) }
        var terminated = 0
        d.terminate = { terminated += 1 }
        #expect(!d.mainWindowShouldClose())
        #expect(stored.isEmpty, "nothing was stored")
        #expect(await eventually { terminated == 1 })
        d.terminationGate.abandonWait()
        let old = delegate(rig.controller, rig.play)
        old.closePreference = { ClosePreference(stored: "stop") }
        var asked = 0
        old.askClose = { _ in asked += 1; return (.cancel, false) }
        #expect(!old.mainWindowShouldClose() && asked == 1, "an earlier build's stored Stop and Quit is read as the question")
    }

    @Test func aRunAlreadyStoppingIsNotAskedAboutAndTheQuitWaitsForIt() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        rig.controller.stop()
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in Issue.record("it is already stopping"); return (.cancel, false) }
        #expect(d.mainWindowShouldClose())
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "the app then ends through the gate, which waits for the save point")
        #expect(await patiently { rig.controller.job == nil })
    }

    @Test func aSecondAttemptToCloseWhileTheQuestionIsUpIsTheFirstOnesBusiness() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var inner: Bool?
        d.askClose = { _ in
            inner = d.mainWindowShouldClose()  // (Command-W again while the alert is up)
            return (.cancel, false)
        }
        #expect(!d.mainWindowShouldClose())
        #expect(inner == false)
    }

    @Test func theOldAppsNeverAskAndQuitWithTheirLastWindow() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        for mode in [LaunchMode.lab, .world] {
            let d = delegate(rig.controller, rig.play, mode: mode)
            d.askClose = { _ in Issue.record("the old apps keep their behaviour"); return (.cancel, false) }
            #expect(d.mainWindowShouldClose())
            #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        }
    }

    @Test func keepingARunAliveClosesTheOpenWorldAndSavesItWhileTheRunGoesOn() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var asked: [CloseQuestion] = []
        d.askClose = { asked.append($0); return (.keepRunning, false) }
        var told: [String?] = []
        d.onKeptRunning = { world, _ in told.append(world) }
        var streams = 0
        d.stopStream = { streams += 1 }
        #expect(d.mainWindowShouldClose())
        #expect(asked[0].detail.contains("saves and closes the open world \u{2018}mini3s_a\u{2019}"))
        d.mainWindowClosed(windows: { [] })
        #expect(told == ["mini3s_a"], "the shell is told which world was closed")
        #expect(streams == 1, "the 3D viewer's stream goes with the window")
        #expect(rig.play.phase == .idle, "the world was asked to stop through its ladder")
        #expect(await patiently { !rig.play.hasProcess }, "it saved and went")
        #expect(rig.controller.job == .run, "and the run never noticed")
    }

    @Test func theShellSaysWhatBecameOfTheWorldAsItBecomesTrueOnThePageThePlayerLeft() async throws {
        let rig = try WorldRig(page: .watch)
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 0, reason: .userPause)
        // the window closes with a run going: the world is stopped, and the note says "being saved" until its process has gone
        rig.play.stop()
        rig.shell.windowClosedWithRun(world: "w")
        #expect(rig.shell.page == .worlds, "Watch has no world to show any more")
        #expect(rig.shell.closedNote?.contains("is being saved and closed") == true && rig.shell.closedNote?.contains("was saved") == false)
        #expect(await rig.settled { rig.shell.closedNote?.contains("was saved and closed") == true }, "and then that it was: \(rig.shell.closedNote ?? "")")
        // a close that had no world does not erase an unread note, and does not move the page
        rig.shell.windowClosedWithRun(world: nil)
        #expect(rig.shell.closedNote?.contains("was saved and closed") == true)
        for page in [ShellPage.setup, .activity, .worlds] {
            let other = try WorldRig(page: page)
            defer { other.finish() }
            other.shell.windowClosedWithRun(world: "w")
            #expect(other.shell.page == page, "the page the player was on stays: \(page)")
        }
        let watching = try WorldRig(page: .watch)
        defer { watching.finish() }
        watching.shell.windowClosedWithRun(world: nil)
        #expect(watching.shell.page == .watch && watching.shell.closedNote == nil, "Watch with no world in it stays Watch")
    }

    @Test func aWorldStillStartingIsNotSaidToHaveBeenSaved() throws {
        let rig = try WorldRig(page: .activity)
        defer { rig.finish() }
        rig.shell.windowClosedWithRun(world: "planet-2", starting: true)
        let note = try #require(rig.shell.closedNote)
        #expect(note.contains("still starting") && note.contains("stopped before it opened") && !note.contains("saved"))
    }

    @Test func aSaveThatDidNotEndWellIsSaidSoAndNotSaved() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // an engine that dies with an error while it "saves"
        let engine = try scratch.script("alife", "cat \"\(Repo.playFixtureURL("3d_n70").path)\"\nwhile IFS= read -r line; do if [ \"$line\" = quit ]; then echo 'disk full' >&2; exit 1; fi; done")
        let suite = "alife-failsave-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let play = PlayController()
        defer { play.shutdown() }
        let session = WorldSession(play: play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        play.start(.resume(PlayResume(run: "w")), executable: engine, directory: scratch.url, arguments: [])
        #expect(await eventually { play.phase == .live })
        play.stop()
        shell.windowClosedWithRun(world: "w")
        #expect(await eventually { shell.closedNote?.contains("did not finish saving") == true }, "\(shell.closedNote ?? "")")
        #expect(shell.closedNote?.contains("exited with code 1: disk full") == true && shell.closedNote?.contains("was saved") == false)
        #expect(play.lastRetiredEnd?.saved == false && play.lastRetiredEnd?.reason?.contains("code 1") == true)
    }

    @Test func aRememberedKeepRunningAloneKeepsTheAppAlive() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        d.closePreference = { .keep }
        d.askClose = { _ in Issue.record("remembered"); return (.cancel, false) }
        #expect(d.mainWindowShouldClose())
        #expect(!d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "a fresh delegate: only the remembered answer says to stay")
    }

    @Test func aWorldThatIsNoLongerOpenIsNotMentionedAndNotToldAbout() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        rig.play.stop()  // its name stays on the controller; it is not open
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var asked: [CloseQuestion] = []
        d.askClose = { asked.append($0); return (.keepRunning, false) }
        var told: [String?] = []
        d.onKeptRunning = { world, _ in told.append(world) }
        #expect(d.closeFacts.world == nil)
        #expect(d.mainWindowShouldClose())
        #expect(!asked[0].detail.contains("mini3s_a"))
        d.mainWindowClosed(windows: { [] })
        #expect(told.count == 1 && told[0] == nil)
    }

    // MARK: Reopening

    private func window(_ id: String?) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        window.identifier = id.map { NSUserInterfaceItemIdentifier($0) }
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.alphaValue = 0
        window.orderFrontRegardless()
        return window
    }

    @Test func aWindowlessAppIsBroughtBackByADockClickAndByAnyCommandThatNeedsAWindow() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        var opened = 0
        d.reopenMainWindow = { opened += 1 }
        d.windowsProvider = { [] }
        d.askClose = { _ in (.keepRunning, false) }
        #expect(d.mainWindowShouldClose())
        d.mainWindowClosed(windows: { [] })
        #expect(!d.mainWindowIsShown)
        #expect(d.applicationShouldHandleReopen(NSApp, hasVisibleWindows: false) == false)
        #expect(opened == 1, "a Dock click opens it")
        #expect(d.showMainWindowIfNeeded() && opened == 2, "and so does a menu command that names a page")
        let main = window(AppShell.mainWindowID)
        defer { main.close() }
        d.windowsProvider = { [main] }
        #expect(d.mainWindowIsShown)
        #expect(!d.showMainWindowIfNeeded() && opened == 2, "with the window showing there is nothing to open")
        #expect(d.applicationShouldHandleReopen(NSApp, hasVisibleWindows: true))
        // the page that was chosen is still chosen and the run is still there
        #expect(rig.controller.job == .run)
    }

    @Test func anAppWithNoReopenClosureOrAnOldModeDoesNothing() throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        let d = delegate(rig.controller, rig.play)
        #expect(!d.showMainWindowIfNeeded(), "no closure yet (the window never appeared)")
        let old = delegate(rig.controller, rig.play, mode: .lab)
        old.reopenMainWindow = { Issue.record("the old apps keep AppKit's own") }
        #expect(!old.showMainWindowIfNeeded())
    }

    // MARK: The close guard of a real window

    @Test func theCloseGuardAsksBeforeTheWindowClosesAndForwardsEverythingElse() throws {
        final class Original: NSObject, NSWindowDelegate {
            var vetoes = false
            var moved = 0
            func windowShouldClose(_ sender: NSWindow) -> Bool { !vetoes }
            func windowDidMove(_ notification: Notification) { moved += 1 }
        }
        var asked = 0
        var answer = true
        let original = Original()
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100), styleMask: [.titled], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        window.delegate = original
        let observer = WindowCloseObserver.ObserverView(frame: .zero)
        observer.shouldClose = { asked += 1; return answer }
        window.contentView = observer
        #expect(window.delegate is CloseGuardDelegate, "the guard is in front of SwiftUI's own delegate")
        #expect(window.delegate?.windowShouldClose?(window) == true && asked == 1)
        answer = false
        #expect(window.delegate?.windowShouldClose?(window) == false && asked == 2, "the answer is the app's")
        original.vetoes = true
        answer = true
        #expect(window.delegate?.windowShouldClose?(window) == false, "the window's own delegate can still say no")
        #expect(asked == 2, "and then the app is not asked")
        // everything else goes to the original
        let delegate = try #require(window.delegate as? NSObject)
        #expect(delegate.responds(to: #selector(NSWindowDelegate.windowDidMove(_:))))
        window.delegate?.windowDidMove?(Notification(name: NSWindow.didMoveNotification))
        #expect(original.moved == 1)
        // SwiftUI puts its delegate back: the observer puts the guard in front again, once
        window.delegate = original
        observer.installCloseGuard()
        #expect(window.delegate is CloseGuardDelegate)
        let installed = window.delegate
        observer.installCloseGuard()
        #expect(window.delegate === installed)
        window.contentView = nil
    }

    @Test func aWindowWithoutAGuardKeepsItsOwnDelegate() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100), styleMask: [.titled], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        let observer = WindowCloseObserver.ObserverView(frame: .zero)
        window.contentView = observer
        #expect(window.delegate == nil, "no shouldClose, no guard")
        window.contentView = nil
    }

    // MARK: Every state of the pipeline against closing the window

    @Test func stageOneKeptRunningThenFinishingWithNoWindowParksBouncesOnceAndStartsNothing() async throws {
        let rig = try PipelineRig(mode: "brief:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in (.keepRunning, false) }
        #expect(d.mainWindowShouldClose())
        rig.looking = false  // the window is closed
        d.mainWindowClosed(windows: { [] })
        #expect(!d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        // it ends by itself with nobody looking
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.notVisible, step: 30_000))
        #expect(rig.bounces == 1, "one bounce, no more")
        try await Task.sleep(for: .milliseconds(600))
        #expect(rig.play.phase == .idle && !rig.exists("planet-1"), "stage two never starts invisibly, and a windowless app does not start it")
        // the window comes back: the bar says what is waiting, and still nothing starts
        rig.looking = true
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.play.phase == .idle)
        #expect(rig.coordinator.banner?.buttons == [.makePlanet, .keepGrowing, .done])
        rig.coordinator.press(.makePlanet)
        #expect(await rig.waitForState { $0 == .live })
    }

    @Test func stageOneStoppedByTheWindowsOwnQuitParksAndTheNextLaunchShowsTheNextAction() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in (.stopAndQuit, false) }
        d.confirmStopRun = { _ in Issue.record("answered at the window"); return false }
        rig.coordinator.isQuitting = { d.terminationGate.isWaiting }
        var terminated = 0
        d.terminate = { terminated += 1 }
        #expect(!d.mainWindowShouldClose())
        #expect(await eventually { terminated == 1 })
        #expect(d.applicationShouldTerminate(NSApp) == .terminateLater)
        #expect(await patiently { rig.controller.job == nil })
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } }, "the run ended at a save point")
        try await Task.sleep(for: .milliseconds(400))
        #expect(rig.play.phase == .idle, "a quit starts nothing")
        d.terminationGate.abandonWait()
    }

    @Test func closingWhileThePlanetIsBeingMadeAsksAndTheQuitCancelsIt() async throws {
        let rig = try PipelineRig(mode: "ok:slow")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { $0 == .making(cancelling: false) })
        #expect(await patiently { rig.coordinator.plan?.making.phase != .starting })
        let d = delegate(rig.controller, rig.play)
        var asked: [CloseQuestion] = []
        d.askClose = { asked.append($0); return (.cancel, false) }
        #expect(!d.mainWindowShouldClose(), "Cancel: the window stays and the planet goes on")
        #expect(asked.count == 1 && asked[0].title == "Stop making \u{2018}planet-1\u{2019} and quit?" && !asked[0].offersKeepRunning)
        #expect(rig.play.phase == .starting && rig.coordinator.plan?.state == .making(cancelling: false))
        // Stop and Quit: the window stays up for the quit, which interrupts the planet being built
        d.askClose = { asked.append($0); return (.stopAndQuit, true) }
        var stored: [ClosePreference] = []
        d.storeClosePreference = { stored.append($0) }
        var terminated = 0
        d.terminate = { terminated += 1 }
        #expect(!d.mainWindowShouldClose())
        #expect(stored.isEmpty && asked.count == 2)
        #expect(await eventually { terminated == 1 })
        rig.looking = false
        rig.coordinator.isQuitting = { d.terminationGate.isWaiting }
        #expect(d.applicationShouldTerminate(NSApp) == .terminateLater, "the gate interrupts the planet being built, and asks nothing more")
        #expect(await patiently { !rig.play.hasProcess }, "the planet being built was interrupted and has gone")
        #expect(rig.play.phase != .live)
        d.terminationGate.abandonWait()
    }

    @Test func leavingTheWatchPageNeverStopsARunOrStartsAnything() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let session = WorldSession(play: rig.play, defaults: try #require(UserDefaults(suiteName: "alife-leave-\(UUID().uuidString)")))
        let shell = ShellModel(session: session, defaults: UserDefaults(suiteName: "alife-leave2-\(UUID().uuidString)")!)
        shell.select(.watch)
        shell.select(.worlds)
        shell.select(.activity)
        shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.controller.job == .run && rig.coordinator.plan?.state == .growing(stopping: false), "R4 pauses a played world; a run is not paused")
        #expect(rig.play.phase == .idle)
    }

    // MARK: The reasons a person is told

    @Test func quittingAfterTheWindowWasClosedWithKeepRunningStillAsksTheQuitQuestion() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in (.keepRunning, false) }
        #expect(d.mainWindowShouldClose())
        d.mainWindowClosed(windows: { [] })
        var asked: [String] = []
        d.confirmStopRun = { asked.append($0); return false }
        #expect(d.applicationShouldTerminate(NSApp) == .terminateCancel, "Cancel on the Dock's Quit keeps the run and the app")
        #expect(asked == ["exp-1"])
        #expect(!rig.controller.stopRequested && !d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "still alive, still running")
    }

    @Test func aLaterCloseWithNoRunQuitsAgain() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig)
        let d = delegate(rig.controller, rig.play)
        d.askClose = { _ in (.keepRunning, false) }
        #expect(d.mainWindowShouldClose() && !d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        rig.controller.stop()
        #expect(await patiently { rig.controller.job == nil })
        #expect(d.mainWindowShouldClose(), "no run now")
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "the flag from the last close does not outlive it")
    }

    // MARK: Review fixes (PR-E review: M1, M5, L1)

    /// The reviewer's test (M5): the player pauses and leaves before the clock says so; the app must not run what it did not pause.
    @Test func aPauseTheClockHasNotConfirmedYetIsNotTheAppsToUndo() async throws {
        let rig = try WorldRig()
        defer { rig.finish() }
        #expect(await rig.start())
        rig.clock(rate: 1, reason: .running)
        rig.play.pause()  // the player's own pause; the clock has not said so yet
        rig.shell.select(.worlds)
        rig.clock(rate: 0, reason: .userPause)
        rig.shell.select(.watch)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.lines("run") == 0, "the player paused it: \(rig.sent)")
        #expect(rig.model.awayHold == .none)
    }

    /// Calls `mainWindowShouldClose` as AppKit does, from the run loop and not from inside an actor job, so that what happens while the alert
    /// is up (a run's exit) can be delivered.
    private func closeAsAppKitDoes(_ d: AppDelegate) async -> Bool {
        await withCheckedContinuation { c in
            RunLoop.main.perform(inModes: [.common]) { c.resume(returning: MainActor.assumeIsolated { d.mainWindowShouldClose() }) }
        }
    }

    private func spinUntil(_ seconds: Double = 10, _ condition: () -> Bool) {
        let until = Date().addingTimeInterval(seconds)
        while !condition(), Date() < until { RunLoop.current.run(until: Date().addingTimeInterval(0.05)) }
    }

    /// The reviewer's test (M1): the run ends while the question is up.
    @Test func aRunThatEndsWhileTheQuestionIsUpLeavesNothingToKeep() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig, name: "exp-1")
        let d = delegate(rig.controller, rig.play)
        var asked = 0
        d.askClose = { _ in
            asked += 1
            rig.controller.stop()  // the run ends while the person reads the question
            self.spinUntil { rig.controller.job == nil }
            return (.keepRunning, false)
        }
        let closed = await closeAsAppKitDoes(d)
        #expect(closed)
        #expect(asked == 1 && rig.controller.job == nil, "not asked again: nothing is left to ask about")
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "no run is going: the app ends with its window, it does not wait for nothing")
    }

    @Test func aPlanetThatBeganWhileTheQuestionWasUpIsAskedAboutAndNotInterruptedBehindKeepRunning() async throws {
        let rig = try PipelineRig(mode: "brief:slow")
        defer { rig.finish() }
        rig.looking = true
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForGrowing())
        let d = delegate(rig.controller, rig.play)
        var asked: [CloseQuestion] = []
        d.askClose = { question in
            asked.append(question)
            if asked.count == 1 {
                // stage 1 ends and stage 2 starts while the person reads the first question
                self.spinUntil(30) { rig.play.phase == .starting }
                return (.keepRunning, false)
            }
            return (.cancel, false)
        }
        let closed = await closeAsAppKitDoes(d)
        #expect(!closed, "the second question was answered Cancel: the window stays")
        #expect(asked.count == 2)
        #expect(asked[0].offersKeepRunning && asked[1].title == "Stop making \u{2018}planet-1\u{2019} and quit?" && !asked[1].offersKeepRunning)
        #expect(rig.play.phase == .starting && rig.play.world == "planet-1", "Keep Running did not interrupt the planet, and nothing claimed it was saved")
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp))
        d.askClose = { _ in (.stopAndQuit, false) }
        var terminated = 0
        d.terminate = { terminated += 1 }
        #expect(!(await closeAsAppKitDoes(d)))
        #expect(await eventually { terminated == 1 })
        d.terminationGate.abandonWait()
    }

    @Test func aCancelIsFinalEvenIfTheRunEndedWhileTheQuestionWasUp() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig, name: "exp-1")
        let d = delegate(rig.controller, rig.play)
        var asked = 0
        d.askClose = { _ in
            asked += 1
            rig.controller.stop()
            self.spinUntil { rig.controller.job == nil }
            return (.cancel, false)
        }
        #expect(!(await closeAsAppKitDoes(d)), "Cancel: the window stays, and the person is not asked again")
        #expect(asked == 1)
    }

    @Test func aRunThatBeganToStopWhileTheQuestionWasUpIsNotKept() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try await growing(rig, name: "exp-1")
        let d = delegate(rig.controller, rig.play)
        var asked = 0
        d.askClose = { _ in
            asked += 1
            rig.controller.stop()  // Stop pressed (Activity, Command-period) while the question is up
            return (.keepRunning, false)
        }
        #expect(await closeAsAppKitDoes(d))
        #expect(asked == 1)
        #expect(d.applicationShouldTerminateAfterLastWindowClosed(NSApp), "a run that is stopping is not kept: the quit waits for its save point")
        #expect(await patiently { rig.controller.job == nil })
    }

    /// A window that says it is minimised (a borderless one cannot be): the Dock click must take it out of the Dock.
    private final class MinimisedWindow: NSWindow {
        var minimised = true
        var restored = 0
        override var isMiniaturized: Bool { minimised }
        override var isVisible: Bool { true }  // (whatever it reports, the check is for `isMiniaturized`)
        override func deminiaturize(_ sender: Any?) {
            minimised = false
            restored += 1
        }
    }

    @Test func aDockClickBringsAMinimisedWindowOutOfTheDockInsteadOfOpeningANewOne() throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        let d = delegate(rig.controller, rig.play)
        var opened = 0
        d.reopenMainWindow = { opened += 1 }
        let window = MinimisedWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        window.identifier = NSUserInterfaceItemIdentifier(AppShell.mainWindowID)
        d.windowsProvider = { [window] }
        #expect(!d.mainWindowIsShown, "a window in the Dock is not shown")
        #expect(d.applicationShouldHandleReopen(NSApp, hasVisibleWindows: false) == false)
        #expect(window.restored == 1 && opened == 0, "taken out of the Dock; no second window asked for")
        #expect(d.mainWindowIsShown)
        #expect(d.applicationShouldHandleReopen(NSApp, hasVisibleWindows: true) == true && window.restored == 1)
        window.minimised = true
        #expect(d.showMainWindowIfNeeded() && window.restored == 2, "a menu command with the window in the Dock does the same")
    }

    @Test func howAWorldThatWasAskedToStopEndedIsToldWithoutGuessing() {
        func end(_ status: Int32, signalled: Bool = false, starting: Bool = false, detail: String? = nil) -> RetiredEnd {
            RetiredEnd(world: "w", starting: starting, status: status, signalled: signalled, detail: detail, serial: 1)
        }
        #expect(end(0).saved && end(0).reason == nil, "quit, then exit 0: saved")
        #expect(end(130).saved, "the runner's own interrupt path saves too")
        #expect(!end(1).saved && end(1, detail: "disk full").reason == "it exited with code 1: disk full")
        #expect(!end(0, signalled: true).saved && end(0, signalled: true).reason == "it was ended before it was done", "a signal may have cut the save, whatever the number says")
        #expect(!end(130, signalled: true).saved)
        #expect(!end(0, starting: true).saved && end(0, starting: true).reason == nil, "a world that was still starting saved nothing, and failed at nothing")
        #expect(!end(137).saved)
    }

    @Test func anEndThatWasNotTheClosedWorldsOwnDoesNotChangeItsNote() throws {
        let rig = try WorldRig(page: .activity)
        defer { rig.finish() }
        rig.shell.windowClosedWithRun(world: "w")
        let saving = rig.shell.closedNote
        rig.shell.retiredEnded(RetiredEnd(world: "other", starting: false, status: 0, signalled: false, detail: nil, serial: 9))
        #expect(rig.shell.closedNote == saving, "another world's end")
        rig.shell.retiredEnded(RetiredEnd(world: "w", starting: false, status: 0, signalled: false, detail: nil, serial: 0))
        #expect(rig.shell.closedNote == saving, "an end that was already over when the window closed")
        rig.shell.retiredEnded(RetiredEnd(world: "W", starting: false, status: 0, signalled: false, detail: nil, serial: 1))
        #expect(rig.shell.closedNote?.contains("was saved and closed") == true, "its own end, whatever the capitals")
    }
}
