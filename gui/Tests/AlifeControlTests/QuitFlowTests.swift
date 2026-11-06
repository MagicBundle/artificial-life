import AlifeCore
import AlifeTestSupport
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The additions to the quit of docs/APP.md §5.4: `abandonWait()`, `onWaiting`, the pre-confirmation before the gate, a run held by
/// the gate, the escalation in `shutdown`, and the quit panel. Stubs and real shell scripts; the shared controllers are not touched
/// (the delegate's own checks run only where the shared state has no process).
@MainActor
@Suite(.serialized) struct QuitFlowTests {
    private final class Stub {
        var running = true
        var stops = 0
        var replies = 0
        var waits: [Duration] = []
    }

    // MARK: The gate

    @Test func abandoningTheWaitRepliesAtOnceAndOnlyOnce() async {
        let gate = TerminationGate(timeout: .seconds(30), poll: .milliseconds(10))
        let stub = Stub()
        let reply = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(reply == .terminateLater && gate.isWaiting)
        try? await Task.sleep(for: .milliseconds(60))
        #expect(stub.replies == 0)
        gate.abandonWait()
        #expect(stub.replies == 1 && !gate.isWaiting)
        gate.abandonWait()
        stub.running = false
        try? await Task.sleep(for: .milliseconds(100))
        #expect(stub.replies == 1, "the cancelled wait does not reply again")
        #expect(stub.stops == 1)
    }

    @Test func abandoningWhenNothingWaitsDoesNothing() {
        let gate = TerminationGate()
        gate.abandonWait()
        #expect(!gate.isWaiting)
    }

    @Test func aGateThatTimedOutCanWaitAgainAndAbandonDoesNotReplyTwice() async {
        let gate = TerminationGate(timeout: .milliseconds(60), poll: .milliseconds(10))
        let stub = Stub()
        _ = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(await eventually { stub.replies == 1 })
        gate.abandonWait()
        #expect(stub.replies == 1)
    }

    @Test func theWaitReportsItsElapsedTimeUntilItEnds() async {
        let gate = TerminationGate(timeout: .seconds(30), poll: .milliseconds(10))
        let stub = Stub()
        gate.onWaiting = { elapsed in stub.waits.append(elapsed) }
        _ = gate.shouldTerminate(hasProcess: { stub.running }, stop: {}, reply: { stub.replies += 1 })
        #expect(await eventually { stub.waits.count >= 5 })
        #expect(stub.waits.first == .zero || stub.waits.first! < .milliseconds(50))
        #expect(stub.waits == stub.waits.sorted(), "elapsed time only grows")
        stub.running = false
        #expect(await eventually { stub.replies == 1 })
        let count = stub.waits.count
        try? await Task.sleep(for: .milliseconds(80))
        #expect(stub.waits.count == count, "nothing is reported after the reply")
    }

    @Test func noWaitMeansNoReport() {
        let gate = TerminationGate()
        let stub = Stub()
        stub.running = false
        gate.onWaiting = { stub.waits.append($0) }
        #expect(gate.shouldTerminate(hasProcess: { false }, stop: {}, reply: {}) == .terminateNow)
        #expect(stub.waits.isEmpty)
    }

    // MARK: The delegate

    @Test func aRunMakesTheDelegateAskFirstAndCancelKeepsEverything() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        run.settings.name = "quit-test-run"
        run.start()
        defer { run.forceStop() }
        #expect(run.hasRunJob && run.runningRunName == "quit-test-run")

        let delegate = makeDelegate(run, PlayController())
        var asked: [String] = []
        delegate.confirmStopRun = { asked.append($0); return false }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateCancel)
        #expect(asked == ["quit-test-run"])
        #expect(!run.stopRequested, "Cancel stops nothing")
        #expect(!delegate.terminationGate.isWaiting)
    }

    @Test func confirmingStopsTheRunThroughTheGateAndTheLastResortEndsWhatStays() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // An engine that ignores SIGINT, so that only the escalation can end it.
        let repo = try fakeRepo(scratch, engine: Self.stubborn)
        let run = controller(repo)
        run.settings.name = "quit-test-stubborn"
        run.start()
        defer { run.forceStop() }
        #expect(run.hasRunJob)
        await ignoresInterrupt(repo)

        let delegate = makeDelegate(run, PlayController())
        delegate.confirmStopRun = { _ in true }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(run.stopRequested, "the gate asked the run to stop (SIGINT)")
        #expect(delegate.terminationGate.isWaiting)
        try? await Task.sleep(for: .milliseconds(300))
        #expect(run.hasRunJob, "SIGINT is ignored: still there")
        // A second quit while waiting is not asked again and does not stop twice.
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        delegate.terminationGate.abandonWait()  // (the reply to AppKit is harmless outside a quit)
        // applicationWillTerminate: the escalation ends it.
        run.shutdown(escalating: true)
        #expect(await eventually(timeout: .seconds(10)) { !run.hasRunJob })
    }

    @Test func withoutEscalationShutdownOnlyAsksOnceToStop() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: Self.stubborn)
        let run = controller(repo)
        run.settings.name = "quit-test-old"
        run.start()
        defer { run.forceStop() }
        await ignoresInterrupt(repo)
        run.shutdown()  // the behaviour from before: one SIGINT, nothing more
        #expect(run.stopRequested)
        try? await Task.sleep(for: .milliseconds(500))
        #expect(run.hasRunJob, "a process that ignores SIGINT is left alone")
    }

    @Test func aSetupOrCheckJobIsEndedByTheEscalationToo() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        run.settings.mode = .resume
        run.refreshRuns()
        run.settings.sourceRun = "src"
        run.inspect()
        defer { run.forceStop() }
        guard run.job == .inspect else { Issue.record("the check did not start: \(run.logText)"); return }
        #expect(!run.hasRunJob, "a check does not hold the quit")
        run.shutdown(escalating: true)
        #expect(await eventually(timeout: .seconds(10)) { run.job == nil })
    }

    @Test func withNothingGoingTheDelegateAsksNothing() {
        let run = RunController(storedPath: nil, savePath: { _ in }, candidates: [], isApp: false)
        let delegate = makeDelegate(run, PlayController())
        var asked = 0
        delegate.confirmStopRun = { _ in asked += 1; return true }
        #expect(!run.hasRunJob)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateNow)
        #expect(asked == 0)
    }

    // MARK: The panel

    @Test func thePanelShowsUpdatesOffersQuitNowAndCloses() {
        let panel = QuitPanel.offscreen()
        var quits = 0
        panel.onQuitNow = { quits += 1 }
        panel.show(QuitPanelModel(worlds: ["storm"], elapsedSeconds: 1, lossMinutes: 7))
        #expect(panel.isVisible)
        #expect(panel.texts.title == "Saving 'storm'...")
        #expect(panel.texts.quitNowHidden)
        #expect(panel.texts.warning.contains("6000 steps"), "the cost is there from the start")
        panel.show(QuitPanelModel(worlds: ["storm"], elapsedSeconds: 12, lossMinutes: 7))
        #expect(panel.texts.detail.contains("12 s of at most 40 s"))
        #expect(!panel.texts.quitNowHidden)
        #expect(panel.texts.warning.contains("7 minutes"))
        let panelWindow = panel.window
        // Not on any display after either show: becoming key (the Quit now line appearing) brought this panel onto the screen in the parallel runs.
        let onADisplay = NSScreen.screens.contains { $0.frame.intersects(panelWindow?.frame ?? .zero) }
        #expect(!onADisplay && panelWindow?.alphaValue == 0 && panelWindow?.isKeyWindow == false,
                "the quit tests never put the panel on a display: \(String(describing: panelWindow?.frame)) alpha \(String(describing: panelWindow?.alphaValue)) screens \(NSScreen.screens.map(\.frame))")
        #expect(panelWindow is NSPanel && (panelWindow as? NSPanel)?.worksWhenModal == true)
        #expect(panelWindow?.styleMask.contains(.closable) == false, "no way to close it but the quit ending")
        panel.close()
        #expect(!panel.isVisible)
    }

    @Test func quitNowHasAKeyboardRouteAsWellAsTheMouse() throws {
        let panel = QuitPanel.offscreen()
        var quits = 0
        panel.onQuitNow = { quits += 1 }
        panel.show(QuitPanelModel(worlds: ["storm"], elapsedSeconds: 11, lossMinutes: 7))
        defer { panel.close() }
        let window = try #require(panel.window)
        func buttons(_ view: NSView) -> [NSButton] { ((view as? NSButton).map { [$0] } ?? []) + view.subviews.flatMap(buttons) }
        let content = try #require(window.contentView)
        let button = try #require(buttons(content).first { $0.title == "Quit now without saving" })
        #expect(window.initialFirstResponder === button, "Space presses it once the panel is key")
        #expect(button.keyEquivalent == "q" && button.keyEquivalentModifierMask == [.command, .option])
        #expect(!button.isHidden && button.toolTip?.contains("Option-Command-Q") == true)
        // (A key event is not sent through the window here: with the panel key, AppKit also offers it to the application's menu,
        // and a stray Command-Q must not reach the test process. The key route is for the owner's hands.)
        // (performClick is avoided: with it the test process exited silently in this suite, cause not found.)
        #expect(button.sendAction(button.action, to: button.target))
        #expect(quits == 1)
    }

    // MARK: The reason, the limit, the other jobs, the sheets, the last resort

    @Test func theEndOfTheSessionStopsTheRunWithoutAskingAnything() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: Self.stubborn)
        let run = controller(repo)
        run.settings.name = "session-end"
        run.start()
        defer { run.forceStop() }
        await_ignoresInterruptSync(repo)
        let delegate = makeDelegate(run, PlayController())
        var asked = 0
        delegate.confirmStopRun = { _ in asked += 1; return false }
        delegate.quitReason = { .sessionEnd }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(asked == 0, "a question nobody answers would hold the log out")
        #expect(run.stopRequested, "the run was asked to stop all the same")
        delegate.terminationGate.abandonWait()
        run.shutdown(escalating: true)
    }

    @Test func aRunWaitsUpToTwoMinutesAndTheOthersFortySeconds() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        run.settings.name = "limit-run"
        run.start()
        defer { run.forceStop() }
        let delegate = makeDelegate(run, PlayController())
        delegate.confirmStopRun = { _ in true }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        let model = try #require(delegate.quitPanelModel(after: .seconds(3)))
        #expect(model.limitSeconds == 120 && model.hasRun && model.runName == "limit-run")
        #expect(model.cost.contains("after 120 s"))
        delegate.terminationGate.abandonWait()
        run.forceStop()
    }

    @Test func theGateTakesItsLimitFromTheCallerAndAStaleWaitCannotEndANewOne() async {
        let gate = TerminationGate(timeout: TestTime.limit(.seconds(30)), poll: .milliseconds(10))
        gate.limit = { .milliseconds(120) }
        let stub = Stub()
        let began = ContinuousClock.now
        _ = gate.shouldTerminate(hasProcess: { stub.running }, stop: {}, reply: { stub.replies += 1 })
        #expect(await eventually { stub.replies == 1 })
        #expect(ContinuousClock.now - began < TestTime.limit(.seconds(5)), "the 120 ms limit, not the 30 s of init")
        // A wait abandoned and a new one begun at once: the old task must not reply for the new one.
        gate.limit = nil
        let again = Stub()
        _ = gate.shouldTerminate(hasProcess: { again.running }, stop: {}, reply: { again.replies += 1 })
        gate.abandonWait()
        #expect(again.replies == 1)
        let third = Stub()
        _ = gate.shouldTerminate(hasProcess: { third.running }, stop: {}, reply: { third.replies += 1 })
        try? await Task.sleep(for: .milliseconds(150))
        #expect(third.replies == 0, "the abandoned wait's task woke up and left the new wait alone")
        gate.abandonWait()
    }

    @Test func aSetupOrCheckJobGetsSIGINTAndTheWaitWithNoQuestion() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // A check that ends cleanly on SIGINT (as uv and `alife inspect` do).
        let repo = try fakeRepo(scratch, engine: "trap 'exit 130' INT\ntouch \"$(dirname \"$0\")/ready\"\nwhile kill -0 $PPID 2>/dev/null && [ $SECONDS -lt 120 ]; do sleep 0.1; done")
        let run = controller(repo)
        run.settings.mode = .resume
        run.refreshRuns()
        run.settings.sourceRun = "src"
        run.inspect()
        defer { run.forceStop() }
        guard run.job == .inspect else { Issue.record("the check did not start: \(run.logText)"); return }
        let ready = repo.appendingPathComponent(".venv/bin/ready").path
        #expect(await eventually(timeout: .seconds(10)) { FileManager.default.fileExists(atPath: ready) })
        let delegate = makeDelegate(run, PlayController())
        var asked = 0
        delegate.confirmStopRun = { _ in asked += 1; return true }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater, "the quit waits for it")
        #expect(asked == 0, "no question for a check or a setup")
        #expect(run.stopRequested)
        let model = try #require(delegate.quitPanelModel(after: .seconds(1)))
        #expect(model.other == "the check" && !model.hasRun && model.limitSeconds == 40)
        #expect(await eventually(timeout: .seconds(10)) { !run.hasJob }, "SIGINT ended it, no SIGTERM needed")
        #expect(run.lastExit?.text.contains("130") == true)
        delegate.terminationGate.abandonWait()
    }

    @Test func theLastResortOfTheDelegateEndsWhatTheWaitLeft() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: Self.stubborn)
        let run = controller(repo)
        run.settings.name = "orphan-test"
        run.start()
        defer { run.forceStop() }
        await ignoresInterrupt(repo)
        let delegate = makeDelegate(run, PlayController())
        delegate.confirmStopRun = { _ in true }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        delegate.terminationGate.abandonWait()  // "Quit now without saving"
        #expect(run.hasRunJob, "SIGINT is ignored: the run is still there when the app is about to go")
        delegate.applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
        #expect(await eventually(timeout: .seconds(10)) { !run.hasJob }, "applicationWillTerminate must end it: no orphan")
    }

    @Test func thePanelModelDecisions() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let world = repo.appendingPathComponent("runs/w1")
        try FileManager.default.createDirectory(at: world, withIntermediateDirectories: true)
        try """
            {"play_version": 1, "base_rate": 10.0, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
             "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": null, "planet": null}
            """.write(to: world.appendingPathComponent("play.json"), atomically: true, encoding: .utf8)
        let play = PlayController()
        play.ladder = EngineSession.Ladder(beforeInterrupt: nil, beforeTerminate: .seconds(30))
        play.start(.resume(PlayResume(run: "w1")), executable: repo.appendingPathComponent(".venv/bin/alife"), directory: repo,
                   arguments: [])
        defer { play.shutdown() }
        let delegate = makeDelegate(controller(repo), play)
        #expect(delegate.quitPanelModel(after: .milliseconds(100)) == nil, "a quick save shows nothing")
        play.stopForQuit()
        #expect(play.retiredWorlds == ["w1"])
        #expect(delegate.quitPanelModel(after: .milliseconds(399)) == nil)
        let model = try #require(delegate.quitPanelModel(after: .milliseconds(1500)))
        #expect(model.worlds == ["w1"] && !model.hasRun && model.other == nil)
        #expect(model.elapsedSeconds == 1 && model.lossMinutes == 10, "from the world's own play.json")
        try FileManager.default.removeItem(at: world.appendingPathComponent("play.json"))
        #expect(delegate.quitPanelModel(after: .seconds(2))?.lossMinutes == 10, "read once, not at every poll")
    }

    @Test func aQuitAskedWhileTheQuestionIsUpIsTheFirstOnesBusiness() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        run.settings.name = "reentrant"
        run.start()
        defer { run.forceStop() }
        let delegate = makeDelegate(run, PlayController())
        var inner: NSApplication.TerminateReply?
        var asked = 0
        delegate.confirmStopRun = { _ in
            asked += 1
            inner = delegate.applicationShouldTerminate(NSApplication.shared)  // the Dock's Quit, in the modal loop
            return false
        }
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateCancel)
        #expect(inner == .terminateCancel && asked == 1, "no second alert, no wait started behind the question")
        #expect(!delegate.terminationGate.isWaiting && !run.stopRequested)
    }

    /// A window that has a sheet, as the launcher sheet and the preset comparison are, without AppKit attaching one: a real sheet moves its
    /// window onto a screen (seen: both flew to the bottom-left corner of the display for half a second), and a test must never put a window
    /// on anyone's display. The quit code asks the window for its `sheets` and ends them with `endSheet`, which is all this stands in for.
    private final class SheetHost: QuietWindow {
        var attached: [NSWindow] = []
        override var sheets: [NSWindow] { attached }
        override func endSheet(_ sheet: NSWindow) { attached.removeAll { $0 === sheet } }
        override func endSheet(_ sheet: NSWindow, returnCode: NSApplication.ModalResponse) { attached.removeAll { $0 === sheet } }
    }

    private func windowWithSheet() -> (NSWindow, NSWindow) {
        let window = SheetHost(contentRect: NSRect(x: -20000, y: -20000, width: 300, height: 200), styleMask: [.borderless], backing: .buffered, defer: true)
        let sheet = QuietWindow(contentRect: NSRect(x: -20000, y: -20000, width: 200, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        sheet.isReleasedWhenClosed = false
        window.attached = [sheet]
        return (window, sheet)
    }

    @Test func theQuitCommandEndsAttachedSheetsThenTerminates() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        let delegate = makeDelegate(run, PlayController())
        var terminated = 0
        var sheetsWhenAsked = -1
        let (window, sheet) = windowWithSheet()
        defer { window.close() }
        delegate.terminate = { terminated += 1; sheetsWhenAsked = window.sheets.count }
        #expect(window.sheets.contains(sheet))
        delegate.requestQuit()
        #expect(await eventually { terminated == 1 })
        #expect(sheetsWhenAsked == 0, "AppKit ignores terminate while a sheet is attached: it is asked for after they went")
    }

    @Test func cancelOnTheQuestionLeavesTheSheetsWhereTheyWere() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let repo = try fakeRepo(scratch, engine: "exec sleep 30")
        let run = controller(repo)
        run.settings.name = "sheet-cancel"
        run.start()
        defer { run.forceStop() }
        let delegate = makeDelegate(run, PlayController())
        var terminated = 0, asked = 0
        delegate.terminate = { terminated += 1 }
        delegate.confirmStopRun = { _ in asked += 1; return false }
        let (window, sheet) = windowWithSheet()
        defer { window.close() }
        delegate.requestQuit()
        #expect(asked == 1 && window.sheets.contains(sheet), "Cancel: the sheet is still there")
        try? await Task.sleep(for: .milliseconds(100))
        #expect(terminated == 0)

        // Stop and Quit: the sheets go, terminate follows, and the question is not asked a second time.
        delegate.confirmStopRun = { _ in asked += 1; return true }
        delegate.requestQuit()
        #expect(asked == 2)
        #expect(await eventually { terminated == 1 && window.sheets.isEmpty })
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(asked == 2, "already asked")
        delegate.terminationGate.abandonWait()
    }

    @Test func aQuitAppleEventCarriesItsReasonToTheDelegate() throws {
        func event(_ reason: String?) -> NSAppleEventDescriptor {
            let e = NSAppleEventDescriptor(eventClass: AEEventClass(kCoreEventClass), eventID: AEEventID(kAEQuitApplication),
                                           targetDescriptor: nil, returnID: AEReturnID(kAutoGenerateReturnID),
                                           transactionID: AETransactionID(kAnyTransactionID))
            if let reason { e.setParam(NSAppleEventDescriptor(enumCode: OSType(QuitReason.fourCC(reason))), forKeyword: AEKeyword(kAEQuitReason)) }
            return e
        }
        #expect(AppDelegate.reason(of: nil) == .user)
        #expect(AppDelegate.reason(of: event(nil)) == .user)
        #expect(AppDelegate.reason(of: event("quia")) == .user)
        #expect(AppDelegate.reason(of: event("rlgo")) == .sessionEnd)
        #expect(AppDelegate.reason(of: event("rsdn")) == .sessionEnd)
        #expect(AppDelegate.reason(of: event("rest")) == .sessionEnd)
    }

    // MARK: Command-R

    @Test func commandRCreatesAWorldOnlyWhereTheCreateButtonIsEnabled() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        await rig.scan()
        let setup = rig.setup
        setup.openCreate(.recipe(Recipes.smallPlanet.id))
        setup.goal = .research
        let window = QuietWindow(contentRect: NSRect(x: -20000, y: -20000, width: 1000, height: 800), styleMask: [.borderless], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: SetupWindowView(setup: setup, controller: rig.controller, library: rig.library).frame(width: 1000, height: 800))
        window.contentView = host
        window.orderFront(nil)
        host.layoutSubtreeIfNeeded()
        defer { window.close() }
        func commandR() throws -> Bool {
            let event = try #require(NSEvent.keyEvent(
                with: .keyDown, location: .zero, modifierFlags: .command, timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: window.windowNumber, context: nil, characters: "r", charactersIgnoringModifiers: "r",
                isARepeat: false, keyCode: 15))
            return window.performKeyEquivalent(with: event)
        }
        // An invalid form: Create is grey and the shortcut does nothing.
        setup.name = ""
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        #expect(!setup.reasons.isEmpty)
        _ = try commandR()
        #expect(rig.controller.job == nil && rig.researchStarts.isEmpty, "Command-R does what a grey Create does: nothing")
        // A valid form: it creates.
        setup.name = "cmd-r"
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        #expect(setup.reasons.isEmpty)
        _ = try commandR()
        #expect(await rig.waitForGrowing("cmd-r"), "Command-R creates the world: \(rig.controller.logText)")
        // A run going: the form is refused, so is the shortcut (a second world is not started).
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        _ = try commandR()
        #expect(rig.researchStarts == ["cmd-r"])
    }

    // MARK: Helpers

    /// A delegate that never answers AppKit: a reply to a quit that is not under way could end the test process.
    private func makeDelegate(_ run: RunController, _ play: PlayController) -> AppDelegate {
        let delegate = AppDelegate(run: run, play: play, quitPanel: .offscreen())
        delegate.replyToTerminate = {}
        delegate.terminate = {}
        return delegate
    }

    /// An engine that ignores SIGINT and says so (a file) once it does: signalling it earlier would kill it.
    private static let stubborn = "trap '' INT\ntouch \"$(dirname \"$0\")/ready\"\nwhile kill -0 $PPID 2>/dev/null && [ $SECONDS -lt 120 ]; do sleep 0.2; done"

    /// `ignoresInterrupt` for a test that is not async: waits on the main run loop.
    private func await_ignoresInterruptSync(_ repo: URL) {
        let ready = repo.appendingPathComponent(".venv/bin/ready").path
        let deadline = Date().addingTimeInterval(10)
        while !FileManager.default.fileExists(atPath: ready), Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
    }

    private func ignoresInterrupt(_ repo: URL) async {
        let ready = repo.appendingPathComponent(".venv/bin/ready").path
        #expect(await eventually(timeout: .seconds(10)) { FileManager.default.fileExists(atPath: ready) })
    }

    /// A controller of its own on `repo`: it starts its jobs in it and reaches no shared state and no saved preference.
    private func controller(_ repo: URL) -> RunController {
        RunController(storedPath: repo.path, savePath: { _ in }, candidates: [], isApp: false)
    }

    /// A repository folder whose `.venv/bin/alife` is `body`.
    private func fakeRepo(_ scratch: Scratch, engine body: String) throws -> URL {
        let repo = scratch.url.appendingPathComponent("repo")
        let bin = repo.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: repo.appendingPathComponent("runs/src"), withIntermediateDirectories: true)
        let script = bin.appendingPathComponent("alife")
        try ("#!/bin/bash\n" + body + "\n").write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return repo
    }
}
