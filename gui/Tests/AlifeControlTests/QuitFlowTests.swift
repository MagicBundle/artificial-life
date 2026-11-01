import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// The additions to the quit of docs/APP.md §5.4: `abandonWait()`, `onWaiting`, the pre-confirmation before the gate, a run held by
/// the gate, the escalation in `shutdown`, and the quit panel. Stubs and real shell scripts; the shared controllers are not touched
/// (the delegate's own checks run only where the shared state has no process).
@MainActor
@Suite struct QuitFlowTests {
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
        let repo = try fakeRepo(scratch, engine: "exec sleep 60")
        let run = controller(repo)
        run.settings.name = "quit-test-run"
        run.start()
        defer { run.forceStop() }
        #expect(run.hasRunJob && run.runningRunName == "quit-test-run")

        let delegate = AppDelegate(run: run, play: PlayController())
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

        let delegate = AppDelegate(run: run, play: PlayController())
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
        let repo = try fakeRepo(scratch, engine: "exec sleep 60")
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
        let delegate = AppDelegate(run: run, play: PlayController())
        var asked = 0
        delegate.confirmStopRun = { _ in asked += 1; return true }
        #expect(!run.hasRunJob)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateNow)
        #expect(asked == 0)
    }

    // MARK: The panel

    @Test func thePanelShowsUpdatesOffersQuitNowAndCloses() {
        let panel = QuitPanel()
        var quits = 0
        panel.onQuitNow = { quits += 1 }
        panel.show(QuitPanelModel(worlds: ["storm"], run: nil, elapsedSeconds: 1, lossMinutes: 7))
        #expect(panel.isVisible)
        #expect(panel.texts.title == "Saving 'storm'...")
        #expect(panel.texts.quitNowHidden)
        panel.show(QuitPanelModel(worlds: ["storm"], run: nil, elapsedSeconds: 12, lossMinutes: 7))
        #expect(panel.texts.detail.contains("12 s"))
        #expect(!panel.texts.quitNowHidden)
        #expect(panel.texts.warning.contains("7 minutes"))
        let panelWindow = panel.window
        #expect(panelWindow is NSPanel && (panelWindow as? NSPanel)?.worksWhenModal == true)
        #expect(panelWindow?.styleMask.contains(.closable) == false, "no way to close it but the quit ending")
        panel.close()
        #expect(!panel.isVisible)
    }

    // MARK: Helpers

    /// An engine that ignores SIGINT and says so (a file) once it does: signalling it earlier would kill it.
    private static let stubborn = "trap '' INT\ntouch \"$(dirname \"$0\")/ready\"\nwhile true; do sleep 0.2; done"

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
