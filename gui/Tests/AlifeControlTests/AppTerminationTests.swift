import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// The quit that waits for a played world to save (DESIGN §33.4.7), with a stub for the world and short durations.
@MainActor
@Suite struct AppTerminationTests {
    private final class Stub {
        var running = true
        var stops = 0
        var replies = 0
    }

    @Test func withNoWorldTheAppQuitsAtOnce() {
        let gate = TerminationGate(timeout: .seconds(5), poll: .milliseconds(10))
        let stub = Stub()
        stub.running = false
        let reply = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(reply == .terminateNow && stub.stops == 0 && stub.replies == 0)
    }

    @Test func aWorldThatSavesHoldsTheQuitUntilItHasGone() async {
        let gate = TerminationGate(timeout: .seconds(20), poll: .milliseconds(10))
        let stub = Stub()
        let reply = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(reply == .terminateLater && stub.stops == 1)
        try? await Task.sleep(for: .milliseconds(100))
        #expect(stub.replies == 0, "still saving")
        stub.running = false
        #expect(await eventually { stub.replies == 1 })
    }

    @Test func aQuitAskedTwiceStopsOnceAndRepliesOnce() async {
        let gate = TerminationGate(timeout: .seconds(20), poll: .milliseconds(10))
        let stub = Stub()
        _ = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        let again = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(again == .terminateLater && stub.stops == 1)
        stub.running = false
        #expect(await eventually { stub.replies == 1 })
        try? await Task.sleep(for: .milliseconds(100))
        #expect(stub.replies == 1)
    }

    @Test func aWorldThatWillNotLeaveIsLeftToTheLastResortAfterTheTimeout() async {
        let gate = TerminationGate(timeout: .milliseconds(150), poll: .milliseconds(10))
        let stub = Stub()
        let began = ContinuousClock.now
        let reply = gate.shouldTerminate(hasProcess: { stub.running }, stop: { stub.stops += 1 }, reply: { stub.replies += 1 })
        #expect(reply == .terminateLater)
        #expect(await eventually { stub.replies == 1 })
        #expect(stub.running, "the process is still there; applicationWillTerminate sends it SIGTERM")
        #expect(ContinuousClock.now - began >= .milliseconds(140))
    }

    @Test func theDelegateAsksThePlayControllerAndQuitsAtOnceWithoutAWorld() {
        // The shared controller has no process in a test run: the real delegate answers `.terminateNow`.
        let delegate = AppDelegate()
        #expect(!PlayController.shared.hasProcess)
        #expect(delegate.applicationShouldTerminate(NSApplication.shared) == .terminateNow)
    }
}
