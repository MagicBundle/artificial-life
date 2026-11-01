import AppKit
import Foundation

/// The answer to "may the app quit now?" while a played world may be saving its checkpoint (686 MB). A world that is live
/// or still saving is asked to stop through its ladder and the app answers `.terminateLater`; it replies when the process
/// has gone, after `timeout` or when the user abandons the wait (`applicationWillTerminate` then ends what is left with
/// SIGTERM). Held apart from the delegate so that a stub can test it with short durations (DESIGN §33.4.7).
@MainActor
final class TerminationGate {
    private let timeout: Duration
    private let poll: Duration
    private var waiting: Task<Void, Never>?
    /// The reply of the wait in progress: taken (and so made at most once) by `finishWait`.
    private var pendingReply: (@MainActor () -> Void)?

    /// Called while the app waits: at the start and then on every poll, with the time waited so far. The quit panel follows it
    /// (the delegate knows which worlds and runs are being saved). Not called when the app quits at once.
    var onWaiting: (@MainActor (_ elapsed: Duration) -> Void)?

    init(timeout: Duration = .seconds(40), poll: Duration = .milliseconds(100)) {
        self.timeout = timeout
        self.poll = poll
    }

    /// Whether a wait is in progress (the quit was answered `.terminateLater` and not yet replied to).
    var isWaiting: Bool { pendingReply != nil }

    /// `.terminateNow` when `hasProcess` is false. Otherwise calls `stop` once, returns `.terminateLater`, and calls `reply`
    /// once when `hasProcess` turns false, `timeout` has passed or `abandonWait()` is called.
    func shouldTerminate(hasProcess: @escaping @MainActor () -> Bool, stop: @MainActor () -> Void,
                         reply: @escaping @MainActor () -> Void) -> NSApplication.TerminateReply {
        guard hasProcess() else { return .terminateNow }
        guard waiting == nil else { return .terminateLater }  // asked twice: the first wait answers
        stop()
        pendingReply = reply
        let (timeout, poll) = (timeout, poll)
        waiting = Task { @MainActor [weak self] in
            let clock = ContinuousClock()
            let began = clock.now
            let deadline = began + timeout
            self?.notifyWaiting(clock.now - began)
            while hasProcess(), clock.now < deadline, !Task.isCancelled {
                try? await Task.sleep(for: poll)
                self?.notifyWaiting(clock.now - began)
            }
            self?.finishWait()
        }
        return .terminateLater
    }

    /// The user gave up waiting ("Quit now without saving"): stops the wait and replies now. What is still running is left to
    /// `applicationWillTerminate`, which ends it with SIGTERM. Nothing happens when no wait is in progress.
    func abandonWait() {
        waiting?.cancel()
        finishWait()
    }

    private func notifyWaiting(_ elapsed: Duration) {
        guard pendingReply != nil else { return }
        onWaiting?(elapsed)
    }

    private func finishWait() {
        waiting = nil
        guard let reply = pendingReply else { return }
        pendingReply = nil
        reply()
    }
}
