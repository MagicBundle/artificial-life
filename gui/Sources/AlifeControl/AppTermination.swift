import AppKit
import Foundation

/// The answer to "may the app quit now?" while a played world may be saving its checkpoint (686 MB). A world that is live
/// or still saving is asked to stop through its ladder and the app answers `.terminateLater`; it replies when the process
/// has gone, or after `timeout` (`applicationWillTerminate` then ends what is left with SIGTERM). Held apart from the
/// delegate so that a stub can test it with short durations (DESIGN §33.4.7).
@MainActor
final class TerminationGate {
    private let timeout: Duration
    private let poll: Duration
    private var waiting: Task<Void, Never>?

    init(timeout: Duration = .seconds(40), poll: Duration = .milliseconds(100)) {
        self.timeout = timeout
        self.poll = poll
    }

    /// `.terminateNow` when `hasProcess` is false. Otherwise calls `stop` once, returns `.terminateLater`, and calls `reply`
    /// once when `hasProcess` turns false or `timeout` has passed.
    func shouldTerminate(hasProcess: @escaping @MainActor () -> Bool, stop: @MainActor () -> Void,
                         reply: @escaping @MainActor () -> Void) -> NSApplication.TerminateReply {
        guard hasProcess() else { return .terminateNow }
        guard waiting == nil else { return .terminateLater }  // asked twice: the first wait answers
        stop()
        let (timeout, poll) = (timeout, poll)
        waiting = Task { @MainActor [weak self] in
            let deadline = ContinuousClock.now + timeout
            while hasProcess(), ContinuousClock.now < deadline {
                try? await Task.sleep(for: poll)
            }
            self?.waiting = nil
            reply()
        }
        return .terminateLater
    }
}
