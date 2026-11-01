import Foundation

/// What the sidebar's Watch row says (docs/APP.md §2.4, review M1): the world's state and, while another page is shown, whether the
/// world is waiting for the player. A chapter or an alert stops the world; `NSApp.requestUserAttention` does nothing for an active
/// app (and the app is active whenever the player is on another page), so the row is the in-app signal.
package enum WatchCue {
    package enum Phase: Equatable, Sendable { case none, starting, live, ended, failedToStart, failed }
    package enum Stop: Equatable, Sendable { case chapter, alert }

    package struct Cue: Equatable, Sendable {
        package var text: String
        /// The world is waiting for the player (a stop, or alerts not yet read): the row is marked.
        package var needsPlayer: Bool
    }

    /// `looking` is `WorldModel.viewerIsLooking`: Watch is shown in the key window. `stop` is why the world is stopped now (a chapter or
    /// an alert with the rate at 0), nil when it runs or the user paused it. `unread` counts the alerts raised while not looking.
    package static func cue(phase: Phase, world: String?, stop: Stop?, chapter: Int, unread: Int, looking: Bool) -> Cue {
        let name = world ?? ""
        switch phase {
        case .none: return Cue(text: "No world open", needsPlayer: false)
        case .starting: return Cue(text: "\(name) · starting", needsPlayer: false)
        case .ended: return Cue(text: "\(name) · ended", needsPlayer: false)
        case .failedToStart: return Cue(text: "\(name) · did not start", needsPlayer: false)
        case .failed: return Cue(text: "\(name) · stopped unexpectedly", needsPlayer: false)
        case .live:
            guard !looking else { return Cue(text: "\(name) · live", needsPlayer: false) }
            var parts = [name]
            switch stop {
            case .chapter: parts.append(chapter > 0 ? "chapter \(chapter) is waiting" : "a chapter is waiting")
            case .alert: parts.append("stopped for an alert")
            case nil: parts.append("live")
            }
            if unread > 0 { parts.append(unread == 1 ? "1 unread alert" : "\(unread) unread alerts") }
            return Cue(text: parts.joined(separator: " · "), needsPlayer: stop != nil || unread > 0)
        }
    }
}
