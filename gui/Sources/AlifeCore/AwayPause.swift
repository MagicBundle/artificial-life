import Foundation

/// R4 of docs/APP.md (owner decision D4 = A): selecting another page of the window pauses the world that is being played, and coming back
/// continues it, with *Pick up where I left off* in Settings. Deactivating the app (Command-Tab) is not leaving: only a page change counts.
///
/// The one rule that matters: **only a pause the app made is ever undone.** A world the player paused, or that stopped by itself at a
/// chapter or an alert, stays paused when the player comes back; resuming it would hide the chapter card they have not read (`run`
/// acknowledges it, `engine play._command`). Pure: the model feeds it the facts of the clock and does what it answers.
package struct AwayPause: Equatable, Sendable {
    /// Whether the app holds a pause, and why the world is paused.
    package enum Hold: Equatable, Sendable {
        case none
        /// The app paused the world because the player is on another page; it continues when they come back and have chosen to pick up.
        case away(world: String)
        /// The player is back and the world stays paused (they chose not to pick up): Watch says it is paused because they left.
        case left(world: String)

        package var world: String? {
            switch self {
            case .none: return nil
            case .away(let w), .left(let w): return w
            }
        }

        /// The app paused the world because the player is on another page.
        package var isAway: Bool {
            if case .away = self { return true }
            return false
        }
    }

    /// What the app is to do to the world.
    package enum Action: Equatable, Sendable {
        case pause
        case resume
    }

    /// What the model reads off the world and the clock.
    package struct Facts: Equatable, Sendable {
        /// The directory name of the world of the session, nil when none.
        package var world: String?
        /// The world is live (it sent its header and has not ended).
        package var live: Bool
        /// The newest clock's rate (0 paused, -1 max), nil before the first clock.
        package var rate: Float?
        package var reason: PlayClock.Reason?
        /// The player (or the app) asked for a pause and the clock has not confirmed it yet.
        package var requestedPause: Bool
        package var chapterPauseArmed: Bool
        package var alertPauseArmed: Bool

        package init(world: String?, live: Bool, rate: Float?, reason: PlayClock.Reason?, requestedPause: Bool = false,
                     chapterPauseArmed: Bool = false, alertPauseArmed: Bool = false) {
            self.world = world
            self.live = live
            self.rate = rate
            self.reason = reason
            self.requestedPause = requestedPause
            self.chapterPauseArmed = chapterPauseArmed
            self.alertPauseArmed = alertPauseArmed
        }

        /// Steps are being taken: live, a clock seen, a rate above zero (or max) and no pause on its way.
        package var isRunning: Bool {
            guard live, let rate, !requestedPause else { return false }
            return rate != 0
        }

        /// The world is running and about to stop by itself: a chapter or an alert announces its card while the save point is written, and
        /// the pause that follows is the world's own. A pause sent now would take its place.
        package var aboutToStop: Bool {
            guard isRunning, let reason else { return false }
            return (reason == .chapter && chapterPauseArmed) || (reason == .alert && alertPauseArmed)
        }

        /// The world stopped itself: a chapter, an alert, the end of its steps or an error. Not the player's pause, and not the app's.
        package var stoppedByTheWorld: Bool {
            guard live, let rate, let reason, !requestedPause, rate == 0 else { return false }
            return [PlayClock.Reason.chapter, .alert, .endOfSteps, .error].contains(reason)
        }
    }

    package private(set) var hold: Hold = .none

    package init() {}

    package mutating func clear() { hold = .none }

    /// The app could not send the pause (no live world to write to): there is nothing to give back.
    package mutating func cancelPause() { if case .away = hold { hold = .none } }

    /// The player went to another page. A running world is paused (once).
    package mutating func leftWatch(_ f: Facts) -> Action? {
        if case .left = hold { hold = .none }  // (a note from the last visit goes: this visit's own pause, if any, replaces it)
        if case .away = hold { return nil }
        guard f.isRunning, !f.aboutToStop, let world = f.world else { return nil }
        hold = .away(world: world)
        return .pause
    }

    /// The player is back on Watch. The app's own pause is undone when `pickUp` is on; the world's, the player's and a chapter's never.
    package mutating func returned(_ f: Facts, pickUp: Bool) -> Action? {
        switch hold {
        case .none, .left:
            return nil
        case .away(let world):
            guard f.live, f.world == world else { hold = .none; return nil }
            // A chapter, an alert, the end of the steps took the world before or just after the app's pause: it is not the app's to continue.
            if f.stoppedByTheWorld || f.aboutToStop || f.isRunning { hold = .none; return nil }
            if pickUp {
                hold = .none
                return .resume
            }
            hold = .left(world: world)
            return nil
        }
    }

    /// A new clock, or a world that went away. Gives the hold back when the world is not the one it was made for or the world has taken
    /// over; pauses a world that is running while the player is on another page (it began, or went on, unseen).
    package mutating func clockChanged(_ f: Facts, watchActive: Bool) -> Action? {
        if let held = hold.world, !f.live || f.world != held { hold = .none }
        switch hold {
        case .left:
            // The player pressed Resume (it runs), or the world stopped by itself: the note no longer tells the truth.
            if f.isRunning || f.stoppedByTheWorld { hold = .none }
        case .away:
            if f.stoppedByTheWorld { hold = .none }
        case .none:
            if !watchActive { return leftWatch(f) }
        }
        return nil
    }
}
