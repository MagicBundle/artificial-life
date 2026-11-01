import AlifeCore
import Foundation

/// R4 of docs/APP.md (owner decision D4 = A): leaving the Watch page pauses the world that is being played, and coming back continues it
/// when *Pick up where I left off* is on. The decisions are `AwayPause`'s; this reads the world's clock for it and sends what it answers.
/// Deactivating the app (Command-Tab) does not call any of this: only `setWatchActive` does, which follows the selected page.
extension WorldModel {
    /// *Pick up where I left off* (Settings): on unless the player turned it off.
    var picksUpWhereLeft: Bool { (defaults.object(forKey: AppShell.pickUpKey) as? Bool) ?? true }

    /// The facts of the world and its clock that `AwayPause` decides from.
    var awayFacts: AwayPause.Facts {
        let clock = play.store.clock
        return AwayPause.Facts(
            world: play.world, live: play.phase == .live, rate: clock?.rate, reason: clock?.reason,
            requestedPause: play.requestedRate == .multiplier(0), chapterPauseArmed: clock?.chapterPauseArmed ?? false,
            alertPauseArmed: clock?.alertPauseArmed ?? false)
    }

    /// The player went to another page.
    func leftWatch() {
        perform(away.leftWatch(awayFacts))
    }

    /// The player is on Watch again.
    func returnedToWatch() {
        perform(away.returned(awayFacts, pickUp: picksUpWhereLeft))
    }

    /// A clock arrived (or the world went away) while another page may be shown.
    func awayClockChanged() {
        perform(away.clockChanged(awayFacts, watchActive: watchActive))
    }

    /// A different world, or none: no pause is held for it. (Not `resetInteraction`: a view that attaches again resets the selection and the
    /// camera, and must not forget that the app paused the world.)
    func clearAway() {
        away.clear()
        syncAwayHold()
    }

    private func perform(_ action: AwayPause.Action?) {
        defer { syncAwayHold() }
        switch action {
        case .pause?:
            if !play.send(.pause) { away.cancelPause() }
        case .resume?:
            play.send(.run)
        case nil:
            break
        }
    }

    private func syncAwayHold() {
        if awayHold != away.hold { awayHold = away.hold }
    }
}
