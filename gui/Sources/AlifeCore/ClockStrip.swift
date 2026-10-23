import Foundation

/// A segment of the clock strip's rate control: `Pause | 0.25x | 0.5x | 1x | 2x | Max`.
package enum ClockSegment: Equatable, Hashable, Sendable {
    case pause
    case rate(PlayRate)

    package static let all: [ClockSegment] = [.pause] + PlayRate.rungs.map { .rate($0) }

    package var label: String {
        switch self {
        case .pause: return "Pause"
        case .rate(.max): return "Max"
        case .rate(let rate): return rate.label
        }
    }

    /// The command that selecting this segment sends.
    package var command: PlayCommand {
        switch self {
        case .pause: return .pause
        case .rate(let rate): return .rate(rate)
        }
    }
}

/// What the clock strip at the top of the dock shows, from the newest `CLCK` and what the window asked since (DESIGN
/// §33.4.6). Pure, so the strip's text is tested without a window.
package struct ClockStrip: Equatable, Sendable {
    /// The selected segment; nil for a rate that is none of the rungs (the readout still names it).
    package var selected: ClockSegment?
    package var paused: Bool
    package var step: UInt64?
    /// "0.98x of 1x" while running, "paused" or the reason it stopped.
    package var readout: String
    /// Steps the schedule gave up on: shown in amber when above zero.
    package var lateSteps: UInt32
    package var saving: Bool
    package var saveStallSteps: UInt32
    /// Why the world is not running, if it is not (`CLCK.reason`).
    package var stopReason: String?
    /// The world stops by itself at the end of a chapter / when an alert class that is armed fires (`CLCK.flags`): the
    /// strip's lock icon.
    package var chapterPauseArmed: Bool
    package var alertPauseArmed: Bool

    /// `clock` is the newest `CLCK` (nil before the first), `requested` what the window asked and the world has not yet
    /// confirmed, `baseRate` the steps per second at 1x. `chapterLabel` ("Chapter 3") and `alertTitle` name the chapter that
    /// ended and the alert that stopped the world, when the window knows them (DESIGN §34.7): "Chapter 3 complete".
    package init(clock: PlayClock?, requested: PlayRate? = nil, baseRate: Double, chapterLabel: String? = nil,
                 alertTitle: String? = nil) {
        step = clock?.head.step
        lateSteps = clock?.lateSteps ?? 0
        saving = clock?.saving ?? false
        saveStallSteps = clock?.saveStallSteps ?? 0
        chapterPauseArmed = clock?.chapterPauseArmed ?? false
        alertPauseArmed = clock?.alertPauseArmed ?? false
        let reason = clock?.reason ?? .running
        switch reason {
        case .chapter: stopReason = chapterLabel.map { $0 + " complete" } ?? Self.text(of: reason)
        case .alert: stopReason = alertTitle.flatMap { $0.isEmpty ? nil : $0 } ?? Self.text(of: reason)
        default: stopReason = Self.text(of: reason)
        }
        let rate: PlayRate? = requested ?? clock.flatMap { PlayRate(clock: $0.rate) }
        // What the window just asked wins over the clock, which lags by at most a quarter of a second.
        // A chapter or alert reason with a rate above zero is the world still running: it announces the card while the
        // checkpoint is written (and stays the reason of a chapter end whose pause is off). The pause itself has rate 0.
        let stopped = [PlayClock.Reason.userPause, .endOfSteps, .error].contains(reason)
        paused = requested.map { $0 == .multiplier(0) } ?? (stopped || clock?.rate == 0)
        if paused {
            selected = .pause
        } else if let rate, PlayRate.rungs.contains(rate) {
            selected = .rate(rate)
        } else {
            selected = nil
        }
        if paused {
            readout = (requested == nil ? stopReason : nil) ?? "paused"
        } else if let clock, let rate {
            let multiplier = baseRate > 0 ? Double(clock.achieved) / baseRate : 0
            readout = String(format: "%.2fx of ", multiplier) + rate.label
        } else {
            readout = "waiting for the clock"
        }
    }

    /// The reason text; nil while running.
    package static func text(of reason: PlayClock.Reason) -> String? {
        switch reason {
        case .running: return nil
        case .userPause: return "paused"
        case .endOfSteps: return "end of steps"
        case .chapter: return "end of chapter"
        case .alert: return "alert"
        case .error: return "stopped by an error"
        }
    }
}
