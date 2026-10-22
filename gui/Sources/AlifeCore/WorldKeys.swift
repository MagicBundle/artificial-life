import Foundation

/// What a key does in the World window (DESIGN §33.4.8).
package enum WorldAction: Equatable, Sendable {
    /// Pause when running, run when paused.
    case togglePause
    /// One step, then paused.
    case step
    /// The next colour mode of the header's list.
    case cycleColor
    case toggleAgents, toggleBonds, toggleTerritory
    /// Follow the selection with the camera, flying in to it.
    case followSelection
    /// One rung down or up the rate ladder.
    case slower, faster
    case resetCamera
    /// Clears the selection and the follow.
    case clearSelection
}

/// The World's own key table. It is not `ViewerKeys`: there `f` is the field slice and `[` `]` the field layers, which in the
/// World are `follow` and the rate.
///
/// ```text
/// space  pause / run        .  step                 c  next colour mode
/// 1 2 3  agents / bonds / territory                 f  follow the selection
/// [ ]    slower / faster along the rate rungs       r  reset the camera        esc  clear
/// ```
package enum WorldKeys {
    /// The action of a key, from `NSEvent.charactersIgnoringModifiers`; nil for a key that does nothing. A command or control
    /// chord is the system's or the menu's (nil here).
    package static func action(for characters: String, command: Bool = false, control: Bool = false) -> WorldAction? {
        if command || control { return nil }
        switch characters.lowercased() {
        case " ": return .togglePause
        case ".": return .step
        case "c": return .cycleColor
        case "1": return .toggleAgents
        case "2": return .toggleBonds
        case "3": return .toggleTerritory
        case "f": return .followSelection
        case "[": return .slower
        case "]": return .faster
        case "r": return .resetCamera
        case "\u{1B}": return .clearSelection
        default: return nil
        }
    }

    /// Every character that does something, for the help text and for tests.
    package static let characters: [String] = [" ", ".", "c", "1", "2", "3", "f", "[", "]", "r", "\u{1B}"]

    /// The mode after `current` in the header's list (the first when `current` is not in it); nil for an empty list.
    package static func nextColorMode(after current: String?, in modes: [String]) -> String? {
        guard !modes.isEmpty else { return nil }
        guard let current, let at = modes.firstIndex(of: current) else { return modes[0] }
        return modes[(at + 1) % modes.count]
    }
}

/// The rungs of the rate control, walked by `[` and `]`. A rate that is none of the rungs (the `rate` line accepts any
/// multiplier in 0.05...16) moves to the nearest rung in the direction asked.
package enum RateLadder {
    /// One rung slower than `rate`; nil at the slowest (the keys do not pause: space does).
    package static func slower(than rate: PlayRate) -> PlayRate? {
        switch rate {
        case .max: return PlayRate.rungs.dropLast().last
        case .multiplier(let m): return PlayRate.rungs.last { rung in rung.factor.map { $0 < m - 1e-9 } ?? false }
        }
    }

    /// One rung faster than `rate`; nil at `max`.
    package static func faster(than rate: PlayRate) -> PlayRate? {
        switch rate {
        case .max: return nil
        case .multiplier(let m): return PlayRate.rungs.first { rung in rung == .max || (rung.factor.map { $0 > m + 1e-9 } ?? false) }
        }
    }
}

extension PlayRate {
    /// The multiple of the base rate; nil for `max`.
    package var factor: Double? {
        if case .multiplier(let m) = self { return m }
        return nil
    }
}
