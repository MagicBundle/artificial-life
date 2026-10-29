import Foundation

/// What a key does in the World window (DESIGN §33.4.8).
package enum WorldAction: Equatable, Sendable {
    /// Pause when running, run when paused.
    case togglePause
    /// One step, then paused.
    case step
    /// The next colour mode of the header's list.
    case cycleColor
    case toggleAgents, toggleBonds
    /// Off, borders, borders and tint, off ... (the View menu's Territory).
    case cycleTerritory
    /// The built material and the farmed soil (DESIGN §45), where the world has them.
    case toggleStructures, toggleSoil
    /// Follow the selection with the camera, flying in to it.
    case followSelection
    /// One rung down or up the rate ladder.
    case slower, faster
    /// Fly to the Home view (`h`, `r`): the flat overview, north up, facing the populated side.
    case home
    /// Turn the map north up (`n`): the heading eases to 0.
    case northUp
    /// The minimap on or off (`i`).
    case toggleMinimap
    /// A standard view of a 3D world, animated: `t` Top, `o` Oblique, `l` Low, `s` Side (`ViewPreset`).
    case viewPreset(ViewPreset)
    /// `Esc`, one level at a time and never resuming or acknowledging (DESIGN §34.7): cancel aiming a power, else drop its
    /// preview, else collapse the chapter card to the map, else clear the selection and the follow.
    case escape
    /// Open the Powers tab.
    case showPowers
    /// Fly to the newest alert of the stack.
    case flyToAlert
    /// Acknowledge the newest alert of the stack that is not acknowledged.
    case acknowledgeAlert
    /// Resume at the rate the world holds (the chapter card's Continue). Never pauses.
    case continueRun
    /// Send the previewed power.
    case confirmPower
}

/// The World's own key table. It is not `ViewerKeys`: there `f` is the field slice and `[` `]` the field layers, which in the
/// World are `follow` and the rate.
///
/// ```text
/// space  pause / run        .  step                 c  next colour mode
/// 1 2 3  agents / bonds / territory (off, borders, tint)   f  follow the selection
/// 5 6    the built material / the farmed soil (a world that has them; `4` is the Lab's field key, which the World leaves unbound)
/// [ ]    slower / faster along the rate rungs       h r  home (the flat overview, a flight)
/// n      north up                                   i  the minimap on or off
/// t o l s  the standard views: Top, Oblique, Low, Side (animated; 3D worlds)
/// g      continue (run)     p  the Powers tab       return  confirm the previewed power
/// a      fly to the newest alert                    k  acknowledge the newest alert
/// esc    one level back: aiming, the preview, the chapter card, the selection (never resumes, never acknowledges)
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
        case "3": return .cycleTerritory
        case "5": return .toggleStructures
        case "6": return .toggleSoil
        case "f": return .followSelection
        case "[": return .slower
        case "]": return .faster
        case "r", "h": return .home
        case "n": return .northUp
        case "i": return .toggleMinimap
        case "t": return .viewPreset(.top)
        case "o": return .viewPreset(.oblique)
        case "l": return .viewPreset(.low)
        case "s": return .viewPreset(.side)
        case "\u{1B}": return .escape
        case "g": return .continueRun
        case "p": return .showPowers
        case "a": return .flyToAlert
        case "k": return .acknowledgeAlert
        case "\r", "\u{3}": return .confirmPower
        default: return nil
        }
    }

    /// Every character that does something, for the help text and for tests.
    package static let characters: [String] = [
        " ", ".", "c", "1", "2", "3", "5", "6", "f", "[", "]", "r", "h", "n", "i", "t", "o", "l", "s", "\u{1B}", "g", "p", "a", "k", "\r", "\u{3}",
    ]

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
