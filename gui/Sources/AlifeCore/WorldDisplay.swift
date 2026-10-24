import Foundation

/// What the Territory layer draws (the View menu, and the `3` key, which cycles it): nothing, only the borders between the
/// territories of different kin groups, or the borders and a weak tint of each cell in its owner's colour.
package enum TerritoryMode: Int, CaseIterable, Sendable {
    case off, borders, tint

    /// The state after this one (`3`): off, borders, tint, off, ...
    package var next: TerritoryMode {
        switch self {
        case .off: return .borders
        case .borders: return .tint
        case .tint: return .off
        }
    }
}

/// Which layers of the World are drawn: the View menu's toggles (DESIGN §33.4.3). Agents are drawn as instances at every level,
/// their opacity a ramp of their drawn radius (§36.7), and at the planet level also as 1 px translucent points if asked for
/// (`agentPoints` is off by default).
package struct WorldDisplay: Equatable, Sendable {
    /// Instanced agents.
    package var agents = true
    /// The 1 px point layer of the planet level.
    package var agentPoints = false
    package var bonds = true
    /// The territory of the kin groups: borders and a 10% tint by default (DESIGN §36.1, F3).
    package var territory = TerritoryMode.tint
    /// The weather map (and, without one, the floor).
    package var map = true
    /// The edges of the box.
    package var frame = true
    /// The map's brightness, a multiplier of its colours (the View menu's Brightness; persisted).
    package var exposure: Float = 1
    /// Organisms as symbols whose size is capped (default), or at their physical size (DESIGN §36.7).
    package var sizeMode = OrganismSizeMode.capped

    /// The brightness presets of the View menu.
    package static let exposurePresets: [Float] = [0.8, 1.0, 1.25, 1.5]

    package init() {}
}
