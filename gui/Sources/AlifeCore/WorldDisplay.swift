import Foundation

/// Which layers of the World are drawn: the View menu's toggles (DESIGN §33.4.3). Agents are drawn as instances at the region
/// and organism levels and, at the planet level, as 1 px translucent points only if asked for: 40,000 organisms cover half to
/// most of the weather map there, which is what that level shows (`agentPoints` is off by default).
package struct WorldDisplay: Equatable, Sendable {
    /// Instanced agents (region and organism levels).
    package var agents = true
    /// The 1 px point layer of the planet level.
    package var agentPoints = false
    package var bonds = true
    package var territory = true
    /// The weather map (and, without one, the floor).
    package var map = true
    /// The edges of the box.
    package var frame = true

    package init() {}
}
