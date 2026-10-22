import Foundation

/// The semantic level of the view (DESIGN §33.4.7). Levels only change what is drawn; the camera is one `Camera`.
package enum WorldLevel: Int, Comparable, CaseIterable, Sendable {
    /// The whole world, or more of it than half: the weather map is the picture, organisms are not instanced.
    case planet
    /// A region: the map under organisms, sphere impostors where large enough, flat discs where small.
    case region
    /// Close to the organisms: full shading, bonds, the map faint.
    case organism

    package static func < (a: WorldLevel, b: WorldLevel) -> Bool { a.rawValue < b.rawValue }
}

package enum WorldLevels {
    /// The planet level begins when the visible half extent at the floor is at least this fraction of `L`.
    package static let planetHalfExtent: Float = 0.5
    /// The organism level begins closer than this fraction of `L`.
    package static let organismDistance: Float = 0.12
    /// An impostor whose projected radius is below this many pixels is a flat disc (no ray test).
    package static let flatDiscPixels: Float = 3
    /// A flat disc is at least this many pixels in radius, so that a small organism does not vanish.
    package static let minDiscPixels: Float = 1.2

    /// The visible half extent at the floor, `d·tan(fov/2)·max(1, aspect)`.
    package static func halfExtent(distance: Float, fovY: Float, aspect: Float) -> Float {
        distance * tan(fovY / 2) * max(1, aspect)
    }

    /// The level at `distance`. The planet level exists to protect the weather map from 40,000 organisms; a world with no map
    /// in view (`mapShown` false: a 2D or dry world, or the map switched off) has nothing to protect and shows its organisms at
    /// the region level however far the camera is.
    package static func level(distance: Float, fovY: Float, aspect: Float, size: Float, mapShown: Bool = true) -> WorldLevel {
        if distance < organismDistance * size { return .organism }
        if mapShown, halfExtent(distance: distance, fovY: fovY, aspect: aspect) >= planetHalfExtent * size { return .planet }
        return .region
    }

    /// How opaque the weather map is: fully at the planet level, under the organisms at the region level, faint close up.
    package static func mapOpacity(_ level: WorldLevel) -> Float {
        switch level {
        case .planet: return 1
        case .region: return 0.6
        case .organism: return 0.25
        }
    }

    /// The alpha of the territory overlay (the density layer at the planet level, §33.1.5): 25%, less close up.
    package static func territoryAlpha(_ level: WorldLevel) -> Float {
        switch level {
        case .planet, .region: return 0.25
        case .organism: return 0.12
        }
    }

    /// How agents are drawn at `level` with the user's toggles: nothing, 1 px translucent points (planet level, if asked
    /// for), or instances.
    package enum AgentLayer: Equatable, Sendable {
        case none, points, instances
    }

    package static func agentLayer(at level: WorldLevel, display: WorldDisplay) -> AgentLayer {
        switch level {
        case .planet: return display.agentPoints ? .points : .none
        case .region, .organism: return display.agents ? .instances : .none
        }
    }

    /// The farthest the camera may be: the visible half extent at the floor stays within one world, so that the repeating
    /// map (two and a half worlds each way from the target's cell) always covers the view. Never nearer than `home`, the
    /// distance of the starting pose in a window of this shape (a wide window, a bottom dock, fits the world by its height
    /// and sits farther out than the rule above; the first pan or zoom must not snap the view in).
    package static func maxDistance(size: Float, fovY: Float, aspect: Float, home: Float = 0) -> Float {
        max(size / (tan(fovY / 2) * max(1, aspect)), home)
    }

    /// The nearest: a few agent radii.
    package static let minDistance: Float = 4

    /// The visible half extent, as a fraction of `L`, a fly-to ends at on a storm, a lineage or a place (the region level).
    package static let flyRegionHalfExtent: Float = 0.2
    /// The distance a fly-to ends at on an organism, as a fraction of `L` (inside the organism level).
    package static let flyOrganismDistance: Float = 0.08

    /// The camera distance a fly-to ends at: `level` is `.region` (a storm, a kin group, a place) or `.organism`; the planet
    /// level is not a destination. Never nearer than `minDistance`.
    package static func flyDistance(_ level: WorldLevel, size: Float, fovY: Float, aspect: Float) -> Float {
        switch level {
        case .organism:
            return max(flyOrganismDistance * size, minDistance)
        case .region, .planet:
            let d = flyRegionHalfExtent * size / (tan(fovY / 2) * max(1, aspect))
            return min(max(d, minDistance), max(maxDistance(size: size, fovY: fovY, aspect: aspect), minDistance))
        }
    }
}
