import Foundation

/// The semantic level of the view (DESIGN §33.4.7). Levels no longer change how the picture looks (§36.1: the map's opacity, the
/// territory and the instances are continuous functions of the distance); they classify the view for the View menu's agent toggle, the
/// picker and the bonds. The camera is one `Camera`.
package enum WorldLevel: Int, Comparable, CaseIterable, Sendable {
    /// The whole world, or more of it than half: the weather map is the picture, organisms are small symbols over it.
    case planet
    /// A region: the map under organisms, sphere impostors where large enough, flat discs where small.
    case region
    /// Close to the organisms: full shading, bonds, the map faint.
    case organism

    package static func < (a: WorldLevel, b: WorldLevel) -> Bool { a.rawValue < b.rawValue }
}

package enum WorldLevels {
    /// The planet level begins when the visible half extent at the floor is at least this fraction of `L`. The view never shows more
    /// than one world (`WorldPose.reach`, 0.49 L), so v1's half a world, reached by a window that shows the whole world and more, is
    /// now the Home view and the last tenth of the zoom range below it. (PR B's globe replaces it by the morph parameter, `s >= 0.5`.)
    package static let planetHalfExtent: Float = 0.45
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

    /// The camera distances (fractions of `L`) and the map opacities of DESIGN §36.1 (F2): 0.7 at the organism level's fly-to,
    /// 0.9 at the region level's, 1.0 from the distance at which the globe would fill 80% of the window (0.296 L).
    package static let mapOpacityNodes: [(distance: Float, opacity: Float)] = [(0.08, 0.7), (0.15, 0.9), (0.296, 1.0)]

    /// How opaque the weather map is at camera `distance` in a world of `size`: a continuous function of `ln d` through the
    /// nodes of `mapOpacityNodes` (linear in `ln d`, each piece eased by a smoothstep so the slope is zero at the nodes), constant
    /// beyond the ends. It replaces v1's three steps (1.0 / 0.6 / 0.25 by level): the map is no longer faded to the clear colour.
    package static func mapOpacity(distance: Float, size: Float) -> Float {
        let nodes = mapOpacityNodes
        guard size > 0, distance > 0 else { return 1 }
        let d = distance / size
        if d <= nodes[0].distance { return nodes[0].opacity }
        for i in 1..<nodes.count where d <= nodes[i].distance {
            let (a, b) = (nodes[i - 1], nodes[i])
            let t = log(d / a.distance) / log(b.distance / a.distance)
            return a.opacity + (b.opacity - a.opacity) * (t * t * (3 - 2 * t))
        }
        return nodes[nodes.count - 1].opacity
    }

    /// The layers agents are drawn with at `level` and the user's toggles.
    package struct AgentLayers: OptionSet, Sendable {
        package let rawValue: Int
        package init(rawValue: Int) { self.rawValue = rawValue }
        /// Impostors: spheres and flat discs, at every level, their opacity a ramp of the drawn radius (§36.7).
        package static let instances = AgentLayers(rawValue: 1)
        /// 1 px translucent points, at the planet level only, if asked for.
        package static let points = AgentLayers(rawValue: 2)
    }

    /// There is no level switch for the instances (v1 hid them at the planet level, where 40,000 organisms covered half to most of
    /// the map: the size policy keeps them to 35% of the view, and the ramp of their alpha has no edge).
    package static func agentLayers(at level: WorldLevel, display: WorldDisplay) -> AgentLayers {
        var layers: AgentLayers = []
        if display.agents { layers.insert(.instances) }
        if level == .planet, display.agentPoints { layers.insert(.points) }
        return layers
    }

    /// The farthest the camera may be, looking straight down at `heading` and height `z`: the picture stays within one world of its
    /// centre (`WorldPose.maxDistance`), for the one-world patch that replaced v1's tiled quad. It is also the distance of the
    /// Home view.
    package static func maxDistance(size: Float, fovY: Float, aspect: Float, heading: Float = 0, z: Float = 0) -> Float {
        WorldPose.maxDistance(size: size, fovY: fovY, aspect: aspect, heading: heading, z: z)
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
