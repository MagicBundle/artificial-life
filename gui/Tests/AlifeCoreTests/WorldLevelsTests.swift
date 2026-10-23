import Foundation
import Testing

@testable import AlifeCore

@Suite struct WorldLevelsTests {
    private let L: Float = 320
    private let fov = Camera.defaultFovY

    private func level(_ distance: Float, aspect: Float = 1) -> WorldLevel {
        WorldLevels.level(distance: distance, fovY: fov, aspect: aspect, size: L)
    }

    @Test(arguments: [Float(1), Float(16) / 9, Float(9) / 16, Float(2.4)])
    func theHomePoseOfThePlanetIsAtThePlanetLevel(aspect: Float) {
        let camera = Camera.planet(size: L, aspect: aspect)
        #expect(level(camera.distance, aspect: aspect) == .planet)
    }

    @Test func zoomingInPassesThroughTheRegionToTheOrganismLevel() {
        let home = Camera.planet(size: L).distance
        var seen: [WorldLevel] = []
        var d = home
        while d > WorldLevels.minDistance {
            let l = level(d)
            if seen.last != l { seen.append(l) }
            d *= 0.9
        }
        #expect(seen == [.planet, .region, .organism])
    }

    @Test func theBoundariesAreThoseOfTheSpec() {
        // planet: the visible half extent at the floor is at least half a world; organism: nearer than 0.12 L.
        let tan = Float(tan(fov / 2))
        #expect(level(0.5 * L / tan + 0.01) == .planet)
        #expect(level(0.5 * L / tan - 0.01) == .region)
        #expect(level(0.12 * L + 0.01) == .region)
        #expect(level(0.12 * L - 0.01) == .organism)
        // A wide window reaches half a world sooner: the half extent is d·tan(fov/2)·max(1, aspect).
        #expect(level(0.5 * L / tan / 2 + 1, aspect: 2) == .planet)
        #expect(level(0.5 * L / tan / 2 + 1, aspect: 1) == .region)
    }

    @Test func theHalfExtentGrowsWithTheDistanceAndTheWidth() {
        #expect(WorldLevels.halfExtent(distance: 100, fovY: fov, aspect: 1) < WorldLevels.halfExtent(distance: 200, fovY: fov, aspect: 1))
        #expect(WorldLevels.halfExtent(distance: 100, fovY: fov, aspect: 2) == 2 * WorldLevels.halfExtent(distance: 100, fovY: fov, aspect: 1))
        #expect(WorldLevels.halfExtent(distance: 100, fovY: fov, aspect: 0.5) == WorldLevels.halfExtent(distance: 100, fovY: fov, aspect: 1))
    }

    @Test func theMapOpacityIsAContinuousFunctionOfTheDistanceThroughTheNodes() {
        func opacity(_ fraction: Float) -> Float { WorldLevels.mapOpacity(distance: fraction * L, size: L) }
        // 0.7 at the organism level's fly-to, 0.9 at the region level's, 1.0 from the globe's Home distance (0.296 L).
        #expect(opacity(0.04) == 0.7 && opacity(0.08) == 0.7)
        #expect(abs(opacity(0.15) - 0.9) < 1e-6)
        #expect(opacity(0.296) == 1 && opacity(0.5) == 1 && opacity(1.2) == 1)
        // Non-decreasing in the distance, no step: the largest change per 0.5% of the distance is small (v1 dropped 40% at a level).
        var previous = opacity(0.04)
        var largest: Float = 0
        var d: Float = 0.04
        while d < 1.5 {
            d *= 1.005
            let o = opacity(d)
            #expect(o >= previous - 1e-6, "monotone at \(d)")
            largest = max(largest, o - previous)
            previous = o
        }
        #expect(largest < 0.01, "the largest step per 0.5% of d: \(largest)")
        // C1 at the nodes: the slope in ln d is zero there.
        for node: Float in [0.08, 0.15, 0.296] {
            let eps: Float = 1e-3
            let slope = (opacity(node * (1 + eps)) - opacity(node * (1 - eps))) / (2 * eps)
            #expect(abs(slope) < 5e-3, "the slope at \(node) L: \(slope)")
        }
        // The world's size is a fraction, not a number of units.
        #expect(WorldLevels.mapOpacity(distance: 0.15 * 48, size: 48) == opacity(0.15))
    }

    @Test func theLevelNoLongerChangesTheOpacityAcrossItsEdges() {
        // v1's steps: the map opacity dropped by 40% at the planet-to-region edge and 58% at the region-to-organism one. Across each
        // edge of the derived level the opacity now changes by less than 1%.
        let tan = Float(tan(fov / 2))
        let planetEdge = 0.5 * L / tan
        let below = WorldLevels.mapOpacity(distance: planetEdge * 0.999, size: L)
        let above = WorldLevels.mapOpacity(distance: planetEdge * 1.001, size: L)
        #expect(abs(above - below) < 0.01)
        #expect(abs(WorldLevels.mapOpacity(distance: 0.1199 * L, size: L) - WorldLevels.mapOpacity(distance: 0.1201 * L, size: L)) < 0.01)
    }

    @Test func organismsAreInstancesAtEveryLevelAndThePointsAreAnExtraAtThePlanetOnly() {
        var display = WorldDisplay()
        for level in WorldLevel.allCases {
            #expect(WorldLevels.agentLayers(at: level, display: display) == .instances, "no level switch: \(level)")
        }
        display.agentPoints = true
        #expect(WorldLevels.agentLayers(at: .planet, display: display) == [.instances, .points])
        #expect(WorldLevels.agentLayers(at: .region, display: display) == .instances, "the points are the planet level's")
        #expect(WorldLevels.agentLayers(at: .organism, display: display) == .instances)
        display.agents = false
        #expect(WorldLevels.agentLayers(at: .planet, display: display) == .points)
        #expect(WorldLevels.agentLayers(at: .region, display: display).isEmpty)
        #expect(WorldLevels.agentLayers(at: .organism, display: display).isEmpty)
    }

    @Test func theTerritoryCyclesThroughItsThreeStatesStartingAtTheTint() {
        #expect(WorldDisplay().territory == .tint, "borders and a weak tint by default")
        #expect(TerritoryMode.off.next == .borders && TerritoryMode.borders.next == .tint && TerritoryMode.tint.next == .off)
        #expect(WorldDisplay().sizeMode == .capped && WorldDisplay().exposure == 1)
        #expect(WorldDisplay.exposurePresets == [0.8, 1.0, 1.25, 1.5])
    }

    @Test func theFarthestViewStaysWithinOneWorldOfTheTarget() {
        for aspect: Float in [1, 16.0 / 9, 9.0 / 16] {
            let d = WorldLevels.maxDistance(size: L, fovY: fov, aspect: aspect)
            #expect(abs(WorldLevels.halfExtent(distance: d, fovY: fov, aspect: aspect) - L) < 1e-2 * L)
            #expect(d > Camera.planet(size: L, aspect: aspect).distance, "the home pose can be zoomed out from")
        }
    }

    @Test func aWorldWithoutAMapHasNoPlanetLevel() {
        // The planet level protects the weather map from the organisms; a 2D or dry world has no map to protect.
        let far = Camera.planet(size: L).distance
        #expect(level(far) == .planet)
        #expect(WorldLevels.level(distance: far, fovY: fov, aspect: 1, size: L, mapShown: false) == .region)
        let near = 0.1 * L
        #expect(WorldLevels.level(distance: near, fovY: fov, aspect: 1, size: L, mapShown: false) == .organism)
        #expect(WorldLevels.agentLayers(at: WorldLevels.level(distance: far, fovY: fov, aspect: 1, size: L, mapShown: false),
                                        display: WorldDisplay()) == .instances)
    }

    @Test(arguments: [Float(2.4), Float(1.923), Float(0.4)])
    func theLimitIsNeverNearerThanTheHomePoseOfTheWindow(aspect: Float) {
        let home = Camera.planet(size: L, aspect: aspect).distance
        let limit = WorldLevels.maxDistance(size: L, fovY: fov, aspect: aspect, home: home)
        #expect(limit >= home, "a wide or tall window starts farther out than the one-world rule")
        #expect(WorldLevels.maxDistance(size: L, fovY: fov, aspect: aspect) == WorldLevels.maxDistance(size: L, fovY: fov, aspect: aspect, home: 0))
    }
}
