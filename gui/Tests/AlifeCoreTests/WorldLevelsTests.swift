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

    @Test func theMapIsFullAtThePlanetAndFaintCloseUp() {
        #expect(WorldLevels.mapOpacity(.planet) == 1)
        #expect(WorldLevels.mapOpacity(.region) == 0.6)
        #expect(WorldLevels.mapOpacity(.organism) < WorldLevels.mapOpacity(.region))
        #expect(WorldLevels.territoryAlpha(.planet) == 0.25 && WorldLevels.territoryAlpha(.region) == 0.25)
        #expect(WorldLevels.territoryAlpha(.organism) < 0.25)
    }

    @Test func organismsAreInstancesBelowThePlanetAndPointsOnlyIfAskedAtIt() {
        var display = WorldDisplay()
        #expect(WorldLevels.agentLayer(at: .planet, display: display) == .none, "40,000 organisms would hide the map")
        #expect(WorldLevels.agentLayer(at: .region, display: display) == .instances)
        #expect(WorldLevels.agentLayer(at: .organism, display: display) == .instances)
        display.agentPoints = true
        #expect(WorldLevels.agentLayer(at: .planet, display: display) == .points)
        display.agents = false
        #expect(WorldLevels.agentLayer(at: .planet, display: display) == .points, "the toggles are per level")
        #expect(WorldLevels.agentLayer(at: .region, display: display) == .none)
        #expect(WorldLevels.agentLayer(at: .organism, display: display) == .none)
    }

    @Test func theFarthestViewStaysWithinOneWorldOfTheTarget() {
        for aspect: Float in [1, 16.0 / 9, 9.0 / 16] {
            let d = WorldLevels.maxDistance(size: L, fovY: fov, aspect: aspect)
            #expect(abs(WorldLevels.halfExtent(distance: d, fovY: fov, aspect: aspect) - L) < 1e-2 * L)
            #expect(d > Camera.planet(size: L, aspect: aspect).distance, "the home pose can be zoomed out from")
        }
    }
}
