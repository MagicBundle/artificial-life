import Foundation
import Testing
import simd

@testable import AlifeCore

@Suite struct TorusTests {
    private let L: Float = 320

    @Test(arguments: [Float(-641), -320, -0.001, 0, 1, 319.999, 320, 321, 1e5, -1e5])
    func aWrappedCoordinateIsInsideTheWorldAndTheSamePoint(v: Float) {
        let w = Torus.wrap(v, size: L)
        #expect(w >= 0 && w < L, "\(v) -> \(w)")
        // The same point modulo L (to the precision of float32 at that magnitude).
        let turns = ((v - w) / L).rounded()
        #expect(abs((v - w) - turns * L) <= max(abs(v), 1) * 1e-5)
    }

    @Test func theMinimumImageIsTheNearestOne() {
        // A point just inside the far edge, a reference just inside the near one: 2 units apart across the seam.
        let image = Torus.nearestImage(of: SIMD2(319, 100), to: SIMD2(1, 100), size: L)
        #expect(image == SIMD2(-1, 100))
        let other = Torus.nearestImage(of: SIMD2(1, 100), to: SIMD2(319, 100), size: L)
        #expect(other == SIMD2(321, 100))
        // Nothing to do when it is already the nearest.
        #expect(Torus.nearestImage(of: SIMD2(150, 90), to: SIMD2(160, 100), size: L) == SIMD2(150, 90))
    }

    @Test func eachAxisIsWrappedOnItsOwnAndZIsLeftAlone() {
        let image = Torus.nearestImage(of: SIMD3(319, 2, 7), to: SIMD3(1, 318, 3), size: L)
        #expect(image == SIMD3(-1, 322, 7))
    }

    @Test func aTieGoesTheWayTheShadersRintGoes() {
        // `rint` rounds half to even: a point exactly half a world away is the image at +L/2, and at 1.5 L it is -L/2.
        #expect(Torus.wrapDelta(160, size: L) == 160)
        #expect(Torus.wrapDelta(-160, size: L) == -160)
        #expect(Torus.wrapDelta(480, size: L) == -160)
        #expect(Torus.wrapDelta(-480, size: L) == 160)
    }

    @Test func theImageIsWithinHalfAWorldOfTheReferenceInEveryAxis() {
        var generator = SystemRandomNumberGenerator()
        for _ in 0..<2000 {
            let p = SIMD2(Float.random(in: 0..<L, using: &generator), Float.random(in: 0..<L, using: &generator))
            let r = SIMD2(Float.random(in: -400..<700, using: &generator), Float.random(in: -400..<700, using: &generator))
            let image = Torus.nearestImage(of: p, to: r, size: L)
            let d = image - r
            #expect(abs(d.x) <= L / 2 + 1e-3 && abs(d.y) <= L / 2 + 1e-3)
            // And it is the same point: the difference is a whole number of worlds.
            let turns = (image - p) / L
            #expect(abs(turns.x - turns.x.rounded()) < 1e-3 && abs(turns.y - turns.y.rounded()) < 1e-3)
        }
    }

    @Test func wrappingTheCameraTargetKeepsItInTheWorldAndTheViewTheSame() {
        var camera = Camera.planet(size: L, aspect: 1)
        camera.target = SIMD3(-5, 700, 3)
        let before = camera
        camera.wrapTarget(size: L)
        #expect(camera.target.x == 315 && camera.target.y == 60 && camera.target.z == 3)
        // A point is drawn at the same place relative to the target either way: its minimum image, projected.
        let p = SIMD3<Float>(310, 70, 0)
        let a = before.ndc(of: Torus.nearestImage(of: p, to: before.target, size: L), aspect: 1)
        let b = camera.ndc(of: Torus.nearestImage(of: p, to: camera.target, size: L), aspect: 1)
        #expect(a != nil && b != nil && simd_length(a! - b!) < 1e-4)
    }

    // MARK: The edge of the tiled map

    private func oblique(pitchDegrees: Float, distance: Float, aspect: Float = 1) -> Camera {
        var camera = Camera.planet(size: L, aspect: aspect)
        camera.pitch = -pitchDegrees * .pi / 180
        camera.distance = distance
        return camera
    }

    @Test func fromStraightAboveTheFloorInViewIsTheFootprintOfTheView() throws {
        let camera = Camera.planet(size: L, aspect: 1)
        let far = try #require(camera.farthestFloorDistance(aspect: 1))
        let half: Float = camera.distance * tan(camera.fovY / 2)
        let corner: Float = half * Float(2).squareRoot()
        #expect(abs(far - corner) < 0.01 * far, "the corner of the footprint: \(far)")
        var same = camera
        #expect(!same.keepFloorInReach(size: L, aspect: 1) && same == camera, "the home pose is within reach")
    }

    @Test func aTiltedViewFarOutSeesPastTheMapsEdgeAndIsTiltedBackDown() throws {
        let limit = WorldLevels.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 1)
        for degrees: Float in [20, 30, 45, 60] {
            var camera = oblique(pitchDegrees: degrees, distance: limit)
            let seen = camera.farthestFloorDistance(aspect: 1)
            let reach = Camera.tiledReach(size: L)
            if degrees <= 45 { #expect(seen == nil || (seen ?? 0) > reach, "\(degrees) degrees sees the edge") }
            camera.keepFloorInReach(size: L, aspect: 1)
            let after = try #require(camera.farthestFloorDistance(aspect: 1))
            #expect(after <= Camera.tiledReach(size: L), "\(degrees) degrees: \(after)")
            let limitPitch: Float = -degrees * .pi / 180 + 1e-5
            #expect(camera.pitch <= limitPitch, "steeper, never shallower")
        }
    }

    @Test func aNearViewKeepsItsPitchWhateverItIs() {
        var camera = oblique(pitchDegrees: 20, distance: 0.05 * L)
        let before = camera
        #expect(!camera.keepFloorInReach(size: L, aspect: 1) && camera == before, "the edge is a hair under the horizon")
        // Higher up, the same pitch shows a band of sky between the horizon and the edge: tilted down.
        var high = oblique(pitchDegrees: 20, distance: 0.5 * L)
        let tilted = high.keepFloorInReach(size: L, aspect: 1)
        #expect(tilted)
    }

    @Test func theLimitHoldsForAWideWindowAndAViewFromBelow() throws {
        var wide = oblique(pitchDegrees: 30, distance: WorldLevels.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 2.4), aspect: 2.4)
        wide.keepFloorInReach(size: L, aspect: 2.4)
        #expect(try #require(wide.farthestFloorDistance(aspect: 2.4)) <= Camera.tiledReach(size: L))
        // The 2D world is seen from below the plane (pitch above the horizon): the same rule, the other way round.
        var below = Camera.topDown(size: L)
        below.pitch = 25 * .pi / 180
        below.distance = WorldLevels.maxDistance(size: L, fovY: below.fovY, aspect: 1)
        #expect(below.position.z < 0)
        below.keepFloorInReach(size: L, aspect: 1)
        let seenBelow = try #require(below.farthestFloorDistance(aspect: 1))
        let tilt: Float = 25 * .pi / 180
        #expect(seenBelow <= Camera.tiledReach(size: L) && below.pitch > tilt)
    }

    @Test func aCameraLookingAwayFromTheFloorHasNothingToProtect() {
        let up: Float = 30 * .pi / 180
        var camera = Camera(target: SIMD3(160, 160, 300), distance: 200, yaw: 0, pitch: up)
        #expect(camera.position.z > 0 && camera.farthestFloorDistance(aspect: 1) == 0)
        #expect(!camera.keepFloorInReach(size: L, aspect: 1) && camera.pitch == up)
    }
}
