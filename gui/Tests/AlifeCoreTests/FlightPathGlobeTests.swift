import Foundation
import Testing
import simd

@testable import AlifeCore

/// The fly-to on the globe: the same van Wijk path, the centre on the great circle (DESIGN §36.10; spec 3.6 and 7.1 `FlightPathTests`).
@Suite struct FlightPathGlobeTests {
    private let L: Float = 320
    private let k: Float = 2 * tan(Camera.defaultFovY / 2) * 1.38

    private func path(_ from: (Float, Float, Float), _ to: (Float, Float, Float), circle: Bool = true) -> FlightPath {
        FlightPath(from: FlightPath.Pose(x: from.0, y: from.1, distance: from.2), to: FlightPath.Pose(x: to.0, y: to.1, distance: to.2),
                   size: L, widthPerDistance: k, greatCircle: circle)
    }

    private func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float { atan2(simd_length(simd_cross(a, b)), simd_dot(a, b)) }

    @Test func theEndpointsAreExactAndTheFlatPathIsUnchanged() {
        let p = path((10, 20, 100), (200, 90, 30))
        #expect(p.pose(at: 0) == p.from && p.pose(at: p.duration) == p.to && p.pose(at: p.duration + 1) == p.to)
        let flat = path((10, 20, 100), (200, 90, 30), circle: false), also = FlightPath(
            from: flat.from, to: flat.to, size: L, widthPerDistance: k)
        #expect(flat.pose(at: 0.3) == also.pose(at: 0.3) && !flat.greatCircle, "the flat metric is the default")
    }

    @Test func theCentreDistanceIsTheGreatCircleLengthInTorusUnitsAtTheEquator() {
        // Along the equator, 80 units of x is a quarter turn: alpha = pi/2, u1 = 80.
        #expect(abs(path((0, 0, 60), (80, 0, 60)).centreDistance - 80) < 0.01)
        // A quarter turn along a meridian is longer in torus units than the rhumb line between the same two rows' ybar
        // (the sphere's way is the shorter one): equator to ybar 100, by x = 0: alpha = gd(2 pi 100 / 320).
        let alpha = GlobeMap.latitude(ybar: 100, L: L)
        #expect(abs(path((0, 0, 60), (0, 100, 60)).centreDistance - alpha * L / (2 * .pi)) < 0.01)
    }

    @Test func theCentreMovesMonotonicallyAlongTheGreatCircleTheShortWay() {
        let p = path((10, 20, 120), (200, 100, 30))
        let c0 = p.centreDirection(at: 0), c1 = p.centreDirection(at: p.duration), total = angle(c0, c1)
        var last: Float = -1
        for i in 0...200 {
            let c = p.centreDirection(at: p.duration * Float(i) / 200)
            let a = angle(c0, c)
            #expect(a >= last - 1e-5, "monotone along the way")
            last = a
            #expect(abs(angle(c0, c) + angle(c, c1) - total) < 1e-4, "on the circle through both ends (the shortest way)")
            #expect(abs(simd_length(c) - 1) < 1e-5)
        }
        #expect(total <= .pi)
    }

    @Test func theWayOverAPoleIsTakenWhenItIsNearer() {
        // ybar 120 at x = 10 and x = 170: half a world apart in x, both high in the north. Along the parallel it is far; over the pole it is
        // near, and the highest latitude on the way exceeds both ends'.
        let p = path((10, 120, 80), (170, 120, 80))
        let end = GlobeMap.latitude(ybar: 120, L: L)
        var top: Float = 0
        for i in 0...100 { top = max(top, asin(p.centreDirection(at: p.duration * Float(i) / 100).z)) }
        #expect(top > end + 0.05, "over the pole: \(top) against \(end)")
        let around = angle(GlobeMap.direction(x: 10, y: 120, L: L), GlobeMap.direction(x: 170, y: 120, L: L))
        #expect(abs(p.centreDistance - around * L / (2 * .pi)) < 0.01 && around < .pi)
    }

    @Test func theAntipodalFlightIsDeterministicAndWellFormed() {
        let p = path((0, 0, 100), (160, 0, 100))  // the antipode on the equator
        for i in 0...20 {
            let c = p.centreDirection(at: p.duration * Float(i) / 20)
            #expect(c.x.isFinite && c.y.isFinite && c.z.isFinite && abs(simd_length(c) - 1) < 1e-4)
        }
        #expect(p.centreDirection(at: 0.2) == path((0, 0, 100), (160, 0, 100)).centreDirection(at: 0.2))
    }

    @Test func theCrossingOfTheXSeamGoesTwentyUnitsWest() {
        let p = path((10, 0, 60), (310, 0, 60))
        #expect(abs(p.centreDistance - 20) < 0.01)
        let mid = p.pose(at: p.duration / 2)
        #expect(abs(Torus.wrapDelta(mid.x - 0, size: L)) < 1.5, "halfway is x = 0, not x = 160: \(mid.x)")
        var last = p.from.x
        for i in 1...50 {
            let x = p.pose(at: p.duration * Float(i) / 50).x
            #expect(Torus.wrapDelta(x - last, size: L) <= 1e-3, "westwards only")
            last = x
        }
    }

    @Test func theDurationIsClampedAndAPureZoomIsExponential() {
        for (a, b) in [((10, 20, 100), (200, 90, 30)), ((10, 20, 30), (14, 22, 31)), ((0, 0, 250), (160, 100, 20))] as [((Float, Float, Float), (Float, Float, Float))] {
            let p = path(a, b)
            #expect(p.duration >= 0.2 && p.duration <= FlightPath.maxDuration)
            if p.centreDistance >= FlightPath.pureZoom { #expect(p.duration >= FlightPath.minDuration) }
        }
        let zoom = path((50, 40, 100), (50, 40, 25))
        #expect(zoom.pose(at: zoom.duration / 2).distance - 50 < 1e-3, "the geometric mean at half time: 50")
        #expect(abs(zoom.pose(at: zoom.duration / 2).x - 50) < 1e-3 && abs(zoom.pose(at: zoom.duration / 2).y - 40) < 1e-3)
    }

    @Test func theOnScreenSpeedOfTheCentreFollowsTheVanWijkProfile() {
        // On the van Wijk path the centre's screen speed u'/w = 1 / (rho cosh(rho s + r0)) is proportional to the view's width
        // w(s) = w0 cosh(r0) / cosh(rho s + r0) (the speed is even in the perceptual metric: slow when zoomed in, fast when out).
        // Measured from the poses on a globe flight (the centre's angle in torus units at the equator over the width): speed / w
        // stays within 25% of its median over the middle of the flight.
        let p = path((0, 10, 150), (140, 60, 40))
        let n = 400
        var ratios: [Float] = []
        for i in (n / 10)..<(n - n / 10) {
            let t0 = p.duration * (Float(i) + 0.5) / Float(n + 1), t1 = t0 + p.duration / Float(n + 1)
            let step = angle(p.centreDirection(at: t0), p.centreDirection(at: t1)) * L / (2 * .pi)
            let width = p.pose(at: (t0 + t1) / 2).distance * k
            ratios.append(step / (t1 - t0) / width / width)
        }
        let median = ratios.sorted()[ratios.count / 2]
        let worst = ratios.map { abs($0 - median) / median }.max()!
        #expect(worst < 0.25, "speed / w is constant on the path (worst deviation \(worst))")
    }

    @Test func aFlightBetweenTwoFlatPosesSoarsHigherThanBothEnds() {
        let p = path((20, 0, 30), (220, 0, 30))
        #expect(p.peakDistance > 60, "peak \(p.peakDistance)")
        #expect(p.centreDistance > 100, "the long way is not taken: 120 units west, \(p.centreDistance)")
    }

    @Test func aStraightEaseIsStillStraightOnTheGlobeAndKeepsItsEnds() {
        let p = FlightPath(from: FlightPath.Pose(x: 10, y: 20, distance: 100), to: FlightPath.Pose(x: 200, y: 40, distance: 30),
                           size: L, widthPerDistance: k, straight: true, greatCircle: true)
        #expect(p.duration == FlightPath.easeDuration && p.pose(at: p.duration) == p.to)
        let d = p.pose(at: p.duration / 2).distance
        #expect(d < 100 && d > 30)
    }
}
