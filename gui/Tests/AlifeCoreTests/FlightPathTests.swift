import Foundation
import Testing
import simd

@testable import AlifeCore

/// The fly-to on the van Wijk and Nuij path (DESIGN §36.5; spec 3.6 and 7.1 `FlightPathTests`, flat).
@Suite struct FlightPathTests {
    private let L: Float = 320
    private let k: Float = 2 * tan(Camera.defaultFovY / 2) * 1.38  // the width of the view per unit of distance

    private func path(_ from: (Float, Float, Float), _ to: (Float, Float, Float), straight: Bool = false) -> FlightPath {
        FlightPath(from: FlightPath.Pose(x: from.0, y: from.1, distance: from.2), to: FlightPath.Pose(x: to.0, y: to.1, distance: to.2),
                   size: L, widthPerDistance: k, straight: straight)
    }

    @Test func theEndpointsAreExactAndTheTimeIsClamped() {
        let p = path((10, 20, 100), (200, 90, 30))
        #expect(p.pose(at: 0) == p.from && p.pose(at: -3) == p.from)
        #expect(p.pose(at: p.duration) == p.to && p.pose(at: p.duration + 5) == p.to, "the end exactly, not within a rounding")
        let inside = p.pose(at: p.duration * 0.999)
        #expect(abs(Torus.wrapDelta(inside.x - 200, size: L)) < 3 && abs(inside.distance - 30) < 3, "(x is unwrapped on the way: 200 is -120)")
    }

    @Test func theDistanceIsPositiveAndHasAtMostOneExtremum() {
        for (from, to) in [((10, 20, 100), (200, 90, 30)), ((10, 20, 30), (200, 90, 30)), ((10, 20, 30), (200, 90, 100)), ((10, 20, 30), (14, 22, 31))] as [((Float, Float, Float), (Float, Float, Float))] {
            let p = path(from, to)
            var signs = 0
            var last = p.pose(at: 0).distance
            var direction: Float = 0
            for i in 1...400 {
                let d = p.pose(at: p.duration * Float(i) / 400).distance
                #expect(d > 0 && d.isFinite)
                let step = d - last
                if abs(step) > 1e-4 * last {
                    if direction != 0, step.sign != direction.sign { signs += 1 }
                    direction = step
                }
                last = d
            }
            #expect(signs <= 1, "\(from) to \(to): \(signs) turns")
        }
    }

    @Test func aLongFlightZoomsOutFirstSoItsPeakIsHigherThanBothEnds() {
        let p = path((100, 160, 30), (300, 160, 30))  // 200 units apart at the same distance
        #expect(p.peakDistance > 1.5 * 30, "the peak is \(p.peakDistance)")
        #expect(path((100, 160, 30), (100.0005, 160, 30)).peakDistance <= 30.001, "a pure zoom has no arc")
        let near = path((100, 160, 60), (104, 162, 60))
        #expect(near.peakDistance < p.peakDistance && near.peakDistance >= 60)
    }

    @Test func theSpeedAcrossTheScreenIsEvenOnVanWijksMetric() {
        // On the path, (d ln w / ds)^2 + rho^4 ((du / ds) / w)^2 = rho^2 at every s: a constant speed in the metric of zooming and panning.
        let p = path((100, 160, 120), (260, 200, 25))
        let u1 = simd_length(p.delta)
        let rho = FlightPath.rho
        let dt = p.duration / 4000
        var worst: Float = 0
        for i in 1..<399 {
            let t = p.duration * Float(i) / 400
            let (a, b) = (p.pose(at: t - dt), p.pose(at: t + dt))
            let progressRate = ((b.x - a.x) * p.delta.x + (b.y - a.y) * p.delta.y) / (u1 * u1 * 2 * dt)  // d(progress)/dt
            let dudt = progressRate * u1
            let w = p.pose(at: t).distance * k
            let dlnw = (log(b.distance) - log(a.distance)) / (2 * dt)
            // s = t * pathLength / duration, so d/ds = (duration / pathLength) d/dt.
            let pathLength = pathLengthOf(p)
            let scale = p.duration / pathLength
            let speed2 = (dlnw * scale) * (dlnw * scale) + pow(rho, 4) * (dudt * scale / w) * (dudt * scale / w)
            worst = max(worst, abs(speed2 - rho * rho) / (rho * rho))
        }
        #expect(worst < 0.02, "the metric speed varies by \(worst)")
    }

    /// S = (r1 - r0) / rho for the path's ends, recomputed from the public ends.
    private func pathLengthOf(_ p: FlightPath) -> Float {
        let rho = FlightPath.rho
        let w0 = p.from.distance * k, w1 = p.to.distance * k, u1 = simd_length(p.delta)
        let b0 = (w1 * w1 - w0 * w0 + pow(rho, 4) * u1 * u1) / (2 * w0 * rho * rho * u1)
        let b1 = (w1 * w1 - w0 * w0 - pow(rho, 4) * u1 * u1) / (2 * w1 * rho * rho * u1)
        return (log(-b1 + (b1 * b1 + 1).squareRoot()) - log(-b0 + (b0 * b0 + 1).squareRoot())) / rho
    }

    @Test func theCentreMovesMonotonicallyTheShortWayAcrossTheSeams() {
        let p = path((10, 100, 40), (310, 100, 40))
        #expect(p.delta == SIMD2(-20, 0), "20 units west, not 300 east")
        var last: Float = 10
        for i in 1...100 {
            let at = p.pose(at: p.duration * Float(i) / 100)
            #expect(at.x <= last + 1e-4 && at.x >= -10 - 1e-3 || abs(at.x - 310) < 1e-3, "x \(at.x)")
            last = at.x
        }
        #expect(abs(p.pose(at: p.duration / 2).x - 0) < 1.5, "halfway is the seam")
        let q = path((100, 150, 40), (100, 10, 40))  // 140 north-south: the short way is not across the wrap
        #expect(q.delta.y == -140)
        let r = path((100, 310, 40), (100, 20, 40))  // across the wrap in y: 30 units
        #expect(r.delta == SIMD2(0, 30))
    }

    @Test func theDurationIsWithinItsRangeAndAPureZoomIsExponentialInTime() {
        for (from, to) in [((10, 20, 100), (200, 90, 30)), ((10, 20, 30), (12, 21, 33)), ((10, 20, 30), (300, 300, 30))] as [((Float, Float, Float), (Float, Float, Float))] {
            let p = path(from, to)
            #expect(p.duration >= FlightPath.minDuration && p.duration <= FlightPath.maxDuration, "\(p.duration)")
        }
        let z = path((50, 50, 100), (50, 50, 25))
        #expect(z.delta == .zero && abs(z.duration - 0.35 * log(4)) < 1e-5)
        for i in 1..<10 {
            let t = z.duration * Float(i) / 10
            #expect(abs(z.pose(at: t).distance - 100 * pow(0.25, Float(i) / 10)) < 1e-3, "exponential in time")
        }
        #expect(path((50, 50, 100), (50, 50, 100.01)).duration == 0.2, "a hair of zoom still takes the shortest time")
        #expect(path((50, 50, 400), (50, 50, 4)).duration == 1.0)
    }

    @Test func theHeadingAndTheTiltEaseWithTheProgressAndTheHeadingTakesTheShortTurn() {
        let from = FlightPath.Pose(x: 0, y: 0, heading: 3, tilt: 0.8, distance: 50)
        let to = FlightPath.Pose(x: 100, y: 0, heading: -3, tilt: 0, distance: 50)
        let p = FlightPath(from: from, to: to, size: L, widthPerDistance: k)
        let half = p.pose(at: p.duration / 2)
        #expect(abs(abs(half.heading) - .pi) < 0.05, "3 to -3 is 0.28 rad across the back, not 6 rad the long way: \(half.heading)")
        #expect(abs(half.tilt - 0.4) < 0.05)
    }

    @Test func reduceMotionIsAStraightEaseInAQuarterOfASecond() {
        let p = path((10, 20, 100), (200, 90, 30), straight: true)
        #expect(p.duration == FlightPath.easeDuration && p.peakDistance <= 100 + 1e-3, "no arc, never farther than the start")
        var last = p.pose(at: 0)
        func progress(_ pose: FlightPath.Pose) -> Float {
            let moved = Torus.wrapDelta(SIMD2(pose.x - p.from.x, pose.y - p.from.y), size: L)
            return simd_dot(moved, p.delta) / simd_length_squared(p.delta)
        }
        for i in 1...50 {
            let at = p.pose(at: p.duration * Float(i) / 50)
            #expect(progress(at) >= progress(last) - 1e-4 && at.distance <= last.distance + 1e-4)
            last = at
        }
        #expect(last == p.to)
        let ease = FlightPath(from: FlightPath.Pose(x: 5, y: 5, heading: 1, distance: 50), to: FlightPath.Pose(x: 5, y: 5, heading: 0, distance: 50),
                              size: L, widthPerDistance: k, straight: true, duration: 0.3)
        #expect(ease.duration == 0.3 && ease.pose(at: 0.15).distance == 50 && abs(ease.pose(at: 0.15).heading - 0.5) < 1e-5)
    }

    @Test func aDegenerateFlightIsOverAtOnce() {
        let p = path((5, 5, 40), (5, 5, 40))
        #expect(p.pose(at: 0.1) == p.to || p.pose(at: 0.1).distance == 40)
        let nan = FlightPath(from: FlightPath.Pose(x: 5, y: 5, distance: 40), to: FlightPath.Pose(x: 6, y: 6, distance: 40), size: 0, widthPerDistance: 0)
        #expect(nan.pose(at: nan.duration) == nan.to)
    }
}
