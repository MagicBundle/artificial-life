import Foundation
import simd

/// A fly-to (DESIGN §36.5, the spec's 3.6): zoom out, pan, zoom in, on the path of van Wijk and Nuij, "Smooth and efficient zooming
/// and panning" (2003). Pure: the window gives it the time that passed.
///
/// With `w` the width of the view (proportional to the distance), `u` the distance along the path, `rho = sqrt 2`, `(w0, w1)` the
/// widths at the ends and `u1` the distance between the two centres, in the units of the width:
///
/// ```text
/// b_i  = (w1^2 - w0^2 + (-1)^i rho^4 u1^2) / (2 w_i rho^2 u1)        r_i = ln(-b_i + sqrt(b_i^2 + 1))      S = (r1 - r0) / rho
/// w(s) = w0 cosh r0 / cosh(rho s + r0)                              u(s) = (w0 / rho^2) (cosh r0 tanh(rho s + r0) - sinh r0)
/// ```
///
/// The speed across the screen is even, the path never overshoots its ends' range by more than the zoom out the pan needs, and it
/// ends exactly at its end. The centre goes the short way round the torus. A pure zoom (the same centre) is exponential in time.
package struct FlightPath: Equatable, Sendable {
    /// The pose of a flight: the centre, the height, the heading, the tilt and the distance.
    package struct Pose: Equatable, Sendable {
        package var x: Float
        package var y: Float
        package var z: Float
        package var heading: Float
        package var tilt: Float
        package var distance: Float

        package init(x: Float, y: Float, z: Float = 0, heading: Float = 0, tilt: Float = 0, distance: Float) {
            self.x = x
            self.y = y
            self.z = z
            self.heading = heading
            self.tilt = tilt
            self.distance = distance
        }

        package init(_ view: WorldPose) {
            self.init(x: view.x, y: view.y, z: view.z, heading: view.heading, tilt: view.tilt, distance: view.distance)
        }
    }

    package static let rho: Float = Float(2).squareRoot()
    /// The speed of the path in its own units per second, and the limits of the duration.
    package static let speed: Float = 1.2
    package static let minDuration: Float = 0.45
    package static let maxDuration: Float = 1.8
    /// Under this a flight is a zoom alone (`u1` in world units).
    package static let pureZoom: Float = 1e-3
    /// The duration of a straight ease (reduce motion).
    package static let easeDuration: Float = 0.25

    package let from: Pose
    package let to: Pose
    /// The shortest way from the start to the end centre on the torus: east, north.
    package let delta: SIMD2<Float>
    package let duration: Float
    /// How wide the view is per unit of distance (`2 tan(fov/2) max(1, aspect)`).
    package let widthPerDistance: Float
    private let u1: Float
    private let r0: Float
    private let r1: Float
    private let pathLength: Float
    private let straight: Bool

    /// A flight from `from` to `to` on a torus of `size`. `widthPerDistance` is the width of the view per unit of distance, so
    /// that the pan and the zoom are in the same units; `straight` is the reduce-motion form: an ease of the centre and the
    /// distance in 0.25 s (`duration` sets another, as the 0.3 s of north up).
    package init(from: Pose, to: Pose, size: Float, widthPerDistance: Float, straight: Bool = false, duration fixed: Float? = nil) {
        self.from = from
        self.to = to
        self.widthPerDistance = max(widthPerDistance, 1e-6)
        self.straight = straight
        let delta = size > 0
            ? Torus.wrapDelta(SIMD2(to.x - from.x, to.y - from.y), size: size) : SIMD2(to.x - from.x, to.y - from.y)
        self.delta = delta
        let u1 = simd_length(delta)
        self.u1 = u1
        let w0 = max(from.distance, 1e-6) * self.widthPerDistance, w1 = max(to.distance, 1e-6) * self.widthPerDistance
        let rho = Self.rho
        if u1 < Self.pureZoom {
            r0 = 0
            r1 = 0
            pathLength = abs(log(w1 / w0))
            duration = fixed ?? (straight ? Self.easeDuration : min(max(0.35 * abs(log(to.distance / max(from.distance, 1e-6))), 0.2), 1.0))
        } else {
            let b0 = (w1 * w1 - w0 * w0 + pow(rho, 4) * u1 * u1) / (2 * w0 * rho * rho * u1)
            let b1 = (w1 * w1 - w0 * w0 - pow(rho, 4) * u1 * u1) / (2 * w1 * rho * rho * u1)
            r0 = log(-b0 + (b0 * b0 + 1).squareRoot())
            r1 = log(-b1 + (b1 * b1 + 1).squareRoot())
            pathLength = (r1 - r0) / rho
            duration = fixed ?? (straight ? Self.easeDuration : min(max(pathLength / Self.speed, Self.minDuration), Self.maxDuration))
        }
    }

    /// The pose `time` seconds in: the end pose exactly from `duration` on.
    package func pose(at time: Float) -> Pose {
        let t = duration > 0 ? min(max(time / duration, 0), 1) : 1
        if t >= 1 { return to }
        if t <= 0 { return from }
        var progress: Float  // the fraction of the way for the centre, the height, the heading and the tilt
        var distance: Float
        if straight {
            progress = t * t * (3 - 2 * t)
            distance = from.distance * pow(to.distance / max(from.distance, 1e-6), progress)
        } else if u1 < Self.pureZoom {
            progress = t
            distance = from.distance * pow(to.distance / max(from.distance, 1e-6), t)
        } else {
            let rho = Self.rho
            let s = t * pathLength
            let w0 = from.distance * widthPerDistance
            let width = w0 * cosh(r0) / cosh(rho * s + r0)
            let along = (w0 / (rho * rho)) * (cosh(r0) * tanh(rho * s + r0) - sinh(r0))
            progress = min(max(along / u1, 0), 1)
            distance = width / widthPerDistance
        }
        let turn = Torus.wrapDelta(to.heading - from.heading, size: 2 * .pi)
        let centre = SIMD2(from.x, from.y) + delta * progress
        return Pose(x: centre.x, y: centre.y, z: from.z + (to.z - from.z) * progress, heading: WorldPose.wrap(from.heading + turn * progress),
                    tilt: from.tilt + (to.tilt - from.tilt) * progress, distance: distance)
    }

    /// The distance the flight reaches at its farthest.
    package var peakDistance: Float {
        var peak = max(from.distance, to.distance)
        for i in 1..<64 { peak = max(peak, pose(at: duration * Float(i) / 64).distance) }
        return peak
    }
}
