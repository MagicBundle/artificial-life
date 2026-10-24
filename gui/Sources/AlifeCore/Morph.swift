import Foundation
import simd

/// The continuous zoom from the globe to the flat map (DESIGN §36.10, the spec's section 3). Pure; the vertex shader mirrors `position`.
///
/// The scene is drawn in *sphere units*: the torus is scaled by `M = 2 pi R / L` (2.827) about the camera's target and the eye is at
/// `E = M d` from the surface point at the centre, so that at `s = 0` the view is the flat map exactly (a similarity) and the plane's
/// equator is as long as the sphere's. One scalar `s` in `[0, 1]` runs the morph; it is a function of the logarithm of the distance `d`
/// (the wheel and the pinch multiply `d`) through a trapezoid velocity profile.
///
/// The morph is **roll, then Mercator-ise**: the scaled plane is rolled up onto a sphere of radius `R / s` tangent at the centre (the
/// exponential map; for every `s` the visible surface is one smooth convex sheet, the far part folds behind it), and over the second
/// half the polar rows move from the roll's antipode to their true latitude on the Mercator sphere of `GlobeMap`.
package enum Morph {
    /// How the two regimes meet. `.roll` is the design; `.crossfade` (the flat layer at `d`, the globe layer at `E = M d`, alpha by `s`) is
    /// the fallback of the spec's 8.4, selected here when the roll fails the criteria of 7.1 on the real schedule.
    package enum Mode: Sendable { case roll, crossfade }
    package static let mode: Mode = .roll

    /// `2 pi R / L`: sphere units per torus unit at the equator.
    package static let M: Float = 2 * .pi * GlobeMap.radiusFactor
    /// `d_f / L`: at or under it the view is exactly the flat map.
    package static let flatLimit: Float = 0.12
    /// The share of the log distance spent in each ease of the trapezoid.
    package static let ease: Float = 0.15
    /// The Mercator tail `w2` runs over `s` in `[mercatorFrom, 1]`.
    package static let mercatorFrom: Float = 0.4
    /// The height scale falls from `M` to `kz` over `s` in `[heightFrom, 1]`.
    package static let heightFrom: Float = 0.3
    /// The farthest the view may be, as a fraction of `L` (the globe at 36% of the height).
    package static let maxDistanceFraction: Float = 0.8
    /// The sphere's share of the smaller view dimension at Home.
    package static let homeFill: Float = 0.80

    /// The eye distance from the surface point at the centre, in sphere units.
    package static func eyeDistance(distance d: Float) -> Float { M * d }

    /// `d_g`: the distance at which the sphere fills `fill` of the smaller view dimension, `(D - R) / M` with
    /// `D = R / sin(atan(fill tan(fov/2) min(1, aspect)))`. The same at every latitude; it is also the Home distance.
    package static func globeDistance(L: Float, fovY: Float, aspect: Float, fill: Float = Morph.homeFill) -> Float {
        let R = GlobeMap.radius(L: L)
        let D = R / sin(atan(fill * tan(fovY / 2) * min(1, aspect)))
        return (D - R) / M
    }

    /// `t` in `[0, 1]`: the position of `d` between `d_f` and `d_g` in the logarithm.
    package static func progress(distance d: Float, L: Float, fovY: Float, aspect: Float) -> Float {
        let df = flatLimit * L, dg = globeDistance(L: L, fovY: fovY, aspect: aspect)
        guard dg > df, d > df else { return 0 }
        return min(max(log(d / df) / log(dg / df), 0), 1)
    }

    /// `s = trap(t)`: 15% ease-in and ease-out, C1 at both ends, a peak slope of `1 / (1 - ease)`.
    package static func trap(_ t: Float) -> Float {
        let a = ease, v = 1 / (1 - a), t = min(max(t, 0), 1)
        if t < a { return v * t * t / (2 * a) }
        if t < 1 - a { return v * (t - a / 2) }
        return 1 - v * (1 - t) * (1 - t) / (2 * a)
    }

    /// The inverse of `trap`.
    package static func trapInverse(_ s: Float) -> Float {
        let a = ease, v = 1 / (1 - a), s = min(max(s, 0), 1)
        let edge = v * a / 2
        if s < edge { return (2 * a * s / v).squareRoot() }
        if s <= 1 - edge { return s / v + a / 2 }
        return 1 - (2 * a * (1 - s) / v).squareRoot()
    }

    /// The morph scalar at a distance.
    package static func s(distance d: Float, L: Float, fovY: Float, aspect: Float) -> Float {
        trap(progress(distance: d, L: L, fovY: fovY, aspect: aspect))
    }

    /// The distance at which the morph scalar is `s`.
    package static func distance(forS s: Float, L: Float, fovY: Float, aspect: Float) -> Float {
        let df = flatLimit * L, dg = globeDistance(L: L, fovY: fovY, aspect: aspect)
        return df * pow(dg / df, trapInverse(s))
    }

    /// `lambda(s) = (1 - s) + s cos(phi_c)`: the plane scale `a = M lambda` is `M` at `s = 0` and `M cos(phi_c)` at `s = 1`, so the
    /// centre's apparent scale `lambda / d` never grows as the player zooms out.
    package static func lambda(s: Float, latitude: Float) -> Float { (1 - s) + s * cos(latitude) }

    /// The weight of the Mercator sphere over the roll.
    package static func w2(s: Float) -> Float { smoothstep(mercatorFrom, 1, s) }

    /// The height scale: true (`M`, as x and y) until `heightFrom`, then compressed into the shell (`kz`).
    package static func zeta(s: Float, L: Float, height: Float) -> Float {
        M + (GlobeMap.shellScale(L: L, height: height) - M) * smoothstep(heightFrom, 1, s)
    }

    /// The alphas of the two layers in `.crossfade` mode: (flat, globe).
    package static func crossfadeAlpha(s: Float) -> (flat: Float, globe: Float) { (1 - s, s) }

    package static func smoothstep(_ lo: Float, _ hi: Float, _ x: Float) -> Float {
        let t = min(max((x - lo) / (hi - lo), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// The local frame at the centre: `(E, N, C)` with the heading applied about `C`. `right` and `up` are the screen axes of a map
    /// turned by `heading` (counter-clockwise seen from above: the same sense as `WorldPose.heading`; 0 is north up).
    package struct Frame: Equatable, Sendable {
        package var x: Float
        package var ybar: Float
        package var latitude: Float
        package var heading: Float
        package var right: SIMD3<Float>
        package var up: SIMD3<Float>
        package var centre: SIMD3<Float>

        /// The frame at a centre direction `c` (a unit vector, anywhere on the sphere, a pole cap included) with `heading`. `ybar` is the
        /// Mercator row of its latitude, held at `+-L/2` inside a cap (the cap has no rows): the two agree wherever the centre is a place.
        package init(centre c: SIMD3<Float>, heading: Float = 0, L: Float) {
            let c = simd_normalize(c)
            let lam = atan2(c.y, c.x)
            x = Torus.wrap(L * lam / (2 * .pi), size: L)
            latitude = asin(min(max(c.z, -1), 1))
            ybar = abs(latitude) < GlobeMap.phiMax ? GlobeMap.ybar(latitude: latitude, L: L) : (latitude < 0 ? -L / 2 : L / 2)
            self.heading = heading
            centre = c
            let east = SIMD3<Float>(-sin(lam), cos(lam), 0)
            let north = simd_cross(c, east)
            let ch = cos(heading), sh = sin(heading)
            right = ch * east + sh * north
            up = -sh * east + ch * north
        }

        /// The heading of a map whose screen-right axis is `right` at the centre `c`: the inverse of the frame's construction.
        package static func heading(right: SIMD3<Float>, centre c: SIMD3<Float>) -> Float {
            let lam = atan2(c.y, c.x)
            let east = SIMD3<Float>(-sin(lam), cos(lam), 0)
            let north = simd_cross(c, east)
            return atan2(simd_dot(right, north), simd_dot(right, east))
        }

        /// The frame at column `x` and signed row `ybar` of a world of size `L`.
        package init(x: Float, ybar: Float, heading: Float = 0, L: Float) {
            self.x = x
            self.ybar = ybar
            self.heading = heading
            let lam = GlobeMap.longitude(x: x, L: L)
            latitude = GlobeMap.latitude(ybar: ybar, L: L)
            centre = GlobeMap.direction(longitude: lam, latitude: latitude)
            let east = SIMD3<Float>(-sin(lam), cos(lam), 0)
            let north = simd_cross(centre, east)
            let ch = cos(heading), sh = sin(heading)
            right = ch * east + sh * north
            up = -sh * east + ch * north
        }
    }

    /// The position in the centre's frame (x right, y up, z towards the eye, the sphere's centre at the origin at `s = 1`) of the mesh
    /// vertex at the offsets `(e, n)` from the centre, east and north in torus units, at height `z`. `n` is the *unwrapped* row offset
    /// (`ybar + kL - ybar_c` for the copy `k`, so a copy's far edge is pulled to the pole and not to the other one). Mirrors the vertex shader.
    package static func position(e: Float, n: Float, z: Float, frame: Frame, s: Float, L: Float, height: Float) -> SIMD3<Float> {
        let roll = rollPosition(e: e, n: n, z: z, frame: frame, s: s, L: L, height: height)
        let w = w2(s: s)
        if w <= 0 { return roll }
        let g = globePosition(e: e, n: n, z: z, frame: frame, L: L, height: height)
        return roll + (g - roll) * w
    }

    /// The outward unit normal of the morphing surface (the shader's lighting normal, and the picker's front-facing test): the rolled
    /// sphere's radial direction mixed with the Mercator sphere's by `w2` and renormalised; `(0, 0, 1)` at `s = 0`.
    package static func normal(e: Float, n: Float, frame: Frame, s: Float, L: Float) -> SIMD3<Float> {
        var roll = SIMD3<Float>(0, 0, 1)
        if s > 1e-6 {
            let a = M * lambda(s: s, latitude: frame.latitude)
            let ch = cos(frame.heading), sh = sin(frame.heading)
            let ux = a * (e * ch + n * sh), uy = a * (-e * sh + n * ch)
            let rho = GlobeMap.radius(L: L) / s, r = hypot(ux, uy)
            let theta = min(r / rho, 2 * .pi - 0.2)
            let k = r > 1e-9 ? sin(theta) / r : 1 / rho
            roll = SIMD3(k * ux, k * uy, cos(theta))
        }
        let w = w2(s: s)
        if w <= 0 { return roll }
        let q = GlobeMap.direction(longitude: GlobeMap.longitude(x: frame.x + e, L: L), latitude: GlobeMap.latitude(ybar: frame.ybar + n, L: L))
        let g = SIMD3(simd_dot(q, frame.right), simd_dot(q, frame.up), simd_dot(q, frame.centre))
        return simd_normalize(roll + (g - roll) * w)
    }

    /// The scaled plane rolled onto a sphere of radius `R / s`, tangent at the centre (the plane itself at `s = 0`), the height carried
    /// along the rolled normal with `zeta`.
    package static func rollPosition(e: Float, n: Float, z: Float, frame: Frame, s: Float, L: Float, height: Float) -> SIMD3<Float> {
        let R = GlobeMap.radius(L: L)
        let a = M * lambda(s: s, latitude: frame.latitude)
        let ch = cos(frame.heading), sh = sin(frame.heading)
        let ux = a * (e * ch + n * sh), uy = a * (-e * sh + n * ch)
        let lift = zeta(s: s, L: L, height: height) * z
        guard s > 1e-6 else { return SIMD3(ux, uy, R + lift) }
        let rho = R / s, r = hypot(ux, uy)
        let theta = min(r / rho, 2 * .pi - 0.2)
        let k = r > 1e-9 ? sin(theta) / r : 1 / rho
        let nhat = SIMD3(k * ux, k * uy, cos(theta))
        // `1 - cos` as `2 sin^2(theta/2)`: at a small `s` the radius `rho` is large and the cancellation would show.
        let versine = 2 * sin(theta / 2) * sin(theta / 2)
        return SIMD3(rho * k * ux, rho * k * uy, R - rho * versine) + lift * nhat
    }

    /// The flat layer of `.crossfade`: the plane at the morph's scale `a = M lambda(s)`, the height scaled by `zeta`.
    package static func flatPosition(e: Float, n: Float, z: Float, frame: Frame, s: Float, L: Float, height: Float) -> SIMD3<Float> {
        let a = M * lambda(s: s, latitude: frame.latitude)
        let ch = cos(frame.heading), sh = sin(frame.heading)
        return SIMD3(a * (e * ch + n * sh), a * (-e * sh + n * ch), GlobeMap.radius(L: L) + zeta(s: s, L: L, height: height) * z)
    }

    /// The Mercator sphere of `GlobeMap` in the centre's frame (`G`), at radius `R + kz z`.
    package static func globePosition(e: Float, n: Float, z: Float, frame: Frame, L: Float, height: Float) -> SIMD3<Float> {
        let q = GlobeMap.direction(longitude: GlobeMap.longitude(x: frame.x + e, L: L), latitude: GlobeMap.latitude(ybar: frame.ybar + n, L: L))
        let radius = GlobeMap.radius(L: L) + GlobeMap.shellScale(L: L, height: height) * z
        return radius * SIMD3(simd_dot(q, frame.right), simd_dot(q, frame.up), simd_dot(q, frame.centre))
    }
}
